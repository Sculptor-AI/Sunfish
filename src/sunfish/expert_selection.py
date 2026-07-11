"""Coverage-constrained expert selection from router calibration statistics.

Implements the selector described in ``docs/architecture.md``: retain router
probability mass, weighted across phase/workload buckets, subject to a
minimum retained-mass fraction in *every* bucket. Frequency-only top-N falls
out as the special case of one bucket and no coverage constraint.

This is a **deterministic multi-start heuristic, not an exact solver**: two
greedy starts (weighted-score top-k and maximin coverage) each undergo
bounded swap repair with cycle detection, and the best candidate wins —
feasible candidates by weighted retained mass, otherwise by worst-bucket
coverage. Single-start swap repair alone can cycle and can misreport a
feasible instance as infeasible; both failure cases are regression-tested
(Codex counterexamples, coordination/channel.md [6]). ``satisfied=False``
therefore means "no candidate found", not a proof of infeasibility — treat
it as a gate failure to investigate, not a certificate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of expert selection for a single layer."""

    selected: tuple[int, ...]
    coverage: dict[str, float]
    weighted_retained: float
    satisfied: bool


def relative_coverage_floor(*, retained_experts: int, source_experts: int, ratio: float) -> float:
    """Convert a size-normalized coverage ratio into an absolute mass floor.

    A load-balanced router has an expected retained mass of ``k / E`` for an
    arbitrary ``k``-expert subset. Expressing the floor relative to that
    baseline keeps the gate comparable across the 64-, 48-, and 32-expert
    ablations.
    """
    if not 0 < retained_experts <= source_experts:
        raise ValueError("retained_experts must be in [1, source_experts]")
    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    floor = ratio * retained_experts / source_experts
    if floor > 1:
        raise ValueError("relative coverage floor cannot exceed 1")
    return floor


def _normalized(mass: Sequence[float]) -> list[float]:
    total = float(sum(mass))
    if total <= 0:
        raise ValueError("bucket mass must have a positive sum")
    return [value / total for value in mass]


def retained_fraction(mass: Sequence[float], selected: Sequence[int]) -> float:
    """Fraction of a bucket's router mass captured by the selected experts."""
    normalized = _normalized(mass)
    return sum(normalized[index] for index in set(selected))


def select_experts(
    bucket_mass: Mapping[str, Sequence[float]],
    *,
    k: int,
    bucket_weights: Mapping[str, float] | None = None,
    min_coverage: float = 0.0,
    max_swap_rounds: int = 256,
) -> SelectionResult:
    """Select ``k`` experts for one layer from per-bucket router mass vectors."""
    if not bucket_mass:
        raise ValueError("bucket_mass must not be empty")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")

    lengths = {len(mass) for mass in bucket_mass.values()}
    if len(lengths) != 1:
        raise ValueError("all buckets must cover the same expert count")
    num_experts = lengths.pop()
    if not 0 < k <= num_experts:
        raise ValueError("k must be in [1, num_experts]")

    weights = dict(bucket_weights) if bucket_weights is not None else {}
    for bucket in bucket_mass:
        weight = weights.setdefault(bucket, 1.0)
        if weight < 0:
            raise ValueError("bucket weights must be non-negative")
    if set(weights) - set(bucket_mass):
        raise ValueError("bucket_weights refers to unknown buckets")

    normalized = {bucket: _normalized(mass) for bucket, mass in bucket_mass.items()}
    score = [
        sum(weights[bucket] * normalized[bucket][expert] for bucket in normalized)
        for expert in range(num_experts)
    ]

    def coverage_of(experts: set[int]) -> dict[str, float]:
        return {
            bucket: sum(normalized[bucket][index] for index in experts)
            for bucket in normalized
        }

    def weighted_greedy_start() -> set[int]:
        return set(sorted(range(num_experts), key=score.__getitem__, reverse=True)[:k])

    def maximin_greedy_start() -> set[int]:
        # Repeatedly add the expert that maximizes the worst bucket's
        # coverage; ties fall back to weighted score. Escapes the "already
        # satisfied bucket blocks every single swap" trap of pure repair.
        selected: set[int] = set()
        running = {bucket: 0.0 for bucket in normalized}
        while len(selected) < k:
            best: tuple[tuple[float, float], int] | None = None
            for expert in range(num_experts):
                if expert in selected:
                    continue
                candidate_min = min(
                    running[bucket] + normalized[bucket][expert] for bucket in normalized
                )
                key = (candidate_min, score[expert])
                if best is None or key > best[0]:
                    best = (key, expert)
            assert best is not None
            selected.add(best[1])
            for bucket in normalized:
                running[bucket] += normalized[bucket][best[1]]
        return selected

    def repair(selected: set[int]) -> set[int]:
        """Bounded swap repair with cycle detection; returns the maximin-best state seen."""
        coverage = coverage_of(selected)
        visited = {frozenset(selected)}
        best_state, best_min = set(selected), min(coverage.values())
        for _ in range(max_swap_rounds):
            violations = {b: c for b, c in coverage.items() if c < min_coverage}
            if not violations:
                break
            worst_bucket = min(violations, key=violations.__getitem__)
            worst_mass = normalized[worst_bucket]

            additions = sorted(
                (index for index in range(num_experts) if index not in selected),
                key=worst_mass.__getitem__,
                reverse=True,
            )
            removals = sorted(selected, key=score.__getitem__)

            chosen: tuple[set[int], dict[str, float]] | None = None
            for addition in additions:
                if worst_mass[addition] <= 0:
                    break
                for removal in removals:
                    if worst_mass[removal] >= worst_mass[addition]:
                        continue
                    candidate = (selected - {removal}) | {addition}
                    if frozenset(candidate) in visited:
                        continue  # cycle detection: never revisit a state
                    candidate_coverage = coverage_of(candidate)
                    # Never push a currently satisfied bucket below the threshold.
                    if any(
                        candidate_coverage[bucket] < min_coverage
                        for bucket in coverage
                        if coverage[bucket] >= min_coverage
                    ):
                        continue
                    chosen = (candidate, candidate_coverage)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                break
            selected, coverage = chosen
            visited.add(frozenset(selected))
            if min(coverage.values()) > best_min:
                best_state, best_min = set(selected), min(coverage.values())
        final_min = min(coverage.values())
        return selected if final_min >= best_min else best_state

    candidates = {
        frozenset(repair(start()))
        for start in (weighted_greedy_start, maximin_greedy_start)
    }

    def candidate_key(experts: frozenset[int]) -> tuple[int, float, float]:
        coverage = coverage_of(set(experts))
        feasible = all(value >= min_coverage for value in coverage.values())
        weighted = sum(weights[bucket] * coverage[bucket] for bucket in coverage)
        # Feasible first; among feasible, best weighted mass; among
        # infeasible, best worst-bucket coverage.
        return (int(feasible), weighted if feasible else min(coverage.values()), weighted)

    winner = set(max(candidates, key=candidate_key))
    coverage = coverage_of(winner)
    weighted_retained = sum(weights[bucket] * coverage[bucket] for bucket in coverage)
    satisfied = all(value >= min_coverage for value in coverage.values())
    return SelectionResult(
        selected=tuple(sorted(winner)),
        coverage=coverage,
        weighted_retained=weighted_retained,
        satisfied=satisfied,
    )


def select_per_layer(
    layers: Sequence[Mapping[str, Sequence[float]]],
    *,
    k: int,
    bucket_weights: Mapping[str, float] | None = None,
    min_coverage: float = 0.0,
) -> list[SelectionResult]:
    """Run selection independently for every layer (selection_is_per_layer)."""
    return [
        select_experts(
            bucket_mass, k=k, bucket_weights=bucket_weights, min_coverage=min_coverage
        )
        for bucket_mass in layers
    ]
