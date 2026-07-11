"""Router-statistics accumulation for expert-pruning calibration.

The training-side hook (JAX) dumps per-layer router probabilities; this module
owns the aggregation format so JAX, PyTorch, and analysis notebooks all speak
the same schema. Buckets are free-form strings and are expected to encode both
phase and workload, e.g. ``"prefill/code_completion"`` or
``"denoise_high/tool_calls"``.

Dependency-free on purpose: it runs anywhere, including inside a TPU VM with
nothing but the standard library.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

_SCHEMA_VERSION = 1


class RouterStatsAccumulator:
    """Accumulate per-bucket, per-layer router probability mass.

    Mass is the sum of router probabilities assigned to each expert across all
    observed tokens, so ``mass / tokens`` is the mean per-token probability.
    """

    def __init__(self, *, num_layers: int, num_experts: int):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._mass: dict[str, list[list[float]]] = {}
        self._tokens: dict[str, int] = {}

    @property
    def buckets(self) -> tuple[str, ...]:
        return tuple(sorted(self._mass))

    def _bucket_mass(self, bucket: str) -> list[list[float]]:
        if bucket not in self._mass:
            self._mass[bucket] = [[0.0] * self.num_experts for _ in range(self.num_layers)]
            self._tokens[bucket] = 0
        return self._mass[bucket]

    def update(self, *, bucket: str, layer: int, probabilities: Sequence[float]) -> None:
        """Add one token's full router distribution for one layer."""
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer out of range")
        if len(probabilities) != self.num_experts:
            raise ValueError("probabilities length must equal num_experts")
        row = self._bucket_mass(bucket)[layer]
        for expert, probability in enumerate(probabilities):
            if probability < 0:
                raise ValueError("probabilities must be non-negative")
            row[expert] += probability

    def update_topk(
        self,
        *,
        bucket: str,
        layer: int,
        expert_indices: Sequence[int],
        probabilities: Sequence[float],
    ) -> None:
        """Add one token's top-k router mass when full distributions are too big to log."""
        if len(expert_indices) != len(probabilities):
            raise ValueError("expert_indices and probabilities must have equal length")
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer out of range")
        row = self._bucket_mass(bucket)[layer]
        for expert, probability in zip(expert_indices, probabilities):
            if not 0 <= expert < self.num_experts:
                raise ValueError("expert index out of range")
            if probability < 0:
                raise ValueError("probabilities must be non-negative")
            row[expert] += probability

    def count_tokens(self, *, bucket: str, tokens: int) -> None:
        """Record how many tokens contributed to a bucket (once per batch)."""
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self._bucket_mass(bucket)
        self._tokens[bucket] += tokens

    def tokens(self, bucket: str) -> int:
        return self._tokens.get(bucket, 0)

    def layer_bucket_mass(self, layer: int) -> dict[str, list[float]]:
        """Return ``bucket -> expert mass vector`` for one layer, for selection."""
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer out of range")
        return {bucket: list(mass[layer]) for bucket, mass in self._mass.items()}

    def merge(self, other: "RouterStatsAccumulator") -> None:
        """Fold another accumulator (e.g. from another TPU host) into this one."""
        if (other.num_layers, other.num_experts) != (self.num_layers, self.num_experts):
            raise ValueError("accumulator shapes do not match")
        for bucket, layers in other._mass.items():
            rows = self._bucket_mass(bucket)
            for layer, layer_mass in enumerate(layers):
                row = rows[layer]
                for expert, mass in enumerate(layer_mass):
                    row[expert] += mass
            self._tokens[bucket] += other._tokens[bucket]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "num_layers": self.num_layers,
                "num_experts": self.num_experts,
                "tokens": self._tokens,
                "mass": self._mass,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "RouterStatsAccumulator":
        data = json.loads(payload)
        if data.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("unsupported router-stats schema version")
        accumulator = cls(num_layers=data["num_layers"], num_experts=data["num_experts"])
        mass: Mapping[str, Sequence[Sequence[float]]] = data["mass"]
        for bucket, layers in mass.items():
            if len(layers) != accumulator.num_layers:
                raise ValueError("layer count mismatch in payload")
            rows = accumulator._bucket_mass(bucket)
            for layer, layer_mass in enumerate(layers):
                if len(layer_mass) != accumulator.num_experts:
                    raise ValueError("expert count mismatch in payload")
                rows[layer] = [float(value) for value in layer_mass]
            accumulator._tokens[bucket] = int(data["tokens"][bucket])
        return accumulator
