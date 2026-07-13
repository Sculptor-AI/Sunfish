"""Dependency-free fixed-window packing for router-calibration corpora."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


CALIBRATION_BUCKET_PERCENT = {
    "code_completion": 35,
    "repo_edit": 20,
    "tool_calls": 15,
    "agent_trajectory": 15,
    "general_control": 10,
    "reasoning_control": 5,
}


def calibration_bucket_token_caps(total_tokens: int) -> dict[str, int]:
    """Allocate the docs/data.md mix exactly, including integer remainder."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    caps = {
        bucket: total_tokens * percent // 100
        for bucket, percent in CALIBRATION_BUCKET_PERCENT.items()
    }
    remainder = total_tokens - sum(caps.values())
    for bucket in CALIBRATION_BUCKET_PERCENT:
        if remainder <= 0:
            break
        caps[bucket] += 1
        remainder -= 1
    return caps


def calibration_windows(
    token_ids: Sequence[int],
    *,
    record_tokens: int,
    min_record_tokens: int,
    remaining_tokens: int,
) -> Iterator[list[int]]:
    """Yield non-overlapping records without exceeding a bucket token cap."""
    if record_tokens < 2:
        raise ValueError("record_tokens must be at least two")
    if not 2 <= min_record_tokens <= record_tokens:
        raise ValueError("min_record_tokens must be in [2, record_tokens]")
    if remaining_tokens < 0:
        raise ValueError("remaining_tokens must be non-negative")

    cursor = 0
    remaining = remaining_tokens
    while remaining >= 2 and cursor < len(token_ids):
        take = min(record_tokens, remaining, len(token_ids) - cursor)
        if remaining - take == 1 and take > min_record_tokens:
            # Never strand a one-token budget tail: leave two tokens for a
            # later record and discard one source-tail token if necessary.
            take -= 1
        if take < min_record_tokens and not (take == remaining and take >= 2):
            break
        yield [int(token) for token in token_ids[cursor : cursor + take]]
        cursor += take
        remaining -= take
