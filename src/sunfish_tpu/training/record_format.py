"""Compact v1 envelope for prompt/response records inside token shards.

``sunfish.datashards`` deliberately stores opaque uint32 records.  This module
defines the training-side envelope without changing that shared storage API.
Loss masks are optional and bit-packed; an omitted mask means all tokens are
supervised.  Keeping prompt and response masks separate is load-bearing for
agent data: tool observations can remain conditioning-only while the next
assistant action still contributes to both diffusion and encoder losses.
"""

from __future__ import annotations

import dataclasses
from array import array
from collections.abc import Iterable, Sequence

MAGIC = 0x53465231  # ASCII-ish "SFR1" marker in one uint32 word.
VERSION = 1
HEADER_WORDS = 7
FLAG_PROMPT_LOSS_MASK = 1 << 0
FLAG_RESPONSE_LOSS_MASK = 1 << 1
_KNOWN_FLAGS = FLAG_PROMPT_LOSS_MASK | FLAG_RESPONSE_LOSS_MASK
_UINT32_MAX = (1 << 32) - 1


@dataclasses.dataclass(frozen=True)
class TrainingRecord:
    prompt: tuple[int, ...]
    response: tuple[int, ...]
    bucket_id: int = 0
    prompt_loss_mask: tuple[bool, ...] | None = None
    response_loss_mask: tuple[bool, ...] | None = None

    def normalized_prompt_mask(self) -> tuple[bool, ...]:
        return _normalize_mask(self.prompt_loss_mask, len(self.prompt), "prompt")

    def normalized_response_mask(self) -> tuple[bool, ...]:
        return _normalize_mask(self.response_loss_mask, len(self.response), "response")


def encode_record(record: TrainingRecord) -> array:
    """Encode one record as uint32 words accepted by ``ShardWriter.add``."""
    _validate_words(record.prompt, "prompt")
    _validate_words(record.response, "response")
    if not 0 <= record.bucket_id <= _UINT32_MAX:
        raise ValueError("bucket_id must fit uint32")

    prompt_mask = record.normalized_prompt_mask()
    response_mask = record.normalized_response_mask()
    flags = 0
    payload = array("I", [MAGIC, VERSION, 0, record.bucket_id, len(record.prompt), len(record.response), 0])
    payload.extend(record.prompt)
    payload.extend(record.response)

    if record.prompt_loss_mask is not None:
        flags |= FLAG_PROMPT_LOSS_MASK
        payload.extend(_pack_mask(prompt_mask))
    if record.response_loss_mask is not None:
        flags |= FLAG_RESPONSE_LOSS_MASK
        payload.extend(_pack_mask(response_mask))
    payload[2] = flags
    payload[6] = len(payload)
    return payload


def decode_record(words: Sequence[int] | array) -> TrainingRecord:
    """Decode and fully validate one v1 training record."""
    if len(words) < HEADER_WORDS:
        raise ValueError("training record is shorter than its header")
    magic, version, flags, bucket_id, prompt_count, response_count, total_words = (
        int(words[index]) for index in range(HEADER_WORDS)
    )
    if magic != MAGIC:
        raise ValueError("training record magic mismatch")
    if version != VERSION:
        raise ValueError(f"unsupported training record version {version}")
    if flags & ~_KNOWN_FLAGS:
        raise ValueError(f"training record has unknown flags 0x{flags:x}")
    if total_words != len(words):
        raise ValueError("training record length does not match its header")

    cursor = HEADER_WORDS
    prompt_end = cursor + prompt_count
    response_end = prompt_end + response_count
    if response_end > len(words):
        raise ValueError("training record token counts exceed its payload")
    prompt = tuple(int(word) for word in words[cursor:prompt_end])
    response = tuple(int(word) for word in words[prompt_end:response_end])
    cursor = response_end

    prompt_mask = None
    if flags & FLAG_PROMPT_LOSS_MASK:
        mask_words = _mask_words(prompt_count)
        prompt_mask = _unpack_mask(words[cursor : cursor + mask_words], prompt_count)
        cursor += mask_words
    response_mask = None
    if flags & FLAG_RESPONSE_LOSS_MASK:
        mask_words = _mask_words(response_count)
        response_mask = _unpack_mask(words[cursor : cursor + mask_words], response_count)
        cursor += mask_words
    if cursor != len(words):
        raise ValueError("training record has trailing payload words")

    return TrainingRecord(
        prompt=prompt,
        response=response,
        bucket_id=bucket_id,
        prompt_loss_mask=prompt_mask,
        response_loss_mask=response_mask,
    )


def _validate_words(values: Iterable[int], name: str) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name}[{index}] must be an integer")
        if not 0 <= value <= _UINT32_MAX:
            raise ValueError(f"{name}[{index}] does not fit uint32")


def _normalize_mask(
    mask: Sequence[bool] | None, length: int, name: str
) -> tuple[bool, ...]:
    if mask is None:
        return (True,) * length
    if len(mask) != length:
        raise ValueError(f"{name}_loss_mask length differs from {name} tokens")
    if any(not isinstance(value, bool) for value in mask):
        raise TypeError(f"{name}_loss_mask values must be bool")
    return tuple(mask)


def _mask_words(length: int) -> int:
    return (length + 31) // 32


def _pack_mask(mask: Sequence[bool]) -> array:
    packed = array("I", [0] * _mask_words(len(mask)))
    for index, enabled in enumerate(mask):
        if enabled:
            packed[index // 32] |= 1 << (index % 32)
    return packed


def _unpack_mask(words: Sequence[int], length: int) -> tuple[bool, ...]:
    expected = _mask_words(length)
    if len(words) != expected:
        raise ValueError("loss mask is truncated")
    if length % 32 and words:
        valid_bits = (1 << (length % 32)) - 1
        if int(words[-1]) & ~valid_bits:
            raise ValueError("loss mask has non-zero padding bits")
    return tuple(bool(int(words[index // 32]) & (1 << (index % 32))) for index in range(length))
