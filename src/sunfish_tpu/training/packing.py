"""Dependency-free conversion from v1 records to fixed training fields."""

from __future__ import annotations

import dataclasses

from sunfish_tpu.training.record_format import TrainingRecord


@dataclasses.dataclass(frozen=True)
class PackedExample:
    prompt: tuple[int, ...]
    canvas: tuple[int, ...]
    canvas_id: tuple[int, ...]
    canvas_mask: tuple[bool, ...]
    canvas_loss_mask: tuple[bool, ...]
    encoder_target: tuple[int, ...]
    encoder_target_mask: tuple[bool, ...]
    bucket_id: int
    record_id: int


def pack_training_record(
    record: TrainingRecord,
    *,
    record_id: int,
    prompt_length: int,
    canvas_size: int,
    num_canvases: int,
    vocab_size: int,
    pad_token: int,
    eos_token: int,
) -> PackedExample:
    """Validate and pack a variable prompt/response without silent truncation."""
    if len(record.prompt) > prompt_length:
        raise ValueError(
            f"record {record_id} prompt has {len(record.prompt)} tokens; "
            f"configured maximum is {prompt_length}"
        )
    capacity = canvas_size * num_canvases
    if not record.response:
        raise ValueError(f"record {record_id} has no response tokens")
    if len(record.response) > capacity:
        raise ValueError(
            f"record {record_id} response has {len(record.response)} tokens; "
            f"configured canvas capacity is {capacity}"
        )
    for field_name, tokens in (("prompt", record.prompt), ("response", record.response)):
        for index, token in enumerate(tokens):
            if not 0 <= token < vocab_size:
                raise ValueError(
                    f"record {record_id} {field_name}[{index}]={token} is outside vocabulary"
                )
    if not 0 <= record_id < 1 << 32:
        raise ValueError("record id does not fit uint32")

    prompt = [pad_token] * prompt_length
    prompt[: len(record.prompt)] = record.prompt
    prompt_valid = [index < len(record.prompt) for index in range(prompt_length)]
    prompt_supervision = [False] * prompt_length
    prompt_supervision[: len(record.prompt)] = record.normalized_prompt_mask()

    canvas = [pad_token] * capacity
    canvas[: len(record.response)] = record.response
    valid_canvases = (len(record.response) + canvas_size - 1) // canvas_size
    valid_canvas_tokens = valid_canvases * canvas_size
    canvas[len(record.response) : valid_canvas_tokens] = [eos_token] * (
        valid_canvas_tokens - len(record.response)
    )
    canvas_mask = [index < valid_canvas_tokens for index in range(capacity)]
    canvas_loss_mask = [False] * capacity
    response_mask = record.normalized_response_mask()
    canvas_loss_mask[: len(response_mask)] = response_mask
    canvas_loss_mask[len(response_mask) : valid_canvas_tokens] = [
        response_mask[-1]
    ] * (valid_canvas_tokens - len(response_mask))
    canvas_id = [index // canvas_size for index in range(capacity)]

    full_sequence = prompt + canvas
    full_valid = prompt_valid + canvas_mask
    full_supervision = prompt_supervision + canvas_loss_mask
    encoder_target = full_sequence[1:] + [pad_token]
    encoder_target_mask = [
        full_valid[index]
        and full_valid[index + 1]
        and full_supervision[index + 1]
        for index in range(len(full_sequence) - 1)
    ] + [False]
    return PackedExample(
        prompt=tuple(prompt),
        canvas=tuple(canvas),
        canvas_id=tuple(canvas_id),
        canvas_mask=tuple(canvas_mask),
        canvas_loss_mask=tuple(canvas_loss_mask),
        encoder_target=tuple(encoder_target),
        encoder_target_mask=tuple(encoder_target_mask),
        bucket_id=record.bucket_id,
        record_id=record_id,
    )
