"""Executable Stage-0 P2-P5 parity harness for DiffusionGemma.

The two 25B text graphs are loaded sequentially so the exact float32 gate fits
on a high-memory CPU host. Full-vocabulary logits are never retained: each
position is projected, hashed, and released before the next position. A hash
match is a bitwise-equality proof; any mismatch fails closed.

This module deliberately imports PyTorch/Transformers only inside execution
functions. Importing the Sunfish package and running the ordinary unit suite
does not require the heavy parity environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sunfish_tpu.parity_dependencies import (
    MODEL_CLASS,
    PARITY_RUNTIME_VERSIONS,
    TRANSFORMERS_RELEASE,
)
from sunfish.source_tree import workspace_source_identity

SCHEMA_VERSION = 1
SEED = 20_260_710
MIN_PROMPT_TOKENS = 256
MAX_PROMPT_TOKENS = 2_048
SLIDING_BOUNDARY_EXERCISE = 1_024
CANVAS_LENGTH = 256
P2_PROMPTS = 32
P3_PROMPTS = 8
P4_PROMPTS = 16
P5_PROMPTS = 4
MAX_MISMATCH_DETAILS = 100
_UPSTREAM_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_upstream_revision(value: str) -> None:
    if _UPSTREAM_REVISION.fullmatch(value) is None:
        raise ValueError("upstream revision must be an exact lowercase 40-hex commit")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Fingerprint checkpoint metadata without re-reading the 50 GB weights."""
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    hashes: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "sunfish_conversion.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ):
        candidate = path / name
        if candidate.is_file():
            hashes[name] = _sha256_file(candidate)
    if "config.json" not in hashes or "model.safetensors.index.json" not in hashes:
        raise FileNotFoundError(
            f"{path} must contain config.json and model.safetensors.index.json"
        )
    return {
        "path": str(path),
        "metadata_sha256": hashes,
        "fingerprint": _canonical_sha256(hashes),
    }


def verify_runtime_contract() -> dict[str, str]:
    """Refuse an unpinned PyTorch parity graph before importing the model."""
    actual: dict[str, str] = {}
    errors: list[str] = []
    for distribution, expected in PARITY_RUNTIME_VERSIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{distribution} is not installed")
            continue
        actual[distribution] = version
        if version != expected:
            errors.append(f"{distribution}=={version}, expected {expected}")
    if errors:
        raise RuntimeError("parity runtime contract failed: " + "; ".join(errors))
    return actual


def _configure_torch(torch) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise RuntimeError(
                "PyTorch interop threads were initialized before parity setup"
            ) from error
    torch.manual_seed(SEED)


def _runtime_environment(torch, versions: dict[str, str]) -> dict[str, Any]:
    torch_config = torch.__config__.show()
    return {
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torch_config_sha256": _sha256_bytes(torch_config.encode("utf-8")),
        "model_class": MODEL_CLASS,
        "transformers_release": TRANSFORMERS_RELEASE,
        "sunfish_source": workspace_source_identity(
            Path(__file__).resolve().parents[2]
        ),
    }


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != P2_PROMPTS:
        raise ValueError(f"parity fixture must contain exactly {P2_PROMPTS} prompts")
    ids: set[str] = set()
    for prompt in prompts:
        prompt_id = prompt.get("id")
        text = prompt.get("text")
        if not isinstance(prompt_id, str) or not isinstance(text, str):
            raise ValueError("every parity prompt needs string id and text fields")
        if prompt_id in ids:
            raise ValueError(f"duplicate prompt id {prompt_id}")
        ids.add(prompt_id)
        actual = _sha256_bytes(text.encode("utf-8"))
        if prompt.get("sha256") != actual:
            raise ValueError(f"fixture text hash mismatch for {prompt_id}")
    return payload


def _token_ids_sha256(token_ids: list[int]) -> str:
    return _canonical_sha256(token_ids)


def _prepare_prompts(tokenizer, fixture_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixture = _load_fixture(fixture_path)
    prepared: list[dict[str, Any]] = []
    for prompt in fixture["prompts"]:
        raw_ids = tokenizer.encode(prompt["text"], add_special_tokens=True)
        if len(raw_ids) < MIN_PROMPT_TOKENS:
            raise ValueError(
                f"{prompt['id']} has {len(raw_ids)} tokens; expected at least {MIN_PROMPT_TOKENS}"
            )
        token_ids = [int(token) for token in raw_ids[:MAX_PROMPT_TOKENS]]
        prepared.append(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "fixture_text_sha256": prompt["sha256"],
                "raw_token_count": len(raw_ids),
                "token_count": len(token_ids),
                "truncated_to_contract_max": len(raw_ids) > MAX_PROMPT_TOKENS,
                "crosses_sliding_window": len(token_ids) > SLIDING_BOUNDARY_EXERCISE,
                "token_ids_sha256": _token_ids_sha256(token_ids),
                "token_ids": token_ids,
            }
        )

    categories: dict[str, int] = {}
    for prompt in prepared:
        categories[prompt["category"]] = categories.get(prompt["category"], 0) + 1
    expected_categories = {"code": 8, "prose": 8, "multilingual": 8, "structured": 8}
    if categories != expected_categories:
        raise ValueError(f"parity fixture category counts changed: {categories}")
    long_count = sum(prompt["crosses_sliding_window"] for prompt in prepared)
    if long_count < 8:
        raise ValueError(f"only {long_count} tokenized prompts cross 1024 tokens")

    metadata = {
        "path": str(fixture_path.resolve()),
        "sha256": _sha256_file(fixture_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "prompts": [
            {key: value for key, value in prompt.items() if key != "token_ids"}
            for prompt in prepared
        ],
    }
    return prepared, metadata


def _balanced_selection(
    prompts: list[dict[str, Any]], *, short_per_category: int, long_per_category: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category in ("code", "prose", "multilingual", "structured"):
        members = sorted(
            (prompt for prompt in prompts if prompt["category"] == category),
            key=lambda prompt: prompt["id"],
        )
        short = [prompt for prompt in members if not prompt["crosses_sliding_window"]]
        long = [prompt for prompt in members if prompt["crosses_sliding_window"]]
        if len(short) < short_per_category or len(long) < long_per_category:
            raise ValueError(f"not enough short/long prompts in {category}")
        selected.extend(short[:short_per_category])
        selected.extend(long[:long_per_category])
    return selected


def _bf16_selection(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _balanced_selection(prompts, short_per_category=0, long_per_category=1)


def _load_model(model_path: Path, dtype_name: str, vision_config_source: Path | None):
    import torch
    from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    config = DiffusionGemmaConfig.from_pretrained(
        str(model_path), local_files_only=True
    )
    injected_vision_config = False
    if config.vision_config is None:
        if vision_config_source is None:
            raise RuntimeError(
                "text-only checkpoint needs --vision-config-source for the shared HF class"
            )
        source_config = DiffusionGemmaConfig.from_pretrained(
            str(vision_config_source), local_files_only=True
        )
        if source_config.vision_config is None:
            raise RuntimeError("vision config source also has vision_config=null")
        # Transformers 5.13.0's model constructor still instantiates the vision
        # tower unconditionally. Injecting only the source architecture lets the
        # text-only checkpoint use the exact same class/graph; its missing vision
        # weights are initialized but never touched by text-only parity prompts.
        config.vision_config = source_config.vision_config
        injected_vision_config = True

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        str(model_path),
        config=config,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    first_device = next(model.parameters()).device
    if first_device.type != "cpu":
        raise RuntimeError(f"parity model was not loaded entirely on CPU: {first_device}")
    if float(model.final_logit_softcapping) != 30.0:
        raise RuntimeError(
            f"unexpected final_logit_softcapping={model.final_logit_softcapping}"
        )
    return model, injected_vision_config


def _tensor_signature(torch, row) -> dict[str, Any]:
    row = row.detach().to(device="cpu").contiguous()
    byte_view = row.view(torch.uint8)
    return {
        "sha256": _sha256_bytes(byte_view.numpy().tobytes()),
        "argmax": int(torch.argmax(row).item()),
        "dtype": str(row.dtype),
        "shape": list(row.shape),
    }


def _project_encoder_position(model, hidden_state):
    import torch

    logits = model.lm_head(hidden_state).to(torch.float32)
    softcap = model.final_logit_softcapping
    return torch.tanh(logits / softcap) * softcap


def _trace_encoder_logits(model, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    results: dict[str, Any] = {}
    errors: list[str] = []
    with torch.inference_mode():
        for prompt in prompts:
            input_ids = torch.tensor([prompt["token_ids"]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            encoder_outputs = model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = encoder_outputs.last_hidden_state
            positions: list[dict[str, Any]] = []
            for position in range(hidden.shape[1]):
                logits = _project_encoder_position(
                    model, hidden[:, position : position + 1, :]
                )
                positions.append(_tensor_signature(torch, logits[0, 0]))
                del logits
            if len(positions) != prompt["token_count"]:
                errors.append(
                    f"{prompt['id']}: traced {len(positions)} positions, expected {prompt['token_count']}"
                )
            results[prompt["id"]] = {
                "category": prompt["category"],
                "raw_token_count": prompt["raw_token_count"],
                "token_count": prompt["token_count"],
                "crosses_sliding_window": prompt["crosses_sliding_window"],
                "token_ids_sha256": prompt["token_ids_sha256"],
                "positions": positions,
            }
            del encoder_outputs, hidden, input_ids, attention_mask
            gc.collect()
    return {"contract_errors": errors, "prompts": results}


class _TraceStreamer:
    """Capture upstream generation drafts without retaining vocab tensors."""

    _takes_logits = True

    def __init__(self, *, capture_logits: bool):
        self.capture_logits = capture_logits
        self._put_calls = 0
        self._current_steps: list[Any] = []
        self.canvases: list[list[Any]] = []

    def put(self, _value) -> None:
        if self._put_calls > 0:
            self.canvases.append(self._current_steps)
            self._current_steps = []
        self._put_calls += 1

    def put_draft(self, *, logits=None, value=None) -> None:
        if logits is None:
            raise RuntimeError("parity streamer requested logits but received tokens")
        if self.capture_logits:
            import torch

            if logits.ndim != 3 or logits.shape[0] != 1:
                raise RuntimeError(f"unexpected draft logits shape {tuple(logits.shape)}")
            self._current_steps.append(
                [_tensor_signature(torch, logits[0, position]) for position in range(logits.shape[1])]
            )
        else:
            self._current_steps.append(None)

    def end(self) -> None:
        if self._current_steps:
            raise RuntimeError("generation ended before the final canvas was streamed")

    @property
    def steps_per_canvas(self) -> list[int]:
        return [len(canvas) for canvas in self.canvases]


def _generation_config(model, *, denoising_steps: int, new_tokens: int, p3: bool):
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationConfig,
        EntropyBoundSamplerConfig,
    )

    text_config = model.config.text_config
    return DiffusionGemmaGenerationConfig(
        max_new_tokens=new_tokens,
        max_denoising_steps=denoising_steps,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_min=0.4,
        t_max=0.8,
        # P3 must execute all four iterations. A history longer than the run
        # keeps the unmodified upstream stopping object from triggering early.
        stability_threshold=denoising_steps + 1 if p3 else 1,
        confidence_threshold=0.005,
        cache_implementation="dynamic",
        disable_compile=True,
        bos_token_id=text_config.bos_token_id,
        pad_token_id=text_config.pad_token_id,
        # P4's contract is exactly two canvases, including when the model emits
        # EOS in the first canvas. An empty EOS set preserves the upstream
        # stopping implementation while making that fixed-length contract
        # explicit. P3 has only one canvas, so its checkpoint EOS is harmless.
        eos_token_id=text_config.eos_token_id if p3 else [],
        return_dict_in_generate=True,
    )


def _trace_p3(model, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    results: dict[str, Any] = {}
    errors: list[str] = []
    config = _generation_config(
        model, denoising_steps=4, new_tokens=CANVAS_LENGTH, p3=True
    )
    for prompt in prompts:
        input_ids = torch.tensor([prompt["token_ids"]], dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        initial_canvas = torch.randint(
            0,
            model.config.text_config.vocab_size,
            (1, CANVAS_LENGTH),
            generator=generator,
            dtype=torch.long,
        )
        initial_canvas_sha256 = _token_ids_sha256(initial_canvas[0].tolist())
        torch.manual_seed(SEED)
        streamer = _TraceStreamer(capture_logits=True)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                decoder_input_ids=initial_canvas,
                generation_config=config,
                streamer=streamer,
            )
        if streamer.steps_per_canvas != [4]:
            errors.append(
                f"{prompt['id']}: P3 steps were {streamer.steps_per_canvas}, expected [4]"
            )
        steps = []
        for index, rows in enumerate(streamer.canvases[0] if streamer.canvases else []):
            steps.append(
                {
                    "remaining_step": 4 - index,
                    "temperature": round(0.8 - (0.1 * index), 1),
                    "positions": rows,
                }
            )
        generated = output.sequences[0, input_ids.shape[1] :].tolist()
        results[prompt["id"]] = {
            "category": prompt["category"],
            "token_count": prompt["token_count"],
            "crosses_sliding_window": prompt["crosses_sliding_window"],
            "token_ids_sha256": prompt["token_ids_sha256"],
            "initial_canvas_sha256": initial_canvas_sha256,
            "steps": steps,
            "steps_per_canvas": streamer.steps_per_canvas,
            "generated_token_ids": generated,
            "tokens_per_forward": output.tokens_per_forward.tolist(),
        }
        del output, streamer, input_ids, initial_canvas
        gc.collect()
    return {
        "contract_errors": errors,
        "seed": SEED,
        "temperatures": [0.8, 0.7, 0.6, 0.5],
        "prompts": results,
    }


def _trace_p4(model, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    results: dict[str, Any] = {}
    errors: list[str] = []
    config = _generation_config(
        model, denoising_steps=48, new_tokens=2 * CANVAS_LENGTH, p3=False
    )
    for prompt in prompts:
        input_ids = torch.tensor([prompt["token_ids"]], dtype=torch.long)
        torch.manual_seed(SEED)
        streamer = _TraceStreamer(capture_logits=False)
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                generation_config=config,
                streamer=streamer,
            )
        generated = output.sequences[0, input_ids.shape[1] :].tolist()
        if len(streamer.steps_per_canvas) != 2:
            errors.append(
                f"{prompt['id']}: generated {len(streamer.steps_per_canvas)} canvases, expected 2"
            )
        if len(generated) != 2 * CANVAS_LENGTH:
            errors.append(
                f"{prompt['id']}: generated {len(generated)} tokens, expected {2 * CANVAS_LENGTH}"
            )
        if any(not 1 <= count <= 48 for count in streamer.steps_per_canvas):
            errors.append(
                f"{prompt['id']}: invalid denoising counts {streamer.steps_per_canvas}"
            )
        results[prompt["id"]] = {
            "category": prompt["category"],
            "token_count": prompt["token_count"],
            "crosses_sliding_window": prompt["crosses_sliding_window"],
            "token_ids_sha256": prompt["token_ids_sha256"],
            "generated_token_ids": generated,
            "steps_per_canvas": streamer.steps_per_canvas,
            "tokens_per_forward": output.tokens_per_forward.tolist(),
        }
        del output, streamer, input_ids
        gc.collect()
    return {
        "contract_errors": errors,
        "seed": SEED,
        "sampler": {
            "entropy_bound": 0.1,
            "max_denoising_steps": 48,
            "t_min": 0.4,
            "t_max": 0.8,
            "stability_threshold": 1,
            "confidence_threshold": 0.005,
            "canvases": 2,
            "eos_stopping": False,
        },
        "prompts": results,
    }


def trace_checkpoint(
    *,
    role: str,
    model_path: Path,
    tokenizer_path: Path,
    vision_config_source: Path | None,
    fixture_path: Path,
    dtype_name: str,
    checks: tuple[str, ...],
    upstream_revision: str,
    output_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    _require_upstream_revision(upstream_revision)
    if output_path.exists() and not force:
        raise FileExistsError(f"trace already exists: {output_path}")
    versions = verify_runtime_contract()
    import torch
    from transformers import AutoTokenizer

    _configure_torch(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, use_fast=True
    )
    prompts, prompt_metadata = _prepare_prompts(tokenizer, fixture_path)
    metadata = checkpoint_metadata(model_path)
    model, injected_vision = _load_model(
        model_path, dtype_name, vision_config_source
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "role": role,
        "dtype": dtype_name,
        "checks_requested": list(checks),
        "upstream_revision": upstream_revision,
        "runtime": _runtime_environment(torch, versions),
        "loader": {
            "model_class": MODEL_CLASS,
            "dtype": dtype_name,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "attention_backend_override": None,
            "compiled": False,
            "injected_source_vision_architecture": injected_vision,
        },
        "checkpoint": metadata,
        "prompt_fixture": prompt_metadata,
        "checks": {},
    }
    try:
        if "p2" in checks:
            if dtype_name != "float32":
                raise ValueError("P2 must run in float32")
            payload["checks"]["p2"] = _trace_encoder_logits(model, prompts)
            if len(payload["checks"]["p2"]["prompts"]) != P2_PROMPTS:
                payload["checks"]["p2"]["contract_errors"].append(
                    f"P2 traced the wrong prompt count"
                )
        if "p3" in checks:
            if dtype_name != "float32":
                raise ValueError("P3 must run in float32")
            selected = _balanced_selection(
                prompts, short_per_category=1, long_per_category=1
            )
            payload["checks"]["p3"] = _trace_p3(model, selected)
        if "p4" in checks:
            if dtype_name != "float32":
                raise ValueError("P4 must run in float32")
            selected = _balanced_selection(
                prompts, short_per_category=2, long_per_category=2
            )
            payload["checks"]["p4"] = _trace_p4(model, selected)
        if "p5" in checks:
            if dtype_name != "bfloat16":
                raise ValueError("P5 must run in bfloat16")
            selected = _bf16_selection(prompts)
            payload["checks"]["p5"] = _trace_encoder_logits(model, selected)
    finally:
        del model
        gc.collect()
    _write_json(output_path, payload)
    return payload


def _record_mismatch(stats: dict[str, Any], path: str, left: Any, right: Any) -> None:
    stats["difference_count"] += 1
    if len(stats["mismatches"]) >= MAX_MISMATCH_DETAILS:
        return
    stats["mismatches"].append({"path": path, "upstream": left, "control": right})


def _compare_tree(left: Any, right: Any, path: str, stats: dict[str, Any]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        signature_keys = {"sha256", "argmax", "dtype", "shape"}
        if signature_keys.issubset(left) and signature_keys.issubset(right):
            stats["signature_count"] += 1
            if left["argmax"] == right["argmax"]:
                stats["argmax_matches"] += 1
            if left["sha256"] != right["sha256"]:
                stats["signature_digest_mismatches"] += 1
            if left != right:
                _record_mismatch(stats, path, left, right)
            return
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            _record_mismatch(stats, path + ".keys", sorted(left_keys), sorted(right_keys))
        for key in sorted(left_keys & right_keys):
            _compare_tree(left[key], right[key], f"{path}.{key}", stats)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            _record_mismatch(stats, path + ".length", len(left), len(right))
        common = min(len(left), len(right))
        if all(not isinstance(item, (dict, list)) for item in left[:common]) and all(
            not isinstance(item, (dict, list)) for item in right[:common]
        ):
            if left != right:
                _record_mismatch(
                    stats,
                    path,
                    {"length": len(left), "sha256": _canonical_sha256(left)},
                    {"length": len(right), "sha256": _canonical_sha256(right)},
                )
            return
        for index in range(common):
            _compare_tree(left[index], right[index], f"{path}[{index}]", stats)
        return
    if type(left) is not type(right) or left != right:
        _record_mismatch(stats, path, left, right)


def compare_check(name: str, upstream: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "difference_count": 0,
        "mismatches": [],
        "signature_count": 0,
        "signature_digest_mismatches": 0,
        "argmax_matches": 0,
    }
    _compare_tree(upstream, control, name, stats)
    contract_errors = sorted(
        set(upstream.get("contract_errors", []))
        | set(control.get("contract_errors", []))
    )
    signatures = stats["signature_count"]
    argmax_agreement = (
        stats["argmax_matches"] / signatures if signatures else None
    )
    exact_logits = (
        stats["signature_digest_mismatches"] == 0
        and stats["difference_count"] == 0
    )
    passed = stats["difference_count"] == 0 and not contract_errors
    result = {
        "passed": passed,
        "contract_errors": contract_errors,
        "differences": stats["difference_count"],
        "mismatches": stats["mismatches"],
        "signature_count": signatures,
        "argmax_agreement": argmax_agreement,
        "max_abs_diff": 0.0 if signatures and exact_logits else None,
    }
    if signatures and not exact_logits:
        result["max_abs_diff_note"] = (
            "not materialized after a streaming hash mismatch; the exactness gate failed"
        )
    return result


def _load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema in {path}")
    return payload


def _require_trace_pair(
    upstream: dict[str, Any], control: dict[str, Any], *, dtype_name: str, checks: set[str]
) -> None:
    if upstream.get("role") != "upstream" or control.get("role") != "control":
        raise ValueError("trace roles must be upstream and control")
    if upstream.get("dtype") != dtype_name or control.get("dtype") != dtype_name:
        raise ValueError(f"trace dtype mismatch; expected {dtype_name}")
    if upstream.get("runtime") != control.get("runtime"):
        raise ValueError("upstream/control traces were not produced in the same runtime")
    if upstream.get("prompt_fixture") != control.get("prompt_fixture"):
        raise ValueError("upstream/control tokenizer outputs or prompt fixture differ")
    if set(upstream.get("checks", {})) != checks or set(control.get("checks", {})) != checks:
        raise ValueError(f"trace checks must be exactly {sorted(checks)}")
    if upstream.get("upstream_revision") != control.get("upstream_revision"):
        raise ValueError("trace upstream revisions differ")


def compare_traces(
    *,
    upstream_fp32_path: Path,
    control_fp32_path: Path,
    upstream_bf16_path: Path,
    control_bf16_path: Path,
    p1_report_path: Path,
    conversion_manifest_path: Path,
    upstream_revision: str,
    output_path: Path,
) -> dict[str, Any]:
    _require_upstream_revision(upstream_revision)
    upstream_fp32 = _load_trace(upstream_fp32_path)
    control_fp32 = _load_trace(control_fp32_path)
    upstream_bf16 = _load_trace(upstream_bf16_path)
    control_bf16 = _load_trace(control_bf16_path)
    _require_trace_pair(
        upstream_fp32, control_fp32, dtype_name="float32", checks={"p2", "p3", "p4"}
    )
    _require_trace_pair(
        upstream_bf16, control_bf16, dtype_name="bfloat16", checks={"p5"}
    )
    if upstream_fp32["prompt_fixture"] != upstream_bf16["prompt_fixture"]:
        raise ValueError("float32 and bfloat16 traces used different prompt tokens")
    if upstream_fp32["runtime"] != upstream_bf16["runtime"]:
        raise ValueError("float32 and bfloat16 traces used different runtimes")
    revisions = {
        upstream_fp32["upstream_revision"],
        upstream_bf16["upstream_revision"],
        upstream_revision,
    }
    if len(revisions) != 1:
        raise ValueError(f"upstream revision mismatch: {sorted(revisions)}")

    p1 = json.loads(p1_report_path.read_text(encoding="utf-8"))
    conversion = json.loads(conversion_manifest_path.read_text(encoding="utf-8"))
    p1_passed = bool(p1.get("passed")) and all(p1.get("checks", {}).values())
    manifest_errors = []
    expected_manifest = {
        "source_experts": 128,
        "retained_experts": 128,
        "top_k": 8,
        "text_only": True,
    }
    for key, expected in expected_manifest.items():
        if conversion.get(key) != expected:
            manifest_errors.append(
                f"conversion manifest {key}={conversion.get(key)!r}, expected {expected!r}"
            )

    checks = {
        "p1": {
            "passed": p1_passed and not manifest_errors,
            "static_report": p1,
            "contract_errors": manifest_errors,
        },
        "p2": compare_check(
            "p2", upstream_fp32["checks"]["p2"], control_fp32["checks"]["p2"]
        ),
        "p3": compare_check(
            "p3", upstream_fp32["checks"]["p3"], control_fp32["checks"]["p3"]
        ),
        "p4": compare_check(
            "p4", upstream_fp32["checks"]["p4"], control_fp32["checks"]["p4"]
        ),
        "p5": compare_check(
            "p5", upstream_bf16["checks"]["p5"], control_bf16["checks"]["p5"]
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "stage": "stage-0-parity",
        "upstream_revision": upstream_revision,
        "sunfish_source": upstream_fp32["runtime"]["sunfish_source"],
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "environment": {
            "float32": upstream_fp32["runtime"],
            "bfloat16": upstream_bf16["runtime"],
            "requirements": PARITY_RUNTIME_VERSIONS,
        },
        "checkpoint_metadata": {
            "upstream": upstream_fp32["checkpoint"],
            "control": control_fp32["checkpoint"],
        },
        "prompt_fixture": upstream_fp32["prompt_fixture"],
        "conversion_manifest": conversion,
        "artifacts": {
            "p1_report": {
                "path": str(p1_report_path.resolve()),
                "sha256": _sha256_file(p1_report_path),
            },
            "conversion_manifest": {
                "path": str(conversion_manifest_path.resolve()),
                "sha256": _sha256_file(conversion_manifest_path),
            },
            "traces": {
                "upstream_float32": _sha256_file(upstream_fp32_path),
                "control_float32": _sha256_file(control_fp32_path),
                "upstream_bfloat16": _sha256_file(upstream_bf16_path),
                "control_bfloat16": _sha256_file(control_bf16_path),
            },
        },
    }
    _write_json(output_path, payload)
    return payload


def _resume_trace_ok(
    path: Path,
    *,
    role: str,
    dtype_name: str,
    checks: tuple[str, ...],
    upstream_revision: str,
    model_path: Path,
    fixture_path: Path,
) -> bool:
    if not path.exists():
        return False
    try:
        trace = _load_trace(path)
        current_source = workspace_source_identity(
            Path(__file__).resolve().parents[2]
        )
        return (
            trace.get("role") == role
            and trace.get("dtype") == dtype_name
            and set(trace.get("checks", {})) == set(checks)
            and trace.get("upstream_revision") == upstream_revision
            and trace.get("checkpoint", {}).get("fingerprint")
            == checkpoint_metadata(model_path)["fingerprint"]
            and trace.get("prompt_fixture", {}).get("sha256") == _sha256_file(fixture_path)
            and trace.get("runtime", {}).get("versions") == PARITY_RUNTIME_VERSIONS
            and trace.get("runtime", {}).get("sunfish_source") == current_source
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_all(args) -> dict[str, Any]:
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    trace_specs = [
        ("upstream", args.source, "float32", ("p2", "p3", "p4")),
        ("control", args.control, "float32", ("p2", "p3", "p4")),
        ("upstream", args.source, "bfloat16", ("p5",)),
        ("control", args.control, "bfloat16", ("p5",)),
    ]
    paths: dict[tuple[str, str], Path] = {}
    for role, model_path, dtype_name, checks in trace_specs:
        output = workdir / f"trace.{role}.{dtype_name}.json"
        paths[(role, dtype_name)] = output
        if args.resume and _resume_trace_ok(
            output,
            role=role,
            dtype_name=dtype_name,
            checks=checks,
            upstream_revision=args.upstream_revision,
            model_path=model_path,
            fixture_path=args.prompts,
        ):
            continue
        trace_checkpoint(
            role=role,
            model_path=model_path,
            tokenizer_path=args.source,
            vision_config_source=args.source,
            fixture_path=args.prompts,
            dtype_name=dtype_name,
            checks=checks,
            upstream_revision=args.upstream_revision,
            output_path=output,
            force=args.force or args.resume,
        )
    return compare_traces(
        upstream_fp32_path=paths[("upstream", "float32")],
        control_fp32_path=paths[("control", "float32")],
        upstream_bf16_path=paths[("upstream", "bfloat16")],
        control_bf16_path=paths[("control", "bfloat16")],
        p1_report_path=args.p1_report,
        conversion_manifest_path=args.conversion_manifest,
        upstream_revision=args.upstream_revision,
        output_path=workdir / "report.json",
    )


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="trace one checkpoint/dtype")
    trace.add_argument("--role", choices=("upstream", "control"), required=True)
    trace.add_argument("--model", type=_path, required=True)
    trace.add_argument("--tokenizer", type=_path, required=True)
    trace.add_argument("--vision-config-source", type=_path)
    trace.add_argument("--prompts", type=_path, required=True)
    trace.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    trace.add_argument("--checks", choices=("p2", "p3", "p4", "p5"), nargs="+", required=True)
    trace.add_argument("--upstream-revision", required=True)
    trace.add_argument("--output", type=_path, required=True)
    trace.add_argument("--force", action="store_true")

    compare = subparsers.add_parser("compare", help="compare four completed traces")
    compare.add_argument("--upstream-fp32", type=_path, required=True)
    compare.add_argument("--control-fp32", type=_path, required=True)
    compare.add_argument("--upstream-bf16", type=_path, required=True)
    compare.add_argument("--control-bf16", type=_path, required=True)
    compare.add_argument("--p1-report", type=_path, required=True)
    compare.add_argument("--conversion-manifest", type=_path, required=True)
    compare.add_argument("--upstream-revision", required=True)
    compare.add_argument("--output", type=_path, required=True)

    run = subparsers.add_parser("run", help="run all traces sequentially and compare")
    run.add_argument("--source", type=_path, required=True)
    run.add_argument("--control", type=_path, required=True)
    run.add_argument(
        "--prompts",
        type=_path,
        default=Path("tests/fixtures/parity_prompts.json"),
    )
    run.add_argument("--p1-report", type=_path, required=True)
    run.add_argument("--conversion-manifest", type=_path, required=True)
    run.add_argument("--upstream-revision", required=True)
    run.add_argument("--workdir", type=_path, required=True)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "trace":
            payload = trace_checkpoint(
                role=args.role,
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                vision_config_source=args.vision_config_source,
                fixture_path=args.prompts,
                dtype_name=args.dtype,
                checks=tuple(dict.fromkeys(args.checks)),
                upstream_revision=args.upstream_revision,
                output_path=args.output,
                force=args.force,
            )
            passed = not any(
                check.get("contract_errors") for check in payload["checks"].values()
            )
        elif args.command == "compare":
            payload = compare_traces(
                upstream_fp32_path=args.upstream_fp32,
                control_fp32_path=args.control_fp32,
                upstream_bf16_path=args.upstream_bf16,
                control_bf16_path=args.control_bf16,
                p1_report_path=args.p1_report,
                conversion_manifest_path=args.conversion_manifest,
                upstream_revision=args.upstream_revision,
                output_path=args.output,
            )
            passed = payload["passed"]
        else:
            payload = run_all(args)
            passed = payload["passed"]
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"sunfish-parity: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": passed}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
