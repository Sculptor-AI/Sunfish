"""Strict dependency-free validator for the Stage-0 P1-P5 parity gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sunfish_tpu.parity_dependencies import (
    MODEL_CLASS,
    PARITY_RUNTIME_VERSIONS,
    TRANSFORMERS_RELEASE,
)
from sunfish_tpu.source_identity import normalize_source_identity

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKS = ("p1", "p2", "p3", "p4", "p5")
_P1_CHECKS = {
    "p1.1_tokenizer_identical",
    "p1.2_config_diff_exact",
    "p1.2_vision_config_null",
    "p1.3_tensor_set_exact",
    "p1.3_all_tensor_hashes_equal",
}
_TRACE_HASHES = {
    "upstream_float32",
    "control_float32",
    "upstream_bfloat16",
    "control_bfloat16",
}
_CATEGORIES = {"code": 8, "prose": 8, "multilingual": 8, "structured": 8}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Stage-0 parity report: {message}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_runtime(
    runtime: Any, *, expected_source: tuple[str, str]
) -> dict[str, Any]:
    _require(isinstance(runtime, Mapping), "runtime evidence is not an object")
    _require(
        runtime.get("versions") == PARITY_RUNTIME_VERSIONS,
        "parity runtime versions differ from the exact lock",
    )
    _require(runtime.get("device") == "cpu", "parity did not run on CPU")
    _require(
        runtime.get("deterministic_algorithms") is True,
        "deterministic PyTorch algorithms were not enabled",
    )
    _require(runtime.get("torch_num_threads") == 1, "PyTorch thread count is not one")
    _require(
        runtime.get("torch_num_interop_threads") == 1,
        "PyTorch interop thread count is not one",
    )
    _require(runtime.get("model_class") == MODEL_CLASS, "model class differs")
    _require(
        runtime.get("transformers_release") == TRANSFORMERS_RELEASE,
        "Transformers release differs",
    )
    _require(
        _is_sha256(runtime.get("torch_config_sha256")),
        "PyTorch build configuration hash is missing",
    )
    source = normalize_source_identity(runtime.get("sunfish_source"))
    _require(source == expected_source, "runtime source identity differs")
    return dict(runtime)


def _validate_prompt_fixture(value: Any) -> None:
    _require(isinstance(value, Mapping), "prompt fixture is not an object")
    _require(_is_sha256(value.get("sha256")), "prompt fixture hash is missing")
    prompts = value.get("prompts")
    _require(isinstance(prompts, list) and len(prompts) == 32, "expected 32 prompts")
    ids: set[str] = set()
    categories: Counter[str] = Counter()
    long_prompts = 0
    for prompt in prompts:
        _require(isinstance(prompt, Mapping), "prompt metadata is not an object")
        prompt_id = prompt.get("id")
        _require(
            isinstance(prompt_id, str) and prompt_id and prompt_id not in ids,
            "prompt IDs are missing or duplicated",
        )
        ids.add(prompt_id)
        category = prompt.get("category")
        _require(isinstance(category, str), f"prompt {prompt_id} has no category")
        categories[category] += 1
        token_count = prompt.get("token_count")
        _require(
            isinstance(token_count, int) and 256 <= token_count <= 2048,
            f"prompt {prompt_id} token count is outside 256..2048",
        )
        _require(
            _is_sha256(prompt.get("fixture_text_sha256"))
            and _is_sha256(prompt.get("token_ids_sha256")),
            f"prompt {prompt_id} hashes are missing",
        )
        crosses = prompt.get("crosses_sliding_window")
        _require(isinstance(crosses, bool), f"prompt {prompt_id} window flag is invalid")
        long_prompts += int(crosses)
    _require(dict(categories) == _CATEGORIES, "prompt category balance differs")
    _require(long_prompts >= 8, "fewer than eight prompts cross 1024 tokens")


def validate_stage0_parity_payload(
    payload: Any, *, expected_source: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one complete all-pass parity report and return a compact pin."""
    _require(isinstance(payload, Mapping), "root is not an object")
    _require(payload.get("schema_version") == 1, "schema version differs")
    _require(payload.get("stage") == "stage-0-parity", "stage differs")
    _require(payload.get("passed") is True, "top-level gate is not passed")
    revision = payload.get("upstream_revision")
    _require(
        isinstance(revision, str) and _REVISION.fullmatch(revision) is not None,
        "upstream revision is not an exact 40-hex commit",
    )
    normalized_expected = normalize_source_identity(expected_source)
    _require(normalized_expected is not None, "expected source identity is invalid")
    _require(
        normalize_source_identity(payload.get("sunfish_source"))
        == normalized_expected,
        "top-level source identity differs",
    )

    checks = payload.get("checks")
    _require(isinstance(checks, Mapping), "checks are not an object")
    _require(set(checks) == set(_CHECKS), "checks must be exactly P1 through P5")
    p1 = checks["p1"]
    _require(isinstance(p1, Mapping) and p1.get("passed") is True, "P1 did not pass")
    _require(p1.get("contract_errors") == [], "P1 has contract errors")
    static = p1.get("static_report")
    _require(isinstance(static, Mapping), "P1 static report is missing")
    _require(static.get("passed") is True, "P1 static report did not pass")
    _require(static.get("tensors_compared") == 691, "P1 did not compare 691 tensors")
    static_checks = static.get("checks")
    _require(
        isinstance(static_checks, Mapping)
        and set(static_checks) == _P1_CHECKS
        and all(value is True for value in static_checks.values()),
        "P1 static checks differ or are not all true",
    )
    for name in ("p2", "p3", "p4", "p5"):
        check = checks[name]
        _require(isinstance(check, Mapping), f"{name.upper()} is not an object")
        _require(check.get("passed") is True, f"{name.upper()} did not pass")
        _require(check.get("contract_errors") == [], f"{name.upper()} has contract errors")
        _require(check.get("differences") == 0, f"{name.upper()} has differences")
        _require(check.get("mismatches") == [], f"{name.upper()} has mismatch details")
        if name in {"p2", "p3", "p5"}:
            _require(
                isinstance(check.get("signature_count"), int)
                and check["signature_count"] > 0,
                f"{name.upper()} compared no logit signatures",
            )
            _require(check.get("max_abs_diff") == 0.0, f"{name.upper()} is not exact")
            _require(
                check.get("argmax_agreement") == 1.0,
                f"{name.upper()} argmax agreement is not exact",
            )

    environment = payload.get("environment")
    _require(isinstance(environment, Mapping), "environment is missing")
    _require(
        environment.get("requirements") == PARITY_RUNTIME_VERSIONS,
        "report requirements differ from the exact lock",
    )
    fp32_runtime = _validate_runtime(
        environment.get("float32"), expected_source=normalized_expected
    )
    bf16_runtime = _validate_runtime(
        environment.get("bfloat16"), expected_source=normalized_expected
    )
    _require(fp32_runtime == bf16_runtime, "float32 and bf16 runtimes differ")
    _validate_prompt_fixture(payload.get("prompt_fixture"))

    conversion = payload.get("conversion_manifest")
    _require(isinstance(conversion, Mapping), "conversion manifest is missing")
    for key, expected in (
        ("source_experts", 128),
        ("retained_experts", 128),
        ("top_k", 8),
        ("text_only", True),
    ):
        _require(conversion.get(key) == expected, f"conversion manifest {key} differs")

    checkpoints = payload.get("checkpoint_metadata")
    _require(isinstance(checkpoints, Mapping), "checkpoint metadata is missing")
    for role in ("upstream", "control"):
        checkpoint = checkpoints.get(role)
        _require(isinstance(checkpoint, Mapping), f"{role} checkpoint metadata is missing")
        _require(_is_sha256(checkpoint.get("fingerprint")), f"{role} fingerprint is invalid")
        metadata_hashes = checkpoint.get("metadata_sha256")
        _require(
            isinstance(metadata_hashes, Mapping)
            and metadata_hashes
            and all(_is_sha256(value) for value in metadata_hashes.values()),
            f"{role} checkpoint metadata hashes are invalid",
        )
    _require(
        "sunfish_conversion.json"
        in checkpoints["control"]["metadata_sha256"],
        "control checkpoint has no conversion manifest hash",
    )

    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, Mapping), "artifact pins are missing")
    for name in ("p1_report", "conversion_manifest"):
        artifact = artifacts.get(name)
        _require(
            isinstance(artifact, Mapping) and _is_sha256(artifact.get("sha256")),
            f"{name} artifact hash is invalid",
        )
    traces = artifacts.get("traces")
    _require(
        isinstance(traces, Mapping)
        and set(traces) == _TRACE_HASHES
        and all(_is_sha256(value) for value in traces.values()),
        "trace artifact hashes differ or are invalid",
    )
    return {
        "schema_version": 1,
        "stage": "stage-0-parity",
        "upstream_revision": revision,
        "sunfish_source": {
            "git_commit": normalized_expected[0],
            "source_tree_sha256": normalized_expected[1],
        },
        "checks_sha256": _canonical_sha256(checks),
        "trace_sha256": dict(traces),
        "p1_tensors_compared": 691,
    }


def validate_stage0_parity_report(
    path: Path, *, expected_source: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    """Read, hash, and validate a parity report from a local controller path."""
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("invalid Stage-0 parity report: invalid JSON") from error
    summary = validate_stage0_parity_payload(payload, expected_source=expected_source)
    return {**summary, "report_sha256": hashlib.sha256(raw).hexdigest()}, raw
