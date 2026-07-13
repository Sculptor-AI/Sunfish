"""Fail-fast static audit of the pinned TPU runtime source contracts.

This module intentionally imports no accelerator-adjacent package.  It reads
installed distribution files through :mod:`importlib.metadata`, parses their
Python source, and proves that the private APIs Sunfish relies on still have
the reviewed shape before bootstrap initializes JAX or a TPU backend.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
)


SOURCE_CONTRACTS: dict[str, tuple[str, str]] = {
    "orbax_standard_checkpointer": (
        "orbax-checkpoint",
        "orbax/checkpoint/_src/checkpointers/standard_checkpointer.py",
    ),
    "orbax_atomicity": (
        "orbax-checkpoint",
        "orbax/checkpoint/_src/path/atomicity_types.py",
    ),
    "kauldron_train_loop": ("kauldron", "kauldron/train/train_loop.py"),
    "kauldron_data": ("kauldron", "kauldron/data/py/base.py"),
    "gemma_models": ("gemma", "gemma/diffusion/_models.py"),
    "gemma_paths": ("gemma", "gemma/diffusion/_paths.py"),
    "gemma_base_model": ("gemma", "gemma/gm/nn/gemma4/_gemma4.py"),
    "gemma_hd_network": (
        "gemma",
        "gemma/diffusion/hackable_diffusion_adapter/hd/hd_gemma_network.py",
    ),
    "gemma_hd_lora": (
        "gemma",
        "gemma/diffusion/hackable_diffusion_adapter/hd/lora.py",
    ),
    "gemma_mask_helpers": (
        "gemma",
        "gemma/diffusion/hackable_diffusion_adapter/hd/mask_helpers.py",
    ),
    "gemma_moe": ("gemma", "gemma/gm/nn/gemma4/_moe.py"),
    "gemma_layers": ("gemma", "gemma/gm/nn/gemma4/_layers.py"),
    "gemma_modules": ("gemma", "gemma/gm/nn/gemma4/_modules.py"),
}

_OFFICIAL_CHECKPOINT = (
    "gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it"
)


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "positional": [
            arg.arg for arg in (*node.args.posonlyargs, *node.args.args)
        ],
        "keyword_only": [arg.arg for arg in node.args.kwonlyargs],
    }


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    return next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name),
        None,
    )


def _find_function(
    tree: ast.AST, name: str, *, class_name: str | None = None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    parent: ast.AST = tree
    if class_name is not None:
        found_class = _find_class(tree, class_name)
        if found_class is None:
            return None
        parent = found_class
    return next(
        (
            node
            for node in getattr(parent, "body", [])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def _call_name(call: ast.Call) -> str:
    parts: list[str] = []
    node: ast.AST = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _named_calls(tree: ast.AST, suffix: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node).endswith(suffix)
    ]


def _literal_assignments(tree: ast.AST) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = literal
    return values


def _keyword_literal(call: ast.Call, name: str) -> Any:
    keyword = next((item for item in call.keywords if item.arg == name), None)
    if keyword is None:
        return None
    try:
        return ast.literal_eval(keyword.value)
    except (ValueError, TypeError):
        return None


def _keyword_is_name(call: ast.Call, keyword: str, name: str) -> bool:
    item = next((item for item in call.keywords if item.arg == keyword), None)
    return item is not None and isinstance(item.value, ast.Name) and item.value.id == name


def _has_process_slice(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
            continue
        lower, step = node.slice.lower, node.slice.step
        if not isinstance(lower, ast.Call) or not isinstance(step, ast.Call):
            continue
        if _call_name(lower) == "jax.process_index" and _call_name(step) == "jax.process_count":
            return node.slice.upper is None
    return False


def _add(
    checks: list[dict[str, str]], name: str, passed: bool, detail: str
) -> None:
    checks.append(
        {"name": name, "status": "pass" if passed else "fail", "detail": detail}
    )


def audit_source_texts(
    sources: Mapping[str, str],
    *,
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit already-loaded source text; useful for tests and offline review."""
    checks: list[dict[str, str]] = []
    trees: dict[str, ast.Module] = {}
    records: list[dict[str, Any]] = []
    metadata = source_metadata or {}
    for key, (distribution, relative_path) in SOURCE_CONTRACTS.items():
        text = sources.get(key)
        if text is None:
            _add(checks, f"source:{key}", False, f"missing {distribution}:{relative_path}")
            continue
        try:
            tree = ast.parse(text, filename=relative_path)
        except SyntaxError as error:
            _add(checks, f"source:{key}", False, f"cannot parse: {error}")
            continue
        trees[key] = tree
        record = {
            "key": key,
            "distribution": distribution,
            "relative_path": relative_path,
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        record.update(metadata.get(key, {}))
        records.append(record)
        _add(checks, f"source:{key}", True, record["sha256"])

    def tree(key: str) -> ast.Module:
        return trees.get(key, ast.Module(body=[], type_ignores=[]))

    restore = _find_function(
        tree("orbax_standard_checkpointer"),
        "restore",
        class_name="StandardCheckpointer",
    )
    restore_signature = _function_signature(restore) if restore is not None else {}
    _add(
        checks,
        "orbax:standard-restore-signature",
        restore_signature
        == {"positional": ["self", "directory", "target"], "keyword_only": ["strict"]},
        json.dumps(restore_signature, sort_keys=True),
    )
    marker = _literal_assignments(tree("orbax_atomicity")).get("COMMIT_SUCCESS_FILE")
    _add(
        checks,
        "orbax:commit-marker",
        marker == "commit_success.txt",
        repr(marker),
    )

    train_tree = tree("kauldron_train_loop")
    composite_calls = _named_calls(train_tree, "checkpoint_state.CheckpointState")
    valid_composite_calls = [
        call
        for call in composite_calls
        if [ast.unparse(arg) for arg in call.args] == ["state", "chrono", "ds_iter"]
    ]
    restore_calls = _named_calls(train_tree, "ckpt.restore")
    save_calls = _named_calls(train_tree, "ckpt.save")
    metric_calls = _named_calls(train_tree, "writer.write_step_metrics")
    next_calls = [
        call
        for call in _named_calls(train_tree, "next")
        if call.args and ast.unparse(call.args[0]) == "ds_iter"
    ]
    step_calls = _named_calls(train_tree, "trainstep.step")
    _add(
        checks,
        "kauldron:composite-checkpoint-state",
        len(valid_composite_calls) >= 2 and bool(restore_calls) and bool(save_calls),
        f"composite={len(valid_composite_calls)}, restore={len(restore_calls)}, save={len(save_calls)}",
    )
    metric_step_ok = bool(metric_calls) and all(
        _keyword_is_name(call, "step", "i") for call in metric_calls
    )
    _add(
        checks,
        "kauldron:metric-step-label",
        metric_step_ok,
        "writer.write_step_metrics uses loop step i",
    )
    save_step_ok = bool(save_calls) and any(
        _keyword_is_name(call, "step", "i") for call in save_calls
    )
    _add(
        checks,
        "kauldron:checkpoint-step-label",
        save_step_ok,
        "ckpt.save uses loop step i",
    )
    ordered = bool(save_calls and next_calls and step_calls and metric_calls)
    if ordered:
        ordered = (
            min(call.lineno for call in save_calls)
            < min(call.lineno for call in next_calls)
            < min(call.lineno for call in step_calls)
            < min(call.lineno for call in metric_calls)
        )
    _add(
        checks,
        "kauldron:checkpoint-input-update-metric-order",
        ordered,
        "checkpoint precedes iterator mutation, update, and metric write",
    )

    data_method = _find_function(
        tree("kauldron_data"), "ds_for_current_process", class_name="DataSourceBase"
    )
    _add(
        checks,
        "kauldron:process-strided-input",
        data_method is not None and _has_process_slice(data_method),
        "ds[jax.process_index()::jax.process_count()]",
    )

    diffusion_model = _find_class(tree("gemma_models"), "DiffusionGemma_26B_A4B")
    _add(
        checks,
        "gemma:diffusion-model",
        diffusion_model is not None
        and _find_function(tree("gemma_models"), "setup", class_name="DiffusionGemma_26B_A4B")
        is not None,
        "DiffusionGemma_26B_A4B.setup",
    )
    checkpoint_literals = _literal_assignments(tree("gemma_paths"))
    _add(
        checks,
        "gemma:official-checkpoint-path",
        checkpoint_literals.get("DIFFUSIONGEMMA_26B_A4B_IT") == _OFFICIAL_CHECKPOINT,
        repr(checkpoint_literals.get("DIFFUSIONGEMMA_26B_A4B_IT")),
    )

    config_calls = _named_calls(tree("gemma_base_model"), "TransformerConfig")
    config_call = next(
        (
            call
            for call in config_calls
            if _keyword_literal(call, "num_experts") == 128
            and _keyword_literal(call, "top_k_experts") == 8
        ),
        None,
    )
    expected_config = {
        "num_embed": 262144,
        "embed_dim": 2816,
        "hidden_dim": 2112,
        "num_experts": 128,
        "expert_dim": 704,
        "top_k_experts": 8,
        "moe_dense_hidden_dim": 2112,
        "final_logit_softcap": 30.0,
        "sliding_window_size": 1024,
    }
    observed_config = {
        name: _keyword_literal(config_call, name) if config_call is not None else None
        for name in expected_config
    }
    _add(
        checks,
        "gemma:audited-base-config",
        observed_config == expected_config,
        json.dumps(observed_config, sort_keys=True),
    )

    hd_tree = tree("gemma_hd_network")
    hd_class = _find_class(hd_tree, "WrappedDiffusionGemmaNetwork")
    hd_ok = hd_class is not None and all(
        _find_function(hd_tree, name, class_name="WrappedDiffusionGemmaNetwork")
        is not None
        for name in ("init_cache", "encoder_call", "__call__")
    )
    hd_ok = hd_ok and _find_function(hd_tree, "prefill_kv_cache_with_encoder") is not None
    _add(
        checks,
        "gemma:hackable-network",
        hd_ok,
        "wrapper init_cache/encoder_call/__call__ plus prefill helper",
    )

    lora_tree = tree("gemma_hd_lora")
    expected_lora_signatures = {
        "_replace_by_lora": {
            "positional": ["module"],
            "keyword_only": ["rank", "dtype", "verbose", "target_modules"],
        },
        "_find_base_weight_key": {
            "positional": ["lora_parent", "original_flat"],
            "keyword_only": [],
        },
        "_compute_lora_delta": {
            "positional": ["a", "b"],
            "keyword_only": ["target_shape"],
        },
    }
    observed_lora_signatures = {}
    for name in expected_lora_signatures:
        node = _find_function(lora_tree, name)
        observed_lora_signatures[name] = _function_signature(node) if node else None
    _add(
        checks,
        "gemma:lora-private-apis",
        observed_lora_signatures == expected_lora_signatures,
        json.dumps(observed_lora_signatures, sort_keys=True),
    )

    mask_tree = tree("gemma_mask_helpers")
    mask_ok = all(
        _find_function(mask_tree, name) is not None
        for name in (
            "build_positions_from_mask",
            "make_causal_prefill_mask",
            "set_cache_end_index",
        )
    )
    _add(
        checks,
        "gemma:mask-cache-helpers",
        mask_ok,
        "positions/prefill-mask/cache-end-index helpers",
    )

    moe_tree = tree("gemma_moe")
    moe_call = _find_function(moe_tree, "__call__", class_name="MoERagged")
    moe_signature = _function_signature(moe_call) if moe_call else {}
    moe_ok = bool(moe_signature) and (
        _find_class(moe_tree, "_Weight") is not None
        and _find_class(moe_tree, "MoERagged") is not None
        and _find_function(moe_tree, "_router", class_name="MoERagged") is not None
        and moe_signature["positional"] == ["self", "x", "unnormalized_x"]
    )
    _add(
        checks,
        "gemma:ragged-moe-private-apis",
        moe_ok,
        json.dumps(moe_signature, sort_keys=True),
    )
    _add(
        checks,
        "gemma:reconstruction-modules",
        _find_class(tree("gemma_layers"), "RMSNorm") is not None
        and _find_class(tree("gemma_modules"), "FeedForward") is not None,
        "gemma4 RMSNorm and FeedForward",
    )

    passed = bool(checks) and all(check["status"] == "pass" for check in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        "gemma_source_commit": GEMMA_SOURCE_COMMIT,
        "sources": sorted(records, key=lambda item: item["key"]),
        "checks": checks,
    }


def audit_installed_runtime_apis() -> dict[str, Any]:
    """Load and audit the installed pinned distributions without importing them."""
    sources: dict[str, str] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    load_errors: list[dict[str, str]] = []
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for key, (name, relative_path) in SOURCE_CONTRACTS.items():
        try:
            distribution = distributions.get(name)
            if distribution is None:
                distribution = importlib.metadata.distribution(name)
                distributions[name] = distribution
            path = Path(distribution.locate_file(relative_path)).resolve()
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, importlib.metadata.PackageNotFoundError) as error:
            load_errors.append(
                {"name": f"installed-source:{key}", "status": "fail", "detail": str(error)}
            )
            continue
        sources[key] = text
        source_metadata[key] = {"installed_path": str(path)}

    report = audit_source_texts(sources, source_metadata=source_metadata)
    version_checks: list[dict[str, str]] = []
    for name in sorted({item[0] for item in SOURCE_CONTRACTS.values()}):
        expected = RUNTIME_VERSIONS[name]
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed = "not-installed"
        _add(
            version_checks,
            f"version:{name}",
            observed == expected,
            f"observed={observed}, expected={expected}",
        )
    try:
        direct_url_text = importlib.metadata.distribution("gemma").read_text(
            "direct_url.json"
        )
        direct_url = json.loads(direct_url_text or "{}")
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        installed_commit = None
    _add(
        version_checks,
        "gemma:installed-source-commit",
        installed_commit == GEMMA_SOURCE_COMMIT,
        f"observed={installed_commit or 'unrecorded'}, expected={GEMMA_SOURCE_COMMIT}",
    )
    report["checks"] = [*load_errors, *version_checks, *report["checks"]]
    report["passed"] = all(
        check["status"] == "pass" for check in report["checks"]
    )
    return report


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statically verify the installed pinned TPU runtime source APIs."
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_installed_runtime_apis()
    if args.output is not None:
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
