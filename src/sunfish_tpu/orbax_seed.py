"""Materialize the exact pruned JAX/Orbax seed used by TPU training.

This is a high-memory, single-host CPU Stage-0 job. It loads Google's official
DiffusionGemma JAX checkpoint, slices the same four expert-dependent leaves as
the audited safetensors converter, saves a normalized intermediate checkpoint,
then asks the pinned public Gemma checkpoint loader to reconcile that checkpoint
against the exact abstract target model tree. The final Orbax Standard
checkpoint is therefore directly consumable by ``ShardedOrbaxInitLoader``.

Never run this on Chase's laptop. The CLI requires an explicit acknowledgement
and verifies physical host memory before importing JAX.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sunfish.checkpoint_convert import load_selection_manifest_bytes
from sunfish.model_budget import DiffusionMoEBudget
from sunfish.source_tree import workspace_source_identity
from sunfish_tpu.seed_manifest import (
    canonical_layer_selection_sha256,
    selection_metadata_bytes,
)
from sunfish_tpu.source_identity import normalize_source_identity

SOURCE_EXPERTS = 128
NUM_LAYERS = 30
_AUDITED_BUDGET = DiffusionMoEBudget()
AUDITED_SOURCE_TEXT_PARAMETERS = _AUDITED_BUDGET.source_text_parameters

# Measured against the official gs://gemma-data JAX/Orbax checkpoint
# (2026-07-18): the Linen tree does not materialize the per-layer layer_scalar
# singletons that the safetensors export carries as separate tensors, and it
# fuses/organizes several embedding, router, and vision leaves differently.
# The net text-parameter difference versus the audited safetensors contract is
# exactly -30 (25,250,986,782 observed vs 25,250,986,812 audited). Safetensors
# headers remain canonical for all Sunfish math (AGENTS.md ground rule 5);
# this constant only reconciles the redundant count pre-gates for JAX-format
# trees, and equality is still enforced exactly. The binding structural gate is
# the exact-tree trace + leaf-by-leaf reconciliation below, which this delta
# does not touch.
OFFICIAL_JAX_TREE_TEXT_DELTA = -30
AUDITED_TARGET_TEXT_PARAMETERS_32E = int(
    _AUDITED_BUDGET.estimate(experts=32, top_k=4)["total_parameters"]
)
DEFAULT_MIN_HOST_MEMORY_GIB = 96
_LAYER = re.compile(r"^layer_(\d+)$")
_EXPECTED_KINDS = frozenset(
    {"gating_einsum", "linear", "router_logits", "per_expert_scale"}
)


def audited_target_text_parameters(experts: int) -> int:
    """Audited-formula target count for any supported expert ablation rung."""
    top_k = min(8, experts)
    return int(
        _AUDITED_BUDGET.estimate(experts=experts, top_k=top_k)[
            "total_parameters"
        ]
    )


@dataclasses.dataclass(frozen=True)
class PrunableLeaf:
    layer: int
    kind: str
    expert_axis: int


@dataclasses.dataclass(frozen=True)
class SelectionSnapshot:
    layers: dict[int, tuple[int, ...]]
    metadata: dict[str, Any]
    sha256: str
    layers_sha256: str


def load_selection_snapshot(
    path: Path,
    *,
    source_experts: int,
    retained_experts: int,
    top_k_experts: int,
) -> SelectionSnapshot:
    """Read selection bytes once and derive every selection attestation from them."""
    manifest_bytes = path.read_bytes()
    layers = load_selection_manifest_bytes(
        manifest_bytes,
        source_experts=source_experts,
        retained_experts=retained_experts,
        top_k_experts=top_k_experts,
        source=str(path),
    )
    if set(layers) != set(range(NUM_LAYERS)):
        raise ValueError("selection must contain exactly layers 0..29")
    metadata = selection_metadata_bytes(manifest_bytes)
    canonical_layers = {
        str(layer): list(layers[layer]) for layer in range(NUM_LAYERS)
    }
    if metadata.get("layers") != canonical_layers:
        raise ValueError("selection metadata layer IDs differ from pruning selection")
    layers_sha256 = canonical_layer_selection_sha256(canonical_layers)
    if metadata.get("layers_sha256") != layers_sha256:
        raise ValueError("selection metadata layer digest differs")
    return SelectionSnapshot(
        layers=layers,
        metadata=metadata,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        layers_sha256=layers_sha256,
    )


def require_unchanged_source_inventory(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    """Fail if the source prefix crossed object generations during its load."""
    if before != after:
        raise RuntimeError(
            "source checkpoint GCS inventory changed while loading parameters"
        )


def classify_prunable_path(path: Sequence[str]) -> PrunableLeaf | None:
    """Recognize nested Gemma 4 MoE leaves, with or without trailing ``w``."""
    layer = None
    layer_index = -1
    for index, component in enumerate(path):
        match = _LAYER.fullmatch(component)
        if match is not None:
            layer = int(match.group(1))
            layer_index = index
            break
    if layer is None:
        return None
    suffix = tuple(path[layer_index + 1 :])
    if suffix in {
        ("mlp", "gating_einsum"),
        ("mlp", "gating_einsum", "w"),
    }:
        return PrunableLeaf(layer, "gating_einsum", 0)
    if suffix in {("mlp", "linear"), ("mlp", "linear", "w")}:
        return PrunableLeaf(layer, "linear", 0)
    if suffix in {
        ("mlp", "router_logits"),
        ("mlp", "router_logits", "w"),
    }:
        return PrunableLeaf(layer, "router_logits", -1)
    if suffix == ("mlp", "per_expert_scale"):
        return PrunableLeaf(layer, "per_expert_scale", 0)
    return None


def validate_prunable_inventory(
    inventory: Mapping[int, set[str]], *, num_layers: int = NUM_LAYERS
) -> None:
    expected_layers = set(range(num_layers))
    if set(inventory) != expected_layers:
        raise ValueError(
            f"JAX checkpoint MoE layers are {sorted(inventory)}, expected {sorted(expected_layers)}"
        )
    for layer in range(num_layers):
        if inventory[layer] != _EXPECTED_KINDS:
            missing = sorted(_EXPECTED_KINDS - inventory[layer])
            extra = sorted(inventory[layer] - _EXPECTED_KINDS)
            raise ValueError(
                f"JAX layer {layer} prunable leaves differ; missing={missing}, extra={extra}"
            )


def physical_memory_bytes() -> int:
    """Return physical RAM without third-party imports, or zero if unknown."""
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0
    return pages * page_size


def _tree_signature(tree: Any, flax) -> dict[str, Any]:
    flat = flax.traverse_util.flatten_dict(tree, sep="/")
    leaves = []
    total_parameters = 0
    for path, value in sorted(flat.items()):
        shape = tuple(int(size) for size in value.shape)
        parameters = math.prod(shape)
        total_parameters += parameters
        leaves.append(
            {
                "path": path,
                "shape": list(shape),
                "dtype": str(value.dtype),
                "parameters": parameters,
            }
        )
    canonical = json.dumps(leaves, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "leaves": len(leaves),
        "parameters": total_parameters,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _metadata_tree(checkpoint_path: Any, ocp) -> Any:
    checkpointer = ocp.StandardCheckpointer()
    try:
        metadata = checkpointer.metadata(checkpoint_path)
    finally:
        checkpointer.close()
    item_metadata = getattr(metadata, "item_metadata", None)
    if item_metadata is None:
        raise ValueError(f"Orbax checkpoint has no item metadata: {checkpoint_path}")
    return item_metadata.tree


def require_parameter_count(
    signature: Mapping[str, Any], *, expected: int, label: str
) -> None:
    actual = int(signature.get("parameters", -1))
    if actual != expected:
        raise RuntimeError(
            f"{label} has {actual:,} parameters; audited contract is {expected:,}"
        )


def _target_abstract_params(
    *, num_experts: int, top_k_experts: int, jax, jnp
) -> Any:
    """Trace the exact wrapped Gemma target tree without allocating 8B weights."""
    from sunfish_tpu.training.model import make_gemma_network

    network = make_gemma_network(
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        dtype="bfloat16",
        use_lora=False,
        lora_rank=1,
    )

    def initialize(key):
        return network.init(
            key,
            time=jnp.zeros((1,), dtype=jnp.float32),
            xt=jnp.zeros((1, 1, 1), dtype=jnp.int32),
            conditioning={},
            is_training=False,
        )

    variables = jax.eval_shape(initialize, jax.random.key(0))
    params = variables["params"]
    if "gemma_model" not in params:
        raise KeyError("wrapped target init did not create params/gemma_model")
    return params["gemma_model"]


def _prune_nested_params(
    params: Any,
    *,
    selection: Mapping[int, tuple[int, ...]],
    retained_experts: int,
    flax,
    jax,
    jnp,
    selection_layers_sha256: str | None = None,
    delete_source_arrays: bool = True,
) -> tuple[Any, dict[str, Any]]:
    canonical_layers = {
        str(layer): list(selection[layer]) for layer in range(NUM_LAYERS)
    }
    actual_selection_layers_sha256 = canonical_layer_selection_sha256(
        canonical_layers
    )
    if (
        selection_layers_sha256 is not None
        and selection_layers_sha256 != actual_selection_layers_sha256
    ):
        raise ValueError("selection layer digest differs from pruning inputs")
    flat = flax.traverse_util.flatten_dict(params)
    inventory: dict[int, set[str]] = {}
    pruned_paths: list[str] = []
    for path, value in list(flat.items()):
        path_strings = tuple(str(component) for component in path)
        leaf = classify_prunable_path(path_strings)
        if leaf is None:
            continue
        inventory.setdefault(leaf.layer, set()).add(leaf.kind)
        if leaf.layer not in selection:
            raise ValueError(f"selection is missing JAX layer {leaf.layer}")
        axis = leaf.expert_axis % value.ndim
        if int(value.shape[axis]) != SOURCE_EXPERTS:
            raise ValueError(
                f"{'/'.join(path_strings)} expert axis is {value.shape[axis]}, "
                f"expected {SOURCE_EXPERTS}"
            )
        indices = jnp.asarray(selection[leaf.layer], dtype=jnp.int32)
        replacement = jnp.take(value, indices, axis=axis)
        jax.block_until_ready(replacement)
        if int(replacement.shape[axis]) != retained_experts:
            raise RuntimeError(f"pruned shape mismatch at {'/'.join(path_strings)}")
        flat[path] = replacement
        if delete_source_arrays and isinstance(value, jax.Array):
            value.delete()
        pruned_paths.append("/".join(path_strings))
    validate_prunable_inventory(inventory)
    if len(pruned_paths) != NUM_LAYERS * len(_EXPECTED_KINDS):
        raise RuntimeError(f"pruned {len(pruned_paths)} leaves, expected 120")
    return flax.traverse_util.unflatten_dict(flat), {
        "pruned_leaves": len(pruned_paths),
        "selection_layers_sha256": actual_selection_layers_sha256,
        "paths_sha256": hashlib.sha256(
            json.dumps(sorted(pruned_paths), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def materialize_orbax_seed(
    *,
    source: str,
    intermediate: str,
    output: str,
    manifest: str,
    selection_path: Path,
    retained_experts: int,
    top_k_experts: int,
    source_revision: str,
    source_anonymous: bool,
    min_host_memory_gib: int,
) -> dict[str, Any]:
    if not 0 < retained_experts < SOURCE_EXPERTS:
        raise ValueError("retained_experts must be in [1, 127]")
    if not 0 < top_k_experts <= retained_experts:
        raise ValueError("top_k_experts must be in [1, retained_experts]")
    if not source_revision.strip():
        raise ValueError("source_revision is required")
    memory = physical_memory_bytes()
    required = min_host_memory_gib * (1024**3)
    if memory == 0 or memory < required:
        raise RuntimeError(
            f"host has {memory / (1024**3):.1f} GiB RAM; "
            f"at least {min_host_memory_gib} GiB is required"
        )

    selection_snapshot = load_selection_snapshot(
        selection_path,
        source_experts=SOURCE_EXPERTS,
        retained_experts=retained_experts,
        top_k_experts=top_k_experts,
    )
    selection = selection_snapshot.layers
    selection_provenance = selection_snapshot.metadata
    if selection_provenance["source_experts"] != SOURCE_EXPERTS:
        raise ValueError("selection metadata source expert count differs")
    if selection_provenance["retained_experts"] != retained_experts:
        raise ValueError("selection metadata retained expert count differs")
    if selection_provenance["top_k_experts"] != top_k_experts:
        raise ValueError("selection metadata top-k differs")
    materializer_source_identity = workspace_source_identity(
        Path(__file__).resolve().parents[2]
    )
    if selection_provenance["promotion_allowed"] and (
        selection_provenance.get("source_revision") != source_revision
        or normalize_source_identity(selection_provenance.get("sunfish_source"))
        != normalize_source_identity(materializer_source_identity)
    ):
        raise ValueError(
            "approved production selection differs from materializer source lineage"
        )
    expected_target_parameters = (
        audited_target_text_parameters(retained_experts)
        + OFFICIAL_JAX_TREE_TEXT_DELTA
    )

    # Heavy imports happen only after the laptop/RAM guard.
    import flax
    from gemma import gm
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from etils import epath

    from sunfish_tpu.training.checkpoint import _validate_exact_tree
    from sunfish_tpu.gcs_inventory import build_gcs_inventory
    from sunfish_tpu.training.runtime import verify_runtime_contract

    versions = verify_runtime_contract(require_tpu=False)
    devices = list(jax.devices())
    if int(jax.process_count()) != 1 or {device.platform for device in devices} != {"cpu"}:
        raise RuntimeError(
            "Orbax seed materialization is a single-process CPU job; set JAX_PLATFORMS=cpu"
        )

    source_path = epath.Path(source)
    intermediate_path = epath.Path(intermediate)
    output_path = epath.Path(output)
    manifest_path = epath.Path(manifest)
    for candidate in (intermediate_path, output_path, manifest_path):
        if candidate.exists():
            raise FileExistsError(f"immutable output already exists: {candidate}")

    source_inventory = build_gcs_inventory(
        source, anonymous=source_anonymous
    )
    expected_source_revision = (
        f"gcs-inventory-sha256:{source_inventory['sha256']}"
    )
    if source_revision != expected_source_revision:
        raise ValueError(
            "source_revision does not match the live source checkpoint inventory: "
            f"expected {expected_source_revision}"
        )

    source_params = gm.ckpts.load_params(source_path, text_only=True)
    source_inventory_post_load = build_gcs_inventory(
        source, anonymous=source_anonymous
    )
    require_unchanged_source_inventory(
        source_inventory, source_inventory_post_load
    )
    source_signature = _tree_signature(source_params, flax)
    require_parameter_count(
        source_signature,
        expected=AUDITED_SOURCE_TEXT_PARAMETERS + OFFICIAL_JAX_TREE_TEXT_DELTA,
        label="official JAX text checkpoint",
    )
    pruned_nested, pruning = _prune_nested_params(
        source_params,
        selection=selection,
        retained_experts=retained_experts,
        flax=flax,
        jax=jax,
        jnp=jnp,
        selection_layers_sha256=selection_snapshot.layers_sha256,
    )
    del source_params
    pruned_signature = _tree_signature(pruned_nested, flax)
    require_parameter_count(
        pruned_signature,
        expected=expected_target_parameters,
        label="pruned nested checkpoint",
    )
    gm.ckpts.save_params(
        pruned_nested, intermediate_path, wait_until_finished=True
    )
    for value in jax.tree.leaves(pruned_nested):
        if isinstance(value, jax.Array):
            value.delete()
    del pruned_nested

    target_abstract = _target_abstract_params(
        num_experts=retained_experts,
        top_k_experts=top_k_experts,
        jax=jax,
        jnp=jnp,
    )
    exact_params = gm.ckpts.load_params(
        intermediate_path,
        params=target_abstract,
        donate=False,
        text_only=True,
    )
    _validate_exact_tree(target_abstract, exact_params)
    target_signature = _tree_signature(target_abstract, flax)
    require_parameter_count(
        target_signature,
        expected=expected_target_parameters,
        label="exact trainer target tree",
    )
    exact_signature = _tree_signature(exact_params, flax)
    if exact_signature != target_signature:
        raise RuntimeError("reconciled exact tree signature differs from target model")
    gm.ckpts.save_params(exact_params, output_path, wait_until_finished=True)
    saved_signature = _tree_signature(_metadata_tree(output_path, ocp), flax)
    if saved_signature != target_signature:
        raise RuntimeError("saved Orbax metadata differs from target model tree")
    output_inventory = build_gcs_inventory(output)

    payload = {
        "schema_version": 1,
        "source": str(source_path),
        "source_revision": source_revision,
        "source_gcs_inventory": source_inventory,
        "source_gcs_inventory_post_load": source_inventory_post_load,
        "source_tree": source_signature,
        "intermediate": str(intermediate_path),
        "pruned_nested_tree": pruned_signature,
        "output": str(output_path),
        "target_exact_tree": target_signature,
        "saved_tree": saved_signature,
        "output_gcs_inventory": output_inventory,
        "source_experts": SOURCE_EXPERTS,
        "retained_experts": retained_experts,
        "top_k_experts": top_k_experts,
        "layers": NUM_LAYERS,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": selection_snapshot.sha256,
        "selection_layers_sha256": selection_snapshot.layers_sha256,
        "selection_metadata": selection_provenance,
        "pruning": pruning,
        "runtime_versions": versions,
        "sunfish_source": materializer_source_identity,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "physical_memory_bytes": memory,
            "jax_devices": [str(device) for device in devices],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="official JAX Orbax checkpoint")
    parser.add_argument("--intermediate", required=True, help="new pruned nested Orbax path")
    parser.add_argument("--output", required=True, help="new exact-tree Orbax seed path")
    parser.add_argument("--manifest", required=True, help="new sidecar JSON path")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--retained-experts", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-anonymous",
        action="store_true",
        help="list the public source checkpoint without application credentials",
    )
    parser.add_argument(
        "--min-host-memory-gib", type=int, default=DEFAULT_MIN_HOST_MEMORY_GIB
    )
    parser.add_argument(
        "--ack-high-memory-cpu",
        action="store_true",
        help="required acknowledgement that this is not running on the laptop",
    )
    args = parser.parse_args(argv)
    if not args.ack_high_memory_cpu:
        parser.error("--ack-high-memory-cpu is required")
    try:
        payload = materialize_orbax_seed(
            source=args.source,
            intermediate=args.intermediate,
            output=args.output,
            manifest=args.manifest,
            selection_path=args.selection,
            retained_experts=args.retained_experts,
            top_k_experts=args.top_k,
            source_revision=args.source_revision,
            source_anonymous=args.source_anonymous,
            min_host_memory_gib=args.min_host_memory_gib,
        )
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"sunfish-orbax-seed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
