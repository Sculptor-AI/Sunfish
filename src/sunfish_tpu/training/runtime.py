"""Runtime provenance and immutable run-identity checks."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
from typing import Any

from etils import epath
from jax.experimental import multihost_utils

from sunfish_tpu.training.data import manifest_sha256
from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)
from sunfish_tpu.training.spec import CheckpointFormat, HarnessConfig
from sunfish_tpu.seed_manifest import validate_seed_manifest_bytes
from sunfish_tpu.gcs_inventory import verify_live_gcs_inventory
from sunfish_tpu.runtime_provenance import resolve_gemma_source_commit
from sunfish_tpu.source_identity import (
    require_launcher_run_id,
    source_identity_from_environment,
)

RUN_MANIFEST_NAME = "sunfish-run.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def verify_runtime_contract(*, require_tpu: bool) -> dict[str, str]:
    """Fail before compilation if the pinned training stack drifted."""
    actual: dict[str, str] = {}
    errors: list[str] = []
    expected_versions = dict(RUNTIME_VERSIONS)
    if require_tpu:
        expected_versions.update(TPU_ONLY_RUNTIME_VERSIONS)
    for distribution, expected in expected_versions.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{distribution} is not installed")
            continue
        actual[distribution] = version
        if version != expected:
            errors.append(f"{distribution}=={version}, expected {expected}")

    if importlib.util.find_spec("gemma.diffusion.hackable_diffusion_adapter") is None:
        errors.append("Gemma DiffusionGemma training adapter is unavailable")
    provenance = "direct-url"
    try:
        installed_commit, provenance = resolve_gemma_source_commit()
    except (OSError, RuntimeError, ValueError) as error:
        installed_commit = None
        provenance = "invalid-offline-bundle"
        errors.append(f"Gemma source provenance could not be verified: {error}")
    if installed_commit != GEMMA_SOURCE_COMMIT:
        errors.append(
            "gemma source commit is "
            f"{installed_commit or 'unrecorded'}, expected {GEMMA_SOURCE_COMMIT} "
            f"({provenance})"
        )
    if errors:
        raise RuntimeError("training runtime contract failed: " + "; ".join(errors))
    return actual


def ensure_run_identity(
    config: HarnessConfig,
    jax: Any,
    *,
    require_tpu_runtime: bool,
) -> dict[str, Any]:
    """Create/read one byte-stable identity document across all workers.

    The identity prevents a resumed Grain cursor from being applied to a
    replaced dataset and prevents checkpoints from being reused under a
    changed objective, topology, model, or dependency stack.
    """
    require_launcher_run_id(config.run.run_id, required=require_tpu_runtime)
    actual_manifest = manifest_sha256(config.data.directory)
    if actual_manifest != config.data.manifest_sha256:
        raise ValueError(
            "configured dataset hash does not match manifest.json: "
            f"expected {config.data.manifest_sha256}, got {actual_manifest}"
        )

    init_path = epath.Path(config.checkpoint.init_path)
    if not init_path.exists():
        raise FileNotFoundError(f"initial Orbax checkpoint does not exist: {init_path}")
    if config.checkpoint.format is CheckpointFormat.KAULDRON_PARAMS:
        promoted_step = (
            init_path
            / "checkpoints"
            / f"ckpt_{config.checkpoint.init_step}"
        )
        if not promoted_step.exists():
            raise FileNotFoundError(
                f"pinned Kauldron checkpoint does not exist: {promoted_step}"
            )
        seed_manifest_sha256 = ""
    else:
        seed_manifest_path = epath.Path(config.checkpoint.init_manifest_path)
        if not seed_manifest_path.exists():
            raise FileNotFoundError(
                f"initial seed manifest does not exist: {seed_manifest_path}"
            )
        seed_payload = validate_seed_manifest_bytes(
            seed_manifest_path.read_bytes(),
            expected_sha256=config.checkpoint.init_manifest_sha256,
            init_path=config.checkpoint.init_path,
            phase=config.run.phase.value,
            expected_num_experts=config.model.num_experts,
            expected_top_k_experts=config.model.top_k_experts,
        )
        if require_tpu_runtime:
            verify_live_gcs_inventory(
                config.checkpoint.init_path,
                seed_payload["output_gcs_inventory"],
            )
        seed_manifest_sha256 = config.checkpoint.init_manifest_sha256

    versions = verify_runtime_contract(require_tpu=require_tpu_runtime)
    source_identity = source_identity_from_environment(
        required=require_tpu_runtime
    )
    config_file_sha256 = os.environ.get("SUNFISH_CONFIG_FILE_SHA256", "")
    if require_tpu_runtime and not _SHA256.fullmatch(config_file_sha256):
        raise RuntimeError(
            "SUNFISH_CONFIG_FILE_SHA256 must be set by the all-host launcher"
        )
    payload = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "config_sha256": config.digest,
        "config_file_sha256": config_file_sha256 or "unrecorded",
        "dataset_manifest_sha256": actual_manifest,
        "init_checkpoint": str(init_path),
        "init_checkpoint_format": config.checkpoint.format.value,
        "init_checkpoint_step": config.checkpoint.init_step,
        "init_checkpoint_manifest": config.checkpoint.init_manifest_path,
        "init_checkpoint_manifest_sha256": seed_manifest_sha256,
        "phase": config.run.phase.value,
        "model": config.canonical_dict()["model"],
        "runtime_versions": versions,
        "gemma_source_commit": GEMMA_SOURCE_COMMIT,
        "sunfish_source": source_identity,
        "topology": {
            "device_count": int(jax.device_count()),
            "process_count": int(jax.process_count()),
            "local_device_count": int(jax.local_device_count()),
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    workdir = epath.Path(config.run.workdir)
    identity_path = workdir / RUN_MANIFEST_NAME

    if int(jax.process_index()) == 0:
        workdir.mkdir(parents=True, exist_ok=True)
        if identity_path.exists():
            _require_same_identity(identity_path, serialized)
        else:
            identity_path.write_text(serialized)
    multihost_utils.sync_global_devices(f"sunfish-run-identity-{config.run.run_id}")
    if not identity_path.exists():
        raise RuntimeError(f"lead worker did not publish {identity_path}")
    _require_same_identity(identity_path, serialized)
    return payload


def _require_same_identity(path: epath.Path, expected: str) -> None:
    actual = path.read_text()
    if actual != expected:
        try:
            actual_payload = json.loads(actual)
            expected_payload = json.loads(expected)
            changed = sorted(
                key
                for key in set(actual_payload) | set(expected_payload)
                if actual_payload.get(key) != expected_payload.get(key)
            )
        except json.JSONDecodeError:
            changed = ["invalid-json"]
        raise RuntimeError(
            f"refusing to reuse {path}; immutable run identity changed: {changed}"
        )
