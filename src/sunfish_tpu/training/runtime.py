"""Runtime provenance and immutable run-identity checks."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
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

RUN_MANIFEST_NAME = "sunfish-run.json"


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
    try:
        direct_url_text = importlib.metadata.distribution("gemma").read_text(
            "direct_url.json"
        )
        direct_url = json.loads(direct_url_text or "{}")
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        installed_commit = None
    if installed_commit != GEMMA_SOURCE_COMMIT:
        errors.append(
            "gemma source commit is "
            f"{installed_commit or 'unrecorded'}, expected {GEMMA_SOURCE_COMMIT}"
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

    versions = verify_runtime_contract(require_tpu=require_tpu_runtime)
    payload = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "config_sha256": config.digest,
        "dataset_manifest_sha256": actual_manifest,
        "init_checkpoint": str(init_path),
        "init_checkpoint_format": config.checkpoint.format.value,
        "init_checkpoint_step": config.checkpoint.init_step,
        "phase": config.run.phase.value,
        "model": config.canonical_dict()["model"],
        "runtime_versions": versions,
        "gemma_source_commit": GEMMA_SOURCE_COMMIT,
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
