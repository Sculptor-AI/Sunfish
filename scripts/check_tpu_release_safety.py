#!/usr/bin/env python3
"""Fail closed if TPU worker transport, egress, or allocation policy regresses."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
IAP_WRAPPER = SCRIPTS / "tpu_iap.sh"
SELF = Path(__file__).resolve()
WORKER_SCRIPTS = (
    SCRIPTS / "bootstrap_tpu.sh",
    SCRIPTS / "deploy_tpu_offline_bundle.sh",
    SCRIPTS / "interrupt_training_attempt.sh",
    SCRIPTS / "launch_tpu_pod.sh",
    SCRIPTS / "probe_tpu_worker_base.sh",
    SCRIPTS / "tpu_host_entrypoint.sh",
    SCRIPTS / "upload_tpu_configs.sh",
)
MUTATING_TPU_VM_COMMANDS = (
    "attach-disk",
    "create",
    "delete",
    "detach-disk",
    "perform-maintenance",
    "reset",
    "restart",
    "resume",
    "simulate-maintenance-event",
    "start",
    "stop",
    "suspend",
    "update",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    wrapper = IAP_WRAPPER.read_text(encoding="utf-8")
    for token in (
        "alpha compute tpus tpu-vm ssh",
        "alpha compute tpus tpu-vm scp",
        "--worker=all",
        "--batch-size=all",
        "--tunnel-through-iap",
    ):
        if token not in wrapper:
            fail(f"IAP transport wrapper is missing {token!r}")
    if "tpu-vm[[:space:]]+[a-z0-9-]+" not in wrapper:
        fail("IAP transport wrapper does not fail closed on worker control-plane commands")

    for path in WORKER_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        if path != IAP_WRAPPER and re.search(r"\bcompute\s+tpus\s+tpu-vm\b", text):
            fail(f"direct TPU VM transport bypasses tpu_iap.sh: {path.name}")
        if re.search(r"https?://|git\+https|\bgit\s+clone\b|\bcurl\b|\bwget\b", text):
            fail(f"worker path contains a public-network operation: {path.name}")
        if "pip install --upgrade" in text or "requirements-gemma-source.lock" in text:
            fail(f"worker bootstrap can resolve online dependencies: {path.name}")

    bootstrap = (SCRIPTS / "bootstrap_tpu.sh").read_text(encoding="utf-8")
    for required in (
        "PIP_NO_INDEX=1",
        "--no-index",
        "--no-deps",
        "--only-binary=:all:",
        "offline-requirements.lock",
        "sunfish_tpu.offline_bundle verify",
        "sunfish_tpu.offline_bundle verify-installed",
    ):
        if required not in bootstrap:
            fail(f"offline worker bootstrap is missing {required!r}")

    host_entrypoint = (SCRIPTS / "tpu_host_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    if "export SUNFISH_TPU_WORKER=1" not in host_entrypoint:
        fail("all-worker entrypoint does not mark the TPU worker environment")
    connected_only = (
        SCRIPTS / "build_tpu_offline_bundle.sh",
        SCRIPTS / "bootstrap_seed_cpu.sh",
        SCRIPTS / "bootstrap_parity.sh",
        SCRIPTS / "bootstrap_tpu_controller.sh",
        SCRIPTS / "preflight_tpu_controller.sh",
    )
    for path in connected_only:
        if "SUNFISH_TPU_WORKER" not in path.read_text(encoding="utf-8"):
            fail(f"connected-only script lacks a TPU-worker refusal: {path.name}")

    probe = (SCRIPTS / "probe_tpu_worker_base.sh").read_text(encoding="utf-8")
    if (
        "preflight_tpu_controller.sh" not in probe
        or probe.index("preflight_tpu_controller.sh") > probe.index("tpu_iap.sh")
    ):
        fail("base-image probe does not run controller preflight before TPU contact")

    implementation_files = [
        *SCRIPTS.glob("*.sh"),
        *SCRIPTS.glob("*.py"),
        *(ROOT / "src").rglob("*.py"),
    ]
    for path in implementation_files:
        text = path.read_text(encoding="utf-8")
        if path.resolve() in {IAP_WRAPPER.resolve(), SELF}:
            continue
        if re.search(
            r"\bgcloud(?:\s+alpha)?\s+compute\s+tpus\s+tpu-vm\s+[a-z0-9-]+\b",
            text,
            re.IGNORECASE,
        ):
            fail(f"implementation bypasses the guarded TPU control plane: {path}")

    operational_docs = (
        ROOT / "AGENTS.md",
        ROOT / "PLAN.md",
        ROOT / "README.md",
        ROOT / "infra/tpu/README.md",
        ROOT / "infra/gcp/README.md",
        ROOT / "docs/compute_call_brief.md",
        ROOT / "docs/training_harness.md",
        ROOT / "coordination/external_tpu_review.md",
        ROOT / "reference/tpu-docs/tpu-pod-launch.md",
    )
    mutating_commands = "|".join(re.escape(name) for name in MUTATING_TPU_VM_COMMANDS)
    lifecycle = re.compile(
        r"gcloud(?:\s+alpha)?\s+compute\s+tpus\s+tpu-vm\s+"
        rf"(?:{mutating_commands})\b",
        re.IGNORECASE,
    )
    for path in operational_docs:
        if lifecycle.search(path.read_text(encoding="utf-8")):
            fail(f"runbook contains a TPU allocation lifecycle command: {path}")

    print("Sunfish TPU release safety policy passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"check_tpu_release_safety: {error}", file=sys.stderr)
        raise SystemExit(2)
