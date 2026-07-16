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
    SCRIPTS / "verify_tpu_bundled_runtime.sh",
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
        "--ssh-flag=-oServerAliveInterval=30",
        "--ssh-flag=-oServerAliveCountMax=6",
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
        "sunfish_tpu.standalone_runtime verify-installed",
        "--require-worker-runtime",
        "python/bin/python3",
    ):
        if required not in bootstrap:
            fail(f"offline worker bootstrap is missing {required!r}")

    host_entrypoint = (SCRIPTS / "tpu_host_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    if "export SUNFISH_TPU_WORKER=1" not in host_entrypoint:
        fail("all-worker entrypoint does not mark the TPU worker environment")
    for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if proxy_name not in host_entrypoint:
            fail(f"all-worker entrypoint does not reject {proxy_name}")
    launcher = (SCRIPTS / "launch_tpu_pod.sh").read_text(encoding="utf-8")
    for text, name in (
        (launcher, "controller launcher"),
        (host_entrypoint, "all-worker entrypoint"),
    ):
        if "XLA_PYTHON_CLIENT_PREALLOCATE" not in text:
            fail(f"{name} does not bind XLA preallocation policy")
    if "--xla-python-client-preallocate" not in launcher:
        fail("controller launcher does not forward XLA preallocation policy")
    for required in (
        "controller_attached_launch.py",
        "interrupt_training_attempt.sh",
        "require_durable_controller.sh",
    ):
        if required not in launcher:
            fail(f"controller launcher is missing lifetime guard {required!r}")
    for required in (
        "publish_pid_file",
        "set -o noclobber",
        "stop_unpublished_child",
        "pid_publish_gate.py",
        "exit 126",
    ):
        if required not in host_entrypoint:
            fail(f"all-worker entrypoint lacks exact PID publication guard {required!r}")
    if "sunfish_tpu.worker_hygiene" not in host_entrypoint:
        fail("all-worker entrypoint lacks the read-only accelerator/lock hygiene gate")
    exact_interrupt = (
        ROOT / "src/sunfish_tpu/exact_process_interrupt.py"
    ).read_text(encoding="utf-8")
    for required in (
        "pidfd_send_signal",
        "start_time_ticks",
        "exact_recorded_descendants",
        "owner intervention is required",
    ):
        if required not in exact_interrupt:
            fail(f"Gate-7 exact process helper is missing {required!r}")
    if "killpg" in exact_interrupt:
        fail("Gate-7 worker helper may not signal a process group")
    hygiene = (ROOT / "src/sunfish_tpu/worker_hygiene.py").read_text(
        encoding="utf-8"
    )
    for required in ("/dev/accel", "/tmp/libtpu_lockfile", "read_only"):
        if required not in hygiene:
            fail(f"worker hygiene check is missing {required!r}")
    for forbidden in ("os.kill(", ".unlink(", "shutil.rmtree"):
        if forbidden in hygiene:
            fail(f"worker hygiene check is not read-only: {forbidden!r}")
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
    for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if proxy_name not in probe:
            fail(f"base-image probe does not reject {proxy_name}")
    if "SUNFISH_STOCK_PYTHON_BIN:-python3" not in probe:
        fail("base-image probe does not use stock Python 3")
    if "ensurepip" in probe or 'sys.version_info[:2]==(3,12)' in probe:
        fail("base-image probe still assumes the bundled Python runtime")
    deploy = (SCRIPTS / "deploy_tpu_offline_bundle.sh").read_text(encoding="utf-8")
    for required in (
        "SUNFISH_STOCK_PYTHON_BIN:-python3",
        "standalone_runtime.py",
        "extract-bundle",
        "verify_tpu_bundled_runtime.sh",
    ):
        if required not in deploy:
            fail(f"offline bundle deployment is missing {required!r}")
    runtime_source = (
        ROOT / "src/sunfish_tpu/standalone_runtime.py"
    ).read_text(encoding="utf-8")
    for exact_pin in (
        'RUNTIME_RELEASE = "20260623"',
        'RUNTIME_PYTHON_VERSION = "3.12.13"',
        "9fa869d69be54f6b8eeae64272fbd9bb0646e0e1a8da9d80e51ba5a3bee48930",
        "RUNTIME_ARCHIVE_SIZE = 111_146_559",
    ):
        if exact_pin not in runtime_source:
            fail(f"standalone Python helper is missing exact pin {exact_pin!r}")

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
