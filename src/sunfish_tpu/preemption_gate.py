"""Controller-side real preemption/recovery orchestrator for readiness gate 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sunfish_tpu.source_identity import normalize_source_identity
from sunfish_tpu.training.spec import HarnessConfig, Phase

_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def checkpoint_commit_marker(workdir: str, step: int) -> str:
    """Pinned Orbax 0.12.1 GCS atomic-finalization marker."""
    return f"{workdir.rstrip('/')}/checkpoints/ckpt_{step}/commit_success.txt"


def build_preemption_plan(
    config: HarnessConfig,
    *,
    preempt_attempt: str,
    resume_attempt: str,
    preempt_after_step: int,
) -> dict[str, Any]:
    if config.run.phase is not Phase.SMOKE:
        raise ValueError("preemption gate requires phase=smoke")
    for name, value in (
        ("preempt_attempt", preempt_attempt),
        ("resume_attempt", resume_attempt),
    ):
        if not _ATTEMPT_ID.fullmatch(value):
            raise ValueError(f"invalid {name}")
    if preempt_attempt == resume_attempt:
        raise ValueError("preempt and resume attempt IDs must differ")
    if not 0 < preempt_after_step < config.training.steps:
        raise ValueError("preempt_after_step must be inside the training run")
    if preempt_after_step % config.training.checkpoint_every_steps:
        raise ValueError("preempt_after_step must be a configured checkpoint step")
    if config.training.log_metrics_every_steps != 1:
        raise ValueError("preemption proof requires log_metrics_every_steps=1")
    resume_metrics = (
        f"{config.run.workdir.rstrip('/')}/readiness/{resume_attempt}/metrics"
    )
    return {
        "run_id": config.run.run_id,
        "workdir": config.run.workdir,
        "config_sha256": config.digest,
        "preempt_attempt": preempt_attempt,
        "resume_attempt": resume_attempt,
        "preempt_after_step": preempt_after_step,
        "preempt_marker": checkpoint_commit_marker(
            config.run.workdir, preempt_after_step
        ),
        "final_marker": checkpoint_commit_marker(
            config.run.workdir, config.training.steps
        ),
        "train_complete": f"{config.run.workdir.rstrip('/')}/train_complete.txt",
        # Kauldron labels the update following ckpt_N as step N. A real restore
        # therefore starts at N, whereas a silent fresh restart emits step 0.
        "resume_first_metric": (
            f"{resume_metrics}/step-{preempt_after_step:09d}.json"
        ),
        "fresh_start_metric": f"{resume_metrics}/step-000000000.json",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gcloud_exists(gcloud: str, uri: str) -> bool:
    result = subprocess.run(
        [gcloud, "storage", "ls", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _gcloud_read_json(gcloud: str, uri: str) -> tuple[dict[str, Any], str]:
    result = subprocess.run(
        [gcloud, "storage", "cat", uri],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"GCS JSON evidence is not an object: {uri}")
    return payload, hashlib.sha256(result.stdout).hexdigest()


def _wait_for_uri(
    gcloud: str,
    uri: str,
    *,
    timeout_seconds: int,
    poll_seconds: int,
    process: subprocess.Popen | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _gcloud_exists(gcloud, uri):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"launch exited with {process.returncode} before producing {uri}"
            )
        time.sleep(poll_seconds)
    raise TimeoutError(f"timed out waiting for {uri}")


def _upload_immutable(gcloud: str, local_path: Path, uri: str) -> None:
    if _gcloud_exists(gcloud, uri):
        raise FileExistsError(f"immutable preemption evidence exists: {uri}")
    subprocess.run(
        [gcloud, "storage", "cp", str(local_path), uri],
        check=True,
    )


def run_preemption_gate(
    *,
    config_path: Path,
    preempt_attempt: str,
    resume_attempt: str,
    preempt_after_step: int,
    evidence_uri: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    config = HarnessConfig.load(config_path)
    plan = build_preemption_plan(
        config,
        preempt_attempt=preempt_attempt,
        resume_attempt=resume_attempt,
        preempt_after_step=preempt_after_step,
    )
    if not evidence_uri.startswith("gs://") or evidence_uri.endswith("/"):
        raise ValueError("evidence_uri must be a complete gs:// object path")
    gcloud = os.environ.get("SUNFISH_GCLOUD_BIN", "gcloud")
    train_bin = os.environ.get("SUNFISH_TRAIN_BIN", ".venv-tpu/bin/sunfish-train")
    remote_config = os.environ.get("SUNFISH_REMOTE_CONFIG", str(config_path))
    root = Path(__file__).resolve().parents[2]
    launcher = root / "scripts/launch_tpu_pod.sh"
    interrupter = root / "scripts/interrupt_training_attempt.sh"

    def launch_command(attempt: str) -> list[str]:
        return [
            str(launcher),
            "--run-id",
            config.run.run_id,
            "--attempt-id",
            attempt,
            "--config",
            str(config_path),
            "--remote-config",
            remote_config,
            "--",
            train_bin,
            "--config",
            remote_config,
        ]

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "gate": 7,
        "plan": plan,
        "started_at": _utc_now(),
        "commands": {
            "preempt_launch": launch_command(preempt_attempt),
            "interrupt_training_processes": [
                str(interrupter),
                "--run-id",
                config.run.run_id,
                "--attempt-id",
                preempt_attempt,
            ],
            "resume_launch": launch_command(resume_attempt),
        },
        "controller_config": str(config_path),
        "remote_config": remote_config,
    }

    preexisting = [
        uri
        for uri in (
            plan["preempt_marker"],
            plan["final_marker"],
            plan["train_complete"],
            plan["resume_first_metric"],
            plan["fresh_start_metric"],
        )
        if _gcloud_exists(gcloud, uri)
    ]
    if preexisting:
        raise FileExistsError(
            "preemption gate requires a fresh workdir; found "
            + ", ".join(preexisting)
        )

    preempt_log = tempfile.TemporaryFile(mode="w+")
    first = subprocess.Popen(
        launch_command(preempt_attempt),
        stdout=preempt_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_uri(
            gcloud,
            plan["preempt_marker"],
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            process=first,
        )
        evidence["checkpoint_finalized_before_process_interrupt_at"] = _utc_now()
        subprocess.run(evidence["commands"]["interrupt_training_processes"], check=True)
        try:
            first_returncode = first.wait(timeout=120)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("preempted all-host launch did not exit") from error
        evidence["preempted_launch_returncode"] = first_returncode
        if first_returncode == 0:
            raise RuntimeError(
                "interrupted launch exited zero; no real process interruption was observed"
            )
        if not _gcloud_exists(gcloud, plan["preempt_marker"]):
            raise RuntimeError("finalized checkpoint disappeared after preemption")

        resumed = subprocess.run(
            launch_command(resume_attempt),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        evidence["resumed_launch_returncode"] = resumed.returncode
        evidence["resumed_output_sha256"] = hashlib.sha256(
            resumed.stdout.encode()
        ).hexdigest()
        if resumed.returncode != 0:
            raise RuntimeError(f"resumed launch failed with {resumed.returncode}")
        _wait_for_uri(
            gcloud,
            plan["resume_first_metric"],
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        if _gcloud_exists(gcloud, plan["fresh_start_metric"]):
            raise RuntimeError("resumed attempt emitted step 0 and restarted from scratch")
        resume_metric, resume_metric_sha256 = _gcloud_read_json(
            gcloud, plan["resume_first_metric"]
        )
        expected_resume_fields = {
            "schema_version": 1,
            "attempt_id": resume_attempt,
            "run_id": config.run.run_id,
            "config_sha256": config.digest,
            "dataset_manifest_sha256": config.data.manifest_sha256,
            "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
            "step": preempt_after_step,
        }
        changed = [
            name
            for name, expected in expected_resume_fields.items()
            if resume_metric.get(name) != expected
        ]
        if changed or normalize_source_identity(
            resume_metric.get("sunfish_source")
        ) is None:
            raise RuntimeError(
                "resumed metric has invalid checkpoint-continuation lineage: "
                + ", ".join(changed or ["sunfish_source"])
            )
        evidence["resume_proof"] = {
            **expected_resume_fields,
            "sunfish_source": resume_metric["sunfish_source"],
            "metric_uri": plan["resume_first_metric"],
            "metric_sha256": resume_metric_sha256,
        }
        _wait_for_uri(
            gcloud,
            plan["train_complete"],
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        _wait_for_uri(
            gcloud,
            plan["final_marker"],
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        evidence.update(
            {
                "completed_at": _utc_now(),
                "finalized_checkpoint_survived": True,
                "automatic_same_workdir_restore": True,
                "resume_continued_from_checkpoint": True,
                "fresh_start_metric_absent": True,
                "manual_gcs_cleanup_performed": False,
                "train_complete_found": True,
                "final_checkpoint_found": True,
                "passed": True,
            }
        )
    finally:
        if first.poll() is None:
            first.terminate()
            try:
                first.wait(timeout=30)
            except subprocess.TimeoutExpired:
                first.kill()
                first.wait(timeout=30)
        preempt_log.seek(0)
        evidence["preempted_output_sha256"] = hashlib.sha256(
            preempt_log.read().encode()
        ).hexdigest()
        preempt_log.close()

    encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "preemption-summary.json"
        local.write_text(encoded, encoding="utf-8")
        _upload_immutable(gcloud, local, evidence_uri)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preempt-attempt", required=True)
    parser.add_argument("--resume-attempt", required=True)
    parser.add_argument("--preempt-after-step", type=int, required=True)
    parser.add_argument("--evidence-uri", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        payload = run_preemption_gate(
            config_path=args.config,
            preempt_attempt=args.preempt_attempt,
            resume_attempt=args.resume_attempt,
            preempt_after_step=args.preempt_after_step,
            evidence_uri=args.evidence_uri,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TimeoutError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"sunfish-preemption-gate: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
