"""Controller-side real preemption/recovery orchestrator for readiness gate 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
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
_REMOTE_INTERRUPT_TIMEOUT_SECONDS = 120
_LOCAL_GROUP_TERM_TIMEOUT_SECONDS = 30
_LOCAL_GROUP_KILL_TIMEOUT_SECONDS = 30
CLEANUP_HARD_STOP_RETURN_CODE = 126
EXPECTED_PREEMPTED_LAUNCH_RETURN_CODES = frozenset(
    (128 + signal.SIGKILL, 128 + signal.SIGTERM)
)
PREEMPTED_LAUNCH_EXIT_POLICY = "signal-status-only-137-or-143"


class _ControllerSignal(RuntimeError):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"controller received signal {signum}")


class OwnerInterventionRequiredError(RuntimeError):
    """Exact worker/local cleanup is unproven; automation must stop."""


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


def _invoke_exact_remote_interrupt(
    interrupt_command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run the all-worker exact-PID helper with a hard controller timeout."""
    return subprocess.run(
        list(interrupt_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=_REMOTE_INTERRUPT_TIMEOUT_SECONDS,
    )


def _require_expected_preempted_launch_returncode(returncode: int) -> None:
    """Refuse a resume unless the first launcher reports a signal exit.

    Status 126 is the attached launcher's explicit fail-closed contract: exact
    remote cleanup is unproven and the allocation owner must investigate.  It
    must never be treated as generic nonzero evidence of a successful test
    interruption.  Other unexpected statuses likewise cannot prove that the
    documented SIGKILL/SIGTERM path is what stopped the launch.
    """
    if returncode == CLEANUP_HARD_STOP_RETURN_CODE:
        raise OwnerInterventionRequiredError(
            "preempted launch returned cleanup hard-stop status 126; exact "
            "remote cleanup is unproven, owner intervention is required, and "
            "automatic resume/retry is forbidden"
        )
    if returncode not in EXPECTED_PREEMPTED_LAUNCH_RETURN_CODES:
        expected = "/".join(
            str(value) for value in sorted(EXPECTED_PREEMPTED_LAUNCH_RETURN_CODES)
        )
        raise RuntimeError(
            f"preempted launch exited with {returncode}; expected signal-style "
            f"status {expected} from the documented exact interrupt path, so "
            "automatic resume is forbidden"
        )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(process_group_id: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _best_effort_interrupt_and_stop(
    process: subprocess.Popen,
    interrupt_command: Sequence[str],
    *,
    interrupt_remote: bool,
) -> bool:
    """Stop an abnormal launch and return whether cleanup is fully proven.

    The remote helper fails closed unless every PID still belongs to the exact
    run/attempt and ``sunfish-train`` command. It must run before terminating
    the controller-local SSH launcher so a dropped SSH session cannot strand
    the user-space training processes on workers.
    """

    def warn(message: str) -> None:
        try:
            print(f"sunfish-preemption-gate cleanup: {message}", file=sys.stderr)
        except BaseException:
            # Cleanup diagnostics must never replace the original gate error.
            pass

    # A failed SSH/gcloud launcher can exit while its remote child survives.
    # Therefore remote cleanup cannot be conditional on the local process still
    # running. Skip it only after the normal guarded interrupt returned zero.
    remote_cleanup_proven = not interrupt_remote
    if interrupt_remote:
        try:
            interrupted = _invoke_exact_remote_interrupt(interrupt_command)
            remote_cleanup_proven = interrupted.returncode == 0
            if not remote_cleanup_proven:
                warn(
                    "exact remote interrupt returned "
                    f"{interrupted.returncode}: {interrupted.stdout.strip()}"
                )
        except BaseException as error:
            warn(f"exact remote interrupt failed: {error}")
            remote_cleanup_proven = False

    try:
        active = process.poll() is None
    except BaseException as error:
        warn(f"unable to re-inspect local launcher: {error}")
        active = True
    if not active and not interrupt_remote:
        return remote_cleanup_proven

    # ``first`` is launched with ``start_new_session=True`` below. Signal that
    # isolated controller-local process group rather than only the Bash wrapper:
    # launch_tpu_pod.sh owns a gcloud/SSH + tee pipeline whose children otherwise
    # survive wrapper termination. This never broadens the worker-side signal;
    # the remote helper above remains exact-PID and allocation-safe.
    process_group_id = process.pid
    if (
        not isinstance(process_group_id, int)
        or process_group_id <= 1
        or process_group_id == os.getpgrp()
    ):
        warn(f"refusing unsafe local process-group cleanup: {process_group_id!r}")
        return False

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return remote_cleanup_proven
    except BaseException as error:
        warn(f"unable to terminate local launcher process group: {error}")
        return False

    try:
        process.wait(timeout=_LOCAL_GROUP_TERM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    except BaseException as error:
        warn(f"unable to wait for local launcher: {error}")
    try:
        terminated = _wait_for_process_group_exit(
            process_group_id, _LOCAL_GROUP_TERM_TIMEOUT_SECONDS
        )
    except BaseException as error:
        warn(f"unable to prove local launcher exit after SIGTERM: {error}")
        terminated = False
    if terminated:
        return remote_cleanup_proven

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return remote_cleanup_proven
    except BaseException as error:
        warn(f"unable to kill local launcher process group: {error}")
        return False
    try:
        process.wait(timeout=_LOCAL_GROUP_KILL_TIMEOUT_SECONDS)
    except BaseException as error:
        warn(f"unable to reap local launcher after SIGKILL: {error}")
    try:
        killed = _wait_for_process_group_exit(
            process_group_id, _LOCAL_GROUP_KILL_TIMEOUT_SECONDS
        )
    except BaseException as error:
        warn(f"unable to prove local launcher exit after SIGKILL: {error}")
        killed = False
    if not killed:
        warn("local launcher process group still exists after SIGKILL")
        return False
    return remote_cleanup_proven


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
            "interrupt_resume_processes": [
                str(interrupter),
                "--run-id",
                config.run.run_id,
                "--attempt-id",
                resume_attempt,
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
    first: subprocess.Popen[str] | None = None
    resumed_process: subprocess.Popen[str] | None = None
    remote_interrupt_succeeded = False
    resume_remote_cleanup_required = False
    caught_signal: int | None = None
    cleanup_armed = False
    spawn_in_progress = False
    previous_handlers: dict[int, Any] = {}

    def handle_controller_signal(signum: int, _frame: Any) -> None:
        nonlocal caught_signal
        caught_signal = signum
        # While Popen is inside the OS spawn path there may be a live child but
        # no Python process object yet. Latch, then raise immediately after
        # Popen returns and the exact cleanup path owns the child.
        if cleanup_armed and not spawn_in_progress:
            raise _ControllerSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, handle_controller_signal)
    try:
        cleanup_armed = True
        if caught_signal is not None:
            raise _ControllerSignal(caught_signal)
        spawn_in_progress = True
        try:
            first = subprocess.Popen(
                launch_command(preempt_attempt),
                stdout=preempt_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        finally:
            spawn_in_progress = False
        if caught_signal is not None:
            raise _ControllerSignal(caught_signal)
        _wait_for_uri(
            gcloud,
            plan["preempt_marker"],
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            process=first,
        )
        evidence["checkpoint_finalized_before_process_interrupt_at"] = _utc_now()
        interrupted = _invoke_exact_remote_interrupt(
            evidence["commands"]["interrupt_training_processes"]
        )
        if interrupted.returncode != 0:
            raise subprocess.CalledProcessError(
                interrupted.returncode,
                evidence["commands"]["interrupt_training_processes"],
                output=interrupted.stdout,
            )
        evidence["interrupt_output_sha256"] = hashlib.sha256(
            interrupted.stdout.encode()
        ).hexdigest()
        evidence["exact_recorded_processes_interrupted"] = True
        evidence["interrupt_process_policy"] = (
            "pre-signal-exact-root-and-descendant-snapshot-with-pidfd"
        )
        evidence["same_attempt_descendants_absent"] = True
        evidence["interrupt_timeout_seconds"] = _REMOTE_INTERRUPT_TIMEOUT_SECONDS
        remote_interrupt_succeeded = True
        try:
            first_returncode = first.wait(timeout=120)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("preempted all-host launch did not exit") from error
        evidence["preempted_launch_returncode"] = first_returncode
        evidence["preempted_launch_exit_policy"] = PREEMPTED_LAUNCH_EXIT_POLICY
        try:
            _require_expected_preempted_launch_returncode(first_returncode)
        except RuntimeError:
            evidence["owner_intervention_required"] = True
            raise
        evidence["owner_intervention_required"] = False
        if not _gcloud_exists(gcloud, plan["preempt_marker"]):
            raise RuntimeError("finalized checkpoint disappeared after preemption")

        if caught_signal is not None:
            raise _ControllerSignal(caught_signal)
        spawn_in_progress = True
        try:
            resumed_process = subprocess.Popen(
                launch_command(resume_attempt),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        finally:
            spawn_in_progress = False
        resume_remote_cleanup_required = True
        if caught_signal is not None:
            raise _ControllerSignal(caught_signal)
        try:
            resumed_output, _ = resumed_process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("resumed all-host launch timed out") from error
        evidence["resumed_launch_returncode"] = resumed_process.returncode
        evidence["resumed_output_sha256"] = hashlib.sha256(
            resumed_output.encode()
        ).hexdigest()
        if resumed_process.returncode == CLEANUP_HARD_STOP_RETURN_CODE:
            evidence["owner_intervention_required"] = True
            raise OwnerInterventionRequiredError(
                "resumed launch returned cleanup hard-stop status 126; exact "
                "remote cleanup is unproven, owner intervention is required, "
                "and automatic retry is forbidden"
            )
        if resumed_process.returncode != 0:
            raise RuntimeError(
                f"resumed launch failed with {resumed_process.returncode}"
            )
        resume_remote_cleanup_required = False
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
        cleanup_proven = True
        cleanup_armed = False
        # Once cleanup starts, a repeated terminal signal must not interrupt
        # exact remote cleanup or local group teardown halfway through.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)

        def prove_cleanup(
            process: subprocess.Popen[str],
            command: Sequence[str],
            *,
            interrupt_remote: bool,
        ) -> bool:
            try:
                return _best_effort_interrupt_and_stop(
                    process,
                    command,
                    interrupt_remote=interrupt_remote,
                )
            except BaseException as error:
                try:
                    print(
                        "sunfish-preemption-gate cleanup proof raised: "
                        f"{error}",
                        file=sys.stderr,
                    )
                except BaseException:
                    pass
                return False

        try:
            if resumed_process is not None and resume_remote_cleanup_required:
                cleanup_proven = prove_cleanup(
                    resumed_process,
                    evidence["commands"]["interrupt_resume_processes"],
                    interrupt_remote=True,
                ) and cleanup_proven
            if first is not None:
                cleanup_proven = prove_cleanup(
                    first,
                    evidence["commands"]["interrupt_training_processes"],
                    interrupt_remote=not remote_interrupt_succeeded,
                ) and cleanup_proven
            preempt_log.seek(0)
            evidence["preempted_output_sha256"] = hashlib.sha256(
                preempt_log.read().encode()
            ).hexdigest()
        finally:
            preempt_log.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if not cleanup_proven:
            raise OwnerInterventionRequiredError(
                "preemption-gate cleanup is unproven; owner intervention is "
                "required and automatic resume/retry is forbidden"
            )

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
    except OwnerInterventionRequiredError as error:
        print(f"sunfish-preemption-gate: {error}", file=sys.stderr)
        return CLEANUP_HARD_STOP_RETURN_CODE
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
