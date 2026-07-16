import dataclasses
import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from sunfish.source_tree import workspace_source_identity
from sunfish_tpu.deployment_config import render_stage05_configs
from sunfish_tpu.preemption_gate import (
    CLEANUP_HARD_STOP_RETURN_CODE,
    EXPECTED_PREEMPTED_LAUNCH_RETURN_CODES,
    OwnerInterventionRequiredError,
    _best_effort_interrupt_and_stop,
    _invoke_exact_remote_interrupt,
    _require_expected_preempted_launch_returncode,
    build_preemption_plan,
    checkpoint_commit_marker,
    main as preemption_main,
    run_preemption_gate,
)
from sunfish_tpu.training.spec import HarnessConfig, Phase
from tests.test_parity_evidence import valid_parity_payload


ROOT = Path(__file__).resolve().parents[1]


class PreemptionGateTests(unittest.TestCase):
    def test_only_documented_signal_statuses_authorize_resume(self):
        for returncode in EXPECTED_PREEMPTED_LAUNCH_RETURN_CODES:
            _require_expected_preempted_launch_returncode(returncode)
        for returncode in (0, 1, 125, 255):
            with self.subTest(returncode=returncode):
                with self.assertRaisesRegex(RuntimeError, "automatic resume"):
                    _require_expected_preempted_launch_returncode(returncode)

    def test_cleanup_hard_stop_requires_owner_and_forbids_resume(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "status 126.*owner intervention.*automatic resume/retry is forbidden",
        ):
            _require_expected_preempted_launch_returncode(
                CLEANUP_HARD_STOP_RETURN_CODE
            )

    def test_first_launch_status_126_never_spawns_resume(self):
        config_path = ROOT / "configs/training/sunfish-smoke.toml"
        first = mock.Mock(pid=98770)
        first.wait.return_value = CLEANUP_HARD_STOP_RETURN_CODE
        marker_checks = 0

        def gcloud_exists(_gcloud, uri):
            nonlocal marker_checks
            if uri.endswith("ckpt_25/commit_success.txt"):
                marker_checks += 1
                return marker_checks > 1
            return False

        with (
            mock.patch(
                "sunfish_tpu.preemption_gate._gcloud_exists",
                side_effect=gcloud_exists,
            ),
            mock.patch("sunfish_tpu.preemption_gate._wait_for_uri"),
            mock.patch(
                "sunfish_tpu.preemption_gate._invoke_exact_remote_interrupt",
                return_value=mock.Mock(returncode=0, stdout="exact cleanup passed"),
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate.subprocess.Popen",
                return_value=first,
            ) as popen,
            mock.patch(
                "sunfish_tpu.preemption_gate._best_effort_interrupt_and_stop",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(
                OwnerInterventionRequiredError,
                "status 126.*owner intervention.*automatic resume/retry is forbidden",
            ):
                run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="hard-stop-first-001",
                    resume_attempt="must-not-spawn-001",
                    preempt_after_step=25,
                    evidence_uri="gs://fake/readiness/preemption.json",
                    timeout_seconds=1,
                    poll_seconds=0,
                )
        popen.assert_called_once()

    def test_resumed_launch_status_126_is_owner_stop(self):
        config_path = ROOT / "configs/training/sunfish-smoke.toml"
        first = mock.Mock(pid=98771)
        first.wait.return_value = 137
        resumed = mock.Mock(pid=98772)
        resumed.returncode = CLEANUP_HARD_STOP_RETURN_CODE
        resumed.communicate.return_value = ("cleanup unproven", None)
        marker_checks = 0

        def gcloud_exists(_gcloud, uri):
            nonlocal marker_checks
            if uri.endswith("ckpt_25/commit_success.txt"):
                marker_checks += 1
                return marker_checks > 1
            return False

        with (
            mock.patch(
                "sunfish_tpu.preemption_gate._gcloud_exists",
                side_effect=gcloud_exists,
            ),
            mock.patch("sunfish_tpu.preemption_gate._wait_for_uri"),
            mock.patch(
                "sunfish_tpu.preemption_gate._invoke_exact_remote_interrupt",
                return_value=mock.Mock(returncode=0, stdout="exact cleanup passed"),
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate.subprocess.Popen",
                side_effect=(first, resumed),
            ) as popen,
            mock.patch(
                "sunfish_tpu.preemption_gate._best_effort_interrupt_and_stop",
                return_value=True,
            ) as cleanup,
        ):
            with self.assertRaisesRegex(
                OwnerInterventionRequiredError,
                "resumed launch.*status 126.*owner intervention.*automatic retry",
            ):
                run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="good-interrupt-001",
                    resume_attempt="hard-stop-resume-001",
                    preempt_after_step=25,
                    evidence_uri="gs://fake/readiness/preemption.json",
                    timeout_seconds=1,
                    poll_seconds=0,
                )
        self.assertEqual(popen.call_count, 2)
        self.assertTrue(
            any(call.args[0] is resumed for call in cleanup.call_args_list)
        )

    def test_controller_signal_latched_inside_spawn_cleans_the_new_child(self):
        config_path = ROOT / "configs/training/sunfish-smoke.toml"
        process = mock.Mock()
        process.pid = 98764
        process.poll.return_value = None
        previous = signal.getsignal(signal.SIGTERM)

        def signal_during_spawn(*_args, **_kwargs):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            return process

        with (
            mock.patch(
                "sunfish_tpu.preemption_gate._gcloud_exists", return_value=False
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate.subprocess.Popen",
                side_effect=signal_during_spawn,
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate._best_effort_interrupt_and_stop",
                return_value=True,
            ) as cleanup,
            mock.patch.dict(
                os.environ,
                {
                    "SUNFISH_TRAIN_BIN": "fake-sunfish-train",
                    "SUNFISH_REMOTE_CONFIG": "/remote/sunfish-smoke.toml",
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "signal 15"):
                run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="spawn-signal-001",
                    resume_attempt="spawn-signal-resume-001",
                    preempt_after_step=25,
                    evidence_uri="gs://fake/readiness/preemption.json",
                    timeout_seconds=1,
                    poll_seconds=0,
                )
        cleanup.assert_called_once()
        self.assertIs(cleanup.call_args.args[0], process)
        self.assertTrue(cleanup.call_args.kwargs["interrupt_remote"])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_controller_signal_with_unproven_cleanup_is_owner_stop(self):
        config_path = ROOT / "configs/training/sunfish-smoke.toml"
        process = mock.Mock(pid=98773)

        def signal_during_spawn(*_args, **_kwargs):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            return process

        with (
            mock.patch(
                "sunfish_tpu.preemption_gate._gcloud_exists", return_value=False
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate.subprocess.Popen",
                side_effect=signal_during_spawn,
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate._best_effort_interrupt_and_stop",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(
                OwnerInterventionRequiredError,
                "cleanup is unproven.*owner intervention.*automatic resume/retry",
            ):
                run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="signal-unproven-001",
                    resume_attempt="signal-unproven-resume-001",
                    preempt_after_step=25,
                    evidence_uri="gs://fake/readiness/preemption.json",
                    timeout_seconds=1,
                    poll_seconds=0,
                )

    def test_cli_maps_owner_stop_to_status_126(self):
        with mock.patch(
            "sunfish_tpu.preemption_gate.run_preemption_gate",
            side_effect=OwnerInterventionRequiredError("cleanup unproven"),
        ):
            returncode = preemption_main(
                [
                    "--config",
                    str(ROOT / "configs/training/sunfish-smoke.toml"),
                    "--preempt-attempt",
                    "cli-stop-001",
                    "--resume-attempt",
                    "cli-stop-resume-001",
                    "--preempt-after-step",
                    "25",
                    "--evidence-uri",
                    "gs://fake/readiness/preemption.json",
                ]
            )
        self.assertEqual(returncode, CLEANUP_HARD_STOP_RETURN_CODE)

    def test_remote_cleanup_runs_even_after_local_ssh_launcher_exited(self):
        process = mock.Mock()
        process.pid = 98765
        process.poll.return_value = 255
        completed = mock.Mock(returncode=0, stdout="")
        with (
            mock.patch(
                "sunfish_tpu.preemption_gate._invoke_exact_remote_interrupt",
                return_value=completed,
            ) as remote,
            mock.patch(
                "sunfish_tpu.preemption_gate.os.killpg",
                side_effect=ProcessLookupError,
            ) as kill_group,
        ):
            self.assertTrue(
                _best_effort_interrupt_and_stop(
                    process,
                    ["interrupt", "--run-id", "run"],
                    interrupt_remote=True,
                )
            )
        remote.assert_called_once()
        kill_group.assert_called_once_with(98765, signal.SIGTERM)

    def test_local_process_group_cleanup_failure_is_unproven(self):
        process = mock.Mock(pid=98774)
        process.poll.return_value = None
        with mock.patch(
            "sunfish_tpu.preemption_gate.os.killpg",
            side_effect=PermissionError("simulated local cleanup denial"),
        ):
            self.assertFalse(
                _best_effort_interrupt_and_stop(
                    process,
                    ["unused-interrupt"],
                    interrupt_remote=False,
                )
            )

    def test_exact_remote_interrupt_has_a_hard_timeout(self):
        completed = mock.Mock(returncode=0, stdout="")
        with mock.patch(
            "sunfish_tpu.preemption_gate.subprocess.run", return_value=completed
        ) as run:
            self.assertIs(
                _invoke_exact_remote_interrupt(["interrupt", "--run-id", "run"]),
                completed,
            )
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertFalse(run.call_args.kwargs["check"])

    def test_local_cleanup_terminates_the_isolated_launcher_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    'sleep 300 & printf "%s\\n" "$!" > "$1"; wait',
                    "bash",
                    str(child_pid_path),
                ],
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid_path.exists():
                    time.sleep(0.01)
                self.assertTrue(child_pid_path.exists())
                child_pid = int(child_pid_path.read_text(encoding="ascii"))
                _best_effort_interrupt_and_stop(
                    process,
                    ["unused-interrupt"],
                    interrupt_remote=False,
                )
                self.assertIsNotNone(process.poll())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_controller_launch_kill_resume_and_upload_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            state.mkdir()
            source = workspace_source_identity(ROOT)
            parity_payload = valid_parity_payload()
            parity_payload["sunfish_source"] = source
            parity_payload["environment"]["float32"]["sunfish_source"] = source
            parity_payload["environment"]["bfloat16"]["sunfish_source"] = source
            parity_path = root / "parity.json"
            parity_path.write_text(json.dumps(parity_payload), encoding="utf-8")
            bundle = root / "bundle"
            render_stage05_configs(
                template_directory=ROOT / "configs/training",
                output_directory=bundle,
                storage_root="gs://bucket/sunfish",
                run_tag="preemption-integration",
                dataset_manifest_sha256="a" * 64,
                seed_manifest_sha256="b" * 64,
                parity_report_path=parity_path,
                expected_devices=32,
                expected_processes=8,
                expected_local_devices=4,
                source_root=ROOT,
            )
            config_path = bundle / "sunfish-preemption-smoke.toml"
            config = HarnessConfig.load(config_path)
            resume_metric_template = state / "resume-metric-template.json"
            resume_metric_template.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "attempt_id": "stage05-resume-integration",
                        "run_id": config.run.run_id,
                        "config_sha256": config.digest,
                        "dataset_manifest_sha256": config.data.manifest_sha256,
                        "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
                        "step": 25,
                        "sunfish_source": workspace_source_identity(ROOT),
                    }
                ),
                encoding="utf-8",
            )
            fake_gcloud = root / "gcloud"
            fake_gcloud.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
state="${SUNFISH_FAKE_GCLOUD_STATE:?}"
if [[ "$1 $2" == "storage ls" ]]; then
  uri="$3"
  case "$uri" in
    *ckpt_25/commit_success.txt) test -f "$state/preempt-marker" ;;
    *ckpt_100/commit_success.txt) test -f "$state/final-marker" ;;
    *train_complete.txt) test -f "$state/train-complete" ;;
    *stage05-resume-integration/metrics/step-000000025.json) test -f "$state/resume-metric" ;;
    *stage05-resume-integration/metrics/step-000000000.json) test -f "$state/fresh-metric" ;;
    *preemption-summary.json) test -f "$state/evidence.json" ;;
    *) exit 1 ;;
  esac
  exit $?
fi
if [[ "$1 $2" == "storage cat" ]]; then
  cat "$state/resume-metric"
  exit 0
fi
if [[ "$1 $2" == "storage cp" ]]; then
  cp "$3" "$state/evidence.json"
  exit 0
fi
if [[ "$1 $2 $3 $4" == "alpha compute tpus tpu-vm" ]]; then
  command=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--command" ]]; then command="$argument"; break; fi
    previous="$argument"
  done
  if [[ "$command" == *"sunfish-exact-process-interrupt.py"* ]]; then
    touch "$state/killed"
    exit 0
  fi
  if [[ "$command" == *"stage05-kill-integration"* ]]; then
    touch "$state/preempt-marker"
    while [[ ! -f "$state/killed" ]]; do sleep 0.01; done
    exit 137
  fi
  if [[ "$command" == *"stage05-resume-integration"* ]]; then
    cp "$state/resume-metric-template.json" "$state/resume-metric"
    touch "$state/final-marker" "$state/train-complete"
    exit 0
  fi
fi
exit 2
""",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "SUNFISH_FAKE_GCLOUD_STATE": str(state),
                "SUNFISH_TRAIN_BIN": "fake-sunfish-train",
                "SUNFISH_CONTROLLER_LOG_DIR": str(root / "logs"),
                "TPU_NAME": "fake-tpu",
                "PROJECT_ID": "fake-project",
                "ZONE": "fake-zone",
                "REMOTE_REPO_DIR": str(ROOT),
                "EXPECTED_TPU_DEVICES": "32",
                "EXPECTED_TPU_PROCESSES": "8",
                "EXPECTED_LOCAL_TPU_DEVICES": "4",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                payload = run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="stage05-kill-integration",
                    resume_attempt="stage05-resume-integration",
                    preempt_after_step=25,
                    evidence_uri=(
                        "gs://fake/sunfish/preemption/preemption-summary.json"
                    ),
                    timeout_seconds=5,
                    poll_seconds=0,
                )
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["preempted_launch_returncode"], 137)
            self.assertEqual(payload["resumed_launch_returncode"], 0)
            self.assertTrue(payload["resume_continued_from_checkpoint"])
            self.assertTrue(payload["exact_recorded_processes_interrupted"])
            self.assertEqual(
                payload["interrupt_process_policy"],
                "pre-signal-exact-root-and-descendant-snapshot-with-pidfd",
            )
            self.assertTrue(payload["same_attempt_descendants_absent"])
            self.assertFalse(payload["owner_intervention_required"])
            self.assertEqual(payload["interrupt_timeout_seconds"], 120)
            self.assertEqual(payload["resume_proof"]["step"], 25)
            self.assertTrue((state / "killed").exists())
            uploaded = json.loads((state / "evidence.json").read_text())
            self.assertTrue(uploaded["automatic_same_workdir_restore"])

    def test_abnormal_cleanup_failure_overrides_with_owner_stop(self):
        config_path = ROOT / "configs/training/sunfish-smoke.toml"
        process = mock.Mock()
        process.pid = 98766
        process.poll.return_value = None
        process.wait.return_value = 143
        events = []

        def failed_remote_interrupt(*_args, **_kwargs):
            events.append("remote-interrupt")
            raise OSError("simulated IAP cleanup failure")

        def terminate_local_group(process_group_id, sent_signal):
            self.assertEqual(process_group_id, process.pid)
            self.assertEqual(sent_signal, signal.SIGTERM)
            events.append("local-group-terminate")

        with (
            mock.patch(
                "sunfish_tpu.preemption_gate._gcloud_exists", return_value=False
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate._wait_for_uri",
                side_effect=TimeoutError("original checkpoint timeout"),
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate.subprocess.Popen",
                return_value=process,
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate._invoke_exact_remote_interrupt",
                side_effect=failed_remote_interrupt,
            ) as cleanup_run,
            mock.patch(
                "sunfish_tpu.preemption_gate.os.killpg",
                side_effect=terminate_local_group,
            ),
            mock.patch(
                "sunfish_tpu.preemption_gate._wait_for_process_group_exit",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {
                    "SUNFISH_TRAIN_BIN": "fake-sunfish-train",
                    "SUNFISH_REMOTE_CONFIG": "/remote/sunfish-smoke.toml",
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(
                OwnerInterventionRequiredError,
                "cleanup is unproven.*owner intervention.*automatic resume/retry",
            ):
                run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="abnormal-interrupt-001",
                    resume_attempt="abnormal-resume-001",
                    preempt_after_step=25,
                    evidence_uri="gs://fake/readiness/preemption.json",
                    timeout_seconds=1,
                    poll_seconds=0,
                )

        self.assertEqual(events, ["remote-interrupt", "local-group-terminate"])
        cleanup_command = cleanup_run.call_args.args[0]
        self.assertTrue(cleanup_command[0].endswith("interrupt_training_attempt.sh"))
        self.assertEqual(
            cleanup_command[-4:],
            [
                "--run-id",
                HarnessConfig.load(config_path).run.run_id,
                "--attempt-id",
                "abnormal-interrupt-001",
            ],
        )

    def test_preemption_template_only_changes_run_identity_and_workdir(self):
        smoke = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        for name in ("sunfish-preemption-smoke.toml", "sunfish-resume-smoke.toml"):
            diagnostic = HarnessConfig.load(ROOT / "configs/training" / name)
            smoke_payload = smoke.canonical_dict()
            diagnostic_payload = diagnostic.canonical_dict()
            for payload in (smoke_payload, diagnostic_payload):
                payload["run"].pop("run_id")
                payload["run"].pop("workdir")
            self.assertEqual(smoke_payload, diagnostic_payload)

    def test_plan_uses_pinned_orbax_gcs_finalization_marker(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        plan = build_preemption_plan(
            config,
            preempt_attempt="interrupt-001",
            resume_attempt="resume-001",
            preempt_after_step=25,
        )
        self.assertEqual(
            plan["preempt_marker"],
            checkpoint_commit_marker(config.run.workdir, 25),
        )
        self.assertTrue(plan["preempt_marker"].endswith("/commit_success.txt"))
        self.assertTrue(plan["train_complete"].endswith("/train_complete.txt"))
        self.assertTrue(
            plan["resume_first_metric"].endswith(
                "/readiness/resume-001/metrics/step-000000025.json"
            )
        )
        self.assertTrue(
            plan["fresh_start_metric"].endswith(
                "/readiness/resume-001/metrics/step-000000000.json"
            )
        )

    def test_preemption_must_happen_at_an_interior_checkpoint(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        with self.assertRaises(ValueError):
            build_preemption_plan(
                config,
                preempt_attempt="same",
                resume_attempt="same",
                preempt_after_step=25,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint step"):
            build_preemption_plan(
                config,
                preempt_attempt="kill",
                resume_attempt="resume",
                preempt_after_step=24,
            )
        non_smoke = dataclasses.replace(
            config,
            run=dataclasses.replace(config.run, phase=Phase.ROUTER),
        )
        with self.assertRaisesRegex(ValueError, "phase=smoke"):
            build_preemption_plan(
                non_smoke,
                preempt_attempt="kill",
                resume_attempt="resume",
                preempt_after_step=25,
            )


if __name__ == "__main__":
    unittest.main()
