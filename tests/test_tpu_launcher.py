import hashlib
import importlib.util
import os
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from sunfish.source_tree import workspace_source_identity
from sunfish_tpu.deployment_config import render_stage05_configs
from tests.test_parity_evidence import valid_parity_payload


class AllHostLauncherTests(unittest.TestCase):
    @staticmethod
    def _render_bundle(root: Path, temporary_path: Path, tag: str) -> Path:
        source = workspace_source_identity(root)
        parity_payload = valid_parity_payload()
        parity_payload["sunfish_source"] = source
        parity_payload["environment"]["float32"]["sunfish_source"] = source
        parity_payload["environment"]["bfloat16"]["sunfish_source"] = source
        parity = temporary_path / f"{tag}-parity.json"
        parity.write_text(json.dumps(parity_payload), encoding="utf-8")
        bundle = temporary_path / f"{tag}-bundle"
        render_stage05_configs(
            template_directory=root / "configs/training",
            output_directory=bundle,
            storage_root="gs://bucket/sunfish",
            run_tag=tag,
            dataset_manifest_sha256="a" * 64,
            seed_manifest_sha256="b" * 64,
            parity_report_path=parity,
            expected_devices=32,
            expected_processes=8,
            expected_local_devices=4,
            source_root=root,
        )
        return bundle

    def test_rendered_config_uploader_copies_and_verifies_every_worker(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            bundle = self._render_bundle(root, temporary_path, "upload-test")
            capture = temporary_path / "gcloud-args.txt"
            fake_gcloud = temporary_path / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\nprintf 'CALL\\n' >> \"$CAPTURE\"\nprintf '%s\\n' \"$@\" >> \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                **os.environ,
                "CAPTURE": str(capture),
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
                "SUNFISH_OFFLINE_BUNDLE_ROOT": "/home/sunfish/releases/a",
            }
            subprocess.run(
                [
                    "bash",
                    str(root / "scripts/upload_tpu_configs.sh"),
                    "--local-dir",
                    str(bundle),
                    "--remote-dir",
                    "/home/sunfish/deploy/grant-001",
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            calls = capture.read_text(encoding="utf-8")
            self.assertEqual(calls.count("CALL\n"), 12)
            self.assertEqual(calls.count("--worker=all\n"), 12)
            self.assertEqual(calls.count("--tunnel-through-iap\n"), 12)
            self.assertEqual(calls.count("--batch-size=all\n"), 7)
            self.assertEqual(calls.count("alpha\n"), 12)
            self.assertIn("tpu-vm\nscp\n", calls)
            self.assertIn("sunfish-preemption-smoke.toml", calls)
            self.assertIn("/home/sunfish/deploy/grant-001", calls)
            self.assertIn(".upload-", calls)
            self.assertIn("prepare", calls)
            self.assertIn("publish-files", calls)
            self.assertIn(
                "/home/sunfish/releases/a/python/bin/python3", calls
            )

    def test_unproven_remote_cleanup_returns_nonretryable_owner_stop(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cleanup = temporary_path / "cleanup"
            cleanup.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
            cleanup.chmod(0o755)
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/controller_attached_launch.py"),
                    "--log",
                    str(temporary_path / "launch.log"),
                    "--cleanup-script",
                    str(cleanup),
                    "--run-id",
                    "owner-stop-run",
                    "--attempt-id",
                    "owner-stop-attempt",
                    "--",
                    "sh",
                    "-c",
                    "exit 42",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 126, result.stderr)
            self.assertIn("non-retryable owner-intervention", result.stderr)

    def test_controller_opens_log_before_spawning_attached_command(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            command_marker = temporary_path / "command-started"
            cleanup = temporary_path / "cleanup"
            cleanup.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cleanup.chmod(0o755)
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/controller_attached_launch.py"),
                    "--log",
                    str(temporary_path),
                    "--cleanup-script",
                    str(cleanup),
                    "--run-id",
                    "log-open-run",
                    "--attempt-id",
                    "log-open-attempt",
                    "--",
                    "sh",
                    "-c",
                    f"touch {command_marker}",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 125, result.stderr)
            self.assertFalse(command_marker.exists())

    def test_controller_signal_after_spawn_completes_cleanup_before_return(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            remote_marker = temporary_path / "remote-live"
            cleanup_marker = temporary_path / "cleanup-ran"
            cleanup = temporary_path / "cleanup"
            cleanup.write_text(
                "#!/bin/sh\n"
                "touch \"$CLEANUP_MARKER\"\n"
                "kill -TERM \"$PPID\"\n"
                "rm -f \"$REMOTE_MARKER\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            cleanup.chmod(0o755)
            command = temporary_path / "signal-controller"
            command.write_text(
                "#!/bin/sh\n"
                "touch \"$REMOTE_MARKER\"\n"
                "kill -TERM \"$PPID\"\n"
                "sleep 300\n",
                encoding="utf-8",
            )
            command.chmod(0o755)
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/controller_attached_launch.py"),
                    "--log",
                    str(temporary_path / "launch.log"),
                    "--cleanup-script",
                    str(cleanup),
                    "--run-id",
                    "signal-run",
                    "--attempt-id",
                    "signal-attempt",
                    "--",
                    str(command),
                ],
                cwd=root,
                env={
                    **os.environ,
                    "REMOTE_MARKER": str(remote_marker),
                    "CLEANUP_MARKER": str(cleanup_marker),
                },
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 128 + signal.SIGTERM, result.stderr)
            self.assertTrue(cleanup_marker.exists())
            self.assertFalse(remote_marker.exists())
            self.assertIn("interrupting the exact remote run/attempt", result.stderr)

    def test_post_spawn_exception_with_unproven_cleanup_is_hard_stop(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "sunfish_controller_attached_launch_test",
            root / "scripts/controller_attached_launch.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cleanup_marker = temporary_path / "cleanup-ran"
            cleanup = temporary_path / "cleanup"
            cleanup.write_text(
                "#!/bin/sh\n"
                "touch \"$CLEANUP_MARKER\"\n"
                "exit 7\n",
                encoding="utf-8",
            )
            cleanup.chmod(0o755)
            with mock.patch.dict(
                os.environ,
                {"CLEANUP_MARKER": str(cleanup_marker)},
            ), mock.patch.object(
                module.threading.Thread,
                "start",
                side_effect=RuntimeError("injected relay-start failure"),
            ):
                result = module.run_attached(
                    ["sh", "-c", "sleep 300"],
                    log_path=temporary_path / "launch.log",
                    cleanup_command=[str(cleanup)],
                )
            self.assertEqual(result, 126)
            self.assertTrue(cleanup_marker.exists())

    def test_uploader_refuses_tampered_bundle_before_gcloud(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            bundle = self._render_bundle(root, temporary_path, "upload-tamper")
            with (bundle / "sunfish-resume-smoke.toml").open(
                "a", encoding="utf-8"
            ) as file:
                file.write("# tampered\n")
            capture = temporary_path / "gcloud-called"
            fake_gcloud = temporary_path / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\ntouch \"$CAPTURE\"\n", encoding="utf-8"
            )
            fake_gcloud.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    str(root / "scripts/upload_tpu_configs.sh"),
                    "--local-dir",
                    str(bundle),
                    "--remote-dir",
                    "/home/sunfish/deploy/upload-tamper",
                ],
                cwd=root,
                env={
                    **os.environ,
                    "CAPTURE": str(capture),
                    "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                    "TPU_NAME": "sunfish-v4",
                    "PROJECT_ID": "sunfish-project",
                    "ZONE": "us-central2-b",
                    "SUNFISH_OFFLINE_BUNDLE_ROOT": "/home/sunfish/releases/a",
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())

    def test_gcloud_targets_all_workers_with_one_run_identity(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            bundle = self._render_bundle(root, temporary_path, "launcher-test")
            capture = temporary_path / "gcloud-args.txt"
            fake_gcloud = temporary_path / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                **os.environ,
                "CAPTURE": str(capture),
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "SUNFISH_CONTROLLER_LOG_DIR": str(temporary_path / "logs"),
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
                "REMOTE_REPO_DIR": "/home/sunfish/sunfish-v2",
                "EXPECTED_TPU_DEVICES": "32",
                "EXPECTED_TPU_PROCESSES": "8",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
            subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "launch_tpu_pod.sh"),
                    "--run-id",
                    "20260711-smoke",
                    "--attempt-id",
                    "attempt-001",
                    "--config",
                    str(bundle / "sunfish-smoke.toml"),
                    "--remote-config",
                    "/home/sunfish/configs/deploy-smoke.toml",
                    "--",
                    ".venv-tpu/bin/sunfish-tpu-preflight",
                    "--require-tpu",
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:5], ["alpha", "compute", "tpus", "tpu-vm", "ssh"])
            self.assertIn("--worker=all", arguments)
            self.assertIn("--batch-size=all", arguments)
            self.assertIn("--tunnel-through-iap", arguments)
            command = arguments[arguments.index("--command") + 1]
            self.assertIn("20260711-smoke", command)
            self.assertIn("--attempt-id attempt-001", command)
            self.assertIn("/home/sunfish/configs/deploy-smoke.toml", command)
            self.assertIn("--expected-devices 32", command)
            self.assertIn("--expected-processes 8", command)
            self.assertIn("--xla-python-client-preallocate false", command)
            self.assertRegex(command, r"--config-sha256 [0-9a-f]{64}")
            self.assertRegex(command, r"--expected-commit [0-9a-f]{40}")
            self.assertRegex(command, r"--source-tree-sha256 [0-9a-f]{64}")

    def test_launcher_rejects_missing_or_invalid_xla_preallocation_policy(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "gcloud-called"
            fake_gcloud = Path(temporary) / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\ntouch \"$CAPTURE\"\n", encoding="utf-8"
            )
            fake_gcloud.chmod(0o755)
            base_environment = {
                **os.environ,
                "CAPTURE": str(capture),
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
                "REMOTE_REPO_DIR": "/home/sunfish/sunfish-v2",
                "EXPECTED_TPU_DEVICES": "32",
                "EXPECTED_TPU_PROCESSES": "8",
            }
            for policy in (None, "FALSE", "0"):
                with self.subTest(policy=policy):
                    environment = dict(base_environment)
                    environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
                    if policy is not None:
                        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = policy
                    result = subprocess.run(
                        [
                            "bash",
                            str(root / "scripts/launch_tpu_pod.sh"),
                            "--run-id",
                            "invalid-xla-policy",
                            "--config",
                            str(root / "configs/sunfish-8b-a3b.toml"),
                            "--",
                            "true",
                        ],
                        cwd=root,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(capture.exists())

    def test_long_attempt_requires_live_tmux_acknowledgements_and_budget(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            bundle = self._render_bundle(root, temporary_path, "durable-launch")
            capture = temporary_path / "gcloud-args.txt"
            fake_gcloud = temporary_path / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            fake_tmux = temporary_path / "tmux"
            fake_tmux.write_text(
                "#!/usr/bin/env bash\n"
                "[[ \"$1\" == display-message && \"$2\" == -p ]] || exit 2\n"
                "printf '$7:sunfish-production\\n'\n",
                encoding="utf-8",
            )
            fake_tmux.chmod(0o755)
            log_directory = temporary_path / "logs"
            environment = {
                **os.environ,
                "CAPTURE": str(capture),
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "SUNFISH_TMUX_BIN": str(fake_tmux),
                "SUNFISH_CONTROLLER_LOG_DIR": str(log_directory),
                "SUNFISH_CONTROLLER_STAYS_AWAKE_ACK": "1",
                "SUNFISH_CONTROLLER_NETWORK_STABLE_ACK": "1",
                "TMUX": "/tmp/tmux-1000/default,1,0",
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
                "REMOTE_REPO_DIR": "/home/sunfish/sunfish-v2",
                "EXPECTED_TPU_DEVICES": "32",
                "EXPECTED_TPU_PROCESSES": "8",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
            command = [
                "bash",
                str(root / "scripts/launch_tpu_pod.sh"),
                "--run-id",
                "production-router",
                "--attempt-id",
                "production-router-002",
                "--config",
                str(bundle / "sunfish-smoke.toml"),
                "--remote-config",
                "/home/sunfish/configs/sunfish-smoke.toml",
                "--require-durable-controller",
                "--attempt-number",
                "2",
                "--max-attempts",
                "3",
                "--",
                ".venv-tpu/bin/sunfish-train",
                "--config",
                "/home/sunfish/configs/sunfish-smoke.toml",
            ]
            subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(capture.exists())
            contract = (
                log_directory
                / "production-router"
                / "production-router-002"
                / "durable-controller-contract.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("tmux_session=$7:sunfish-production", contract)
            self.assertIn("attempt_number=2", contract)
            self.assertIn("max_attempts=3", contract)
            self.assertIn("ssh_server_alive_interval_seconds=30", contract)
            self.assertIn("cleanup_hard_stop_exit_status=126", contract)
            self.assertIn("cleanup_hard_stop_retry_allowed=0", contract)

            for missing in (
                "TMUX",
                "SUNFISH_CONTROLLER_STAYS_AWAKE_ACK",
                "SUNFISH_CONTROLLER_NETWORK_STABLE_ACK",
            ):
                with self.subTest(missing=missing):
                    capture.unlink(missing_ok=True)
                    failed_environment = dict(environment)
                    failed_environment.pop(missing)
                    result = subprocess.run(
                        command,
                        cwd=root,
                        env=failed_environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(capture.exists())

            capture.unlink(missing_ok=True)
            over_budget = list(command)
            over_budget[over_budget.index("2")] = "4"
            result = subprocess.run(
                over_budget,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())
            self.assertIn("exceeds", result.stderr)

    def test_failed_iap_launch_interrupts_exact_remote_attempt_before_return(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            bundle = self._render_bundle(root, temporary_path, "failed-iap")
            capture = temporary_path / "gcloud-calls.txt"
            remote_marker = temporary_path / "remote-trainer-alive"
            local_orphan_pid = temporary_path / "local-orphan.pid"
            fake_gcloud = temporary_path / "gcloud"
            fake_gcloud.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf 'CALL\n' >> "$CAPTURE"
printf '%s\n' "$@" >> "$CAPTURE"
cleanup=0
for argument in "$@"; do
  case "$argument" in
    *'sunfish-exact-process-interrupt.py'*) cleanup=1 ;;
  esac
done
if [[ "$cleanup" == 1 ]]; then
  rm -f "$REMOTE_MARKER"
  exit 0
fi
touch "$REMOTE_MARKER"
sleep 300 &
printf '%s\n' "$!" > "$LOCAL_ORPHAN_PID"
exit 42
""",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                **os.environ,
                "CAPTURE": str(capture),
                "REMOTE_MARKER": str(remote_marker),
                "LOCAL_ORPHAN_PID": str(local_orphan_pid),
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "SUNFISH_CONTROLLER_LOG_DIR": str(temporary_path / "logs"),
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
                "REMOTE_REPO_DIR": "/home/sunfish/sunfish-v2",
                "EXPECTED_TPU_DEVICES": "32",
                "EXPECTED_TPU_PROCESSES": "8",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
            result = subprocess.run(
                [
                    "bash",
                    str(root / "scripts/launch_tpu_pod.sh"),
                    "--run-id",
                    "failed-iap-run",
                    "--attempt-id",
                    "failed-iap-attempt",
                    "--config",
                    str(bundle / "sunfish-smoke.toml"),
                    "--remote-config",
                    "/home/sunfish/configs/sunfish-smoke.toml",
                    "--",
                    ".venv-tpu/bin/sunfish-train",
                    "--config",
                    "/home/sunfish/configs/sunfish-smoke.toml",
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertFalse(remote_marker.exists())
            orphan_pid = int(local_orphan_pid.read_text(encoding="ascii"))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(orphan_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.kill(orphan_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.fail(f"controller-local gcloud descendant survived: {orphan_pid}")
            calls = capture.read_text(encoding="utf-8")
            self.assertEqual(calls.count("CALL\n"), 2)
            self.assertIn("--run-id failed-iap-run", calls)
            self.assertIn("--attempt-id failed-iap-attempt", calls)
            self.assertIn("sunfish-exact-process-interrupt.py", calls)
            self.assertIn("interrupting the exact remote run/attempt", result.stderr)

    def test_preemption_interrupt_targets_only_one_attempt_on_all_workers(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            capture = temporary_path / "gcloud-args.txt"
            fake_gcloud = temporary_path / "gcloud"
            fake_gcloud.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                **os.environ,
                "CAPTURE": str(capture),
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
            }
            subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "interrupt_training_attempt.sh"),
                    "--run-id",
                    "sunfish-smoke",
                    "--attempt-id",
                    "interrupt-001",
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:5], ["alpha", "compute", "tpus", "tpu-vm", "ssh"])
            self.assertIn("--worker=all", arguments)
            self.assertIn("--batch-size=all", arguments)
            self.assertIn("--tunnel-through-iap", arguments)
            command = arguments[arguments.index("--command") + 1]
            self.assertIn("sunfish-exact-process-interrupt.py", command)
            self.assertIn("--run-id sunfish-smoke", command)
            self.assertIn("--attempt-id interrupt-001", command)
            self.assertIn("--pid-root", command)
            self.assertIn("SUNFISH_PID_ROOT", command)
            helper = (root / "src/sunfish_tpu/exact_process_interrupt.py").read_text()
            self.assertIn("pidfd_send_signal", helper)
            self.assertIn("exact_recorded_descendants", helper)
            self.assertIn("owner intervention is required", helper)
            self.assertNotIn("killpg", helper)
            subprocess.run(["bash", "-n", "-c", command], check=True)

    def test_host_entrypoint_writes_a_per_host_log(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            log_root = Path(temporary) / "host-logs"
            environment = {
                **os.environ,
                "SUNFISH_HOST_LOG_ROOT": str(log_root),
                "SUNFISH_PID_ROOT": str(Path(temporary) / "pids"),
                "SUNFISH_PYTHON_BIN": str(Path(temporary) / "missing-python"),
                "SUNFISH_REMOTE_PYTHON_BIN": sys.executable,
            }
            expected_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source_digest = subprocess.run(
                ["python3", "scripts/source_tree_digest.py", "--root", "."],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            config_sha256 = subprocess.run(
                [
                    "python3",
                    "-c",
                    "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('configs/sunfish-8b-a3b.toml').read_bytes()).hexdigest())",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "tpu_host_entrypoint.sh"),
                    "--run-id",
                    "20260711-host-log",
                    "--attempt-id",
                    "attempt-host-001",
                    "--config",
                    "configs/sunfish-8b-a3b.toml",
                    "--config-sha256",
                    config_sha256,
                    "--expected-devices",
                    "64",
                    "--expected-processes",
                    "8",
                    "--xla-python-client-preallocate",
                    "false",
                    "--expected-commit",
                    expected_commit,
                    "--source-tree-sha256",
                    source_digest,
                    "--",
                    "bash",
                    "-c",
                    "test \"$SUNFISH_TPU_WORKER\" = 1 && test \"$XLA_PYTHON_CLIENT_PREALLOCATE\" = false && echo host-command-ran",
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            logs = list(
                (log_root / "20260711-host-log" / "attempt-host-001").glob("*.log")
            )
            self.assertEqual(len(logs), 1)
            content = logs[0].read_text(encoding="utf-8")
            self.assertIn("run_id=20260711-host-log", content)
            self.assertIn("attempt_id=attempt-host-001", content)
            self.assertIn(f"commit={expected_commit}", content)
            self.assertIn(f"source_tree={source_digest}", content)
            self.assertIn("host-command-ran", content)
            self.assertEqual(list((Path(temporary) / "pids").glob("*.pid")), [])

    def test_host_entrypoint_forwards_term_and_cleans_the_exact_pid_file(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            ready = temporary_path / "child-ready"
            signal_marker = temporary_path / "child-signal"
            child = temporary_path / "signal-child.sh"
            child.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
trap 'printf TERM > "${SIGNAL_MARKER}"; exit 42' TERM
printf '%s\n' "$$" > "${READY_MARKER}"
while true; do sleep 0.05; done
""",
                encoding="utf-8",
            )
            child.chmod(0o755)
            pid_root = temporary_path / "pids"
            source = workspace_source_identity(root)
            config = root / "configs/sunfish-8b-a3b.toml"
            config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
            environment = {
                **os.environ,
                "READY_MARKER": str(ready),
                "SIGNAL_MARKER": str(signal_marker),
                "SUNFISH_HOST_LOG_ROOT": str(temporary_path / "host-logs"),
                "SUNFISH_PID_ROOT": str(pid_root),
                "SUNFISH_PYTHON_BIN": str(temporary_path / "missing-python"),
                "SUNFISH_REMOTE_PYTHON_BIN": sys.executable,
            }
            process = subprocess.Popen(
                [
                    "bash",
                    str(root / "scripts/tpu_host_entrypoint.sh"),
                    "--run-id",
                    "host-signal-test",
                    "--attempt-id",
                    "term-001",
                    "--config",
                    str(config),
                    "--config-sha256",
                    config_sha256,
                    "--expected-devices",
                    "32",
                    "--expected-processes",
                    "8",
                    "--xla-python-client-preallocate",
                    "false",
                    "--expected-commit",
                    source["git_commit"],
                    "--source-tree-sha256",
                    source["source_tree_sha256"],
                    "--",
                    str(child),
                ],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not ready.exists():
                time.sleep(0.01)
            if not ready.exists():
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
                self.fail(f"host child did not start\nstdout={stdout}\nstderr={stderr}")

            pid_files = list(pid_root.glob("*.pid"))
            self.assertEqual(len(pid_files), 1)
            child_pid = int(pid_files[0].read_text(encoding="utf-8"))
            self.assertEqual(child_pid, int(ready.read_text(encoding="utf-8")))
            process.send_signal(signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate(timeout=5)
                self.fail("host entrypoint did not finish after forwarding TERM")

            self.assertNotEqual(process.returncode, 0, (stdout, stderr))
            self.assertEqual(signal_marker.read_text(encoding="utf-8"), "TERM")
            self.assertEqual(list(pid_root.glob("*.pid")), [])
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

            host_source = (root / "scripts/tpu_host_entrypoint.sh").read_text(
                encoding="utf-8"
            )
            for name in ("TERM", "HUP", "INT"):
                self.assertIn(f"trap 'forward_signal {name}' {name}", host_source)

    def test_host_entrypoint_logger_exits_when_a_descendant_retains_stdout(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            ready = temporary_path / "descendant-ready"
            child = temporary_path / "descendant-child.sh"
            child.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
trap 'exit 42' TERM
(while true; do printf 'descendant-still-writing\n'; sleep 0.01; done) &
printf '%s\n' "$!" > "${READY_MARKER}"
while true; do sleep 0.05; done
""",
                encoding="utf-8",
            )
            child.chmod(0o755)
            pid_root = temporary_path / "pids"
            log_root = temporary_path / "host-logs"
            source = workspace_source_identity(root)
            config = root / "configs/sunfish-8b-a3b.toml"
            environment = {
                **os.environ,
                "READY_MARKER": str(ready),
                "SUNFISH_HOST_LOG_ROOT": str(log_root),
                "SUNFISH_PID_ROOT": str(pid_root),
                "SUNFISH_PYTHON_BIN": str(temporary_path / "missing-python"),
                "SUNFISH_REMOTE_PYTHON_BIN": sys.executable,
            }
            process = subprocess.Popen(
                [
                    "bash",
                    str(root / "scripts/tpu_host_entrypoint.sh"),
                    "--run-id",
                    "host-descendant-test",
                    "--attempt-id",
                    "term-001",
                    "--config",
                    str(config),
                    "--config-sha256",
                    hashlib.sha256(config.read_bytes()).hexdigest(),
                    "--expected-devices",
                    "32",
                    "--expected-processes",
                    "8",
                    "--xla-python-client-preallocate",
                    "false",
                    "--expected-commit",
                    source["git_commit"],
                    "--source-tree-sha256",
                    source["source_tree_sha256"],
                    "--",
                    str(child),
                ],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            descendant_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not ready.exists():
                    time.sleep(0.01)
                if not ready.exists():
                    stdout, stderr = process.communicate(timeout=5)
                    self.fail(
                        "host child did not create descendant\n"
                        f"stdout={stdout}\nstderr={stderr}"
                    )
                descendant_pid = int(ready.read_text(encoding="ascii"))
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
                self.assertNotEqual(process.returncode, 0, (stdout, stderr))
                self.assertEqual(list(pid_root.glob("*.pid")), [])
                relay_artifacts = [
                    path
                    for path in log_root.rglob("*")
                    if any(marker in path.name for marker in (".pipe.", ".stop.", ".ready."))
                ]
                self.assertEqual(relay_artifacts, [])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
                if descendant_pid is not None:
                    try:
                        os.kill(descendant_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_host_entrypoint_stop_marker_does_not_follow_a_raced_symlink(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            ready = temporary_path / "child-ready"
            child = temporary_path / "quiet-child.sh"
            child.write_text(
                """#!/usr/bin/env bash
trap 'exit 42' TERM
printf ready > "${READY_MARKER}"
while true; do sleep 0.05; done
""",
                encoding="utf-8",
            )
            child.chmod(0o755)
            log_root = temporary_path / "host-logs"
            source = workspace_source_identity(root)
            config = root / "configs/sunfish-8b-a3b.toml"
            process = subprocess.Popen(
                [
                    "bash",
                    str(root / "scripts/tpu_host_entrypoint.sh"),
                    "--run-id",
                    "host-stop-symlink-test",
                    "--attempt-id",
                    "term-001",
                    "--config",
                    str(config),
                    "--config-sha256",
                    hashlib.sha256(config.read_bytes()).hexdigest(),
                    "--expected-devices",
                    "32",
                    "--expected-processes",
                    "8",
                    "--xla-python-client-preallocate",
                    "false",
                    "--expected-commit",
                    source["git_commit"],
                    "--source-tree-sha256",
                    source["source_tree_sha256"],
                    "--",
                    str(child),
                ],
                cwd=root,
                env={
                    **os.environ,
                    "READY_MARKER": str(ready),
                    "SUNFISH_HOST_LOG_ROOT": str(log_root),
                    "SUNFISH_PID_ROOT": str(temporary_path / "pids"),
                    "SUNFISH_PYTHON_BIN": str(temporary_path / "missing-python"),
                    "SUNFISH_REMOTE_PYTHON_BIN": sys.executable,
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not ready.exists():
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                relay_ready = list(log_root.rglob("*.ready.*"))
                self.assertEqual(len(relay_ready), 1)
                stop_marker = Path(str(relay_ready[0]).replace(".ready.", ".stop."))
                victim = temporary_path / "victim"
                victim.write_text("unchanged", encoding="utf-8")
                stop_marker.symlink_to(victim)
                time.sleep(0.2)
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=10)
                self.assertNotEqual(process.returncode, 0, (stdout, stderr))
                self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged")
                self.assertFalse(stop_marker.exists())
                self.assertFalse(stop_marker.is_symlink())
                self.assertIn("stop marker", stderr)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

    def test_raced_pid_symlink_stops_and_reaps_the_unpublished_child(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            pip_ready = temporary_path / "pip-ready"
            release_pip = temporary_path / "release-pip"
            fake_python = temporary_path / "fake-python"
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf ready > "${PIP_READY}"
while [[ ! -e "${RELEASE_PIP}" ]]; do sleep 0.01; done
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            workload_started = temporary_path / "workload-started"
            child = temporary_path / "long-child.sh"
            child.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
sleep 300 &
printf '%s %s\n' "$$" "$!" > "${WORKLOAD_STARTED}"
wait
""",
                encoding="utf-8",
            )
            child.chmod(0o755)
            pid_root = temporary_path / "pids"
            source = workspace_source_identity(root)
            config = root / "configs/sunfish-8b-a3b.toml"
            run_id = "host-pid-race-test"
            attempt_id = "race-001"
            process = subprocess.Popen(
                [
                    "bash",
                    str(root / "scripts/tpu_host_entrypoint.sh"),
                    "--run-id",
                    run_id,
                    "--attempt-id",
                    attempt_id,
                    "--config",
                    str(config),
                    "--config-sha256",
                    hashlib.sha256(config.read_bytes()).hexdigest(),
                    "--expected-devices",
                    "32",
                    "--expected-processes",
                    "8",
                    "--xla-python-client-preallocate",
                    "false",
                    "--expected-commit",
                    source["git_commit"],
                    "--source-tree-sha256",
                    source["source_tree_sha256"],
                    "--",
                    str(child),
                ],
                cwd=root,
                env={
                    **os.environ,
                    "PIP_READY": str(pip_ready),
                    "RELEASE_PIP": str(release_pip),
                    "WORKLOAD_STARTED": str(workload_started),
                    "SUNFISH_HOST_LOG_ROOT": str(temporary_path / "host-logs"),
                    "SUNFISH_PID_ROOT": str(pid_root),
                    "SUNFISH_PYTHON_BIN": str(fake_python),
                    "SUNFISH_REMOTE_PYTHON_BIN": sys.executable,
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not pip_ready.exists():
                    time.sleep(0.01)
                self.assertTrue(pip_ready.exists())
                raw_host = subprocess.run(
                    ["hostname"], check=True, capture_output=True, text=True
                ).stdout
                host = "".join(
                    character
                    if character.isascii()
                    and (character.isalnum() or character in "._-")
                    else "_"
                    for character in raw_host
                )
                pid_file = pid_root / f"{run_id}.{attempt_id}.{host}.pid"
                victim = temporary_path / "pid-victim"
                victim.write_text("unchanged", encoding="utf-8")
                pid_file.symlink_to(victim)
                release_pip.write_text("go", encoding="ascii")
                stdout, stderr = process.communicate(timeout=15)
                self.assertNotEqual(process.returncode, 0, (stdout, stderr))
                match = re.search(r"child_pid=([1-9][0-9]*)", stderr)
                self.assertIsNotNone(match, stderr)
                child_pid = int(match.group(1))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertTrue(pid_file.is_symlink())
                self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged")
                self.assertIn("did not publish", stderr)
                self.assertFalse(
                    workload_started.exists(),
                    "the PID-publication gate must prevent root/descendant spawn",
                )
            finally:
                release_pip.touch(exist_ok=True)
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_host_entrypoint_rejects_config_bytes_before_command(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "ran"
            expected_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source_digest = subprocess.run(
                ["python3", "scripts/source_tree_digest.py", "--root", "."],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result = subprocess.run(
                [
                    "bash",
                    str(root / "scripts/tpu_host_entrypoint.sh"),
                    "--run-id",
                    "config-mismatch",
                    "--config",
                    "configs/sunfish-8b-a3b.toml",
                    "--config-sha256",
                    "0" * 64,
                    "--expected-devices",
                    "8",
                    "--expected-processes",
                    "2",
                    "--xla-python-client-preallocate",
                    "false",
                    "--expected-commit",
                    expected_commit,
                    "--source-tree-sha256",
                    source_digest,
                    "--",
                    "touch",
                    str(marker),
                ],
                cwd=root,
                env={**os.environ, "SUNFISH_REMOTE_PYTHON_BIN": sys.executable},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("config differs", result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
