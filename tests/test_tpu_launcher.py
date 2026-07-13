import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

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
            self.assertIn("tpu-vm\nscp\n", calls)
            self.assertIn("sunfish-preemption-smoke.toml", calls)
            self.assertIn("/home/sunfish/deploy/grant-001", calls)
            self.assertIn(".upload-", calls)
            self.assertIn("test ! -e", calls)
            self.assertIn("mv", calls)

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
            self.assertIn("--worker=all", arguments)
            command = arguments[arguments.index("--command") + 1]
            self.assertIn("20260711-smoke", command)
            self.assertIn("--attempt-id attempt-001", command)
            self.assertIn("/home/sunfish/configs/deploy-smoke.toml", command)
            self.assertIn("--expected-devices 32", command)
            self.assertIn("--expected-processes 8", command)
            self.assertRegex(command, r"--config-sha256 [0-9a-f]{64}")
            self.assertRegex(command, r"--expected-commit [0-9a-f]{40}")
            self.assertRegex(command, r"--source-tree-sha256 [0-9a-f]{64}")

    def test_preemption_kill_targets_only_one_attempt_on_all_workers(self):
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
                    str(root / "scripts" / "kill_tpu_attempt.sh"),
                    "--run-id",
                    "sunfish-smoke",
                    "--attempt-id",
                    "kill-001",
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertIn("--worker=all", arguments)
            command = arguments[arguments.index("--command") + 1]
            self.assertIn("sunfish-smoke.kill-001.", command)
            self.assertIn("*.pid", command)
            self.assertIn("kill -KILL", command)

    def test_host_entrypoint_writes_a_per_host_log(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            log_root = Path(temporary) / "host-logs"
            environment = {
                **os.environ,
                "SUNFISH_HOST_LOG_ROOT": str(log_root),
                "SUNFISH_PID_ROOT": str(Path(temporary) / "pids"),
                "SUNFISH_PYTHON_BIN": str(Path(temporary) / "missing-python"),
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
                    "--expected-commit",
                    expected_commit,
                    "--source-tree-sha256",
                    source_digest,
                    "--",
                    "bash",
                    "-c",
                    "echo host-command-ran",
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
                    "--expected-commit",
                    expected_commit,
                    "--source-tree-sha256",
                    source_digest,
                    "--",
                    "touch",
                    str(marker),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("config differs", result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
