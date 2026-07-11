import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class AllHostLauncherTests(unittest.TestCase):
    def test_gcloud_targets_all_workers_with_one_run_identity(self):
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
                "SUNFISH_CONTROLLER_LOG_DIR": str(temporary_path / "logs"),
                "TPU_NAME": "sunfish-v4",
                "PROJECT_ID": "sunfish-project",
                "ZONE": "us-central2-b",
                "REMOTE_REPO_DIR": "/home/sunfish/sunfish-v2",
                "EXPECTED_TPU_DEVICES": "64",
                "EXPECTED_TPU_PROCESSES": "8",
            }
            subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "launch_tpu_pod.sh"),
                    "--run-id",
                    "20260711-smoke",
                    "--config",
                    "configs/sunfish-8b-a3b.toml",
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
            self.assertIn("configs/sunfish-8b-a3b.toml", command)
            self.assertIn("--expected-devices 64", command)
            self.assertIn("--expected-processes 8", command)

    def test_host_entrypoint_writes_a_per_host_log(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            log_root = Path(temporary) / "host-logs"
            environment = {
                **os.environ,
                "SUNFISH_HOST_LOG_ROOT": str(log_root),
                "SUNFISH_PYTHON_BIN": str(Path(temporary) / "missing-python"),
            }
            subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "tpu_host_entrypoint.sh"),
                    "--run-id",
                    "20260711-host-log",
                    "--config",
                    "configs/sunfish-8b-a3b.toml",
                    "--expected-devices",
                    "64",
                    "--expected-processes",
                    "8",
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
            logs = list((log_root / "20260711-host-log").glob("*.log"))
            self.assertEqual(len(logs), 1)
            content = logs[0].read_text(encoding="utf-8")
            self.assertIn("run_id=20260711-host-log", content)
            self.assertIn("host-command-ran", content)


if __name__ == "__main__":
    unittest.main()
