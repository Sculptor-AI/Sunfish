import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sunfish_tpu.pid_publish_gate import publish_gate, wait_for_gate


class PidPublishGateTests(unittest.TestCase):
    def test_gate_is_private_atomic_and_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary) / "launch.gate"
            token = "a" * 64
            publish_gate(gate, token)
            self.assertEqual(gate.stat().st_mode & 0o777, 0o600)
            wait_for_gate(gate, token, timeout_seconds=1)
            self.assertFalse(gate.exists())

    def test_existing_gate_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary) / "launch.gate"
            gate.write_text("victim", encoding="ascii")
            with self.assertRaises(FileExistsError):
                publish_gate(gate, "b" * 64)
            self.assertEqual(gate.read_text(encoding="ascii"), "victim")

    def test_waiter_exec_preserves_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary) / "launch.gate"
            output = Path(temporary) / "pid"
            token = "c" * 64
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sunfish_tpu.pid_publish_gate",
                    "--gate",
                    str(gate),
                    "--token",
                    token,
                    "--",
                    sys.executable,
                    "-c",
                    "import os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))",
                    str(output),
                ],
                env={**os.environ, "PYTHONPATH": "src"},
            )
            try:
                publish_gate(gate, token)
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual(int(output.read_text(encoding="ascii")), process.pid)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
