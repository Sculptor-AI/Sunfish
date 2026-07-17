import importlib.util
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sunfish_test_host_log_relay", ROOT / "scripts/host_log_relay.py"
)
assert SPEC is not None and SPEC.loader is not None
host_log_relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host_log_relay)


class HostLogRelayTests(unittest.TestCase):
    def test_ready_path_appears_only_after_the_complete_pid_is_staged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "host.ready"
            expected = f"{os.getpid()}\n".encode("ascii")
            partial_written = threading.Event()
            allow_completion = threading.Event()
            write_all = host_log_relay._write_all

            def paused_write(fd: int, payload: bytes) -> None:
                self.assertEqual(payload, expected)
                write_all(fd, payload[:1])
                partial_written.set()
                if not allow_completion.wait(timeout=5):
                    raise TimeoutError("test did not release the staged ready write")
                write_all(fd, payload[1:])

            with mock.patch.object(
                host_log_relay, "_write_all", side_effect=paused_write
            ):
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(host_log_relay._create_ready, ready)
                    self.assertTrue(partial_written.wait(timeout=5))
                    try:
                        self.assertFalse(ready.exists())
                        staged = list(root.iterdir())
                        self.assertEqual(len(staged), 1)
                        self.assertEqual(staged[0].read_bytes(), expected[:1])
                    finally:
                        allow_completion.set()
                    future.result(timeout=5)

            self.assertEqual(ready.read_bytes(), expected)
            self.assertEqual(stat.S_IMODE(ready.stat().st_mode), 0o600)
            self.assertEqual(list(root.iterdir()), [ready])

    def test_existing_ready_path_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "host.ready"
            ready.write_text("tampered\n", encoding="ascii")
            with self.assertRaises(FileExistsError):
                host_log_relay._create_ready(ready)
            self.assertEqual(ready.read_text(encoding="ascii"), "tampered\n")
            self.assertEqual(list(root.iterdir()), [ready])


if __name__ == "__main__":
    unittest.main()
