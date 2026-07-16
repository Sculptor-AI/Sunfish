import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from sunfish_tpu.exact_process_interrupt import (
    _published_roots,
    _snapshot_processes,
    interrupt_attempt,
)


def _fake_process(
    proc_root: Path,
    pid: int,
    ppid: int,
    *,
    run_id: str,
    attempt_id: str,
    command: list[str],
    exact_environment: bool = True,
) -> None:
    process = proc_root / str(pid)
    (process / "fd").mkdir(parents=True)
    environment = ["PATH=/usr/bin"]
    if exact_environment:
        environment.extend(
            [f"SUNFISH_RUN_ID={run_id}", f"SUNFISH_ATTEMPT_ID={attempt_id}"]
        )
    (process / "environ").write_bytes(
        b"\0".join(value.encode() for value in environment) + b"\0"
    )
    (process / "cmdline").write_bytes(
        b"\0".join(os.fsencode(value) for value in command) + b"\0"
    )
    fields = ["S", str(ppid), *("0" for _ in range(17)), str(pid * 100)]
    (process / "stat").write_text(
        f"{pid} (fixture command) {' '.join(fields)}\n", encoding="ascii"
    )
    (process / "status").write_text(f"Name:\tfixture\nPPid:\t{ppid}\n")


class ExactProcessInterruptTests(unittest.TestCase):
    def test_snapshot_records_verified_descendants_before_any_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            pid_root = root / "pids"
            pid_root.mkdir()
            run_id = "run-001"
            attempt_id = "attempt-001"
            _fake_process(
                proc,
                1001,
                900,
                run_id=run_id,
                attempt_id=attempt_id,
                command=["/runtime/python", "/venv/bin/sunfish-train"],
            )
            _fake_process(
                proc,
                1002,
                1001,
                run_id=run_id,
                attempt_id=attempt_id,
                command=["/runtime/python", "-c", "multiprocessing.spawn"],
            )
            _fake_process(
                proc,
                1003,
                1002,
                run_id=run_id,
                attempt_id=attempt_id,
                command=["/runtime/python", "grain-worker"],
            )
            _fake_process(
                proc,
                2000,
                1,
                run_id="other",
                attempt_id="other",
                command=["sleep", "300"],
            )
            (pid_root / f"{run_id}.{attempt_id}.host.pid").write_text("1001\n")
            roots = _published_roots(pid_root, run_id, attempt_id)
            snapshot = _snapshot_processes(proc, roots, run_id, attempt_id)
            self.assertEqual([entry["pid"] for entry in snapshot], [1001, 1002, 1003])
            self.assertEqual(
                [entry["role"] for entry in snapshot],
                ["root", "descendant", "descendant"],
            )

    def test_unidentified_descendant_is_a_pre_signal_hard_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            run_id = "run-002"
            attempt_id = "attempt-002"
            _fake_process(
                proc,
                1101,
                900,
                run_id=run_id,
                attempt_id=attempt_id,
                command=["/venv/bin/sunfish-train"],
            )
            _fake_process(
                proc,
                1102,
                1101,
                run_id=run_id,
                attempt_id=attempt_id,
                command=["grain-worker"],
                exact_environment=False,
            )
            with self.assertRaisesRegex(ValueError, "no process was signaled"):
                _snapshot_processes(proc, [1101], run_id, attempt_id)

    @unittest.skipUnless(Path("/proc").is_dir(), "Linux /proc integration")
    def test_linux_helper_kills_the_exact_recorded_root_and_orphan_prone_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_pid_file = root / "child.pid"
            train = root / "sunfish-train"
            train.write_text(
                "#!/bin/sh\n"
                "sleep 300 &\n"
                "printf '%s\\n' \"$!\" > \"$CHILD_PID_FILE\"\n"
                "wait\n",
                encoding="utf-8",
            )
            train.chmod(0o755)
            run_id = "linux-run"
            attempt_id = "linux-attempt"
            environment = {
                **os.environ,
                "CHILD_PID_FILE": str(child_pid_file),
                "SUNFISH_RUN_ID": run_id,
                "SUNFISH_ATTEMPT_ID": attempt_id,
            }
            process = subprocess.Popen([str(train)], env=environment)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid_file.exists():
                    time.sleep(0.01)
                self.assertTrue(child_pid_file.exists())
                child_pid = int(child_pid_file.read_text())
                pid_root = root / "pids"
                pid_root.mkdir()
                (pid_root / f"{run_id}.{attempt_id}.host.pid").write_text(
                    f"{process.pid}\n"
                )
                result = interrupt_attempt(
                    run_id, attempt_id, pid_root=pid_root, wait_seconds=5
                )
                self.assertTrue(result["same_attempt_processes_absent"])
                self.assertEqual(result["exact_recorded_roots"], 1)
                self.assertGreaterEqual(result["exact_recorded_descendants"], 1)
                self.assertIn(process.pid, result["signaled_pids"])
                self.assertIn(child_pid, result["signaled_pids"])
            finally:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_posix_sigkill_of_a_parent_does_not_clean_up_its_child(self):
        if os.name != "posix":
            self.skipTest("POSIX process semantics required")
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_file = Path(temporary) / "child.pid"
            process = subprocess.Popen(
                [
                    "sh",
                    "-c",
                    'sleep 300 & printf "%s\\n" "$!" > "$1"; wait',
                    "sh",
                    str(child_pid_file),
                ]
            )
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid_file.exists():
                    time.sleep(0.01)
                self.assertTrue(child_pid_file.exists())
                child_pid = int(child_pid_file.read_text())
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
                # This is the load-bearing regression: SIGKILL is not
                # recursively delivered, so a direct-PID-only Gate 7 leaks the
                # Grain-like child and can leave /dev/accel0 occupied.
                os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
