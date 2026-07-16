import os
from pathlib import Path
import tempfile
import unittest

from sunfish_tpu.worker_hygiene import inspect_worker_hygiene


def _fake_process(
    proc_root: Path,
    pid: int,
    *,
    run_id: str,
    attempt_id: str,
    descriptors: list[str],
) -> None:
    process = proc_root / str(pid)
    fd_root = process / "fd"
    fd_root.mkdir(parents=True)
    (process / "environ").write_bytes(
        f"SUNFISH_RUN_ID={run_id}\0SUNFISH_ATTEMPT_ID={attempt_id}\0".encode()
    )
    (process / "cmdline").write_bytes(b"python\0worker\0")
    for number, target in enumerate(descriptors):
        (fd_root / str(number)).symlink_to(target)


class WorkerHygieneTests(unittest.TestCase):
    def test_clean_worker_passes_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            report = inspect_worker_hygiene(
                "run-clean",
                "attempt-clean",
                proc_root=proc,
                lockfile=root / "libtpu_lockfile",
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["read_only"])

    def test_accelerator_holder_is_a_read_only_owner_intervention_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            _fake_process(
                proc,
                1201,
                run_id="old-run",
                attempt_id="old-attempt",
                descriptors=["/dev/accel0"],
            )
            report = inspect_worker_hygiene(
                "new-run",
                "new-attempt",
                proc_root=proc,
                lockfile=root / "libtpu_lockfile",
            )
            self.assertFalse(report["passed"])
            self.assertTrue(report["owner_intervention_required"])
            self.assertEqual(report["accelerator_holders"][0]["pid"], 1201)
            self.assertTrue((proc / "1201").exists())

    def test_stale_libtpu_lockfile_is_never_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            lockfile = root / "libtpu_lockfile"
            lockfile.write_text("stale\n")
            report = inspect_worker_hygiene(
                "new-run",
                "new-attempt",
                proc_root=proc,
                lockfile=lockfile,
            )
            self.assertFalse(report["passed"])
            self.assertTrue(report["lockfile_exists"])
            self.assertTrue(lockfile.exists())
            self.assertEqual(lockfile.read_text(), "stale\n")

    def test_lockfile_with_exact_current_attempt_fd_owner_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            proc.mkdir()
            lockfile = root / "libtpu_lockfile"
            lockfile.write_text("active\n")
            _fake_process(
                proc,
                1301,
                run_id="current-run",
                attempt_id="current-attempt",
                descriptors=[str(lockfile)],
            )
            report = inspect_worker_hygiene(
                "current-run",
                "current-attempt",
                proc_root=proc,
                lockfile=lockfile,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["lockfile_verified_current_attempt_owner"])
            self.assertTrue(lockfile.exists())


if __name__ == "__main__":
    unittest.main()
