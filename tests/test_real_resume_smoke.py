import copy
import os
import signal
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sunfish_tpu.real_resume_smoke import (
    _broadcast_process0_error,
    _payload_sha256,
    _phase_command,
    _run_child_process,
    _run_orchestrated_restart,
    run_real_resume_phase,
    verify_real_resume_evidence,
    verify_real_resume_prepare_evidence,
)
from sunfish_tpu.training.spec import Phase


def host(process, prepare_summary=None):
    if prepare_summary is None:
        prepare_summary = make_prepare_summary()
    prepare = prepare_summary["hosts"][process]
    payload = {
        "schema_version": 2,
        "gate": 6,
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": "resume-real-1",
        "config_sha256": "a" * 64,
        "config_file_sha256": "9" * 64,
        "checkpoint_step": 1,
        "process_index": process,
        "process_count": 2,
        "global_device_count": 8,
        "local_device_count": 4,
        "attempt_id": "resume-proof",
        "launcher_attempt_id": prepare["launcher_attempt_id"],
        "dataset_manifest_sha256": "b" * 64,
        "seed_manifest_sha256": "c" * 64,
        "prepare_summary_sha256": _payload_sha256(prepare_summary),
        "prepare_launcher_attempt_id": prepare["launcher_attempt_id"],
        "restart_mode": "separate-python-processes",
        "prepare_process_token": prepare["process_token"],
        "resume_process_token": f"{process + 11:064x}",
        "prepare_process_pid": prepare["process_pid"],
        "resume_process_pid": 200 + process,
        "topology": {"ready": True},
        "next_batch_exact": True,
        "next_loss_exact": True,
        "next_trainable_gradients_exact": True,
        "next_trainable_updates_exact": True,
        "next_trainable_params_exact": True,
        "next_optimizer_state_exact": True,
        "next_collections_exact": True,
        "next_step_exact": True,
        "control_frozen_params_unchanged": True,
        "resumed_frozen_params_unchanged": True,
        "passed": True,
        "digests": {},
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
    }
    for name, digest in prepare["control_digests"].items():
        payload["digests"][name] = {"control": digest, "resumed": digest}
    return payload


def prepare_host(process):
    payload = {
        "schema_version": 2,
        "gate": 6,
        "phase": "prepare",
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": "resume-real-1",
        "proof_id": "resume-proof",
        "launcher_attempt_id": "resume-launch",
        "config_sha256": "a" * 64,
        "config_file_sha256": "9" * 64,
        "checkpoint_step": 1,
        "process_index": process,
        "process_count": 2,
        "global_device_count": 8,
        "local_device_count": 4,
        "dataset_manifest_sha256": "b" * 64,
        "seed_manifest_sha256": "c" * 64,
        "topology": {"ready": True},
        "process_token": f"{process + 1:064x}",
        "process_pid": 100 + process,
        "control_frozen_params_unchanged": True,
        "control_digests": {},
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
        "passed": True,
    }
    for index, name in enumerate(
        (
            "batch",
            "loss",
            "gradients",
            "updates",
            "params",
            "opt_state",
            "collections",
            "step",
        ),
        start=20,
    ):
        payload["control_digests"][name] = f"{index:064x}"
    return payload


def make_prepare_summary():
    return verify_real_resume_prepare_evidence(
        [prepare_host(0), prepare_host(1)],
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
    )


class RealResumeEvidenceTests(unittest.TestCase):
    def test_orchestrator_runs_prepare_then_resume_as_module_processes(self):
        args = Namespace(
            config=Path("config.toml"),
            attempt_id="proof",
            evidence_dir="gs://bucket/evidence",
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        prepare = _phase_command(args, "prepare")
        resume = _phase_command(args, "resume")
        self.assertEqual(
            prepare[:3],
            [__import__("sys").executable, "-m", "sunfish_tpu.real_resume_smoke"],
        )
        self.assertEqual(prepare[-2:], ["--process-phase", "prepare"])
        self.assertEqual(resume[-2:], ["--process-phase", "resume"])
        with mock.patch(
            "sunfish_tpu.real_resume_smoke._run_child_process",
            side_effect=[0, 0],
        ) as run:
            self.assertEqual(_run_orchestrated_restart(args), 0)
        self.assertEqual(
            [call.kwargs["phase"] for call in run.call_args_list],
            ["prepare", "resume"],
        )

    def test_orchestrator_never_starts_resume_after_prepare_failure(self):
        args = Namespace(
            config=Path("config.toml"),
            attempt_id="proof",
            evidence_dir="gs://bucket/evidence",
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        with mock.patch(
            "sunfish_tpu.real_resume_smoke._run_child_process",
            return_value=2,
        ) as run:
            self.assertEqual(_run_orchestrated_restart(args), 2)
        run.assert_called_once()

    def test_signal_before_popen_returns_is_latched_forwarded_and_fails(self):
        handlers = {}

        class FakeChild:
            def __init__(self):
                self.pid = 7001
                self.running = True

            def poll(self):
                return None if self.running else 0

            def wait(self, timeout=None):
                self.running = False
                return 0

        child = FakeChild()

        def install_handler(signum, handler):
            if callable(handler):
                handlers[signum] = handler
            else:
                handlers.pop(signum, None)

        def spawn_after_signal(*_args, **_kwargs):
            self.assertIn(signal.SIGTERM, handlers)
            self.assertTrue(_kwargs.get("start_new_session"))
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return child

        group_signals = []

        def signal_group(process_group, signum):
            self.assertEqual(process_group, child.pid)
            if signum == 0 and not child.running:
                raise ProcessLookupError
            if signum:
                group_signals.append(signum)

        with (
            mock.patch(
                "sunfish_tpu.real_resume_smoke.signal.getsignal",
                return_value="previous",
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.signal.signal",
                side_effect=install_handler,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.subprocess.Popen",
                side_effect=spawn_after_signal,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.os.killpg",
                side_effect=signal_group,
            ),
        ):
            returncode = _run_child_process(["python", "child.py"], phase="prepare")
        self.assertEqual(returncode, 143)
        self.assertEqual(group_signals, [signal.SIGTERM])

    def test_signal_latched_before_spawn_does_not_create_a_phase_process(self):
        handlers = {}
        delivered = False

        def install_handler(signum, handler):
            nonlocal delivered
            if callable(handler):
                handlers[signum] = handler
                if signum == signal.SIGTERM and not delivered:
                    delivered = True
                    handler(signal.SIGTERM, None)
            else:
                handlers.pop(signum, None)

        with (
            mock.patch(
                "sunfish_tpu.real_resume_smoke.signal.getsignal",
                return_value="previous",
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.signal.signal",
                side_effect=install_handler,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.subprocess.Popen",
            ) as popen,
        ):
            returncode = _run_child_process(
                ["python", "child.py"], phase="prepare"
            )
        self.assertEqual(returncode, 143)
        popen.assert_not_called()

    def test_latched_signal_bounds_wait_then_terminates_and_kills_child(self):
        handlers = {}

        class IgnoringChild:
            def __init__(self):
                self.pid = 7002
                self.running = True
                self.killed = False

            def poll(self):
                return None if self.running else -int(signal.SIGKILL)

            def wait(self, timeout=None):
                if self.killed:
                    self.running = False
                    return -int(signal.SIGKILL)
                raise __import__("subprocess").TimeoutExpired("child", timeout)

        child = IgnoringChild()

        def install_handler(signum, handler):
            if callable(handler):
                handlers[signum] = handler
            else:
                handlers.pop(signum, None)

        def spawn_after_signal(*_args, **_kwargs):
            self.assertTrue(_kwargs.get("start_new_session"))
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return child

        group_signals = []

        def signal_group(process_group, signum):
            self.assertEqual(process_group, child.pid)
            if signum == 0:
                if not child.running:
                    raise ProcessLookupError
                return
            group_signals.append(signum)
            if signum == signal.SIGKILL:
                child.killed = True

        with (
            mock.patch(
                "sunfish_tpu.real_resume_smoke.signal.getsignal",
                return_value="previous",
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.signal.signal",
                side_effect=install_handler,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.subprocess.Popen",
                side_effect=spawn_after_signal,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.os.killpg",
                side_effect=signal_group,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke._SIGNAL_GRACE_SECONDS", 0.0
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke._TERMINATE_GRACE_SECONDS", 0.0
            ),
        ):
            returncode = _run_child_process(["python", "child.py"], phase="prepare")
        self.assertEqual(returncode, 143)
        self.assertTrue(child.killed)
        self.assertEqual(
            group_signals,
            [signal.SIGTERM, signal.SIGTERM, signal.SIGKILL],
        )

    def test_stuck_descendant_group_overrides_direct_child_success(self):
        class ExitedChild:
            pid = 7003

            @staticmethod
            def poll():
                return 0

            @staticmethod
            def wait(timeout=None):
                del timeout
                return 0

        signals = []

        def group_never_disappears(process_group, signum):
            self.assertEqual(process_group, ExitedChild.pid)
            if signum:
                signals.append(signum)

        with (
            mock.patch(
                "sunfish_tpu.real_resume_smoke.subprocess.Popen",
                return_value=ExitedChild(),
            ) as popen,
            mock.patch(
                "sunfish_tpu.real_resume_smoke.os.killpg",
                side_effect=group_never_disappears,
            ),
            mock.patch(
                "sunfish_tpu.real_resume_smoke._TERMINATE_GRACE_SECONDS", 0.0
            ),
        ):
            returncode = _run_child_process(
                ["python", "child.py"], phase="prepare"
            )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(returncode, 137)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "killpg"),
        "process-group regression requires POSIX sessions",
    )
    def test_real_descendant_is_killed_with_its_phase_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            descendant_pid_path = Path(directory) / "descendant.pid"
            descendant_ready_path = Path(directory) / "descendant.ready"
            descendant = (
                "import pathlib,signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"pathlib.Path({str(descendant_ready_path)!r}).write_text('ready'); "
                "time.sleep(60)"
            )
            phase = (
                "import os,pathlib,signal,subprocess,sys,time\n"
                f"child=subprocess.Popen([{sys.executable!r},'-c',{descendant!r}])\n"
                f"ready=pathlib.Path({str(descendant_ready_path)!r})\n"
                "while not ready.exists():\n"
                "    time.sleep(0.01)\n"
                f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid))\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "os.kill(os.getppid(), signal.SIGTERM)\n"
                "time.sleep(60)\n"
            )
            descendant_pid = None
            try:
                with (
                    mock.patch(
                        "sunfish_tpu.real_resume_smoke._SIGNAL_GRACE_SECONDS", 0.1
                    ),
                    mock.patch(
                        "sunfish_tpu.real_resume_smoke._TERMINATE_GRACE_SECONDS",
                        0.2,
                    ),
                    mock.patch(
                        "sunfish_tpu.real_resume_smoke._WAIT_POLL_SECONDS", 0.02
                    ),
                ):
                    returncode = _run_child_process(
                        [sys.executable, "-c", phase], phase="prepare"
                    )
                self.assertEqual(returncode, 143)
                descendant_pid = int(descendant_pid_path.read_text())
                deadline = time.monotonic() + 2.0
                while (
                    self._process_is_live(descendant_pid)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertFalse(
                    self._process_is_live(descendant_pid),
                    "phase descendant survived owned process-group shutdown",
                )
            finally:
                if descendant_pid is not None and self._process_is_live(
                    descendant_pid
                ):
                    try:
                        os.kill(descendant_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "killpg"),
        "process-group regression requires POSIX sessions",
    )
    def test_lingering_descendant_is_removed_after_phase_root_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            descendant_pid_path = Path(directory) / "descendant.pid"
            descendant_ready_path = Path(directory) / "descendant.ready"
            descendant = (
                "import pathlib,signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"pathlib.Path({str(descendant_ready_path)!r}).write_text('ready'); "
                "time.sleep(60)"
            )
            phase = (
                "import pathlib,subprocess,sys,time\n"
                f"child=subprocess.Popen([{sys.executable!r},'-c',{descendant!r}])\n"
                f"ready=pathlib.Path({str(descendant_ready_path)!r})\n"
                "while not ready.exists():\n"
                "    time.sleep(0.01)\n"
                f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid))\n"
            )
            descendant_pid = None
            try:
                with (
                    mock.patch(
                        "sunfish_tpu.real_resume_smoke._TERMINATE_GRACE_SECONDS",
                        0.2,
                    ),
                    mock.patch(
                        "sunfish_tpu.real_resume_smoke._WAIT_POLL_SECONDS", 0.02
                    ),
                ):
                    returncode = _run_child_process(
                        [sys.executable, "-c", phase], phase="prepare"
                    )
                self.assertEqual(returncode, 0)
                descendant_pid = int(descendant_pid_path.read_text())
                deadline = time.monotonic() + 2.0
                while (
                    self._process_is_live(descendant_pid)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertFalse(
                    self._process_is_live(descendant_pid),
                    "phase descendant survived after its direct parent exited",
                )
            finally:
                if descendant_pid is not None and self._process_is_live(
                    descendant_pid
                ):
                    try:
                        os.kill(descendant_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @staticmethod
    def _process_is_live(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.exists():
            try:
                # A killed orphan can remain as a non-executing zombie until
                # the container's PID 1 reaps it. That is not a surviving
                # workload process and cannot hold TPU resources.
                return proc_stat.read_text().split()[2] != "Z"
            except (IndexError, OSError):
                return False
        return True

    def test_process0_error_broadcast_reaches_every_host(self):
        class Buffer(bytearray):
            def tolist(self):
                return list(self)

        class FakeNumpy:
            uint8 = object()

            @staticmethod
            def zeros(shape, _dtype):
                return Buffer(shape[0])

            @staticmethod
            def frombuffer(value, _dtype):
                return value

            @staticmethod
            def asarray(value):
                return Buffer(value)

        class BroadcastBus:
            payload = None

            def broadcast_one_to_all(self, value):
                if any(value):
                    self.payload = Buffer(value)
                self.assert_payload()
                return Buffer(self.payload)

            def assert_payload(self):
                if self.payload is None:
                    raise AssertionError("process 0 did not publish a payload")

        bus = BroadcastBus()
        process0 = _broadcast_process0_error(
            bus, FakeNumpy, "FileExistsError: divergent summary", limit=128
        )
        peer = _broadcast_process0_error(bus, FakeNumpy, None, limit=128)
        self.assertEqual(process0, "FileExistsError: divergent summary")
        self.assertEqual(peer, process0)

    def test_local_evidence_path_fails_before_distributed_jax(self):
        config = SimpleNamespace(
            run=SimpleNamespace(phase=Phase.SMOKE, run_id="resume-real-1"),
            topology=SimpleNamespace(
                expected_devices=8,
                expected_processes=2,
                expected_local_devices=4,
            ),
        )
        environment = {
            "SUNFISH_REAL_RESUME_ORCHESTRATED": "1",
            "SUNFISH_REAL_RESUME_PROCESS_TOKEN": "1" * 64,
            "SUNFISH_ATTEMPT_ID": "resume-launch",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.HarnessConfig.load",
                return_value=config,
            ),
            mock.patch("sunfish_tpu.real_resume_smoke.require_launcher_run_id"),
            mock.patch(
                "sunfish_tpu.real_resume_smoke.initialize_distributed_jax"
            ) as initialize,
        ):
            with self.assertRaisesRegex(ValueError, "GCS workdir"):
                run_real_resume_phase(
                    config_path=Path("config.toml"),
                    attempt_id="resume-proof",
                    evidence_dir="/tmp/not-shared",
                    expected_devices=8,
                    expected_processes=2,
                    expected_local_devices=4,
                    process_phase="prepare",
                )
        initialize.assert_not_called()

    def test_prepare_requires_all_hosts_and_valid_control_digests(self):
        result = verify_real_resume_prepare_evidence(
            [prepare_host(0), prepare_host(1)],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(result["passed"], result["errors"])

        broken = copy.deepcopy(prepare_host(1))
        broken["control_digests"]["loss"] = "not-a-digest"
        result = verify_real_resume_prepare_evidence(
            [prepare_host(0), broken],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("loss" in error for error in result["errors"]))

    def test_all_exact_host_comparisons_pass(self):
        prepare = make_prepare_summary()
        result = verify_real_resume_evidence(
            [host(0, prepare), host(1, prepare)],
            prepare_summary=prepare,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(result["passed"], result["errors"])

    def test_any_boolean_or_digest_mismatch_fails(self):
        prepare = make_prepare_summary()
        broken = copy.deepcopy(host(1, prepare))
        broken["next_loss_exact"] = False
        broken["digests"]["gradients"]["resumed"] = "different"
        result = verify_real_resume_evidence(
            [host(0, prepare), broken],
            prepare_summary=prepare,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("next_loss_exact" in error for error in result["errors"]))
        self.assertTrue(any("gradients" in error for error in result["errors"]))

    def test_reusing_one_process_token_fails_restart_proof(self):
        prepare = make_prepare_summary()
        broken = copy.deepcopy(host(1, prepare))
        broken["resume_process_token"] = broken["prepare_process_token"]
        result = verify_real_resume_evidence(
            [host(0, prepare), broken],
            prepare_summary=prepare,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("process tokens" in error for error in result["errors"]))

    def test_equal_missing_digest_values_cannot_pass(self):
        prepare = make_prepare_summary()
        broken = copy.deepcopy(host(1, prepare))
        broken["digests"]["loss"] = {"control": None, "resumed": None}
        result = verify_real_resume_evidence(
            [host(0, prepare), broken],
            prepare_summary=prepare,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("loss" in error for error in result["errors"]))

    def test_embedded_prepare_summary_is_recomputed_and_hash_bound(self):
        prepare = make_prepare_summary()
        hosts = [host(0, prepare), host(1, prepare)]
        tampered = copy.deepcopy(prepare)
        tampered["passed"] = False
        result = verify_real_resume_evidence(
            hosts,
            prepare_summary=tampered,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("embedded prepare" in error for error in result["errors"]))
        self.assertTrue(any("prepare-summary" in error for error in result["errors"]))

    def test_launcher_attempt_must_match_across_processes(self):
        prepare = make_prepare_summary()
        hosts = [host(0, prepare), host(1, prepare)]
        for item in hosts:
            item["launcher_attempt_id"] = "different-launch"
        result = verify_real_resume_evidence(
            hosts,
            prepare_summary=prepare,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("launcher" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
