#!/usr/bin/env python3
"""Run attached all-worker SSH with exact remote cleanup on abnormal exit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

_TERM_SECONDS = 5.0
_KILL_SECONDS = 5.0
_CLEANUP_TIMEOUT_SECONDS = 120.0
_CLEANUP_HARD_STOP = 126
_SIGNAL_POLL_SECONDS = 0.25


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group_exit(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_exists(process_group):
            return True
        time.sleep(0.05)
    return not _group_exists(process_group)


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    process_group = process.pid
    if process_group <= 1 or process_group == os.getpgrp():
        raise RuntimeError(f"refusing unsafe controller process group {process_group}")
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_TERM_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    if _wait_group_exit(process_group, _TERM_SECONDS):
        return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_KILL_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    if not _wait_group_exit(process_group, _KILL_SECONDS):
        raise RuntimeError(
            f"controller process group {process_group} survived SIGKILL"
        )


def _run_cleanup(command: list[str]) -> int:
    print(
        "attached TPU launch failed; interrupting the exact remote run/attempt",
        file=sys.stderr,
        flush=True,
    )
    cleanup = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = cleanup.communicate(timeout=_CLEANUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_group(cleanup)
        output = b""
        returncode = 124
    else:
        returncode = cleanup.returncode
        if _group_exists(cleanup.pid):
            _terminate_group(cleanup)
            if returncode == 0:
                returncode = 125
    if output:
        sys.stderr.buffer.write(output)
        sys.stderr.buffer.flush()
    if returncode != 0:
        print(
            "exact remote cleanup did not pass; do not relaunch this attempt "
            f"until cleanup is confirmed (status {returncode})",
            file=sys.stderr,
            flush=True,
        )
    return returncode


def _tee_output(
    source, destination, errors: list[BaseException]
) -> None:
    try:
        while block := source.read(64 * 1024):
            destination.write(block)
            destination.flush()
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()
    except BaseException as error:
        errors.append(error)


def _cleanup_after_abnormal_exit(
    child: subprocess.Popen[bytes],
    *,
    tee: threading.Thread | None,
    tee_started: bool,
    tee_errors: list[BaseException],
    cleanup_command: list[str],
) -> int:
    """Run remote cleanup first, then prove the local process group is gone."""
    try:
        remote_status = _run_cleanup(cleanup_command)
    except BaseException as error:
        print(f"exact remote cleanup raised: {error}", file=sys.stderr, flush=True)
        remote_status = 125

    local_status = 0
    try:
        _terminate_group(child)
    except BaseException as error:
        print(f"local launcher cleanup failed: {error}", file=sys.stderr, flush=True)
        local_status = 125

    if tee is not None and tee_started:
        try:
            tee.join(timeout=_TERM_SECONDS + _KILL_SECONDS)
        except BaseException as error:
            print(f"controller log relay join failed: {error}", file=sys.stderr)
            local_status = 125
        else:
            if tee.is_alive():
                print("controller log relay did not stop", file=sys.stderr)
                local_status = 125
    if tee_errors:
        print(f"controller log relay failed: {tee_errors[0]}", file=sys.stderr)
        local_status = 125

    if remote_status != 0 or local_status != 0:
        print(
            "remote/local process cleanup is unproven; returning the "
            "non-retryable owner-intervention status 126",
            file=sys.stderr,
        )
        return _CLEANUP_HARD_STOP
    return 0


def run_attached(
    command: list[str], *, log_path: Path, cleanup_command: list[str]
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        child: subprocess.Popen[bytes] | None = None
        tee: threading.Thread | None = None
        tee_started = False
        tee_errors: list[BaseException] = []
        previous_handlers: dict[int, signal.Handlers] = {}
        signal_state: dict[str, int | None] = {"pending": None}
        cleanup_performed = False

        def record_signal(signum, _frame):
            if signal_state["pending"] is None:
                signal_state["pending"] = signum

        try:
            # Install non-raising handlers before Popen. A signal delivered
            # during Popen is recorded and handled immediately after the child
            # object is safely assigned, eliminating the unowned-child window.
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, record_signal)

            pending = signal_state["pending"]
            if isinstance(pending, int):
                return 128 + pending

            child = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            assert child.stdout is not None
            tee = threading.Thread(
                target=_tee_output,
                args=(child.stdout, log, tee_errors),
                daemon=True,
                name="sunfish-controller-log-relay",
            )
            tee.start()
            tee_started = True

            interrupted = None
            while True:
                pending = signal_state["pending"]
                if isinstance(pending, int):
                    interrupted = pending
                    returncode = 128 + pending
                    break
                try:
                    returncode = child.wait(timeout=_SIGNAL_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    continue
                break

            abnormal = (
                returncode != 0
                or bool(tee_errors)
                or _group_exists(child.pid)
            )
            if abnormal:
                cleanup_performed = True
                cleanup_status = _cleanup_after_abnormal_exit(
                    child,
                    tee=tee,
                    tee_started=tee_started,
                    tee_errors=tee_errors,
                    cleanup_command=cleanup_command,
                )
                if cleanup_status != 0:
                    return cleanup_status
                if interrupted is not None:
                    return 128 + interrupted
                return returncode or cleanup_status or 125

            tee.join(timeout=_TERM_SECONDS)
            if tee.is_alive() or tee_errors:
                cleanup_performed = True
                cleanup_status = _cleanup_after_abnormal_exit(
                    child,
                    tee=tee,
                    tee_started=tee_started,
                    tee_errors=tee_errors,
                    cleanup_command=cleanup_command,
                )
                return cleanup_status or 125
            return 0
        except BaseException:
            if child is not None and not cleanup_performed:
                cleanup_performed = True
                cleanup_status = _cleanup_after_abnormal_exit(
                    child,
                    tee=tee,
                    tee_started=tee_started,
                    tee_errors=tee_errors,
                    cleanup_command=cleanup_command,
                )
                if cleanup_status != 0:
                    return cleanup_status
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cleanup-script", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("missing attached command after --")
    cleanup = [
        str(args.cleanup_script),
        "--run-id",
        args.run_id,
        "--attempt-id",
        args.attempt_id,
    ]
    try:
        return run_attached(command, log_path=args.log, cleanup_command=cleanup)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"controller-attached-launch: {error}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
