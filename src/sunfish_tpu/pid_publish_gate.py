"""Exec a worker command only after its exact PID file is published.

The waiting process never forks.  Once the controller shell has exclusively
published that process's PID, it atomically publishes a private token file;
the waiter consumes the token and ``exec`` replaces it with the real workload.
Thus the recorded PID is the eventual ``sunfish-train`` PID, while a PID-file
publication race cannot leave workload descendants that were never recorded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
import sys
import time


_TOKEN = re.compile(r"^[0-9a-f]{64}$")


def _validate(gate: Path, token: str) -> None:
    if not gate.is_absolute():
        raise ValueError("PID publication gate path must be absolute")
    if not _TOKEN.fullmatch(token):
        raise ValueError("PID publication gate token must be 64 lowercase hex")


def publish_gate(gate: Path, token: str) -> None:
    """Atomically create a mode-0600 gate without following/replacing paths."""
    _validate(gate, token)
    temporary = gate.with_name(f"{gate.name}.tmp.{token}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        payload = f"{token}\n".encode("ascii")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, gate, follow_symlinks=False)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def wait_for_gate(gate: Path, token: str, *, timeout_seconds: float) -> None:
    """Wait for and consume the exact private gate, without spawning children."""
    _validate(gate, token)
    if timeout_seconds <= 0:
        raise ValueError("PID publication gate timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    while True:
        try:
            descriptor = os.open(gate, flags)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for exact PID publication")
            time.sleep(0.01)
            continue
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("PID publication gate is not a regular file")
            if opened.st_uid != os.getuid() or opened.st_mode & 0o077:
                raise ValueError("PID publication gate ownership/mode is unsafe")
            payload = os.read(descriptor, 66)
            if payload != f"{token}\n".encode("ascii"):
                raise ValueError("PID publication gate token differs")
        finally:
            os.close(descriptor)
        current = gate.lstat()
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("PID publication gate changed while being consumed")
        gate.unlink()
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        if args.publish:
            if command:
                raise ValueError("publisher does not accept a command")
            publish_gate(args.gate, args.token)
            return 0
        if not command:
            raise ValueError("missing command after --")
        wait_for_gate(args.gate, args.token, timeout_seconds=args.timeout_seconds)
        os.execvpe(command[0], command, os.environ)
    except (FileExistsError, FileNotFoundError, OSError, TimeoutError, ValueError) as error:
        print(f"sunfish-pid-publish-gate: {error}", file=sys.stderr)
        return 125
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
