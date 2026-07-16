#!/usr/bin/env python3
"""Bounded FIFO relay for one TPU host command.

The relay owns the FIFO reader and exits when the entrypoint creates the stop
marker. It drains bytes already queued at that point but does not wait for EOF,
because multiprocessing descendants can retain inherited stdout after the
recorded training parent has exited.
"""

from __future__ import annotations

import argparse
import errno
import os
import select
import stat
import sys
import time
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024
_MAX_READS_PER_POLL = 64


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _write_stream_best_effort(fd: int | None, payload: bytes) -> int | None:
    if fd is None:
        return None
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except BlockingIOError:
            # The immutable worker log remains complete even if the controller
            # stops consuming its best-effort live stream.
            return fd
        except (BrokenPipeError, OSError) as error:
            if isinstance(error, OSError) and error.errno not in {
                errno.EBADF,
                errno.EPIPE,
            }:
                raise
            os.close(fd)
            return None
        view = view[written:]
    return fd


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"host log is not a regular file: {path}")
    return fd


def _create_ready(path: Path) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
    finally:
        os.close(fd)


def _stop_requested(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise ValueError(f"host log relay stop marker is not a regular file: {path}")
    return True


def relay(*, pipe: Path, log: Path, stop: Path, ready: Path) -> None:
    mode = pipe.lstat().st_mode
    if not stat.S_ISFIFO(mode):
        raise ValueError(f"host log pipe is not a FIFO: {pipe}")
    if stop.exists() or stop.is_symlink() or ready.exists() or ready.is_symlink():
        raise FileExistsError("host log relay control path already exists")

    # O_RDWR keeps the FIFO readable before the entrypoint opens its writer and
    # avoids EOF busy loops. Shutdown is controlled solely by the stop marker.
    pipe_fd = os.open(pipe, os.O_RDWR | os.O_NONBLOCK)
    log_fd = _open_log(log)
    stream_fd: int | None = os.dup(sys.stdout.fileno())
    os.set_blocking(stream_fd, False)
    try:
        _create_ready(ready)
        while True:
            readable, _, _ = select.select([pipe_fd], [], [], 0.1)
            if readable:
                # Bound each drain pass so a continuously writing orphan
                # cannot starve the stop-marker check below.
                for _ in range(_MAX_READS_PER_POLL):
                    try:
                        payload = os.read(pipe_fd, _CHUNK_SIZE)
                    except BlockingIOError:
                        break
                    if not payload:
                        break
                    _write_all(log_fd, payload)
                    stream_fd = _write_stream_best_effort(stream_fd, payload)

            if _stop_requested(stop):
                # The direct child is already reaped before the stop marker is
                # published. Drain its queued bytes once, then exit even when a
                # descendant still holds or writes the inherited FIFO.
                drain_deadline = time.monotonic() + 1.0
                while time.monotonic() < drain_deadline:
                    try:
                        payload = os.read(pipe_fd, _CHUNK_SIZE)
                    except BlockingIOError:
                        break
                    if not payload:
                        break
                    _write_all(log_fd, payload)
                    stream_fd = _write_stream_best_effort(stream_fd, payload)
                return
    finally:
        os.close(pipe_fd)
        os.close(log_fd)
        if stream_fd is not None:
            os.close(stream_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    try:
        relay(pipe=args.pipe, log=args.log, stop=args.stop, ready=args.ready)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"host-log-relay: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
