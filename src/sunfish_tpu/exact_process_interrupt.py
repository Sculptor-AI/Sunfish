"""Interrupt only pre-recorded, exact run/attempt worker processes on Linux.

The module is dependency-free and CPython-3.10-compatible because the
controller embeds it into the all-worker IAP command.  It never signals a
process group.  Before the first signal it snapshots every current-user
descendant of the published ``sunfish-train`` roots, verifies the exact
run/attempt environment, and records PID plus Linux start time.  Linux pidfds
then bind each SIGKILL to that exact process even if a numeric PID is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any


SNAPSHOT_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PID = re.compile(r"^[1-9][0-9]*$")
_WAIT_SECONDS = 30.0


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as source:
        return source.read()


def _environment_matches(raw: bytes, run_id: str, attempt_id: str) -> bool:
    values = set(raw.split(b"\0"))
    return (
        f"SUNFISH_RUN_ID={run_id}".encode() in values
        and f"SUNFISH_ATTEMPT_ID={attempt_id}".encode() in values
    )


def _command_arguments(raw: bytes) -> list[str]:
    return [
        os.fsdecode(value)
        for value in raw.rstrip(b"\0").split(b"\0")
        if value
    ]


def _proc_start_time(stat_bytes: bytes) -> int:
    # /proc/PID/stat field 2 is parenthesized and may contain spaces or ')'.
    closing = stat_bytes.rfind(b")")
    if closing < 0:
        raise ValueError("process stat has no command terminator")
    fields = stat_bytes[closing + 1 :].split()
    # fields[0] is field 3 (state); starttime is Linux proc field 22.
    if len(fields) <= 19:
        raise ValueError("process stat is missing start time")
    return int(fields[19])


def _proc_parent(status_bytes: bytes) -> int:
    for line in status_bytes.splitlines():
        if line.startswith(b"PPid:"):
            return int(line.split(b":", 1)[1].strip())
    raise ValueError("process status is missing PPid")


def _read_process(proc_root: Path, pid: int, uid: int) -> dict[str, Any] | None:
    directory = proc_root / str(pid)
    try:
        info = directory.stat(follow_symlinks=False)
        if info.st_uid != uid or not directory.is_dir() or directory.is_symlink():
            return None
        environ = _read_bytes(directory / "environ")
        cmdline = _read_bytes(directory / "cmdline")
        start_time = _proc_start_time(_read_bytes(directory / "stat"))
        parent = _proc_parent(_read_bytes(directory / "status"))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return {
        "pid": pid,
        "ppid": parent,
        "start_time_ticks": start_time,
        "environ": environ,
        "cmdline": cmdline,
    }


def _published_roots(pid_root: Path, run_id: str, attempt_id: str) -> list[int]:
    roots: list[int] = []
    for path in sorted(pid_root.glob(f"{run_id}.{attempt_id}.*.pid")):
        if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid():
            continue
        raw = path.read_text(encoding="ascii").strip()
        if _PID.fullmatch(raw) is None or "\n" in raw:
            raise ValueError(f"published attempt PID file is invalid: {path}")
        pid = int(raw)
        if pid <= 1 or pid in roots:
            raise ValueError(f"published attempt PID is invalid or duplicate: {pid}")
        roots.append(pid)
    if not roots:
        raise ValueError("no exact published training PID was found")
    return roots


def _snapshot_processes(
    proc_root: Path, roots: Sequence[int], run_id: str, attempt_id: str
) -> list[dict[str, Any]]:
    uid = os.getuid()
    observed: dict[int, dict[str, Any]] = {}
    for path in proc_root.iterdir():
        if _PID.fullmatch(path.name) is None:
            continue
        process = _read_process(proc_root, int(path.name), uid)
        if process is not None:
            observed[process["pid"]] = process

    for pid in roots:
        process = observed.get(pid)
        if process is None:
            raise ValueError(f"published training PID is not a current-user process: {pid}")
        arguments = _command_arguments(process["cmdline"])
        if not any(Path(argument).name == "sunfish-train" for argument in arguments):
            raise ValueError(f"published PID is not an exact sunfish-train root: {pid}")
        if not _environment_matches(process["environ"], run_id, attempt_id):
            raise ValueError(f"published training PID has another run/attempt: {pid}")

    descendants: set[int] = set()
    frontier = set(roots)
    while frontier:
        children = {
            pid
            for pid, process in observed.items()
            if process["ppid"] in frontier and pid not in descendants and pid not in roots
        }
        descendants.update(children)
        frontier = children

    # Every descendant must carry the exact exec-time environment.  If not,
    # stop before signaling anything: the allocation owner did not authorize
    # a broad ancestry- or process-group-based kill.
    for pid in sorted(descendants):
        if not _environment_matches(observed[pid]["environ"], run_id, attempt_id):
            raise ValueError(
                "current-user training descendant lacks the exact run/attempt "
                f"identity; no process was signaled: {pid}"
            )

    targets: list[dict[str, Any]] = []
    root_set = set(roots)
    for pid in [*roots, *sorted(descendants)]:
        process = observed[pid]
        targets.append(
            {
                "pid": pid,
                "ppid": process["ppid"],
                "start_time_ticks": process["start_time_ticks"],
                "role": "root" if pid in root_set else "descendant",
                "cmdline_sha256": hashlib.sha256(process["cmdline"]).hexdigest(),
            }
        )
    return targets


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _publish_snapshot(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    encoded = _canonical_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("exact process snapshot is not a trusted regular file")
    observed_bytes = _read_bytes(path)
    try:
        observed = json.loads(observed_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("exact process snapshot is invalid") from error
    if observed != payload:
        raise ValueError("existing exact process snapshot differs from this attempt")
    return observed, hashlib.sha256(observed_bytes).hexdigest()


def _load_snapshot(path: Path, run_id: str, attempt_id: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("exact process snapshot is not a trusted regular file")
    encoded = _read_bytes(path)
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("exact process snapshot is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or payload.get("run_id") != run_id
        or payload.get("attempt_id") != attempt_id
        or payload.get("uid") != os.getuid()
        or not isinstance(payload.get("targets"), list)
        or not payload["targets"]
    ):
        raise ValueError("exact process snapshot belongs to another attempt")
    return payload, hashlib.sha256(encoded).hexdigest()


def _live_record(
    proc_root: Path, record: Mapping[str, Any], run_id: str, attempt_id: str
) -> dict[str, Any] | None:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        raise ValueError("exact process snapshot has an invalid PID")
    process = _read_process(proc_root, pid, os.getuid())
    if process is None:
        return None
    if (
        process["start_time_ticks"] != record.get("start_time_ticks")
        or not _environment_matches(process["environ"], run_id, attempt_id)
        or hashlib.sha256(process["cmdline"]).hexdigest()
        != record.get("cmdline_sha256")
    ):
        # The numeric PID has exited or been reused.  It is no longer the exact
        # recorded target and must never be signaled.
        return None
    if record.get("role") == "root":
        arguments = _command_arguments(process["cmdline"])
        if not any(Path(argument).name == "sunfish-train" for argument in arguments):
            return None
    elif record.get("role") != "descendant":
        raise ValueError("exact process snapshot has an invalid role")
    return process


def _signal_snapshot(
    proc_root: Path, payload: Mapping[str, Any], run_id: str, attempt_id: str
) -> tuple[list[int], list[int]]:
    live: list[tuple[Mapping[str, Any], int | None]] = []
    already_exited: list[int] = []
    use_pidfd = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    for record in payload["targets"]:
        process = _live_record(proc_root, record, run_id, attempt_id)
        if process is None:
            already_exited.append(int(record["pid"]))
            continue
        pidfd = os.pidfd_open(process["pid"], 0) if use_pidfd else None
        # Recheck after opening the pidfd so even the fallback never signals a
        # process whose identity changed during snapshot publication.
        if _live_record(proc_root, record, run_id, attempt_id) is None:
            if pidfd is not None:
                os.close(pidfd)
            already_exited.append(process["pid"])
            continue
        live.append((record, pidfd))

    signaled: list[int] = []
    # Stop roots first so they cannot spawn replacements, then signal every
    # descendant that was independently exact-recorded before the first kill.
    live.sort(key=lambda item: item[0]["role"] != "root")
    try:
        for record, pidfd in live:
            pid = int(record["pid"])
            try:
                if pidfd is not None:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                elif _live_record(proc_root, record, run_id, attempt_id) is not None:
                    os.kill(pid, signal.SIGKILL)
                else:
                    already_exited.append(pid)
                    continue
            except ProcessLookupError:
                already_exited.append(pid)
                continue
            signaled.append(pid)
    finally:
        for _record, pidfd in live:
            if pidfd is not None:
                os.close(pidfd)
    return signaled, already_exited


def _same_attempt_processes(
    proc_root: Path, run_id: str, attempt_id: str
) -> list[int]:
    leftovers: list[int] = []
    for path in proc_root.iterdir():
        if _PID.fullmatch(path.name) is None:
            continue
        process = _read_process(proc_root, int(path.name), os.getuid())
        if process is not None and _environment_matches(
            process["environ"], run_id, attempt_id
        ):
            leftovers.append(process["pid"])
    return sorted(leftovers)


def interrupt_attempt(
    run_id: str,
    attempt_id: str,
    *,
    pid_root: Path,
    proc_root: Path = Path("/proc"),
    wait_seconds: float = _WAIT_SECONDS,
) -> dict[str, Any]:
    for value in (run_id, attempt_id):
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("invalid run/attempt ID")
    if not pid_root.is_dir() or pid_root.is_symlink():
        raise ValueError("attempt PID root is not a regular directory")
    if not proc_root.is_dir() or proc_root.is_symlink():
        raise ValueError("Linux procfs is unavailable")
    hostname = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname())
    snapshot_path = pid_root / f"{run_id}.{attempt_id}.{hostname}.interrupt.json"
    if snapshot_path.exists() or snapshot_path.is_symlink():
        snapshot, snapshot_sha256 = _load_snapshot(
            snapshot_path, run_id, attempt_id
        )
    else:
        roots = _published_roots(pid_root, run_id, attempt_id)
        targets = _snapshot_processes(proc_root, roots, run_id, attempt_id)
        snapshot, snapshot_sha256 = _publish_snapshot(
            snapshot_path,
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "uid": os.getuid(),
                "hostname": hostname,
                "signal": "SIGKILL",
                "targets": targets,
            },
        )
    signaled, already_exited = _signal_snapshot(
        proc_root, snapshot, run_id, attempt_id
    )
    deadline = time.monotonic() + wait_seconds
    leftovers = _same_attempt_processes(proc_root, run_id, attempt_id)
    while leftovers and time.monotonic() < deadline:
        time.sleep(0.1)
        leftovers = _same_attempt_processes(proc_root, run_id, attempt_id)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "snapshot": str(snapshot_path),
        "snapshot_sha256": snapshot_sha256,
        "exact_recorded_targets": len(snapshot["targets"]),
        "exact_recorded_roots": sum(
            record.get("role") == "root" for record in snapshot["targets"]
        ),
        "exact_recorded_descendants": sum(
            record.get("role") == "descendant" for record in snapshot["targets"]
        ),
        "signaled_pids": signaled,
        "already_exited_pids": sorted(set(already_exited)),
        "same_attempt_processes_absent": not leftovers,
        "owner_intervention_required": bool(leftovers),
        "leftover_pids": leftovers,
    }
    if leftovers:
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        raise RuntimeError(
            "same-attempt processes survived the exact recorded-PID interrupt; "
            "owner intervention is required and automated retry is forbidden"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--pid-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = interrupt_attempt(
            args.run_id, args.attempt_id, pid_root=args.pid_root
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"sunfish-exact-process-interrupt: {error}", file=sys.stderr)
        return 7
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
