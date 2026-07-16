"""Read-only TPU worker hygiene check before a workload starts.

The check never signals a process and never removes a lock file.  It reports a
hard owner-intervention stop when another current-user process holds a TPU
accelerator device, or when ``/tmp/libtpu_lockfile`` has no current-attempt
current-user owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from collections.abc import Sequence
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PID = re.compile(r"^[1-9][0-9]*$")


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as source:
        return source.read()


def _environment_matches(raw: bytes, run_id: str, attempt_id: str) -> bool:
    values = set(raw.split(b"\0"))
    return (
        f"SUNFISH_RUN_ID={run_id}".encode() in values
        and f"SUNFISH_ATTEMPT_ID={attempt_id}".encode() in values
    )


def _current_user_processes(proc_root: Path) -> list[Path]:
    uid = os.getuid()
    result: list[Path] = []
    for path in proc_root.iterdir():
        if _PID.fullmatch(path.name) is None:
            continue
        try:
            if (
                path.stat(follow_symlinks=False).st_uid == uid
                and path.is_dir()
                and not path.is_symlink()
            ):
                result.append(path)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(result, key=lambda path: int(path.name))


def inspect_worker_hygiene(
    run_id: str,
    attempt_id: str,
    *,
    proc_root: Path = Path("/proc"),
    lockfile: Path = Path("/tmp/libtpu_lockfile"),
) -> dict[str, Any]:
    for value in (run_id, attempt_id):
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("invalid run/attempt ID")
    if not proc_root.is_dir() or proc_root.is_symlink():
        raise ValueError("Linux procfs is unavailable")

    accelerator_holders: list[dict[str, Any]] = []
    lockfile_owners: list[dict[str, Any]] = []
    for process in _current_user_processes(proc_root):
        pid = int(process.name)
        try:
            environment = _read_bytes(process / "environ")
            cmdline = _read_bytes(process / "cmdline")
            descriptors = list((process / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        exact_attempt = _environment_matches(environment, run_id, attempt_id)
        command_hash = hashlib.sha256(cmdline).hexdigest()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, OSError, PermissionError):
                continue
            if target.startswith("/dev/accel"):
                accelerator_holders.append(
                    {
                        "pid": pid,
                        "device": target,
                        "cmdline_sha256": command_hash,
                        "exact_attempt": exact_attempt,
                    }
                )
            if target == str(lockfile):
                lockfile_owners.append(
                    {
                        "pid": pid,
                        "cmdline_sha256": command_hash,
                        "exact_attempt": exact_attempt,
                    }
                )

    lockfile_exists = lockfile.exists() or lockfile.is_symlink()
    lockfile_regular = (
        lockfile_exists
        and lockfile.is_file()
        and not lockfile.is_symlink()
        and lockfile.stat().st_uid == os.getuid()
    )
    lockfile_verified_owner = bool(lockfile_owners) and all(
        owner["exact_attempt"] for owner in lockfile_owners
    )
    errors: list[str] = []
    if accelerator_holders:
        errors.append("another current-user process holds /dev/accel*")
    if lockfile_exists and (not lockfile_regular or not lockfile_verified_owner):
        errors.append(
            "/tmp/libtpu_lockfile exists without a verified current-attempt owner"
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "uid": os.getuid(),
        "accelerator_holders": accelerator_holders,
        "lockfile": str(lockfile),
        "lockfile_exists": lockfile_exists,
        "lockfile_regular_current_user": lockfile_regular,
        "lockfile_owners": lockfile_owners,
        "lockfile_verified_current_attempt_owner": lockfile_verified_owner,
        "read_only": True,
        "passed": not errors,
        "owner_intervention_required": bool(errors),
        "errors": errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = inspect_worker_hygiene(args.run_id, args.attempt_id)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(f"hygiene report already exists: {args.output}")
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        if not result["passed"]:
            print(
                "sunfish-worker-hygiene: owner intervention is required; "
                "do not remove/kill automatically and do not retry",
                file=sys.stderr,
            )
            return 7
    except (OSError, ValueError) as error:
        print(f"sunfish-worker-hygiene: {error}", file=sys.stderr)
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
