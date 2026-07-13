"""Dependency-free deterministic identity for a Git workspace."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPLOYMENT_PATHS = (
    "src",
    "scripts",
    "configs",
    "pyproject.toml",
    "requirements*.lock",
    "reference/upstream",
)


def source_tree_digest(root: Path) -> tuple[str, int]:
    root = root.resolve()
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *_DEPLOYMENT_PATHS,
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    names = sorted(name for name in result.stdout.split(b"\0") if name)
    if not names:
        raise RuntimeError("deployment source tree contains no versioned files")
    digest = hashlib.sha256()
    for raw_name in names:
        name = raw_name.decode("utf-8", errors="strict")
        path = root / name
        if path.is_symlink():
            kind = b"symlink"
            payload = os.readlink(path).encode("utf-8")
        elif path.is_file():
            kind = b"file"
            payload = path.read_bytes()
        else:
            raise RuntimeError(f"source entry is not a file or symlink: {name}")
        mode = path.lstat().st_mode & 0o777
        digest.update(len(raw_name).to_bytes(8, "little"))
        digest.update(raw_name)
        digest.update(kind)
        digest.update(mode.to_bytes(2, "little"))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest(), len(names)


def workspace_source_identity(root: Path) -> dict[str, str | int]:
    root = root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _COMMIT.fullmatch(commit):
        raise RuntimeError("workspace is not on a lowercase 40-hex Git commit")
    digest, files = source_tree_digest(root)
    return {
        "git_commit": commit,
        "source_tree_sha256": digest,
        "source_files": files,
    }
