"""Dependency-free deterministic identity for a Git or exported release tree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPLOYMENT_PATHS = (
    "src",
    "scripts",
    "configs",
    "pyproject.toml",
    "requirements*.lock",
    "reference/upstream",
)
_RELEASE_MANIFEST = ".sunfish-release.json"
_IGNORED_EXPORT_PARTS = {"__pycache__", ".pytest_cache"}


def _valid_relative_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _git_toplevel(root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    try:
        return Path(result.stdout.strip()).resolve()
    except OSError:
        return None


def _git_source_names(root: Path) -> list[bytes]:
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
    return sorted(
        name
        for name in result.stdout.split(b"\0")
        if name
        and (
            (root / name.decode("utf-8", errors="strict")).exists()
            or (root / name.decode("utf-8", errors="strict")).is_symlink()
        )
    )


def _read_release_manifest(root: Path) -> dict[str, Any]:
    path = root / _RELEASE_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid exported source identity: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported exported source identity")
    commit = payload.get("git_commit")
    digest = payload.get("source_tree_sha256")
    count = payload.get("source_files")
    names = payload.get("deployment_files")
    if (
        not isinstance(commit, str)
        or not _COMMIT.fullmatch(commit)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(names, list)
        or len(names) != count
        or any(not isinstance(name, str) or not _valid_relative_name(name) for name in names)
        or names != sorted(set(names))
    ):
        raise RuntimeError("malformed exported source identity")
    return payload


def _is_ignored_export_path(path: Path) -> bool:
    return bool(_IGNORED_EXPORT_PARTS.intersection(path.parts)) or path.suffix in {
        ".pyc",
        ".pyo",
    }


def _exported_source_names(root: Path, payload: dict[str, Any]) -> list[bytes]:
    declared = payload["deployment_files"]
    observed: set[str] = set()
    for pattern in _DEPLOYMENT_PATHS:
        roots = root.glob(pattern) if "*" in pattern else (root / pattern,)
        for candidate in roots:
            if not candidate.exists() and not candidate.is_symlink():
                continue
            entries = candidate.rglob("*") if candidate.is_dir() else (candidate,)
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    continue
                relative = entry.relative_to(root)
                if _is_ignored_export_path(relative):
                    continue
                observed.add(relative.as_posix())
    if observed != set(declared):
        missing = sorted(set(declared) - observed)
        extra = sorted(observed - set(declared))
        raise RuntimeError(
            "exported source file set changed: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return [name.encode("utf-8") for name in declared]


def _digest_source_names(root: Path, names: list[bytes]) -> tuple[str, int]:
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
        # Git records only whether an entry is executable, not checkout-local
        # read/write bits inherited from the host umask. Encode that one bit
        # canonically so equivalent 0644/0664 and 0755/0775 trees agree.
        mode = 0o755 if path.lstat().st_mode & 0o100 else 0o644
        digest.update(len(raw_name).to_bytes(8, "little"))
        digest.update(raw_name)
        digest.update(kind)
        digest.update(mode.to_bytes(2, "little"))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest(), len(names)


def source_tree_digest(root: Path) -> tuple[str, int]:
    root = root.resolve()
    if _git_toplevel(root) == root:
        return _digest_source_names(root, _git_source_names(root))
    release = _read_release_manifest(root)
    actual = _digest_source_names(root, _exported_source_names(root, release))
    expected = (release["source_tree_sha256"], release["source_files"])
    if actual != expected:
        raise RuntimeError("exported source digest differs from its release identity")
    return actual


def workspace_source_identity(root: Path) -> dict[str, str | int]:
    root = root.resolve()
    if _git_toplevel(root) == root:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        commit = _read_release_manifest(root)["git_commit"]
    if not _COMMIT.fullmatch(commit):
        raise RuntimeError("workspace is not on a lowercase 40-hex Git commit")
    digest, files = source_tree_digest(root)
    return {
        "git_commit": commit,
        "source_tree_sha256": digest,
        "source_files": files,
    }
