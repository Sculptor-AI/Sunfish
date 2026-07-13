"""Dependency-free deployed-source identity contract."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def source_identity_from_environment(*, required: bool) -> dict[str, str]:
    commit = os.environ.get("SUNFISH_GIT_COMMIT", "")
    tree = os.environ.get("SUNFISH_SOURCE_TREE_SHA256", "")
    if not commit and not tree and not required:
        return {"git_commit": "unrecorded", "source_tree_sha256": "unrecorded"}
    if not _COMMIT.fullmatch(commit):
        raise RuntimeError("SUNFISH_GIT_COMMIT must be a lowercase 40-hex commit")
    if not _SHA256.fullmatch(tree):
        raise RuntimeError(
            "SUNFISH_SOURCE_TREE_SHA256 must be a lowercase SHA-256 digest"
        )
    return {"git_commit": commit, "source_tree_sha256": tree}


def normalize_source_identity(value: Any) -> tuple[str, str] | None:
    """Return a hashable strict identity, or None for malformed evidence."""
    if not isinstance(value, Mapping):
        return None
    commit = value.get("git_commit")
    tree = value.get("source_tree_sha256")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return None
    if not isinstance(tree, str) or not _SHA256.fullmatch(tree):
        return None
    return commit, tree


def require_launcher_run_id(run_id: str, *, required: bool = True) -> None:
    """Bind a hardware program's CLI/config run ID to its all-host launch."""
    launcher_run_id = os.environ.get("SUNFISH_RUN_ID", "")
    if not launcher_run_id and not required:
        return
    if launcher_run_id != run_id:
        raise RuntimeError(
            f"launcher run ID {launcher_run_id!r} differs from program run ID {run_id!r}"
        )
