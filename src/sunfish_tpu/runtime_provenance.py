"""Resolve Gemma source provenance without importing accelerator packages."""

from __future__ import annotations

import importlib.metadata
import json
import os
from collections.abc import Mapping
from pathlib import Path

from sunfish_tpu.offline_bundle import BUNDLE_MANIFEST_NAME, verify_bundle


def resolve_gemma_source_commit(
    *, environment: Mapping[str, str] | None = None
) -> tuple[str | None, str]:
    """Return the installed Gemma commit and the provenance mechanism used.

    Worker installs come from a locally installed wheel, whose PEP 610
    ``direct_url.json`` describes that wheel archive rather than the VCS input
    used to build it.  The immutable offline-bundle manifest is therefore the
    authoritative worker-side source record.  Connected development installs
    retain the direct-VCS fallback.
    """

    env = os.environ if environment is None else environment
    manifest_value = env.get("SUNFISH_OFFLINE_BUNDLE_MANIFEST", "")
    root_value = env.get("SUNFISH_OFFLINE_BUNDLE_ROOT", "")
    if manifest_value or root_value:
        if manifest_value:
            manifest_path = Path(manifest_value).expanduser().resolve()
            bundle_root = manifest_path.parent
            if manifest_path != bundle_root / BUNDLE_MANIFEST_NAME:
                raise ValueError(
                    "SUNFISH_OFFLINE_BUNDLE_MANIFEST must name offline-bundle.json"
                )
        else:
            bundle_root = Path(root_value).expanduser().resolve()
            manifest_path = bundle_root / BUNDLE_MANIFEST_NAME
        if root_value and Path(root_value).expanduser().resolve() != bundle_root:
            raise ValueError(
                "offline bundle root and manifest environment variables disagree"
            )
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("offline bundle manifest is missing or not a regular file")
        manifest = verify_bundle(bundle_root, verify_file_hashes=False)
        commit = manifest.get("gemma_source_commit")
        return (commit if isinstance(commit, str) else None), "verified-offline-bundle"

    try:
        direct_url_text = importlib.metadata.distribution("gemma").read_text(
            "direct_url.json"
        )
        direct_url = json.loads(direct_url_text or "{}")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        return None, "direct-url"
    if not isinstance(direct_url, Mapping):
        return None, "direct-url"
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, Mapping):
        return None, "direct-url"
    commit = vcs_info.get("commit_id")
    return (commit if isinstance(commit, str) else None), "direct-url"
