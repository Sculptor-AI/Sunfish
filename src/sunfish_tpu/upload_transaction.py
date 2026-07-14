"""Content-addressed, retry-safe directory publication for TPU workers.

This module is dependency-free because release deployment embeds its exact
source in a one-line remote Python command before Sunfish is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from collections.abc import Mapping, Sequence


MARKER_NAME = ".sunfish-upload.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validated_paths(final: Path, identity: str) -> tuple[Path, Path]:
    if _SHA256.fullmatch(identity) is None:
        raise ValueError("upload identity must be a lowercase SHA-256")
    final_text = str(final)
    pure = PurePosixPath(final_text)
    if (
        not pure.is_absolute()
        or pure == PurePosixPath("/")
        or ".." in pure.parts
        or "\n" in final_text
    ):
        raise ValueError("final upload directory must be a safe non-root absolute path")
    final = Path(final_text)
    temporary = Path(f"{final_text}.upload-{identity[:16]}")
    return final, temporary


def _marker_payload(final: Path, identity: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "identity": identity,
        "final_directory": str(final),
    }


def _marker_path(directory: Path) -> Path:
    return directory / MARKER_NAME


def _read_marker(directory: Path, final: Path, identity: str) -> None:
    marker = _marker_path(directory)
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"upload transaction has no trusted marker: {directory}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid upload transaction marker: {directory}") from error
    if payload != _marker_payload(final, identity):
        raise ValueError(f"upload transaction marker belongs to another release: {directory}")


def _remove_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise ValueError(f"refusing to remove unusual upload object: {path}")


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"uploaded artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_files(values: Sequence[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for value in values:
        name, separator, digest = value.partition("=")
        if (
            not separator
            or not name
            or name in {".", "..", MARKER_NAME}
            or PurePosixPath(name).name != name
            or "\n" in name
            or _SHA256.fullmatch(digest) is None
            or name in files
        ):
            raise ValueError(f"invalid upload file specification: {value!r}")
        files[name] = digest
    if not files:
        raise ValueError("at least one upload file specification is required")
    return files


def prepare_transaction(final: Path, identity: str) -> Path:
    """Create or reconcile a marked content-addressed staging directory."""

    final, temporary = _validated_paths(final, identity)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists() or temporary.is_symlink():
        if not temporary.is_dir() or temporary.is_symlink():
            raise ValueError(f"upload staging path is not a regular directory: {temporary}")
        _read_marker(temporary, final, identity)
    else:
        temporary.mkdir(mode=0o700)
        marker = _marker_path(temporary)
        marker.write_text(
            json.dumps(_marker_payload(final, identity), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _read_marker(temporary, final, identity)
    for child in temporary.iterdir():
        if child.name != MARKER_NAME:
            _remove_entry(child)
    return temporary


def verify_uploaded_file(final: Path, identity: str, name: str, sha256: str) -> Path:
    """Verify one staged file without trusting SCP's exit status alone."""

    final, temporary = _validated_paths(final, identity)
    _read_marker(temporary, final, identity)
    files = _parse_files([f"{name}={sha256}"])
    path = temporary / name
    actual = _sha256_file(path)
    if actual != files[name]:
        raise ValueError(f"uploaded artifact hash mismatch: {path}")
    return path


def _verify_file_set(
    directory: Path,
    files: Mapping[str, str],
    *,
    final: Path,
    identity: str,
    marker_allowed: bool,
) -> bool:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"published upload is not a regular directory: {directory}")
    names = {path.name for path in directory.iterdir()}
    marker_present = MARKER_NAME in names
    expected = set(files)
    if marker_present:
        if not marker_allowed:
            raise ValueError(f"unexpected upload marker in published directory: {directory}")
        _read_marker(directory, final, identity)
        expected.add(MARKER_NAME)
    if names != expected:
        raise ValueError(f"upload file inventory mismatch: {directory}")
    for name, expected_hash in files.items():
        if _sha256_file(directory / name) != expected_hash:
            raise ValueError(f"uploaded artifact hash mismatch: {directory / name}")
    return marker_present


def cleanup_transaction(final: Path, identity: str) -> None:
    """Remove only a staging directory carrying the exact trusted marker."""

    final, temporary = _validated_paths(final, identity)
    _read_marker(temporary, final, identity)
    for child in temporary.iterdir():
        if child.name != MARKER_NAME:
            _remove_entry(child)
    _marker_path(temporary).unlink()
    temporary.rmdir()


def publish_files(final: Path, identity: str, files: Mapping[str, str]) -> Path:
    """Atomically publish files, accepting an already-published exact result."""

    normalized = _parse_files([f"{name}={digest}" for name, digest in files.items()])
    final, temporary = _validated_paths(final, identity)
    _read_marker(temporary, final, identity)
    _verify_file_set(
        temporary,
        normalized,
        final=final,
        identity=identity,
        marker_allowed=True,
    )
    if final.exists() or final.is_symlink():
        marker_present = _verify_file_set(
            final,
            normalized,
            final=final,
            identity=identity,
            marker_allowed=True,
        )
        if marker_present:
            _marker_path(final).unlink()
            _verify_file_set(
                final,
                normalized,
                final=final,
                identity=identity,
                marker_allowed=False,
            )
        cleanup_transaction(final, identity)
        return final

    temporary.rename(final)
    marker_present = _verify_file_set(
        final,
        normalized,
        final=final,
        identity=identity,
        marker_allowed=True,
    )
    if not marker_present:
        raise RuntimeError("atomic upload publication lost its transaction marker")
    _marker_path(final).unlink()
    _verify_file_set(
        final,
        normalized,
        final=final,
        identity=identity,
        marker_allowed=False,
    )
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify-file", "cleanup", "publish-files"):
        child = subparsers.add_parser(command)
        child.add_argument("--final", type=Path, required=True)
        child.add_argument("--identity", required=True)
        if command == "verify-file":
            child.add_argument("--name", required=True)
            child.add_argument("--sha256", required=True)
        elif command == "publish-files":
            child.add_argument("--file", action="append", default=[], required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_transaction(args.final, args.identity)
        elif args.command == "verify-file":
            result = verify_uploaded_file(
                args.final, args.identity, args.name, args.sha256
            )
        elif args.command == "cleanup":
            cleanup_transaction(args.final, args.identity)
            result = args.final
        else:
            result = publish_files(
                args.final, args.identity, _parse_files(args.file)
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"sunfish-upload-transaction: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
