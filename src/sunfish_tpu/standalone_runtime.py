"""Stock-Python-compatible installation of Sunfish's pinned TPU runtime.

This module is deliberately dependency-free and compatible with CPython 3.10.
The controller embeds its source in the pre-deployment command, before the
Sunfish wheel or the pinned CPython runtime exists on a worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


RUNTIME_METADATA_SCHEMA_VERSION = 1
RUNTIME_KIND = "python-build-standalone"
RUNTIME_RELEASE = "20260623"
RUNTIME_PYTHON_VERSION = "3.12.13"
RUNTIME_PLATFORM = "x86_64-unknown-linux-gnu"
RUNTIME_FLAVOR = "install_only"
RUNTIME_ARCHIVE_NAME = (
    "cpython-3.12.13+20260623-"
    "x86_64-unknown-linux-gnu-install_only.tar.gz"
)
RUNTIME_ARCHIVE_SHA256 = (
    "9fa869d69be54f6b8eeae64272fbd9bb0646e0e1a8da9d80e51ba5a3bee48930"
)
RUNTIME_ARCHIVE_SIZE = 111_146_559
RUNTIME_ARCHIVE_DIRECTORY = "python-runtime"
RUNTIME_METADATA_NAME = "python-runtime.json"
RUNTIME_INSTALL_DIRECTORY = "python"
RUNTIME_EXECUTABLE = "python/bin/python3"
RUNTIME_MARKER_NAME = ".sunfish-python-runtime.json"
BUNDLE_DIRECTORY_NAME = "sunfish-tpu-offline"
BUNDLE_MANIFEST_NAME = "offline-bundle.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARCHIVE_MEMBERS = 500_000
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024


def expected_runtime_metadata() -> dict[str, Any]:
    """Return the exact public-runtime identity without any network URL."""

    return {
        "schema_version": RUNTIME_METADATA_SCHEMA_VERSION,
        "kind": RUNTIME_KIND,
        "release": RUNTIME_RELEASE,
        "python_version": RUNTIME_PYTHON_VERSION,
        "platform": RUNTIME_PLATFORM,
        "flavor": RUNTIME_FLAVOR,
        "archive": f"{RUNTIME_ARCHIVE_DIRECTORY}/{RUNTIME_ARCHIVE_NAME}",
        "archive_sha256": RUNTIME_ARCHIVE_SHA256,
        "archive_size": RUNTIME_ARCHIVE_SIZE,
        "install_directory": RUNTIME_INSTALL_DIRECTORY,
        "python_executable": RUNTIME_EXECUTABLE,
    }


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"runtime artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(encoded)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def write_runtime_metadata(archive: Path, output: Path) -> dict[str, Any]:
    """Verify the downloaded asset and write its URL-free immutable metadata."""

    metadata = expected_runtime_metadata()
    if archive.name != RUNTIME_ARCHIVE_NAME:
        raise ValueError("standalone Python archive filename differs from the pin")
    if archive.stat().st_size != RUNTIME_ARCHIVE_SIZE:
        raise ValueError("standalone Python archive size differs from the pin")
    if _sha256_file(archive) != RUNTIME_ARCHIVE_SHA256:
        raise ValueError("standalone Python archive SHA-256 differs from the pin")
    _write_json_exclusive(output, metadata)
    return metadata


def verify_runtime_artifacts(
    bundle_root: Path, *, require_bundle_manifest: bool = False
) -> dict[str, Any]:
    """Verify the exact runtime archive, metadata, and optional bundle binding."""

    bundle_root = bundle_root.resolve()
    metadata_path = bundle_root / RUNTIME_METADATA_NAME
    metadata = _read_json_object(metadata_path, "standalone Python metadata")
    expected = expected_runtime_metadata()
    if metadata != expected:
        raise ValueError("standalone Python metadata differs from the audited pin")
    archive = bundle_root / metadata["archive"]
    if archive.stat().st_size != metadata["archive_size"]:
        raise ValueError("standalone Python archive size differs from metadata")
    if _sha256_file(archive) != metadata["archive_sha256"]:
        raise ValueError("standalone Python archive hash differs from metadata")

    if require_bundle_manifest:
        manifest = _read_json_object(
            bundle_root / BUNDLE_MANIFEST_NAME, "offline bundle manifest"
        )
        if manifest.get("python_runtime") != metadata:
            raise ValueError("offline bundle does not bind the standalone runtime")
        records = manifest.get("files")
        if not isinstance(records, list):
            raise ValueError("offline bundle file inventory is missing")
        by_path: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("offline bundle has an invalid file record")
            relative = record.get("path")
            if not isinstance(relative, str) or relative in by_path:
                raise ValueError("offline bundle has a malformed file record")
            by_path[relative] = record
        expected_records = {
            RUNTIME_METADATA_NAME: {
                "size": metadata_path.stat().st_size,
                "sha256": _sha256_file(metadata_path),
            },
            metadata["archive"]: {
                "size": metadata["archive_size"],
                "sha256": metadata["archive_sha256"],
            },
        }
        for relative, values in expected_records.items():
            record = by_path.get(relative)
            if (
                record is None
                or record.get("size") != values["size"]
                or record.get("sha256") != values["sha256"]
            ):
                raise ValueError(
                    f"offline bundle runtime inventory differs: {relative}"
                )
    return metadata


def _member_path(name: str, required_prefix: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\n" in name:
        raise ValueError("archive contains an invalid member name")
    path = PurePosixPath(name.rstrip("/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"archive member escapes its root: {name!r}")
    if path.parts[0] != required_prefix:
        raise ValueError(f"archive member has an unexpected root: {name!r}")
    return path


def _resolved_link_parts(
    member: PurePosixPath, linkname: str, required_prefix: str
) -> tuple[str, ...]:
    if (
        not isinstance(linkname, str)
        or not linkname
        or "\x00" in linkname
        or "\n" in linkname
    ):
        raise ValueError(f"archive link has an invalid target: {member}")
    target = PurePosixPath(linkname)
    if target.is_absolute():
        raise ValueError(f"archive link target is absolute: {member}")
    parts = list(member.parent.parts)
    for part in target.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"archive link target escapes its root: {member}")
            parts.pop()
        else:
            parts.append(part)
    if not parts or parts[0] != required_prefix:
        raise ValueError(f"archive link target escapes its root: {member}")
    return tuple(parts)


def _ensure_real_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if not current.is_dir() or current.is_symlink():
                raise ValueError(f"archive path crosses a non-directory: {relative}")
        else:
            current.mkdir(mode=0o700)
    return current


def _validated_members(
    archive: tarfile.TarFile, required_prefix: str
) -> list[tuple[tarfile.TarInfo, PurePosixPath]]:
    raw_members = archive.getmembers()
    if not raw_members or len(raw_members) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError("archive member count is invalid")
    validated: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
    observed: dict[PurePosixPath, str] = {}
    uncompressed = 0
    for member in raw_members:
        relative = _member_path(member.name, required_prefix)
        if relative in observed:
            raise ValueError(f"archive contains a duplicate member: {relative}")
        if member.isdir():
            kind = "directory"
        elif member.isfile():
            kind = "file"
            uncompressed += member.size
        elif member.issym():
            kind = "symlink"
            _resolved_link_parts(relative, member.linkname, required_prefix)
        elif member.islnk():
            kind = "hardlink"
            _member_path(member.linkname, required_prefix)
        else:
            raise ValueError(f"archive contains a forbidden member type: {relative}")
        if uncompressed > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError("archive uncompressed size exceeds the safety bound")
        observed[relative] = kind
        validated.append((member, relative))

    non_directories = {
        path for path, kind in observed.items() if kind != "directory"
    }
    for path in observed:
        for depth in range(1, len(path.parts)):
            if PurePosixPath(*path.parts[:depth]) in non_directories:
                raise ValueError(f"archive path crosses a non-directory: {path}")
    for member, relative in validated:
        if member.islnk():
            target = _member_path(member.linkname, required_prefix)
            if observed.get(target) != "file":
                raise ValueError(f"archive hardlink target is not a file: {relative}")
    return validated


def safe_extract_archive(
    archive_path: Path, destination: Path, *, required_prefix: str
) -> Path:
    """Extract a tar archive without trusting member paths or link traversal."""

    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError(f"archive is not a regular file: {archive_path}")
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError(f"archive destination is not a regular directory: {destination}")
    prefix_path = destination / required_prefix
    if prefix_path.exists() or prefix_path.is_symlink():
        raise FileExistsError(f"archive target already exists: {prefix_path}")

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = _validated_members(archive, required_prefix)
        for member, relative in sorted(
            (item for item in members if item[0].isdir()),
            key=lambda item: len(item[1].parts),
        ):
            _ensure_real_directory(destination, relative)

        for member, relative in (item for item in members if item[0].isfile()):
            parent = _ensure_real_directory(destination, relative.parent)
            target = parent / relative.name
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive file has no payload: {relative}")
            descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                member.mode & 0o777,
            )
            try:
                with source, os.fdopen(descriptor, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fchmod(output.fileno(), member.mode & 0o777)
            except BaseException:
                try:
                    target.unlink()
                except OSError:
                    pass
                raise
            if target.stat().st_size != member.size:
                raise ValueError(f"archive file size changed during extraction: {relative}")

        for member, relative in (item for item in members if item[0].islnk()):
            parent = _ensure_real_directory(destination, relative.parent)
            target = parent / relative.name
            link_target = destination / _member_path(
                member.linkname, required_prefix
            ).as_posix()
            if not link_target.is_file() or link_target.is_symlink():
                raise ValueError(f"archive hardlink target changed: {relative}")
            os.link(link_target, target, follow_symlinks=False)

        for member, relative in (item for item in members if item[0].issym()):
            parent = _ensure_real_directory(destination, relative.parent)
            target = parent / relative.name
            _resolved_link_parts(relative, member.linkname, required_prefix)
            os.symlink(member.linkname, target)
        # Apply potentially read-only directory modes only after all children
        # exist; otherwise a legitimate 0555 directory could block its own
        # extraction midway through the transaction.
        for member, relative in sorted(
            (item for item in members if item[0].isdir()),
            key=lambda item: len(item[1].parts),
            reverse=True,
        ):
            os.chmod(
                destination / relative.as_posix(),
                (member.mode & 0o777) | 0o700,
            )
    if not prefix_path.is_dir() or prefix_path.is_symlink():
        raise ValueError("archive did not produce the required root directory")
    return prefix_path


def extract_bundle_archive(archive: Path, destination: Path) -> Path:
    return safe_extract_archive(
        archive, destination, required_prefix=BUNDLE_DIRECTORY_NAME
    )


def _runtime_tree_identity(runtime_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    entries = 0
    stack: list[tuple[Path, PurePosixPath]] = [(runtime_root, PurePosixPath("."))]
    while stack:
        directory, relative_directory = stack.pop()
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: entry.name, reverse=True)
        for entry in children:
            relative = (
                PurePosixPath(entry.name)
                if relative_directory == PurePosixPath(".")
                else relative_directory / entry.name
            )
            if relative.as_posix() == RUNTIME_MARKER_NAME:
                continue
            if entry.name == "__pycache__" or entry.name.endswith(".pyc"):
                # CPython bytecode caches are volatile: any legitimate run of the
                # bundled interpreter may (re)write them, so they cannot be part
                # of the immutable tree identity.
                continue
            path = Path(entry.path)
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                _resolved_link_parts(
                    PurePosixPath(RUNTIME_INSTALL_DIRECTORY) / relative,
                    target,
                    RUNTIME_INSTALL_DIRECTORY,
                )
                record = f"L\0{relative.as_posix()}\0{mode:o}\0{target}\0"
            elif stat.S_ISDIR(info.st_mode):
                record = f"D\0{relative.as_posix()}\0{mode:o}\0"
                stack.append((path, relative))
            elif stat.S_ISREG(info.st_mode):
                record = (
                    f"F\0{relative.as_posix()}\0{mode:o}\0{info.st_size}\0"
                    f"{_sha256_file(path)}\0"
                )
            else:
                raise ValueError(f"installed runtime has a forbidden object: {relative}")
            digest.update(record.encode("utf-8"))
            entries += 1
    return {"sha256": digest.hexdigest(), "entries": entries}


def _run_runtime_probe(python: Path) -> dict[str, Any]:
    program = (
        "import ensurepip,json,platform,sys,venv; "
        "print(json.dumps({"
        "'implementation':platform.python_implementation(),"
        "'python_version':platform.python_version(),"
        "'system':platform.system(),"
        "'machine':platform.machine(),"
        "'libc':list(platform.libc_ver()),"
        "'ensurepip':ensurepip.version()"
        "},sort_keys=True))"
    )
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", program],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("bundled Python probe returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("bundled Python probe must return an object")
    if (
        payload.get("implementation") != "CPython"
        or payload.get("python_version") != RUNTIME_PYTHON_VERSION
        or payload.get("system") != "Linux"
        or str(payload.get("machine", "")).lower() not in {"x86_64", "amd64"}
        or not isinstance(payload.get("ensurepip"), str)
        or not payload["ensurepip"]
    ):
        raise ValueError("installed bundled Python differs from the runtime contract")
    libc = payload.get("libc")
    if (
        not isinstance(libc, list)
        or len(libc) != 2
        or str(libc[0]).lower() != "glibc"
        or not libc[1]
    ):
        raise ValueError("installed bundled Python does not report glibc")
    return payload


def verify_installed_runtime(
    bundle_root: Path,
    *,
    destination: Path | None = None,
    require_bundle_manifest: bool = False,
) -> dict[str, Any]:
    metadata = verify_runtime_artifacts(
        bundle_root, require_bundle_manifest=require_bundle_manifest
    )
    destination = bundle_root if destination is None else destination
    runtime_root = destination / metadata["install_directory"]
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise ValueError(f"bundled Python directory is invalid: {runtime_root}")
    marker = _read_json_object(
        runtime_root / RUNTIME_MARKER_NAME, "bundled Python install marker"
    )
    if marker.get("schema_version") != 1 or marker.get("runtime") != metadata:
        raise ValueError("bundled Python install marker differs from the pin")
    identity = _runtime_tree_identity(runtime_root)
    if marker.get("tree") != identity:
        raise ValueError("bundled Python installed tree differs from its marker")
    python = destination / metadata["python_executable"]
    probe = _run_runtime_probe(python)
    return {"runtime": metadata, "tree": identity, "probe": probe}


def install_runtime(
    bundle_root: Path,
    *,
    destination: Path | None = None,
    require_bundle_manifest: bool = False,
) -> dict[str, Any]:
    """Safely derive an immutable ``python/`` tree from the pinned archive."""

    bundle_root = bundle_root.resolve()
    metadata = verify_runtime_artifacts(
        bundle_root, require_bundle_manifest=require_bundle_manifest
    )
    destination = bundle_root if destination is None else destination.resolve()
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or destination.is_symlink():
            raise ValueError(f"runtime destination is not a regular directory: {destination}")
    else:
        destination.mkdir(mode=0o700)
    runtime_root = destination / metadata["install_directory"]
    if runtime_root.exists() or runtime_root.is_symlink():
        return verify_installed_runtime(
            bundle_root,
            destination=destination,
            require_bundle_manifest=require_bundle_manifest,
        )

    staging = destination / f".sunfish-python-install-{RUNTIME_ARCHIVE_SHA256[:16]}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"bundled Python staging path already exists: {staging}")
    staging.mkdir(mode=0o700)
    installed = False
    try:
        archive = bundle_root / metadata["archive"]
        extracted = safe_extract_archive(
            archive, staging, required_prefix=RUNTIME_INSTALL_DIRECTORY
        )
        tree = _runtime_tree_identity(extracted)
        _write_json_exclusive(
            extracted / RUNTIME_MARKER_NAME,
            {"schema_version": 1, "runtime": metadata, "tree": tree},
        )
        _run_runtime_probe(staging / metadata["python_executable"])
        extracted.rename(runtime_root)
        installed = True
        staging.rmdir()
    finally:
        if not installed and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
    return verify_installed_runtime(
        bundle_root,
        destination=destination,
        require_bundle_manifest=require_bundle_manifest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("write-metadata")
    metadata.add_argument("--archive", type=Path, required=True)
    metadata.add_argument("--output", type=Path, required=True)

    extract = subparsers.add_parser("extract-bundle")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)

    for name in ("install", "verify-installed"):
        child = subparsers.add_parser(name)
        child.add_argument("--bundle-root", type=Path, required=True)
        child.add_argument("--destination", type=Path)
        child.add_argument("--require-bundle-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-metadata":
            result: Any = write_runtime_metadata(args.archive, args.output)
        elif args.command == "extract-bundle":
            result = {
                "bundle_root": str(
                    extract_bundle_archive(args.archive, args.destination)
                )
            }
        elif args.command == "install":
            result = install_runtime(
                args.bundle_root,
                destination=args.destination,
                require_bundle_manifest=args.require_bundle_manifest,
            )
        else:
            result = verify_installed_runtime(
                args.bundle_root,
                destination=args.destination,
                require_bundle_manifest=args.require_bundle_manifest,
            )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as error:
        print(f"sunfish-standalone-runtime: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
