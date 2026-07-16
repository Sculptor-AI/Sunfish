"""Build-time and worker-side validation for an immutable offline TPU bundle."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default as default_email_policy
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sunfish.source_tree import (
    _DEPLOYMENT_PATHS,
    workspace_source_identity,
)
from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)
from sunfish_tpu import standalone_runtime

BUNDLE_SCHEMA_VERSION = 3
BUNDLE_KIND = "sunfish-tpu-offline-bundle"
BUNDLE_DIRECTORY_NAME = "sunfish-tpu-offline"
BUNDLE_MANIFEST_NAME = "offline-bundle.json"
RELEASE_MANIFEST_NAME = ".sunfish-release.json"
RESOLVED_LOCK_NAME = "offline-requirements.lock"
WHEELHOUSE_NAME = "wheelhouse"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")
_NORMALIZE_NAME = re.compile(r"[-_.]+")
_MANYLINUX_TAG = re.compile(r"^manylinux_(\d+)_(\d+)_x86_64$")
_GLIBC_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_BOOTSTRAP_DISTRIBUTIONS = {"pip", "setuptools"}
_LEGACY_MANYLINUX_GLIBC = {
    "manylinux1_x86_64": (2, 5),
    "manylinux2010_x86_64": (2, 12),
    "manylinux2014_x86_64": (2, 17),
}
_WORKER_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def normalize_distribution_name(name: str) -> str:
    return _NORMALIZE_NAME.sub("-", name).lower()


def _version_tuple(value: str) -> tuple[int, ...]:
    if not _GLIBC_VERSION.fullmatch(value):
        raise ValueError(f"invalid glibc version: {value!r}")
    return tuple(int(part) for part in value.split("."))


def _format_version(value: Sequence[int]) -> str:
    return ".".join(str(part) for part in value)


def _wheel_minimum_glibc(path: Path) -> tuple[int, ...] | None:
    """Return the least glibc version admitted by one wheel's platform tags."""
    if path.suffix != ".whl":
        raise ValueError(f"not a wheel: {path.name}")
    try:
        platform_tags = path.name[:-4].rsplit("-", 3)[3].split(".")
    except IndexError as error:
        raise ValueError(f"invalid wheel filename: {path.name}") from error
    if "any" in platform_tags:
        return None
    requirements: list[tuple[int, ...]] = []
    for tag in platform_tags:
        if tag in _LEGACY_MANYLINUX_GLIBC:
            requirements.append(_LEGACY_MANYLINUX_GLIBC[tag])
            continue
        match = _MANYLINUX_TAG.fullmatch(tag)
        if match is not None:
            requirements.append((int(match.group(1)), int(match.group(2))))
    if not requirements:
        raise ValueError(
            "native wheel has no versioned manylinux x86_64 compatibility tag: "
            f"{path.name}"
        )
    # A wheel with several platform tags is usable when any one tag matches.
    return min(requirements)


def minimum_glibc_for_wheels(wheels: Sequence[Path]) -> str:
    """Compute the worker glibc floor encoded by the complete wheel set."""
    minimum = (2, 17)
    for wheel in wheels:
        requirement = _wheel_minimum_glibc(wheel)
        if requirement is not None:
            minimum = max(minimum, requirement)
    return _format_version(minimum)


def verify_worker_runtime_compatibility(
    manifest: Mapping[str, Any],
    *,
    system: str | None = None,
    machine: str | None = None,
    python_version: str | None = None,
    libc: tuple[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
    discovered_proxies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed when the current worker cannot run the immutable bundle."""
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    python_version = platform.python_version() if python_version is None else python_version
    libc = platform.libc_ver() if libc is None else libc
    environment = os.environ if environment is None else environment
    discovered_proxies = (
        urllib.request.getproxies()
        if discovered_proxies is None
        else discovered_proxies
    )
    target = manifest.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("offline bundle target is missing")
    minimum_glibc = target.get("minimum_glibc")
    if not isinstance(minimum_glibc, str):
        raise ValueError("offline bundle glibc floor is missing")
    active_environment = {
        name: environment[name]
        for name in _WORKER_PROXY_VARIABLES
        if environment.get(name)
    }
    active_discovered = {
        str(name): value
        for name, value in discovered_proxies.items()
        if str(name).lower() in {"http", "https", "all"} and value
    }
    if active_environment or active_discovered:
        names = sorted({*active_environment, *active_discovered})
        raise ValueError(
            "worker HTTP(S)/ALL proxy settings are forbidden before JAX startup: "
            + ", ".join(names)
        )
    if system != "Linux" or machine.lower() not in {"x86_64", "amd64"}:
        raise ValueError(
            f"worker platform is incompatible with bundle: {system}/{machine}"
        )
    if python_version != standalone_runtime.RUNTIME_PYTHON_VERSION:
        raise ValueError(
            f"worker Python {python_version} differs from required Python "
            f"{standalone_runtime.RUNTIME_PYTHON_VERSION}"
        )
    if libc[0].lower() != "glibc" or not libc[1]:
        raise ValueError(f"worker does not report glibc: {libc!r}")
    observed_glibc = _version_tuple(libc[1])
    required_glibc = _version_tuple(minimum_glibc)
    width = max(len(observed_glibc), len(required_glibc))
    if observed_glibc + (0,) * (width - len(observed_glibc)) < required_glibc + (
        0,
    ) * (width - len(required_glibc)):
        raise ValueError(
            f"worker glibc {libc[1]} is older than bundle floor {minimum_glibc}"
        )
    return {
        "operating_system": "linux",
        "machine": "x86_64",
        "python": python_version,
        "glibc": libc[1],
        "minimum_glibc": minimum_glibc,
        "proxy_environment_clear": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_resolved_lock(path: Path) -> dict[str, str]:
    """Parse the URL-free, fully resolved worker lock generated by the builder."""
    distributions: dict[str, str] = {}
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or any(token in line for token in ("@", ";", "[", "]")):
            raise ValueError(f"offline lock line {number} is not name==version")
        name, version = line.split("==", 1)
        if (
            not _DIST_NAME.fullmatch(name)
            or not _VERSION.fullmatch(version)
        ):
            raise ValueError(f"invalid offline lock line {number}")
        normalized = normalize_distribution_name(name)
        if normalized in distributions:
            raise ValueError(f"duplicate offline distribution: {normalized}")
        distributions[normalized] = version
    if not distributions:
        raise ValueError("offline lock is empty")
    return dict(sorted(distributions.items()))


def _git_bytes(root: Path, arguments: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def export_release_source(repository: Path, output: Path) -> dict[str, Any]:
    """Export only deployable committed files plus a verifiable release identity."""
    repository = repository.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"release source output already exists: {output}")
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("release source output must be outside the repository")

    status = _git_bytes(
        repository,
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_DEPLOYMENT_PATHS,
        ],
    )
    if status:
        raise RuntimeError("deployable source tree must be clean before bundling")
    raw_names = _git_bytes(
        repository,
        ["ls-files", "--cached", "-z", "--", *_DEPLOYMENT_PATHS],
    )
    names = sorted(name.decode("utf-8") for name in raw_names.split(b"\0") if name)
    if not names:
        raise RuntimeError("repository has no deployable source files")

    identity = workspace_source_identity(repository)
    output.mkdir(parents=True)
    for name in names:
        source = repository / name
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            os.symlink(os.readlink(source), destination)
        else:
            shutil.copy2(source, destination)
    release = {
        "schema_version": 1,
        **identity,
        "deployment_files": names,
    }
    _write_json_atomic(output / RELEASE_MANIFEST_NAME, release)
    if workspace_source_identity(output) != identity:
        raise RuntimeError("exported source identity differs from repository")
    return release


def _installed_distributions(python: Path) -> dict[str, str]:
    program = (
        "import importlib.metadata,json; "
        "print(json.dumps(sorted((d.metadata['Name'], d.version) "
        "for d in importlib.metadata.distributions())))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    entries = json.loads(result.stdout)
    observed: dict[str, str] = {}
    for name, version in entries:
        normalized = normalize_distribution_name(name)
        previous = observed.setdefault(normalized, version)
        if previous != version:
            raise RuntimeError(f"multiple installed versions for {normalized}")
    return dict(sorted(observed.items()))


def write_resolved_lock(python: Path, output: Path) -> dict[str, str]:
    observed = {
        name: version
        for name, version in _installed_distributions(python).items()
        if name not in _BOOTSTRAP_DISTRIBUTIONS
    }
    if not observed:
        raise RuntimeError("validation environment contains no runtime distributions")
    if "requests" not in observed:
        raise RuntimeError(
            "resolved TPU runtime is missing requests required by jax[tpu] "
            "distributed initialization"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Generated and offline-validated by build_tpu_offline_bundle.sh.\n"
        + "".join(f"{name}=={version}\n" for name, version in observed.items()),
        encoding="utf-8",
    )
    resolved = parse_resolved_lock(output)
    if resolved.get("requests") != observed["requests"]:
        raise RuntimeError("resolved TPU lock did not preserve exactly one requests pin")
    return observed


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"bundle artifact is not a regular file: {relative}")
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_sunfish_wheel_matches_source(source_root: Path, wheels: list[Path]) -> None:
    candidates = [
        path
        for path in wheels
        if path.name.lower().startswith("sunfish_diffusion-")
    ]
    if len(candidates) != 1:
        raise ValueError("wheelhouse must contain exactly one Sunfish wheel")
    expected: dict[str, bytes] = {}
    for package in ("sunfish", "sunfish_tpu"):
        package_root = source_root / "src" / package
        for path in package_root.rglob("*.py"):
            expected[path.relative_to(source_root / "src").as_posix()] = path.read_bytes()
    if not expected:
        raise ValueError("exported source has no Sunfish Python modules")
    try:
        with zipfile.ZipFile(candidates[0]) as wheel:
            observed = {
                name: wheel.read(name)
                for name in wheel.namelist()
                if name.endswith(".py")
                and (name.startswith("sunfish/") or name.startswith("sunfish_tpu/"))
            }
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("invalid Sunfish wheel") from error
    if observed != expected:
        raise ValueError("Sunfish wheel modules differ from exported source")


def _wheel_identity(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_files = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError("wheel must contain exactly one METADATA file")
            metadata = BytesParser(policy=default_email_policy).parsebytes(
                wheel.read(metadata_files[0])
            )
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"invalid wheel metadata: {path.name}") from error
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError(f"wheel is missing Name/Version metadata: {path.name}")
    return normalize_distribution_name(name), version


def _verify_wheelhouse_matches_lock(
    wheels: Sequence[Path], distributions: Mapping[str, str]
) -> None:
    observed: dict[str, str] = {}
    for wheel in wheels:
        name, version = _wheel_identity(wheel)
        if name in observed:
            raise ValueError(f"wheelhouse contains duplicate distribution: {name}")
        observed[name] = version
    if observed != distributions:
        missing = sorted(set(distributions) - set(observed))
        extra = sorted(set(observed) - set(distributions))
        changed = {
            name: {"expected": distributions[name], "observed": observed[name]}
            for name in sorted(set(observed) & set(distributions))
            if observed[name] != distributions[name]
        }
        raise ValueError(
            "wheelhouse differs from resolved lock: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def create_bundle_manifest(bundle_root: Path) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    source_root = bundle_root / "source"
    lock_path = bundle_root / RESOLVED_LOCK_NAME
    wheelhouse = bundle_root / WHEELHOUSE_NAME
    identity = workspace_source_identity(source_root)
    distributions = parse_resolved_lock(lock_path)
    python_runtime = standalone_runtime.verify_runtime_artifacts(bundle_root)

    expected_direct = {
        normalize_distribution_name(name): version
        for name, version in {**RUNTIME_VERSIONS, **TPU_ONLY_RUNTIME_VERSIONS}.items()
    }
    expected_direct["sunfish-diffusion"] = "0.1.0"
    mismatches = {
        name: {"expected": version, "observed": distributions.get(name)}
        for name, version in expected_direct.items()
        if distributions.get(name) != version
    }
    if mismatches:
        raise ValueError(f"offline environment direct-pin mismatch: {mismatches}")

    wheels = sorted(path for path in wheelhouse.iterdir() if path.is_file())
    if not wheels or any(path.suffix != ".whl" for path in wheels):
        raise ValueError("wheelhouse must contain wheels only")
    if any(path.is_dir() for path in wheelhouse.iterdir()):
        raise ValueError("wheelhouse may not contain directories")
    _verify_wheelhouse_matches_lock(wheels, distributions)
    _verify_sunfish_wheel_matches_source(source_root, wheels)

    relative_files = [
        f"source/{RELEASE_MANIFEST_NAME}",
        RESOLVED_LOCK_NAME,
        standalone_runtime.RUNTIME_METADATA_NAME,
        python_runtime["archive"],
        *(f"{WHEELHOUSE_NAME}/{path.name}" for path in wheels),
    ]
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "network_policy": "worker-no-egress",
        "transport_policy": "gcloud-alpha-tpu-vm-iap-all-workers",
        "target": {
            "operating_system": "linux",
            "machine": "x86_64",
            "python": standalone_runtime.RUNTIME_PYTHON_VERSION,
            "minimum_glibc": minimum_glibc_for_wheels(wheels),
        },
        "builder": {
            "operating_system": platform.system().lower(),
            "machine": (
                "x86_64"
                if platform.machine().lower() in {"x86_64", "amd64"}
                else platform.machine().lower()
            ),
            "python": platform.python_version(),
            "libc": list(platform.libc_ver()),
        },
        "python_runtime": python_runtime,
        "sunfish_source": identity,
        "gemma_source_commit": GEMMA_SOURCE_COMMIT,
        "resolved_distributions": distributions,
        "files": [_file_record(bundle_root, name) for name in relative_files],
    }
    _write_json_atomic(bundle_root / BUNDLE_MANIFEST_NAME, payload)
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid offline bundle manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("offline bundle manifest must be an object")
    return payload


def verify_bundle(
    bundle_root: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    verify_file_hashes: bool = True,
    require_worker_runtime: bool = False,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    manifest = _load_manifest(bundle_root / BUNDLE_MANIFEST_NAME)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported offline bundle schema")
    if manifest.get("kind") != BUNDLE_KIND:
        raise ValueError("not a Sunfish TPU offline bundle")
    if manifest.get("network_policy") != "worker-no-egress":
        raise ValueError("offline bundle does not forbid worker egress")
    if manifest.get("transport_policy") != "gcloud-alpha-tpu-vm-iap-all-workers":
        raise ValueError("offline bundle does not require all-worker IAP transport")
    target = manifest.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("operating_system") != "linux"
        or target.get("machine") != "x86_64"
        or target.get("python") != standalone_runtime.RUNTIME_PYTHON_VERSION
        or not isinstance(target.get("minimum_glibc"), str)
    ):
        raise ValueError("offline bundle target differs from TPU worker contract")
    _version_tuple(target["minimum_glibc"])
    builder = manifest.get("builder")
    if (
        not isinstance(builder, Mapping)
        or builder.get("operating_system") != "linux"
        or builder.get("machine") != "x86_64"
        or builder.get("python") != standalone_runtime.RUNTIME_PYTHON_VERSION
        or not isinstance(builder.get("libc"), list)
        or len(builder["libc"]) != 2
        or any(not isinstance(item, str) for item in builder["libc"])
        or builder["libc"][0].lower() != "glibc"
        or not builder["libc"][1]
    ):
        raise ValueError("offline bundle was not built on the required Linux host")
    if manifest.get("gemma_source_commit") != GEMMA_SOURCE_COMMIT:
        raise ValueError("offline Gemma source commit differs from the audited pin")

    source = workspace_source_identity(bundle_root / "source")
    if manifest.get("sunfish_source") != source:
        raise ValueError("offline source identity differs from bundle manifest")
    if expected_commit is not None and source.get("git_commit") != expected_commit:
        raise ValueError("offline source commit differs from controller")
    if expected_tree is not None and source.get("source_tree_sha256") != expected_tree:
        raise ValueError("offline source tree differs from controller")

    lock = parse_resolved_lock(bundle_root / RESOLVED_LOCK_NAME)
    if manifest.get("resolved_distributions") != lock:
        raise ValueError("offline resolved lock differs from bundle manifest")
    wheel_entries = list((bundle_root / WHEELHOUSE_NAME).iterdir())
    if not wheel_entries or any(
        not path.is_file() or path.is_symlink() or path.suffix != ".whl"
        for path in wheel_entries
    ):
        raise ValueError("offline wheelhouse contains a non-wheel artifact")
    wheel_paths = {
        f"{WHEELHOUSE_NAME}/{path.name}"
        for path in wheel_entries
    }
    _verify_wheelhouse_matches_lock(wheel_entries, lock)
    if target["minimum_glibc"] != minimum_glibc_for_wheels(wheel_entries):
        raise ValueError("offline bundle glibc floor differs from wheel tags")
    expected_paths = {
        f"source/{RELEASE_MANIFEST_NAME}",
        RESOLVED_LOCK_NAME,
        standalone_runtime.RUNTIME_METADATA_NAME,
        standalone_runtime.expected_runtime_metadata()["archive"],
        *wheel_paths,
    }
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("offline bundle file inventory is missing")
    observed_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("invalid offline bundle file record")
        relative = record.get("path")
        size = record.get("size")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in observed_paths
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError("malformed offline bundle file record")
        path = bundle_root / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
            raise ValueError(f"offline bundle file size/type mismatch: {relative}")
        if verify_file_hashes and _sha256_file(path) != digest:
            raise ValueError(f"offline bundle file hash mismatch: {relative}")
        observed_paths.add(relative)
    if observed_paths != expected_paths:
        raise ValueError("offline bundle inventory/file set mismatch")
    standalone_runtime.verify_runtime_artifacts(
        bundle_root, require_bundle_manifest=True
    )
    if require_worker_runtime:
        verify_worker_runtime_compatibility(manifest)
    return manifest


def verify_installed_environment(bundle_root: Path, python: Path) -> dict[str, str]:
    manifest = verify_bundle(bundle_root)
    expected = manifest["resolved_distributions"]
    observed = _installed_distributions(python)
    missing_or_changed = {
        name: {"expected": version, "observed": observed.get(name)}
        for name, version in expected.items()
        if observed.get(name) != version
    }
    unexpected = sorted(set(observed) - set(expected) - _BOOTSTRAP_DISTRIBUTIONS)
    if missing_or_changed or unexpected:
        raise RuntimeError(
            "installed offline environment differs from manifest: "
            f"mismatch={missing_or_changed}, unexpected={unexpected}"
        )
    return observed


def pack_bundle(bundle_root: Path, output: Path) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    output = output.resolve()
    try:
        output.relative_to(bundle_root)
    except ValueError:
        pass
    else:
        raise ValueError("offline bundle archive must be outside the bundle root")
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("offline bundle archive or sidecar already exists")
    verify_bundle(bundle_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(bundle_root.rglob("*")):
            relative = path.relative_to(bundle_root)
            arcname = Path(BUNDLE_DIRECTORY_NAME) / relative
            info = archive.gettarinfo(str(path), arcname.as_posix())
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as source:
                    archive.addfile(info, source)
            else:
                archive.addfile(info)
    digest = _sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {
        "archive": str(output),
        "sha256": digest,
        "sha256_file": str(sidecar),
        "size": output.stat().st_size,
    }


def read_archive_sidecar(archive: Path) -> str:
    sidecar = archive.with_name(archive.name + ".sha256")
    try:
        line = sidecar.read_text(encoding="ascii").strip()
    except OSError as error:
        raise ValueError(f"missing offline bundle SHA-256 sidecar: {sidecar}") from error
    parts = line.split()
    if len(parts) != 2 or not _SHA256.fullmatch(parts[0]) or parts[1] != archive.name:
        raise ValueError("invalid offline bundle SHA-256 sidecar")
    return parts[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-source")
    export.add_argument("--repository", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    lock = subparsers.add_parser("write-lock")
    lock.add_argument("--python", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--bundle-root", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle-root", type=Path, required=True)
    verify.add_argument("--expected-commit")
    verify.add_argument("--expected-tree")
    verify.add_argument("--skip-file-hashes", action="store_true")
    verify.add_argument("--require-worker-runtime", action="store_true")

    installed = subparsers.add_parser("verify-installed")
    installed.add_argument("--bundle-root", type=Path, required=True)
    installed.add_argument("--python", type=Path, required=True)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--bundle-root", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)

    sidecar = subparsers.add_parser("read-sidecar")
    sidecar.add_argument("--archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export-source":
            result: Any = export_release_source(args.repository, args.output)
        elif args.command == "write-lock":
            result = write_resolved_lock(args.python, args.output)
        elif args.command == "create-manifest":
            result = create_bundle_manifest(args.bundle_root)
        elif args.command == "verify":
            result = verify_bundle(
                args.bundle_root,
                expected_commit=args.expected_commit,
                expected_tree=args.expected_tree,
                verify_file_hashes=not args.skip_file_hashes,
                require_worker_runtime=args.require_worker_runtime,
            )
        elif args.command == "verify-installed":
            result = verify_installed_environment(args.bundle_root, args.python)
        elif args.command == "pack":
            result = pack_bundle(args.bundle_root, args.output)
        elif args.command == "read-sidecar":
            result = {"sha256": read_archive_sidecar(args.archive)}
        else:  # pragma: no cover - argparse owns the command set.
            raise AssertionError(args.command)
    except (FileExistsError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"sunfish-offline-bundle: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
