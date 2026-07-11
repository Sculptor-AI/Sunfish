"""Initialize and validate a distributed Sunfish JAX environment.

The ordering in this module is load-bearing: ``jax.distributed.initialize()``
is attempted immediately after importing :mod:`jax`, before any call that can
initialize a backend.  TPU entrypoints import this module before Kauldron,
Orbax, or ``jax.numpy`` and share :func:`initialize_distributed_jax`.

Local CPU/GPU development remains possible.  With ``require_distributed``
false, failed environment auto-detection is a warning and backend inspection
continues as a single process.  TPU and pod launchers always require it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal["pass", "warn", "fail"]
_GCS_URI = re.compile(r"^gs://([^/]+)(?:/(.*))?$")
_REQUIRED_DISTRIBUTIONS = (
    "jax",
    "jaxlib",
    "libtpu",
    "gemma",
    "hackable-diffusion",
    "kauldron",
    "orbax-checkpoint",
    "grain",
    "etils",
    "google-cloud-storage",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def validate_gcs_uri(uri: str) -> tuple[str, str]:
    """Return bucket and prefix for a non-root GCS work directory."""
    match = _GCS_URI.fullmatch(uri)
    if match is None or not match.group(1):
        raise ValueError("GCS workdir must use gs://bucket/prefix syntax")
    bucket, prefix = match.group(1), (match.group(2) or "").strip("/")
    if not prefix:
        raise ValueError("GCS workdir must include a project-specific prefix")
    return bucket, prefix


def _package_checks(*, require_tpu: bool) -> list[Check]:
    checks: list[Check] = []
    for distribution in _REQUIRED_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                Check(
                    f"package:{distribution}",
                    "fail" if require_tpu else "warn",
                    "not installed",
                )
            )
        else:
            checks.append(Check(f"package:{distribution}", "pass", version))
    return checks


def initialize_distributed_jax(*, require_distributed: bool) -> tuple[Any, Check]:
    """Import JAX and initialize its process cluster before backend access.

    Importing :mod:`jax` itself is safe.  Importing backend-adjacent packages
    or asking JAX for devices before this call is not safe on a pod.
    """
    jax = importlib.import_module("jax")
    distributed = jax.distributed
    is_initialized = getattr(distributed, "is_initialized", lambda: False)
    if bool(is_initialized()):
        return jax, Check(
            "jax-distributed-initialize",
            "pass",
            "already initialized before backend inspection",
        )

    try:
        distributed.initialize()
    except Exception as error:
        detail = f"auto-initialize unavailable: {error}"
        if require_distributed:
            raise RuntimeError(detail) from error
        return jax, Check(
            "jax-distributed-initialize",
            "warn",
            f"{detail}; continuing in single-process development mode",
        )
    return jax, Check(
        "jax-distributed-initialize",
        "pass",
        "initialized before backend inspection",
    )


def _mismatch_status(require_tpu: bool) -> Status:
    return "fail" if require_tpu else "warn"


def _topology_checks(
    jax: Any,
    jnp: Any,
    *,
    require_tpu: bool,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> list[Check]:
    """Inspect the initialized global topology and execute a real psum."""
    checks: list[Check] = []
    try:
        devices = list(jax.devices())
        local_devices = list(jax.local_devices())
        process_count = int(jax.process_count())
        process_index = int(jax.process_index())
    except Exception as error:
        return [Check("jax-runtime", "fail", f"topology inspection failed: {error}")]

    platforms = sorted({str(device.platform).lower() for device in devices})
    kinds = sorted({str(getattr(device, "device_kind", "unknown")) for device in devices})
    topology_detail = (
        f"global_devices={len(devices)}, local_devices={len(local_devices)}, "
        f"processes={process_count}, process_index={process_index}, "
        f"platforms={platforms}, kinds={kinds}"
    )
    if require_tpu and (not devices or platforms != ["tpu"]):
        checks.append(Check("jax-tpu-topology", "fail", topology_detail))
    elif platforms == ["tpu"]:
        checks.append(Check("jax-tpu-topology", "pass", topology_detail))
    else:
        checks.append(Check("jax-tpu-topology", "warn", topology_detail))

    if expected_devices > 0 and len(devices) != expected_devices:
        checks.append(
            Check(
                "jax-global-device-count",
                _mismatch_status(require_tpu),
                f"found {len(devices)}; expected {expected_devices}",
            )
        )
    elif not devices:
        checks.append(Check("jax-global-device-count", "fail", "no devices found"))
    else:
        checks.append(Check("jax-global-device-count", "pass", str(len(devices))))

    if expected_processes > 0 and process_count != expected_processes:
        checks.append(
            Check(
                "jax-process-count",
                _mismatch_status(require_tpu),
                f"found {process_count}; expected {expected_processes}",
            )
        )
    elif process_count < 1:
        checks.append(Check("jax-process-count", "fail", str(process_count)))
    else:
        checks.append(Check("jax-process-count", "pass", str(process_count)))

    if not 0 <= process_index < process_count:
        checks.append(
            Check(
                "jax-process-index",
                "fail",
                f"process_index={process_index} outside [0, {process_count})",
            )
        )
    else:
        checks.append(Check("jax-process-index", "pass", str(process_index)))

    device_process_indices = [int(device.process_index) for device in devices]
    unique_process_indices = sorted(set(device_process_indices))
    expected_indices = list(range(process_count))
    if unique_process_indices != expected_indices:
        checks.append(
            Check(
                "jax-unique-process-indices",
                "fail",
                f"devices report {unique_process_indices}; expected {expected_indices}",
            )
        )
    else:
        checks.append(
            Check(
                "jax-unique-process-indices",
                "pass",
                json.dumps(unique_process_indices),
            )
        )

    devices_per_process = Counter(device_process_indices)
    metadata_local_count = devices_per_process.get(process_index, 0)
    local_count_ok = bool(local_devices) and len(local_devices) == metadata_local_count
    if expected_local_devices > 0:
        local_count_ok = local_count_ok and len(local_devices) == expected_local_devices
    local_detail = (
        f"local={len(local_devices)}, global-metadata={dict(sorted(devices_per_process.items()))}"
    )
    if expected_local_devices > 0:
        local_detail += f", expected-local={expected_local_devices}"
    checks.append(
        Check(
            "jax-local-device-count",
            "pass" if local_count_ok else _mismatch_status(require_tpu),
            local_detail,
        )
    )

    try:
        if not local_devices:
            raise ValueError("no local devices available for collective")
        # All processes execute the same pmap.  In multi-process JAX the named
        # axis spans every global device, so this is a real cross-host psum.
        psum = jax.pmap(
            lambda value: jax.lax.psum(value, "sunfish_devices"),
            axis_name="sunfish_devices",
        )
        collective = psum(jnp.ones((len(local_devices),), dtype=jnp.int32))
        collective.block_until_ready()
        observed = int(collective[0])
        expected = len(devices)
        if observed != expected:
            raise ValueError(f"psum returned {observed}; expected {expected}")
    except Exception as error:
        checks.append(Check("jax-global-psum", "fail", str(error)))
    else:
        checks.append(
            Check(
                "jax-global-psum",
                "pass",
                f"one from every global device summed to {observed}",
            )
        )
    return checks


def _jax_checks(
    *,
    require_tpu: bool,
    require_distributed: bool,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> list[Check]:
    try:
        jax, initialization = initialize_distributed_jax(
            require_distributed=require_distributed
        )
    except Exception as error:
        return [Check("jax-distributed-initialize", "fail", str(error))]

    # This import is deliberately after distributed initialization.
    try:
        jnp = importlib.import_module("jax.numpy")
    except Exception as error:
        return [initialization, Check("jax-runtime", "fail", f"import failed: {error}")]
    return [
        initialization,
        *_topology_checks(
            jax,
            jnp,
            require_tpu=require_tpu,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        ),
    ]


def _gcs_read_check(uri: str) -> Check:
    try:
        from google.cloud import storage

        bucket_name, prefix = validate_gcs_uri(uri)
        client = storage.Client()
        # This is intentionally read-only. The checkpoint smoke performs the
        # required distributed write/restore validation when explicitly run.
        next(iter(client.list_blobs(bucket_name, prefix=prefix, max_results=1)), None)
    except Exception as error:
        return Check("gcs-read", "fail", str(error))
    return Check("gcs-read", "pass", f"read/list access to {uri}")


def run_preflight(
    *,
    require_tpu: bool,
    expected_devices: int,
    gcs_workdir: str | None,
    require_gcs: bool,
    probe_gcs_read: bool,
    require_distributed: bool = False,
    expected_processes: int = 0,
    expected_local_devices: int = 0,
) -> list[Check]:
    checks = [
        Check(
            "python",
            "pass" if sys.version_info >= (3, 12) else "fail",
            sys.version.split()[0],
        )
    ]
    checks.extend(_package_checks(require_tpu=require_tpu))
    checks.extend(
        _jax_checks(
            require_tpu=require_tpu,
            require_distributed=require_distributed or require_tpu,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        )
    )

    if gcs_workdir is None:
        checks.append(
            Check(
                "gcs-workdir",
                "fail" if require_gcs else "warn",
                "not supplied; use --gcs-workdir gs://bucket/sunfish/run-prefix",
            )
        )
    else:
        try:
            bucket, prefix = validate_gcs_uri(gcs_workdir)
        except ValueError as error:
            checks.append(Check("gcs-workdir", "fail", str(error)))
        else:
            checks.append(Check("gcs-workdir", "pass", f"bucket={bucket}, prefix={prefix}"))
            if probe_gcs_read:
                checks.append(_gcs_read_check(gcs_workdir))
    return checks


def report(checks: list[Check]) -> dict[str, object]:
    counts = {
        status: sum(check.status == status for check in checks)
        for status in ("pass", "warn", "fail")
    }
    return {
        "ready": counts["fail"] == 0,
        "summary": counts,
        "checks": [asdict(check) for check in checks],
    }


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--require-distributed", action="store_true")
    parser.add_argument(
        "--expected-devices",
        type=int,
        default=_environment_int("EXPECTED_TPU_DEVICES", 8),
        help="global JAX device count; use 0 to disable the exact-count gate",
    )
    parser.add_argument(
        "--expected-processes",
        type=int,
        default=_environment_int("EXPECTED_TPU_PROCESSES", 0),
        help="JAX process count; use 0 when the allocation host count is unknown",
    )
    parser.add_argument(
        "--expected-local-devices",
        type=int,
        default=_environment_int("EXPECTED_LOCAL_TPU_DEVICES", 0),
        help="devices owned by this process; use 0 to derive from global metadata",
    )
    parser.add_argument("--gcs-workdir", help="gs://bucket/project-prefix")
    parser.add_argument("--require-gcs", action="store_true")
    parser.add_argument(
        "--probe-gcs-read",
        action="store_true",
        help="perform a read-only authenticated list request under --gcs-workdir",
    )
    args = parser.parse_args()
    for name in ("expected_devices", "expected_processes", "expected_local_devices"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if args.probe_gcs_read and not args.gcs_workdir:
        parser.error("--probe-gcs-read requires --gcs-workdir")

    payload = report(
        run_preflight(
            require_tpu=args.require_tpu,
            require_distributed=args.require_distributed,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
            gcs_workdir=args.gcs_workdir,
            require_gcs=args.require_gcs,
            probe_gcs_read=args.probe_gcs_read,
        )
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["ready"] else 1)


if __name__ == "__main__":
    main()
