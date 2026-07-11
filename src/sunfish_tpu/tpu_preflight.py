"""Validate a Sunfish JAX environment before allocating a real training run.

This lives in the separate ``sunfish_tpu`` package so core checkpoint and
selection utilities remain dependency-free. Optional JAX/Cloud imports stay
lazy to keep ``--help`` and pure validation usable off-TPU.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["pass", "warn", "fail"]
_GCS_URI = re.compile(r"^gs://([^/]+)(?:/(.*))?$")
_REQUIRED_DISTRIBUTIONS = (
    "jax",
    "gemma",
    "hackable-diffusion",
    "kauldron",
    "orbax-checkpoint",
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


def _package_checks() -> list[Check]:
    checks: list[Check] = []
    for distribution in _REQUIRED_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            checks.append(Check(f"package:{distribution}", "fail", "not installed"))
        else:
            checks.append(Check(f"package:{distribution}", "pass", version))
    return checks


def _jax_checks(*, require_tpu: bool, expected_devices: int) -> list[Check]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as error:  # JAX can fail during backend initialization.
        return [Check("jax-runtime", "fail", f"import failed: {error}")]

    checks: list[Check] = []
    try:
        devices = jax.devices()
        platforms = sorted({str(device.platform).lower() for device in devices})
        kinds = sorted({str(getattr(device, "device_kind", "unknown")) for device in devices})
        runtime_detail = (
            f"devices={len(devices)}, processes={jax.process_count()}, "
            f"platforms={platforms}, kinds={kinds}"
        )
        if require_tpu and (not devices or platforms != ["tpu"]):
            checks.append(Check("jax-tpu-topology", "fail", runtime_detail))
        elif platforms == ["tpu"]:
            checks.append(Check("jax-tpu-topology", "pass", runtime_detail))
        else:
            checks.append(Check("jax-tpu-topology", "warn", runtime_detail))

        if expected_devices > 0 and len(devices) != expected_devices:
            checks.append(
                Check(
                    "jax-device-count",
                    "fail" if require_tpu else "warn",
                    f"found {len(devices)}; expected {expected_devices}",
                )
            )
        else:
            checks.append(Check("jax-device-count", "pass", str(len(devices))))

        result = jnp.add(jnp.asarray(1), jnp.asarray(1)).block_until_ready()
        if int(result) != 2:
            raise ValueError(f"unexpected result {result}")
        checks.append(Check("jax-compiled-operation", "pass", "1 + 1 = 2"))
    except Exception as error:
        checks.append(Check("jax-runtime", "fail", f"device operation failed: {error}"))
    return checks


def _gcs_read_check(uri: str) -> Check:
    try:
        from google.cloud import storage

        bucket_name, prefix = validate_gcs_uri(uri)
        client = storage.Client()
        # This is intentionally read-only. The separate checkpoint smoke test
        # performs the required write/restore validation when explicitly run.
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
) -> list[Check]:
    checks = [
        Check(
            "python",
            "pass" if sys.version_info >= (3, 12) else "fail",
            sys.version.split()[0],
        )
    ]
    checks.extend(_package_checks())
    checks.extend(_jax_checks(require_tpu=require_tpu, expected_devices=expected_devices))

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
    counts = {status: sum(check.status == status for check in checks) for status in ("pass", "warn", "fail")}
    return {
        "ready": counts["fail"] == 0,
        "summary": counts,
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument(
        "--expected-devices",
        type=int,
        default=8,
        help="visible global JAX device count; use 0 to disable the exact-count gate",
    )
    parser.add_argument("--gcs-workdir", help="gs://bucket/project-prefix")
    parser.add_argument("--require-gcs", action="store_true")
    parser.add_argument(
        "--probe-gcs-read",
        action="store_true",
        help="perform a read-only authenticated list request under --gcs-workdir",
    )
    args = parser.parse_args()
    if args.expected_devices < 0:
        parser.error("--expected-devices cannot be negative")
    if args.probe_gcs_read and not args.gcs_workdir:
        parser.error("--probe-gcs-read requires --gcs-workdir")

    payload = report(
        run_preflight(
            require_tpu=args.require_tpu,
            expected_devices=args.expected_devices,
            gcs_workdir=args.gcs_workdir,
            require_gcs=args.require_gcs,
            probe_gcs_read=args.probe_gcs_read,
        )
    )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["ready"] else 1)


if __name__ == "__main__":
    main()
