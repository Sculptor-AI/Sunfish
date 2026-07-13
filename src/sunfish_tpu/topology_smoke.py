"""All-host Stage-0.5 topology/collective evidence collector."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from sunfish_tpu.tpu_preflight import (
    _package_checks,
    _topology_checks,
    initialize_distributed_jax,
    report,
)
from sunfish_tpu.source_identity import (
    normalize_source_identity,
    require_launcher_run_id,
    source_identity_from_environment,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED_PASS_CHECKS = {
    "jax-distributed-initialize",
    "jax-tpu-topology",
    "jax-global-device-count",
    "jax-process-count",
    "jax-process-index",
    "jax-unique-process-indices",
    "jax-local-device-count",
    "jax-global-psum",
}


def verify_topology_evidence(
    hosts: Sequence[Mapping[str, Any]],
    *,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    """Verify every process published a passing view of one exact topology."""
    errors: list[str] = []
    if len(hosts) != expected_processes:
        errors.append(f"found {len(hosts)} host reports; expected {expected_processes}")
    indices = [host.get("process_index") for host in hosts]
    if sorted(indices) != list(range(expected_processes)):
        errors.append(f"process indices are {sorted(indices)}")
    run_ids = {host.get("run_id") for host in hosts}
    if len(run_ids) != 1 or None in run_ids:
        errors.append(f"run IDs differ: {sorted(str(value) for value in run_ids)}")
    sources = [normalize_source_identity(host.get("sunfish_source")) for host in hosts]
    if any(source is None for source in sources) or len(set(sources)) != 1:
        errors.append("host source identities are missing or differ")
    for host in hosts:
        process = host.get("process_index")
        if host.get("schema_version") != 1:
            errors.append(f"process {process} has an unsupported schema")
        if host.get("process_count") != expected_processes:
            errors.append(f"process {process} reports the wrong process count")
        if host.get("global_device_count") != expected_devices:
            errors.append(f"process {process} reports the wrong global device count")
        if host.get("local_device_count") != expected_local_devices:
            errors.append(f"process {process} reports the wrong local device count")
        preflight = host.get("preflight")
        if not isinstance(preflight, Mapping) or preflight.get("ready") is not True:
            errors.append(f"process {process} preflight did not pass")
            continue
        checks = {
            item.get("name"): item.get("status")
            for item in preflight.get("checks", ())
            if isinstance(item, Mapping)
        }
        for name in sorted(_REQUIRED_PASS_CHECKS):
            if checks.get(name) != "pass":
                errors.append(f"process {process} check {name}={checks.get(name)!r}")
    return {
        "schema_version": 1,
        "gate": 1,
        "run_id": next(iter(run_ids), None),
        "passed": not errors,
        "errors": errors,
        "expected": {
            "global_device_count": expected_devices,
            "process_count": expected_processes,
            "local_device_count": expected_local_devices,
        },
        "hosts": list(hosts),
        "sunfish_source": hosts[0].get("sunfish_source") if hosts else None,
    }


def _write_immutable(path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"immutable topology evidence changed at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def run_topology_smoke(
    *,
    output_dir: str,
    run_id: str,
    expected_devices: int,
    expected_processes: int,
    expected_local_devices: int,
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run ID")
    if min(expected_devices, expected_processes, expected_local_devices) <= 0:
        raise ValueError("all expected topology counts must be positive")
    require_launcher_run_id(run_id)

    # No backend-adjacent import may move above this call.
    jax, initialization = initialize_distributed_jax(require_distributed=True)
    import jax.numpy as jnp
    from etils import epath
    from jax.experimental import multihost_utils

    checks = [
        initialization,
        *_package_checks(require_tpu=True),
        *_topology_checks(
            jax,
            jnp,
            require_tpu=True,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        ),
    ]
    preflight = report(checks)
    host = {
        "schema_version": 1,
        "gate": 1,
        "run_id": run_id,
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "global_device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "preflight": preflight,
        "sunfish_source": source_identity_from_environment(required=True),
    }
    root = epath.Path(output_dir) / run_id
    host_path = root / f"host-{int(jax.process_index()):05d}.json"
    _write_immutable(host_path, host)
    multihost_utils.sync_global_devices(f"sunfish-topology-hosts-{run_id}")

    summary = None
    if int(jax.process_index()) == 0:
        hosts = [
            json.loads((root / f"host-{process:05d}.json").read_text())
            for process in range(expected_processes)
        ]
        summary = verify_topology_evidence(
            hosts,
            expected_devices=expected_devices,
            expected_processes=expected_processes,
            expected_local_devices=expected_local_devices,
        )
        _write_immutable(root / "summary.json", summary)
    multihost_utils.sync_global_devices(f"sunfish-topology-summary-{run_id}")
    if not preflight["ready"]:
        raise RuntimeError("local topology preflight failed")
    return summary if summary is not None else host


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        payload = run_topology_smoke(
            output_dir=args.output_dir,
            run_id=args.run_id,
            expected_devices=args.expected_devices,
            expected_processes=args.expected_processes,
            expected_local_devices=args.expected_local_devices,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"sunfish-topology-smoke: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("passed", payload.get("preflight", {}).get("ready")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
