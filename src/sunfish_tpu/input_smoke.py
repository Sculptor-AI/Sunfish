"""Stage-0.5 process-disjoint GCS input proof using the production loader.

All TPU hosts run this command. Distributed JAX initializes before importing
Grain, Kauldron, or the GCS data implementation. Each host then constructs the
same ``SunfishData`` pipeline, exercises Kauldron's pinned process slice before
shuffle, and writes immutable record-ID/read-throughput evidence. Process 0
merges every host artifact and fails unless coverage is exhaustive and pairwise
disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sunfish_tpu.tpu_preflight import (
    _topology_checks,
    initialize_distributed_jax,
    report,
)
from sunfish_tpu.training.spec import HarnessConfig
from sunfish_tpu.source_identity import (
    normalize_source_identity,
    require_launcher_run_id,
    source_identity_from_environment,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def verify_evidence(
    evidence: list[dict[str, Any]], *, total_records: int, expected_processes: int
) -> dict[str, Any]:
    """Dependency-free exact-coverage verifier for per-host evidence."""
    errors: list[str] = []
    if expected_processes <= 0:
        raise ValueError("expected_processes must be positive")
    if total_records <= 0:
        errors.append("dataset has no records")
    if len(evidence) != expected_processes:
        errors.append(
            f"found {len(evidence)} host artifacts, expected {expected_processes}"
        )
    process_indices = [item.get("process_index") for item in evidence]
    if not all(isinstance(index, int) for index in process_indices):
        errors.append(f"non-integer process indices: {process_indices}")
        normalized_process_indices: list[int] = []
    else:
        normalized_process_indices = sorted(process_indices)
    if normalized_process_indices != list(range(expected_processes)):
        errors.append(
            f"process indices are {normalized_process_indices}, expected {list(range(expected_processes))}"
        )

    owners: dict[int, int] = {}
    duplicate_ids: list[int] = []
    manifests = set()
    run_ids = set()
    sources = []
    for item in evidence:
        process_index = int(item.get("process_index", -1))
        run_ids.add(item.get("run_id"))
        manifests.add(item.get("manifest_sha256"))
        sources.append(normalize_source_identity(item.get("sunfish_source")))
        if item.get("schema_version") != 1:
            errors.append(f"process {process_index} has an unsupported schema")
        if item.get("process_count") != expected_processes:
            errors.append(f"process {process_index} reports the wrong process count")
        if item.get("total_records") != total_records:
            errors.append(f"process {process_index} reports the wrong record total")
        topology = item.get("topology")
        if not isinstance(topology, dict) or topology.get("ready") is not True:
            errors.append(f"process {process_index} topology did not pass")
        record_ids = item.get("record_ids")
        if not isinstance(record_ids, list):
            errors.append(f"process {process_index} record_ids is not a list")
            continue
        if len(record_ids) != len(set(record_ids)):
            errors.append(f"process {process_index} read a duplicate record")
        expected_record_hash = hashlib.sha256(
            json.dumps(record_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if item.get("record_ids_sha256") != expected_record_hash:
            errors.append(f"process {process_index} record-ID hash differs")
        for record_id in record_ids:
            if not isinstance(record_id, int):
                errors.append(f"process {process_index} emitted non-integer record id")
                continue
            if record_id in owners:
                duplicate_ids.append(record_id)
            else:
                owners[record_id] = process_index
        metrics = item.get("read_metrics", {})
        if metrics.get("records_read") != len(record_ids):
            errors.append(f"process {process_index} read counter differs from evidence")
        if record_ids and metrics.get("payload_bytes_read", 0) <= 0:
            errors.append(f"process {process_index} recorded no payload bytes")
        wall_seconds = item.get("wall_seconds")
        if (
            not isinstance(wall_seconds, (int, float))
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) <= 0.0
        ):
            errors.append(f"process {process_index} has invalid wall time")

    if len(run_ids) != 1 or None in run_ids:
        errors.append(f"host run IDs differ: {sorted(map(str, run_ids))}")
    if len(manifests) != 1 or None in manifests:
        errors.append(f"hosts used different manifests: {sorted(map(str, manifests))}")
    if any(source is None for source in sources) or len(set(sources)) != 1:
        errors.append("host source identities are missing or differ")
    if duplicate_ids:
        errors.append(f"cross-process duplicate record ids: {sorted(set(duplicate_ids))[:20]}")
    expected_ids = set(range(total_records))
    observed_ids = set(owners)
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if missing:
        errors.append(f"missing record ids: {missing[:20]}")
    if extra:
        errors.append(f"out-of-range record ids: {extra[:20]}")

    total_bytes = sum(
        int(item.get("read_metrics", {}).get("payload_bytes_read", 0))
        for item in evidence
    )
    wall_seconds = max(
        (float(item.get("wall_seconds", 0.0)) for item in evidence), default=0.0
    )
    return {
        "schema_version": 1,
        "run_id": next(iter(run_ids), None),
        "ready": not errors,
        "errors": errors,
        "expected_processes": expected_processes,
        "total_records": total_records,
        "records_observed": len(observed_ids),
        "payload_bytes_read": total_bytes,
        "max_host_wall_seconds": wall_seconds,
        "aggregate_records_per_second": (
            len(observed_ids) / wall_seconds if wall_seconds > 0.0 else 0.0
        ),
        "aggregate_payload_mib_per_second": (
            total_bytes / (1024.0 * 1024.0) / wall_seconds
            if wall_seconds > 0.0
            else 0.0
        ),
        "hosts": evidence,
        "sunfish_source": evidence[0].get("sunfish_source") if evidence else None,
    }


def run_input_smoke(
    *,
    config_path: Path,
    output_dir: str,
    run_id: str,
    max_total_records: int,
    allow_non_tpu: bool = False,
) -> dict[str, Any]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    if not 1 <= max_total_records <= 100_000:
        raise ValueError("max_total_records must be in [1, 100000]")
    spec = HarnessConfig.load(config_path)
    require_launcher_run_id(run_id, required=not allow_non_tpu)

    # No Grain/Kauldron/GCS/JAX backend import may move above this call.
    jax, initialization = initialize_distributed_jax(
        require_distributed=not allow_non_tpu
    )
    import time

    import jax.numpy as jnp
    from etils import epath
    from kauldron import random as kd_random
    from jax.experimental import multihost_utils

    from sunfish_tpu.training.data import SunfishData

    expected_devices = 0 if allow_non_tpu else spec.topology.expected_devices
    expected_processes = 1 if allow_non_tpu else spec.topology.expected_processes
    expected_local_devices = (
        0 if allow_non_tpu else spec.topology.expected_local_devices
    )
    topology = report(
        [
            initialization,
            *_topology_checks(
                jax,
                jnp,
                require_tpu=not allow_non_tpu,
                expected_devices=expected_devices,
                expected_processes=expected_processes,
                expected_local_devices=expected_local_devices,
            ),
        ]
    )
    if not topology["ready"]:
        raise RuntimeError(f"distributed topology failed: {json.dumps(topology)}")

    data_root = epath.Path(spec.data.directory)
    manifest_bytes = (data_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    total_records = int(manifest.get("total_records", -1))
    if not 1 <= total_records <= max_total_records:
        raise ValueError(
            f"input smoke requires a tiny dataset with 1..{max_total_records} records; got {total_records}"
        )

    pipeline = SunfishData(
        directory=spec.data.directory,
        expected_manifest_sha256=spec.data.manifest_sha256,
        verify_shard_hashes=spec.data.verify_shard_hashes,
        prompt_length=spec.data.prompt_length,
        canvas_size=spec.data.canvas_size,
        num_canvases=spec.data.num_canvases,
        vocab_size=spec.model.vocab_size,
        pad_token=spec.data.pad_token,
        eos_token=spec.data.eos_token,
        batch_size=None,
        seed=spec.run.seed,
        shuffle=spec.data.shuffle,
        num_epochs=1,
        batch_drop_remainder=False,
        num_workers=0,
        shard_by_process=True,
    )
    rng = kd_random.PRNGKey(spec.run.seed).fold_in("kmix")
    process_dataset = pipeline.ds_for_current_process(rng)
    started = time.perf_counter()
    record_ids = [
        int(process_dataset[index]["record_id"])
        for index in range(len(process_dataset))
    ]
    wall_seconds = time.perf_counter() - started
    evidence = {
        "schema_version": 1,
        "run_id": run_id,
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "manifest_sha256": pipeline.data_source.manifest_sha256,
        "total_records": total_records,
        "record_ids": record_ids,
        "record_ids_sha256": hashlib.sha256(
            json.dumps(record_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "wall_seconds": wall_seconds,
        "read_metrics": pipeline.data_source.read_metrics,
        "topology": topology,
        "sunfish_source": source_identity_from_environment(
            required=not allow_non_tpu
        ),
    }

    destination = epath.Path(output_dir) / run_id
    destination.mkdir(parents=True, exist_ok=True)
    host_path = destination / f"host{jax.process_index()}.json"
    if host_path.exists():
        raise FileExistsError(f"immutable input evidence already exists: {host_path}")
    host_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    multihost_utils.sync_global_devices(f"sunfish-input-smoke-write-{run_id}")

    summary_path = destination / "summary.json"
    if int(jax.process_index()) == 0:
        all_evidence = [
            json.loads((destination / f"host{index}.json").read_text())
            for index in range(expected_processes)
        ]
        summary = verify_evidence(
            all_evidence,
            total_records=total_records,
            expected_processes=expected_processes,
        )
        summary["schema_version"] = 1
        summary["run_id"] = run_id
        summary["destination"] = str(destination)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    multihost_utils.sync_global_devices(f"sunfish-input-smoke-summary-{run_id}")
    summary = json.loads(summary_path.read_text())
    if not summary.get("ready"):
        raise RuntimeError(f"process-disjoint input proof failed: {json.dumps(summary)}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-total-records", type=int, default=10_000)
    parser.add_argument("--allow-non-tpu", action="store_true")
    args = parser.parse_args(argv)
    summary = run_input_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        run_id=args.run_id,
        max_total_records=args.max_total_records,
        allow_non_tpu=args.allow_non_tpu,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
