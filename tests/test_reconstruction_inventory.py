import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sunfish_tpu.reconstruction_inventory import (
    build_artifact_inventory,
    paths_for_process,
    validate_artifact_inventory,
    verify_live_artifact_inventory,
)


FIELDS = (
    "shared_pre_router_residual",
    "topk_indices",
    "final_scaled_topk_weights",
)
BUCKETS = ("prefill/code_completion", "denoise_high/code_completion")


def write_manifest(root: Path, process_index: int, artifact_id: str, bucket: str):
    host = root / f"host-{process_index:05d}"
    host.mkdir(parents=True, exist_ok=True)
    fields = {}
    for name in FIELDS:
        payload = f"{artifact_id}:{name}".encode()
        path = host / f"{artifact_id}.{name}.bin"
        path.write_bytes(payload)
        fields[name] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "run_id": "calibration-v1",
        "calibration_run_sha256": "a" * 64,
        "process_index": process_index,
        "artifact_id": artifact_id,
        "bucket": bucket,
        "tokens": 3,
        "fields": fields,
    }
    (host / f"{artifact_id}.json").write_text(json.dumps(manifest))


class ReconstructionInventoryTests(unittest.TestCase):
    def test_inventory_binds_exact_manifest_set_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root, 0, "step-000000000-prefill", BUCKETS[0])
            write_manifest(root, 1, "step-000000000-denoise_high", BUCKETS[1])
            inventory = build_artifact_inventory(
                root,
                run_id="calibration-v1",
                calibration_run_sha256="a" * 64,
                expected_processes=2,
                allowed_buckets=BUCKETS,
                field_names=FIELDS,
            )
            validated = validate_artifact_inventory(
                inventory,
                root=root,
                run_id="calibration-v1",
                calibration_run_sha256="a" * 64,
                expected_processes=2,
                allowed_buckets=BUCKETS,
                field_names=FIELDS,
            )
            verify_live_artifact_inventory(root, validated)
            self.assertEqual(len(paths_for_process(root, validated, 1)), 1)
            write_manifest(root, 1, "step-000000001-prefill", BUCKETS[0])
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                verify_live_artifact_inventory(root, validated)

    def test_inventory_rejects_manifest_lineage_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_manifest(root, 0, "step-000000000-prefill", BUCKETS[0])
            path = next((root / "host-00000").glob("*.json"))
            payload = json.loads(path.read_text())
            payload["calibration_run_sha256"] = "b" * 64
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "lineage"):
                build_artifact_inventory(
                    root,
                    run_id="calibration-v1",
                    calibration_run_sha256="a" * 64,
                    expected_processes=1,
                    allowed_buckets=BUCKETS,
                    field_names=FIELDS,
                )


if __name__ == "__main__":
    unittest.main()
