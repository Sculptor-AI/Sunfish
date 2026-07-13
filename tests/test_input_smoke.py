import unittest
import hashlib
import json

from sunfish_tpu.input_smoke import verify_evidence

SOURCE = {"git_commit": "c" * 40, "source_tree_sha256": "d" * 64}


def host(index, ids, *, manifest="a" * 64, payload_bytes=100):
    return {
        "schema_version": 1,
        "run_id": "input-smoke",
        "process_index": index,
        "process_count": 2,
        "manifest_sha256": manifest,
        "total_records": 6,
        "record_ids": ids,
        "record_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "wall_seconds": 1.0,
        "read_metrics": {
            "records_read": len(ids),
            "payload_bytes_read": payload_bytes if ids else 0,
        },
        "topology": {"ready": True},
        "sunfish_source": SOURCE,
    }


class InputSmokeEvidenceTests(unittest.TestCase):
    def test_disjoint_exhaustive_process_slices_pass(self):
        result = verify_evidence(
            [host(0, [0, 2, 4]), host(1, [1, 3, 5])],
            total_records=6,
            expected_processes=2,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["records_observed"], 6)
        self.assertEqual(result["aggregate_records_per_second"], 6.0)

    def test_overlap_and_missing_record_fail(self):
        first = host(0, [0, 2])
        second = host(1, [1, 2])
        first["total_records"] = second["total_records"] = 5
        result = verify_evidence(
            [first, second],
            total_records=5,
            expected_processes=2,
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("duplicate" in error for error in result["errors"]))
        self.assertTrue(any("missing" in error for error in result["errors"]))

    def test_manifest_or_read_counter_drift_fails(self):
        first = host(0, [0])
        second = host(1, [1], manifest="b" * 64)
        first["total_records"] = second["total_records"] = 2
        second["read_metrics"]["records_read"] = 0
        result = verify_evidence(
            [first, second], total_records=2, expected_processes=2
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("manifests" in error for error in result["errors"]))
        self.assertTrue(any("counter" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
