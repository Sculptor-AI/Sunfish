import copy
import unittest

from sunfish_tpu.calibration_data_inventory import (
    build_calibration_data_inventory,
    validate_calibration_data_inventory,
    verify_live_calibration_data_inventory,
)


class _Blob:
    def __init__(self, name, generation, size, crc32c):
        self.name = name
        self.generation = generation
        self.size = size
        self.crc32c = crc32c


class _Client:
    def __init__(self, blobs):
        self.blobs = blobs
        self.calls = []

    def list_blobs(self, bucket, *, prefix):
        self.calls.append((bucket, prefix))
        return list(self.blobs)


def _manifest():
    return {
        "manifest_version": 1,
        "shards": [
            {
                "bucket": "code_completion",
                "bin": "code_completion.bin",
                "idx": "code_completion.idx",
                "records": 2,
                "tokens": 3,
                "sha256_bin": "a" * 64,
                "sha256_idx": "b" * 64,
            }
        ],
    }


def _blobs(*, bin_generation=10, bin_crc="bin-crc"):
    prefix = "calibration/reviewed/"
    return [
        _Blob(prefix + "manifest.json", 1, 900, "manifest-crc"),
        _Blob(prefix + "source-receipt.json", 2, 700, "receipt-crc"),
        _Blob(prefix + "code_completion.bin", bin_generation, 12, bin_crc),
        _Blob(prefix + "code_completion.idx", 11, 16, "idx-crc"),
        _Blob(prefix + "unreferenced.tmp", 12, 99, "extra-crc"),
    ]


class CalibrationDataInventoryTests(unittest.TestCase):
    def test_inventory_binds_only_manifest_artifacts_and_declared_hashes(self):
        client = _Client(_blobs())
        inventory = build_calibration_data_inventory(
            "gs://bucket/calibration/reviewed", _manifest(), client=client
        )
        self.assertEqual(client.calls, [("bucket", "calibration/reviewed/")])
        self.assertEqual(
            [item["name"] for item in inventory["artifacts"]],
            ["code_completion.bin", "code_completion.idx"],
        )
        self.assertEqual(
            [item["sha256"] for item in inventory["artifacts"]],
            ["a" * 64, "b" * 64],
        )
        self.assertEqual(inventory["artifact_count"], 2)
        self.assertEqual(inventory["total_bytes"], 28)
        self.assertEqual(
            validate_calibration_data_inventory(_manifest(), inventory), inventory
        )

    def test_equal_length_generation_crc_mutation_is_rejected(self):
        expected = build_calibration_data_inventory(
            "gs://bucket/calibration/reviewed",
            _manifest(),
            client=_Client(_blobs()),
        )
        changed = _Client(_blobs(bin_generation=99, bin_crc="changed-crc"))
        with self.assertRaisesRegex(RuntimeError, "inventory changed"):
            verify_live_calibration_data_inventory(
                "gs://bucket/calibration/reviewed",
                _manifest(),
                expected,
                client=changed,
            )

    def test_manifest_declared_hash_is_part_of_canonical_inventory(self):
        inventory = build_calibration_data_inventory(
            "gs://bucket/calibration/reviewed",
            _manifest(),
            client=_Client(_blobs()),
        )
        changed_manifest = _manifest()
        changed_manifest["shards"][0]["sha256_bin"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_calibration_data_inventory(changed_manifest, inventory)

    def test_size_or_artifact_set_tampering_is_rejected(self):
        inventory = build_calibration_data_inventory(
            "gs://bucket/calibration/reviewed",
            _manifest(),
            client=_Client(_blobs()),
        )
        wrong_size = copy.deepcopy(inventory)
        wrong_size["artifacts"][0]["size"] = 16
        with self.assertRaises(ValueError):
            validate_calibration_data_inventory(_manifest(), wrong_size)

        missing = copy.deepcopy(inventory)
        missing["artifacts"].pop()
        with self.assertRaises(ValueError):
            validate_calibration_data_inventory(_manifest(), missing)


if __name__ == "__main__":
    unittest.main()
