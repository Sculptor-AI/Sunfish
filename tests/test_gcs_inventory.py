import copy
import unittest
from unittest import mock

from sunfish_tpu.gcs_inventory import (
    build_gcs_inventory,
    compare_gcs_inventory_contents,
    gcs_inventory_from_objects,
    probe_gcs_object_reads,
    validate_gcs_inventory,
    verify_live_gcs_inventory,
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
        return self.blobs


class GcsInventoryTests(unittest.TestCase):
    def test_inventory_is_sorted_and_binds_generation_size_and_crc(self):
        client = _Client(
            [
                _Blob("checkpoints/seed/z", 3, 20, "crc-z"),
                _Blob("checkpoints/seed/a", 2, 10, "crc-a"),
            ]
        )
        payload = build_gcs_inventory(
            "gs://bucket/checkpoints/seed", client=client
        )
        self.assertEqual(client.calls, [("bucket", "checkpoints/seed/")])
        self.assertEqual([row["name"] for row in payload["objects"]], ["a", "z"])
        self.assertEqual(payload["total_bytes"], 30)
        self.assertEqual(
            validate_gcs_inventory(
                payload, expected_uri="gs://bucket/checkpoints/seed/"
            ),
            payload,
        )

        changed = copy.deepcopy(payload)
        changed["objects"][0]["generation"] += 1
        with self.assertRaisesRegex(ValueError, "sha256"):
            validate_gcs_inventory(
                changed, expected_uri="gs://bucket/checkpoints/seed"
            )

    def test_directory_sidecar_with_shared_name_prefix_is_excluded(self):
        client = _Client([])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_gcs_inventory("gs://bucket/checkpoints/seed", client=client)
        self.assertEqual(client.calls, [("bucket", "checkpoints/seed/")])

    def test_live_verifier_detects_replaced_object(self):
        expected = build_gcs_inventory(
            "gs://bucket/checkpoints/seed",
            client=_Client([_Blob("checkpoints/seed/a", 1, 10, "crc")]),
        )
        changed = gcs_inventory_from_objects(
            "gs://bucket/checkpoints/seed",
            [{"name": "a", "generation": 2, "size": 10, "crc32c": "crc"}],
        )
        with (
            mock.patch(
                "sunfish_tpu.gcs_inventory.build_gcs_inventory",
                return_value=changed,
            ),
            self.assertRaisesRegex(RuntimeError, "inventory changed"),
        ):
            verify_live_gcs_inventory(
                "gs://bucket/checkpoints/seed", expected
            )

    def test_staged_content_match_ignores_generation_but_not_bytes(self):
        source = gcs_inventory_from_objects(
            "gs://public/checkpoint",
            [
                {"name": "a", "generation": 1, "size": 3, "crc32c": "abc="},
                {"name": "b", "generation": 2, "size": 4, "crc32c": "def="},
            ],
        )
        staged = gcs_inventory_from_objects(
            "gs://project/staged/checkpoint",
            [
                {"name": "a", "generation": 101, "size": 3, "crc32c": "abc="},
                {"name": "b", "generation": 102, "size": 4, "crc32c": "def="},
            ],
        )
        receipt = compare_gcs_inventory_contents(source, staged)
        self.assertTrue(receipt["matched"])
        self.assertEqual(receipt["object_count"], 2)
        self.assertEqual(receipt["total_bytes"], 7)
        self.assertEqual(receipt["source_inventory_sha256"], source["sha256"])
        self.assertEqual(receipt["staged_inventory_sha256"], staged["sha256"])

        changed = gcs_inventory_from_objects(
            "gs://project/staged/checkpoint",
            [
                {"name": "a", "generation": 101, "size": 3, "crc32c": "WRONG="},
                {"name": "b", "generation": 102, "size": 4, "crc32c": "def="},
            ],
        )
        with self.assertRaisesRegex(ValueError, "names/sizes/CRC32Cs"):
            compare_gcs_inventory_contents(source, changed)

    def test_bounded_read_probe_reads_one_byte_per_nonempty_object(self):
        payload = gcs_inventory_from_objects(
            "gs://project/staged/checkpoint",
            [
                {"name": "a", "generation": 1, "size": 3, "crc32c": "abc="},
                {"name": "empty", "generation": 2, "size": 0, "crc32c": "def="},
            ],
        )

        class FakePath:
            def __init__(self, raw):
                self.raw = raw

            def open(self, mode):
                self.assert_mode = mode
                from io import BytesIO

                return BytesIO(b"payload")

            def __str__(self):
                return self.raw

        self.assertEqual(
            probe_gcs_object_reads(
                "gs://project/staged/checkpoint",
                payload,
                path_factory=FakePath,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
