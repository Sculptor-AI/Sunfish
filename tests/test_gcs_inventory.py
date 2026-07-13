import copy
import unittest
from unittest import mock

from sunfish_tpu.gcs_inventory import (
    build_gcs_inventory,
    gcs_inventory_from_objects,
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


if __name__ == "__main__":
    unittest.main()
