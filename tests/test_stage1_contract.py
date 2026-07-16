import hashlib
import json
import unittest

from sunfish.calibration_records import (
    CALIBRATION_BUCKET_PERCENT,
    calibration_bucket_token_caps,
)
from sunfish_tpu.calibration_data_inventory import (
    calibration_data_inventory_from_objects,
)
from sunfish_tpu.stage1_contract import validate_calibration_source_contract


def _reviewed_shards():
    total_tokens = 75_006_144
    record_tokens = 768
    bucket_caps = calibration_bucket_token_caps(total_tokens)
    return [
        {
            "bucket": bucket,
            "source_id": f"hf://datasets/example/{bucket}",
            "source_revision": f"{index + 1:040x}",
            "bin": f"{bucket}.bin",
            "idx": f"{bucket}.idx",
            "sha256_bin": f"{index * 2 + 1:064x}",
            "sha256_idx": f"{index * 2 + 2:064x}",
            "records": (bucket_caps[bucket] + record_tokens - 1) // record_tokens,
            "tokens": bucket_caps[bucket],
            "record_tokens": record_tokens,
        }
        for index, bucket in enumerate(CALIBRATION_BUCKET_PERCENT)
    ]


def reviewed_inventory():
    manifest = {"shards": _reviewed_shards()}
    objects = []
    for index, shard in enumerate(manifest["shards"]):
        objects.extend(
            [
                {
                    "name": shard["bin"],
                    "generation": 100 + index * 2,
                    "size": shard["tokens"] * 4,
                    "crc32c": f"bin-crc-{index}",
                },
                {
                    "name": shard["idx"],
                    "generation": 101 + index * 2,
                    "size": shard["records"] * 8,
                    "crc32c": f"idx-crc-{index}",
                },
            ]
        )
    return calibration_data_inventory_from_objects(
        "gs://bucket/calibration/reviewed-v1", manifest, objects
    )


def reviewed_receipt_bytes(*, inventory=None):
    if inventory is None:
        inventory = reviewed_inventory()
    payload = {
        "schema_version": 2,
        "purpose": "stage-1-calibration-source-receipt",
        "approval_scope": "stage-1-router-calibration-promotion",
        "review_status": "approved",
        "promotion_allowed": True,
        "source_profile": "reviewed-stage1-first-run-v1",
        "reviewed_by": "allocation-owner-and-data-reviewer",
        "reviewed_at_utc": "2026-07-16T20:00:00Z",
        "approval_reference": "sunfish-first-run-source-review-001",
        "corpus_gcs_inventory": inventory,
        "bucket_sources": {
            bucket: {
                "source_id": f"hf://datasets/example/{bucket}",
                "source_revision": f"{index + 1:040x}",
                "license_policy": "reviewed-permissive-or-dataset-approved",
                "filters": ["quality-v1", "dedup-v1"],
                "decontamination": ["lineage-v1", "sentinel-v1"],
                "token_share": percent / 100.0,
            }
            for index, (bucket, percent) in enumerate(
                CALIBRATION_BUCKET_PERCENT.items()
            )
        },
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def reviewed_manifest(receipt_sha256):
    total_tokens = 75_006_144
    record_tokens = 768
    bucket_caps = calibration_bucket_token_caps(total_tokens)
    shards = _reviewed_shards()
    inventory = reviewed_inventory()
    return {
        "manifest_version": 1,
        "promotion_allowed": True,
        "source_profile": "reviewed-stage1-first-run-v1",
        "source_receipt_sha256": receipt_sha256,
        "corpus_gcs_inventory_sha256": inventory["sha256"],
        "failures": [],
        "total_tokens": total_tokens,
        "total_records": sum(shard["records"] for shard in shards),
        "record_tokens": record_tokens,
        "bucket_token_caps": bucket_caps,
        "bucket_token_shares": {
            bucket: percent / 100.0
            for bucket, percent in CALIBRATION_BUCKET_PERCENT.items()
        },
        "shards": shards,
    }


class Stage1SourceContractTests(unittest.TestCase):
    def test_reviewed_immutable_source_receipt_binds_every_bucket(self):
        receipt = reviewed_receipt_bytes()
        digest = hashlib.sha256(receipt).hexdigest()
        result = validate_calibration_source_contract(
            reviewed_manifest(digest),
            receipt,
            expected_receipt_sha256=digest,
        )
        self.assertTrue(result["promotion_allowed"])
        self.assertEqual(result["source_receipt_sha256"], digest)
        self.assertEqual(
            result["corpus_gcs_inventory_sha256"], reviewed_inventory()["sha256"]
        )
        self.assertEqual(
            set(result["bucket_sources"]), set(CALIBRATION_BUCKET_PERCENT)
        )

    def test_structurally_complete_75m_pilot_manifest_cannot_promote(self):
        receipt = reviewed_receipt_bytes()
        digest = hashlib.sha256(receipt).hexdigest()
        pilot = reviewed_manifest(digest)
        pilot.update(
            {
                "promotion_allowed": False,
                "source_profile": "pilot-substitutes-v1",
                "source_receipt_sha256": None,
                "non_promotion_reason": "substitute sources",
            }
        )
        with self.assertRaisesRegex(ValueError, "non-promotable"):
            validate_calibration_source_contract(
                pilot,
                receipt,
                expected_receipt_sha256=digest,
            )

    def test_mutable_or_mismatched_source_revision_is_rejected(self):
        receipt_payload = json.loads(reviewed_receipt_bytes())
        receipt_payload["bucket_sources"]["code_completion"][
            "source_revision"
        ] = "main"
        receipt = (json.dumps(receipt_payload, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(receipt).hexdigest()
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            validate_calibration_source_contract(
                reviewed_manifest(digest),
                receipt,
                expected_receipt_sha256=digest,
            )

    def test_manifest_cannot_fake_the_75m_bucket_token_contract(self):
        receipt = reviewed_receipt_bytes()
        digest = hashlib.sha256(receipt).hexdigest()
        mutations = {}

        below_minimum = reviewed_manifest(digest)
        below_minimum["total_tokens"] = 74_999_999
        mutations["minimum"] = below_minimum

        bad_caps = reviewed_manifest(digest)
        bad_caps["bucket_token_caps"] = dict(bad_caps["bucket_token_caps"])
        bad_caps["bucket_token_caps"]["code_completion"] -= 1
        mutations["caps"] = bad_caps

        bad_shard_tokens = reviewed_manifest(digest)
        bad_shard_tokens["shards"][0]["tokens"] -= 1
        mutations["shard-tokens"] = bad_shard_tokens

        no_records = reviewed_manifest(digest)
        no_records["shards"][0]["records"] = 0
        mutations["records"] = no_records

        changed_record_size = reviewed_manifest(digest)
        changed_record_size["shards"][0]["record_tokens"] = 512
        mutations["record-tokens"] = changed_record_size

        for name, manifest in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_calibration_source_contract(
                        manifest,
                        receipt,
                        expected_receipt_sha256=digest,
                    )

    def test_receipt_and_manifest_must_bind_exact_corpus_inventory(self):
        inventory = reviewed_inventory()
        receipt = reviewed_receipt_bytes(inventory=inventory)
        digest = hashlib.sha256(receipt).hexdigest()

        missing_binding = reviewed_manifest(digest)
        missing_binding.pop("corpus_gcs_inventory_sha256")
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_calibration_source_contract(
                missing_binding,
                receipt,
                expected_receipt_sha256=digest,
            )

        changed_inventory = json.loads(json.dumps(inventory))
        changed_inventory["artifacts"][0]["generation"] += 1
        changed_receipt = reviewed_receipt_bytes(inventory=changed_inventory)
        changed_digest = hashlib.sha256(changed_receipt).hexdigest()
        with self.assertRaisesRegex(ValueError, "not canonical"):
            validate_calibration_source_contract(
                reviewed_manifest(changed_digest),
                changed_receipt,
                expected_receipt_sha256=changed_digest,
            )


if __name__ == "__main__":
    unittest.main()
