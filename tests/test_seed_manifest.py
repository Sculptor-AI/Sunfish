import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sunfish_tpu.seed_manifest import (
    STAGE05_PURPOSE,
    selection_metadata,
    validate_seed_manifest_bytes,
)
from sunfish_tpu.gcs_inventory import gcs_inventory_from_objects


def _manifest(*, promotion_allowed: bool, purpose: str = STAGE05_PURPOSE) -> bytes:
    source_inventory = gcs_inventory_from_objects(
        "gs://gemma-data/checkpoints/source",
        [{"name": "array", "generation": 1, "size": 2, "crc32c": "crc"}],
    )
    output_inventory = gcs_inventory_from_objects(
        "gs://bucket/seed",
        [{"name": "array", "generation": 2, "size": 3, "crc32c": "crc2"}],
    )
    payload = {
        "schema_version": 1,
        "output": "gs://bucket/seed",
        "source": "gs://gemma-data/checkpoints/source",
        "source_revision": f"gcs-inventory-sha256:{source_inventory['sha256']}",
        "source_gcs_inventory": source_inventory,
        "output_gcs_inventory": output_inventory,
        "selection_sha256": "a" * 64,
        "selection_metadata": {
            "purpose": purpose,
            "promotion_allowed": promotion_allowed,
            "selection_method": "test",
            "source_experts": 128,
            "retained_experts": 32,
            "top_k_experts": 4,
        },
        "source_experts": 128,
        "retained_experts": 32,
        "top_k_experts": 4,
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
        "target_exact_tree": {"sha256": "b" * 64},
        "saved_tree": {"sha256": "b" * 64},
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode()


class SeedManifestTests(unittest.TestCase):
    def test_stage05_seed_is_accepted_only_for_smoke(self):
        encoded = _manifest(promotion_allowed=False)
        digest = hashlib.sha256(encoded).hexdigest()
        validated = validate_seed_manifest_bytes(
            encoded,
            expected_sha256=digest,
            init_path="gs://bucket/seed",
            phase="smoke",
        )
        self.assertFalse(validated["selection_metadata"]["promotion_allowed"])
        with self.assertRaisesRegex(ValueError, "non-promotable"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=digest,
                init_path="gs://bucket/seed",
                phase="router",
            )

    def test_research_seed_is_rejected_for_readiness_smoke(self):
        encoded = _manifest(promotion_allowed=True, purpose="stage-2-approved")
        with self.assertRaisesRegex(ValueError, "smoke phase"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="smoke",
            )

    def test_manifest_hash_and_output_are_binding(self):
        encoded = _manifest(promotion_allowed=True, purpose="stage-2-approved")
        digest = hashlib.sha256(encoded).hexdigest()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256="0" * 64,
                init_path="gs://bucket/seed",
                phase="router",
            )
        with self.assertRaisesRegex(ValueError, "init_path"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=digest,
                init_path="gs://bucket/other",
                phase="router",
            )

    def test_selection_metadata_requires_explicit_promotion_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "purpose": STAGE05_PURPOSE,
                        "promotion_allowed": False,
                        "selection_method": "first-32",
                        "source_experts": 128,
                        "retained_experts": 32,
                        "top_k_experts": 4,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(selection_metadata(path)["promotion_allowed"])
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "promotion_allowed"):
                selection_metadata(path)

    def test_seed_top_k_is_bound_even_though_tree_shapes_match(self):
        encoded = _manifest(promotion_allowed=False)
        with self.assertRaisesRegex(ValueError, "top-k"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="smoke",
                expected_num_experts=32,
                expected_top_k_experts=8,
            )


if __name__ == "__main__":
    unittest.main()
