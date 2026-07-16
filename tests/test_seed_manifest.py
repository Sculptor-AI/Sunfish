import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sunfish_tpu.seed_manifest import (
    STAGE1_APPROVED_METHOD,
    STAGE1_APPROVED_PURPOSE,
    STAGE05_PURPOSE,
    canonical_layer_selection_sha256,
    selection_metadata,
    validate_seed_manifest_bytes,
)
from sunfish_tpu.gcs_inventory import gcs_inventory_from_objects


def _layers():
    return {str(layer): list(range(32)) for layer in range(30)}


def _approved_selection(*, include_layers: bool = True):
    payload = {
        "schema_version": 1,
        "purpose": STAGE1_APPROVED_PURPOSE,
        "promotion_allowed": True,
        "selection_method": STAGE1_APPROVED_METHOD,
        "source_experts": 128,
        "retained_experts": 32,
        "top_k_experts": 8,
        "source_revision": "gcs-inventory-sha256:" + "f" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
        "mass_gate_satisfied": True,
        "reconstruction_gate_satisfied": True,
        "mass_min_coverage": 0.225,
        "mass_candidate_sha256": "1" * 64,
        "calibration_run_sha256": "2" * 64,
        "calibration_summary_sha256": "3" * 64,
        "calibration_artifact_inventory_sha256": "4" * 64,
        "calibration_corpus_gcs_inventory_sha256": "7" * 64,
        "reconstruction_run_sha256": "5" * 64,
        "reconstruction_summary_sha256": "6" * 64,
        "calibration_observed_input_tokens": 75_000_000,
        "calibration_minimum_observed_input_tokens": 75_000_000,
        "calibration_full_usable_records": 97_664,
        "calibration_processed_records": 97_664,
        "thresholds": {
            "max_relative_rmse": 0.15,
            "min_cosine_similarity": 0.99,
            "min_total_tokens": 100_000,
            "min_tokens_per_bucket": 4_000,
        },
        "worst_relative_rmse": {"value": 0.1, "bucket_layer": "x"},
        "worst_cosine_similarity": {"value": 0.995, "bucket_layer": "x"},
    }
    if include_layers:
        payload["layers"] = _layers()
    return payload


def _stage05_selection(*, purpose: str = STAGE05_PURPOSE):
    return {
        "schema_version": 1,
        "purpose": purpose,
        "promotion_allowed": False,
        "selection_method": "test",
        "source_experts": 128,
        "retained_experts": 32,
        "top_k_experts": 4,
        "layers": _layers(),
    }


def _manifest(*, promotion_allowed: bool, purpose: str = STAGE05_PURPOSE) -> bytes:
    source_inventory = gcs_inventory_from_objects(
        "gs://gemma-data/checkpoints/source",
        [{"name": "array", "generation": 1, "size": 2, "crc32c": "crc"}],
    )
    output_inventory = gcs_inventory_from_objects(
        "gs://bucket/seed",
        [{"name": "array", "generation": 2, "size": 3, "crc32c": "crc2"}],
    )
    selection = (
        _approved_selection()
        if promotion_allowed
        else _stage05_selection(purpose=purpose)
    )
    layers_sha256 = canonical_layer_selection_sha256(selection["layers"])
    selection["layers_sha256"] = layers_sha256
    payload = {
        "schema_version": 1,
        "output": "gs://bucket/seed",
        "source": "gs://gemma-data/checkpoints/source",
        "source_revision": f"gcs-inventory-sha256:{source_inventory['sha256']}",
        "source_gcs_inventory": source_inventory,
        "source_gcs_inventory_post_load": source_inventory,
        "output_gcs_inventory": output_inventory,
        "selection_sha256": "a" * 64,
        "selection_layers_sha256": layers_sha256,
        "selection_metadata": selection,
        "source_experts": 128,
        "retained_experts": 32,
        "top_k_experts": 8 if promotion_allowed else 4,
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
        "target_exact_tree": {"sha256": "b" * 64},
        "saved_tree": {"sha256": "b" * 64},
        "pruning": {"selection_layers_sha256": layers_sha256},
    }
    if promotion_allowed:
        payload["selection_metadata"]["source_revision"] = payload[
            "source_revision"
        ]
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
        encoded = _manifest(promotion_allowed=True)
        with self.assertRaisesRegex(ValueError, "smoke phase"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="smoke",
            )

    def test_manifest_hash_and_output_are_binding(self):
        encoded = _manifest(promotion_allowed=True)
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
                    _stage05_selection()
                ),
                encoding="utf-8",
            )
            self.assertFalse(selection_metadata(path)["promotion_allowed"])
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "promotion_allowed"):
                selection_metadata(path)

    def test_production_selection_requires_complete_approved_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "purpose": "hand-written",
                        "promotion_allowed": True,
                        "selection_method": "trust-me",
                        "source_experts": 128,
                        "retained_experts": 32,
                        "top_k_experts": 8,
                        "layers": {
                            str(layer): list(range(32)) for layer in range(30)
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "canonical Stage-1"):
                selection_metadata(path)
            approved = _approved_selection(include_layers=True)
            path.write_text(json.dumps(approved))
            self.assertTrue(selection_metadata(path)["promotion_allowed"])
            approved["source_revision"] = "unversioned"
            path.write_text(json.dumps(approved))
            with self.assertRaisesRegex(ValueError, "GCS inventory"):
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

    def test_production_seed_rejects_legacy_selection_without_corpus_inventory(self):
        payload = json.loads(_manifest(promotion_allowed=True))
        payload["selection_metadata"].pop(
            "calibration_corpus_gcs_inventory_sha256"
        )
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with self.assertRaisesRegex(
            ValueError, "calibration_corpus_gcs_inventory_sha256"
        ):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="router",
                expected_num_experts=32,
                expected_top_k_experts=8,
            )

    def test_runtime_rejects_tampered_layer_ids_and_pruning_binding(self):
        payload = json.loads(_manifest(promotion_allowed=False))
        payload["selection_metadata"]["layers"]["0"][-1] = 32
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "layer digest"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="smoke",
            )

        payload = json.loads(_manifest(promotion_allowed=False))
        payload["pruning"]["selection_layers_sha256"] = "0" * 64
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "pruning is not bound"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="smoke",
            )

    def test_runtime_requires_matching_post_load_source_inventory(self):
        payload = json.loads(_manifest(promotion_allowed=False))
        payload["source_gcs_inventory_post_load"] = gcs_inventory_from_objects(
            "gs://gemma-data/checkpoints/source",
            [{"name": "array", "generation": 2, "size": 2, "crc32c": "crc"}],
        )
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            validate_seed_manifest_bytes(
                encoded,
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                init_path="gs://bucket/seed",
                phase="smoke",
            )

if __name__ == "__main__":
    unittest.main()
