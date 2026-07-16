import inspect
import unittest

from sunfish_tpu.calibration_data_inventory import (
    calibration_data_inventory_from_objects,
)
from sunfish_tpu.reconstruction_gate import (
    NUM_LAYERS,
    RECONSTRUCTION_BUCKETS,
    approved_selection_payload,
    run_reconstruction_gate,
    summarize_reconstruction,
    validate_calibration_for_reconstruction,
)

SOURCE = {"git_commit": "c" * 40, "source_tree_sha256": "d" * 64}
CALIBRATION_RECORDS = 97_664
CALIBRATION_STEPS = 12_208
CALIBRATION_SOURCE_TOKENS = 75_005_952


def corpus_inventory():
    manifest = {
        "shards": [
            {
                "bucket": "code_completion",
                "bin": "code_completion.bin",
                "idx": "code_completion.idx",
                "records": CALIBRATION_RECORDS,
                "tokens": CALIBRATION_SOURCE_TOKENS,
                "sha256_bin": "7" * 64,
                "sha256_idx": "8" * 64,
            }
        ]
    }
    return calibration_data_inventory_from_objects(
        "gs://bucket/calibration/data",
        manifest,
        [
            {
                "name": "code_completion.bin",
                "generation": 70,
                "size": CALIBRATION_SOURCE_TOKENS * 4,
                "crc32c": "bin-crc",
            },
            {
                "name": "code_completion.idx",
                "generation": 71,
                "size": CALIBRATION_RECORDS * 8,
                "crc32c": "idx-crc",
            },
        ],
    )


def calibration_evidence():
    run_sha256 = "a" * 64
    summary_sha256 = "b" * 64
    candidate_sha256 = "c" * 64
    candidate_path = "gs://bucket/calibration/selection-mass-candidate.json"
    inventory = corpus_inventory()
    identity = {
        "schema_version": 1,
        "run_id": "calibration-v1",
        "run_mode": "full",
        "debug_run": False,
        "max_records": 0,
        "source_records": CALIBRATION_RECORDS,
        "full_usable_records": CALIBRATION_RECORDS,
        "usable_records": CALIBRATION_RECORDS,
        "collective_steps": CALIBRATION_STEPS,
        "source_tokens": CALIBRATION_SOURCE_TOKENS,
        "record_tokens": 768,
        "maximum_usable_input_tokens": CALIBRATION_SOURCE_TOKENS,
        "minimum_source_tokens": 75_000_000,
        "reconstruction_sample_tokens": 100_000,
        "coverage_floors": {"32_experts": 0.225, "48_experts": 0.3375},
        "source_revision": "generation:123",
        "topology": {"processes": 8},
        "dataset_manifest_sha256": "e" * 64,
        "dataset_directory": "gs://bucket/calibration/data",
        "corpus_gcs_inventory": inventory,
        "corpus_gcs_inventory_sha256": inventory["sha256"],
        "sunfish_source": SOURCE,
    }
    candidate = {
        "schema_version": 1,
        "source_revision": "generation:123",
        "dataset_manifest_sha256": "e" * 64,
        "corpus_gcs_inventory_sha256": inventory["sha256"],
        "source_experts": 128,
        "retained_experts": 32,
        "top_k_experts": 8,
        "sunfish_source": SOURCE,
        "mass_gate_satisfied": True,
        "min_coverage": 0.225,
        "promotion_allowed": False,
        "promotion_eligible": True,
        "calibration_run_sha256": run_sha256,
        "calibration_mode": "full",
        "debug_run": False,
        "full_usable_records": CALIBRATION_RECORDS,
        "processed_records": CALIBRATION_RECORDS,
        "full_corpus_consumed": True,
        "observed_input_tokens": 75_000_000,
        "minimum_observed_input_tokens": 75_000_000,
        "minimum_tokens_observed": True,
        "layers": {str(layer): list(range(32)) for layer in range(30)},
    }
    summary = {
        "schema_version": 1,
        "run_id": "calibration-v1",
        "execution_completed": True,
        "run_succeeded": True,
        "artifact_sample_satisfied": True,
        "promotion_eligible": True,
        "full_corpus_consumed": True,
        "minimum_tokens_observed": True,
        "run_mode": "full",
        "debug_run": False,
        "max_records": 0,
        "calibration_run_sha256": run_sha256,
        "dataset_manifest_sha256": "e" * 64,
        "corpus_gcs_inventory_sha256": inventory["sha256"],
        "source_records": CALIBRATION_RECORDS,
        "full_usable_records": CALIBRATION_RECORDS,
        "processed_records": CALIBRATION_RECORDS,
        "collective_steps": CALIBRATION_STEPS,
        "maximum_usable_input_tokens": CALIBRATION_SOURCE_TOKENS,
        "observed_input_tokens": 75_000_000,
        "minimum_observed_input_tokens": 75_000_000,
        "candidate": candidate_path,
        "candidate_sha256": candidate_sha256,
        "mass_gate_satisfied": True,
    }
    return (
        identity,
        summary,
        candidate,
        run_sha256,
        summary_sha256,
        candidate_sha256,
        candidate_path,
    )


def metric_rows(relative_error=0.1, cosine=0.995):
    # teacher norm = candidate norm = 1; dot selects cosine. Error is supplied
    # independently because the pure gate deliberately consumes additive sums.
    error_squared = relative_error**2
    return [
        [[error_squared, 1.0, cosine, 1.0] for _ in range(NUM_LAYERS)]
        for _ in RECONSTRUCTION_BUCKETS
    ]


class ReconstructionGateTests(unittest.TestCase):
    def test_reconstruction_revalidates_calibration_readiness_receipt(self):
        source = inspect.getsource(run_reconstruction_gate)
        self.assertIn("readiness_ledger_sha256", source)
        self.assertIn("validate_readiness_unlock", source)
        self.assertLess(
            source.index("validate_calibration_for_reconstruction"),
            source.index("initialize_distributed_jax"),
        )
        self.assertLess(
            source.index("validate_readiness_unlock"),
            source.index("initialize_distributed_jax"),
        )
        self.assertIn("_broadcast_process0_error", source)

    def test_reconstruction_thresholds_cannot_be_weakened(self):
        with self.assertRaisesRegex(ValueError, "canonical and non-overridable"):
            run_reconstruction_gate(
                source_checkpoint="gs://bucket/teacher",
                source_anonymous=True,
                calibration_dir="gs://bucket/calibration",
                raw_dir="gs://bucket/raw",
                candidate_path="gs://bucket/candidate.json",
                output_dir="gs://bucket/output",
                run_id="reconstruction-v1",
                expected_devices=64,
                expected_processes=8,
                expected_local_devices=8,
                max_relative_rmse=1.0,
                min_cosine_similarity=0.99,
                min_total_tokens=100_000,
                min_tokens_per_bucket=4_000,
            )

    def test_all_bucket_layer_thresholds_pass(self):
        summary = summarize_reconstruction(
            metric_rows(),
            [4_200] * len(RECONSTRUCTION_BUCKETS),
            route_mismatches=0,
            max_relative_rmse=0.15,
            min_cosine_similarity=0.99,
            min_total_tokens=100_000,
            min_tokens_per_bucket=4_000,
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["total_tokens"], 100_800)

    def test_one_bad_layer_or_teacher_route_fails(self):
        rows = metric_rows()
        rows[3][7][0] = 0.25**2
        summary = summarize_reconstruction(
            rows,
            [4_200] * len(RECONSTRUCTION_BUCKETS),
            route_mismatches=1,
            max_relative_rmse=0.15,
            min_cosine_similarity=0.99,
            min_total_tokens=100_000,
            min_tokens_per_bucket=4_000,
        )
        self.assertFalse(summary["passed"])
        self.assertTrue(any("layer7" in error for error in summary["errors"]))
        self.assertTrue(any("teacher route" in error for error in summary["errors"]))

    def test_promotion_requires_both_gates_and_pins_provenance(self):
        summary = summarize_reconstruction(
            metric_rows(),
            [4_200] * len(RECONSTRUCTION_BUCKETS),
            route_mismatches=0,
            max_relative_rmse=0.15,
            min_cosine_similarity=0.99,
            min_total_tokens=100_000,
            min_tokens_per_bucket=4_000,
        )
        (
            identity,
            calibration_summary,
            candidate,
            calibration_run_sha256,
            calibration_summary_sha256,
            candidate_sha256,
            candidate_path,
        ) = calibration_evidence()
        provenance = validate_calibration_for_reconstruction(
            identity,
            calibration_summary,
            candidate,
            calibration_run_sha256=calibration_run_sha256,
            calibration_summary_sha256=calibration_summary_sha256,
            mass_candidate_sha256=candidate_sha256,
            mass_candidate_path=candidate_path,
        )
        provenance["artifact_inventory_sha256"] = "2" * 64
        summary["calibration_provenance"] = provenance
        approved = approved_selection_payload(
            candidate,
            summary,
            mass_candidate_sha256=candidate_sha256,
            reconstruction_run_sha256="f" * 64,
            reconstruction_summary_sha256="1" * 64,
            calibration_provenance=provenance,
        )
        self.assertTrue(approved["promotion_allowed"])
        self.assertTrue(approved["reconstruction_gate_satisfied"])
        self.assertEqual(
            approved["calibration_summary_sha256"], calibration_summary_sha256
        )
        self.assertEqual(approved["calibration_observed_input_tokens"], 75_000_000)
        self.assertEqual(
            approved["calibration_corpus_gcs_inventory_sha256"],
            corpus_inventory()["sha256"],
        )
        changed_summary = dict(summary)
        changed_summary["calibration_provenance"] = {
            **provenance,
            "observed_input_tokens": 75_000_001,
        }
        with self.assertRaisesRegex(ValueError, "changed calibration provenance"):
            approved_selection_payload(
                candidate,
                changed_summary,
                mass_candidate_sha256=candidate_sha256,
                reconstruction_run_sha256="f" * 64,
                reconstruction_summary_sha256="1" * 64,
                calibration_provenance=provenance,
            )
        candidate["mass_gate_satisfied"] = False
        with self.assertRaisesRegex(ValueError, "mass"):
            approved_selection_payload(
                candidate,
                summary,
                mass_candidate_sha256=candidate_sha256,
                reconstruction_run_sha256="f" * 64,
                reconstruction_summary_sha256="1" * 64,
                calibration_provenance=provenance,
            )

    def test_reconstruction_rejects_capped_or_short_calibration(self):
        (
            identity,
            calibration_summary,
            candidate,
            run_sha256,
            summary_sha256,
            candidate_sha256,
            candidate_path,
        ) = calibration_evidence()
        identity["run_mode"] = "debug-capped"
        identity["debug_run"] = True
        identity["max_records"] = CALIBRATION_RECORDS
        with self.assertRaisesRegex(ValueError, "debug/capped"):
            validate_calibration_for_reconstruction(
                identity,
                calibration_summary,
                candidate,
                calibration_run_sha256=run_sha256,
                calibration_summary_sha256=summary_sha256,
                mass_candidate_sha256=candidate_sha256,
                mass_candidate_path=candidate_path,
            )

        identity, calibration_summary, candidate, *_ = calibration_evidence()
        calibration_summary["observed_input_tokens"] = 74_999_999
        with self.assertRaisesRegex(ValueError, "fewer input tokens"):
            validate_calibration_for_reconstruction(
                identity,
                calibration_summary,
                candidate,
                calibration_run_sha256=run_sha256,
                calibration_summary_sha256=summary_sha256,
                mass_candidate_sha256=candidate_sha256,
                mass_candidate_path=candidate_path,
            )

    def test_reconstruction_rejects_candidate_not_recorded_by_summary(self):
        (
            identity,
            calibration_summary,
            candidate,
            run_sha256,
            summary_sha256,
            _,
            candidate_path,
        ) = calibration_evidence()
        with self.assertRaisesRegex(ValueError, "recorded by calibration"):
            validate_calibration_for_reconstruction(
                identity,
                calibration_summary,
                candidate,
                calibration_run_sha256=run_sha256,
                calibration_summary_sha256=summary_sha256,
                mass_candidate_sha256="9" * 64,
                mass_candidate_path=candidate_path,
            )

    def test_reconstruction_rejects_legacy_or_tampered_corpus_inventory(self):
        (
            identity,
            calibration_summary,
            candidate,
            run_sha256,
            summary_sha256,
            candidate_sha256,
            candidate_path,
        ) = calibration_evidence()
        identity.pop("corpus_gcs_inventory")
        with self.assertRaisesRegex(ValueError, "no corpus GCS inventory"):
            validate_calibration_for_reconstruction(
                identity,
                calibration_summary,
                candidate,
                calibration_run_sha256=run_sha256,
                calibration_summary_sha256=summary_sha256,
                mass_candidate_sha256=candidate_sha256,
                mass_candidate_path=candidate_path,
            )

        identity, calibration_summary, candidate, *_ = calibration_evidence()
        identity["corpus_gcs_inventory"]["artifacts"][0]["generation"] += 1
        with self.assertRaisesRegex(ValueError, "not canonical"):
            validate_calibration_for_reconstruction(
                identity,
                calibration_summary,
                candidate,
                calibration_run_sha256=run_sha256,
                calibration_summary_sha256=summary_sha256,
                mass_candidate_sha256=candidate_sha256,
                mass_candidate_path=candidate_path,
            )

        identity, calibration_summary, candidate, *_ = calibration_evidence()
        calibration_summary["corpus_gcs_inventory_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "summary corpus inventory"):
            validate_calibration_for_reconstruction(
                identity,
                calibration_summary,
                candidate,
                calibration_run_sha256=run_sha256,
                calibration_summary_sha256=summary_sha256,
                mass_candidate_sha256=candidate_sha256,
                mass_candidate_path=candidate_path,
            )


if __name__ == "__main__":
    unittest.main()
