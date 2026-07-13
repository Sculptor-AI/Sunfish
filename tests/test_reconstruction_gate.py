import inspect
import unittest

from sunfish_tpu.reconstruction_gate import (
    NUM_LAYERS,
    RECONSTRUCTION_BUCKETS,
    approved_selection_payload,
    run_reconstruction_gate,
    summarize_reconstruction,
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
            source.index("validate_readiness_unlock"),
            source.index("initialize_distributed_jax"),
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
        candidate = {
            "source_revision": "generation:123",
            "dataset_manifest_sha256": "c" * 64,
            "retained_experts": 32,
            "top_k_experts": 8,
            "sunfish_source": {
                "git_commit": "c" * 40,
                "source_tree_sha256": "d" * 64,
            },
            "mass_gate_satisfied": True,
            "promotion_allowed": False,
            "layers": {str(layer): list(range(32)) for layer in range(30)},
        }
        approved = approved_selection_payload(
            candidate,
            summary,
            mass_candidate_sha256="a" * 64,
            reconstruction_run_sha256="b" * 64,
            reconstruction_summary_sha256="d" * 64,
        )
        self.assertTrue(approved["promotion_allowed"])
        self.assertTrue(approved["reconstruction_gate_satisfied"])
        candidate["mass_gate_satisfied"] = False
        with self.assertRaisesRegex(ValueError, "mass"):
            approved_selection_payload(
                candidate,
                summary,
                mass_candidate_sha256="a" * 64,
                reconstruction_run_sha256="b" * 64,
                reconstruction_summary_sha256="d" * 64,
            )


if __name__ == "__main__":
    unittest.main()
