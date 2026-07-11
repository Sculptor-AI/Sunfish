import unittest

from sunfish.expert_selection import (
    relative_coverage_floor,
    retained_fraction,
    select_experts,
    select_per_layer,
)
from sunfish.router_stats import RouterStatsAccumulator


class RetainedFractionTests(unittest.TestCase):
    def test_relative_floor_scales_with_candidate_size(self):
        self.assertAlmostEqual(
            relative_coverage_floor(retained_experts=32, source_experts=128, ratio=0.9),
            0.225,
        )
        self.assertAlmostEqual(
            relative_coverage_floor(retained_experts=48, source_experts=128, ratio=0.9),
            0.3375,
        )

    def test_relative_floor_validation(self):
        with self.assertRaises(ValueError):
            relative_coverage_floor(retained_experts=0, source_experts=128, ratio=0.9)
        with self.assertRaises(ValueError):
            relative_coverage_floor(retained_experts=32, source_experts=128, ratio=-0.1)

    def test_full_selection_retains_everything(self):
        self.assertAlmostEqual(retained_fraction([1.0, 2.0, 3.0], [0, 1, 2]), 1.0)

    def test_partial_selection(self):
        self.assertAlmostEqual(retained_fraction([1.0, 1.0, 2.0], [2]), 0.5)

    def test_rejects_zero_mass(self):
        with self.assertRaises(ValueError):
            retained_fraction([0.0, 0.0], [0])


class SelectExpertsTests(unittest.TestCase):
    def test_single_bucket_is_topk_by_mass(self):
        result = select_experts({"all": [5.0, 1.0, 3.0, 0.5]}, k=2)
        self.assertEqual(result.selected, (0, 2))
        self.assertTrue(result.satisfied)
        self.assertAlmostEqual(result.coverage["all"], 8.0 / 9.5)

    def test_weighted_buckets_shift_selection(self):
        bucket_mass = {
            "code": [10.0, 0.0, 0.0, 0.1],
            "denoise": [0.0, 10.0, 0.0, 0.1],
        }
        heavy_code = select_experts(bucket_mass, k=1, bucket_weights={"code": 10.0, "denoise": 0.1})
        self.assertEqual(heavy_code.selected, (0,))
        heavy_denoise = select_experts(bucket_mass, k=1, bucket_weights={"code": 0.1, "denoise": 10.0})
        self.assertEqual(heavy_denoise.selected, (1,))

    def test_coverage_repair_rescues_downweighted_bucket(self):
        # Expert 3 carries nearly all of a low-weight bucket but is worthless
        # to the high-weight buckets, so unconstrained weighted top-2 drops it.
        bucket_mass = {
            "majority_a": [10.0, 9.0, 8.0, 0.1],
            "majority_b": [10.0, 9.0, 8.0, 0.1],
            "minority": [0.1, 0.1, 0.1, 9.7],
        }
        weights = {"majority_a": 1.0, "majority_b": 1.0, "minority": 0.05}

        unconstrained = select_experts(bucket_mass, k=2, bucket_weights=weights)
        self.assertNotIn(3, unconstrained.selected)
        self.assertLess(unconstrained.coverage["minority"], 0.35)

        constrained = select_experts(
            bucket_mass, k=2, bucket_weights=weights, min_coverage=0.35
        )
        self.assertIn(3, constrained.selected)
        self.assertTrue(constrained.satisfied)
        for coverage in constrained.coverage.values():
            self.assertGreaterEqual(coverage, 0.35)

    def test_infeasible_coverage_reports_unsatisfied(self):
        bucket_mass = {
            "spread": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
        result = select_experts(bucket_mass, k=2, min_coverage=0.9)
        self.assertFalse(result.satisfied)
        self.assertEqual(len(result.selected), 2)

    def test_validation(self):
        with self.assertRaises(ValueError):
            select_experts({}, k=1)
        with self.assertRaises(ValueError):
            select_experts({"a": [1.0, 2.0]}, k=3)
        with self.assertRaises(ValueError):
            select_experts({"a": [1.0, 2.0], "b": [1.0]}, k=1)
        with self.assertRaises(ValueError):
            select_experts({"a": [1.0, 2.0]}, k=1, bucket_weights={"missing": 1.0})


class CodexCounterexampleRegressions(unittest.TestCase):
    """Regression cases from coordination/channel.md [6] where single-start
    swap repair returned false-infeasible or cycled despite a feasible set."""

    def test_one_swap_local_trap_is_escaped(self):
        # Old behavior: greedy {0,2} repaired to {0,1} and stopped,
        # reporting infeasible; the feasible answer is {2,3}.
        bucket_mass = {
            "b0": [12.0, 2.0, 8.0, 4.0],
            "b1": [0.0, 5.0, 4.0, 3.0],
        }
        result = select_experts(
            bucket_mass,
            k=2,
            bucket_weights={"b0": 0.5, "b1": 0.05},
            min_coverage=0.45,
        )
        self.assertTrue(result.satisfied)
        self.assertEqual(result.selected, (2, 3))
        for coverage in result.coverage.values():
            self.assertGreaterEqual(coverage, 0.45)

    def test_two_cycle_is_broken_and_feasible_found(self):
        # Old behavior: repair oscillated (1,2) -> (0,2) -> (1,2) until the
        # round cap; the feasible answer is {0,1}.
        bucket_mass = {
            "b0": [12.0, 15.0, 4.0, 13.0],
            "b1": [16.0, 16.0, 18.0, 2.0],
            "b2": [18.0, 14.0, 7.0, 17.0],
        }
        result = select_experts(
            bucket_mass,
            k=2,
            bucket_weights={"b0": 0.1, "b1": 10.0, "b2": 0.01},
            min_coverage=0.45,
        )
        self.assertTrue(result.satisfied)
        self.assertEqual(result.selected, (0, 1))
        for coverage in result.coverage.values():
            self.assertGreaterEqual(coverage, 0.45)


class PerLayerTests(unittest.TestCase):
    def test_layers_are_independent(self):
        layers = [
            {"all": [9.0, 1.0, 1.0]},
            {"all": [1.0, 9.0, 1.0]},
        ]
        results = select_per_layer(layers, k=1)
        self.assertEqual(results[0].selected, (0,))
        self.assertEqual(results[1].selected, (1,))


class RouterStatsTests(unittest.TestCase):
    def test_accumulate_serialize_and_select(self):
        stats = RouterStatsAccumulator(num_layers=2, num_experts=4)
        stats.update(bucket="prefill/code", layer=0, probabilities=[0.7, 0.1, 0.1, 0.1])
        stats.update(bucket="prefill/code", layer=0, probabilities=[0.6, 0.2, 0.1, 0.1])
        stats.update_topk(
            bucket="denoise_high/code", layer=0, expert_indices=[3, 1], probabilities=[0.5, 0.3]
        )
        stats.count_tokens(bucket="prefill/code", tokens=2)
        stats.count_tokens(bucket="denoise_high/code", tokens=1)

        restored = RouterStatsAccumulator.from_json(stats.to_json())
        self.assertEqual(restored.tokens("prefill/code"), 2)
        layer_mass = restored.layer_bucket_mass(0)
        self.assertAlmostEqual(layer_mass["prefill/code"][0], 1.3)
        self.assertAlmostEqual(layer_mass["denoise_high/code"][3], 0.5)

        result = select_experts(layer_mass, k=2, min_coverage=0.3)
        self.assertIn(0, result.selected)
        self.assertIn(3, result.selected)
        self.assertTrue(result.satisfied)

    def test_merge_across_hosts(self):
        left = RouterStatsAccumulator(num_layers=1, num_experts=2)
        right = RouterStatsAccumulator(num_layers=1, num_experts=2)
        left.update(bucket="b", layer=0, probabilities=[1.0, 0.0])
        right.update(bucket="b", layer=0, probabilities=[0.0, 1.0])
        left.count_tokens(bucket="b", tokens=1)
        right.count_tokens(bucket="b", tokens=1)
        left.merge(right)
        self.assertEqual(left.tokens("b"), 2)
        self.assertEqual(left.layer_bucket_mass(0)["b"], [1.0, 1.0])

    def test_validation(self):
        stats = RouterStatsAccumulator(num_layers=1, num_experts=2)
        with self.assertRaises(ValueError):
            stats.update(bucket="b", layer=1, probabilities=[0.5, 0.5])
        with self.assertRaises(ValueError):
            stats.update(bucket="b", layer=0, probabilities=[0.5])
        with self.assertRaises(ValueError):
            stats.update(bucket="b", layer=0, probabilities=[-0.1, 0.5])


if __name__ == "__main__":
    unittest.main()
