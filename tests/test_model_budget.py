import unittest

from sunfish.model_budget import DiffusionMoEBudget


class DiffusionMoEBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = DiffusionMoEBudget()

    def test_published_expert_tensor_math(self) -> None:
        self.assertEqual(self.budget.parameters_per_expert_per_layer, 5_947_392)
        self.assertEqual(self.budget.parameters_per_expert, 178_421_760)

    def test_source_count_matches_audited_shard_headers(self) -> None:
        # Audited 2026-07-10 from the real upstream shard headers
        # (reference/upstream/audit.json): text = 25,823,781,228 total minus
        # the 572,794,416-param vision tower.
        estimate = self.budget.estimate(experts=128, top_k=8)
        self.assertEqual(estimate["total_parameters"], 25_250_986_812)
        self.assertEqual(estimate["active_parameters"], 3_840_375_612)

    def test_sunfish_8b_a3b_candidate(self) -> None:
        estimate = self.budget.estimate(experts=32, top_k=4)
        self.assertEqual(estimate["total_parameters"], 8_114_384_892)
        self.assertEqual(estimate["active_parameters"], 3_118_575_612)

    def test_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            self.budget.estimate(experts=4, top_k=8)


if __name__ == "__main__":
    unittest.main()
