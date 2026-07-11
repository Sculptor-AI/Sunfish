import random
import unittest

from sunfish.sampling import (
    StableAndConfidentStopper,
    accept_canvas,
    entropy_bound_mask,
    linear_temperature,
    renoise_canvas,
)


class EntropyBoundSamplerTest(unittest.TestCase):
    def test_temperature_counts_reverse_steps(self) -> None:
        self.assertEqual(
            linear_temperature(remaining_step=48, max_steps=48, minimum=0.4, maximum=0.8),
            0.8,
        )
        self.assertAlmostEqual(
            linear_temperature(remaining_step=24, max_steps=48, minimum=0.4, maximum=0.8),
            0.6,
        )

    def test_entropy_bound_selects_lowest_entropy_positions(self) -> None:
        # Sorted values are .01, .04, .08, .20. The cumulative-minus-current
        # values are 0, .01, .05, and .13, so the final position is rejected.
        mask = entropy_bound_mask([0.20, 0.01, 0.08, 0.04], bound=0.1)
        self.assertEqual(mask, (False, True, True, True))

    def test_accept_then_renoise_changes_only_unaccepted_positions(self) -> None:
        accepted, mask = accept_canvas(
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [0.20, 0.01, 0.08, 0.04],
            bound=0.1,
        )
        self.assertEqual(accepted, (10, 21, 22, 23))
        renoised = renoise_canvas(accepted, mask, vocab_size=100, rng=random.Random(7))
        self.assertEqual(renoised[1:], (21, 22, 23))
        self.assertNotEqual(renoised[0], 10)

    def test_stopping_requires_both_stability_and_confidence(self) -> None:
        stopper = StableAndConfidentStopper(
            stability_steps=1, mean_entropy_threshold=0.005
        )
        self.assertFalse(stopper([1, 2], [0.001, 0.001]))
        self.assertTrue(stopper([1, 2], [0.001, 0.001]))
        self.assertFalse(stopper([1, 3], [0.001, 0.001]))
        self.assertFalse(stopper([1, 3], [0.01, 0.01]))


if __name__ == "__main__":
    unittest.main()
