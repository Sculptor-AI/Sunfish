import tempfile
import unittest
from pathlib import Path

try:
    import jax  # noqa: F401

    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

from sunfish.router_stats import RouterStatsAccumulator


@unittest.skipUnless(HAVE_JAX, "jax not installed (heavy dependency)")
class CalibrationCoreTests(unittest.TestCase):
    BUCKETS = ["prefill/code", "denoise_high/code", "denoise_low/general"]
    LAYERS, EXPERTS = 2, 4

    def make_probs(self, rows):
        import jax.numpy as jnp

        # rows: list of per-token single-layer distributions; tile to LAYERS.
        return jnp.asarray([[row] * self.LAYERS for row in rows], jnp.float32)

    def test_accumulate_masks_padding_and_sums_by_bucket(self):
        import jax.numpy as jnp

        from sunfish_tpu.calibration import accumulate, init_state

        state = init_state(len(self.BUCKETS), self.LAYERS, self.EXPERTS)
        probs = self.make_probs(
            [[0.7, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]]
        )
        bucket_ids = jnp.asarray([0, 0, -1])  # third token is padding
        state = accumulate(state, probs, bucket_ids)

        self.assertEqual(int(state.tokens[0]), 2)
        self.assertEqual(int(state.tokens[1]), 0)
        self.assertAlmostEqual(float(state.mass[0, 0, 0]), 1.1, places=5)
        self.assertAlmostEqual(float(state.mass[0].sum()), 2 * self.LAYERS, places=4)
        self.assertEqual(float(state.mass[1:].sum()), 0.0)  # padding contributed nothing

    def test_flush_merge_roundtrip_matches_direct_totals(self):
        import jax.numpy as jnp

        from sunfish_tpu.calibration import accumulate, flush_host, init_state, merge_flushes

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            # Two "hosts" accumulate disjoint tokens, flush separately.
            for host, dist in enumerate([[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.6]]):
                state = init_state(len(self.BUCKETS), self.LAYERS, self.EXPERTS)
                state = accumulate(
                    state, self.make_probs([dist]), jnp.asarray([1])
                )
                flush_host(
                    state,
                    bucket_names=self.BUCKETS,
                    process_index=host,
                    shard_id="0",
                    output_dir=outdir,
                )
            merged = merge_flushes(outdir)
            self.assertEqual(merged.tokens("denoise_high/code"), 2)
            mass = merged.layer_bucket_mass(0)["denoise_high/code"]
            self.assertAlmostEqual(mass[0], 0.7, places=5)
            self.assertAlmostEqual(mass[3], 0.7, places=5)
            self.assertAlmostEqual(sum(mass), 2.0, places=4)

    def test_flush_sanity_catches_bad_mass(self):
        import jax.numpy as jnp

        from sunfish_tpu.calibration import accumulate, flush_host, init_state

        state = init_state(len(self.BUCKETS), self.LAYERS, self.EXPERTS)
        bad = self.make_probs([[0.2, 0.1, 0.1, 0.1]])  # sums to 0.5, not 1
        state = accumulate(state, bad, jnp.asarray([0]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                flush_host(
                    state,
                    bucket_names=self.BUCKETS,
                    process_index=0,
                    shard_id="0",
                    output_dir=Path(tmp),
                )

    def test_selection_adapter_feeds_expert_selection(self):
        import jax.numpy as jnp

        from sunfish.expert_selection import select_experts
        from sunfish_tpu.calibration import (
            accumulate,
            flush_host,
            init_state,
            merge_flushes,
            selection_inputs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            state = init_state(len(self.BUCKETS), self.LAYERS, self.EXPERTS)
            state = accumulate(
                state,
                self.make_probs([[0.9, 0.05, 0.03, 0.02], [0.05, 0.9, 0.03, 0.02]]),
                jnp.asarray([0, 2]),
            )
            flush_host(
                state, bucket_names=self.BUCKETS, process_index=0,
                shard_id="0", output_dir=Path(tmp),
            )
            merged = merge_flushes(Path(tmp))
            bucket_mass = {
                k: v for k, v in selection_inputs(merged, 0).items() if sum(v) > 0
            }
            result = select_experts(bucket_mass, k=2, min_coverage=0.5)
            self.assertEqual(result.selected, (0, 1))
            self.assertTrue(result.satisfied)


if __name__ == "__main__":
    unittest.main()
