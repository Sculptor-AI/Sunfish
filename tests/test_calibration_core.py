import json
import tempfile
import unittest
from pathlib import Path

try:
    import jax  # noqa: F401

    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

try:
    import flax.linen as nn  # noqa: F401

    HAVE_FLAX = True
except Exception:
    HAVE_FLAX = False

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

    def test_state_and_accumulate_are_jittable_pytrees(self):
        import jax
        import jax.numpy as jnp

        from sunfish_tpu.calibration import accumulate, init_state

        state = init_state(len(self.BUCKETS), self.LAYERS, self.EXPERTS)
        compiled = jax.jit(accumulate)
        state = compiled(
            state,
            self.make_probs([[0.4, 0.3, 0.2, 0.1]]),
            jnp.asarray([2]),
        )
        self.assertEqual(int(state.tokens[2]), 1)
        self.assertAlmostEqual(float(state.mass[2, 0].sum()), 1.0, places=5)

    def test_accumulate_rejects_shape_mismatch_during_trace(self):
        import jax
        import jax.numpy as jnp

        from sunfish_tpu.calibration import accumulate, init_state

        state = init_state(len(self.BUCKETS), self.LAYERS, self.EXPERTS)
        with self.assertRaises(ValueError):
            jax.jit(accumulate)(
                state,
                jnp.ones((1, self.LAYERS + 1, self.EXPERTS), jnp.float32),
                jnp.asarray([0]),
            )

    @unittest.skipUnless(HAVE_FLAX, "flax not installed (heavy dependency)")
    def test_router_interceptor_returns_plain_softmax_under_jit(self):
        import flax.linen as nn
        import jax
        import jax.numpy as jnp

        from sunfish_tpu.calibration import call_with_router_probabilities

        class TinyRouter(nn.Module):
            num_experts: int = 4
            num_experts_per_datapoint: int = 2

            def _router(self, router_logits):
                probabilities = jax.nn.softmax(router_logits, axis=-1)
                choices = jnp.argsort(router_logits, axis=-1)[..., -2:]
                return probabilities, choices

            @nn.compact
            def __call__(self, logits):
                return self._router(logits)

        class TwoLayers(nn.Module):
            @nn.compact
            def __call__(self, logits):
                first = TinyRouter(name="layer_0")(logits)
                second = TinyRouter(name="layer_1")(logits + 1.0)
                return first, second

        model = TwoLayers()
        logits = jnp.asarray([[[1.0, 2.0, 3.0, 4.0]]], jnp.float32)
        variables = model.init(jax.random.key(0), logits)

        @jax.jit
        def run(value):
            return call_with_router_probabilities(
                lambda current: model.apply(variables, current),
                value,
                expected_layers=2,
            )

        _, probabilities = run(logits)
        self.assertEqual(probabilities.shape, (1, 2, 4))
        expected = jax.nn.softmax(logits[0, 0], axis=-1)
        self.assertTrue(jnp.allclose(probabilities[0, 0], expected))
        self.assertTrue(jnp.allclose(probabilities[0, 1], expected))

    @unittest.skipUnless(HAVE_FLAX, "flax not installed (heavy dependency)")
    def test_router_artifact_interceptor_captures_reconstruction_contract(self):
        import flax.linen as nn
        import jax
        import jax.numpy as jnp

        from sunfish_tpu.calibration import call_with_router_artifacts

        class TinyMoE(nn.Module):
            num_experts: int = 4
            num_experts_per_datapoint: int = 2

            def setup(self):
                self.per_expert_scale = self.param(
                    "per_expert_scale", nn.initializers.ones, (4,)
                )

            def _router(self, router_logits):
                probabilities = jax.nn.softmax(router_logits, axis=-1)
                _, choices = jax.lax.top_k(router_logits, 2)
                selected = jnp.take_along_axis(probabilities, choices, axis=-1)
                normalizer = selected.sum(axis=-1, keepdims=True)
                weights = probabilities / normalizer
                return weights, choices

            def __call__(self, x, unnormalized_x=None):
                residual = x if unnormalized_x is None else unnormalized_x
                logits = residual[..., :4]
                weights, choices = self._router(logits)
                return weights, choices

        class TwoLayers(nn.Module):
            @nn.compact
            def __call__(self, x):
                first = TinyMoE(name="layer_0")(x, x + 1.0)
                second = TinyMoE(name="layer_1")(x, x + 2.0)
                return first, second

        model = TwoLayers()
        x = jnp.asarray([[[1.0, 2.0, 3.0, 4.0]]], jnp.float32)
        variables = model.init(jax.random.key(0), x)

        @jax.jit
        def run(value):
            return call_with_router_artifacts(
                lambda current: model.apply(variables, current),
                value,
                expected_layers=2,
            )

        _, artifacts = run(x)
        self.assertEqual(artifacts["probabilities"].shape, (1, 2, 4))
        self.assertEqual(
            artifacts["shared_pre_router_residual"].shape, (1, 2, 4)
        )
        self.assertEqual(artifacts["topk_indices"].shape, (1, 2, 2))
        self.assertEqual(
            artifacts["final_scaled_topk_weights"].shape, (1, 2, 2)
        )
        self.assertTrue(
            jnp.array_equal(
                artifacts["shared_pre_router_residual"][0, 0], x[0, 0] + 1.0
            )
        )

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
            merged = merge_flushes(outdir, require_phase_coverage=False)
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
            merged = merge_flushes(Path(tmp), require_phase_coverage=False)
            bucket_mass = {
                k: v for k, v in selection_inputs(merged, 0).items() if sum(v) > 0
            }
            result = select_experts(bucket_mass, k=2, min_coverage=0.5)
            self.assertEqual(result.selected, (0, 1))
            self.assertTrue(result.satisfied)


class CalibrationCoverageTests(unittest.TestCase):
    def make_complete(self):
        from sunfish_tpu.calibration import PHASES, WORKLOADS

        accumulator = RouterStatsAccumulator(num_layers=1, num_experts=2)
        for phase in PHASES:
            for workload in WORKLOADS:
                accumulator.update(
                    bucket=f"{phase}/{workload}",
                    layer=0,
                    probabilities=[0.5, 0.5],
                )
                accumulator.count_tokens(
                    bucket=f"{phase}/{workload}", tokens=1
                )
        return accumulator

    def test_complete_phase_workload_taxonomy_passes(self):
        from sunfish_tpu.calibration import phase_coverage_errors

        self.assertEqual(phase_coverage_errors(self.make_complete()), [])

    def test_missing_phase_and_malformed_position_are_reported(self):
        from sunfish_tpu.calibration import phase_coverage_errors

        accumulator = self.make_complete()
        payload = json.loads(accumulator.to_json())
        del payload["mass"]["denoise_low/repo_edit"]
        del payload["tokens"]["denoise_low/repo_edit"]
        payload["mass"]["prefill/code_completion/pos0"] = [
            [0.5, 0.5]
        ]
        payload["tokens"]["prefill/code_completion/pos0"] = 1
        errors = phase_coverage_errors(
            RouterStatsAccumulator.from_json(json.dumps(payload))
        )
        self.assertTrue(any("missing tokens for denoise_low/repo_edit" in error for error in errors))
        self.assertTrue(any("prefill bucket must not" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
