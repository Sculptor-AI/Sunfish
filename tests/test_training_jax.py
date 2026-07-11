import unittest

try:
    from flax import linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sunfish_tpu.training.lora import expert_lora_delta, fuse_lora_params
    from sunfish_tpu.training.checkpoint import _copy_set_path
    from sunfish_tpu.training.prefix_amortized import (
        PrefixStratifiedTimeSampler,
        gather_selected_canvas,
    )
except ImportError as error:  # CPU-light development environment.
    JAX_IMPORT_ERROR = error
else:
    JAX_IMPORT_ERROR = None


@unittest.skipIf(JAX_IMPORT_ERROR is not None, f"JAX training stack unavailable: {JAX_IMPORT_ERROR}")
class TrainingJaxTests(unittest.TestCase):
    def test_linen_vmap_shares_params_and_captures_two_decoder_passes(self):
        class ToyNetwork(nn.Module):
            @nn.compact
            def __call__(self, values):
                return nn.Dense(3, use_bias=False)(values)

        class TwoPass(nn.Module):
            @nn.compact
            def __call__(self, draws):
                network = ToyNetwork()

                def run(values):
                    return nn.vmap(
                        lambda module, draw: module(draw),
                        variable_axes={"params": None, "intermediates": 0},
                        split_rngs={"params": False},
                        in_axes=0,
                        out_axes=0,
                        axis_size=draws.shape[0],
                    )(network, values)

                return run(draws), run(draws)

        model = TwoPass()
        draws = jnp.ones((4, 2, 5), dtype=jnp.float32)
        variables = model.init(
            {"params": jax.random.key(3)},
            draws,
            capture_intermediates=True,
        )
        (first, second), _ = model.apply(
            variables,
            draws,
            mutable=True,
            capture_intermediates=True,
        )
        self.assertEqual(first.shape, (4, 2, 3))
        np.testing.assert_array_equal(first, second)
        kernel = variables["params"]["ToyNetwork_0"]["Dense_0"]["kernel"]
        self.assertEqual(kernel.shape, (5, 3))

    def test_prefix_sampler_covers_every_noise_stratum(self):
        data = jnp.zeros((3, 8, 1), dtype=jnp.int32)
        times = PrefixStratifiedTimeSampler(safety_epsilon=0.0).sample_draws(
            jax.random.key(7), data, 4
        )
        self.assertEqual(times.shape, (4, 3, 1, 1))
        sorted_times = np.sort(np.asarray(times[:, :, 0, 0]), axis=0)
        for stratum in range(4):
            self.assertTrue(np.all(sorted_times[stratum] >= stratum / 4))
            self.assertTrue(np.all(sorted_times[stratum] < (stratum + 1) / 4))

    def test_selected_canvas_gather_is_per_example(self):
        values = jnp.arange(2 * 12).reshape(2, 12)
        selected = jnp.asarray([0, 2])
        actual = gather_selected_canvas(values, selected, 4)
        np.testing.assert_array_equal(actual, np.asarray([[0, 1, 2, 3], [20, 21, 22, 23]]))

    def test_expert_lora_contracts_rank_per_expert(self):
        a = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2)
        b = jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4)
        actual = expert_lora_delta(a, b, (2, 3, 4))
        expected = np.stack([np.asarray(a[i]) @ np.asarray(b[i]) for i in range(2)])
        np.testing.assert_allclose(actual, expected)

    def test_expert_lora_fusion_preserves_exact_tree(self):
        base = jnp.zeros((2, 3, 4), dtype=jnp.float32)
        a = jnp.ones((2, 3, 1), dtype=jnp.float32)
        b = jnp.ones((2, 1, 4), dtype=jnp.float32)
        params = {
            "layer_0": {
                "mlp": {
                    "linear": {
                        "w": base,
                        "lora": {"a": a, "b": b},
                    }
                }
            }
        }
        fused = fuse_lora_params(params)
        self.assertNotIn("lora", fused["layer_0"]["mlp"]["linear"])
        np.testing.assert_array_equal(
            fused["layer_0"]["mlp"]["linear"]["w"], np.ones_like(base)
        )

    def test_checkpoint_path_copy_does_not_mutate_source(self):
        source = {"outer": {"model": {"weight": 1}, "other": 2}}
        replaced = _copy_set_path(source, ["outer", "model"], {"shape": 3})
        self.assertEqual(source["outer"]["model"], {"weight": 1})
        self.assertEqual(replaced["outer"]["model"], {"shape": 3})
        self.assertEqual(replaced["outer"]["other"], 2)

    def test_shared_prefix_gradient_matches_k_independent_encodes(self):
        params = {
            "encoder": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10,
            "decoder": jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 10,
        }
        prefix = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10
        noises = jnp.arange(4 * 2 * 2, dtype=jnp.float32).reshape(4, 2, 2) / 100

        def amortized_loss(current):
            cache = prefix @ current["encoder"]
            predictions = jax.vmap(lambda noise: cache @ current["decoder"] + noise)(noises)
            return jnp.mean(jnp.square(predictions))

        def independent_loss(current):
            def one(noise):
                cache = prefix @ current["encoder"]
                return jnp.mean(jnp.square(cache @ current["decoder"] + noise))

            return jnp.mean(jax.vmap(one)(noises))

        amortized_grad = jax.grad(amortized_loss)(params)
        independent_grad = jax.grad(independent_loss)(params)
        for left, right in zip(
            jax.tree.leaves(amortized_grad),
            jax.tree.leaves(independent_grad),
            strict=True,
        ):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
