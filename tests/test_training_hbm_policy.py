import importlib.util
import inspect
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from sunfish_tpu.training.hbm_policy import (
    rematerialize_transformer_blocks,
    trainable_gradient_globs,
)
from sunfish_tpu.training.remat import validate_block_call_signature


ROOT = Path(__file__).resolve().parents[1]


def _heavy_stack_available() -> bool:
    for name in ("jax", "flax", "kauldron", "gemma"):
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


class TrainingHBMPolicyTests(unittest.TestCase):
    def test_partial_phases_select_only_their_trainable_gradient_families(self):
        self.assertEqual(trainable_gradient_globs("smoke"), ("**.lora.**",))
        self.assertEqual(trainable_gradient_globs("lora"), ("**.lora.**",))
        self.assertEqual(
            trainable_gradient_globs("router"),
            (
                "**.router_logits.**",
                "**.per_expert_scale",
                "**.router_scale",
            ),
        )
        self.assertIsNone(trainable_gradient_globs("full"))
        with self.assertRaisesRegex(ValueError, "unsupported training phase"):
            trainable_gradient_globs("unknown")

    def test_recovery_and_full_training_enable_block_remat(self):
        self.assertFalse(rematerialize_transformer_blocks("smoke"))
        self.assertFalse(rematerialize_transformer_blocks("router"))
        self.assertTrue(rematerialize_transformer_blocks("lora"))
        self.assertTrue(rematerialize_transformer_blocks("full"))

    def test_pinned_block_signature_guard_is_dependency_free(self):
        def pinned(
            self,
            x,
            segment_pos,
            cache,
            attn_mask,
            per_layer_input=None,
            kv_shared_cache=None,
            skip_sliding_mask=False,
        ):
            del self, x, segment_pos, cache, attn_mask
            del per_layer_input, kv_shared_cache, skip_sliding_mask

        validate_block_call_signature(pinned)

        def drifted(self, x, segment_pos, cache, attn_mask, new_argument=None):
            del self, x, segment_pos, cache, attn_mask, new_argument

        with self.assertRaisesRegex(RuntimeError, "signature drifted"):
            validate_block_call_signature(drifted)

    def test_config_wires_selected_norm_and_upstream_boundary_remat(self):
        config = (ROOT / "src/sunfish_tpu/training/kauldron_config.py").read_text()
        model = (ROOT / "src/sunfish_tpu/training/model.py").read_text()
        remat = (ROOT / "src/sunfish_tpu/training/remat.py").read_text()
        self.assertIn("trainable_gradient_globs(spec.run.phase.value)", config)
        self.assertIn("if gradient_globs is None:", config)
        self.assertIn("sunfish_metrics.SelectedTreeNorm", config)
        self.assertIn("rematerialize_blocks=rematerialize_transformer_blocks", config)
        self.assertIn("enable_upstream_block_rematerialization()", model)
        self.assertIn("jax.checkpoint_policies.nothing_saveable", remat)
        self.assertIn("static_argnums=7", remat)

    @unittest.skipUnless(
        _heavy_stack_available(),
        "pinned JAX/Flax/Kauldron/Gemma stack is not installed",
    )
    def test_selected_norm_ignores_frozen_gradient_leaves(self):
        import jax.numpy as jnp

        from sunfish_tpu.training.metrics import SelectedTreeNorm

        context = SimpleNamespace(
            grads={
                "layer_0": {
                    "mlp": {
                        "router_logits": {"w": jnp.asarray([3.0, 4.0])},
                        "per_expert_scale": jnp.asarray([12.0]),
                        "router_scale": jnp.asarray([84.0]),
                        # A NaN in a frozen expert bank makes accidental full-tree
                        # consumption unambiguously visible in this test.
                        "linear": {"w": jnp.asarray([math.nan])},
                    }
                }
            }
        )
        metric = SelectedTreeNorm(
            include=trainable_gradient_globs("router") or (),
        )
        value = float(metric.get_state_from_context(context).compute())
        self.assertEqual(value, 85.0)

        lora_context = SimpleNamespace(
            grads={
                "layer_0": {
                    "attn": {
                        "lora": {
                            "a": jnp.asarray([3.0]),
                            "b": jnp.asarray([4.0]),
                        },
                        "w": jnp.asarray([math.nan]),
                    }
                }
            }
        )
        lora_metric = SelectedTreeNorm(
            include=trainable_gradient_globs("lora") or (),
        )
        lora_value = float(
            lora_metric.get_state_from_context(lora_context).compute()
        )
        self.assertEqual(lora_value, 5.0)

    @unittest.skipUnless(
        _heavy_stack_available(),
        "pinned JAX/Flax/Kauldron/Gemma stack is not installed",
    )
    def test_pinned_gemma_block_accepts_and_installs_upstream_remat(self):
        from gemma.gm.nn.gemma4 import _modules

        from sunfish_tpu.training.remat import (
            enable_upstream_block_rematerialization,
        )

        original = _modules.Block.__call__
        try:
            validate_block_call_signature(original)
            enable_upstream_block_rematerialization()
            patched = _modules.Block.__call__
            self.assertIsNot(patched, original)
            self.assertEqual(
                tuple(inspect.signature(patched).parameters),
                (
                    "self",
                    "x",
                    "segment_pos",
                    "cache",
                    "attn_mask",
                    "per_layer_input",
                    "kv_shared_cache",
                    "skip_sliding_mask",
                ),
            )
            enable_upstream_block_rematerialization()
            self.assertIs(_modules.Block.__call__, patched)
        finally:
            _modules.Block.__call__ = original


if __name__ == "__main__":
    unittest.main()
