import unittest

from sunfish_tpu.runtime_api_audit import SOURCE_CONTRACTS, audit_source_texts


def _passing_sources():
    sources = {key: "pass\n" for key in SOURCE_CONTRACTS}
    sources.update(
        {
            "orbax_standard_checkpointer": """
class StandardCheckpointer:
  def restore(self, directory, target=None, *, strict=True):
    pass
""",
            "orbax_atomicity": "COMMIT_SUCCESS_FILE = 'commit_success.txt'\n",
            "kauldron_train_loop": """
def train_impl():
  state, chrono, ds_iter = ckpt.restore(
      checkpoint_state.CheckpointState(state, chrono, ds_iter)
  )
  for i in steps:
    ckpt.save(
        checkpoint_state.CheckpointState(state, chrono, ds_iter), step=i
    )
    batch = next(ds_iter)
    state, aux = trainstep.step(state, batch)
    writer.write_step_metrics(step=i)
""",
            "kauldron_data": """
class DataSourceBase:
  def ds_for_current_process(self, rng):
    ds = ds[jax.process_index() :: jax.process_count()]
    return ds
""",
            "gemma_models": """
class DiffusionGemma_26B_A4B(Base):
  def setup(self):
    pass
""",
            "gemma_paths": """
class CheckpointPath:
  DIFFUSIONGEMMA_26B_A4B_IT = 'gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it'
""",
            "gemma_base_model": """
class Gemma4_26B_A4B:
  config = TransformerConfig(
      num_embed=262144, embed_dim=2816, hidden_dim=2112,
      num_experts=128, expert_dim=704, top_k_experts=8,
      moe_dense_hidden_dim=2112, final_logit_softcap=30.0,
      sliding_window_size=1024,
  )
""",
            "gemma_hd_network": """
def prefill_kv_cache_with_encoder(): pass
class WrappedDiffusionGemmaNetwork:
  def init_cache(self): pass
  def encoder_call(self): pass
  def __call__(self): pass
""",
            "gemma_hd_lora": """
def _replace_by_lora(module, *, rank, dtype, verbose, target_modules=None): pass
def _find_base_weight_key(lora_parent, original_flat): pass
def _compute_lora_delta(a, b, *, target_shape): pass
""",
            "gemma_mask_helpers": """
def build_positions_from_mask(mask): pass
def make_causal_prefill_mask(token_mask, cache_length): pass
def set_cache_end_index(kv_cache, end_index): pass
""",
            "gemma_moe": """
class _Weight: pass
class MoERagged:
  def _router(self, router_logits): pass
  def __call__(self, x, unnormalized_x=None): pass
""",
            "gemma_layers": "class RMSNorm: pass\n",
            "gemma_modules": "class FeedForward: pass\n",
        }
    )
    return sources


class RuntimeApiAuditTests(unittest.TestCase):
    def test_reviewed_source_contract_passes(self):
        report = audit_source_texts(_passing_sources())
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["sources"]), len(SOURCE_CONTRACTS))
        self.assertTrue(all(item["sha256"] for item in report["sources"]))

    def test_private_api_drift_fails_closed(self):
        sources = _passing_sources()
        sources["gemma_hd_lora"] = sources["gemma_hd_lora"].replace(
            "target_shape", "shape"
        )
        report = audit_source_texts(sources)
        self.assertFalse(report["passed"])
        failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
        self.assertIn("gemma:lora-private-apis", failed)

    def test_missing_source_fails_closed(self):
        sources = _passing_sources()
        del sources["kauldron_train_loop"]
        report = audit_source_texts(sources)
        self.assertFalse(report["passed"])
        self.assertIn(
            "source:kauldron_train_loop",
            {check["name"] for check in report["checks"] if check["status"] == "fail"},
        )

    def test_checkpoint_cursor_order_is_load_bearing(self):
        sources = _passing_sources()
        sources["kauldron_train_loop"] = sources["kauldron_train_loop"].replace(
            "    ckpt.save(\n        checkpoint_state.CheckpointState(state, chrono, ds_iter), step=i\n    )\n    batch = next(ds_iter)",
            "    batch = next(ds_iter)\n    ckpt.save(\n        checkpoint_state.CheckpointState(state, chrono, ds_iter), step=i\n    )",
        )
        report = audit_source_texts(sources)
        self.assertFalse(report["passed"])
        failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
        self.assertIn("kauldron:checkpoint-input-update-metric-order", failed)


if __name__ == "__main__":
    unittest.main()
