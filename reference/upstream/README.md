# Upstream checkpoint reference (audited 2026-07-10)

Header-only copies of the 11 `google/diffusiongemma-26B-A4B-it` shards,
fetched via HTTP range requests (~156 KB total — headers parse fine without
tensor data; do NOT try to load weights from these files). Plus the upstream
`config.json`, our real-name group rules, and the generated `audit.json`.

## Verified facts (from real shard headers + config)

- **Totals**: 25,823,781,228 params incl. vision; text = **25,250,986,812**
  (vision tower = 572,794,416). Design math validated to the digit for expert
  banks (22,837,985,280) and routers (10,901,760); the original estimate was
  short exactly 60 params — the per-layer `layer_scalar` singletons
  (30 decoder + 30 encoder).
- **Expert banks are fused 3D tensors, not per-expert matrices**:
  `model.decoder.layers.N.experts.gate_up_proj` `[128, 1408, 2816]`
  (gate+up fused, 1408 = 2×704) and `.experts.down_proj` `[128, 2816, 704]`.
  **Pruning = slicing dim 0 with the retained-expert index list.**
- **Router = three tensors per layer**: `router.proj.weight` `[128, 2816]`,
  `router.scale` `[2816]`, `router.per_expert_scale` `[128]` — matching
  `model_budget.router_parameters` exactly. Pruning slices `proj.weight` and
  `per_expert_scale` on the expert dim; `scale` is untouched.
- **Encoder shares decoder weights**: `model.encoder.language_model.*`
  contains ONLY per-layer `layer_scalar` `[1]` tensors. The text-only strip
  keeps them; everything under `model.encoder.vision_tower.*` and
  `model.encoder.embed_vision.*` is dropped.
- **Self-conditioning** is a small decoder-level module (4 tensors, ~17.8M
  params): `model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}`.
  Never pruned.
- **Tied embeddings** (`tie_word_embeddings = true`): one `[262144, 2816]`
  tensor serves as both embedding and output head.
- **Config quirks the converter/inference must carry over**:
  `final_logit_softcapping = 30.0`; GQA 16 Q heads / 8 KV heads, head_dim 256;
  global layers use `global_head_dim = 512` with `num_global_key_value_heads = 2`;
  dual RoPE (`proportional` theta 1e6 + `partial_rotary_factor 0.25` for full
  attention; `default` theta 1e4 for sliding); `sliding_window = 1024`;
  shared-MLP `intermediate_size = 2112`; `moe_intermediate_size = 704`;
  `canvas_length = 256`; `use_bidirectional_attention = "vision"`.

## Known upstream metadata quirk

The live Hugging Face index reports `total_parameters = 25,823,778,864`, but
the 1,047 BF16 header shapes sum to 25,823,781,228 (2,364 more) and exactly
match `total_size / 2` (Codex, 2026-07-10). Header shapes are canonical for
all Sunfish math (AGENTS.md ground rule 5); the index metadata is recorded
here as an upstream inconsistency, nothing more.

## Regenerate

```bash
PYTHONPATH=src python -m sunfish.checkpoint_audit \
  --dir reference/upstream --rules reference/upstream/audit_rules.json
```
