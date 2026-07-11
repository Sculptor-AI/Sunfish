# Upstream checkpoint contract

This file freezes the metadata assumptions used by the Stage 0 converter. It
was checked against the public metadata for
`google/diffusiongemma-26B-A4B-it` on 2026-07-10. The full gated weight shards
were not downloaded during this audit.

## Published metadata snapshot

| Field | Value |
| --- | ---: |
| Safetensors shards | 11 |
| Indexed tensors | 1,047 |
| Materialized parameters | 25,823,778,864 |
| Weight bytes | 51,647,562,456 |
| Decoder layers | 30 |
| Routed experts per layer | 128 |
| Active routed experts | 8 |
| Vision tensors removed by text-only conversion | 356 |

Metadata file hashes:

```text
config.json
13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506

model.safetensors.index.json
6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a
```

These hashes identify the audited metadata, not the 11 weight shards. The
downloaded checkpoint must still be audited from its actual safetensors
headers before conversion.

## Tensor-name contract

Every decoder layer has exactly these expert-axis tensors:

```text
model.decoder.layers.{L}.experts.down_proj
model.decoder.layers.{L}.experts.gate_up_proj
model.decoder.layers.{L}.router.per_expert_scale
model.decoder.layers.{L}.router.proj.weight
```

The converter slices axis 0 of those four tensors according to the per-layer
selection. `model.decoder.layers.{L}.router.scale` and `.mlp.*` are copied
unchanged. The always-active shared MLP is represented by `.mlp.*`; it is not
part of the routed bank.

Text-only conversion drops only tensors with these prefixes:

```text
model.encoder.embed_vision.
model.encoder.vision_tower.
```

It retains the tied encoder-language state, including
`model.encoder.language_model.layers.{L}.layer_scalar`, and changes top-level
`vision_config` to `null`. No unrecognized provenance keys are inserted into
the strict Hugging Face config; conversion details live in
`sunfish_conversion.json` instead.

## Required control sequence

1. Audit all downloaded shard headers and compare the reported totals with the
   published index.
2. Convert 128 experts / Top-8 with text-only mode. Untouched text tensors must
   remain byte-identical.
3. Load upstream and control implementations and compare seeded text logits
   and generated canvases. Header or byte equality does not replace this gate.
4. Only then convert a pruned candidate using a complete 30-layer selection
   manifest.

Sources: [model repository](https://huggingface.co/google/diffusiongemma-26B-A4B-it),
[DiffusionGemma documentation](https://ai.google.dev/gemma/docs/diffusiongemma).
