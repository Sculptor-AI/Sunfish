# Sunfish: Discrete Masked Diffusion LM

Sunfish is a discrete masked diffusion language model built around a pretrained
Qwen3 base. Instead of adding Gaussian noise to embeddings, it masks tokens and
predicts them with cross-entropy loss, then generates by iterative unmasking.

## Key Features

- Discrete masking with a `[MASK]` token
- Cross-entropy loss computed only on masked positions
- Confidence-based iterative unmasking sampler
- Infilling between prefix and suffix text
- Gradient checkpointing for 16GB-class GPUs

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate on CPU (downloads the base model if not cached):

```bash
python validate_cpu.py
```

Train:

```bash
python train_masked.py
python train_masked.py --cpu
python train_masked.py --use-shift
```

Generate:

```bash
python sample_masked.py checkpoints/masked/last.ckpt
python sample_masked.py checkpoints/masked/last.ckpt --prompt "Once upon"
python sample_masked.py checkpoints/masked/last.ckpt --mode infill --text "The [MASK] ran."
python sample_masked.py checkpoints/masked/last.ckpt --mode interactive
```

## Project Structure

```
config/
  qwen_masked_config.py   # Masked diffusion config
data/
  qwen_datamodule.py      # Qwen tokenizer + streaming dataset
models/
  masked_diffusion_lm.py  # Masked diffusion model
  discrete_sampler.py     # Iterative unmasking sampler
train_masked.py           # Training script
sample_masked.py          # Generation script
validate_cpu.py           # CPU validation script
requirements.txt          # Dependencies
```

## Notes

- Training and sampling require downloading the base model and dataset from
  Hugging Face unless they are already cached locally.
- CPU validation uses a synthetic dataset but still loads the base tokenizer
  and model.
