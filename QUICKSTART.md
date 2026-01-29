# Quickstart

This project uses a discrete masked diffusion model based on Qwen3.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Validate (CPU)

```bash
python validate_cpu.py
```

If you want to use a different base model:

```bash
python validate_cpu.py --base-model <model-id-or-path>
```

## 3) Train

```bash
python train_masked.py
python train_masked.py --cpu
python train_masked.py --use-shift
```

## 4) Generate

```bash
python sample_masked.py checkpoints/masked/last.ckpt
python sample_masked.py checkpoints/masked/last.ckpt --prompt "Once upon"
python sample_masked.py checkpoints/masked/last.ckpt --mode infill --text "The [MASK] ran."
python sample_masked.py checkpoints/masked/last.ckpt --mode interactive
```
