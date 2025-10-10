# 🐟 SunFish Portable Demo

A self-contained demo of the SunFish diffusion language model.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo (unconditional)
python demo.py --samples 3 --length 128

# 3. Run with prompt (conditional)
python demo.py --prompt "Write about nature" --guidance 7.0 --samples 3
```

## What's Included

- `model.ckpt` - Trained SunFish model (403MB)
- `demo.py` - Simple generation script
- `models/` - Model architecture code
- `config/` - Model configuration

## Model Info

- **Type**: Diffusion Language Model
- **Size**: ~42M parameters
- **Training**: 6000 steps on CPU
- **Inference**: CPU-friendly, ~2-3 seconds per sample

## Usage Examples

### Unconditional Generation
```bash
python demo.py --samples 5 --length 256
```

### Conditional Generation (with prompt)
```bash
python demo.py --prompt "Write a story about robots" --guidance 7.0
```

### Adjust Quality
```bash
# Higher guidance = more prompt adherence (less creative)
python demo.py --prompt "Describe the ocean" --guidance 15.0

# Lower guidance = more creative (ignores prompt more)
python demo.py --prompt "Describe the ocean" --guidance 3.0
```

## Notes

- First run downloads GPT-2 tokenizer (~500MB)
- Generation takes ~2-5 seconds per sample on CPU
- This is a small experimental model - expect repetition and limited coherence
- Works best with short prompts and sequences

## Transfer to Another Machine

```bash
# Zip the entire directory
zip -r sunfish_demo.zip sunfish_demo_6k/

# Copy to another machine and unzip
# Then run: pip install -r requirements.txt && python demo.py
```
