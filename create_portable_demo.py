#!/usr/bin/env python3
"""
Create a portable demo package for SunFish model
Packages everything needed to run the model elsewhere
"""

import torch
import argparse
import json
import shutil
from pathlib import Path


def create_portable_demo(checkpoint_path: str, output_dir: str = "sunfish_demo"):
    """Create a self-contained demo package."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"🐟 Creating portable SunFish demo package...")

    # 1. Copy the model checkpoint
    print(f"\n1️⃣ Copying model checkpoint...")
    shutil.copy(checkpoint_path, output_path / "model.ckpt")

    # 2. Copy necessary code files
    print(f"2️⃣ Copying code files...")
    files_to_copy = [
        "models/diffusion_transformer.py",
        "models/schedulers.py",
        "models/__init__.py",
        "config/model_config.py",
        "config/tiny_config.py",
        "config/micro_config.py",
        "config/nano_config.py",
        "config/__init__.py",
        "sample.py",
    ]

    for file in files_to_copy:
        src = Path(file)
        if src.exists():
            dst = output_path / file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    # 3. Create a simple demo script
    print(f"3️⃣ Creating demo script...")
    demo_script = '''#!/usr/bin/env python3
"""
SunFish Demo - Portable Text Generation
"""

import torch
from models import SunFishTransformer, DDIMScheduler
from transformers import AutoTokenizer

def generate_text(prompt=None, num_samples=3, seq_len=128, guidance_scale=7.0):
    """Generate text with the SunFish model."""

    # Load model
    print("🐟 Loading SunFish model...")
    model = SunFishTransformer.load_from_checkpoint("model.ckpt", map_location="cpu")
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create sampler
    sampler = DDIMScheduler(model, eta=0.0)

    print(f"🔮 Generating {num_samples} samples...")

    # Generate embeddings
    shape = (num_samples, seq_len, model.config.n_embd)
    embeddings = sampler.sample(
        shape,
        num_steps=20,
        show_progress=True,
        prompt=prompt,
        guidance_scale=guidance_scale if prompt else 1.0
    )

    # Round to tokens
    vocab_embeddings = model.token_embedding.weight
    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)
    vocab_norm = torch.nn.functional.normalize(vocab_embeddings, dim=-1)
    similarities = torch.matmul(embeddings_norm, vocab_norm.T)
    token_ids = similarities.argmax(dim=-1)

    # Decode
    print("\\n" + "="*80)
    for i, tokens in enumerate(token_ids):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\\nSample {i+1}:")
        print("-"*80)
        print(text)
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SunFish Demo")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--length", type=int, default=128, help="Sequence length")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")

    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        num_samples=args.samples,
        seq_len=args.length,
        guidance_scale=args.guidance
    )
'''

    with open(output_path / "demo.py", "w") as f:
        f.write(demo_script)

    # 4. Create requirements.txt
    print(f"4️⃣ Creating requirements.txt...")
    requirements = """torch>=2.0.0
transformers>=4.30.0
pytorch-lightning>=2.0.0
tqdm
"""
    with open(output_path / "requirements.txt", "w") as f:
        f.write(requirements)

    # 5. Create README
    print(f"5️⃣ Creating README...")
    readme = f"""# 🐟 SunFish Portable Demo

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

- `model.ckpt` - Trained SunFish model ({Path(checkpoint_path).stat().st_size / 1024 / 1024:.0f}MB)
- `demo.py` - Simple generation script
- `models/` - Model architecture code
- `config/` - Model configuration

## Model Info

- **Type**: Diffusion Language Model
- **Size**: ~42M parameters
- **Training**: {checkpoint_path.split('step=')[-1].split('.')[0] if 'step=' in checkpoint_path else 'Unknown'} steps on CPU
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
zip -r sunfish_demo.zip {output_dir}/

# Copy to another machine and unzip
# Then run: pip install -r requirements.txt && python demo.py
```
"""

    with open(output_path / "README.md", "w") as f:
        f.write(readme)

    # 6. Create a .gitignore
    with open(output_path / ".gitignore", "w") as f:
        f.write("__pycache__/\n*.pyc\n.DS_Store\n")

    # 7. Make demo.py executable
    (output_path / "demo.py").chmod(0o755)

    print(f"\n✅ Portable demo package created: {output_path}/")
    print(f"\n📦 Package contents:")
    print(f"   - model.ckpt ({Path(checkpoint_path).stat().st_size / 1024 / 1024:.0f}MB)")
    print(f"   - demo.py (simple generation script)")
    print(f"   - models/ (architecture code)")
    print(f"   - requirements.txt")
    print(f"   - README.md")

    print(f"\n🚀 To use:")
    print(f"   cd {output_dir}")
    print(f"   pip install -r requirements.txt")
    print(f"   python demo.py --prompt 'Write about nature'")

    print(f"\n📦 To transfer to another machine:")
    print(f"   zip -r sunfish_demo.zip {output_dir}/")
    print(f"   # Copy sunfish_demo.zip to other machine and unzip")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create portable SunFish demo")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="sunfish_demo", help="Output directory")

    args = parser.parse_args()
    create_portable_demo(args.checkpoint, args.output)
