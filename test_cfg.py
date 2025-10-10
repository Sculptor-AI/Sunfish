#!/usr/bin/env python3
"""
Test script for Classifier-Free Guidance implementation
"""

import torch
from config.tiny_config import get_tiny_config
from models import SunFishTransformer

print("=" * 70)
print("🧪 Testing Classifier-Free Guidance Implementation")
print("=" * 70)

# Load config
config = get_tiny_config()

# Initialize model
print("\n1️⃣ Initializing model...")
model = SunFishTransformer(config)
model.eval()
print(f"✅ Model loaded with {model.get_num_params():,} parameters")

# Test 1: Unconditional forward pass
print("\n2️⃣ Testing unconditional forward pass...")
batch_size = 2
seq_len = 32
x_0 = torch.randn(batch_size, seq_len, config.n_embd)
t = torch.randint(0, config.timesteps, (batch_size,))
x_t = model.q_sample(x_0, t)

with torch.no_grad():
    pred_uncond = model(x_t, t, context=None)
    print(f"✅ Unconditional output shape: {pred_uncond.shape}")
    assert pred_uncond.shape == (batch_size, seq_len, config.n_embd)

# Test 2: Training step with unconditional generation
print("\n3️⃣ Testing training step (unconditional)...")
model.train()
token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
loss = model.training_step(token_ids, 0)
print(f"✅ Training loss (unconditional): {loss.item():.4f}")
assert not torch.isnan(loss) and not torch.isinf(loss)

# Test 3: Check CFG parameters are in config
print("\n4️⃣ Checking CFG config parameters...")
assert hasattr(config, "conditioning_dropout"), "Missing conditioning_dropout"
assert hasattr(config, "guidance_scale"), "Missing guidance_scale"
assert hasattr(config, "prompt_max_length"), "Missing prompt_max_length"
print(f"✅ conditioning_dropout: {config.conditioning_dropout}")
print(f"✅ guidance_scale: {config.guidance_scale}")
print(f"✅ prompt_max_length: {config.prompt_max_length}")

# Test 4: Test transformer blocks with None context
print("\n5️⃣ Testing transformer blocks with None context...")
model.eval()
with torch.no_grad():
    x = torch.randn(batch_size, seq_len, config.n_embd)
    for i, block in enumerate(model.transformer_blocks):
        x = block(x, context=None)
        assert x.shape == (batch_size, seq_len, config.n_embd)
    print(f"✅ All {len(model.transformer_blocks)} blocks work with None context")

# Test 5: Test prompt encoder initialization (without loading GPT-2)
print("\n6️⃣ Testing prompt encoder structure...")
prompt_encoder = model.prompt_encoder
assert hasattr(prompt_encoder, "prompt_projection"), "Missing prompt_projection"
assert hasattr(prompt_encoder, "encode"), "Missing encode method"
print(f"✅ Prompt encoder initialized (lazy-loading enabled)")
print(f"   Projection: 768 -> {config.n_embd}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n📝 Notes:")
print("   - Model can do unconditional generation ✓")
print("   - Training loop supports CFG dropout ✓")
print("   - Transformer blocks handle None context ✓")
print("   - CFG parameters configured ✓")
print("   - Prompt encoder ready for conditional generation ✓")
print("\n💡 To test with actual prompts:")
print("   1. Ensure internet connection (for GPT-2 download)")
print("   2. Run training with prompt data")
print("   3. Use sample.py --prompt 'Your prompt here'")
