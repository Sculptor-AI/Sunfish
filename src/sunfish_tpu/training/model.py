"""Construct the audited text-only 32-expert DiffusionGemma network."""

from __future__ import annotations

import dataclasses

from gemma.diffusion import _models
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
import jax.numpy as jnp

from sunfish_tpu.training.lora import SunfishLoRA


def make_gemma_network(
    *,
    num_experts: int,
    top_k_experts: int,
    dtype: str,
    use_lora: bool,
    lora_rank: int,
):
    """Create Sunfish by changing only audited expert-count routing fields."""
    if dtype == "bfloat16":
        jax_dtype = jnp.bfloat16
    elif dtype == "float32":
        jax_dtype = jnp.float32
    else:
        raise ValueError(f"unsupported model dtype {dtype}")
    model_config = dataclasses.replace(
        _models.DiffusionGemma_26B_A4B.config,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        vision_encoder=None,
        audio_encoder=None,
    )
    gemma_model = _models.DiffusionGemma_26B_A4B(
        config=model_config,
        dtype=jax_dtype,
        text_only=True,
    )
    network = hd_gemma_network.WrappedDiffusionGemmaNetwork(
        gemma_model=gemma_model
    )
    if use_lora:
        network = SunfishLoRA(
            rank=lora_rank,
            model=network,
            dtype=jax_dtype,
        )
    return network
