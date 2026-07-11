"""Training losses kept free of upstream evaluator import side effects."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from kauldron import kd
import optax


@dataclasses.dataclass(frozen=True, kw_only=True)
class EncoderARLoss(kd.losses.Loss):
    """Masked causal auxiliary loss for the clean prefix encoder."""

    encoder_logits: kd.kontext.Key = "preds.encoder_logits"
    encoder_target: kd.kontext.Key = "preds.encoder_target"
    encoder_target_mask: kd.kontext.Key = "preds.encoder_target_mask"

    def get_values(
        self,
        encoder_logits: jnp.ndarray,
        encoder_target: jnp.ndarray,
        encoder_target_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        token_loss = optax.softmax_cross_entropy_with_integer_labels(
            encoder_logits, encoder_target
        )
        mask = encoder_target_mask.astype(token_loss.dtype)
        return jnp.sum(token_loss * mask, axis=-1) / jnp.maximum(
            jnp.sum(mask, axis=-1), 1.0
        )
