"""Prefix-amortized multi-noise DiffusionGemma training objective."""

from __future__ import annotations

import dataclasses
from typing import Any

from flax import linen as nn
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
import jax
import jax.numpy as jnp
from kauldron import kd


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrefixStratifiedTimeSampler:
    """Give every prefix one random time in each of K equal noise strata."""

    safety_epsilon: float = 1e-4

    def sample_draws(
        self, key: jax.Array, data_spec: jax.Array, num_draws: int
    ) -> jax.Array:
        if num_draws <= 0:
            raise ValueError("num_draws must be positive")
        batch_size = data_spec.shape[0]
        uniform_key, order_key = jax.random.split(key)
        offsets = jax.random.uniform(
            uniform_key, shape=(num_draws, batch_size), dtype=jnp.float32
        )
        times = (jnp.arange(num_draws, dtype=jnp.float32)[:, None] + offsets) / num_draws
        # Independently randomize stratum order for each example without relying
        # on version-sensitive ``random.permutation(independent=True)``.
        order = jnp.argsort(
            jax.random.uniform(order_key, shape=(num_draws, batch_size)), axis=0
        )
        times = jnp.take_along_axis(times, order, axis=0)
        epsilon = jnp.asarray(self.safety_epsilon, dtype=jnp.float32)
        times = epsilon + times * (1.0 - 2.0 * epsilon)
        return times.reshape((num_draws, batch_size) + (1,) * (data_spec.ndim - 1))

    def __call__(self, key: jax.Array, data_spec: jax.Array) -> jax.Array:
        return self.sample_draws(key, data_spec, 1)[0]


class PrefixAmortizedSFTDiffusion(nn.Module):
    """Encode one long prefix/canvas history, then denoise K corruptions.

    Canvas selection is shared across the K draws because the KV-cache boundary
    depends on that selection.  Noise times, corruption, and self-conditioning
    are independent.  The K and batch axes are flattened only at the output so
    Hackable Diffusion's stock loss computes the mean of exactly the same
    examples as K independent forward passes, while gradients flow through one
    shared prefix encode.
    """

    gemma_network: Any
    corruption_process: Any
    time_sampler: Any
    prompt_len: int
    canvas_size: int
    num_canvases: int
    noise_draws: int

    x0: kd.kontext.Key
    prompt: kd.kontext.Key
    canvas_id: kd.kontext.Key
    canvas_mask: kd.kontext.Key
    canvas_loss_mask: kd.kontext.Key
    encoder_target: kd.kontext.Key
    encoder_target_mask: kd.kontext.Key

    pad_token: int = 0
    stop_gradient_from_denoiser_to_encoder: bool = False
    self_cond_prob: float = 0.5

    @property
    def total_canvas_len(self) -> int:
        return self.num_canvases * self.canvas_size

    @nn.compact
    def __call__(
        self,
        x0: jax.Array,
        prompt: jax.Array,
        canvas_id: jax.Array,
        canvas_mask: jax.Array,
        canvas_loss_mask: jax.Array,
        encoder_target: jax.Array,
        encoder_target_mask: jax.Array,
        is_training: bool = True,
    ) -> dict[str, Any]:
        batch_size = x0.shape[0]
        if self.noise_draws < 1:
            raise ValueError("noise_draws must be positive")

        first_token_indices = jnp.arange(self.num_canvases) * self.canvas_size
        canvas_validity = canvas_mask[:, first_token_indices]
        num_valid_canvases = jnp.maximum(jnp.sum(canvas_validity, axis=-1), 1)
        selected_canvas_idx = jax.random.randint(
            self.make_rng("sampling"),
            shape=num_valid_canvases.shape,
            minval=0,
            maxval=num_valid_canvases,
        )
        x0_tokens = x0[..., 0] if x0.ndim == 3 else x0
        selected_x0 = gather_selected_canvas(
            x0, selected_canvas_idx, self.canvas_size
        )
        selected_canvas_mask = gather_selected_canvas(
            canvas_mask, selected_canvas_idx, self.canvas_size
        )
        selected_canvas_loss_mask = gather_selected_canvas(
            canvas_loss_mask, selected_canvas_idx, self.canvas_size
        )
        selected_canvas_id = gather_selected_canvas(
            canvas_id, selected_canvas_idx, self.canvas_size
        )

        encoder_logits, kv_cache, positions, prompt_mask = encode_prefix(
            gemma_network=self.gemma_network,
            prompt=prompt,
            x0_tokens=x0_tokens,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=self.prompt_len,
            total_canvas_len=self.total_canvas_len,
            canvas_size=self.canvas_size,
            pad_token=self.pad_token,
        )
        if self.stop_gradient_from_denoiser_to_encoder:
            kv_cache = jax.lax.stop_gradient(kv_cache)

        times = sample_noise_draws(
            self.time_sampler,
            self.make_rng("sampling"),
            selected_x0,
            self.noise_draws,
        )
        corruption_keys = jax.random.split(
            self.make_rng("sampling"), self.noise_draws
        )
        xt, target_info = jax.vmap(
            lambda key, time: self.corruption_process.corrupt(
                key, selected_x0, time
            )
        )(corruption_keys, times)
        # CategoricalProcess emits a one-hot target logits tensor even though
        # NoWeightDiscreteLoss consumes integer x0 labels. At a 262K vocab this
        # dead field is enormous; remove it from the staged output tree so it
        # can never survive compiler DCE or inflate checkpoint/summary state.
        target_info = dict(target_info)
        target_info.pop("logits", None)

        def decode_draw(module, xt_draw, time_draw, sc_logits=None):
            return decode_selected_canvas(
                gemma_network=module,
                xt=xt_draw,
                time=time_draw,
                kv_cache=kv_cache,
                positions=positions,
                prompt_mask=prompt_mask,
                canvas_mask=canvas_mask,
                selected_canvas_idx=selected_canvas_idx,
                prompt_len=self.prompt_len,
                total_canvas_len=self.total_canvas_len,
                canvas_size=self.canvas_size,
                sc_logits=sc_logits,
                is_training=is_training,
            )

        # This must be Linen's lifted transform rather than raw ``jax.vmap``:
        # the target is a bound Module with parameter and intermediates
        # collections. Parameters are shared across draws while captured
        # intermediates acquire a draw axis. XLA can consequently fold K into
        # the effective batch instead of materializing K Python model calls.
        first_pass = nn.vmap(
            lambda module, draw_x, draw_t: decode_draw(
                module, draw_x, draw_t
            ),
            variable_axes={"params": None, "intermediates": 0},
            split_rngs={"params": False},
            in_axes=(0, 0),
            out_axes=0,
            axis_size=self.noise_draws,
        )(self.gemma_network, xt, times)
        converted_first = jax.vmap(
            lambda prediction, draw_x, draw_t: self.corruption_process.convert_predictions(
                prediction, draw_x, draw_t
            )
        )(first_pass, xt, times)
        converted_first = jax.lax.stop_gradient(converted_first)
        sc_logits = converted_first["logits"]
        do_self_condition = jax.random.uniform(
            self.make_rng("sampling"), shape=(self.noise_draws, batch_size)
        ) < self.self_cond_prob
        do_self_condition = do_self_condition.reshape(
            (self.noise_draws, batch_size) + (1,) * (sc_logits.ndim - 2)
        )
        sc_logits = jnp.where(do_self_condition, sc_logits, jnp.zeros_like(sc_logits))
        denoiser_output = nn.vmap(
            lambda module, draw_x, draw_t, draw_sc: decode_draw(
                module, draw_x, draw_t, draw_sc
            ),
            variable_axes={"params": None, "intermediates": 0},
            split_rngs={"params": False},
            in_axes=(0, 0, 0),
            out_axes=0,
            axis_size=self.noise_draws,
        )(self.gemma_network, xt, times, sc_logits)
        converted = jax.vmap(
            lambda prediction, draw_x, draw_t: self.corruption_process.convert_predictions(
                prediction, draw_x, draw_t
            )
        )(denoiser_output, xt, times)
        # The configured categorical loss needs logits only. Dropping the
        # derived argmax prediction keeps the train-step auxiliary tree small.
        converted = {"logits": converted["logits"]}
        noise_info = jax.vmap(self.corruption_process.get_schedule_info)(times)

        target_mask = (
            selected_canvas_mask
            & selected_canvas_loss_mask
            & (selected_canvas_id == selected_canvas_idx[:, None])
        )
        draw_target_mask = jnp.broadcast_to(
            target_mask[None, ..., None],
            (self.noise_draws,) + target_mask.shape + (1,),
        )
        target_info["is_corrupted"] = (
            target_info["is_corrupted"] & draw_target_mask
        )
        target_info["target_mask"] = draw_target_mask

        return {
            "output": flatten_draw_batch(converted, self.noise_draws, batch_size),
            "target": flatten_draw_batch(target_info, self.noise_draws, batch_size),
            "xt": flatten_draw_batch(xt, self.noise_draws, batch_size),
            "noise_info": flatten_draw_batch(
                noise_info, self.noise_draws, batch_size
            ),
            "encoder_logits": encoder_logits,
            "encoder_target": encoder_target,
            "encoder_target_mask": encoder_target_mask,
            "selected_canvas_idx": selected_canvas_idx,
        }


def sample_noise_draws(
    sampler: Any,
    key: jax.Array,
    data_spec: jax.Array,
    num_draws: int,
) -> jax.Array:
    if hasattr(sampler, "sample_draws"):
        return sampler.sample_draws(key, data_spec, num_draws)
    keys = jax.random.split(key, num_draws)
    return jax.vmap(lambda draw_key: sampler(draw_key, data_spec))(keys)


def encode_prefix(
    *,
    gemma_network: Any,
    prompt: jax.Array,
    x0_tokens: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    pad_token: int,
) -> tuple[jax.Array, Any, jax.Array, jax.Array]:
    """Prefill the clean prefix without importing upstream evaluator code."""
    del total_canvas_len
    full_sequence = jnp.concatenate([prompt, x0_tokens], axis=1)
    prompt_mask = prompt != pad_token
    full_sequence_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)
    kv_cache, encoder_logits, positions, _ = (
        hd_gemma_network.prefill_kv_cache_with_encoder(
            tokens=full_sequence,
            input_mask=full_sequence_mask,
            init_cache_fn=gemma_network.init_cache,
            encoder_fn=gemma_network.encoder_call,
        )
    )
    end_index = prompt_len + selected_canvas_idx * canvas_size
    kv_cache = mask_helpers.set_cache_end_index(kv_cache, end_index)
    return encoder_logits, kv_cache, positions, prompt_mask


def gather_selected_canvas(
    values: jax.Array, selected_canvas_idx: jax.Array, canvas_size: int
) -> jax.Array:
    """Gather one different contiguous canvas per batch element."""
    indices = selected_canvas_idx[:, None] * canvas_size + jnp.arange(canvas_size)
    indices = indices.reshape(indices.shape + (1,) * (values.ndim - 2))
    indices = jnp.broadcast_to(
        indices, indices.shape[:2] + values.shape[2:]
    )
    return jnp.take_along_axis(values, indices, axis=1)


def decode_selected_canvas(
    gemma_network: Any,
    *,
    xt: jax.Array,
    time: jax.Array,
    kv_cache: Any,
    positions: jax.Array,
    prompt_mask: jax.Array,
    canvas_mask: jax.Array,
    selected_canvas_idx: jax.Array,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    sc_logits: jax.Array | None = None,
    is_training: bool = True,
) -> dict[str, jax.Array]:
    """Decode only the selected 256-token canvas against the shared cache.

    Google's reference SFT helper computes vocabulary logits for every canvas
    and masks all but one in the loss. At 262K vocabulary that defeats the
    memory win of prefix amortization. This equivalent query gathers the
    selected positions and emits exactly ``canvas_size`` logits while prior
    clean canvases remain available in the prefilled cache.
    """
    attention_mask = mask_helpers.create_decoder_attention_mask(
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_size,
        num_queries=canvas_size,
    )
    all_canvas_positions = positions[:, prompt_len:]
    query_positions = gather_selected_canvas(
        all_canvas_positions, selected_canvas_idx, canvas_size
    )
    conditioning = {
        "kv_cache": kv_cache,
        "positions": query_positions,
        "attention_mask": attention_mask,
    }
    if sc_logits is not None:
        conditioning["sc_logits"] = sc_logits
    return gemma_network(
        xt=xt,
        time=time,
        conditioning=conditioning,
        is_training=is_training,
    )


def flatten_draw_batch(tree: Any, num_draws: int, batch_size: int) -> Any:
    """Flatten leading ``[K, B]`` axes without touching remaining structure."""

    def flatten(value):
        if value.shape[:2] != (num_draws, batch_size):
            raise ValueError(
                f"expected leading draw/batch shape {(num_draws, batch_size)}, "
                f"got {value.shape}"
            )
        return value.reshape((num_draws * batch_size,) + value.shape[2:])

    return jax.tree.map(flatten, tree)
