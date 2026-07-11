"""Kauldron config generated from the strict Sunfish TOML run contract."""

from __future__ import annotations

import os

import jax
from kauldron import konfig

from sunfish_tpu.training.runtime import ensure_run_identity
from sunfish_tpu.training.spec import (
    CheckpointFormat,
    HarnessConfig,
    Phase,
    TimeSampling,
)

# These imports become serializable Konfig constructor proxies. They resolve
# only after every worker has initialized distributed JAX in ``sunfish-train``.
with konfig.imports():
    from hackable_diffusion import hd
    from hackable_diffusion.kdiff import core
    from hackable_diffusion.lib.training import discrete_loss
    from kauldron import kd
    import optax
    from sunfish_tpu.training import checkpoint as sunfish_checkpoint
    from sunfish_tpu.training import data as sunfish_data
    from sunfish_tpu.training import losses as sunfish_losses
    from sunfish_tpu.training import model as sunfish_model
    from sunfish_tpu.training import prefix_amortized
    from sunfish_tpu.training import sharding as sunfish_sharding


def get_config():
    """Return one fully specified, resume-safe Kauldron trainer."""
    config_path = os.environ.get("SUNFISH_TRAIN_CONFIG")
    if not config_path:
        raise RuntimeError("SUNFISH_TRAIN_CONFIG is not set; launch with sunfish-train")
    spec = HarnessConfig.load(config_path)
    allow_non_tpu = os.environ.get("SUNFISH_ALLOW_NON_TPU") == "1"
    identity = ensure_run_identity(
        spec, jax, require_tpu_runtime=spec.topology.require_tpu and not allow_non_tpu
    )

    cfg = kd.train.Trainer()
    cfg.seed = spec.run.seed
    cfg.workdir = spec.run.workdir
    cfg.aux = {
        "run_id": spec.run.run_id,
        "phase": spec.run.phase.value,
        "config_sha256": spec.digest,
        "dataset_manifest_sha256": spec.data.manifest_sha256,
        "runtime_identity": identity,
        "noise_draws": spec.objective.noise_draws,
    }

    cfg.sharding = sunfish_sharding.make_training_sharding_for(
        phase=spec.run.phase.value,
        num_experts=spec.model.num_experts,
    )
    # Keep the adapter tree present (but optimizer-frozen) in the router phase.
    # Its checkpoint can then seed LoRA recovery without a structural rewrite.
    use_lora = spec.run.phase in {Phase.SMOKE, Phase.ROUTER, Phase.LORA}
    gemma_network = sunfish_model.make_gemma_network(
        num_experts=spec.model.num_experts,
        top_k_experts=spec.model.top_k_experts,
        dtype=spec.model.dtype,
        use_lora=use_lora,
        lora_rank=spec.optimizer.lora_rank,
    )
    corruption_process = hd.corruption.CategoricalProcess.uniform_process(
        num_categories=spec.model.vocab_size,
        schedule=hd.corruption.RFSchedule(),
    )
    if spec.objective.time_sampling is TimeSampling.PREFIX_STRATIFIED:
        time_sampler = prefix_amortized.PrefixStratifiedTimeSampler(
            safety_epsilon=spec.objective.safety_epsilon
        )
    elif spec.objective.time_sampling is TimeSampling.LOGIT_NORMAL:
        time_sampler = hd.training.time_sampling.LogitNormalTimeSampler(
            mean=spec.objective.logit_normal_mean,
            scale=spec.objective.logit_normal_scale,
            span=hd.jax_helpers.SafeSpan(
                safety_epsilon=spec.objective.safety_epsilon
            ),
        )
    else:
        time_sampler = hd.training.time_sampling.UniformTimeSampler(
            span=hd.jax_helpers.SafeSpan(
                safety_epsilon=spec.objective.safety_epsilon
            )
        )

    cfg.model = prefix_amortized.PrefixAmortizedSFTDiffusion(
        x0="batch.canvas",
        prompt="batch.prompt",
        canvas_id="batch.canvas_id",
        canvas_mask="batch.canvas_mask",
        canvas_loss_mask="batch.canvas_loss_mask",
        encoder_target="batch.encoder_target",
        encoder_target_mask="batch.encoder_target_mask",
        gemma_network=gemma_network,
        corruption_process=corruption_process,
        time_sampler=time_sampler,
        prompt_len=spec.data.prompt_length,
        canvas_size=spec.data.canvas_size,
        num_canvases=spec.data.num_canvases,
        noise_draws=spec.objective.noise_draws,
        pad_token=spec.data.pad_token,
        self_cond_prob=spec.objective.self_condition_probability,
        stop_gradient_from_denoiser_to_encoder=(
            spec.objective.stop_gradient_from_denoiser_to_encoder
        ),
    )
    cfg.train_losses = {
        "diffusion_loss": core.KauldronLossWrapper(
            loss=discrete_loss.NoWeightDiscreteLoss(
                use_mask=True,
                mask_key="target_mask",
            ),
            weight=spec.objective.decoder_loss_weight,
        ),
        "encoder_loss": sunfish_losses.EncoderARLoss(
            encoder_logits="preds.encoder_logits",
            encoder_target="preds.encoder_target",
            encoder_target_mask="preds.encoder_target_mask",
            weight=spec.objective.encoder_loss_weight,
        ),
    }

    cfg.num_train_steps = spec.training.steps
    cfg.log_metrics_every = spec.training.log_metrics_every_steps
    cfg.log_summaries_every = spec.training.log_summaries_every_steps
    cfg.schedules = {
        "learning_rate": optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=spec.optimizer.peak_learning_rate,
            end_value=spec.optimizer.end_learning_rate,
            warmup_steps=spec.optimizer.warmup_steps,
            decay_steps=spec.training.steps,
        )
    }
    base_optimizer = kd.optim.named_chain(**{
        "clip": optax.clip_by_global_norm(spec.optimizer.gradient_clip_norm),
        "adam": optax.scale_by_adam(
            b1=spec.optimizer.adam_beta1,
            b2=spec.optimizer.adam_beta2,
            eps=spec.optimizer.adam_epsilon,
        ),
        "decay": optax.add_decayed_weights(spec.optimizer.weight_decay),
        "learning_rate": optax.scale_by_learning_rate(
            cfg.ref.schedules["learning_rate"]
        ),
    })
    base_optimizer = optax.apply_if_finite(
        # Never let a persistent non-finite gradient poison model or optimizer
        # state. The loss remains non-finite and visible in logs, but every
        # affected update is rejected for the entire bounded run.
        base_optimizer,
        max_consecutive_errors=spec.training.steps,
    )
    if spec.run.phase in {Phase.SMOKE, Phase.LORA}:
        cfg.optimizer = kd.optim.partial_updates(
            optimizer=base_optimizer,
            mask=kd.optim.select("lora"),
        )
    elif spec.run.phase is Phase.ROUTER:
        cfg.optimizer = kd.optim.partial_updates(
            optimizer=base_optimizer,
            mask=kd.optim.select(
                ["router_logits", "per_expert_scale", "router_scale"]
            ),
        )
    else:
        cfg.optimizer = base_optimizer

    cfg.train_ds = sunfish_data.SunfishData(
        directory=spec.data.directory,
        expected_manifest_sha256=spec.data.manifest_sha256,
        verify_shard_hashes=spec.data.verify_shard_hashes,
        prompt_length=spec.data.prompt_length,
        canvas_size=spec.data.canvas_size,
        num_canvases=spec.data.num_canvases,
        vocab_size=spec.model.vocab_size,
        pad_token=spec.data.pad_token,
        eos_token=spec.data.eos_token,
        batch_size=spec.data.global_batch_size,
        seed=spec.run.seed,
        shuffle=spec.data.shuffle,
        num_epochs=None,
        batch_drop_remainder=True,
        num_workers=spec.data.num_workers,
        per_worker_buffer_size=2,
        shard_by_process=True,
    )
    if spec.checkpoint.format is CheckpointFormat.EXACT_TREE:
        cfg.init_transform = sunfish_checkpoint.ShardedOrbaxInitLoader(
            path=spec.checkpoint.init_path,
            model_param_path=spec.checkpoint.model_param_path,
        )
    else:
        cfg.init_transform = sunfish_checkpoint.ShardedKauldronParamsInitLoader(
            workdir=spec.checkpoint.init_path,
            step=spec.checkpoint.init_step,
            model_param_path=spec.checkpoint.model_param_path,
        )
    cfg.checkpointer = kd.ckpts.Checkpointer(
        fast=False,
        never_save_step_zero=True,
        save_interval_steps=spec.training.checkpoint_every_steps,
        max_to_keep=spec.training.max_checkpoints_to_keep,
    )
    cfg.rng_streams = kd.train.RngStreams([
        kd.train.RngStream("default", train=True, eval=True),
        kd.train.RngStream("sampling", train=True, eval=True),
    ])
    cfg.evals = {}
    cfg._konfig_experimental_nofreeze = True  # pylint: disable=protected-access
    return cfg
