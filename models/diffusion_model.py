"""
Diffusion Transformer Model for Language Modeling
Based on RND1 and LLaDA research

This model implements:
- Bidirectional transformer encoder
- Timestep conditioning via sinusoidal embeddings + MLP
- Continuous diffusion in embedding space
- Noise prediction objective (epsilon-prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.

    Uses the same formula as in Attention Is All You Need but for scalar timesteps.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] tensor of timestep indices

        Returns:
            [batch_size, dim] positional encodings
        """
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding module.

    Converts scalar timesteps to d_model-dimensional embeddings via:
    1. Sinusoidal positional encoding
    2. MLP projection
    """

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEncoding(d_model, max_period)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] tensor of timestep indices

        Returns:
            [batch_size, d_model] timestep embeddings
        """
        emb = self.sinusoidal(timesteps)
        emb = self.mlp(emb)
        return emb


class DiffusionTransformer(pl.LightningModule):
    """
    Transformer-based diffusion model for language generation.

    Architecture:
    - Token embedding layer
    - Learned positional embeddings
    - Timestep embedding MLP
    - N transformer encoder layers (bidirectional attention)
    - Output projection to predict noise in embedding space
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # ========================================================================
        # Embeddings
        # ========================================================================
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Learned positional embeddings (for sequence positions)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )

        # Timestep embedding (for diffusion timesteps)
        self.timestep_embedding = TimestepEmbedding(config.n_embd)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.dropout)

        # ========================================================================
        # Transformer Encoder
        # ========================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=config.layer_norm_epsilon,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layer,
        )

        # ========================================================================
        # Output Layers
        # ========================================================================
        self.final_layer_norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Noise prediction head (predicts noise in embedding space)
        self.noise_pred_head = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)

        # ========================================================================
        # Diffusion Schedule
        # ========================================================================
        self._init_diffusion_schedule()

        # ========================================================================
        # Initialize Weights
        # ========================================================================
        self.apply(self._init_weights)

        # Special initialization for certain layers
        self._init_special_weights()

        logger.info(f"Model initialized with {self.count_parameters():,} parameters")

    def _init_diffusion_schedule(self):
        """
        Precompute diffusion schedule constants.

        Following DDPM, we compute:
        - betas: variance schedule
        - alphas: 1 - betas
        - alphas_cumprod: cumulative product of alphas
        - sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod: for q(x_t | x_0)
        """
        # Beta schedule
        if self.config.beta_schedule == "linear":
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.timesteps,
                dtype=torch.float32,
            )
        elif self.config.beta_schedule == "cosine":
            # Cosine schedule (improved DDPM)
            steps = self.config.timesteps + 1
            x = torch.linspace(0, self.config.timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / self.config.timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.config.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register as buffers (non-trainable, but part of state)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Precompute sqrt_recip_alphas_cumprod for denoising
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

    def _init_weights(self, module):
        """Initialize weights (GPT-style initialization)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _init_special_weights(self):
        """Special initialization for specific layers."""
        # Initialize position embeddings from N(0, 0.02)
        torch.nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict noise given noisy embeddings and timestep.

        Args:
            noisy_embeddings: [batch, seq_len, d_model] noisy token embeddings
            timesteps: [batch] diffusion timestep indices
            attention_mask: [batch, seq_len] optional attention mask (1=attend, 0=ignore)

        Returns:
            predicted_noise: [batch, seq_len, d_model] predicted noise
        """
        batch_size, seq_len, _ = noisy_embeddings.shape

        # Get timestep embeddings [batch, d_model]
        t_emb = self.timestep_embedding(timesteps)  # [batch, d_model]
        t_emb = t_emb.unsqueeze(1)  # [batch, 1, d_model]

        # Add positional embeddings to noisy input
        pos_emb = self.position_embedding[:, :seq_len, :]  # [1, seq_len, d_model]

        # Combine: noisy_embeddings + positional + timestep
        x = noisy_embeddings + pos_emb + t_emb

        # Apply embedding dropout
        x = self.emb_dropout(x)

        # Create attention mask for transformer (convert 0/1 to True/False)
        # Transformer expects: True = ignore, False = attend
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Pass through transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Final layer norm
        x = self.final_layer_norm(x)

        # Predict noise
        predicted_noise = self.noise_pred_head(x)

        return predicted_noise

    def get_noisy_embeddings(
        self,
        token_ids: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get noisy embeddings using the forward diffusion process.

        Args:
            token_ids: [batch, seq_len] token indices
            timesteps: [batch] timestep indices
            noise: [batch, seq_len, d_model] optional pre-sampled noise

        Returns:
            noisy_embeddings: [batch, seq_len, d_model]
            noise: [batch, seq_len, d_model] the noise that was added
        """
        # Get clean embeddings (x_0)
        clean_embeddings = self.token_embedding(token_ids)

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(clean_embeddings)

        # Get schedule coefficients for the given timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]  # [batch]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]  # [batch]

        # Reshape for broadcasting: [batch] -> [batch, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_embeddings = sqrt_alpha_prod * clean_embeddings + sqrt_one_minus_alpha_prod * noise

        return noisy_embeddings, noise

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step: predict noise added to embeddings.

        Args:
            batch: [batch, seq_len] token indices
            batch_idx: batch index

        Returns:
            loss: scalar loss tensor
        """
        token_ids = batch

        # Sample random timesteps for each sequence in the batch
        batch_size = token_ids.shape[0]
        timesteps = torch.randint(
            0,
            self.config.timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Get noisy embeddings via forward diffusion
        noisy_embeddings, noise = self.get_noisy_embeddings(token_ids, timesteps)

        # Predict the noise
        predicted_noise = self.forward(noisy_embeddings, timesteps)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters by layer type if using layer-wise LR
        if self.config.use_layer_wise_lr:
            attention_params = []
            ffn_params = []
            other_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue

                # Classify parameter
                if "self_attn" in name or "norm1" in name:
                    attention_params.append(param)
                elif "linear1" in name or "linear2" in name or "norm2" in name:
                    ffn_params.append(param)
                else:
                    other_params.append(param)

            # Create parameter groups with different LRs
            param_groups = [
                {
                    "params": attention_params,
                    "lr": self.config.learning_rate * self.config.attention_lr_multiplier,
                    "name": "attention",
                },
                {
                    "params": ffn_params,
                    "lr": self.config.learning_rate * self.config.ffn_lr_multiplier,
                    "name": "ffn",
                },
                {"params": other_params, "lr": self.config.learning_rate, "name": "other"},
            ]

            logger.info(
                f"Using layer-wise LR: "
                f"attention={self.config.learning_rate * self.config.attention_lr_multiplier:.2e}, "
                f"ffn={self.config.learning_rate * self.config.ffn_lr_multiplier:.2e}"
            )
        else:
            param_groups = self.parameters()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

        # Create learning rate scheduler
        if self.config.lr_scheduler == "cosine":

            def lr_lambda(current_step: int) -> float:
                # Warmup
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))

                # Cosine decay
                progress = (current_step - self.config.warmup_steps) / float(
                    max(1, self.config.max_steps - self.config.warmup_steps)
                )
                progress = min(1.0, max(0.0, progress))

                # Cosine annealing from 1.0 to min_lr_ratio
                min_lr_ratio = self.config.min_lr / self.config.learning_rate
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        elif self.config.lr_scheduler == "linear":

            def lr_lambda(current_step: int) -> float:
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return max(
                    0.0,
                    float(self.config.max_steps - current_step)
                    / float(max(1, self.config.max_steps - self.config.warmup_steps)),
                )

        else:  # constant

            def lr_lambda(current_step: int) -> float:
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    # Test the model
    import sys

    sys.path.append("..")
    from config.model_config import get_config_tiny

    logging.basicConfig(level=logging.INFO)

    config = get_config_tiny()
    model = DiffusionTransformer(config)

    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    seq_len = config.block_size
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    timesteps = torch.randint(0, config.timesteps, (batch_size,))

    noisy_emb, noise = model.get_noisy_embeddings(token_ids, timesteps)
    predicted_noise = model(noisy_emb, timesteps)

    print(f"\nInput shape: {token_ids.shape}")
    print(f"Noisy embeddings shape: {noisy_emb.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Loss: {F.mse_loss(predicted_noise, noise).item():.4f}")
