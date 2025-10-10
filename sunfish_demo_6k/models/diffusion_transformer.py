"""
SunFish Diffusion Transformer Model
Implements continuous embedding space diffusion (Section 6-7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from typing import Optional, Union
from transformers import GPT2Tokenizer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps (Section 6)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: tensor of shape [batch_size] with timestep indices
        Returns:
            tensor of shape [batch_size, d_model]
        """
        return self.pe[t]


class TimestepEmbedding(nn.Module):
    """MLP for processing timestep embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class PromptEncoder(nn.Module):
    """
    Encodes text prompts for conditional generation.
    Uses GPT-2 tokenizer and learnable projection to model dimension.
    """

    def __init__(self, dim: int, max_length: int = 77):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self._tokenizer = None
        self._prompt_embeddings = None

        # Learnable projection from GPT-2 embedding space to model space
        # GPT-2 uses 768-dim embeddings
        self.prompt_projection = nn.Sequential(
            nn.Linear(768, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def prompt_embeddings(self):
        """Lazy load frozen GPT-2 embeddings."""
        if self._prompt_embeddings is None:
            from transformers import GPT2Model
            gpt2 = GPT2Model.from_pretrained("gpt2")
            self._prompt_embeddings = gpt2.wte  # Word token embeddings
            # Freeze the embeddings
            for param in self._prompt_embeddings.parameters():
                param.requires_grad = False
        return self._prompt_embeddings

    def encode(self, prompts: Union[str, list]) -> torch.Tensor:
        """
        Encode text prompts to continuous embeddings.

        Args:
            prompts: Single string or list of strings

        Returns:
            Prompt embeddings [batch_size, max_length, dim]
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize prompts
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(self.prompt_embeddings.weight.device)

        # Get frozen GPT-2 embeddings
        with torch.no_grad():
            prompt_emb = self.prompt_embeddings(input_ids)  # [batch, max_length, 768]

        # Project to model dimension
        prompt_emb = self.prompt_projection(prompt_emb)  # [batch, max_length, dim]

        return prompt_emb


class TransformerBlockWithCrossAttention(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and FFN.
    Implements pre-norm architecture for stability.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention on noisy embeddings
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to prompt embeddings
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence [batch, seq_len, d_model]
            context: Optional prompt embeddings [batch, prompt_len, d_model]

        Returns:
            Output sequence [batch, seq_len, d_model]
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        # Pre-norm cross-attention (if context provided)
        if context is not None:
            residual = x
            x = self.norm2(x)
            x, _ = self.cross_attn(x, context, context)
            x = residual + x

        # Pre-norm feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        return x


class SunFishTransformer(pl.LightningModule):
    """
    SunFish Diffusion Transformer model.

    This model implements continuous embedding space diffusion for language modeling.
    Unlike traditional autoregressive LLMs, it generates text by denoising in embedding space.

    Architecture:
    - Token embeddings for discrete -> continuous conversion
    - Positional embeddings for sequence positions
    - Timestep embeddings for diffusion time conditioning
    - Transformer encoder backbone
    - Noise prediction head

    Reference: Technical report Sections 6-7
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # ====================================================================
        # Embedding Layers
        # ====================================================================

        # Token embedding: vocab -> continuous space
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional embedding: sequence positions
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )

        # Timestep encoding and projection
        self.time_encoder = PositionalEncoding(config.n_embd, config.timesteps)
        self.time_mlp = TimestepEmbedding(config.n_embd)

        # Prompt encoder for conditional generation
        prompt_max_length = getattr(config, "prompt_max_length", 77)
        self.prompt_encoder = PromptEncoder(config.n_embd, max_length=prompt_max_length)

        # ====================================================================
        # Transformer Backbone
        # ====================================================================

        # Custom transformer with cross-attention for prompt conditioning
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithCrossAttention(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
            )
            for _ in range(config.n_layer)
        ])

        # ====================================================================
        # Output Layers
        # ====================================================================

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.noise_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # ====================================================================
        # Diffusion Schedule (Section 2)
        # ====================================================================

        self._setup_diffusion_schedule()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using GPT-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _setup_diffusion_schedule(self):
        """
        Pre-compute diffusion schedule constants (Section 2).

        Uses linear beta schedule: β_t = β_start + t * (β_end - β_start) / T
        """
        betas = torch.linspace(
            self.config.beta_start, self.config.beta_end, self.config.timesteps
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register as buffers (not parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the denoiser network.

        Args:
            x_t: Noisy embeddings [batch, seq_len, n_embd]
            t: Timesteps [batch] - integer indices into diffusion schedule
            context: Optional prompt embeddings [batch, prompt_len, n_embd]

        Returns:
            predicted_noise: [batch, seq_len, n_embd]
        """
        batch_size, seq_len, _ = x_t.shape

        # Get time embeddings
        time_emb = self.time_encoder(t)  # [batch, n_embd]
        time_emb = self.time_mlp(time_emb)  # [batch, n_embd]
        time_emb = time_emb.unsqueeze(1)  # [batch, 1, n_embd]

        # Add time conditioning to noisy input
        x = x_t + time_emb

        # Pass through transformer blocks with cross-attention
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Final layer norm and noise prediction
        x = self.ln_f(x)
        predicted_noise = self.noise_head(x)

        return predicted_noise

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).

        Adds noise to clean embeddings using the closed-form equation:
        x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε

        Args:
            x_0: Clean embeddings [batch, seq_len, n_embd]
            t: Timesteps [batch]
            noise: Optional pre-sampled noise (if None, samples new noise)

        Returns:
            x_t: Noisy embeddings [batch, seq_len, n_embd]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1
        )

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t

    def training_step(self, batch, batch_idx):
        """
        Training step implementing classifier-free guidance training.

        Loss: E[||ε - ε_θ(x_t, t, c)||²]
        where c is randomly dropped out for unconditional training

        Args:
            batch: Token IDs [batch, seq_len] or dict with 'tokens' and 'prompts'
            batch_idx: Batch index

        Returns:
            loss: MSE between true noise and predicted noise
        """
        # Handle both simple token batches and dict batches with prompts
        if isinstance(batch, dict):
            token_ids = batch['tokens']
            prompts = batch.get('prompts', None)
        else:
            token_ids = batch
            prompts = None

        # 1. Get clean embeddings (x_0) from token IDs
        x_0 = self.token_embedding(token_ids)
        x_0 = x_0 + self.pos_embedding[:, : token_ids.shape[1], :]

        # 2. Sample random timesteps uniformly
        t = torch.randint(
            0, self.config.timesteps, (x_0.shape[0],), device=self.device
        ).long()

        # 3. Sample noise from standard Gaussian
        noise = torch.randn_like(x_0)

        # 4. Create noisy sample x_t using forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # 5. Prepare prompt conditioning with classifier-free guidance dropout
        context = None
        conditioning_dropout = getattr(self.config, "conditioning_dropout", 0.1)

        if prompts is not None and len(prompts) > 0:
            # Encode prompts
            prompt_embeddings = self.prompt_encoder.encode(prompts)

            # Apply conditioning dropout (randomly drop prompts for CFG training)
            if self.training and conditioning_dropout > 0:
                batch_size = len(prompts)
                # Randomly select which samples should be unconditional
                drop_mask = torch.rand(batch_size, device=self.device) < conditioning_dropout
                # Set dropped prompts to None by zeroing them out
                prompt_embeddings = prompt_embeddings * (~drop_mask).view(-1, 1, 1).float()

            context = prompt_embeddings

        # 6. Predict noise using the model (with or without prompt conditioning)
        predicted_noise = self(x_t, t, context=context)

        # 7. Compute simplified MSE loss
        loss = F.mse_loss(noise, predicted_noise)

        # 8. Log metrics (only if attached to trainer)
        try:
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            # Log learning rate
            if self.trainer and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log("lr", current_lr, on_step=True, logger=True)

            # Log conditioning dropout rate
            if context is not None:
                self.log("cfg_dropout", conditioning_dropout, on_step=False, on_epoch=True, logger=True)
        except (RuntimeError, AttributeError):
            pass  # Not attached to trainer yet, skip logging

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing schedule."""

        # AdamW optimizer (Section 8)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.config.warmup_steps:
                return current_step / max(1, self.config.warmup_steps)

            # Cosine annealing phase
            progress = (current_step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def get_num_params(self, non_embedding: bool = False):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.numel()

        return n_params


if __name__ == "__main__":
    # Quick test
    from config.tiny_config import get_tiny_config

    config = get_tiny_config()
    model = SunFishTransformer(config)

    print(f"\n✅ Model initialized successfully!")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        # Get embeddings
        x_0 = model.token_embedding(token_ids) + model.pos_embedding[:, :seq_len, :]

        # Sample timestep
        t = torch.randint(0, config.timesteps, (batch_size,))

        # Add noise
        noise = torch.randn_like(x_0)
        x_t = model.q_sample(x_0, t, noise)

        # Predict noise
        pred_noise = model(x_t, t)

        print(f"\n✅ Forward pass successful!")
        print(f"Input shape: {x_t.shape}")
        print(f"Output shape: {pred_noise.shape}")
        print(f"Noise MSE: {F.mse_loss(noise, pred_noise):.4f}")
