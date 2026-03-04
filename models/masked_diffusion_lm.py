"""
Masked Diffusion Language Model
Implements discrete masked diffusion following Dream 7B and RND1 approaches.
Uses Qwen3-0.6B as pretrained base.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer


class MaskedDiffusionLM(pl.LightningModule):
    """
    Discrete Masked Diffusion Language Model.

    Key differences from continuous diffusion:
    - Uses [MASK] token instead of Gaussian noise
    - Cross-entropy loss instead of MSE
    - Direct token prediction instead of embedding denoising
    - Pretrained initialization from Qwen3-0.6B
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # ====================================================================
        # Load Pretrained Qwen3 Model
        # ====================================================================
        print(f"Loading pretrained model: {config.base_model}")
        model_dtype = torch.bfloat16 if "bf16" in config.precision else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            dtype=model_dtype,  # Use dtype instead of deprecated torch_dtype
            trust_remote_code=True,
        )

        # Disable causal masking for bidirectional attention
        if getattr(config, "bidirectional", True):
            self._disable_causal_masking()

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # ====================================================================
        # Load Tokenizer and Add [MASK] Token
        # ====================================================================
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )

        # Add [MASK] token if not present
        added_mask_token = False
        if self.tokenizer.mask_token is None:
            special_tokens = {"mask_token": "[MASK]"}
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                added_mask_token = True
                print(f"Added [MASK] token, new vocab size: {len(self.tokenizer)}")

        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = len(self.tokenizer)

        # Initialize [MASK] embedding if newly added
        if added_mask_token:
            self._init_mask_embedding()

        # ====================================================================
        # Store Original Attention Configuration for Bidirectional
        # ====================================================================
        # We'll handle bidirectional attention by passing full attention mask

    def _init_mask_embedding(self):
        """Initialize [MASK] embedding as mean of existing embeddings."""
        with torch.no_grad():
            embed = self.model.get_input_embeddings()
            # Use mean of all embeddings except the newly added mask token
            embed.weight[self.mask_token_id] = embed.weight[:-1].mean(dim=0)
            print(f"Initialized [MASK] embedding (id={self.mask_token_id})")

    def _disable_causal_masking(self):
        """Disable causal masking flags throughout the model."""
        disabled = 0
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "is_causal"):
                self.model.config.is_causal = False
                disabled += 1
            if hasattr(self.model.config, "is_decoder"):
                self.model.config.is_decoder = False
                disabled += 1

        for module in self.model.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
                disabled += 1
            if hasattr(module, "is_decoder"):
                module.is_decoder = False
                disabled += 1

        print(f"Bidirectional attention enabled (disabled {disabled} causal flags)")

    def get_mask_rate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get mask rate alpha_t for timestep t.

        Linear schedule: alpha_t = t / T
        At t=0: no masking
        At t=T: fully masked
        """
        if self.config.mask_schedule == "linear":
            return t.float() / self.config.timesteps
        elif self.config.mask_schedule == "cosine":
            # Cosine schedule for smoother masking
            return 1 - torch.cos(t.float() / self.config.timesteps * math.pi / 2)
        else:
            return t.float() / self.config.timesteps

    def _get_move_chance(self, t_float: torch.Tensor) -> torch.Tensor:
        """Masking probability at normalized time t in [0, 1] (for sampling).

        t=0 means clean (no masking), t=1 means fully masked.
        """
        if self.config.mask_schedule == "cosine":
            return 1.0 - torch.cos(t_float * math.pi / 2)
        return t_float

    def forward_mask(self, token_ids: torch.Tensor, t: torch.Tensor):
        """
        Apply masking at timestep t.

        Args:
            token_ids: [batch, seq] - original token IDs
            t: [batch] - timesteps (0 to timesteps-1)

        Returns:
            masked_tokens: [batch, seq] - tokens with some replaced by [MASK]
            mask: [batch, seq] - boolean mask where True = position was masked
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Get mask rate for each sample
        alpha_t = self.get_mask_rate(t)  # [batch]
        mask_prob = alpha_t.unsqueeze(1).expand(batch_size, seq_len)  # [batch, seq]

        # Sample mask
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob

        # Apply masking
        masked_tokens = token_ids.clone()
        masked_tokens[mask] = self.mask_token_id

        return masked_tokens, mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass with bidirectional attention.

        Args:
            input_ids: [batch, seq] - input token IDs (with [MASK] tokens)
            attention_mask: [batch, seq] - attention mask (optional)

        Returns:
            logits: [batch, seq, vocab_size] or [batch, seq-1, vocab_size] if shift
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Build a non-causal attention mask mapping for Qwen3.
        # Qwen3Model treats a dict as a pre-built mask mapping and skips causal mask creation.
        # Additive mask in model dtype: 0 = attend, -inf = ignore.
        mask_dtype = next(self.parameters()).dtype
        if not mask_dtype.is_floating_point:
            mask_dtype = torch.float32
        if attention_mask is None:
            full_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                dtype=mask_dtype,
                device=device,
            )
        else:
            # attention_mask is [batch, seq] with 1 for tokens to keep, 0 for padding
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D [batch, seq]")
            keep = attention_mask[:, None, None, :].to(dtype=mask_dtype, device=device)
            full_mask = (1.0 - keep) * torch.finfo(mask_dtype).min
            full_mask = full_mask.expand(batch_size, 1, seq_len, seq_len)

        # Build mask mapping keyed by layer type
        layer_types = getattr(self.model.config, "layer_types", ["full_attention"])
        mask_mapping = {layer_type: full_mask for layer_type in set(layer_types)}

        # Forward through model with explicit bidirectional attention mask mapping
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=mask_mapping,
            output_hidden_states=True,
            use_cache=False,  # Disable KV cache for bidirectional
        )

        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden_dim]

        # Apply shift operation if configured (predict i+1 from hidden i)
        if self.config.use_shift:
            hidden = hidden[:, :-1, :]  # [batch, seq-1, hidden_dim]

        # Get logits from language model head
        logits = self.model.lm_head(hidden)  # [batch, seq, vocab_size]

        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step with masked cross-entropy loss.

        Loss is computed only on masked positions.
        """
        # Handle different batch formats
        if isinstance(batch, dict):
            token_ids = batch["input_ids"]
        else:
            token_ids = batch

        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Sample random timesteps
        t = torch.randint(
            1, self.config.timesteps + 1,  # Include full-mask timestep
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        # Apply masking
        masked_tokens, mask = self.forward_mask(token_ids, t)

        # Forward pass
        logits = self.forward(masked_tokens)  # [batch, seq, vocab] or [batch, seq-1, vocab]

        # Compute loss
        if self.config.use_shift:
            # Shift targets to align with predictions
            targets = token_ids[:, 1:]  # [batch, seq-1]
            mask_shifted = mask[:, 1:]  # [batch, seq-1]
        else:
            targets = token_ids
            mask_shifted = mask

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, self.vocab_size)  # [batch*seq, vocab]
        if self.mask_token_id is not None:
            logits_flat = logits_flat.clone()
            logits_flat[:, self.mask_token_id] = float("-inf")
        targets_flat = targets.reshape(-1)  # [batch*seq]
        mask_flat = mask_shifted.reshape(-1).float()  # [batch*seq]

        # Cross-entropy loss on all positions
        loss_per_token = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
        )

        # Apply mask: only count loss on masked positions
        masked_loss = loss_per_token * mask_flat
        num_masked = mask_flat.sum().clamp(min=1)
        loss = masked_loss.sum() / num_masked

        # Logging (only if attached to a trainer)
        try:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("num_masked", num_masked, on_step=True, logger=True)
            self.log("mask_rate", mask_flat.mean(), on_step=True, logger=True)

            # Log learning rate
            if self.trainer and self.trainer.optimizers:
                current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log("lr", current_lr, on_step=True, logger=True)
        except RuntimeError:
            # Not attached to trainer (e.g., during testing)
            pass

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, dict):
            token_ids = batch["input_ids"]
        else:
            token_ids = batch

        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Use middle timestep for validation
        t = torch.full((batch_size,), self.config.timesteps // 2, device=device, dtype=torch.long)

        masked_tokens, mask = self.forward_mask(token_ids, t)
        logits = self.forward(masked_tokens)

        if self.config.use_shift:
            targets = token_ids[:, 1:]
            mask_shifted = mask[:, 1:]
        else:
            targets = token_ids
            mask_shifted = mask

        logits_flat = logits.reshape(-1, self.vocab_size)
        if self.mask_token_id is not None:
            logits_flat = logits_flat.clone()
            logits_flat[:, self.mask_token_id] = float("-inf")
        targets_flat = targets.reshape(-1)
        mask_flat = mask_shifted.reshape(-1).float()

        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        masked_loss = loss_per_token * mask_flat
        num_masked = mask_flat.sum().clamp(min=1)
        loss = masked_loss.sum() / num_masked

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # Compute accuracy on masked positions
        predictions = logits.argmax(dim=-1)
        if self.config.use_shift:
            correct = (predictions == targets) & mask_shifted
        else:
            correct = (predictions == targets) & mask
        accuracy = correct.float().sum() / num_masked

        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer with A2D warmup schedule.

        Phase 1 (0 to a2d_warmup_steps): 10% LR for bidirectional adaptation
        Phase 2 (a2d_warmup_steps to warmup_steps): Linear warmup to full LR
        Phase 3 (warmup_steps to max_steps): Cosine decay
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        def lr_lambda(current_step):
            a2d_warmup = self.config.a2d_warmup_steps
            warmup = self.config.warmup_steps
            max_steps = self.config.max_steps

            if current_step < a2d_warmup:
                # Phase 1: A2D warmup at 10% LR
                return 0.1

            elif current_step < warmup:
                # Phase 2: Linear warmup from 10% to 100%
                progress = (current_step - a2d_warmup) / max(1, warmup - a2d_warmup)
                return 0.1 + 0.9 * progress

            else:
                # Phase 3: Cosine decay
                progress = (current_step - warmup) / max(1, max_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            embed_params = self.model.get_input_embeddings().weight.numel()
            n_params -= embed_params

        return n_params

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """
        Generate text using MDLM ancestral posterior sampling.

        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of sequences
            num_steps: Number of reverse diffusion steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            generated_ids: [batch_size, seq_len] - generated token IDs
        """
        device = next(self.parameters()).device

        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        eps = 1e-4
        timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i]
            s = timesteps[i + 1]
            move_chance_t = self._get_move_chance(t)
            move_chance_s = self._get_move_chance(s)

            logits = self.forward(tokens) / temperature
            logits[..., self.mask_token_id] = float("-inf")

            if top_k > 0:
                indices_to_remove = (
                    logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
                )
                logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            p_x0 = F.softmax(logits, dim=-1).to(torch.float64)
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[..., self.mask_token_id] = move_chance_s

            gumbel_norm = 1e-10 - (torch.rand_like(q_xs) + 1e-10).log()
            sampled = (q_xs / gumbel_norm).argmax(dim=-1)

            is_masked = tokens == self.mask_token_id
            tokens = torch.where(is_masked, sampled, tokens)

        still_masked = tokens == self.mask_token_id
        if still_masked.any():
            logits = self.forward(tokens)
            logits[..., self.mask_token_id] = float("-inf")
            final_preds = logits.argmax(dim=-1)
            tokens = torch.where(still_masked, final_preds, tokens)

        return tokens

    def decode(self, token_ids: torch.Tensor) -> list:
        """Decode token IDs to text."""
        texts = []
        for ids in token_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        return texts


if __name__ == "__main__":
    # Quick test
    from config.qwen_masked_config import get_qwen_masked_config_cpu

    config = get_qwen_masked_config_cpu()
    print(f"\nConfiguration:\n{config}")

    print("\nInitializing model...")
    model = MaskedDiffusionLM(config)

    print(f"\nModel initialized!")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Mask token ID: {model.mask_token_id}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 32
    token_ids = torch.randint(0, model.vocab_size - 1, (batch_size, seq_len))

    with torch.no_grad():
        # Test masking
        t = torch.tensor([500, 250])
        masked_tokens, mask = model.forward_mask(token_ids, t)
        print(f"Mask rate: {mask.float().mean():.2%}")

        # Test forward
        logits = model.forward(masked_tokens)
        print(f"Logits shape: {logits.shape}")

    print("\nTest passed!")
