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
        # Use float32 masks to satisfy XLA/SDPA dtype requirements.
        mask_dtype = torch.float32
        if attention_mask is None:
            full_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len,
                dtype=mask_dtype,
                device=device,
            )
        else:
            # attention_mask is [batch, seq] with 1 for tokens to keep, 0 for padding
            # Convert to additive mask: 0 for keep, -inf for pad
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D [batch, seq]")
            keep = attention_mask[:, None, None, :].to(dtype=mask_dtype, device=device)
            neg_inf = torch.finfo(mask_dtype).min
            full_mask = (1.0 - keep) * neg_inf
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
        Generate text using iterative unmasking.

        Args:
            batch_size: Number of sequences to generate
            seq_len: Length of sequences
            num_steps: Number of unmasking steps
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            generated_ids: [batch_size, seq_len] - generated token IDs
        """
        device = next(self.parameters()).device

        # Start fully masked
        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        is_masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        for step in range(num_steps):
            # Get predictions
            logits = self.forward(tokens) / temperature  # [batch, seq, vocab]

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample tokens
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1,
            ).view(batch_size, -1)

            # Confidence scores for masked positions
            confidence = probs.max(dim=-1).values  # [batch, seq]
            confidence[~is_masked] = -1  # Don't re-unmask

            # Calculate how many to unmask this step
            remaining_steps = num_steps - step
            remaining_masked = is_masked.sum().item()
            num_to_unmask = max(1, remaining_masked // remaining_steps)

            # Find top confident positions across batch
            flat_confidence = confidence.view(-1)
            flat_is_masked = is_masked.view(-1)

            # Only consider masked positions
            masked_indices = flat_is_masked.nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                break

            masked_confidences = flat_confidence[masked_indices]
            num_to_unmask = min(num_to_unmask, len(masked_indices))

            _, top_indices = masked_confidences.topk(num_to_unmask)
            unmask_indices = masked_indices[top_indices]

            # Update tokens and mask
            flat_tokens = tokens.view(-1)
            flat_sampled = sampled.view(-1)

            flat_tokens[unmask_indices] = flat_sampled[unmask_indices]
            flat_is_masked[unmask_indices] = False

            tokens = flat_tokens.view(batch_size, -1)
            is_masked = flat_is_masked.view(batch_size, -1)

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
