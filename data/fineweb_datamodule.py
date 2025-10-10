"""
FineWeb DataModule for SunFish
Implements streaming data pipeline (Section 5)
"""

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Iterator
import warnings

from data.simple_text import SimpleTextDataset


class FineWebStreamDataset(IterableDataset):
    """
    Streaming dataset for FineWeb (Section 5).

    Streams data from HuggingFace without downloading entire dataset.
    Tokenizes on-the-fly and yields fixed-length sequences.
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 2048,
        tokenizer_name: str = "gpt2",
        streaming: bool = True,
    ):
        """
        Args:
            split: Dataset split ('train' or 'validation')
            block_size: Sequence length
            tokenizer_name: Tokenizer to use
            streaming: Whether to stream data (True) or download (False)
        """
        self.block_size = block_size
        self.streaming = streaming

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        try:
            self.dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name="default",
                split=split,
                streaming=streaming,
                trust_remote_code=True,
            )
        except Exception as e:
            warnings.warn(f"Failed to load FineWeb: {e}. Using fallback dataset.")
            # Fallback to a smaller dataset for testing
            self.dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=split if split == "train" else "validation",
                streaming=streaming,
            )

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over tokenized sequences."""
        buffer = []

        for item in self.dataset:
            # Get text field (FineWeb uses 'text', WikiText uses 'text')
            text = item.get("text", "")

            if not text or len(text.strip()) == 0:
                continue

            # Tokenize
            token_ids = self.tokenizer(
                text, truncation=False, add_special_tokens=False
            )["input_ids"]

            buffer.extend(token_ids)

            # Yield full blocks
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
                buffer = buffer[self.block_size :]


class TinyTextDataset(Dataset):
    """
    Tiny synthetic dataset for CPU testing.

    Generates random token sequences for quick validation.
    """

    def __init__(
        self, num_samples: int = 1000, block_size: int = 128, vocab_size: int = 1024
    ):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Generate random token sequence
        return torch.randint(0, self.vocab_size, (self.block_size,), dtype=torch.long)


class FineWebDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FineWeb.

    Handles data loading, preprocessing, and batching.
    Supports both streaming (for large-scale training) and local (for testing).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""

        # Detect config type
        is_tiny = (self.config.vocab_size <= 1024 and self.config.n_layer <= 2)
        is_nano_or_micro = (self.config.vocab_size <= 10000 and 3 <= self.config.n_layer <= 12)

        # Nano/Micro config: use real simple text
        if is_nano_or_micro and not is_tiny:
            print("🐟 Using SimpleTextDataset for micro config (real text!)")
            self.train_dataset = SimpleTextDataset(
                block_size=self.config.block_size,
                tokenizer_name=self.config.tokenizer_name,
                num_repeats=2000,  # More repeats for more training data
                vocab_size=self.config.vocab_size,  # Match model vocab
            )
            self.val_dataset = SimpleTextDataset(
                block_size=self.config.block_size,
                tokenizer_name=self.config.tokenizer_name,
                num_repeats=200,  # Less for validation
                vocab_size=self.config.vocab_size,  # Match model vocab
            )
            return

        # Tiny config or CPU: use synthetic data
        if is_tiny or self.config.accelerator == "cpu":
            print("🐟 Using synthetic dataset for CPU/tiny config testing")
            self.train_dataset = TinyTextDataset(
                num_samples=1000,
                block_size=self.config.block_size,
                vocab_size=self.config.vocab_size,
            )
            self.val_dataset = TinyTextDataset(
                num_samples=100,
                block_size=self.config.block_size,
                vocab_size=self.config.vocab_size,
            )
            return

        # For GPU training, use streaming FineWeb
        if stage == "fit" or stage is None:
            self.train_dataset = FineWebStreamDataset(
                split="train",
                block_size=self.config.block_size,
                tokenizer_name=self.config.tokenizer_name,
                streaming=True,
            )

        # Validation dataset (optional)
        if stage == "validate" or stage is None:
            try:
                self.val_dataset = FineWebStreamDataset(
                    split="validation",
                    block_size=self.config.block_size,
                    tokenizer_name=self.config.tokenizer_name,
                    streaming=True,
                )
            except:
                # Validation might not be available
                self.val_dataset = None

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.accelerator == "gpu",
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.accelerator == "gpu",
        )


if __name__ == "__main__":
    # Test with tiny config
    from config.tiny_config import get_tiny_config

    config = get_tiny_config()
    dm = FineWebDataModule(config)
    dm.setup()

    print("\n✅ DataModule initialized successfully!")

    # Test dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"✅ Loaded first batch!")
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Min token ID: {batch.min()}, Max token ID: {batch.max()}")

    # Test a few batches
    print(f"\nTesting data throughput...")
    import time

    start = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
    elapsed = time.time() - start

    print(f"✅ Loaded 10 batches in {elapsed:.2f}s")
    print(f"Throughput: {10 * config.batch_size / elapsed:.1f} samples/sec")
