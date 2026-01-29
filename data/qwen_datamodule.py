"""
Qwen DataModule for Masked Diffusion Training
Uses Qwen3 tokenizer for data preprocessing.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Iterator
import warnings


class QwenStreamDataset(IterableDataset):
    """
    Streaming dataset using Qwen3 tokenizer.

    Streams data from HuggingFace and tokenizes on-the-fly.
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 512,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        streaming: bool = True,
        dataset_name: str = "wikitext-103",
    ):
        """
        Args:
            split: Dataset split ('train' or 'validation')
            block_size: Sequence length
            tokenizer_name: Qwen tokenizer to use
            streaming: Whether to stream data
            dataset_name: Dataset name - supported options:
                - "wikitext-2": Tiny (~2MB), instant load
                - "wikitext-103": Small (~500MB), fast load (RECOMMENDED)
                - "openwebtext": Medium (~12GB), good quality
                - "fineweb": Large, streaming only
        """
        self.block_size = block_size
        self.streaming = streaming

        # Load Qwen tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add mask token if not present
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        # Map dataset names to HuggingFace identifiers
        # Note: wikitext moved to Salesforce/wikitext
        dataset_configs = {
            "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            "openwebtext": ("openwebtext", None),
            "fineweb": ("HuggingFaceFW/fineweb", "default"),
        }

        # Load dataset
        if dataset_name in dataset_configs:
            hf_name, hf_config = dataset_configs[dataset_name]
            # wikitext doesn't need streaming - it's small enough to cache
            use_streaming = streaming and dataset_name in ["fineweb", "openwebtext"]

            try:
                if hf_config:
                    self.dataset = load_dataset(
                        hf_name,
                        hf_config,
                        split=split,
                        streaming=use_streaming,
                    )
                else:
                    self.dataset = load_dataset(
                        hf_name,
                        split=split,
                        streaming=use_streaming,
                    )
                self.streaming = use_streaming
                print(f"Loaded {dataset_name} ({split} split, streaming={use_streaming})")
            except Exception as e:
                warnings.warn(f"Failed to load {dataset_name}: {e}. Using wikitext-2 fallback.")
                self.dataset = load_dataset(
                    "wikitext",
                    "wikitext-2-raw-v1",
                    split=split,
                    streaming=False,
                )
                self.streaming = False
                print("Using wikitext-2 fallback dataset")
        else:
            # Try loading as a direct HuggingFace dataset name
            try:
                self.dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                )
                print(f"Loaded {dataset_name} ({split} split)")
            except Exception as e:
                warnings.warn(f"Failed to load {dataset_name}: {e}. Using wikitext-2 fallback.")
                self.dataset = load_dataset(
                    "wikitext",
                    "wikitext-2-raw-v1",
                    split=split,
                    streaming=False,
                )
                self.streaming = False
                print("Using wikitext-2 fallback dataset")

    def __iter__(self) -> Iterator[dict]:
        """Iterate over tokenized sequences."""
        buffer = []

        for item in self.dataset:
            text = item.get("text", "")

            if not text or len(text.strip()) == 0:
                continue

            # Tokenize without truncation
            token_ids = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            buffer.extend(token_ids)

            # Yield full blocks
            while len(buffer) >= self.block_size:
                tokens = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                buffer = buffer[self.block_size:]
                yield {"input_ids": tokens}


class QwenCachedDataset(Dataset):
    """
    Cached dataset that pre-tokenizes everything into memory.
    Much faster for small/medium datasets like wikitext.
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 512,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        dataset_name: str = "wikitext-103",
        max_samples: int = None,
    ):
        """
        Args:
            split: Dataset split ('train', 'validation', 'test')
            block_size: Sequence length
            tokenizer_name: Qwen tokenizer to use
            dataset_name: "wikitext-2", "wikitext-103", or HF dataset name
            max_samples: Limit number of samples (None = all)
        """
        self.block_size = block_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        # Map dataset names (wikitext moved to Salesforce/wikitext)
        dataset_configs = {
            "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
        }

        # Load dataset
        print(f"Loading {dataset_name} ({split})...")
        if dataset_name in dataset_configs:
            hf_name, hf_config = dataset_configs[dataset_name]
            raw_dataset = load_dataset(hf_name, hf_config, split=split)
        else:
            raw_dataset = load_dataset(dataset_name, split=split)

        # Tokenize all text
        print("Tokenizing dataset (one-time cost, cached after)...")
        all_tokens = []
        for item in raw_dataset:
            text = item.get("text", "")
            if text and len(text.strip()) > 0:
                tokens = self.tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                )["input_ids"]
                all_tokens.extend(tokens)

        # Chunk into blocks
        self.samples = []
        for i in range(0, len(all_tokens) - block_size + 1, block_size):
            self.samples.append(
                torch.tensor(all_tokens[i:i + block_size], dtype=torch.long)
            )
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"Created {len(self.samples):,} samples of {block_size} tokens each")
        print(f"Total tokens: {len(all_tokens):,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.samples[idx]}


class QwenTextDataset(Dataset):
    """
    Non-streaming dataset for smaller data.
    Useful for validation or testing.
    """

    def __init__(
        self,
        texts: list,
        block_size: int = 512,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
    ):
        """
        Args:
            texts: List of text strings
            block_size: Sequence length
            tokenizer_name: Qwen tokenizer to use
        """
        self.block_size = block_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            all_tokens.extend(tokens)

        # Chunk into blocks
        self.samples = []
        for i in range(0, len(all_tokens) - block_size + 1, block_size):
            self.samples.append(torch.tensor(all_tokens[i:i + block_size], dtype=torch.long))

        print(f"Created dataset with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.samples[idx]}


class SyntheticQwenDataset(Dataset):
    """
    Synthetic dataset for testing.
    Generates random token sequences.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        block_size: int = 512,
        vocab_size: int = 151936,  # Qwen3-0.6B vocab size
    ):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        tokens = torch.randint(0, self.vocab_size, (self.block_size,), dtype=torch.long)
        return {"input_ids": tokens}


class QwenDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Qwen-based masked diffusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

        # Get tokenizer info for vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""

        # Use synthetic data if configured
        use_synthetic = getattr(self.config, 'use_synthetic_data', False)
        if use_synthetic:
            print("Using synthetic dataset for fast testing")
            self.train_dataset = SyntheticQwenDataset(
                num_samples=10000,
                block_size=self.config.block_size,
                vocab_size=len(self.tokenizer),
            )
            self.val_dataset = SyntheticQwenDataset(
                num_samples=100,
                block_size=self.config.block_size,
                vocab_size=len(self.tokenizer),
            )
            return

        # Get dataset name from config
        dataset_name = getattr(self.config, 'dataset_name', 'wikitext-103')

        # For wikitext datasets, use cached version (much faster)
        use_cached = dataset_name in ["wikitext-2", "wikitext-103"]

        if stage == "fit" or stage is None:
            print(f"Setting up training dataset: {dataset_name}...")
            if use_cached:
                self.train_dataset = QwenCachedDataset(
                    split="train",
                    block_size=self.config.block_size,
                    tokenizer_name=self.config.base_model,
                    dataset_name=dataset_name,
                )
            else:
                self.train_dataset = QwenStreamDataset(
                    split="train",
                    block_size=self.config.block_size,
                    tokenizer_name=self.config.base_model,
                    streaming=True,
                    dataset_name=dataset_name,
                )

        if stage in {"fit", "validate"} or stage is None:
            try:
                if use_cached:
                    self.val_dataset = QwenCachedDataset(
                        split="validation",
                        block_size=self.config.block_size,
                        tokenizer_name=self.config.base_model,
                        dataset_name=dataset_name,
                    )
                else:
                    self.val_dataset = QwenStreamDataset(
                        split="validation",
                        block_size=self.config.block_size,
                        tokenizer_name=self.config.base_model,
                        streaming=True,
                        dataset_name=dataset_name,
                    )
            except Exception:
                print("Validation dataset not available, using synthetic")
                self.val_dataset = SyntheticQwenDataset(
                    num_samples=100,
                    block_size=self.config.block_size,
                    vocab_size=len(self.tokenizer),
                )

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
    from config.qwen_masked_config import get_qwen_masked_config_cpu

    config = get_qwen_masked_config_cpu()

    print("Testing QwenDataModule...")
    dm = QwenDataModule(config)
    dm.setup()

    print(f"\nDataModule initialized!")
    print(f"Tokenizer vocab size: {len(dm.tokenizer)}")

    # Test dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"\nFirst batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  dtype: {batch['input_ids'].dtype}")
    print(f"  min: {batch['input_ids'].min()}, max: {batch['input_ids'].max()}")

    print("\nTest passed!")
