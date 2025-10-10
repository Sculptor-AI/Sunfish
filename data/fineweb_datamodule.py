"""
FineWeb Data Pipeline for Diffusion LLM Training
Implements efficient streaming data loading with tokenization
Supports both streaming from HuggingFace and local downloaded datasets
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Iterator, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FineWebStreamDataset(IterableDataset):
    """
    Streaming dataset for FineWeb that yields tokenized sequences.

    This dataset:
    - Streams data from HuggingFace without downloading the full 108TB corpus
    - Tokenizes text on-the-fly
    - Packs tokens into fixed-length sequences (block_size)
    - Maintains a buffer to avoid wasting tokens at document boundaries
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_config: Optional[str] = "default",
        split: str = "train",
        block_size: int = 2048,
        tokenizer_name: str = "gpt2",
    ):
        """
        Args:
            dataset_name: HuggingFace dataset identifier
            dataset_config: Dataset configuration/subset
            split: Dataset split to use
            block_size: Sequence length (number of tokens)
            tokenizer_name: HuggingFace tokenizer identifier
        """
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.block_size = block_size

        # Load dataset in streaming mode
        logger.info(f"Loading dataset: {dataset_name} (config={dataset_config}, split={split})")
        self.dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            f"Dataset initialized - Vocab size: {len(self.tokenizer)}, "
            f"Block size: {block_size}"
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset, yielding tensors of shape [block_size].

        The iterator maintains a buffer of tokens and yields complete blocks,
        ensuring no tokens are wasted.
        """
        buffer = []
        buffer_size = 0

        for sample in self.dataset:
            # Extract text from the sample (field names may vary)
            text = sample.get("text") or sample.get("content") or sample.get("data")

            if text is None or not text.strip():
                continue

            # Tokenize the text
            try:
                token_ids = self.tokenizer(
                    text,
                    add_special_tokens=False,  # We'll manage special tokens ourselves
                    truncation=False,  # Don't truncate - we handle splitting
                    return_attention_mask=False,
                )["input_ids"]

                # Add to buffer
                buffer.extend(token_ids)
                buffer_size += len(token_ids)

                # Yield complete blocks from the buffer
                while buffer_size >= self.block_size:
                    chunk = buffer[: self.block_size]
                    buffer = buffer[self.block_size :]
                    buffer_size -= self.block_size

                    # Convert to tensor
                    yield torch.tensor(chunk, dtype=torch.long)

            except Exception as e:
                logger.warning(f"Error tokenizing text: {e}")
                continue


class FineWebLocalDataset(IterableDataset):
    """
    Dataset for locally downloaded FineWeb data.

    This dataset:
    - Loads from local JSONL files (downloaded via download_dataset.py)
    - Tokenizes text on-the-fly
    - Much faster than streaming (no network latency)
    - Recommended for production training
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int = 2048,
        tokenizer_name: str = "gpt2",
    ):
        """
        Args:
            data_dir: Directory containing downloaded JSONL chunks
            block_size: Sequence length (number of tokens)
            tokenizer_name: HuggingFace tokenizer identifier
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size

        # Find all chunk files
        self.chunk_files = sorted(self.data_dir.glob("chunk_*.jsonl"))
        if not self.chunk_files:
            raise FileNotFoundError(
                f"No chunk files found in {data_dir}. "
                f"Download data first with: python download_dataset.py --size 500GB --output {data_dir}"
            )

        logger.info(f"Loading local dataset from: {data_dir}")
        logger.info(f"Found {len(self.chunk_files)} chunk files")

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            f"Local dataset initialized - Vocab size: {len(self.tokenizer)}, "
            f"Block size: {block_size}"
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over local data files, yielding tokenized sequences.
        """
        import json

        buffer = []
        buffer_size = 0

        # Iterate through all chunk files
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        text = sample.get('text', '')

                        if not text.strip():
                            continue

                        # Tokenize
                        token_ids = self.tokenizer(
                            text,
                            add_special_tokens=False,
                            truncation=False,
                            return_attention_mask=False,
                        )["input_ids"]

                        # Add to buffer
                        buffer.extend(token_ids)
                        buffer_size += len(token_ids)

                        # Yield complete blocks
                        while buffer_size >= self.block_size:
                            chunk = buffer[: self.block_size]
                            buffer = buffer[self.block_size :]
                            buffer_size -= self.block_size

                            yield torch.tensor(chunk, dtype=torch.long)

                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Error processing line in {chunk_file}: {e}")
                        continue


class FineWebDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FineWeb dataset.

    Handles setup and provides dataloaders for training.
    Supports both streaming mode and local dataset mode.
    """

    def __init__(self, config, local_data_dir: Optional[str] = None):
        """
        Args:
            config: DiffusionConfig object with all hyperparameters
            local_data_dir: Path to local dataset (if None, uses streaming)
        """
        super().__init__()
        self.config = config
        self.local_data_dir = local_data_dir
        self.train_dataset = None

    def prepare_data(self):
        """
        Download/prepare data if needed.
        With streaming, this is mostly a no-op.
        """
        # Test that we can load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        logger.info(f"Tokenizer loaded successfully: {self.config.tokenizer_name}")
        logger.info(f"Vocabulary size: {len(tokenizer)}")

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            # Choose dataset type based on local_data_dir
            if self.local_data_dir is not None:
                # Use local dataset
                logger.info(f"Using LOCAL dataset from: {self.local_data_dir}")
                self.train_dataset = FineWebLocalDataset(
                    data_dir=self.local_data_dir,
                    block_size=self.config.block_size,
                    tokenizer_name=self.config.tokenizer_name,
                )
            else:
                # Use streaming dataset
                logger.info("Using STREAMING dataset from HuggingFace")
                logger.warning(
                    "Streaming mode requires stable internet connection. "
                    "For production training, download data first: "
                    "python download_dataset.py --size 500GB --output data/fineweb_local"
                )
                self.train_dataset = FineWebStreamDataset(
                    dataset_name=self.config.dataset_name,
                    dataset_config=self.config.dataset_config,
                    split=self.config.dataset_split,
                    block_size=self.config.block_size,
                    tokenizer_name=self.config.tokenizer_name,
                )

            # Update config vocab size if needed
            if hasattr(self.train_dataset, "tokenizer"):
                actual_vocab_size = len(self.train_dataset.tokenizer)
                if actual_vocab_size != self.config.vocab_size:
                    logger.warning(
                        f"Updating vocab_size from {self.config.vocab_size} "
                        f"to {actual_vocab_size} (from tokenizer)"
                    )
                    self.config.vocab_size = actual_vocab_size

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Validation dataloader.
        For unsupervised pretraining, we don't have a separate validation set.
        Instead, we generate samples periodically.
        """
        return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """Test dataloader (not used for pretraining)."""
        return None


# Utility function to test the data pipeline
def test_data_pipeline(config, num_batches: int = 10):
    """
    Test the data pipeline by loading a few batches.

    Args:
        config: DiffusionConfig object
        num_batches: Number of batches to load for testing
    """
    import time
    from tqdm import tqdm

    logger.info("Testing data pipeline...")
    data_module = FineWebDataModule(config)
    data_module.setup()

    loader = data_module.train_dataloader()
    start_time = time.time()

    for i, batch in enumerate(tqdm(loader, total=num_batches, desc="Loading batches")):
        if i >= num_batches:
            break

        # Validate batch
        assert batch.ndim == 2, f"Expected 2D batch, got {batch.ndim}D"
        assert batch.shape[1] == config.block_size, (
            f"Expected sequence length {config.block_size}, " f"got {batch.shape[1]}"
        )
        assert batch.dtype == torch.long, f"Expected dtype torch.long, got {batch.dtype}"

        logger.info(
            f"Batch {i}: shape={batch.shape}, dtype={batch.dtype}, "
            f"min={batch.min()}, max={batch.max()}"
        )

    elapsed = time.time() - start_time
    sequences_per_sec = (num_batches * config.batch_size) / elapsed
    tokens_per_sec = sequences_per_sec * config.block_size

    logger.info(f"\n{'='*60}")
    logger.info(f"Data Pipeline Test Results:")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Throughput: {sequences_per_sec:.2f} sequences/sec")
    logger.info(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the data module
    import sys

    sys.path.append("..")
    from config.model_config import get_config_tiny

    logging.basicConfig(level=logging.INFO)

    # Use tiny config for fast testing
    config = get_config_tiny()
    test_data_pipeline(config, num_batches=5)
