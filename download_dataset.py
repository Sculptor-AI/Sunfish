#!/usr/bin/env python3
"""
Download FineWeb Dataset for Offline Training

This script downloads a portion of FineWeb to local storage for faster,
more reliable training without internet dependency.

Usage:
    # Download 500GB (recommended minimum)
    python download_dataset.py --size 500GB --output data/fineweb_local

    # Download 1TB (full training)
    python download_dataset.py --size 1TB --output data/fineweb_local

    # Download specific number of samples
    python download_dataset.py --num-samples 10000000 --output data/fineweb_local
"""

import argparse
import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_size(size_str):
    """Parse size string like '500GB' or '1TB' to bytes."""
    size_str = size_str.upper().strip()

    units = {
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4,
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            number = float(size_str[:-len(unit)])
            return int(number * multiplier)

    # Try parsing as plain number (bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '500GB' or '1TB'")


def estimate_samples_from_size(target_bytes):
    """
    Estimate number of samples needed to reach target size.

    FineWeb documents average ~2-5KB each, so we estimate 3KB average.
    """
    avg_doc_size = 3 * 1024  # 3KB average
    estimated_samples = target_bytes // avg_doc_size
    logger.info(f"Estimating ~{estimated_samples:,} samples needed for {target_bytes / 1e9:.1f}GB")
    return estimated_samples


def download_fineweb_chunk(
    output_dir: str,
    num_samples: int = None,
    target_size: int = None,
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "default",
    split: str = "train",
):
    """
    Download a chunk of FineWeb dataset.

    Args:
        output_dir: Directory to save downloaded data
        num_samples: Number of samples to download (if specified)
        target_size: Target size in bytes (if specified, overrides num_samples)
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine number of samples to download
    if target_size is not None:
        num_samples = estimate_samples_from_size(target_size)
    elif num_samples is None:
        raise ValueError("Must specify either --num-samples or --size")

    logger.info(f"Starting download to: {output_path}")
    logger.info(f"Dataset: {dataset_name} ({dataset_config})")
    logger.info(f"Target samples: {num_samples:,}")

    # Load dataset in streaming mode
    logger.info("Connecting to HuggingFace...")
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    # Download samples
    samples = []
    total_bytes = 0
    total_chars = 0

    logger.info("Downloading samples...")
    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Downloading")):
        if i >= num_samples:
            break

        # Extract text
        text = sample.get('text') or sample.get('content') or sample.get('data')
        if not text:
            continue

        # Track stats
        text_bytes = len(text.encode('utf-8'))
        total_bytes += text_bytes
        total_chars += len(text)

        # Store sample
        samples.append({
            'text': text,
            'id': sample.get('id', f'sample_{i}'),
            'url': sample.get('url', ''),
            'timestamp': sample.get('timestamp', ''),
        })

        # Periodically save (every 10k samples)
        if (i + 1) % 10000 == 0:
            save_chunk(samples, output_path, chunk_num=i // 10000)
            logger.info(
                f"Progress: {i+1:,} samples, {total_bytes / 1e9:.2f}GB, "
                f"Avg size: {total_bytes / (i+1) / 1024:.1f}KB"
            )
            samples = []  # Clear for next chunk

    # Save remaining samples
    if samples:
        save_chunk(samples, output_path, chunk_num=(num_samples // 10000))

    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'total_bytes': total_bytes,
        'total_chars': total_chars,
        'avg_bytes_per_sample': total_bytes / num_samples,
        'avg_chars_per_sample': total_chars / num_samples,
        'dataset_name': dataset_name,
        'dataset_config': dataset_config,
        'split': split,
    }

    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("Download complete!")
    logger.info("=" * 80)
    logger.info(f"Samples downloaded: {num_samples:,}")
    logger.info(f"Total size: {total_bytes / 1e9:.2f} GB")
    logger.info(f"Average per sample: {total_bytes / num_samples / 1024:.1f} KB")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("=" * 80)

    # Print usage instructions
    print("\n" + "=" * 80)
    print("To use this dataset for training:")
    print("=" * 80)
    print("\n1. Update config/model_config.py:")
    print(f"   dataset_name: str = 'json'")
    print(f"   dataset_config: str = None")
    print(f"   data_files: str = '{output_path}/chunk_*.jsonl'")
    print("\n2. Or use the --local-dataset flag:")
    print(f"   python train.py --local-dataset {output_path}")
    print("\n" + "=" * 80)


def save_chunk(samples, output_dir, chunk_num):
    """Save a chunk of samples to JSONL file."""
    chunk_file = output_dir / f"chunk_{chunk_num:04d}.jsonl"

    with open(chunk_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    logger.info(f"Saved chunk {chunk_num} ({len(samples)} samples) to {chunk_file}")


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb dataset for offline training")

    # Download target
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--size',
        type=str,
        help='Target download size (e.g., "500GB", "1TB")',
    )
    group.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to download',
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='data/fineweb_local',
        help='Output directory for downloaded data (default: data/fineweb_local)',
    )

    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        default='HuggingFaceFW/fineweb',
        help='HuggingFace dataset name (default: HuggingFaceFW/fineweb)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Dataset configuration (default: default)',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split (default: train)',
    )

    args = parser.parse_args()

    # Parse target size if specified
    target_size = None
    if args.size:
        target_size = parse_size(args.size)
        logger.info(f"Target size: {target_size / 1e9:.2f} GB ({target_size:,} bytes)")

    # Download
    try:
        download_fineweb_chunk(
            output_dir=args.output,
            num_samples=args.num_samples,
            target_size=target_size,
            dataset_name=args.dataset,
            dataset_config=args.config,
            split=args.split,
        )
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        logger.info("Partial download saved - you can resume or use what was downloaded")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nDownload failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
