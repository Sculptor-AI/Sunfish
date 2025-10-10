# Data Setup Guide

## Overview

Sunfish requires 500GB-1TB of training data from FineWeb. You have two options:

1. **Download locally** (RECOMMENDED) - Faster, more reliable
2. **Stream during training** - No pre-download needed, but slower

## Option 1: Download Dataset Locally (Recommended)

### Why Download Locally?

✅ **10-100× faster training** (no network latency)
✅ **More reliable** (no internet drops during multi-day training)
✅ **Better GPU utilization** (no data loading bottleneck)
❌ Requires 500GB-1TB disk space
❌ Takes 2-6 hours to download

### Download Instructions

#### For 500GB (Minimum Recommended)

```bash
python download_dataset.py --size 500GB --output data/fineweb_local
```

This will:
- Download ~166 million samples from FineWeb
- Save to `data/fineweb_local/` as JSONL chunks
- Take 2-4 hours on a good internet connection
- Require ~500GB disk space

#### For 1TB (Full Training)

```bash
python download_dataset.py --size 1TB --output data/fineweb_local
```

This will:
- Download ~333 million samples
- Save to `data/fineweb_local/` as JSONL chunks
- Take 4-6 hours
- Require ~1TB disk space

#### Specific Number of Samples

```bash
# Download exactly 10 million samples (~30GB)
python download_dataset.py --num-samples 10000000 --output data/fineweb_local
```

### Monitor Download Progress

The script shows:
- Progress bar
- Current size downloaded
- Average document size
- Time elapsed

Example output:
```
Downloading: 47%|████████████         | 470000/1000000 [00:15<00:17, 30567.89it/s]
Progress: 470,000 samples, 1.38GB, Avg size: 3.1KB
```

### Resume Interrupted Downloads

If download is interrupted:
- Already downloaded chunks are saved
- Just re-run the same command
- It will continue where it left off (duplicate data is fine - just wastes a bit of space)

### Verify Download

After download completes:

```bash
# Check metadata
cat data/fineweb_local/metadata.json

# Should show something like:
{
  "num_samples": 166000000,
  "total_bytes": 536870912000,
  "avg_bytes_per_sample": 3234,
  "dataset_name": "HuggingFaceFW/fineweb"
}
```

### Use Local Dataset for Training

```bash
# Train with local dataset
python train.py --config 1.4B --local-dataset data/fineweb_local
```

## Option 2: Stream During Training

### Why Stream?

✅ **No pre-download** needed - start training immediately
✅ **No disk space** required
❌ 10-100× slower training
❌ Requires constant internet
❌ Network issues can crash training

### Usage

```bash
# Just run training without --local-dataset flag
python train.py --config 1.4B
```

The data module will automatically stream from HuggingFace.

### Streaming Caveats

**Internet Requirements:**
- Stable connection for 3-5 days
- Bandwidth: ~50-100 Mbps recommended
- No data caps (will download 500GB-1TB total)

**Performance Impact:**
- GPU utilization may drop to 30-60% (waiting for data)
- Effective training speed reduced by 50-90%
- May need to reduce `num_workers` to 2-4

**Reliability Issues:**
- Network hiccups can cause training to hang
- If connection drops, training may crash
- Need to resume from checkpoint

### Streaming Performance Tips

If you must stream:

1. **Reduce workers** to avoid overwhelming connection:
   ```bash
   python train.py --config 1.4B --num-workers 2
   ```

2. **Increase prefetch** to buffer more data:
   ```python
   # In config/model_config.py
   prefetch_factor: int = 4  # Increase from 2
   ```

3. **Use wired connection** (not WiFi)

4. **Monitor bandwidth** usage:
   ```bash
   # Watch network stats
   watch -n 1 ifstat
   ```

## Disk Space Planning

### Recommended Setup

| Component | Size | Notes |
|-----------|------|-------|
| Dataset | 500GB - 1TB | Downloaded chunks |
| Checkpoints | 50-100GB | ~5.6GB each, save top 3 + last |
| Logs | 1-5GB | Training logs, tensorboard |
| **Total** | **550GB - 1.1TB** | Use NVMe SSD for best performance |

### Storage Tips

- **Use NVMe SSD** for dataset (10× faster than HDD)
- Place checkpoints on same drive as dataset
- Logs can go on slower storage
- Consider RAID0 for faster data loading

### Cleaning Up

After training:

```bash
# Remove dataset to free space (keep checkpoints!)
rm -rf data/fineweb_local

# Can always re-download if needed
```

## Dataset Format

Downloaded data is stored as JSONL (JSON Lines):

```json
{"text": "Article content here...", "id": "12345", "url": "https://...", "timestamp": "2024-01-01"}
{"text": "Another article...", "id": "12346", "url": "https://...", "timestamp": "2024-01-02"}
```

Each chunk file contains ~10,000 samples and is ~30-50MB.

## Troubleshooting

### Download Fails

**Error:** `ConnectionResetError` or `timeout`

**Solution:**
- Check internet connection
- Try downloading smaller chunk: `--size 100GB`
- Retry - partial progress is saved

**Error:** `No space left on device`

**Solution:**
- Free up disk space
- Use different output directory on larger drive
- Download smaller dataset: `--size 250GB`

### Download Very Slow

**Causes:**
- Slow internet connection
- HuggingFace servers busy
- Network throttling

**Solutions:**
- Download overnight
- Try different time of day
- Use cloud instance with fast internet

### Corrupted Downloads

**Symptoms:** Training crashes with `JSONDecodeError`

**Solution:**
- Delete the corrupted chunk file
- Re-run download to replace it
- Or delete entire dataset and re-download

## Advanced Options

### Download Specific Subset

```bash
# Download only high-quality educational subset
python download_dataset.py \
    --size 500GB \
    --dataset HuggingFaceFW/fineweb \
    --config CC-MAIN-2024-10 \
    --output data/fineweb_edu
```

### Download in Parallel (Future)

Current script is single-threaded. For faster downloads:

```bash
# Split into 4 parallel downloads
python download_dataset.py --size 125GB --output data/chunk1 &
python download_dataset.py --size 125GB --output data/chunk2 &
python download_dataset.py --size 125GB --output data/chunk3 &
python download_dataset.py --size 125GB --output data/chunk4 &

# Then concatenate
cat data/chunk*/chunk_*.jsonl > data/fineweb_local/
```

(Note: This isn't officially supported but works in practice)

## FAQ

**Q: How long does 500GB download take?**
A: 2-4 hours on 50 Mbps, 1-2 hours on 100 Mbps, 30-60 min on 500 Mbps.

**Q: Can I use a different dataset?**
A: Yes! Any HuggingFace dataset works. Just specify `--dataset <name>`.

**Q: Can I mix local and streaming?**
A: No, choose one. But you can switch between training runs.

**Q: How much data is actually used in 500k steps?**
A: ~131B tokens = ~500GB with default batch size. So 500GB is the minimum.

**Q: What if I run out of disk space mid-download?**
A: Download will fail. Delete partial data, free space, and restart.

**Q: Can I download to external drive?**
A: Yes, but use USB 3.0+ or Thunderbolt. USB 2.0 will bottleneck training.

---

**Recommendation:** Download 500GB minimum for your first training run. You can always download more later if you want to train longer.

**Ready to download?**

```bash
python download_dataset.py --size 500GB --output data/fineweb_local
```
