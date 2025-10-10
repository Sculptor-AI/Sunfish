#!/usr/bin/env python3
"""
Train Micro SunFish and Monitor Progress
Periodically check for coherent text generation
"""

import subprocess
import time
import os
import sys
from pathlib import Path


def start_training():
    """Start training in background."""
    print("🐟 Starting Micro SunFish Training...")
    print("=" * 70)

    cmd = [
        "venv/bin/python",
        "train.py",
        "--config", "micro",
        "--cpu",
        "--name", "micro-coherent-gen"
    ]

    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    print(f"✅ Training started (PID: {process.pid})")
    return process


def check_for_checkpoint():
    """Find the latest checkpoint."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Get most recent
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def test_generation(checkpoint_path, step_num):
    """Test text generation with current checkpoint."""
    print(f"\n{'='*70}")
    print(f"🎲 Testing Generation at Step {step_num}")
    print(f"{'='*70}")

    cmd = [
        "venv/bin/python",
        "sample.py",
        checkpoint_path,
        "--num_samples", "3",
        "--seq_len", "64",  # Short sequences
        "--num_steps", "20",  # Fast sampling
        "--scheduler", "ddim",
        "--no_progress"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            # Extract and show generated text
            lines = result.stdout.split('\n')
            in_sample = False
            samples = []
            current_sample = []

            for line in lines:
                if 'Sample ' in line and ':' in line:
                    if current_sample:
                        samples.append('\n'.join(current_sample))
                    current_sample = []
                    in_sample = True
                elif in_sample and line.strip() and not line.startswith('─'):
                    current_sample.append(line.strip())

            if current_sample:
                samples.append('\n'.join(current_sample))

            if samples:
                print("\n📝 Generated Samples:")
                for i, sample in enumerate(samples[:3], 1):
                    print(f"\n  Sample {i}: {sample[:100]}...")

                # Check for coherence (simple heuristic)
                coherent_words = 0
                for sample in samples:
                    words = sample.lower().split()
                    # Count common English words
                    common_words = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'to',
                                  'of', 'for', 'on', 'with', 'at', 'by', 'from'}
                    coherent_words += sum(1 for w in words if w in common_words)

                coherence_score = coherent_words / max(1, len(' '.join(samples).split()))
                print(f"\n  Coherence Score: {coherence_score:.2%}")

                if coherence_score > 0.2:
                    print("  🎉 FOUND COHERENT WORDS!")
                    return True
        else:
            print(f"  ⚠️ Generation failed: {result.stderr[:200]}")

    except Exception as e:
        print(f"  ❌ Error testing generation: {e}")

    return False


def monitor_training(process, check_interval=600):
    """
    Monitor training and periodically test generation.

    Args:
        process: Training subprocess
        check_interval: Seconds between generation tests (default: 10min)
    """
    print(f"\n🔍 Monitoring training (checking every {check_interval//60} minutes)...")
    print("Press Ctrl+C to stop\n")

    last_check = time.time()
    last_step = 0
    found_coherent = False

    try:
        while process.poll() is None:
            # Read training output
            line = process.stdout.readline()
            if line:
                # Look for step info
                if 'step=' in line.lower() or 'loss' in line.lower():
                    print(f"  {line.strip()}")

                # Extract step number if possible
                if 'step=' in line.lower():
                    try:
                        step_str = line.split('step=')[1].split()[0]
                        current_step = int(step_str)
                        last_step = current_step
                    except:
                        pass

            # Check if it's time to test generation
            if time.time() - last_check >= check_interval:
                checkpoint = check_for_checkpoint()

                if checkpoint:
                    found_coherent = test_generation(checkpoint, last_step)

                    if found_coherent:
                        print("\n🎊 SUCCESS! Model generating coherent text!")
                        print("You can stop training or let it continue to improve.\n")

                last_check = time.time()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        process.terminate()
        process.wait()

        # Final generation test
        checkpoint = check_for_checkpoint()
        if checkpoint:
            print("\n🔍 Final generation test...")
            test_generation(checkpoint, last_step)

    finally:
        if process.poll() is None:
            process.terminate()

    print(f"\n✅ Training complete! Final checkpoint: {check_for_checkpoint()}")


def main():
    print("\n" + "🐟" * 35)
    print("SUNFISH MICRO TRAINING - COHERENT TEXT GENERATION")
    print("🐟" * 35 + "\n")

    print("This will:")
    print("  1. Train a 32M parameter model on CPU")
    print("  2. Use real text data (not synthetic)")
    print("  3. Test generation every 10 minutes")
    print("  4. Report when coherent words appear")
    print("\n⏱️  Expected time: 2-6 hours for coherent output\n")

    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Start training
    process = start_training()

    # Monitor
    monitor_training(process, check_interval=600)  # Check every 10 min


if __name__ == "__main__":
    main()
