"""Assemble the stage-1 router-calibration corpus.

Produces, per workload bucket, a packed uint32 token file plus a manifest —
the exact input shape the JAX calibration hook consumes (docs/
calibration_hook.md; bucket taxonomy from docs/data.md).

Sources are streamed from Hugging Face and capped per bucket; nothing is
mirrored (infra/gcp rules). Gated or unreachable sources are skipped with a
warning so a partial assembly is still a valid pilot corpus — the manifest
records exactly what went in.

Run inside a venv with `datasets` and `tokenizers`:

  ~/sunfish-cache/.venv-tools/bin/python scripts/assemble_calibration.py \
      --tokenizer ~/sunfish-cache/diffusiongemma-26B-A4B-it/tokenizer.json \
      --output ~/sunfish-cache/calibration \
      --tokens-per-bucket 2000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

def _messages_text(ex: dict) -> str:
    """Render a messages list (or JSON-string of one) as plain text."""
    import json as _json

    messages = ex.get("messages") or []
    if isinstance(messages, str):  # some datasets store messages as a JSON string
        try:
            messages = _json.loads(messages)
        except ValueError:
            return ""
    return "\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '')}"
        for m in messages
        if isinstance(m, dict)
    )


# Bucket -> (dataset, config, split, text-builder). Builders return one
# plain-text document per example; calibration needs raw workload text, not
# chat templates (docs/data.md).
SOURCES = {
    "code_completion": {
        # Pilot source: raw file contents from commitpackft's JavaScript
        # slice (ungated JSONL; disjoint language from repo_edit's Python).
        # For the full 75M assembly, swap in The Stack v2 once its terms are
        # accepted (one click) — bucket format is unchanged.
        "dataset": ("json", "hf://datasets/bigcode/commitpackft/data/javascript/*.jsonl", "train"),
        "build": lambda ex: ex.get("new_contents") or "",
    },
    "repo_edit": {
        # commitpackft's loader script is deprecated in datasets>=4; load its
        # raw per-language JSONL directly.
        "dataset": ("json", "hf://datasets/bigcode/commitpackft/data/python/*.jsonl", "train"),
        "build": lambda ex: (
            f"commit: {ex.get('message', '')}\n--- before\n{ex.get('old_contents', '')}"
            f"\n--- after\n{ex.get('new_contents', '')}"
        ),
    },
    "tool_calls": {
        "dataset": ("glaiveai/glaive-function-calling-v2", "default", "train"),
        "build": lambda ex: f"{ex.get('system', '')}\n{ex.get('chat', '')}",
    },
    "agent_trajectory": {
        # Splits are format-named: 'tool' (function-call format) matches our
        # canonical grammar (docs/data.md).
        "dataset": ("SWE-bench/SWE-smith-trajectories", "default", "tool"),
        "build": _messages_text,
    },
    "general_control": {
        "dataset": ("HuggingFaceTB/smoltalk", "all", "train"),
        "build": _messages_text,
    },
    "reasoning_control": {
        "dataset": ("openai/gsm8k", "main", "train"),
        "build": lambda ex: f"{ex.get('question', '')}\n{ex.get('answer', '')}",
    },
}


def assemble_bucket(
    name: str,
    spec: dict,
    tokenizer,
    output_dir: Path,
    token_cap: int,
    min_doc_tokens: int = 16,
) -> dict:
    """Stream, tokenize, and write one bucket as an indexed immutable shard.

    Uses sunfish.datashards ShardWriter (.bin/.idx) so every document is a
    randomly-accessible record — required by the process-disjoint sampler and
    the calibration hook. One shard per bucket; bucket identity lives in the
    manifest entry, not inside records.
    """
    from datasets import load_dataset

    from sunfish.datashards import ShardWriter

    dataset_id, config, split = spec["dataset"]
    build = spec["build"]
    started = time.time()
    if dataset_id == "json":  # raw-file loading; config carries the hf:// glob
        stream = load_dataset("json", data_files={split: config}, split=split, streaming=True)
    else:
        stream = load_dataset(dataset_id, config, split=split, streaming=True)

    writer = ShardWriter(output_dir / name)
    for example in stream:
        text = build(example)
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text).ids
        if len(ids) < min_doc_tokens:
            continue
        writer.add(ids)
        if writer.tokens >= token_cap:
            break
    info = writer.close()
    return {
        "bucket": name,
        "dataset": dataset_id,
        "config": config,
        "seconds": round(time.time() - started, 1),
        **info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-bucket", type=int, default=2_000_000)
    parser.add_argument("--buckets", nargs="*", help="subset of buckets to build")
    args = parser.parse_args()

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from tokenizers import Tokenizer

    from sunfish.datashards import write_manifest

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output.mkdir(parents=True, exist_ok=True)

    results, failures = [], []
    wanted = args.buckets or list(SOURCES)
    for name in wanted:
        spec = SOURCES[name]
        try:
            result = assemble_bucket(
                name, spec, tokenizer, args.output, args.tokens_per_bucket
            )
            results.append(result)
            print(f"[ok] {name}: {result['tokens']:,} tokens / {result['records']:,} records "
                  f"({result['seconds']}s)", flush=True)
        except Exception as error:  # gated dataset, schema drift, network
            failures.append({"bucket": name, "error": str(error)[:300]})
            print(f"[skip] {name}: {error}", file=sys.stderr, flush=True)

    manifest_hash = write_manifest(
        args.output,
        results,
        tokenizer=str(args.tokenizer),
        vocab_size=tokenizer.get_vocab_size(),
        tokens_per_bucket_cap=args.tokens_per_bucket,
        failures=failures,
        purpose="stage-1 router calibration",
    )
    print(json.dumps({
        "total_tokens": sum(r["tokens"] for r in results),
        "manifest_hash": manifest_hash,
        "failures": failures,
    }, indent=2))


if __name__ == "__main__":
    main()
