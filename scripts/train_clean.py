#!/usr/bin/env python3
"""Phase 2: One Clean Training Run — deduplicate, split THEN weight, train once.

Fixes C3: splits unique traces into train/valid BEFORE weighting.
No data leakage between train and validation sets.

Usage:
    # Merge all collected traces and train
    python scripts/train_clean.py --traces experiments/w1/traces.jsonl experiments/w2/traces.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct --output experiments/clean_lora_v1
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def merge_and_dedup(trace_files: list[str]) -> list[dict]:
    """Load and deduplicate traces from multiple files."""
    all_traces = []
    for f in trace_files:
        p = Path(f)
        if not p.exists():
            print(f"  WARNING: {f} not found, skipping", flush=True)
            continue
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    all_traces.append(json.loads(line))

    # Deduplicate by problem text (first 200 chars)
    seen = set()
    unique = []
    for t in all_traces:
        key = t.get("prompt", t.get("text", ""))[:200]
        if key and key not in seen:
            seen.add(key)
            unique.append(t)

    return unique


def clean_split(traces: list[dict], valid_ratio: float = 0.1, seed: int = 42) -> tuple[list, list]:
    """Split UNIQUE traces into train/valid BEFORE any weighting.

    This is the C3 fix — no duplicate can appear in both sets.
    """
    rng = random.Random(seed)
    shuffled = list(traces)
    rng.shuffle(shuffled)

    n_valid = max(5, int(len(shuffled) * valid_ratio))
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]

    return train, valid


def weight_train_set(train: list[dict], max_weight: int = 3) -> list[dict]:
    """Weight training traces by difficulty — ONLY on the train split."""
    weights = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    weighted = []
    for t in train:
        d = t.get("difficulty", 1)
        w = min(weights.get(d, 1), max_weight)
        weighted.extend([t] * w)
    random.shuffle(weighted)
    return weighted


def save_split(data: list[dict], path: Path):
    """Save traces in {"text": ...} format for training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in data:
            text = t.get("text", t.get("prompt", "") + "\n" + t.get("completion", ""))
            f.write(json.dumps({"text": text}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clean LoRA Training")
    parser.add_argument("--traces", nargs="+", required=True, help="Trace JSONL files to merge")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", default="experiments/clean_lora", help="Output adapter path")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Just prepare data, don't train")

    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"  CLEAN TRAINING PIPELINE", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Merge and deduplicate
    traces = merge_and_dedup(args.traces)
    by_diff = {}
    for t in traces:
        d = t.get("difficulty", 1)
        by_diff[d] = by_diff.get(d, 0) + 1
    print(f"  Unique traces: {len(traces)}", flush=True)
    print(f"  By difficulty: {by_diff}", flush=True)

    if len(traces) < 50:
        print(f"  WARNING: Only {len(traces)} traces. Recommend 300+ for clean training.", flush=True)

    # 2. Split BEFORE weighting (C3 fix)
    train_raw, valid_raw = clean_split(traces, seed=args.seed)
    print(f"  Split: {len(train_raw)} train / {len(valid_raw)} valid (BEFORE weighting)", flush=True)

    # 3. Weight ONLY the train set
    train_weighted = weight_train_set(train_raw)
    print(f"  Weighted train: {len(train_weighted)} examples", flush=True)

    # 4. Save
    output_dir = Path(args.output)
    data_dir = output_dir / "data"
    save_split(train_weighted, data_dir / "train.jsonl")
    save_split(valid_raw, data_dir / "valid.jsonl")  # validation is NEVER weighted
    print(f"  Data saved to {data_dir}", flush=True)

    # Save config for reproducibility
    config = {
        "model": args.model,
        "trace_files": args.traces,
        "unique_traces": len(traces),
        "train_raw": len(train_raw),
        "valid_raw": len(valid_raw),
        "train_weighted": len(train_weighted),
        "by_difficulty": by_diff,
        "rank": args.rank,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.dry_run:
        print(f"\n  DRY RUN — data prepared, no training.", flush=True)
        return

    # 5. Train
    print(f"\n  Starting LoRA training...", flush=True)
    from scripts.cloud_spiral import train_lora
    train_lora(
        model_name=args.model,
        data_dir=data_dir,
        adapter_path=output_dir / "adapter",
        prev_adapter=None,
        config={
            "rank": args.rank,
            "alpha": args.rank * 2,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "grad_accum": 2,
            "warmup": 50,
        },
    )

    print(f"\n  Training complete. Adapter: {output_dir / 'adapter'}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
