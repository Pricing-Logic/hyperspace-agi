#!/usr/bin/env python3
"""Freeze a deterministic benchmark for consistent evaluation.

Generates N problems per domain per tier, verifies all have correct answers,
saves to JSONL with SHA256 hash for integrity checking.

Usage:
    python scripts/freeze_benchmark.py
    python scripts/freeze_benchmark.py --per-domain-tier 20 --output experiments/bench_v2.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def freeze_benchmark(
    output_path: str = "experiments/frozen_benchmark_v1.jsonl",
    seed: int = 42,
    per_domain_tier: int = 10,
) -> dict:
    """Generate and freeze a benchmark dataset.

    Args:
        output_path: Where to save the benchmark JSONL.
        seed: Random seed for reproducibility.
        per_domain_tier: Problems per domain per tier.

    Returns:
        Metadata dict with counts, hash, and verification status.
    """
    from src.loop.generator import generate_problem, DOMAINS

    rng = random.Random(seed)
    random.seed(seed)

    problems = []
    failures = []
    tiers = [1, 2, 3, 4, 5]

    print(f"Generating {len(DOMAINS)} domains x {len(tiers)} tiers x {per_domain_tier} = "
          f"{len(DOMAINS) * len(tiers) * per_domain_tier} problems...", flush=True)

    for domain in DOMAINS:
        for tier in tiers:
            generated = 0
            attempts = 0
            max_attempts = per_domain_tier * 5  # allow retries for dedup

            seen_texts = set()

            while generated < per_domain_tier and attempts < max_attempts:
                attempts += 1
                try:
                    p = generate_problem(difficulty=tier, domain=domain)
                except Exception:
                    # Tier 5 is composite, not domain-specific
                    if tier == 5:
                        p = generate_problem(difficulty=5)
                    else:
                        continue

                # Dedup by problem text
                text_key = p.text.strip()[:200]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                # Verify the problem has a correct answer
                try:
                    is_valid = p.verification_func(f"ANSWER: {p.correct_answer}")
                except Exception:
                    failures.append({"domain": domain, "tier": tier, "text": p.text, "error": "verification_func crashed"})
                    continue

                if not is_valid:
                    failures.append({"domain": domain, "tier": tier, "text": p.text, "answer": p.correct_answer, "error": "self-verify failed"})
                    continue

                problems.append({
                    "text": p.text,
                    "correct_answer": p.correct_answer,
                    "domain": p.domain,
                    "difficulty": p.difficulty,
                    "tier": tier,
                })
                generated += 1

            if generated < per_domain_tier:
                print(f"  WARNING: {domain} tier {tier}: only generated {generated}/{per_domain_tier}", flush=True)

    # Shuffle deterministically
    rng.shuffle(problems)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    content = ""
    for p in problems:
        content += json.dumps(p) + "\n"

    out.write_text(content)
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Count stats
    by_domain = {}
    by_tier = {}
    for p in problems:
        by_domain[p["domain"]] = by_domain.get(p["domain"], 0) + 1
        by_tier[p["difficulty"]] = by_tier.get(p["difficulty"], 0) + 1

    meta = {
        "total_problems": len(problems),
        "per_domain_tier": per_domain_tier,
        "seed": seed,
        "sha256": content_hash,
        "by_domain": by_domain,
        "by_tier": by_tier,
        "failures": len(failures),
        "output_path": str(out),
    }

    meta_path = out.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Re-verify: load back and check integrity
    loaded = []
    with open(out) as f:
        for line in f:
            loaded.append(json.loads(line))

    reload_hash = hashlib.sha256(out.read_text().encode()).hexdigest()
    assert reload_hash == content_hash, "Hash mismatch after save!"
    assert len(loaded) == len(problems), "Count mismatch after save!"

    print(f"\nBenchmark frozen:", flush=True)
    print(f"  Total: {len(problems)} problems", flush=True)
    print(f"  By domain: {by_domain}", flush=True)
    print(f"  By tier: {by_tier}", flush=True)
    print(f"  Failures: {len(failures)}", flush=True)
    print(f"  SHA256: {content_hash}", flush=True)
    print(f"  Saved: {out}", flush=True)
    print(f"  Meta: {meta_path}", flush=True)

    if failures:
        print(f"\n  Failed problems:", flush=True)
        for f_item in failures[:5]:
            print(f"    {f_item['domain']} T{f_item['tier']}: {f_item['error']}", flush=True)

    return meta


def verify_benchmark(benchmark_path: str) -> bool:
    """Load and re-verify a frozen benchmark."""
    path = Path(benchmark_path)
    meta_path = path.with_suffix(".meta.json")

    if not path.exists():
        print(f"ERROR: {path} not found")
        return False

    content = path.read_text()
    current_hash = hashlib.sha256(content.encode()).hexdigest()

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        expected_hash = meta.get("sha256")
        if current_hash != expected_hash:
            print(f"ERROR: Hash mismatch! Expected {expected_hash}, got {current_hash}")
            return False

    problems = [json.loads(line) for line in content.strip().split("\n")]
    print(f"Verified: {len(problems)} problems, hash {current_hash[:16]}...")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze a benchmark")
    parser.add_argument("--output", default="experiments/frozen_benchmark_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-domain-tier", type=int, default=10)
    parser.add_argument("--verify", type=str, default=None, help="Verify an existing benchmark")

    args = parser.parse_args()

    if args.verify:
        ok = verify_benchmark(args.verify)
        sys.exit(0 if ok else 1)

    meta = freeze_benchmark(args.output, args.seed, args.per_domain_tier)

    if meta["failures"] > 0:
        print(f"\nWARNING: {meta['failures']} problems failed verification during generation.")
        print("These were excluded from the benchmark.")

    # Final integrity check
    ok = verify_benchmark(args.output)
    if ok:
        print("\nBenchmark is LOCKED and ready for evaluation.")
    else:
        print("\nERROR: Benchmark failed integrity check!")
        sys.exit(1)
