#!/usr/bin/env python3
"""A/B test: compare base model vs LoRA adapter on a FROZEN benchmark.

Uses verify_answer (string comparison) not verify_with_function (closure),
so it works with the frozen JSONL without regenerating problems.

Usage:
    python scripts/ab_test.py --model Qwen/Qwen2.5-7B-Instruct \
        --adapter experiments/clean_lora_v1/adapter \
        --benchmark experiments/frozen_benchmark_v1.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_benchmark(path: str) -> list[dict]:
    """Load frozen benchmark from JSONL."""
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def run_test(llm, problems, label):
    from src.loop.verifier import verify_answer

    correct = 0
    parse_failures = 0
    by_tier = {}

    for i, p in enumerate(problems):
        prompt = (
            "Solve the following problem. Think step by step. Show your work.\n"
            "Put your final answer on the last line after ANSWER:\n\n"
            f"PROBLEM:\n{p['text']}\n\nSOLUTION:"
        )
        resp = llm.generate(prompt, max_tokens=512, temperature=0.3)

        # Use verify_answer (string-based, no closure needed)
        result = verify_answer(resp, p["correct_answer"])

        tier = p.get("difficulty", p.get("tier", 1))
        if tier not in by_tier:
            by_tier[tier] = {"correct": 0, "total": 0, "parse_fail": 0}
        by_tier[tier]["total"] += 1

        if result.parsed_answer == "[PARSE_FAILED]":
            parse_failures += 1
            by_tier[tier]["parse_fail"] += 1
        elif result.is_correct:
            correct += 1
            by_tier[tier]["correct"] += 1

        if (i + 1) % 50 == 0:
            print(f"    {label}: {i+1}/{len(problems)} done, {correct} correct, {parse_failures} parse failures", flush=True)

    total = len(problems)
    pct = correct / total * 100 if total else 0
    print(f"\n  {label}: {correct}/{total} ({pct:.1f}%)", flush=True)
    print(f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)", flush=True)
    for t in sorted(by_tier):
        d = by_tier[t]
        if d["total"] > 0:
            print(f"    Tier {t}: {d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%) | parse_fail: {d['parse_fail']}", flush=True)
    return correct, parse_failures, by_tier


def main():
    parser = argparse.ArgumentParser(description="A/B Test on Frozen Benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--benchmark", default="experiments/frozen_benchmark_v1.jsonl")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--backend", choices=["cuda", "mlx"], default="cuda")
    args = parser.parse_args()

    import os
    os.environ["LLM_BACKEND"] = args.backend

    if args.backend == "cuda":
        from src.llm.cuda_backend import CUDABackend as Backend
    else:
        from src.llm.mlx_backend import MLXBackend as Backend

    # Load frozen benchmark
    problems = load_benchmark(args.benchmark)
    if args.max_problems:
        random.seed(42)
        random.shuffle(problems)
        problems = problems[:args.max_problems]

    print(f"A/B Test on Frozen Benchmark", flush=True)
    print(f"  Benchmark: {args.benchmark} ({len(problems)} problems)", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Adapter: {args.adapter}", flush=True)
    print(flush=True)

    # Base model
    print("Loading BASE...", flush=True)
    load_kwargs = {"load_in_4bit": True} if args.backend == "cuda" else {}
    llm_base = Backend(args.model, **load_kwargs)
    base_score, base_pf, base_tiers = run_test(llm_base, problems, "BASE")
    llm_base.unload()
    del llm_base

    print(flush=True)

    # LoRA model
    print("Loading LORA...", flush=True)
    llm_lora = Backend(args.model, adapter_path=args.adapter, **load_kwargs)
    lora_score, lora_pf, lora_tiers = run_test(llm_lora, problems, "LORA")
    llm_lora.unload()
    del llm_lora

    # Verdict
    delta = lora_score - base_score
    n = len(problems)

    print(f"\n{'='*60}", flush=True)
    print(f"  A/B TEST RESULT", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  BASE:  {base_score}/{n} ({base_score/n*100:.1f}%) | parse_fail: {base_pf}", flush=True)
    print(f"  LORA:  {lora_score}/{n} ({lora_score/n*100:.1f}%) | parse_fail: {lora_pf}", flush=True)
    print(f"  Delta: {delta:+d} ({delta/n*100:+.1f}%)", flush=True)
    print(f"  Parse failure delta: {lora_pf - base_pf:+d}", flush=True)
    print(f"", flush=True)
    if lora_pf < base_pf and delta > 0:
        print(f"  NOTE: LoRA has fewer parse failures — improvement may be formatting, not reasoning.", flush=True)
    elif delta > 0 and lora_pf >= base_pf:
        print(f"  NOTE: LoRA improved with same/more parse failures — likely genuine reasoning improvement.", flush=True)
    print(f"{'='*60}", flush=True)

    # Save results
    results = {
        "benchmark": args.benchmark,
        "model": args.model,
        "adapter": args.adapter,
        "total_problems": n,
        "base_score": base_score,
        "base_parse_failures": base_pf,
        "lora_score": lora_score,
        "lora_parse_failures": lora_pf,
        "delta": delta,
    }
    out_path = Path(args.adapter).parent / "ab_test_frozen.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
