#!/usr/bin/env python3
"""A/B test: compare base model vs LoRA adapter on a fixed problem set."""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_test(llm, problems, label):
    from src.loop.verifier import verify_with_function

    correct = 0
    by_tier = {}
    for p in problems:
        prompt = (
            "Solve the following problem. Think step by step. Show your work.\n"
            "Put your final answer on the last line after ANSWER:\n\n"
            f"PROBLEM:\n{p.text}\n\nSOLUTION:"
        )
        resp = llm.generate(prompt, max_tokens=512, temperature=0.3)
        result = verify_with_function(resp, p.verification_func, p.correct_answer)
        tier = p.difficulty
        if tier not in by_tier:
            by_tier[tier] = {"correct": 0, "total": 0}
        by_tier[tier]["total"] += 1
        if result.is_correct:
            correct += 1
            by_tier[tier]["correct"] += 1

    total = len(problems)
    pct = correct / total * 100 if total else 0
    print(f"{label}: {correct}/{total} ({pct:.0f}%)", flush=True)
    for t in sorted(by_tier):
        tc = by_tier[t]["correct"]
        tt = by_tier[t]["total"]
        tp = tc / tt * 100 if tt else 0
        print(f"  Tier {t}: {tc}/{tt} ({tp:.0f}%)", flush=True)
    return correct


def main():
    parser = argparse.ArgumentParser(description="A/B Test: Base vs LoRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--problems", type=int, default=30)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    import os
    os.environ["LLM_BACKEND"] = "cuda"

    from src.llm.cuda_backend import CUDABackend
    from src.loop.generator import generate_batch

    random.seed(args.seed)
    n = args.problems
    problems = generate_batch(n // 3, difficulty=1) + generate_batch(n // 3, difficulty=2) + generate_batch(n - 2 * (n // 3), difficulty=3)
    random.shuffle(problems)

    print(f"A/B Test: {len(problems)} problems (seed={args.seed})", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Adapter: {args.adapter}", flush=True)
    print(flush=True)

    print("Loading BASE...", flush=True)
    llm_base = CUDABackend(args.model, load_in_4bit=True)
    base = run_test(llm_base, problems, "BASE")
    llm_base.unload()
    del llm_base

    print(flush=True)
    print("Loading CLEAN-LORA...", flush=True)
    llm_lora = CUDABackend(args.model, adapter_path=args.adapter, load_in_4bit=True)
    lora = run_test(llm_lora, problems, "CLEAN-LORA")
    llm_lora.unload()
    del llm_lora

    print(flush=True)
    delta = lora - base
    print(f"=== RESULT: Delta = {delta:+d} ({delta / len(problems) * 100:+.0f}%) ===", flush=True)


if __name__ == "__main__":
    main()
