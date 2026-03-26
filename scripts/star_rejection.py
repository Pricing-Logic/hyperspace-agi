#!/usr/bin/env python3
"""STaR + Rejection Sampling: collect high-value training traces.

Generates k=8 samples per problem. Keeps "rescued" cases (pass@1 fail but pass@k succeed).
Does STaR retry on failures: shows correct answer, asks for derivation.
Mixes in hard GSM8K train problems for transfer.

Usage:
    # Local smoke test
    python scripts/star_rejection.py --backend mlx --model mlx-community/Qwen2.5-3B-Instruct-4bit \
        --problems 20 --k 4 --tag smoke

    # Full collection
    python scripts/star_rejection.py --backend cuda --model Qwen/Qwen2.5-3B-Instruct \
        --problems 1000 --k 8 --tag star_v1
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def classify_problem(correct_counts: int, k: int) -> str:
    """Classify a problem based on pass@k results."""
    if correct_counts == k:
        return "easy"        # always right — low training value
    elif correct_counts > 0:
        return "rescued"     # sometimes right — HIGH training value
    else:
        return "unsolved"    # never right — needs STaR retry


def star_retry(llm, problem_text: str, correct_answer: str, failed_response: str) -> str | None:
    """STaR: show the correct answer, ask model to derive it step by step."""
    prompt = (
        f"A student attempted this problem but got the wrong answer.\n\n"
        f"PROBLEM:\n{problem_text}\n\n"
        f"The correct answer is: {correct_answer}\n\n"
        f"Please derive the correct answer step by step, showing clear reasoning.\n"
        f"End with ANSWER: {correct_answer}"
    )
    response = llm.generate(prompt, max_tokens=1024, temperature=0.3)
    return response


def run_collection(args):
    os.environ["LLM_BACKEND"] = args.backend

    if args.backend == "mlx":
        from src.llm.mlx_backend import MLXBackend as Backend
        llm = Backend(args.model)
    else:
        from src.llm.cuda_backend import CUDABackend as Backend
        llm = Backend(args.model, load_in_4bit=True)

    from src.loop.generator import generate_problem
    from src.loop.verifier import verify_answer

    # Load benchmark hashes for contamination firewall
    benchmark_texts = set()
    for bench_file in ["experiments/frozen_benchmark_v1.jsonl", "experiments/frozen_benchmark_v1_clean.jsonl"]:
        p = Path(bench_file)
        if p.exists():
            with open(p) as f:
                for line in f:
                    item = json.loads(line)
                    benchmark_texts.add(item["text"][:200])

    output_dir = PROJECT_ROOT / "experiments" / f"star_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / "traces.jsonl"
    stats_file = output_dir / "collection_stats.json"

    random.seed(args.seed)
    tiers = [int(t) for t in args.tiers.split(",")]

    print(f"{'='*60}", flush=True)
    print(f"  STaR + REJECTION SAMPLING", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Problems: {args.problems} | k={args.k} | Tiers: {tiers}", flush=True)
    print(f"  Benchmark firewall: {len(benchmark_texts)} blocked texts", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    stats = {"easy": 0, "rescued": 0, "unsolved": 0, "star_recovered": 0,
             "traces_saved": 0, "contamination_blocked": 0, "total_problems": 0}
    t0 = time.time()

    for i in range(args.problems):
        tier = tiers[i % len(tiers)]
        problem = generate_problem(difficulty=tier)

        # Contamination firewall
        if problem.text[:200] in benchmark_texts:
            stats["contamination_blocked"] += 1
            continue

        stats["total_problems"] += 1

        # Generate k samples
        correct_responses = []
        all_responses = []
        for j in range(args.k):
            temp = 0.3 + (j / args.k) * 0.6  # 0.3 → 0.9
            resp = llm.generate(
                f"Solve the following problem. Think step by step. Show your work.\n"
                f"Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{problem.text}\n\nSOLUTION:",
                max_tokens=1024,
                temperature=temp,
            )
            result = verify_answer(resp, problem.correct_answer)
            all_responses.append(resp)
            if result.is_correct:
                correct_responses.append(resp)

        classification = classify_problem(len(correct_responses), args.k)
        stats[classification] += 1

        # Save traces based on classification
        if classification == "rescued":
            # HIGH VALUE: keep at most 2 correct traces (Codex recommendation)
            for resp in correct_responses[:2]:
                trace = {
                    "text": problem.text + "\n" + resp,
                    "prompt": problem.text,
                    "completion": resp,
                    "domain": problem.domain,
                    "difficulty": problem.difficulty,
                    "classification": "rescued",
                    "pass_at_k": len(correct_responses),
                    "k": args.k,
                    "star_retry": False,
                }
                with open(traces_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")
                stats["traces_saved"] += 1

        elif classification == "easy":
            # LOW VALUE: keep only 1 trace for anchor/base distribution
            if random.random() < 0.2:  # keep 20% of easy problems
                trace = {
                    "text": problem.text + "\n" + correct_responses[0],
                    "prompt": problem.text,
                    "completion": correct_responses[0],
                    "domain": problem.domain,
                    "difficulty": problem.difficulty,
                    "classification": "easy_anchor",
                    "pass_at_k": len(correct_responses),
                    "k": args.k,
                    "star_retry": False,
                }
                with open(traces_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")
                stats["traces_saved"] += 1

        elif classification == "unsolved" and args.star:
            # STaR RETRY: show correct answer, ask for derivation
            star_resp = star_retry(llm, problem.text, problem.correct_answer, all_responses[0])
            if star_resp:
                star_result = verify_answer(star_resp, problem.correct_answer)
                if star_result.is_correct:
                    trace = {
                        "text": problem.text + "\n" + star_resp,
                        "prompt": problem.text,
                        "completion": star_resp,
                        "domain": problem.domain,
                        "difficulty": problem.difficulty,
                        "classification": "star_recovered",
                        "pass_at_k": 0,
                        "k": args.k,
                        "star_retry": True,
                    }
                    with open(traces_file, "a") as f:
                        f.write(json.dumps(trace) + "\n")
                    stats["star_recovered"] += 1
                    stats["traces_saved"] += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{args.problems}] easy={stats['easy']} rescued={stats['rescued']} "
                  f"unsolved={stats['unsolved']} star={stats['star_recovered']} "
                  f"traces={stats['traces_saved']} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    stats["elapsed_s"] = round(elapsed, 1)
    stats["model"] = args.model
    stats["k"] = args.k
    stats["tiers"] = tiers

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  COLLECTION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Easy: {stats['easy']} | Rescued: {stats['rescued']} | Unsolved: {stats['unsolved']}", flush=True)
    print(f"  STaR recovered: {stats['star_recovered']}", flush=True)
    print(f"  Total traces saved: {stats['traces_saved']}", flush=True)
    print(f"  Contamination blocked: {stats['contamination_blocked']}", flush=True)
    print(f"  Time: {elapsed:.0f}s", flush=True)
    print(f"  Output: {traces_file}", flush=True)
    print(f"{'='*60}", flush=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description="STaR + Rejection Sampling")
    parser.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--problems", type=int, default=1000)
    parser.add_argument("--k", type=int, default=8, help="Samples per problem")
    parser.add_argument("--tiers", default="1,2,3,4", help="Difficulty tiers to cycle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True, help="Run tag")
    parser.add_argument("--no-star", dest="star", action="store_false", help="Disable STaR retry")
    parser.set_defaults(star=True)

    args = parser.parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
