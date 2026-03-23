#!/usr/bin/env python3
"""Phase 1: Data Collection Sprint — collect 500+ traces with NO training.

Runs inference-only loop across multiple difficulty tiers. Optionally uses
self-consistency voting (majority vote over N samples per problem).

Each instance gets isolated output dirs (C1 fix). Traces flush per-generation (C2 fix).

Usage:
    # Quick local test
    python scripts/collect_traces.py --backend mlx --gens 5 --problems 5

    # Cloud: collect at scale
    python scripts/collect_traces.py --backend cuda --model Qwen/Qwen2.5-7B-Instruct \
        --gens 20 --problems 10 --tiers 1,2,3 --voting 5 --run-id worker-1

    # Multiple workers on different machines (each gets unique run-id):
    # worker-1: --tiers 1,2 --run-id w1
    # worker-2: --tiers 2,3 --run-id w2
    # worker-3: --tiers 3,4 --run-id w3
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def solve_with_voting(llm, prompt: str, n_votes: int = 5) -> tuple[str, str, float]:
    """Generate N answers and return majority vote.

    Returns: (best_answer, best_response, confidence)
    """
    from src.loop.verifier import extract_answer

    responses = []
    parsed_answers = []

    for i in range(n_votes):
        temp = 0.3 + (i * 0.15)  # vary temperature: 0.3, 0.45, 0.6, 0.75, 0.9
        resp = llm.generate(prompt, max_tokens=1024, temperature=temp)
        responses.append(resp)
        answer = extract_answer(resp)
        if answer:
            # Normalize for voting (strip whitespace, lowercase for booleans)
            normalized = answer.strip().lower() if answer.strip().lower() in ('true', 'false') else answer.strip()
            parsed_answers.append((normalized, resp))

    if not parsed_answers:
        return "", responses[0] if responses else "", 0.0

    # Majority vote on normalized answers
    answer_counts = Counter(a for a, _ in parsed_answers)
    best_answer, best_count = answer_counts.most_common(1)[0]
    confidence = best_count / len(parsed_answers)

    # Return the response that produced the winning answer
    best_response = next(resp for ans, resp in parsed_answers if ans == best_answer)

    return best_answer, best_response, confidence


def run_collection(args):
    """Collect traces via inference only — no training."""

    # Isolated output dir per run (C1 fix)
    run_id = args.run_id or f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = PROJECT_ROOT / "experiments" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_file = output_dir / "traces.jsonl"

    # Parse tiers
    tiers = [int(t) for t in args.tiers.split(",")]

    # Load LLM
    if args.backend == "cuda":
        os.environ["LLM_BACKEND"] = "cuda"
        from src.llm.cuda_backend import CUDABackend
        llm = CUDABackend(args.model, load_in_4bit=True)
    elif args.backend == "mlx":
        os.environ["LLM_BACKEND"] = "mlx"
        from src.llm.mlx_backend import MLXBackend
        llm = MLXBackend(args.model)
    else:
        os.environ["LLM_BACKEND"] = "api"
        from src.llm.api_backend import AnthropicBackend
        llm = AnthropicBackend(args.model)

    from src.loop.generator import generate_problem
    from src.loop.verifier import verify_with_function, extract_answer

    print(f"\n{'='*60}", flush=True)
    print(f"  TRACE COLLECTION SPRINT", flush=True)
    print(f"  Run ID:     {run_id}", flush=True)
    print(f"  Model:      {args.model}", flush=True)
    print(f"  Backend:    {args.backend}", flush=True)
    print(f"  Gens:       {args.gens}", flush=True)
    print(f"  Problems:   {args.problems}", flush=True)
    print(f"  Tiers:      {tiers}", flush=True)
    print(f"  Voting:     {args.voting}x" if args.voting > 1 else "  Voting:     off", flush=True)
    print(f"  Output:     {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    total_correct = 0
    total_attempted = 0
    traces_collected = 0
    start_time = time.time()

    for gen in range(args.gens):
        gen_start = time.time()
        tier = tiers[gen % len(tiers)]  # cycle through tiers
        gen_correct = 0

        print(f"  [Gen {gen+1}/{args.gens}] Tier {tier}, {args.problems} problems...", flush=True)

        for i in range(args.problems):
            problem = generate_problem(difficulty=tier)

            prompt = (
                f"Solve the following problem. Think step by step. Show your work.\n"
                f"Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{problem.text}\n\nSOLUTION:"
            )

            if args.voting > 1:
                answer, response, confidence = solve_with_voting(llm, prompt, args.voting)
                result = verify_with_function(response, problem.verification_func, problem.correct_answer)
            else:
                response = llm.generate(prompt, max_tokens=1024, temperature=0.7)
                result = verify_with_function(response, problem.verification_func, problem.correct_answer)
                confidence = 1.0 if result.is_correct else 0.0

            total_attempted += 1

            if result.is_correct:
                gen_correct += 1
                total_correct += 1
                traces_collected += 1

                # Flush trace immediately (C2 fix — no data loss on crash)
                trace = {
                    "text": problem.text + "\n" + response,
                    "prompt": problem.text,
                    "completion": response,
                    "domain": problem.domain,
                    "difficulty": problem.difficulty,
                    "confidence": confidence,
                    "generation": gen,
                    "tier": tier,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(traces_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")

            if args.verbose:
                status = "OK" if result.is_correct else "XX"
                print(f"    [{i+1:2d}/{args.problems}] [{status}] [{problem.domain:^15s}] "
                      f"exp={problem.correct_answer} got={result.parsed_answer}", flush=True)

        gen_elapsed = time.time() - gen_start
        gen_acc = gen_correct / args.problems * 100
        print(f"    -> {gen_correct}/{args.problems} ({gen_acc:.0f}%) in {gen_elapsed:.1f}s | "
              f"Total traces: {traces_collected}", flush=True)

    # Summary
    elapsed = time.time() - start_time
    overall_acc = total_correct / total_attempted * 100 if total_attempted else 0

    summary = {
        "run_id": run_id,
        "model": args.model,
        "backend": args.backend,
        "tiers": tiers,
        "voting": args.voting,
        "total_problems": total_attempted,
        "total_correct": total_correct,
        "accuracy": overall_acc,
        "traces_collected": traces_collected,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  COLLECTION COMPLETE", flush=True)
    print(f"  Traces: {traces_collected} ({overall_acc:.0f}% accuracy)", flush=True)
    print(f"  Time:   {elapsed:.0f}s", flush=True)
    print(f"  Output: {traces_file}", flush=True)
    print(f"{'='*60}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Trace Collection Sprint")
    parser.add_argument("--backend", choices=["mlx", "cuda", "api"], default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--gens", type=int, default=20, help="Number of generations")
    parser.add_argument("--problems", type=int, default=10, help="Problems per generation")
    parser.add_argument("--tiers", default="1,2,3", help="Comma-separated difficulty tiers to cycle")
    parser.add_argument("--voting", type=int, default=1, help="Self-consistency votes per problem (1=off)")
    parser.add_argument("--run-id", default=None, help="Unique run ID for isolation")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
