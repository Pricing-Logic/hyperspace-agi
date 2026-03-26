#!/usr/bin/env python3
"""Evaluate a model on the frozen GSM8K test set.

Strict GSM8K answer extraction. Per-item output for auditability.
Supports both MLX (local) and CUDA (Modal) backends.

Usage:
    # Base model
    python scripts/eval_gsm8k.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx --tag base

    # LoRA adapter
    python scripts/eval_gsm8k.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx \
        --adapter experiments/adapters/lora_v1 --tag lora_v1

    # Quick smoke test (20 problems)
    python scripts/eval_gsm8k.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx --tag smoke --limit 20
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Two prompt modes per Codex design
PROMPT_GSM8K = "Question: {question}\n\nLet's think step by step.\nThen write the final answer on the last line as: #### <integer>"
PROMPT_HOUSE = "Solve the following problem. Think step by step. Show your work.\nPut your final answer on the last line after ANSWER:\n\nPROBLEM:\n{question}\n\nSOLUTION:"


def verify_gsm8k_answer(response: str, gold: int) -> tuple[bool, str | None, str]:
    """Strict GSM8K answer extraction.

    Returns: (is_correct, parsed_answer, parse_mode)
    parse_mode is one of: 'hash', 'answer_tag', 'last_line_int', 'parse_failed'
    """
    text = response.strip()

    # Mode 1: #### <int> (GSM8K standard — allow optional angle brackets)
    matches = re.findall(r'####\s*<?(-?[\d,]+)>?', text)
    if matches:
        parsed = int(matches[-1].replace(',', ''))
        return parsed == gold, str(parsed), "hash"

    # Mode 2: \boxed{int} (LaTeX — common from instruct models)
    matches = re.findall(r'\\boxed\{(-?[\d,]+)\}', text)
    if matches:
        parsed = int(matches[-1].replace(',', ''))
        return parsed == gold, str(parsed), "boxed"

    # Mode 3: ANSWER: <int> (our house format)
    matches = re.findall(r'(?i)ANSWER\s*[:=]\s*(-?[\d,]+)', text)
    if matches:
        parsed = int(matches[-1].replace(',', ''))
        return parsed == gold, str(parsed), "answer_tag"

    # Mode 4: Last line contains a standalone integer
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        # Strip common wrappers
        last = re.sub(r'\\\((.+?)\\\)', r'\1', last)
        last = re.sub(r'\$(.+?)\$', r'\1', last)
        last = re.sub(r'\\boxed\{(.+?)\}', r'\1', last)
        last = re.sub(r'\*\*(.+?)\*\*', r'\1', last)
        last = last.strip().rstrip('.,;')
        last = last.replace(',', '')
        # Extract integer from the line
        int_match = re.search(r'(-?\d+)\s*$', last)
        if int_match:
            parsed = int(int_match.group(1))
            return parsed == gold, str(parsed), "last_line_int"

    return False, None, "parse_failed"


def run_eval(args):
    import os
    os.environ["LLM_BACKEND"] = args.backend

    if args.backend == "mlx":
        from src.llm.mlx_backend import MLXBackend as Backend
        llm = Backend(args.model, adapter_path=args.adapter)
    else:
        from src.llm.cuda_backend import CUDABackend as Backend
        llm = Backend(args.model, adapter_path=args.adapter, load_in_4bit=True)

    # Load frozen GSM8K
    benchmark_path = Path(args.benchmark)
    problems = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    if args.limit:
        problems = problems[:args.limit]

    prompt_template = PROMPT_GSM8K if args.prompt == "gsm8k" else PROMPT_HOUSE

    print(f"GSM8K Evaluation", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Adapter: {args.adapter or 'none (base)'}", flush=True)
    print(f"  Problems: {len(problems)}", flush=True)
    print(f"  Prompt: {args.prompt}", flush=True)
    print(f"  Tag: {args.tag}", flush=True)
    print(flush=True)

    results = []
    correct = 0
    parse_modes = {"hash": 0, "boxed": 0, "answer_tag": 0, "last_line_int": 0, "parse_failed": 0}

    t0 = time.time()

    for i, p in enumerate(problems):
        prompt = prompt_template.format(question=p["question"])

        start = time.time()
        response = llm.generate(prompt, max_tokens=1024, temperature=0.0)
        latency = time.time() - start

        is_correct, parsed, parse_mode = verify_gsm8k_answer(response, p["gold_answer"])
        parse_modes[parse_mode] += 1

        if is_correct:
            correct += 1

        results.append({
            "idx": p["idx"],
            "question": p["question"],
            "gold_answer": p["gold_answer"],
            "parsed_answer": parsed,
            "parse_mode": parse_mode,
            "is_correct": is_correct,
            "response_chars": len(response),
            "question_length": p["question_length"],
            "latency_s": round(latency, 2),
            "raw_response": response,
        })

        if (i + 1) % 50 == 0:
            acc = correct / (i + 1) * 100
            pf = parse_modes["parse_failed"]
            print(f"  [{i+1}/{len(problems)}] acc={acc:.1f}% parse_fail={pf} ({pf/(i+1)*100:.1f}%)", flush=True)

    elapsed = time.time() - t0
    total = len(problems)
    acc = correct / total * 100

    # Complexity bin analysis (by question length quartiles)
    lengths = sorted(r["question_length"] for r in results)
    if len(lengths) > 4:
        q_cuts = [lengths[len(lengths) // 4], lengths[len(lengths) // 2], lengths[3 * len(lengths) // 4]]
    else:
        q_cuts = [999999]

    bins = {"short": [], "medium": [], "long": [], "very_long": []}
    for r in results:
        ql = r["question_length"]
        if ql <= q_cuts[0]:
            bins["short"].append(r)
        elif len(q_cuts) > 1 and ql <= q_cuts[1]:
            bins["medium"].append(r)
        elif len(q_cuts) > 2 and ql <= q_cuts[2]:
            bins["long"].append(r)
        else:
            bins["very_long"].append(r)

    print(f"\n{'='*60}", flush=True)
    print(f"  GSM8K RESULT: {args.tag}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Accuracy: {correct}/{total} ({acc:.1f}%)", flush=True)
    print(f"  Parse modes: {parse_modes}", flush=True)
    print(f"  Parse failure rate: {parse_modes['parse_failed']}/{total} ({parse_modes['parse_failed']/total*100:.1f}%)", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/total:.1f}s/problem)", flush=True)
    print(flush=True)
    print(f"  By complexity:", flush=True)
    for bin_name, bin_items in bins.items():
        if bin_items:
            bc = sum(1 for r in bin_items if r["is_correct"])
            bt = len(bin_items)
            print(f"    {bin_name}: {bc}/{bt} ({bc/bt*100:.1f}%)", flush=True)
    print(f"{'='*60}", flush=True)

    # Save per-item results
    out_dir = Path(f"experiments/gsm8k_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "per_item.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "prompt_mode": args.prompt,
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "parse_modes": parse_modes,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results: {out_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    parser.add_argument("--benchmark", default="experiments/gsm8k_test_v1.jsonl")
    parser.add_argument("--prompt", choices=["gsm8k", "house"], default="gsm8k")
    parser.add_argument("--tag", required=True, help="Label for this run (e.g. base, lora_v1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N problems (for smoke test)")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
