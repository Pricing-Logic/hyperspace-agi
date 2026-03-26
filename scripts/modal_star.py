"""Modal A100: Full STaR + Rejection Sampling → Train → Four-Benchmark Eval.

The definitive experiment: can a 3B model self-improve toward 7B baseline
using rescued traces + STaR retry + mixed training?

Usage:
    modal run scripts/modal_star.py
"""

import modal

app = modal.App("hyperspace-star-v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers>=4.44", "peft>=0.7", "datasets>=2.16",
        "accelerate>=0.25", "bitsandbytes>=0.42", "sentencepiece", "protobuf",
        "numpy", "sympy",
    )
)

train_image = image.add_local_dir(
    "/Users/sam/Projects/tests/Hyperspace AGI",
    remote_path="/root/hyperspace",
    ignore=modal.FilePatternMatcher(
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/data/arc/**", "**/experiments/adapters/**",
        "**/experiments/gsm8k_*/**", "**/experiments/spiral_*/**",
    ),
)

vol = modal.Volume.from_name("hyperspace-star-results", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"


@app.function(image=train_image, gpu="A100", timeout=14400, volumes={"/results": vol})
def run_star_experiment():
    import sys
    sys.path.insert(0, "/root/hyperspace")

    import json, os, random, time, gc, shutil
    from pathlib import Path

    os.environ["LLM_BACKEND"] = "cuda"

    import torch
    print(f"{'='*60}", flush=True)
    print(f"  STaR EXPERIMENT — MODAL A100", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    results_dir = Path("/results/star_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # =============================================
    # PHASE 1: Collect rescued traces via rejection sampling + STaR
    # =============================================
    print("=== PHASE 1: REJECTION SAMPLING + STaR ===", flush=True)

    from src.llm.cuda_backend import CUDABackend
    from src.loop.generator import generate_problem
    from src.loop.verifier import verify_answer

    llm = CUDABackend(MODEL, load_in_4bit=True)

    # Load benchmark texts for contamination firewall
    benchmark_texts = set()
    for bf in ["/root/hyperspace/experiments/frozen_benchmark_v1.jsonl",
               "/root/hyperspace/experiments/frozen_benchmark_v1_clean.jsonl"]:
        if Path(bf).exists():
            with open(bf) as f:
                for line in f:
                    benchmark_texts.add(json.loads(line)["text"][:200])

    random.seed(42)
    traces = []
    stats = {"easy": 0, "rescued": 0, "unsolved": 0, "star_recovered": 0}
    tiers = [1, 2, 3, 4]
    k = 8
    n_problems = 500  # 500 problems x 8 samples = 4000 LLM calls
    t0 = time.time()

    for i in range(n_problems):
        tier = tiers[i % len(tiers)]
        problem = generate_problem(difficulty=tier)

        if problem.text[:200] in benchmark_texts:
            continue

        correct_responses = []
        all_responses = []
        for j in range(k):
            temp = 0.3 + (j / k) * 0.6
            resp = llm.generate(
                f"Solve the following problem. Think step by step. Show your work.\n"
                f"Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{problem.text}\n\nSOLUTION:",
                max_tokens=1024, temperature=temp,
            )
            result = verify_answer(resp, problem.correct_answer)
            all_responses.append(resp)
            if result.is_correct:
                correct_responses.append(resp)

        n_correct = len(correct_responses)

        if n_correct == k:
            stats["easy"] += 1
            if random.random() < 0.15:  # 15% anchor sampling
                traces.append({"text": problem.text + "\n" + correct_responses[0],
                               "prompt": problem.text, "completion": correct_responses[0],
                               "domain": problem.domain, "difficulty": problem.difficulty,
                               "classification": "easy_anchor", "star_retry": False})
        elif n_correct > 0:
            stats["rescued"] += 1
            for resp in correct_responses[:2]:
                traces.append({"text": problem.text + "\n" + resp,
                               "prompt": problem.text, "completion": resp,
                               "domain": problem.domain, "difficulty": problem.difficulty,
                               "classification": "rescued", "star_retry": False})
        else:
            stats["unsolved"] += 1
            # STaR retry
            star_prompt = (
                f"A student attempted this problem but got the wrong answer.\n\n"
                f"PROBLEM:\n{problem.text}\n\n"
                f"The correct answer is: {problem.correct_answer}\n\n"
                f"Please derive the correct answer step by step.\n"
                f"End with ANSWER: {problem.correct_answer}"
            )
            star_resp = llm.generate(star_prompt, max_tokens=1024, temperature=0.3)
            star_result = verify_answer(star_resp, problem.correct_answer)
            if star_result.is_correct:
                stats["star_recovered"] += 1
                traces.append({"text": problem.text + "\n" + star_resp,
                               "prompt": problem.text, "completion": star_resp,
                               "domain": problem.domain, "difficulty": problem.difficulty,
                               "classification": "star_recovered", "star_retry": True})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_problems}] easy={stats['easy']} rescued={stats['rescued']} "
                  f"unsolved={stats['unsolved']} star={stats['star_recovered']} "
                  f"traces={len(traces)}", flush=True)

    collect_time = time.time() - t0
    print(f"\n  Collection: {len(traces)} traces in {collect_time:.0f}s", flush=True)
    print(f"  {stats}", flush=True)

    # Save traces
    traces_file = results_dir / "traces.jsonl"
    with open(traces_file, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    llm.unload(); del llm; gc.collect(); torch.cuda.empty_cache()

    # =============================================
    # PHASE 2: Mix with GSM8K hard problems + train fresh LoRA
    # =============================================
    print("\n=== PHASE 2: MIXED TRAINING ===", flush=True)

    from scripts.train_clean import merge_and_dedup, clean_split, save_split

    # Load GSM8K train for mixing
    from datasets import load_dataset
    gsm8k_train = load_dataset("gsm8k", "main", split="train")

    # Filter hard GSM8K problems (longer questions = harder)
    gsm8k_traces = []
    for item in gsm8k_train:
        if len(item["question"]) > 200:  # harder problems
            import re
            gold_match = re.search(r'####\s*(-?[\d,]+)', item["answer"])
            if gold_match:
                gsm8k_traces.append({
                    "text": item["question"] + "\n" + item["answer"],
                    "prompt": item["question"],
                    "completion": item["answer"],
                    "domain": "gsm8k",
                    "difficulty": 3,
                    "classification": "gsm8k_train",
                    "star_retry": False,
                })

    random.shuffle(gsm8k_traces)
    n_gsm8k = int(len(traces) * 0.3 / 0.7)  # 30% GSM8K mix
    gsm8k_subset = gsm8k_traces[:n_gsm8k]

    all_traces = traces + gsm8k_subset
    print(f"  Synthetic: {len(traces)} | GSM8K: {len(gsm8k_subset)} | Total: {len(all_traces)}", flush=True)

    # Dedup and clean split
    unique = merge_and_dedup([str(traces_file)])
    # Add GSM8K traces (already unique)
    unique.extend(gsm8k_subset)
    random.shuffle(unique)

    train_raw, valid_raw = clean_split(unique, seed=42)

    # No weighting — Codex said keep it simple for this experiment
    data_dir = results_dir / "data"
    save_split(train_raw, data_dir / "train.jsonl")
    save_split(valid_raw, data_dir / "valid.jsonl")
    print(f"  Split: {len(train_raw)} train / {len(valid_raw)} valid", flush=True)

    # Train FRESH LoRA from base (not from v1 adapter — key insight)
    from scripts.cloud_spiral import train_lora
    adapter_path = results_dir / "adapter"
    train_lora(
        model_name=MODEL,
        data_dir=data_dir,
        adapter_path=adapter_path,
        prev_adapter=None,  # FRESH from base
        config={"rank": 4, "alpha": 8, "lr": 3e-5, "batch_size": 4, "epochs": 2, "grad_accum": 2, "warmup": 20},
    )

    # =============================================
    # PHASE 3: Four-benchmark evaluation
    # =============================================
    print("\n=== PHASE 3: FOUR-BENCHMARK EVALUATION ===", flush=True)

    from scripts.eval_gsm8k import verify_gsm8k_answer

    def eval_frozen(llm_eval, bench_path, label):
        problems = []
        with open(bench_path) as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        correct = 0
        pf = 0
        for p in problems:
            prompt = (f"Solve the following problem. Think step by step. Show your work.\n"
                      f"Put your final answer on the last line after ANSWER:\n\n"
                      f"PROBLEM:\n{p['text']}\n\nSOLUTION:")
            resp = llm_eval.generate(prompt, max_tokens=512, temperature=0.0)
            result = verify_answer(resp, p["correct_answer"])
            if result.is_correct:
                correct += 1
            if result.parsed_answer == "[PARSE_FAILED]":
                pf += 1
        n = len(problems)
        print(f"  {label}: {correct}/{n} ({correct/n*100:.1f}%) | pf={pf}", flush=True)
        return {"correct": correct, "total": n, "accuracy": round(correct/n*100, 1), "parse_failures": pf}

    def eval_gsm8k(llm_eval, bench_path, label):
        problems = []
        with open(bench_path) as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        # Use first 300 for speed
        problems = problems[:300]
        correct = 0
        pf = 0
        for p in problems:
            prompt = f"Question: {p['question']}\n\nLet's think step by step.\nThen write the final answer on the last line as: #### <integer>"
            resp = llm_eval.generate(prompt, max_tokens=1024, temperature=0.0)
            is_correct, parsed, mode = verify_gsm8k_answer(resp, p["gold_answer"])
            if is_correct:
                correct += 1
            if mode == "parse_failed":
                pf += 1
        n = len(problems)
        print(f"  {label}: {correct}/{n} ({correct/n*100:.1f}%) | pf={pf}", flush=True)
        return {"correct": correct, "total": n, "accuracy": round(correct/n*100, 1), "parse_failures": pf}

    bench_frozen = "/root/hyperspace/experiments/frozen_benchmark_v1_clean.jsonl"
    bench_gsm8k = "/root/hyperspace/experiments/gsm8k_test_v1.jsonl"

    all_results = {}

    # 3B BASE
    print("\n  --- 3B BASE ---", flush=True)
    llm_base = CUDABackend(MODEL, load_in_4bit=True)
    all_results["3b_base_frozen"] = eval_frozen(llm_base, bench_frozen, "3B-BASE frozen")
    all_results["3b_base_gsm8k"] = eval_gsm8k(llm_base, bench_gsm8k, "3B-BASE gsm8k")
    llm_base.unload(); del llm_base; gc.collect(); torch.cuda.empty_cache()

    # 3B STaR-LORA
    print("\n  --- 3B STaR-LORA ---", flush=True)
    llm_star = CUDABackend(MODEL, adapter_path=str(adapter_path), load_in_4bit=True)
    all_results["3b_star_frozen"] = eval_frozen(llm_star, bench_frozen, "3B-STaR frozen")
    all_results["3b_star_gsm8k"] = eval_gsm8k(llm_star, bench_gsm8k, "3B-STaR gsm8k")
    llm_star.unload(); del llm_star; gc.collect(); torch.cuda.empty_cache()

    # 7B BASE (the target baseline)
    print("\n  --- 7B BASE (target) ---", flush=True)
    llm_7b = CUDABackend(MODEL_7B, load_in_4bit=True)
    all_results["7b_base_frozen"] = eval_frozen(llm_7b, bench_frozen, "7B-BASE frozen")
    all_results["7b_base_gsm8k"] = eval_gsm8k(llm_7b, bench_gsm8k, "7B-BASE gsm8k")
    llm_7b.unload(); del llm_7b; gc.collect(); torch.cuda.empty_cache()

    # =============================================
    # VERDICT
    # =============================================
    print(f"\n{'='*60}", flush=True)
    print(f"  FOUR-BENCHMARK VERDICT", flush=True)
    print(f"{'='*60}", flush=True)

    frozen_base = all_results["3b_base_frozen"]["accuracy"]
    frozen_star = all_results["3b_star_frozen"]["accuracy"]
    frozen_7b = all_results["7b_base_frozen"]["accuracy"]
    gsm_base = all_results["3b_base_gsm8k"]["accuracy"]
    gsm_star = all_results["3b_star_gsm8k"]["accuracy"]
    gsm_7b = all_results["7b_base_gsm8k"]["accuracy"]

    print(f"  {'Model':<20} {'Frozen':>10} {'GSM8K':>10}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'3B BASE':<20} {frozen_base:>9.1f}% {gsm_base:>9.1f}%", flush=True)
    print(f"  {'3B STaR-LORA':<20} {frozen_star:>9.1f}% {gsm_star:>9.1f}%", flush=True)
    print(f"  {'7B BASE (target)':<20} {frozen_7b:>9.1f}% {gsm_7b:>9.1f}%", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'Frozen delta':<20} {frozen_star-frozen_base:>+9.1f}pp", flush=True)
    print(f"  {'GSM8K delta':<20} {gsm_star-gsm_base:>+9.1f}pp", flush=True)
    print(f"  {'Gap to 7B (frozen)':<20} {frozen_star-frozen_7b:>+9.1f}pp", flush=True)
    print(f"  {'Gap to 7B (GSM8K)':<20} {gsm_star-gsm_7b:>+9.1f}pp", flush=True)
    print(f"{'='*60}", flush=True)

    # Decision
    frozen_improved = (frozen_star - frozen_base) >= 5
    gsm_not_degraded = (gsm_star - gsm_base) >= -2
    if frozen_improved and gsm_not_degraded:
        verdict = "PASS — self-improvement with transfer preservation. Scale up."
    elif frozen_improved:
        verdict = "PARTIAL — in-distribution gain but GSM8K regression. Adjust mixing ratio."
    else:
        verdict = "FAIL — no meaningful improvement. Revisit approach."

    print(f"\n  VERDICT: {verdict}", flush=True)

    # Save everything
    final = {
        "results": all_results,
        "collection_stats": stats,
        "collection_time_s": round(collect_time, 1),
        "traces_count": len(traces),
        "gsm8k_mixed": len(gsm8k_subset),
        "verdict": verdict,
    }
    with open(results_dir / "final_results.json", "w") as f:
        json.dump(final, f, indent=2)

    vol.commit()
    return final


@app.local_entrypoint()
def main():
    result = run_star_experiment.remote()

    print(f"\n{'='*60}")
    print(f"  MODAL STaR EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Traces: {result['traces_count']} + {result['gsm8k_mixed']} GSM8K")
    for name, r in result["results"].items():
        print(f"  {name}: {r['accuracy']}%")
    print(f"{'='*60}")
