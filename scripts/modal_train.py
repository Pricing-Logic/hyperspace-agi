"""Modal A100: 7B self-trace experiment.

Phase 1: Collect 300+ traces from 7B BASE (its own reasoning)
Phase 2: Clean LoRA training on those self-traces
Phase 3: 200-problem A/B test (statistically meaningful)

Decision gate: +5 points = 7B self-improvement works. Flat/negative = abandon 7B.

Usage:
    modal run scripts/modal_train.py
"""

import modal

app = modal.App("hyperspace-7b-selftrace")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.44",
        "peft>=0.7",
        "datasets>=2.16",
        "accelerate>=0.25",
        "bitsandbytes>=0.42",
        "sentencepiece",
        "protobuf",
        "numpy",
        "sympy",
    )
)

train_image = image.add_local_dir(
    "/Users/sam/Projects/tests/Hyperspace AGI",
    remote_path="/root/hyperspace",
    ignore=modal.FilePatternMatcher("**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/data/arc/**", "**/experiments/adapters/**"),
)

vol = modal.Volume.from_name("hyperspace-results-v2", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


@app.function(
    image=train_image,
    gpu="A100",
    timeout=7200,
    volumes={"/results": vol},
)
def run_7b_selftrace_experiment():
    import sys
    sys.path.insert(0, "/root/hyperspace")

    import json, os, random, time
    from pathlib import Path
    from datetime import datetime

    os.environ["LLM_BACKEND"] = "cuda"

    import torch
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))

    print("=" * 60, flush=True)
    print("  7B SELF-TRACE EXPERIMENT — MODAL A100", flush=True)
    print("=" * 60, flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram/1e9:.0f}GB)", flush=True)
    print(f"  Model: {MODEL}", flush=True)
    print(f"  Plan: Collect 300+ self-traces → Clean LoRA → 200-problem A/B", flush=True)
    print("=" * 60, flush=True)

    from src.llm.cuda_backend import CUDABackend
    from src.loop.generator import generate_problem
    from src.loop.verifier import verify_with_function, extract_answer

    results_dir = Path("/results/7b_selftrace")
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_file = results_dir / "traces.jsonl"

    # ========================================
    # PHASE 1: Collect fresh 7B self-traces
    # ========================================
    print("\n=== PHASE 1: COLLECT 7B SELF-TRACES ===", flush=True)

    llm = CUDABackend(MODEL, load_in_4bit=True)
    t0 = time.time()

    total_attempted = 0
    total_correct = 0
    tiers = [1, 2, 3]

    for gen in range(40):
        tier = tiers[gen % len(tiers)]
        gen_correct = 0

        for i in range(10):
            problem = generate_problem(difficulty=tier)
            prompt = (
                "Solve the following problem. Think step by step. Show your work.\n"
                "Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{problem.text}\n\nSOLUTION:"
            )
            resp = llm.generate(prompt, max_tokens=1024, temperature=0.7)
            result = verify_with_function(resp, problem.verification_func, problem.correct_answer)

            total_attempted += 1
            if result.is_correct:
                gen_correct += 1
                total_correct += 1
                trace = {
                    "text": problem.text + "\n" + resp,
                    "prompt": problem.text,
                    "completion": resp,
                    "domain": problem.domain,
                    "difficulty": problem.difficulty,
                    "tier": tier,
                    "generation": gen,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(traces_file, "a") as f:
                    f.write(json.dumps(trace) + "\n")

        acc = gen_correct / 10 * 100
        print(f"  Gen {gen+1:2d}/40 [Tier {tier}]: {gen_correct}/10 ({acc:.0f}%) | Total traces: {total_correct}", flush=True)

    collect_time = time.time() - t0
    collect_acc = total_correct / total_attempted * 100

    print(f"\n  Collection done: {total_correct} traces ({collect_acc:.0f}%) in {collect_time:.0f}s", flush=True)

    llm.unload()
    del llm
    torch.cuda.empty_cache()

    if total_correct < 50:
        print("  ERROR: Too few traces collected. Aborting.", flush=True)
        return {"error": "insufficient traces", "traces": total_correct}

    # ========================================
    # PHASE 2: Clean LoRA training
    # ========================================
    print("\n=== PHASE 2: CLEAN LORA TRAINING ===", flush=True)

    from scripts.train_clean import merge_and_dedup, clean_split, weight_train_set, save_split

    traces = merge_and_dedup([str(traces_file)])
    by_diff = {}
    for t in traces:
        d = t.get("difficulty", 1)
        by_diff[d] = by_diff.get(d, 0) + 1
    print(f"  Unique traces: {len(traces)}", flush=True)
    print(f"  By difficulty: {by_diff}", flush=True)

    train_raw, valid_raw = clean_split(traces, seed=42)
    train_weighted = weight_train_set(train_raw)
    print(f"  Split: {len(train_raw)} train / {len(valid_raw)} valid", flush=True)
    print(f"  Weighted: {len(train_weighted)} examples", flush=True)

    data_dir = results_dir / "data"
    save_split(train_weighted, data_dir / "train.jsonl")
    save_split(valid_raw, data_dir / "valid.jsonl")

    from scripts.cloud_spiral import train_lora
    adapter_path = results_dir / "adapter"
    train_lora(
        model_name=MODEL,
        data_dir=data_dir,
        adapter_path=adapter_path,
        prev_adapter=None,
        config={"rank": 16, "alpha": 32, "lr": 5e-5, "batch_size": 4, "epochs": 2, "grad_accum": 2, "warmup": 30},
    )

    # ========================================
    # PHASE 3: 200-problem A/B test
    # ========================================
    print("\n=== PHASE 3: 200-PROBLEM A/B TEST ===", flush=True)

    from src.loop.generator import generate_batch

    random.seed(42)
    problems = (
        generate_batch(80, difficulty=1)
        + generate_batch(70, difficulty=2)
        + generate_batch(50, difficulty=3)
    )
    random.shuffle(problems)
    print(f"  Problems: {len(problems)} (80 T1 + 70 T2 + 50 T3)", flush=True)

    def run_ab(llm, label):
        correct = 0
        by_tier = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
        for i, p in enumerate(problems):
            prompt = (
                "Solve the following problem. Think step by step. Show your work.\n"
                "Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{p.text}\n\nSOLUTION:"
            )
            resp = llm.generate(prompt, max_tokens=512, temperature=0.3)
            result = verify_with_function(resp, p.verification_func, p.correct_answer)
            tier = p.difficulty
            by_tier[tier][1] += 1
            if result.is_correct:
                correct += 1
                by_tier[tier][0] += 1
            if (i + 1) % 50 == 0:
                print(f"    {label}: {i+1}/{len(problems)} done, {correct} correct so far...", flush=True)

        total = len(problems)
        print(f"\n  {label}: {correct}/{total} ({correct/total*100:.1f}%)", flush=True)
        for t in sorted(by_tier):
            c, tot = by_tier[t]
            if tot > 0:
                print(f"    Tier {t}: {c}/{tot} ({c/tot*100:.1f}%)", flush=True)
        return correct, by_tier

    # Base model
    print("\n  Loading BASE 7B...", flush=True)
    llm_base = CUDABackend(MODEL, load_in_4bit=True)
    base_score, base_tiers = run_ab(llm_base, "BASE")
    llm_base.unload()
    del llm_base
    torch.cuda.empty_cache()

    # Self-trace LoRA
    print("\n  Loading 7B + SELF-TRACE LORA...", flush=True)
    llm_lora = CUDABackend(MODEL, adapter_path=str(adapter_path), load_in_4bit=True)
    lora_score, lora_tiers = run_ab(llm_lora, "SELF-TRACE-LORA")
    llm_lora.unload()
    del llm_lora

    # ========================================
    # VERDICT
    # ========================================
    delta = lora_score - base_score
    n = len(problems)
    gate_pass = delta >= 5

    print(f"\n{'='*60}", flush=True)
    print(f"  7B SELF-TRACE EXPERIMENT — VERDICT", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  BASE:            {base_score}/{n} ({base_score/n*100:.1f}%)", flush=True)
    print(f"  SELF-TRACE-LORA: {lora_score}/{n} ({lora_score/n*100:.1f}%)", flush=True)
    print(f"  Delta:           {delta:+d} ({delta/n*100:+.1f}%)", flush=True)
    print(f"  Gate (+5):       {'PASS — 7B self-improvement WORKS' if gate_pass else 'FAIL — 7B does not self-improve'}", flush=True)
    print(f"{'='*60}", flush=True)

    final = {
        "model": MODEL,
        "traces_collected": total_correct,
        "traces_unique": len(traces),
        "collection_accuracy": collect_acc,
        "collection_time_s": round(collect_time, 1),
        "base_score": base_score,
        "lora_score": lora_score,
        "delta": delta,
        "total_problems": n,
        "gate_pass": gate_pass,
        "by_difficulty": by_diff,
        "base_tiers": {str(k): v for k, v in base_tiers.items()},
        "lora_tiers": {str(k): v for k, v in lora_tiers.items()},
    }
    with open(results_dir / "experiment_results.json", "w") as f:
        json.dump(final, f, indent=2)

    vol.commit()
    return final


@app.local_entrypoint()
def main():
    result = run_7b_selftrace_experiment.remote()

    if "error" in result:
        print(f"\nExperiment failed: {result['error']}")
        return

    n = result["total_problems"]
    print(f"\n{'='*60}")
    print(f"  7B SELF-TRACE EXPERIMENT — FINAL")
    print(f"{'='*60}")
    print(f"  Traces collected: {result['traces_collected']} ({result['collection_accuracy']:.0f}%)")
    print(f"  BASE:            {result['base_score']}/{n}")
    print(f"  SELF-TRACE-LORA: {result['lora_score']}/{n}")
    print(f"  Delta:           {result['delta']:+d} ({result['delta']/n*100:+.1f}%)")
    print(f"  Gate:            {'PASS' if result['gate_pass'] else 'FAIL'}")
    print(f"{'='*60}")
