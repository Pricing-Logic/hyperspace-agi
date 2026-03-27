"""Modal v2: Sharded STaR collection → merge/train → eval. Crash-safe.

Three separate functions, each with its own timeout. Traces flush to volume
every 10 problems. Collection runs in parallel shards.

Usage:
    modal run scripts/modal_star_v2.py
"""

import modal

app = modal.App("hyperspace-star-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers>=4.44", "peft>=0.7", "datasets>=2.16",
        "accelerate>=0.25", "bitsandbytes>=0.42", "sentencepiece", "protobuf",
        "numpy", "sympy", "fastapi[standard]",
    )
)

train_image = image.add_local_dir(
    "/Users/sam/Projects/tests/Hyperspace AGI",
    remote_path="/root/hyperspace",
    ignore=modal.FilePatternMatcher(
        "**/.venv/**", "**/.git/**", "**/__pycache__/**",
        "**/data/arc/**", "**/experiments/adapters/**",
        "**/experiments/gsm8k_*/**", "**/experiments/spiral_*/**",
        "**/experiments/star_*/**",
    ),
)

vol = modal.Volume.from_name("hyperspace-star-v2", create_if_missing=True)

MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
FLUSH_EVERY = 10


# =============================================
# STAGE 1: Collect traces (sharded, parallel)
# =============================================
@app.function(image=train_image, gpu="A100", timeout=7200, volumes={"/results": vol})
def collect_shard(shard_id: int, n_problems: int = 60, k: int = 8, seed: int = 42):
    """Collect rescued traces for one shard. Flushes to volume every FLUSH_EVERY problems."""
    import sys; sys.path.insert(0, "/root/hyperspace")
    import json, os, random, time, gc
    from pathlib import Path

    os.environ["LLM_BACKEND"] = "cuda"
    import torch
    from src.llm.cuda_backend import CUDABackend
    from src.loop.generator import generate_problem
    from src.loop.verifier import verify_answer

    random.seed(seed + shard_id * 1000)

    # Contamination firewall
    benchmark_texts = set()
    for bf in ["/root/hyperspace/experiments/frozen_benchmark_v1.jsonl",
               "/root/hyperspace/experiments/frozen_benchmark_v1_clean.jsonl"]:
        if Path(bf).exists():
            with open(bf) as f:
                for line in f:
                    benchmark_texts.add(json.loads(line)["text"][:200])

    shard_dir = Path(f"/results/collect/shard_{shard_id:03d}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    traces_file = shard_dir / "traces.jsonl"
    stats_file = shard_dir / "stats.json"

    print(f"[Shard {shard_id}] Starting: {n_problems} problems, k={k}", flush=True)
    print(f"[Shard {shard_id}] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    llm = CUDABackend(MODEL_3B, load_in_4bit=True)
    tiers = [1, 2, 3, 4]

    stats = {"easy": 0, "rescued": 0, "unsolved": 0, "star_recovered": 0,
             "traces_saved": 0, "problems_done": 0}
    buffer = []
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
            if random.random() < 0.15:
                buffer.append({"text": problem.text + "\n" + correct_responses[0],
                               "prompt": problem.text, "completion": correct_responses[0],
                               "domain": problem.domain, "difficulty": problem.difficulty,
                               "classification": "easy_anchor", "star_retry": False})
        elif n_correct > 0:
            stats["rescued"] += 1
            for resp in correct_responses[:2]:
                buffer.append({"text": problem.text + "\n" + resp,
                               "prompt": problem.text, "completion": resp,
                               "domain": problem.domain, "difficulty": problem.difficulty,
                               "classification": "rescued", "star_retry": False})
        else:
            stats["unsolved"] += 1
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
                buffer.append({"text": problem.text + "\n" + star_resp,
                               "prompt": problem.text, "completion": star_resp,
                               "domain": problem.domain, "difficulty": problem.difficulty,
                               "classification": "star_recovered", "star_retry": True})

        stats["problems_done"] = i + 1

        # Flush every FLUSH_EVERY problems
        if (i + 1) % FLUSH_EVERY == 0 and buffer:
            with open(traces_file, "a") as f:
                for t in buffer:
                    f.write(json.dumps(t) + "\n")
            stats["traces_saved"] += len(buffer)
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            vol.commit()
            print(f"[Shard {shard_id}] [{i+1}/{n_problems}] "
                  f"rescued={stats['rescued']} star={stats['star_recovered']} "
                  f"traces={stats['traces_saved']} (flushed)", flush=True)
            buffer = []

    # Final flush
    if buffer:
        with open(traces_file, "a") as f:
            for t in buffer:
                f.write(json.dumps(t) + "\n")
        stats["traces_saved"] += len(buffer)

    stats["elapsed_s"] = round(time.time() - t0, 1)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    vol.commit()

    print(f"[Shard {shard_id}] DONE: {stats}", flush=True)
    return stats


# =============================================
# STAGE 2: Merge traces + train fresh LoRA
# =============================================
@app.function(image=train_image, gpu="A100", timeout=7200, volumes={"/results": vol})
def merge_and_train():
    """Merge shard traces, mix with GSM8K, train fresh LoRA from base."""
    import sys; sys.path.insert(0, "/root/hyperspace")
    import json, os, random, gc
    from pathlib import Path

    os.environ["LLM_BACKEND"] = "cuda"
    import torch

    print("=== STAGE 2: MERGE + TRAIN ===", flush=True)
    vol.reload()

    # Merge all shard traces
    all_traces = []
    collect_dir = Path("/results/collect")
    for shard_dir in sorted(collect_dir.glob("shard_*")):
        traces_file = shard_dir / "traces.jsonl"
        if traces_file.exists():
            with open(traces_file) as f:
                for line in f:
                    if line.strip():
                        all_traces.append(json.loads(line))
        stats_file = shard_dir / "stats.json"
        if stats_file.exists():
            print(f"  {shard_dir.name}: {json.loads(stats_file.read_text())}", flush=True)

    print(f"  Total traces merged: {len(all_traces)}", flush=True)

    # Dedup
    seen = set()
    unique = []
    for t in all_traces:
        key = t.get("prompt", t.get("text", ""))[:200]
        if key not in seen:
            seen.add(key)
            unique.append(t)

    by_class = {}
    for t in unique:
        c = t.get("classification", "unknown")
        by_class[c] = by_class.get(c, 0) + 1
    print(f"  Unique: {len(unique)} | By class: {by_class}", flush=True)

    # Mix with GSM8K hard train problems (30%)
    from datasets import load_dataset
    import re
    gsm8k_train = load_dataset("gsm8k", "main", split="train")
    gsm8k_traces = []
    for item in gsm8k_train:
        if len(item["question"]) > 200:
            gold_match = re.search(r'####\s*(-?[\d,]+)', item["answer"])
            if gold_match:
                gsm8k_traces.append({
                    "text": item["question"] + "\n" + item["answer"],
                    "prompt": item["question"], "completion": item["answer"],
                    "domain": "gsm8k", "difficulty": 3,
                    "classification": "gsm8k_train", "star_retry": False,
                })

    random.seed(42)
    random.shuffle(gsm8k_traces)
    n_gsm8k = int(len(unique) * 0.3 / 0.7)
    gsm8k_subset = gsm8k_traces[:n_gsm8k]
    print(f"  GSM8K mix: {len(gsm8k_subset)} hard train problems", flush=True)

    combined = unique + gsm8k_subset
    random.shuffle(combined)

    # Clean split
    n_valid = max(10, int(len(combined) * 0.1))
    valid_data = combined[:n_valid]
    train_data = combined[n_valid:]

    data_dir = Path("/results/training/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for path, data in [(data_dir / "train.jsonl", train_data), (data_dir / "valid.jsonl", valid_data)]:
        with open(path, "w") as f:
            for t in data:
                f.write(json.dumps({"text": t["text"]}) + "\n")

    print(f"  Split: {len(train_data)} train / {len(valid_data)} valid", flush=True)

    # Train FRESH LoRA from base
    from scripts.cloud_spiral import train_lora
    adapter_path = Path("/results/training/adapter")
    train_lora(
        model_name=MODEL_3B,
        data_dir=data_dir,
        adapter_path=adapter_path,
        prev_adapter=None,
        config={"rank": 4, "alpha": 8, "lr": 3e-5, "batch_size": 4, "epochs": 2, "grad_accum": 2, "warmup": 20},
    )

    vol.commit()
    print("  Training complete. Adapter saved.", flush=True)
    return {"traces": len(unique), "gsm8k_mixed": len(gsm8k_subset),
            "train": len(train_data), "valid": len(valid_data)}


# =============================================
# STAGE 3: Four-benchmark evaluation
# =============================================
@app.function(image=train_image, gpu="A100", timeout=10800, volumes={"/results": vol})
def eval_all():
    """Evaluate 3B base, 3B STaR-LoRA, and 7B base on frozen + GSM8K benchmarks."""
    import sys; sys.path.insert(0, "/root/hyperspace")
    import json, os, gc, time
    from pathlib import Path

    os.environ["LLM_BACKEND"] = "cuda"
    import torch
    from src.llm.cuda_backend import CUDABackend
    from src.loop.verifier import verify_answer
    from scripts.eval_gsm8k import verify_gsm8k_answer

    vol.reload()
    adapter_path = "/results/training/adapter"

    bench_frozen = "/root/hyperspace/experiments/frozen_benchmark_v1_clean.jsonl"
    bench_gsm8k = "/root/hyperspace/experiments/gsm8k_test_v1.jsonl"

    def eval_frozen(llm, bench_path, label, limit=None):
        problems = [json.loads(l) for l in open(bench_path) if l.strip()]
        if limit:
            problems = problems[:limit]
        correct = pf = 0
        for p in problems:
            resp = llm.generate(
                f"Solve the following problem. Think step by step. Show your work.\n"
                f"Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{p['text']}\n\nSOLUTION:",
                max_tokens=512, temperature=0.0)
            result = verify_answer(resp, p["correct_answer"])
            if result.is_correct: correct += 1
            if result.parsed_answer == "[PARSE_FAILED]": pf += 1
        n = len(problems)
        print(f"  {label}: {correct}/{n} ({correct/n*100:.1f}%) pf={pf}", flush=True)
        return {"correct": correct, "total": n, "acc": round(correct/n*100, 1), "pf": pf}

    def eval_gsm8k(llm, bench_path, label, limit=300):
        problems = [json.loads(l) for l in open(bench_path) if l.strip()][:limit]
        correct = pf = 0
        for p in problems:
            resp = llm.generate(
                f"Question: {p['question']}\n\nLet's think step by step.\n"
                f"Then write the final answer on the last line as: #### <integer>",
                max_tokens=1024, temperature=0.0)
            ok, parsed, mode = verify_gsm8k_answer(resp, p["gold_answer"])
            if ok: correct += 1
            if mode == "parse_failed": pf += 1
        n = len(problems)
        print(f"  {label}: {correct}/{n} ({correct/n*100:.1f}%) pf={pf}", flush=True)
        return {"correct": correct, "total": n, "acc": round(correct/n*100, 1), "pf": pf}

    results = {}

    # 3B BASE
    print("\n--- 3B BASE ---", flush=True)
    llm = CUDABackend(MODEL_3B, load_in_4bit=True)
    results["3b_base_frozen"] = eval_frozen(llm, bench_frozen, "3B-BASE frozen")
    results["3b_base_gsm8k"] = eval_gsm8k(llm, bench_gsm8k, "3B-BASE gsm8k")
    llm.unload(); del llm; gc.collect(); torch.cuda.empty_cache()

    # 3B STaR-LORA
    print("\n--- 3B STaR-LORA ---", flush=True)
    llm = CUDABackend(MODEL_3B, adapter_path=adapter_path, load_in_4bit=True)
    results["3b_star_frozen"] = eval_frozen(llm, bench_frozen, "3B-STaR frozen")
    results["3b_star_gsm8k"] = eval_gsm8k(llm, bench_gsm8k, "3B-STaR gsm8k")
    llm.unload(); del llm; gc.collect(); torch.cuda.empty_cache()

    # 7B BASE
    print("\n--- 7B BASE (target) ---", flush=True)
    llm = CUDABackend(MODEL_7B, load_in_4bit=True)
    results["7b_base_frozen"] = eval_frozen(llm, bench_frozen, "7B-BASE frozen")
    results["7b_base_gsm8k"] = eval_gsm8k(llm, bench_gsm8k, "7B-BASE gsm8k")
    llm.unload(); del llm; gc.collect(); torch.cuda.empty_cache()

    # Verdict
    fb = results["3b_base_frozen"]["acc"]
    fs = results["3b_star_frozen"]["acc"]
    f7 = results["7b_base_frozen"]["acc"]
    gb = results["3b_base_gsm8k"]["acc"]
    gs = results["3b_star_gsm8k"]["acc"]
    g7 = results["7b_base_gsm8k"]["acc"]

    print(f"\n{'='*60}", flush=True)
    print(f"  {'Model':<20} {'Frozen':>10} {'GSM8K':>10}", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'3B BASE':<20} {fb:>9.1f}% {gb:>9.1f}%", flush=True)
    print(f"  {'3B STaR-LORA':<20} {fs:>9.1f}% {gs:>9.1f}%", flush=True)
    print(f"  {'7B BASE (target)':<20} {f7:>9.1f}% {g7:>9.1f}%", flush=True)
    print(f"  {'-'*40}", flush=True)
    print(f"  {'Frozen delta':<20} {fs-fb:>+9.1f}pp", flush=True)
    print(f"  {'GSM8K delta':<20} {gs-gb:>+9.1f}pp", flush=True)
    print(f"  {'Gap to 7B (frozen)':<20} {fs-f7:>+9.1f}pp", flush=True)
    print(f"{'='*60}", flush=True)

    frozen_ok = (fs - fb) >= 5
    gsm_ok = (gs - gb) >= -2
    if frozen_ok and gsm_ok:
        verdict = "PASS — self-improvement WITH transfer preservation"
    elif frozen_ok:
        verdict = "PARTIAL — in-distribution gain, GSM8K regression"
    else:
        verdict = "FAIL — no meaningful improvement"

    print(f"  VERDICT: {verdict}", flush=True)

    results["verdict"] = verdict
    with open("/results/final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    return results


# =============================================
# Orchestrator (runs server-side, survives terminal crash)
# =============================================
@app.function(image=train_image, timeout=21600, volumes={"/results": vol})
def run_pipeline():
    """Server-side orchestrator. Triggered via web endpoint or .remote()."""
    import json
    from pathlib import Path

    print("=== STAGE 1: PARALLEL COLLECTION (3 shards) ===", flush=True)
    shard_futures = []
    for shard_id in [0, 1, 2]:
        shard_futures.append(collect_shard.spawn(shard_id, n_problems=60, k=8))

    shard_results = [f.get() for f in shard_futures]

    total_traces = sum(s.get("traces_saved", 0) for s in shard_results)
    print(f"\nCollection done: {total_traces} traces", flush=True)
    for i, s in enumerate(shard_results):
        print(f"  Shard {i}: {s}", flush=True)

    if total_traces < 50:
        return {"error": "Too few traces", "traces": total_traces}

    print("\n=== STAGE 2: MERGE + TRAIN ===", flush=True)
    train_result = merge_and_train.remote()
    print(f"Training: {train_result}", flush=True)

    print("\n=== STAGE 3: FOUR-BENCHMARK EVAL ===", flush=True)
    eval_result = eval_all.remote()

    print(f"\nVERDICT: {eval_result.get('verdict', '?')}", flush=True)

    # Save final summary
    vol.reload()
    with open("/results/pipeline_complete.json", "w") as f:
        json.dump({"shards": shard_results, "training": train_result,
                    "eval": eval_result, "verdict": eval_result.get("verdict")}, f, indent=2)
    vol.commit()

    return eval_result


@app.function(image=train_image, timeout=30)
@modal.fastapi_endpoint(method="GET")
def trigger():
    """GET endpoint to trigger the full pipeline. Returns immediately."""
    call = run_pipeline.spawn()
    return {"status": "launched", "call_id": call.object_id,
            "message": "Pipeline running. Check Modal dashboard for logs."}


@app.function(image=train_image, timeout=30)
@modal.fastapi_endpoint(method="GET")
def trigger_train_eval():
    """Skip collection — go straight to train + eval using existing shard data."""
    call = train_and_eval_only.spawn()
    return {"status": "launched", "call_id": call.object_id,
            "message": "Train+Eval running on existing traces."}


@app.function(image=train_image, timeout=21600, volumes={"/results": vol})
def train_and_eval_only():
    """Run just Stage 2 + 3 using traces already on the volume."""
    import json
    from pathlib import Path

    vol.reload()

    # Verify traces exist
    traces_count = 0
    for s in range(3):
        tf = Path(f"/results/collect/shard_{s:03d}/traces.jsonl")
        if tf.exists():
            with open(tf) as f:
                traces_count += sum(1 for _ in f)

    print(f"Found {traces_count} traces on volume. Proceeding to train+eval.", flush=True)

    if traces_count < 20:
        return {"error": "Too few traces on volume", "count": traces_count}

    print("=== STAGE 2: MERGE + TRAIN ===", flush=True)
    train_result = merge_and_train.remote()
    print(f"Training: {train_result}", flush=True)

    print("=== STAGE 3: FOUR-BENCHMARK EVAL ===", flush=True)
    eval_result = eval_all.remote()
    print(f"Eval: {eval_result}", flush=True)

    vol.reload()
    with open("/results/pipeline_complete.json", "w") as f:
        json.dump({"training": train_result, "eval": eval_result,
                    "verdict": eval_result.get("verdict"), "traces_used": traces_count}, f, indent=2)
    vol.commit()

    return eval_result


@app.function(image=train_image, timeout=60, volumes={"/results": vol})
@modal.fastapi_endpoint(method="GET")
def status():
    """GET endpoint to check results."""
    import json
    from pathlib import Path

    vol.reload()
    result_file = Path("/results/pipeline_complete.json")
    if result_file.exists():
        return json.loads(result_file.read_text())

    # Check shard progress
    shard_stats = {}
    for s in range(3):
        sf = Path(f"/results/collect/shard_{s:03d}/stats.json")
        if sf.exists():
            shard_stats[f"shard_{s}"] = json.loads(sf.read_text())

    if shard_stats:
        return {"status": "in_progress", "shards": shard_stats}
    return {"status": "not_started"}


@app.local_entrypoint()
def main():
    """Local entrypoint — deploy first, then trigger via web."""
    print("This app should be DEPLOYED, not run directly.")
    print("Run: modal deploy scripts/modal_star_v2.py")
    print("Then trigger via the web endpoint URL shown after deploy.")
    print("")
    print("Or to run directly (foreground, will die if terminal closes):")
    result = run_pipeline.remote()
    print(f"\nVERDICT: {result.get('verdict', result)}")


@app.function(image=train_image, gpu="A100", timeout=7200, volumes={"/results": vol})
def eval_single(model_name: str, adapter_path: str = None, tag: str = "base", bench_type: str = "frozen"):
    """Evaluate a single model on one benchmark. Saves result immediately."""
    import sys; sys.path.insert(0, "/root/hyperspace")
    import json, os, gc
    from pathlib import Path
    os.environ["LLM_BACKEND"] = "cuda"
    import torch
    from src.llm.cuda_backend import CUDABackend
    from src.loop.verifier import verify_answer
    from scripts.eval_gsm8k import verify_gsm8k_answer

    vol.reload()
    llm = CUDABackend(model_name, adapter_path=adapter_path, load_in_4bit=True)

    if bench_type == "frozen":
        bench = "/root/hyperspace/experiments/frozen_benchmark_v1_clean.jsonl"
        problems = [json.loads(l) for l in open(bench) if l.strip()]
        correct = pf = 0
        for p in problems:
            resp = llm.generate(
                f"Solve the following problem. Think step by step. Show your work.\n"
                f"Put your final answer on the last line after ANSWER:\n\n"
                f"PROBLEM:\n{p['text']}\n\nSOLUTION:",
                max_tokens=512, temperature=0.0)
            result = verify_answer(resp, p["correct_answer"])
            if result.is_correct: correct += 1
            if result.parsed_answer == "[PARSE_FAILED]": pf += 1
    else:
        bench = "/root/hyperspace/experiments/gsm8k_test_v1.jsonl"
        problems = [json.loads(l) for l in open(bench) if l.strip()][:300]
        correct = pf = 0
        for p in problems:
            resp = llm.generate(
                f"Question: {p['question']}\n\nLet's think step by step.\n"
                f"Then write the final answer on the last line as: #### <integer>",
                max_tokens=1024, temperature=0.0)
            ok, parsed, mode = verify_gsm8k_answer(resp, p["gold_answer"])
            if ok: correct += 1
            if mode == "parse_failed": pf += 1

    n = len(problems)
    result = {"tag": tag, "bench": bench_type, "correct": correct, "total": n,
              "acc": round(correct/n*100, 1), "pf": pf}
    print(f"  {tag} [{bench_type}]: {correct}/{n} ({result['acc']}%) pf={pf}", flush=True)

    with open(f"/results/eval_{tag}_{bench_type}.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    llm.unload(); del llm; gc.collect(); torch.cuda.empty_cache()
    return result


@app.function(image=train_image, timeout=30)
@modal.fastapi_endpoint(method="GET")
def trigger_eval():
    """Trigger parallel eval of all models."""
    calls = []
    calls.append(eval_single.spawn("Qwen/Qwen2.5-3B-Instruct", None, "3b_base", "frozen"))
    calls.append(eval_single.spawn("Qwen/Qwen2.5-3B-Instruct", None, "3b_base", "gsm8k"))
    calls.append(eval_single.spawn("Qwen/Qwen2.5-3B-Instruct", "/results/training/adapter", "3b_star", "frozen"))
    calls.append(eval_single.spawn("Qwen/Qwen2.5-3B-Instruct", "/results/training/adapter", "3b_star", "gsm8k"))
    calls.append(eval_single.spawn("Qwen/Qwen2.5-7B-Instruct", None, "7b_base", "frozen"))
    calls.append(eval_single.spawn("Qwen/Qwen2.5-7B-Instruct", None, "7b_base", "gsm8k"))
    return {"status": "launched", "evals": 6, "message": "6 parallel evals running (3 models x 2 benchmarks)"}
