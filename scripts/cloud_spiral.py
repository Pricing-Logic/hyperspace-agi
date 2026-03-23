#!/usr/bin/env python3
"""Cloud Self-Improvement Spiral — runs the full generate→train→evaluate→repeat loop.

Designed for CUDA GPUs on vast.ai / Modal. Runs unattended for hours.
Fixed per red-team review: 4-bit quant, dynamic padding, difficulty weighting, checkpointing.

Usage:
    python scripts/cloud_spiral.py --model Qwen/Qwen2.5-7B-Instruct --rounds 5
    python scripts/cloud_spiral.py --model Qwen/Qwen2.5-3B-Instruct --rounds 3 --quick
    python scripts/cloud_spiral.py --resume experiments/spiral_20260323_120000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_lora(model_name: str, data_dir: Path, adapter_path: Path, prev_adapter: Path | None, config: dict):
    """Run LoRA training via PEFT/transformers with 4-bit quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    from datasets import load_dataset

    print(f"\n{'='*60}", flush=True)
    print(f"  LORA TRAINING: {adapter_path.name}", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Data: {data_dir}", flush=True)
    if prev_adapter:
        print(f"  Base adapter: {prev_adapter}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in 4-bit to fit training in 24GB VRAM (C6 fix)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if torch.cuda.is_available() else None

    load_kwargs = {"trust_remote_code": True}
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if prev_adapter and prev_adapter.exists():
        model = PeftModel.from_pretrained(model, str(prev_adapter))
        model = model.merge_and_unload()
        print(f"  Merged previous adapter: {prev_adapter}", flush=True)

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.get("rank", 16),
        lora_alpha=config.get("alpha", 32),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "valid.jsonl"),
    })

    # Dynamic padding — no more max_length waste (C4 fix)
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=str(adapter_path),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("grad_accum", 2),
        learning_rate=config.get("lr", 5e-5),
        warmup_steps=config.get("warmup", 50),
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    from transformers import Trainer, DataCollatorForLanguageModeling
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  Adapter saved to {adapter_path}", flush=True)

    # Proper cleanup (W6 fix)
    try:
        model.cpu()
    except Exception:
        pass
    del model, trainer
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_path


def run_spiral(args):
    """Main spiral loop: generate → solve → verify → train → repeat."""
    os.environ["LLM_BACKEND"] = "cuda"

    from src.llm.cuda_backend import CUDABackend
    from src.loop.runner import run_loop, save_report

    results_dir = Path(args.resume) if args.resume else (
        PROJECT_ROOT / "experiments" / f"spiral_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    all_traces_dir = results_dir / "traces"
    all_traces_dir.mkdir(exist_ok=True)
    adapters_dir = results_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)

    lora_config = {
        "rank": args.rank,
        "alpha": args.rank * 2,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "grad_accum": 2,
        "warmup": 50,
    }

    # Resume support (W1 fix): load checkpoint state if resuming
    current_adapter = None
    current_difficulty = args.start_difficulty
    cumulative_traces = []
    round_summaries = []
    start_round = 0

    checkpoint_path = results_dir / "checkpoint.json"
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        start_round = checkpoint.get("completed_rounds", 0)
        current_difficulty = checkpoint.get("difficulty", args.start_difficulty)
        adapter_name = checkpoint.get("adapter")
        if adapter_name:
            current_adapter = Path(adapter_name)
        round_summaries = checkpoint.get("round_summaries", [])
        # Reload cumulative traces
        for r in range(1, start_round + 1):
            trace_file = all_traces_dir / f"round_{r}_full.jsonl"
            if trace_file.exists():
                with open(trace_file) as f:
                    for line in f:
                        cumulative_traces.append(json.loads(line))
        print(f"  [RESUME] Resuming from round {start_round + 1}, {len(cumulative_traces)} traces loaded", flush=True)

    print(f"\n{'*'*60}", flush=True)
    print(f"  HYPERSPACE AGI — CLOUD SPIRAL", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Rounds: {args.rounds} (starting from {start_round + 1})", flush=True)
    print(f"  Gens/round: {args.gens_per_round}", flush=True)
    print(f"  Problems/gen: {args.problems_per_gen}", flush=True)
    print(f"  LoRA config: rank={lora_config['rank']}, lr={lora_config['lr']}", flush=True)
    print(f"  4-bit quantization: enabled", flush=True)
    print(f"  Output: {results_dir}", flush=True)
    print(f"{'*'*60}\n", flush=True)

    spiral_start = time.time()

    for round_num in range(start_round, args.rounds):
        round_start = time.time()
        print(f"\n{'#'*60}", flush=True)
        print(f"  ROUND {round_num + 1}/{args.rounds}", flush=True)
        print(f"  Adapter: {current_adapter or 'BASE'}", flush=True)
        print(f"  Starting difficulty: Tier {current_difficulty}", flush=True)
        print(f"{'#'*60}\n", flush=True)

        # 1. Load model with 4-bit quant
        llm = CUDABackend(args.model, adapter_path=str(current_adapter) if current_adapter else None, load_in_4bit=True)

        # 2. Run the loop
        summary = run_loop(
            llm=llm,
            generations=args.gens_per_round,
            problems_per_gen=args.problems_per_gen,
            start_difficulty=current_difficulty,
            fine_tune_threshold=999999,
            verbose=True,
            model_name=f"{args.model}+LoRA-v{round_num}" if current_adapter else args.model,
        )

        # Save report
        report_path = results_dir / f"round_{round_num + 1}_report.json"
        save_report(summary, report_path)

        # Collect traces — use full traces with difficulty metadata (W2 fix)
        new_traces = summary.get("total_traces", 0)
        full_traces_file = summary.get("full_traces_file")
        traces_file = summary.get("traces_file")

        source_file = full_traces_file or traces_file
        if source_file and Path(source_file).exists():
            import shutil
            dest = all_traces_dir / f"round_{round_num + 1}_full.jsonl"
            shutil.copy(source_file, dest)
            with open(source_file) as f:
                for line in f:
                    cumulative_traces.append(json.loads(line))

        # Update difficulty
        if summary.get("generations"):
            last_gen = summary["generations"][-1]
            current_difficulty = last_gen.get("new_difficulty", current_difficulty)

        round_elapsed = time.time() - round_start
        round_acc = summary.get("overall_accuracy", 0)

        round_summaries.append({
            "round": round_num + 1,
            "accuracy": round_acc,
            "traces": new_traces,
            "cumulative_traces": len(cumulative_traces),
            "difficulty": current_difficulty,
            "adapter": str(current_adapter) if current_adapter else "BASE",
            "elapsed": round(round_elapsed, 1),
        })

        print(f"\n  Round {round_num + 1} complete: {round_acc*100:.1f}% accuracy, {new_traces} new traces", flush=True)
        print(f"  Cumulative traces: {len(cumulative_traces)}", flush=True)

        # 3. Free model memory
        llm.unload()
        del llm

        # 4. Train LoRA if we have enough traces and not the last round
        if len(cumulative_traces) >= 20 and round_num < args.rounds - 1:
            import random
            random.seed(42 + round_num)

            # Weight by difficulty — using full traces which HAVE difficulty (W2 fix)
            weighted = []
            for t in cumulative_traces:
                d = t.get("difficulty", 1)
                # Cap multiplier at 3x to avoid overfitting (W5 fix)
                weight = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}.get(d, 1)
                weighted.extend([t] * weight)
            random.shuffle(weighted)

            split = max(int(len(weighted) * 0.9), len(weighted) - 10)
            train_data = weighted[:split]
            valid_data = weighted[split:]

            train_path = all_traces_dir / "train.jsonl"
            valid_path = all_traces_dir / "valid.jsonl"
            for path, data in [(train_path, train_data), (valid_path, valid_data)]:
                with open(path, "w") as f:
                    for t in data:
                        text = t.get("text", t.get("prompt", "") + "\n" + t.get("completion", ""))
                        f.write(json.dumps({"text": text}) + "\n")

            print(f"\n  Training data: {len(train_data)} train / {len(valid_data)} valid", flush=True)

            new_adapter = adapters_dir / f"lora_v{round_num + 1}"
            train_lora(args.model, all_traces_dir, new_adapter, current_adapter, lora_config)
            current_adapter = new_adapter

        # Save checkpoint after every round (W1 fix)
        checkpoint_path.write_text(json.dumps({
            "completed_rounds": round_num + 1,
            "difficulty": current_difficulty,
            "adapter": str(current_adapter) if current_adapter else None,
            "round_summaries": round_summaries,
            "cumulative_traces": len(cumulative_traces),
            "timestamp": datetime.now().isoformat(),
        }, indent=2))
        print(f"  Checkpoint saved: {checkpoint_path}", flush=True)

    # Final summary
    total_elapsed = time.time() - spiral_start

    print(f"\n{'*'*60}", flush=True)
    print(f"  SPIRAL COMPLETE", flush=True)
    print(f"{'*'*60}", flush=True)
    print(f"  Total rounds: {len(round_summaries)}", flush=True)
    print(f"  Total traces: {len(cumulative_traces)}", flush=True)
    print(f"  Total time: {total_elapsed/60:.1f} minutes", flush=True)
    print(flush=True)
    print(f"  Accuracy by round:", flush=True)
    for r in round_summaries:
        bar = "#" * int(r["accuracy"] * 20)
        print(f"    Round {r['round']}: {bar:<20s} {r['accuracy']*100:.1f}% (Tier {r['difficulty']}, {r['cumulative_traces']} traces)", flush=True)
    print(f"\n  Results: {results_dir}", flush=True)
    print(f"{'*'*60}\n", flush=True)

    with open(results_dir / "spiral_summary.json", "w") as f:
        json.dump({
            "model": args.model,
            "rounds": round_summaries,
            "total_traces": len(cumulative_traces),
            "total_elapsed": round(total_elapsed, 1),
            "lora_config": lora_config,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Cloud Self-Improvement Spiral")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model name")
    parser.add_argument("--rounds", type=int, default=5, help="Number of train-evaluate rounds")
    parser.add_argument("--gens-per-round", type=int, default=10, help="Generations per round")
    parser.add_argument("--problems-per-gen", type=int, default=10, help="Problems per generation")
    parser.add_argument("--start-difficulty", type=int, default=1, help="Starting difficulty tier")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per round")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a spiral directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 gens, 5 problems")

    args = parser.parse_args()
    if args.quick:
        args.gens_per_round = 3
        args.problems_per_gen = 5

    run_spiral(args)


if __name__ == "__main__":
    main()
