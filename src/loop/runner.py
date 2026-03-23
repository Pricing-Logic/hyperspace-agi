"""Main Loop Orchestrator — the beating heart of the self-improving system.

Runs the generate -> solve -> verify -> collect -> fine-tune loop across
multiple generations, tracking progress and adapting difficulty via curriculum.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import DB_PATH, EXPERIMENTS_DIR
from src.experiment_log import ExperimentLogger
from src.loop.generator import (
    CurriculumManager,
    Problem,
    generate_batch,
    DOMAINS,
)
from src.loop.verifier import verify_with_function, VerificationResult
from src.loop.trainer import TrainingDataCollector, LoRAFineTuner


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

COT_SYSTEM = (
    "You are a precise mathematical and logical reasoning engine. "
    "Solve problems step by step, showing your work clearly."
)

COT_TEMPLATE = """Solve the following problem. Think step by step. Show your work.
Put your final answer on the last line after 'ANSWER:'.

PROBLEM:
{problem}

SOLUTION:
Let me work through this step by step.
"""


def build_prompt(problem_text: str) -> str:
    """Build the chain-of-thought prompt for a problem."""
    return COT_TEMPLATE.format(problem=problem_text)


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

class CollapseDetector:
    """Detects if the model has entered a degenerate state."""

    def __init__(self, zero_streak_limit: int = 3):
        self.zero_streak_limit = zero_streak_limit
        self.zero_streak: int = 0
        self.recent_accuracies: list[float] = []
        self.recent_responses: list[str] = []

    def record_generation(self, accuracy: float, responses: list[str]) -> None:
        self.recent_accuracies.append(accuracy)
        self.recent_responses = responses  # keep latest gen's responses

        if accuracy == 0.0:
            self.zero_streak += 1
        else:
            self.zero_streak = 0

    def is_collapsed(self) -> tuple[bool, str]:
        """Check for collapse. Returns (collapsed, reason)."""
        # Zero accuracy streak
        if self.zero_streak >= self.zero_streak_limit:
            return True, f"Zero accuracy for {self.zero_streak} consecutive generations"

        # All responses identical (model stuck in a loop)
        if len(self.recent_responses) > 3:
            unique = set(r.strip()[:100] for r in self.recent_responses)
            if len(unique) == 1:
                return True, "All responses are identical — model is stuck"

        return False, ""

    def should_diversify(self) -> bool:
        """Should we try to diversify problem generation?"""
        if len(self.recent_accuracies) >= 2:
            # Accuracy trending down
            if self.recent_accuracies[-1] < self.recent_accuracies[-2] * 0.5:
                return True
        return False


# ---------------------------------------------------------------------------
# Generation dashboard
# ---------------------------------------------------------------------------

def print_dashboard(
    generation: int,
    total_generations: int,
    correct: int,
    total: int,
    old_diff: int,
    new_diff: int,
    domain_report: dict[str, float],
    total_traces: int,
    fine_tune_status: str | None,
    elapsed: float,
    collapse_warning: str | None = None,
) -> None:
    """Print an exciting mad-science dashboard for this generation."""

    acc = correct / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * correct / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)

    diff_arrow = ""
    if new_diff > old_diff:
        diff_arrow = f" -> Tier {new_diff} [ADVANCING]"
    elif new_diff < old_diff:
        diff_arrow = f" -> Tier {new_diff} [RETREATING]"

    print()
    print("=" * 68)
    print(f"  GENERATION {generation + 1}/{total_generations}               {elapsed:.1f}s elapsed")
    print("=" * 68)
    print(f"  Accuracy:    [{bar}] {correct}/{total} ({acc:.1f}%)")
    print(f"  Difficulty:  Tier {old_diff}{diff_arrow}")
    print()
    print("  Domain Breakdown:")

    for domain, domain_acc in sorted(domain_report.items()):
        pct = domain_acc * 100
        indicator = ">>>" if pct >= 80 else "   " if pct >= 50 else "!!!"
        domain_bar_filled = int(15 * domain_acc)
        domain_bar = "=" * domain_bar_filled + "-" * (15 - domain_bar_filled)
        print(f"    {indicator} {domain:<17s} [{domain_bar}] {pct:5.1f}%")

    print()
    print(f"  Total traces collected: {total_traces}")

    if fine_tune_status:
        print(f"  Fine-tuning: {fine_tune_status}")

    if collapse_warning:
        print()
        print(f"  *** WARNING: {collapse_warning} ***")

    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    llm=None,
    generations: int = 10,
    problems_per_gen: int = 20,
    start_difficulty: int = 1,
    fine_tune_threshold: int = 50,
    verbose: bool = False,
    model_name: str | None = None,
) -> dict:
    """Run the self-improving loop.

    Args:
        llm: An LLMInterface instance. If None, loads from config.
        generations: Number of generations to run.
        problems_per_gen: Problems per generation.
        start_difficulty: Starting difficulty tier (1-5).
        fine_tune_threshold: Trigger fine-tuning after this many traces.
        verbose: Print detailed per-problem output.
        model_name: Model name (for fine-tuner). Auto-detected if None.

    Returns:
        A summary dict with all stats.
    """
    # --- Setup ---
    if llm is None:
        from src.config import get_llm
        llm = get_llm()

    if model_name is None:
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))

    run_id = f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    logger = ExperimentLogger(DB_PATH)
    collector = TrainingDataCollector()
    curriculum = CurriculumManager(start_difficulty=start_difficulty)
    collapse = CollapseDetector()
    fine_tuner = LoRAFineTuner(model_name=model_name)

    # Track overall stats
    all_results: list[dict] = []
    generation_summaries: list[dict] = []
    total_correct = 0
    total_attempted = 0
    fine_tune_triggered = False
    stopped_early = False
    stop_reason = ""

    print()
    print("=" * 68)
    print("  HYPERSPACE AGI -- SELF-IMPROVING LOOP")
    print(f"  Run ID:     {run_id}")
    print(f"  Model:      {model_name}")
    print(f"  Generations: {generations} x {problems_per_gen} problems")
    print(f"  Start Tier: {start_difficulty}")
    print("=" * 68)
    print()

    loop_start = time.time()

    # --- Main loop ---
    for gen in range(generations):
        gen_start = time.time()
        gen_correct = 0
        gen_responses: list[str] = []

        # Reset per-generation curriculum stats
        curriculum.reset_generation_stats()

        # Generate problems at current difficulty
        difficulty = curriculum.difficulty
        problems = generate_batch(n=problems_per_gen, difficulty=difficulty)

        if verbose:
            print(f"\n  [Gen {gen+1}] Generating {len(problems)} problems at Tier {difficulty}...\n")

        # Solve each problem
        for i, problem in enumerate(problems):
            prompt = build_prompt(problem.text)

            try:
                response = llm.generate(
                    prompt,
                    max_tokens=1024,
                    temperature=0.7,
                )
            except Exception as e:
                response = f"[ERROR: {e}]"

            gen_responses.append(response)

            # Verify
            result = verify_with_function(
                response,
                problem.verification_func,
                problem.correct_answer,
            )

            is_correct = result.is_correct

            # Record for curriculum
            curriculum.record(problem.domain, is_correct)

            if is_correct:
                gen_correct += 1
                total_correct += 1
                # Collect successful trace
                collector.add_trace(
                    problem=problem.text,
                    response=response,
                    domain=problem.domain,
                    difficulty=problem.difficulty,
                    generation=gen,
                )

            total_attempted += 1

            # Log to DB
            logger.log_loop_iteration(
                generation=gen,
                problem=problem.text,
                solution=result.parsed_answer,
                verified=is_correct,
                reasoning_trace=response if is_correct else None,
            )

            # Per-problem detail
            if verbose:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"    [{i+1:2d}/{problems_per_gen}] [{problem.domain:^15s}] "
                      f"[{status:^7s}] "
                      f"Expected: {problem.correct_answer:>10s}  "
                      f"Got: {result.parsed_answer:>10s}")

            # Store result
            all_results.append({
                "generation": gen,
                "problem": problem.text,
                "domain": problem.domain,
                "difficulty": problem.difficulty,
                "correct": is_correct,
                "expected": problem.correct_answer,
                "parsed": result.parsed_answer,
            })

        # --- Post-generation ---
        gen_accuracy = gen_correct / problems_per_gen if problems_per_gen > 0 else 0
        old_diff, new_diff = curriculum.update_difficulty()
        domain_report = curriculum.get_domain_report()

        # Collapse detection
        collapse.record_generation(gen_accuracy, gen_responses)
        collapsed, collapse_reason = collapse.is_collapsed()

        # Fine-tuning check
        ft_status = None
        if collector.total_collected >= fine_tune_threshold and not fine_tune_triggered:
            ft_status = "TRIGGERED"
            fine_tune_triggered = True

            print(f"\n  [FINE-TUNE] {collector.total_collected} traces collected -- initiating LoRA training")
            train_path, valid_path = collector.prepare_train_valid_split()
            ft_result = fine_tuner.fine_tune(train_path, valid_path, generation=gen, llm=llm)
            ft_status = f"{ft_result['status'].upper()} (gen {gen+1})"

            # Try to reload model with adapter if successful
            if ft_result["status"] == "success":
                adapter_path = fine_tuner.get_latest_adapter()
                if adapter_path:
                    _try_reload_with_adapter(llm, adapter_path)
        elif fine_tune_triggered:
            # Check if we should do another round
            new_traces = collector.total_collected - fine_tune_threshold
            if new_traces >= fine_tune_threshold:
                ft_status = "RE-TRIGGERED"
                print(f"\n  [FINE-TUNE] {new_traces} new traces -- re-training LoRA")
                train_path, valid_path = collector.prepare_train_valid_split()
                ft_result = fine_tuner.fine_tune(train_path, valid_path, generation=gen, llm=llm)
                ft_status = f"{ft_result['status'].upper()} (gen {gen+1}, round 2)"

        gen_elapsed = time.time() - gen_start

        # Dashboard
        print_dashboard(
            generation=gen,
            total_generations=generations,
            correct=gen_correct,
            total=problems_per_gen,
            old_diff=old_diff,
            new_diff=new_diff,
            domain_report=domain_report,
            total_traces=collector.total_collected,
            fine_tune_status=ft_status,
            elapsed=gen_elapsed,
            collapse_warning=collapse_reason if collapsed else None,
        )

        # Save generation summary
        generation_summaries.append({
            "generation": gen,
            "accuracy": gen_accuracy,
            "correct": gen_correct,
            "total": problems_per_gen,
            "difficulty": old_diff,
            "new_difficulty": new_diff,
            "domain_accuracy": domain_report,
            "traces_collected": collector.total_collected,
            "fine_tune_status": ft_status,
            "elapsed_seconds": round(gen_elapsed, 2),
        })

        # Log generation to experiment DB
        logger.log_experiment(
            project="self_improving_loop",
            run_id=run_id,
            status="generation_complete",
            params={"generation": gen, "difficulty": old_diff},
            metrics={
                "accuracy": gen_accuracy,
                "correct": gen_correct,
                "total": problems_per_gen,
                "traces": collector.total_collected,
            },
        )

        # Stop on collapse
        if collapsed:
            print(f"\n  [COLLAPSE] Stopping early: {collapse_reason}")
            stopped_early = True
            stop_reason = collapse_reason
            break

        # Diversify if needed
        if collapse.should_diversify():
            print("  [DIVERSIFY] Accuracy dropping fast -- mixing in lower-tier problems next gen")
            if curriculum.difficulty > 1:
                curriculum.difficulty -= 1

    # --- Final summary ---
    total_elapsed = time.time() - loop_start

    # Save traces
    traces_path = collector.save_traces()
    full_traces_path = collector.save_full_traces()

    summary = {
        "run_id": run_id,
        "model": model_name,
        "total_generations": len(generation_summaries),
        "total_problems": total_attempted,
        "total_correct": total_correct,
        "overall_accuracy": total_correct / total_attempted if total_attempted > 0 else 0,
        "total_traces": collector.total_collected,
        "traces_file": str(traces_path),
        "full_traces_file": str(full_traces_path),
        "fine_tune_triggered": fine_tune_triggered,
        "fine_tune_history": fine_tuner.training_history,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "generations": generation_summaries,
        "collector_stats": collector.get_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Log final experiment
    logger.log_experiment(
        project="self_improving_loop",
        run_id=run_id,
        status="completed" if not stopped_early else "collapsed",
        params={
            "generations": generations,
            "problems_per_gen": problems_per_gen,
            "start_difficulty": start_difficulty,
            "model": model_name,
        },
        metrics={
            "overall_accuracy": summary["overall_accuracy"],
            "total_correct": total_correct,
            "total_traces": collector.total_collected,
        },
    )

    logger.close()

    # Print final report
    _print_final_report(summary)

    return summary


def _try_reload_with_adapter(llm, adapter_path: Path) -> None:
    """Try to reload the MLX model with a new LoRA adapter."""
    try:
        if hasattr(llm, '_model') and hasattr(llm, '_tokenizer'):
            from mlx_lm import load
            print(f"  [RELOAD] Loading model with adapter from {adapter_path}...")
            model, tokenizer = load(
                llm.model_name,
                adapter_path=str(adapter_path),
            )
            llm._model = model
            llm._tokenizer = tokenizer
            print("  [RELOAD] Model updated with new adapter weights")
        else:
            print("  [RELOAD] Skipped -- API backend does not support adapter loading")
    except Exception as e:
        print(f"  [RELOAD] Failed to load adapter: {e}")


def _print_final_report(summary: dict) -> None:
    """Print the final run report."""
    print()
    print("*" * 68)
    print("  SELF-IMPROVING LOOP -- FINAL REPORT")
    print("*" * 68)
    print(f"  Run ID:            {summary['run_id']}")
    print(f"  Model:             {summary['model']}")
    print(f"  Generations:       {summary['total_generations']}")
    print(f"  Total problems:    {summary['total_problems']}")
    print(f"  Total correct:     {summary['total_correct']}")
    print(f"  Overall accuracy:  {summary['overall_accuracy']*100:.1f}%")
    print(f"  Traces collected:  {summary['total_traces']}")
    print(f"  Fine-tuning:       {'Yes' if summary['fine_tune_triggered'] else 'No'}")
    print(f"  Elapsed:           {summary['total_elapsed_seconds']:.1f}s")

    if summary['stopped_early']:
        print(f"  Early stop:        {summary['stop_reason']}")

    # Show accuracy trend
    if summary['generations']:
        accs = [g['accuracy'] * 100 for g in summary['generations']]
        print()
        print("  Accuracy by generation:")
        for i, acc in enumerate(accs):
            bar_len = int(acc / 5)
            bar = "#" * bar_len
            print(f"    Gen {i+1:2d}: {bar:<20s} {acc:.1f}%")

    # Domain breakdown across all generations
    stats = summary.get("collector_stats", {})
    domain_counts = stats.get("by_domain", {})
    if domain_counts:
        print()
        print("  Traces by domain:")
        for d, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"    {d:<17s} {count:>4d} traces")

    print()
    print(f"  Traces saved to:   {summary['traces_file']}")
    print("*" * 68)
    print()


def save_report(summary: dict, path: Path | None = None) -> Path:
    """Save the summary report to JSON."""
    if path is None:
        path = EXPERIMENTS_DIR / "loop_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make the summary JSON-serializable
    serializable = _make_serializable(summary)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Report saved to {path}")
    return path


def _make_serializable(obj):
    """Recursively convert Path objects etc. to strings for JSON."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj
