#!/usr/bin/env python3
"""Entry point for the Self-Improving Loop.

Usage:
    python -m scripts.run_loop --backend mlx --generations 10 --problems-per-gen 20
    python -m scripts.run_loop --backend api --model claude-sonnet-4-6 --verbose
    python -m scripts.run_loop --start-difficulty 2 --generations 5

Run from the project root directory.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Hyperspace AGI -- Self-Improving Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --backend mlx --generations 5
  %(prog)s --backend api --model claude-sonnet-4-6 --verbose
  %(prog)s --start-difficulty 3 --problems-per-gen 10
        """,
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "api"],
        default=os.environ.get("LLM_BACKEND", "mlx"),
        help="LLM backend to use (default: mlx, or $LLM_BACKEND)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/path. Defaults: mlx-community/Qwen2.5-3B-Instruct-4bit (mlx) "
             "or claude-sonnet-4-6 (api)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations to run (default: 10)",
    )
    parser.add_argument(
        "--problems-per-gen",
        type=int,
        default=20,
        help="Problems per generation (default: 20)",
    )
    parser.add_argument(
        "--start-difficulty",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Starting difficulty tier (default: 1)",
    )
    parser.add_argument(
        "--fine-tune-threshold",
        type=int,
        default=50,
        help="Number of traces to collect before triggering fine-tuning (default: 50)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-problem output",
    )

    args = parser.parse_args()

    # Override env vars based on CLI args
    os.environ["LLM_BACKEND"] = args.backend

    if args.model:
        if args.backend == "mlx":
            os.environ["MLX_MODEL"] = args.model
        else:
            os.environ["API_MODEL"] = args.model

    # Import after setting env vars so config picks them up
    from src.config import get_llm, MLX_MODEL, API_MODEL
    from src.loop.runner import run_loop, save_report

    # Determine model name
    if args.model:
        model_name = args.model
    elif args.backend == "mlx":
        model_name = MLX_MODEL
    else:
        model_name = API_MODEL

    # Initialize LLM
    print(f"\n  Initializing {args.backend} backend with model: {model_name}")
    try:
        llm = get_llm()
    except Exception as e:
        print(f"\n  ERROR: Failed to initialize LLM backend: {e}")
        print(f"  Make sure the required packages are installed:")
        if args.backend == "mlx":
            print(f"    pip install mlx-lm")
        else:
            print(f"    pip install anthropic")
            print(f"    export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    # Run the loop
    try:
        summary = run_loop(
            llm=llm,
            generations=args.generations,
            problems_per_gen=args.problems_per_gen,
            start_difficulty=args.start_difficulty,
            fine_tune_threshold=args.fine_tune_threshold,
            verbose=args.verbose,
            model_name=model_name,
        )
    except KeyboardInterrupt:
        print("\n\n  Loop interrupted by user.")
        sys.exit(0)

    # Save report
    report_path = save_report(summary)
    print(f"\n  Full report: {report_path}")
    print(f"  Database:    {PROJECT_ROOT / 'experiments' / 'experiments.db'}")
    print()


if __name__ == "__main__":
    main()
