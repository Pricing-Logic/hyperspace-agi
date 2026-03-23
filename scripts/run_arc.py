#!/usr/bin/env python3
"""CLI entry point for the ARC-AGI solver.

Usage:
    python -m scripts.run_arc --backend api --puzzles 5
    python -m scripts.run_arc --puzzle-id 007bbfb7 --verbose
    python -m scripts.run_arc --backend mlx --model mlx-community/Qwen2.5-3B-Instruct-4bit --puzzles 10
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_dataset():
    """Download the ARC dataset if it doesn't exist."""
    from src.config import ARC_DATA_DIR

    training_dir = ARC_DATA_DIR / "training"
    if training_dir.exists() and any(training_dir.glob("*.json")):
        return  # already downloaded

    print("ARC dataset not found. Downloading...")
    try:
        from scripts.download_arc import download
        download()
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("You can download manually: python scripts/download_arc.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ARC-AGI Puzzle Solver — Program Synthesis via LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --backend api --puzzles 5          Solve 5 random puzzles using Claude API
  %(prog)s --puzzle-id 007bbfb7              Solve a specific puzzle
  %(prog)s --backend mlx --puzzles 3 -v      Solve 3 puzzles with local MLX model
  %(prog)s --puzzles 10 --seed 42            Reproducible random selection
        """,
    )

    parser.add_argument(
        "--backend", choices=["mlx", "api"], default=None,
        help="LLM backend to use. Overrides LLM_BACKEND env var. (default: from config)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name/path. Overrides MLX_MODEL or API_MODEL env var.",
    )
    parser.add_argument(
        "--puzzles", type=int, default=5,
        help="Number of random puzzles to attempt (default: 5)",
    )
    parser.add_argument(
        "--puzzle-id", type=str, default=None,
        help="Solve a specific puzzle by ID (overrides --puzzles)",
    )
    parser.add_argument(
        "--split", choices=["training", "evaluation"], default="training",
        help="Which puzzle split to use (default: training)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for puzzle selection",
    )
    parser.add_argument(
        "--population", type=int, default=None,
        help="Number of initial candidates per puzzle (default: from config)",
    )
    parser.add_argument(
        "--generations", type=int, default=None,
        help="Max evolution generations (default: from config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    # Apply backend/model overrides via environment
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        backend = args.backend or os.environ.get("LLM_BACKEND", "mlx")
        if backend == "mlx":
            os.environ["MLX_MODEL"] = args.model
        else:
            os.environ["API_MODEL"] = args.model

    # Import after env vars are set so config picks them up
    from src.config import get_llm, LLM_BACKEND, POPULATION_SIZE, MAX_GENERATIONS
    from src.arc.solver import ARCSolver
    from src.arc.dsl import DSLLibrary

    verbose = not args.quiet

    # Ensure dataset exists
    ensure_dataset()

    # Initialize components
    if verbose:
        backend_name = args.backend or LLM_BACKEND
        model_name = args.model or ("(default)" )
        print(f"\n  ARC-AGI Solver")
        print(f"  Backend: {backend_name}")
        print(f"  Model:   {model_name}")
        print()

    llm = get_llm()
    dsl = DSLLibrary()

    pop_size = args.population or POPULATION_SIZE
    max_gen = args.generations or MAX_GENERATIONS

    solver = ARCSolver(
        llm=llm,
        dsl=dsl,
        population_size=pop_size,
        max_generations=max_gen,
        verbose=verbose,
    )

    # Solve
    if args.puzzle_id:
        if verbose:
            print(f"Solving puzzle: {args.puzzle_id}")
        result = solver.solve_by_id(args.puzzle_id)
        if result.solved:
            print(f"\nPuzzle {args.puzzle_id}: SOLVED")
        else:
            print(f"\nPuzzle {args.puzzle_id}: NOT SOLVED (best score: {result.best_score:.2f})")
    else:
        if verbose:
            print(f"Solving {args.puzzles} random {args.split} puzzles"
                  + (f" (seed={args.seed})" if args.seed else ""))
        batch = solver.solve_random(
            n=args.puzzles,
            split=args.split,
            seed=args.seed,
        )
        print(batch.summary())

        # Exit with code reflecting results
        if batch.solved_count == 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
