"""Main ARC solver orchestrator — ties synthesizer, evolution, and logging together.

For each puzzle:
1. Load and display the puzzle
2. Generate initial candidates via synthesizer
3. Test all candidates
4. If none are perfect, evolve the best ones
5. Apply the best program to test input
6. Log results
"""

import time
from dataclasses import dataclass, field

import numpy as np

from src.arc.dsl import DSLLibrary
from src.arc.evolution import ProgramEvolver, EvolutionStats
from src.arc.grid import Grid, display_pair, grids_match
from src.arc.loader import Puzzle, load_puzzle, load_all_puzzles, load_random_puzzles
from src.arc.sandbox import execute_program
from src.arc.synthesizer import ProgramSynthesizer, Candidate
from src.config import DB_PATH, POPULATION_SIZE, MAX_GENERATIONS
from src.experiment_log import ExperimentLogger


@dataclass
class PuzzleResult:
    """Result of solving a single puzzle."""
    puzzle_id: str
    solved: bool
    best_score: float
    best_code: str
    test_predictions: list[np.ndarray | None]
    test_correct: list[bool]
    generations_used: int
    total_candidates: int
    time_seconds: float
    evolution_stats: EvolutionStats | None = None


@dataclass
class BatchResult:
    """Result of solving a batch of puzzles."""
    results: list[PuzzleResult] = field(default_factory=list)
    total_puzzles: int = 0
    solved_count: int = 0
    total_time_seconds: float = 0.0

    @property
    def solve_rate(self) -> float:
        return self.solved_count / self.total_puzzles if self.total_puzzles > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "  ARC SOLVER — BATCH RESULTS",
            "=" * 60,
            f"  Puzzles attempted:  {self.total_puzzles}",
            f"  Puzzles solved:     {self.solved_count}",
            f"  Solve rate:         {self.solve_rate:.1%}",
            f"  Total time:         {self.total_time_seconds:.1f}s",
            f"  Avg time/puzzle:    {self.total_time_seconds / max(1, self.total_puzzles):.1f}s",
            "",
        ]

        if self.results:
            lines.append("  Per-puzzle results:")
            for r in self.results:
                status = "SOLVED" if r.solved else f"FAILED (best: {r.best_score:.2f})"
                lines.append(f"    {r.puzzle_id}: {status}  [{r.time_seconds:.1f}s, {r.total_candidates} candidates]")

        lines.append("=" * 60)
        return "\n".join(lines)


class ARCSolver:
    """Main ARC-AGI puzzle solver.

    Orchestrates the full pipeline: synthesis -> evaluation -> evolution -> testing.
    """

    def __init__(
        self,
        llm,
        dsl: DSLLibrary | None = None,
        population_size: int | None = None,
        max_generations: int | None = None,
        logger: ExperimentLogger | None = None,
        verbose: bool = True,
    ):
        """
        Args:
            llm: An LLMInterface instance.
            dsl: Shared DSLLibrary. Created fresh if None.
            population_size: Number of initial candidates per puzzle.
            max_generations: Max evolution generations.
            logger: ExperimentLogger for recording results. Auto-created if None.
            verbose: Print progress and visualizations.
        """
        self.llm = llm
        self.dsl = dsl or DSLLibrary()
        self.population_size = population_size or POPULATION_SIZE
        self.max_generations = max_generations or MAX_GENERATIONS
        self.verbose = verbose

        self.synthesizer = ProgramSynthesizer(
            llm=self.llm,
            dsl=self.dsl,
            population_size=self.population_size,
        )
        self.evolver = ProgramEvolver(
            llm=self.llm,
            dsl=self.dsl,
            max_generations=self.max_generations,
            population_size=self.population_size,
        )

        if logger is not None:
            self.logger = logger
        else:
            self.logger = ExperimentLogger(DB_PATH)

    def solve_puzzle(self, puzzle: Puzzle) -> PuzzleResult:
        """Solve a single ARC puzzle through the full pipeline.

        Args:
            puzzle: The puzzle to solve.

        Returns:
            PuzzleResult with the outcome.
        """
        start_time = time.time()

        if self.verbose:
            self._display_puzzle(puzzle)

        # Phase 1: Generate initial candidates
        if self.verbose:
            print(f"\n[Phase 1] Generating {self.population_size} initial candidates...")

        candidates = self.synthesizer.generate_candidates(
            puzzle, n=self.population_size, verbose=self.verbose,
        )

        if not candidates:
            if self.verbose:
                print("  No valid candidates generated.")
            elapsed = time.time() - start_time
            return PuzzleResult(
                puzzle_id=puzzle.puzzle_id,
                solved=False,
                best_score=0.0,
                best_code="",
                test_predictions=[],
                test_correct=[],
                generations_used=0,
                total_candidates=0,
                time_seconds=elapsed,
            )

        best = candidates[0]
        total_candidates = len(candidates)

        if self.verbose:
            print(f"\n  Best initial score: {best.score:.2f} "
                  f"({best.pairs_passed}/{best.total_pairs})")

        # Phase 2: Evolution (if not already perfect)
        evolution_stats = None
        if best.score < 1.0:
            if self.verbose:
                print(f"\n[Phase 2] Evolving (max {self.max_generations} generations)...")

            best, evolution_stats = self.evolver.evolve(
                puzzle, candidates, verbose=self.verbose,
            )
            if evolution_stats:
                total_candidates += evolution_stats.total_candidates_evaluated

        # Phase 3: Apply to test inputs
        if self.verbose:
            print(f"\n[Phase 3] Testing on {puzzle.num_test} test input(s)...")

        test_predictions = []
        test_correct = []

        if best is not None and best.code:
            for i, test_pair in enumerate(puzzle.test_pairs):
                result, error = execute_program(best.code, test_pair.input, self.dsl)

                if result is not None:
                    test_predictions.append(result)
                    correct = grids_match(result, test_pair.output)
                    test_correct.append(correct)

                    if self.verbose:
                        status = "CORRECT" if correct else "WRONG"
                        print(f"  Test {i + 1}: {status}")
                        if not correct:
                            print(f"    Expected shape: {test_pair.output.shape}")
                            print(f"    Got shape:      {result.shape}")
                else:
                    test_predictions.append(None)
                    test_correct.append(False)
                    if self.verbose:
                        print(f"  Test {i + 1}: ERROR — {error}")
        else:
            test_predictions = [None] * puzzle.num_test
            test_correct = [False] * puzzle.num_test

        solved = all(test_correct) and len(test_correct) > 0
        elapsed = time.time() - start_time

        # Log the attempt
        self.logger.log_arc_attempt(
            puzzle_id=puzzle.puzzle_id,
            attempt=total_candidates,
            program=best.code if best else "",
            success=solved,
            pairs_passed=best.pairs_passed if best else 0,
            total_pairs=best.total_pairs if best else len(puzzle.train_pairs),
        )

        result = PuzzleResult(
            puzzle_id=puzzle.puzzle_id,
            solved=solved,
            best_score=best.score if best else 0.0,
            best_code=best.code if best else "",
            test_predictions=test_predictions,
            test_correct=test_correct,
            generations_used=evolution_stats.generation if evolution_stats else 0,
            total_candidates=total_candidates,
            time_seconds=elapsed,
            evolution_stats=evolution_stats,
        )

        if self.verbose:
            self._display_result(result)

        return result

    def solve_batch(
        self,
        puzzles: list[Puzzle],
    ) -> BatchResult:
        """Solve a batch of puzzles and track overall results.

        Args:
            puzzles: List of puzzles to solve.

        Returns:
            BatchResult with aggregated outcomes.
        """
        batch = BatchResult(total_puzzles=len(puzzles))
        batch_start = time.time()

        for i, puzzle in enumerate(puzzles):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"  PUZZLE {i + 1}/{len(puzzles)}: {puzzle.puzzle_id}")
                print(f"{'=' * 60}")

            result = self.solve_puzzle(puzzle)
            batch.results.append(result)

            if result.solved:
                batch.solved_count += 1

            if self.verbose:
                running_rate = batch.solved_count / (i + 1)
                print(f"\n  Running solve rate: {batch.solved_count}/{i + 1} ({running_rate:.1%})")

        batch.total_time_seconds = time.time() - batch_start

        if self.verbose:
            print(batch.summary())

        return batch

    def solve_by_id(self, puzzle_id: str) -> PuzzleResult:
        """Load and solve a puzzle by its ID.

        Args:
            puzzle_id: The puzzle filename stem (e.g. "007bbfb7").

        Returns:
            PuzzleResult.
        """
        puzzle = load_puzzle(puzzle_id)
        return self.solve_puzzle(puzzle)

    def solve_random(self, n: int, split: str = "training", seed: int | None = None) -> BatchResult:
        """Solve n random puzzles.

        Args:
            n: Number of puzzles.
            split: "training" or "evaluation".
            seed: Random seed for reproducibility.

        Returns:
            BatchResult.
        """
        puzzles = load_random_puzzles(n, split=split, seed=seed)
        return self.solve_batch(puzzles)

    def _display_puzzle(self, puzzle: Puzzle):
        """Print a puzzle's training pairs to the terminal."""
        print(f"\n  Puzzle: {puzzle.puzzle_id} ({puzzle.source})")
        print(f"  Training pairs: {puzzle.num_train}, Test pairs: {puzzle.num_test}")
        print()

        for i, pair in enumerate(puzzle.train_pairs):
            print(display_pair(pair.input, pair.output, f"Train {i + 1}"))

    def _display_result(self, result: PuzzleResult):
        """Print a summary dashboard for a puzzle result."""
        print()
        print("-" * 40)
        status = "SOLVED" if result.solved else "NOT SOLVED"
        print(f"  {result.puzzle_id}: {status}")
        print(f"  Best training score: {result.best_score:.2f}")
        print(f"  Generations used:    {result.generations_used}")
        print(f"  Candidates tried:    {result.total_candidates}")
        print(f"  Time:                {result.time_seconds:.1f}s")

        if result.test_correct:
            correct = sum(result.test_correct)
            total = len(result.test_correct)
            print(f"  Test accuracy:       {correct}/{total}")

        if result.best_code:
            print(f"  Solution ({len(result.best_code)} chars):")
            for line in result.best_code.split("\n")[:15]:
                print(f"    {line}")
            if result.best_code.count("\n") > 15:
                print(f"    ...({result.best_code.count(chr(10)) - 15} more lines)")

        print("-" * 40)
