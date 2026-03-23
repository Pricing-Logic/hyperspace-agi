"""Evolutionary program improvement for ARC solver.

Takes partially-successful programs and improves them via:
- LLM-guided mutation (show what went wrong, ask for a fix)
- Crossover (combine working parts of two programs)
- Fitness tracking and plateau detection
"""

import re
import random
from dataclasses import dataclass, field

import numpy as np

from src.arc.dsl import DSLLibrary
from src.arc.grid import format_grid_for_prompt, grids_match
from src.arc.loader import Puzzle
from src.arc.sandbox import execute_program
from src.arc.synthesizer import Candidate, evaluate_candidate, _extract_code
from src.config import MAX_GENERATIONS, MUTATION_RATE, POPULATION_SIZE


@dataclass
class EvolutionStats:
    """Tracks evolution progress across generations."""
    generation: int = 0
    best_score: float = 0.0
    best_code: str = ""
    fitness_history: list[float] = field(default_factory=list)
    total_candidates_evaluated: int = 0
    plateau_count: int = 0  # generations without improvement


def _build_mutation_prompt(
    candidate: Candidate,
    puzzle: Puzzle,
    dsl: DSLLibrary,
) -> str:
    """Build a prompt asking the LLM to fix a partially-correct program.

    Shows the LLM:
    - The original code
    - Which training pairs it got right/wrong
    - The specific errors or mismatches
    - Asks for a corrected version
    """
    parts = [
        "You are fixing an ARC-AGI puzzle solution that partially works.",
        "",
        "=== TRAINING EXAMPLES ===",
    ]

    # Show all training pairs with pass/fail status
    for i, pair in enumerate(puzzle.train_pairs):
        in_h, in_w = pair.input.shape
        out_h, out_w = pair.output.shape

        # Run the candidate to see its actual output
        result, run_error = execute_program(candidate.code, pair.input, dsl)

        status = "PASS" if (result is not None and grids_match(result, pair.output)) else "FAIL"

        parts.append(f"Example {i + 1} [{status}]:")
        parts.append(f"  Input ({in_h}x{in_w}):")
        for row in pair.input:
            parts.append("    " + "".join(str(int(v)) for v in row))
        parts.append(f"  Expected Output ({out_h}x{out_w}):")
        for row in pair.output:
            parts.append("    " + "".join(str(int(v)) for v in row))

        if status == "FAIL":
            if run_error:
                parts.append(f"  ERROR: {run_error[:200]}")
            elif result is not None:
                parts.append(f"  Actual Output ({result.shape[0]}x{result.shape[1]}):")
                for row in result:
                    parts.append("    " + "".join(str(int(v)) for v in row))
                if result.shape == pair.output.shape:
                    diff = (result != pair.output)
                    diff_positions = list(zip(*np.where(diff)))
                    n_diff = len(diff_positions)
                    parts.append(f"  {n_diff} cells differ")
                    for r, c in diff_positions[:5]:
                        parts.append(f"    ({r},{c}): expected {pair.output[r,c]}, got {result[r,c]}")
                    if n_diff > 5:
                        parts.append(f"    ...and {n_diff - 5} more")
                else:
                    parts.append(f"  Shape mismatch: expected {pair.output.shape}, got {result.shape}")
        parts.append("")

    parts.extend([
        "=== CURRENT (BUGGY) CODE ===",
        "```python",
        candidate.code,
        "```",
        "",
        f"Score: {candidate.pairs_passed}/{candidate.total_pairs} training pairs correct",
        "",
        "=== AVAILABLE PRIMITIVES ===",
        dsl.get_prompt_description(),
        "",
        "=== YOUR TASK ===",
        "Fix the code so it correctly transforms ALL training examples.",
        "Look carefully at:",
        "- What the code gets right (to keep that logic)",
        "- What it gets wrong (specific cells or shapes that differ)",
        "- The pattern across ALL examples",
        "",
        "Write the complete corrected Python function. Include any helper functions.",
        "The function must be named `transform` and take/return np.ndarray.",
        "",
        "```python",
    ])

    return "\n".join(parts)


def _build_crossover_prompt(
    parent_a: Candidate,
    parent_b: Candidate,
    puzzle: Puzzle,
    dsl: DSLLibrary,
) -> str:
    """Build a prompt to combine two partially-working programs.

    Shows both programs and their scores, asks the LLM to synthesize
    the best parts of each into a better solution.
    """
    examples_str = ""
    for i, pair in enumerate(puzzle.train_pairs):
        examples_str += f"Example {i + 1}:\n"
        examples_str += f"  Input ({pair.input.shape[0]}x{pair.input.shape[1]}):\n"
        for row in pair.input:
            examples_str += "    " + "".join(str(int(v)) for v in row) + "\n"
        examples_str += f"  Output ({pair.output.shape[0]}x{pair.output.shape[1]}):\n"
        for row in pair.output:
            examples_str += "    " + "".join(str(int(v)) for v in row) + "\n"
        examples_str += "\n"

    parts = [
        "You are combining two partial ARC puzzle solutions into a better one.",
        "",
        "=== TRAINING EXAMPLES ===",
        examples_str,
        f"=== PROGRAM A (score: {parent_a.score:.2f}, {parent_a.pairs_passed}/{parent_a.total_pairs}) ===",
        "```python",
        parent_a.code,
        "```",
        "",
        f"=== PROGRAM B (score: {parent_b.score:.2f}, {parent_b.pairs_passed}/{parent_b.total_pairs}) ===",
        "```python",
        parent_b.code,
        "```",
        "",
        "=== YOUR TASK ===",
        "Create a new solution that combines the best insights from both programs.",
        "The goal is to pass ALL training examples.",
        "Focus on what each program gets right and merge those strategies.",
        "",
        "Write the complete Python function `transform(grid: np.ndarray) -> np.ndarray`.",
        "Include any needed helper functions.",
        "",
        "```python",
    ]
    return "\n".join(parts)


class ProgramEvolver:
    """Evolves partially-successful programs toward perfect solutions.

    Uses an evolutionary strategy:
    1. Start with a population of candidates from the synthesizer
    2. Each generation: mutate the best candidates, try crossover
    3. Keep the best candidates, discard the rest
    4. Stop when a perfect solution is found, generations exhausted,
       or fitness plateaus
    """

    def __init__(
        self,
        llm,
        dsl: DSLLibrary | None = None,
        max_generations: int | None = None,
        population_size: int | None = None,
        mutation_rate: float | None = None,
    ):
        """
        Args:
            llm: An LLMInterface instance.
            dsl: DSLLibrary instance. Created fresh if None.
            max_generations: Maximum number of evolution generations.
            population_size: How many candidates to keep per generation.
            mutation_rate: Probability of mutation vs crossover.
        """
        self.llm = llm
        self.dsl = dsl or DSLLibrary()
        self.max_generations = max_generations or MAX_GENERATIONS
        self.population_size = population_size or POPULATION_SIZE
        self.mutation_rate = mutation_rate or MUTATION_RATE

    def evolve(
        self,
        puzzle: Puzzle,
        initial_candidates: list[Candidate],
        verbose: bool = False,
    ) -> tuple[Candidate | None, EvolutionStats]:
        """Evolve a population of candidates toward a perfect solution.

        Args:
            puzzle: The puzzle being solved.
            initial_candidates: Starting population from the synthesizer.
            verbose: Print progress.

        Returns:
            (best_candidate, stats) — best_candidate is None if population was empty.
        """
        stats = EvolutionStats()

        # Initialize population with the best initial candidates
        population = sorted(initial_candidates, key=lambda c: -c.score)
        population = population[:self.population_size]

        if not population:
            return None, stats

        stats.best_score = population[0].score
        stats.best_code = population[0].code
        stats.fitness_history.append(stats.best_score)
        stats.total_candidates_evaluated = len(initial_candidates)

        if stats.best_score == 1.0:
            if verbose:
                print("  Already have a perfect solution!")
            return population[0], stats

        plateau_threshold = 3  # stop after this many generations without improvement

        for gen in range(1, self.max_generations + 1):
            stats.generation = gen

            if verbose:
                print(f"\n  Generation {gen}/{self.max_generations} "
                      f"(best: {stats.best_score:.2f}, plateau: {stats.plateau_count})")

            new_candidates = []

            # Determine how many mutations vs crossovers
            n_mutations = max(1, int(self.population_size * self.mutation_rate))
            n_crossovers = max(1, self.population_size - n_mutations)

            # Mutations: take the best candidates and mutate them
            for i in range(min(n_mutations, len(population))):
                parent = population[i]
                child = self._mutate(parent, puzzle, verbose)
                if child is not None:
                    new_candidates.append(child)
                    stats.total_candidates_evaluated += 1

            # Crossover: combine pairs of good candidates
            if len(population) >= 2:
                for _ in range(n_crossovers):
                    # Tournament selection: pick two parents from top half
                    top_half = population[:max(2, len(population) // 2)]
                    parent_a = random.choice(top_half)
                    parent_b = random.choice(top_half)
                    if parent_a.code == parent_b.code:
                        continue  # skip if same program

                    child = self._crossover(parent_a, parent_b, puzzle, verbose)
                    if child is not None:
                        new_candidates.append(child)
                        stats.total_candidates_evaluated += 1

            # Merge new candidates with existing population
            all_candidates = population + new_candidates

            # Deduplicate by code (keep the one with higher score if same code)
            seen_code: dict[str, Candidate] = {}
            for c in all_candidates:
                stripped = c.code.strip()
                if stripped not in seen_code or c.score > seen_code[stripped].score:
                    seen_code[stripped] = c
            all_candidates = list(seen_code.values())

            # Select the best for next generation
            all_candidates.sort(key=lambda c: (-c.score, len(c.code)))
            population = all_candidates[:self.population_size]

            # Update stats
            current_best = population[0].score if population else 0.0
            stats.fitness_history.append(current_best)

            if current_best > stats.best_score:
                stats.best_score = current_best
                stats.best_code = population[0].code
                stats.plateau_count = 0
                if verbose:
                    print(f"  New best: {stats.best_score:.2f}")
            else:
                stats.plateau_count += 1

            # Update generation number on all candidates
            for c in population:
                c.generation = gen

            # Check stop conditions
            if stats.best_score == 1.0:
                if verbose:
                    print("  Perfect solution found!")
                break

            if stats.plateau_count >= plateau_threshold:
                if verbose:
                    print(f"  Fitness plateau ({plateau_threshold} generations without improvement). Stopping.")
                break

        best = population[0] if population else None
        return best, stats

    def _mutate(self, parent: Candidate, puzzle: Puzzle, verbose: bool = False) -> Candidate | None:
        """Mutate a candidate by asking the LLM to fix its errors.

        Args:
            parent: The candidate to mutate.
            puzzle: The puzzle being solved.
            verbose: Print progress.

        Returns:
            A new Candidate, or None if mutation failed.
        """
        if verbose:
            print(f"    Mutating (parent score: {parent.score:.2f})...", end=" ", flush=True)

        prompt = _build_mutation_prompt(parent, puzzle, self.dsl)

        try:
            raw_response = self.llm.generate(prompt, max_tokens=4096, temperature=0.3)
            code = _extract_code(raw_response)

            if not code or "def transform" not in code:
                if verbose:
                    print("SKIP")
                return None

            child = evaluate_candidate(code, puzzle, self.dsl)
            child.generation = parent.generation + 1

            if verbose:
                delta = child.score - parent.score
                direction = "+" if delta > 0 else ""
                print(f"score={child.score:.2f} ({direction}{delta:.2f})")

            return child

        except Exception as e:
            if verbose:
                print(f"ERROR ({e})")
            return None

    def _crossover(self, parent_a: Candidate, parent_b: Candidate, puzzle: Puzzle, verbose: bool = False) -> Candidate | None:
        """Combine two candidates by asking the LLM to merge their approaches.

        Args:
            parent_a: First parent candidate.
            parent_b: Second parent candidate.
            puzzle: The puzzle being solved.
            verbose: Print progress.

        Returns:
            A new Candidate, or None if crossover failed.
        """
        if verbose:
            print(f"    Crossover (scores: {parent_a.score:.2f} x {parent_b.score:.2f})...", end=" ", flush=True)

        prompt = _build_crossover_prompt(parent_a, parent_b, puzzle, self.dsl)

        try:
            raw_response = self.llm.generate(prompt, max_tokens=4096, temperature=0.4)
            code = _extract_code(raw_response)

            if not code or "def transform" not in code:
                if verbose:
                    print("SKIP")
                return None

            child = evaluate_candidate(code, puzzle, self.dsl)
            child.generation = max(parent_a.generation, parent_b.generation) + 1

            if verbose:
                print(f"score={child.score:.2f}")

            return child

        except Exception as e:
            if verbose:
                print(f"ERROR ({e})")
            return None
