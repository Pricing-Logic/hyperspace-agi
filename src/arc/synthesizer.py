"""Program synthesis engine — generates candidate Python programs for ARC puzzles.

Uses the LLM to generate transform() functions, tests them against training
pairs, and returns ranked candidates.
"""

import re
import time
from dataclasses import dataclass

import numpy as np

from src.arc.dsl import DSLLibrary
from src.arc.grid import format_grid_for_prompt, grids_match
from src.arc.loader import Puzzle
from src.arc.sandbox import execute_program
from src.config import POPULATION_SIZE


@dataclass
class Candidate:
    """A candidate program with its evaluation results."""
    code: str
    score: float  # fraction of training pairs correct (0.0 to 1.0)
    pairs_passed: int
    total_pairs: int
    errors: list[str]  # error messages from failed pairs
    generation: int = 0


def _format_training_examples(puzzle: Puzzle) -> str:
    """Format all training pairs for the LLM prompt.

    Shows each pair as compact digit grids with clear labeling.
    """
    parts = []
    for i, pair in enumerate(puzzle.train_pairs):
        in_h, in_w = pair.input.shape
        out_h, out_w = pair.output.shape
        parts.append(f"Example {i + 1}:")
        parts.append(f"  Input ({in_h}x{in_w}):")
        for row in pair.input:
            parts.append("    " + "".join(str(int(v)) for v in row))
        parts.append(f"  Output ({out_h}x{out_w}):")
        for row in pair.output:
            parts.append("    " + "".join(str(int(v)) for v in row))
        parts.append("")
    return "\n".join(parts)


def _format_analysis_hints(puzzle: Puzzle) -> str:
    """Generate analysis hints about the puzzle structure.

    These hints help the LLM notice patterns without giving away the answer.
    """
    hints = []

    # Size relationships
    size_changes = []
    for pair in puzzle.train_pairs:
        in_h, in_w = pair.input.shape
        out_h, out_w = pair.output.shape
        if in_h == out_h and in_w == out_w:
            size_changes.append("same_size")
        elif out_h == in_h * 2 and out_w == in_w * 2:
            size_changes.append("doubled")
        elif out_h == in_h * 3 and out_w == in_w * 3:
            size_changes.append("tripled")
        elif out_h > in_h or out_w > in_w:
            size_changes.append(f"grew({in_h}x{in_w}->{out_h}x{out_w})")
        elif out_h < in_h or out_w < in_w:
            size_changes.append(f"shrunk({in_h}x{in_w}->{out_h}x{out_w})")
        else:
            size_changes.append(f"changed({in_h}x{in_w}->{out_h}x{out_w})")

    if len(set(size_changes)) == 1:
        hints.append(f"Size pattern: consistently {size_changes[0]}")
    else:
        hints.append(f"Size changes: {', '.join(size_changes)}")

    # Color analysis
    for i, pair in enumerate(puzzle.train_pairs):
        in_colors = set(int(v) for v in np.unique(pair.input))
        out_colors = set(int(v) for v in np.unique(pair.output))
        new_colors = out_colors - in_colors
        removed_colors = in_colors - out_colors
        if new_colors:
            hints.append(f"Example {i + 1}: new colors in output: {new_colors}")
        if removed_colors:
            hints.append(f"Example {i + 1}: colors removed: {removed_colors}")

    return "\n".join(hints)


def build_synthesis_prompt(puzzle: Puzzle, dsl: DSLLibrary, attempt: int = 0, previous_errors: list[str] | None = None) -> str:
    """Build the full prompt for the LLM to generate a transform function.

    This is the most important function in the system. The quality of this
    prompt directly determines the quality of generated programs.

    Args:
        puzzle: The ARC puzzle to solve.
        dsl: The DSL library (primitives are described in the prompt).
        attempt: Which attempt this is (0-indexed). Later attempts get
                 different instructions to encourage diversity.
        previous_errors: Error messages from prior failed attempts.

    Returns:
        The complete prompt string.
    """
    examples = _format_training_examples(puzzle)
    analysis = _format_analysis_hints(puzzle)
    dsl_desc = dsl.get_prompt_description()

    # Vary the approach instruction based on attempt number to encourage diversity
    approach_hints = [
        "Think step by step. What is the simplest rule that explains ALL the examples?",
        "Look for geometric transformations: rotations, reflections, scaling, tiling.",
        "Look for color-based rules: replacement, counting, pattern matching.",
        "Consider object-level reasoning: find distinct objects and how they relate.",
        "Look for symmetry, repetition, or spatial relationships between elements.",
        "Think about the output as being constructed from parts of the input.",
        "Consider if the transformation involves sorting, reordering, or selecting parts.",
        "Look for boundary/edge effects, padding, cropping, or framing operations.",
    ]
    approach = approach_hints[attempt % len(approach_hints)]

    prompt_parts = [
        "You are solving an ARC-AGI puzzle. ARC puzzles involve discovering a transformation rule from input/output grid examples, then applying that rule to new inputs.",
        "",
        "Each grid is a 2D array of integers 0-9, where each integer represents a color:",
        "  0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=magenta 7=orange 8=cyan 9=maroon",
        "",
        "=== TRAINING EXAMPLES ===",
        examples,
        "=== ANALYSIS ===",
        analysis,
        "",
        f"=== AVAILABLE PRIMITIVES ===",
        dsl_desc,
        "",
        "=== YOUR TASK ===",
        f"{approach}",
        "",
        "Write a Python function `transform(grid: np.ndarray) -> np.ndarray` that:",
        "1. Takes an input grid (2D numpy array of ints 0-9)",
        "2. Returns the correct output grid",
        "3. Works for ALL training examples, not just one",
        "",
        "You have access to numpy (as np) and all the primitives listed above.",
        "The function should handle any valid input grid, not just the specific examples shown.",
        "",
        "IMPORTANT RULES:",
        "- Return a numpy array of dtype int",
        "- Do NOT hardcode outputs for specific examples",
        "- Keep the code concise and clear",
        "- You can use numpy operations directly (slicing, indexing, np.where, etc.)",
        "- You can define helper functions before transform() if needed",
        "",
    ]

    if previous_errors:
        prompt_parts.append("=== PREVIOUS ERRORS (avoid these) ===")
        for err in previous_errors[-3:]:  # show last 3 errors
            truncated = err[:300] if len(err) > 300 else err
            prompt_parts.append(f"  - {truncated}")
        prompt_parts.append("")

    prompt_parts.extend([
        "Write ONLY the Python code. Start with any helper functions, then the transform function.",
        "```python",
    ])

    return "\n".join(prompt_parts)


def _extract_code(raw_response: str) -> str:
    """Extract Python code from LLM response, handling markdown fences."""
    # Try to find code in ```python ... ``` blocks
    python_blocks = re.findall(r"```python\s*(.*?)```", raw_response, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()  # take the last one (most likely the complete version)

    # Try generic ``` blocks
    code_blocks = re.findall(r"```\s*(.*?)```", raw_response, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # If the response contains 'def transform', extract from there
    if "def transform" in raw_response:
        # Find the start of the function definition
        lines = raw_response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if "def transform" in line or (in_code and (line.startswith(" ") or line.startswith("\t") or line.strip() == "" or line.startswith("def "))):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.startswith("def "):
                # Likely hit prose after the code
                if not line.strip().startswith("#"):
                    break
                code_lines.append(line)
        return "\n".join(code_lines).strip()

    # Last resort: return the whole thing
    return raw_response.strip()


def evaluate_candidate(code: str, puzzle: Puzzle, dsl: DSLLibrary) -> Candidate:
    """Test a candidate program against all training pairs.

    Args:
        code: Python source code with a transform() function.
        puzzle: The puzzle to test against.
        dsl: DSL library for sandbox injection.

    Returns:
        Candidate with score and error information.
    """
    total = len(puzzle.train_pairs)
    passed = 0
    errors = []

    for i, pair in enumerate(puzzle.train_pairs):
        result, error = execute_program(code, pair.input, dsl)

        if error is not None:
            errors.append(f"Pair {i + 1}: {error}")
        elif result is None:
            errors.append(f"Pair {i + 1}: No result returned")
        elif not grids_match(result, pair.output):
            # Show what went wrong
            expected_shape = pair.output.shape
            got_shape = result.shape
            if expected_shape != got_shape:
                errors.append(
                    f"Pair {i + 1}: Wrong shape. Expected {expected_shape}, got {got_shape}"
                )
            else:
                diff_count = int(np.sum(result != pair.output))
                total_cells = int(np.prod(pair.output.shape))
                errors.append(
                    f"Pair {i + 1}: {diff_count}/{total_cells} cells differ"
                )
        else:
            passed += 1

    score = passed / total if total > 0 else 0.0
    return Candidate(
        code=code,
        score=score,
        pairs_passed=passed,
        total_pairs=total,
        errors=errors,
    )


class ProgramSynthesizer:
    """Generates candidate Python programs for ARC puzzles using an LLM.

    The synthesizer:
    1. Constructs a detailed prompt from the puzzle
    2. Asks the LLM to generate multiple candidate programs
    3. Tests each against training pairs
    4. Returns candidates ranked by score
    """

    def __init__(self, llm, dsl: DSLLibrary | None = None, population_size: int | None = None):
        """
        Args:
            llm: An LLMInterface instance.
            dsl: DSLLibrary instance. Created fresh if None.
            population_size: Number of candidates to generate per puzzle.
        """
        self.llm = llm
        self.dsl = dsl or DSLLibrary()
        self.population_size = population_size or POPULATION_SIZE

    def generate_candidates(
        self,
        puzzle: Puzzle,
        n: int | None = None,
        verbose: bool = False,
    ) -> list[Candidate]:
        """Generate and evaluate candidate programs for a puzzle.

        Args:
            puzzle: The puzzle to solve.
            n: Number of candidates to generate. Defaults to self.population_size.
            verbose: If True, print progress.

        Returns:
            List of Candidates sorted by score (best first).
        """
        n = n or self.population_size
        candidates = []
        all_errors: list[str] = []

        for attempt in range(n):
            if verbose:
                print(f"  Generating candidate {attempt + 1}/{n}...", end=" ", flush=True)

            # Build prompt, incorporating errors from previous attempts
            prompt = build_synthesis_prompt(
                puzzle, self.dsl, attempt=attempt,
                previous_errors=all_errors[-5:] if all_errors else None,
            )

            # Generate code via LLM
            try:
                # Vary temperature slightly across attempts for diversity
                temp = 0.2 + (attempt * 0.05)
                temp = min(temp, 0.9)

                raw_response = self.llm.generate(
                    prompt,
                    max_tokens=4096,
                    temperature=temp,
                )
                code = _extract_code(raw_response)

                if not code or "def transform" not in code:
                    if verbose:
                        print("SKIP (no transform function)")
                    all_errors.append("LLM did not produce a transform() function")
                    continue

            except Exception as e:
                if verbose:
                    print(f"ERROR ({e})")
                all_errors.append(f"LLM error: {e}")
                continue

            # Evaluate the candidate
            candidate = evaluate_candidate(code, puzzle, self.dsl)
            candidate.generation = 0  # initial generation
            candidates.append(candidate)

            if verbose:
                print(f"score={candidate.score:.2f} ({candidate.pairs_passed}/{candidate.total_pairs})")

            # Collect errors for future prompts
            all_errors.extend(candidate.errors)

            # Early exit if we found a perfect solution
            if candidate.score == 1.0:
                if verbose:
                    print("  Perfect solution found!")
                break

        # Sort by score descending, then by code length (prefer shorter)
        candidates.sort(key=lambda c: (-c.score, len(c.code)))
        return candidates

    def generate_single(self, puzzle: Puzzle, attempt: int = 0, previous_errors: list[str] | None = None) -> Candidate | None:
        """Generate and evaluate a single candidate.

        Useful for the evolution loop which generates one at a time.

        Args:
            puzzle: The puzzle to solve.
            attempt: Attempt number (affects prompt variation).
            previous_errors: Errors from prior attempts.

        Returns:
            A Candidate, or None if generation failed.
        """
        prompt = build_synthesis_prompt(
            puzzle, self.dsl, attempt=attempt,
            previous_errors=previous_errors,
        )

        try:
            temp = 0.2 + (attempt * 0.05)
            temp = min(temp, 0.9)
            raw_response = self.llm.generate(prompt, max_tokens=4096, temperature=temp)
            code = _extract_code(raw_response)

            if not code or "def transform" not in code:
                return None

            candidate = evaluate_candidate(code, puzzle, self.dsl)
            return candidate

        except Exception:
            return None
