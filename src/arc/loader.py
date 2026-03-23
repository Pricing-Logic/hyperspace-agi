"""ARC puzzle loader — reads puzzle JSON files into structured dataclasses."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import ARC_DATA_DIR


@dataclass
class Pair:
    """A single input/output example pair."""
    input: np.ndarray
    output: np.ndarray

    def __post_init__(self):
        if not isinstance(self.input, np.ndarray):
            self.input = np.array(self.input, dtype=int)
        if not isinstance(self.output, np.ndarray):
            self.output = np.array(self.output, dtype=int)


@dataclass
class Puzzle:
    """An ARC puzzle with training and test pairs."""
    puzzle_id: str
    train_pairs: list[Pair] = field(default_factory=list)
    test_pairs: list[Pair] = field(default_factory=list)
    source: str = "training"  # "training" or "evaluation"

    @property
    def num_train(self) -> int:
        return len(self.train_pairs)

    @property
    def num_test(self) -> int:
        return len(self.test_pairs)

    def summary(self) -> str:
        """One-line summary of puzzle dimensions."""
        train_dims = [
            f"{p.input.shape} -> {p.output.shape}" for p in self.train_pairs
        ]
        return (
            f"Puzzle {self.puzzle_id}: "
            f"{self.num_train} train, {self.num_test} test | "
            f"dims: {', '.join(train_dims)}"
        )


def _parse_puzzle(puzzle_id: str, data: dict, source: str = "training") -> Puzzle:
    """Parse a puzzle dict (from JSON) into a Puzzle dataclass."""
    train_pairs = [
        Pair(
            input=np.array(example["input"], dtype=int),
            output=np.array(example["output"], dtype=int),
        )
        for example in data["train"]
    ]
    test_pairs = [
        Pair(
            input=np.array(example["input"], dtype=int),
            output=np.array(example["output"], dtype=int),
        )
        for example in data["test"]
    ]
    return Puzzle(
        puzzle_id=puzzle_id,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        source=source,
    )


def load_puzzle(puzzle_id: str, split: str | None = None) -> Puzzle:
    """Load a single puzzle by ID.

    Args:
        puzzle_id: The puzzle filename stem (e.g. "007bbfb7").
        split: "training" or "evaluation". If None, searches both.

    Returns:
        The loaded Puzzle.

    Raises:
        FileNotFoundError: If the puzzle ID doesn't exist.
    """
    if not puzzle_id.endswith(".json"):
        filename = f"{puzzle_id}.json"
    else:
        filename = puzzle_id
        puzzle_id = puzzle_id.replace(".json", "")

    splits_to_check = [split] if split else ["training", "evaluation"]

    for s in splits_to_check:
        path = ARC_DATA_DIR / s / filename
        if path.exists():
            data = json.loads(path.read_text())
            return _parse_puzzle(puzzle_id, data, source=s)

    raise FileNotFoundError(
        f"Puzzle '{puzzle_id}' not found in {ARC_DATA_DIR}. "
        f"Run scripts/download_arc.py first."
    )


def load_all_puzzles(split: str = "training") -> list[Puzzle]:
    """Load all puzzles from a split.

    Args:
        split: "training" or "evaluation".

    Returns:
        List of all puzzles in the split, sorted by ID.
    """
    split_dir = ARC_DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Directory {split_dir} not found. Run scripts/download_arc.py first."
        )

    puzzles = []
    for path in sorted(split_dir.glob("*.json")):
        data = json.loads(path.read_text())
        puzzle_id = path.stem
        puzzles.append(_parse_puzzle(puzzle_id, data, source=split))

    return puzzles


def load_random_puzzles(n: int, split: str = "training", seed: int | None = None) -> list[Puzzle]:
    """Load a random subset of puzzles.

    Args:
        n: Number of puzzles to load.
        split: "training" or "evaluation".
        seed: Random seed for reproducibility. None for random.

    Returns:
        List of n randomly selected puzzles.
    """
    split_dir = ARC_DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Directory {split_dir} not found. Run scripts/download_arc.py first."
        )

    all_files = sorted(split_dir.glob("*.json"))
    if n > len(all_files):
        n = len(all_files)

    rng = random.Random(seed)
    selected = rng.sample(all_files, n)

    puzzles = []
    for path in selected:
        data = json.loads(path.read_text())
        puzzle_id = path.stem
        puzzles.append(_parse_puzzle(puzzle_id, data, source=split))

    return puzzles
