"""Training Data Collector & LoRA Fine-tuner.

Collects successful chain-of-thought reasoning traces and manages LoRA
fine-tuning via MLX when enough data has accumulated.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TRACES_DIR = EXPERIMENTS_DIR / "traces"
ADAPTERS_DIR = EXPERIMENTS_DIR / "adapters"


class TrainingDataCollector:
    """Accumulates successful reasoning traces for fine-tuning."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or TRACES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traces: list[dict] = []
        self.total_collected: int = 0
        self.generation_counts: dict[int, int] = {}

    def add_trace(
        self,
        problem: str,
        response: str,
        domain: str,
        difficulty: int,
        generation: int,
    ) -> None:
        """Add a successful reasoning trace.

        Args:
            problem: The problem text.
            response: The full model response (chain-of-thought + answer).
            domain: Problem domain.
            difficulty: Difficulty tier.
            generation: Which loop generation produced this trace.
        """
        trace = {
            "prompt": problem,
            "completion": response,
            "domain": domain,
            "difficulty": difficulty,
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.traces.append(trace)
        self.total_collected += 1
        self.generation_counts[generation] = self.generation_counts.get(generation, 0) + 1

    def save_traces(self, filename: str | None = None) -> Path:
        """Save all collected traces to a JSONL file.

        Returns the path to the saved file.
        Uses "text" field format compatible with mlx_lm.lora.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traces_{timestamp}.jsonl"

        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            for trace in self.traces:
                # MLX LoRA expects {"text": "full prompt + completion"} format
                ft_record = {
                    "text": trace["prompt"] + "\n" + trace["completion"],
                }
                f.write(json.dumps(ft_record) + "\n")

        return filepath

    def save_full_traces(self, filename: str | None = None) -> Path:
        """Save traces with full metadata (for analysis)."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traces_full_{timestamp}.jsonl"

        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            for trace in self.traces:
                f.write(json.dumps(trace) + "\n")

        return filepath

    def prepare_train_valid_split(
        self, train_ratio: float = 0.9
    ) -> tuple[Path, Path]:
        """Split traces into train/validation sets and save both.

        Returns (train_path, valid_path).
        """
        import random

        shuffled = list(self.traces)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_data = shuffled[:split_idx]
        valid_data = shuffled[split_idx:]

        # Ensure validation set has at least a few examples
        if len(valid_data) < 3 and len(shuffled) > 3:
            split_idx = len(shuffled) - 3
            train_data = shuffled[:split_idx]
            valid_data = shuffled[split_idx:]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_path = self.output_dir / f"train_{timestamp}.jsonl"
        valid_path = self.output_dir / f"valid_{timestamp}.jsonl"

        for path, data in [(train_path, train_data), (valid_path, valid_data)]:
            with open(path, "w") as f:
                for trace in data:
                    # MLX LoRA expects {"text": "..."} format
                    record = {
                        "text": trace["prompt"] + "\n" + trace["completion"],
                    }
                    f.write(json.dumps(record) + "\n")

        return train_path, valid_path

    def get_stats(self) -> dict:
        """Return collection statistics."""
        domain_counts: dict[str, int] = {}
        difficulty_counts: dict[int, int] = {}

        for trace in self.traces:
            d = trace.get("domain", "unknown")
            diff = trace.get("difficulty", 0)
            domain_counts[d] = domain_counts.get(d, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        return {
            "total_traces": self.total_collected,
            "by_domain": domain_counts,
            "by_difficulty": difficulty_counts,
            "by_generation": dict(self.generation_counts),
        }


# ---------------------------------------------------------------------------
# LoRA Fine-tuner
# ---------------------------------------------------------------------------

class LoRAFineTuner:
    """Manages MLX LoRA fine-tuning of the local model."""

    def __init__(
        self,
        model_name: str,
        adapter_dir: Path | None = None,
        rank: int = 8,
        alpha: int = 16,
        target_modules: list[str] | None = None,
        num_steps: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.adapter_dir = adapter_dir or ADAPTERS_DIR
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_history: list[dict] = []

    def fine_tune(
        self,
        train_path: Path,
        valid_path: Path,
        generation: int,
        llm=None,
    ) -> dict:
        """Attempt LoRA fine-tuning using MLX via subprocess.

        Uses subprocess to avoid double model loading (OOM risk on 48GB Mac).

        Args:
            train_path: Path to training JSONL.
            valid_path: Path to validation JSONL.
            generation: Current generation number (for naming the adapter).
            llm: Optional LLM instance — will be unloaded before training to free RAM.

        Returns:
            Dict with status, adapter_path, and any error info.
        """
        adapter_name = f"adapter_gen{generation:03d}"
        adapter_path = self.adapter_dir / adapter_name

        result = {
            "status": "pending",
            "adapter_path": str(adapter_path),
            "generation": generation,
            "train_size": _count_lines(train_path),
            "valid_size": _count_lines(valid_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Unload inference model to free RAM before training
        if llm is not None and hasattr(llm, 'unload'):
            print("  [TRAINER] Unloading inference model to free RAM...")
            llm.unload()

        try:
            return self._run_mlx_subprocess(train_path, valid_path, adapter_path, result)
        except FileNotFoundError:
            result["status"] = "skipped"
            result["reason"] = "mlx_lm not installed or python not found"
            result["instructions"] = (
                "To enable fine-tuning, install mlx-lm:\n"
                "  pip install mlx-lm\n"
                f"Training data saved at: {train_path}\n"
                f"Manual fine-tune command:\n"
                f"  python -m mlx_lm.lora --model {self.model_name} "
                f"--train --data {train_path.parent} "
                f"--adapter-path {adapter_path} --iters {self.num_steps}"
            )
            print(f"\n  [TRAINER] {result['instructions']}")
            self.training_history.append(result)
            return result
        except Exception as e:
            result["status"] = "failed"
            result["reason"] = str(e)
            print(f"\n  [TRAINER] Fine-tuning failed: {e}")
            self.training_history.append(result)
            return result

    def _run_mlx_subprocess(
        self, train_path: Path, valid_path: Path, adapter_path: Path, result: dict
    ) -> dict:
        """Run MLX LoRA fine-tuning via subprocess (avoids double model loading)."""
        import subprocess

        adapter_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self.model_name,
            "--train",
            "--data", str(train_path.parent),
            "--adapter-path", str(adapter_path),
            "--iters", str(self.num_steps),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--lora-rank", str(self.rank),
        ]

        print(f"\n  [TRAINER] Starting LoRA training via subprocess:")
        print(f"  [TRAINER]   {' '.join(cmd)}")
        print(f"  [TRAINER]   Steps={self.num_steps}, LR={self.learning_rate}, Rank={self.rank}")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        elapsed = time.time() - t0

        if proc.returncode != 0:
            stderr = proc.stderr[-500:] if len(proc.stderr) > 500 else proc.stderr
            raise RuntimeError(f"LoRA training failed (exit {proc.returncode}):\n{stderr}")

        result["status"] = "success"
        result["training_time_seconds"] = round(elapsed, 2)
        print(f"  [TRAINER] Fine-tuning complete in {elapsed:.1f}s")
        print(f"  [TRAINER] Adapter saved to {adapter_path}")

        self.training_history.append(result)
        return result

    def get_latest_adapter(self) -> Path | None:
        """Return the path to the most recently trained adapter."""
        if not self.training_history:
            return None
        successful = [h for h in self.training_history if h["status"] == "success"]
        if not successful:
            return None
        return Path(successful[-1]["adapter_path"])


def _count_lines(path: Path) -> int:
    """Count lines in a file."""
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
