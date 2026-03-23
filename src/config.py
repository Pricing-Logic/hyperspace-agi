"""Shared configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARC_DATA_DIR = DATA_DIR / "arc"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DB_PATH = EXPERIMENTS_DIR / "experiments.db"

# LLM config
LLM_BACKEND = os.environ.get("LLM_BACKEND", "mlx")  # "mlx" or "api"
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
API_MODEL = os.environ.get("API_MODEL", "claude-sonnet-4-6")

# Sandbox config
SANDBOX_TIMEOUT_SECONDS = 5
MAX_PROGRAM_LENGTH = 2000

# Evolution config
POPULATION_SIZE = 20
MAX_GENERATIONS = 15
MUTATION_RATE = 0.3


def get_llm():
    """Get configured LLM backend. Reads env vars at call time, not import time."""
    backend = os.environ.get("LLM_BACKEND", "mlx")
    if backend == "cuda":
        from src.llm import CUDABackend
        model = os.environ.get("CUDA_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        use_4bit = os.environ.get("LOAD_IN_4BIT", "0") == "1"
        return CUDABackend(model, load_in_4bit=use_4bit)
    elif backend == "mlx":
        from src.llm import MLXBackend
        model = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
        return MLXBackend(model)
    else:
        from src.llm import AnthropicBackend
        model = os.environ.get("API_MODEL", "claude-sonnet-4-6")
        return AnthropicBackend(model)
