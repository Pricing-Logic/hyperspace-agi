# Project 1: The Self-Improving Loop

**Hard problem: Self-improvement**
**Priority: SELECTED — Top 2 Pick**
**Hardware: Mac, 48GB RAM, Apple Silicon (MLX)**

## Concept

A training loop where a small local model generates problems, solves them, verifies answers programmatically, and fine-tunes itself on successful reasoning traces — continuously.

The core question: **does the loop converge to something smarter, or does it collapse?**

DeepSeek-R1 proved this works at massive scale. Nobody knows the minimum viable scale. That's what we're testing.

## Architecture

```
while true:
  1. Model generates a hard problem for itself (math, logic, code)
  2. Model attempts to solve it with chain-of-thought reasoning
  3. Verifier checks the answer PROGRAMMATICALLY (not another LLM)
  4. Score the reasoning trace
  5. Fine-tune on successful traces (LoRA via MLX)
  6. Measure: can it now solve problems it couldn't before?
  7. Log everything. Goto 1.
```

## Key Design Decisions

- **Model**: Qwen 2.5 3B or Gemma 3 1B (must fit alongside existing loaded model in 48GB)
- **Fine-tuning**: LoRA adapters via MLX (fast, memory-efficient on Apple Silicon)
- **Verification domain**: Start with math (sympy for symbolic verification, eval for numeric)
- **Difficulty curriculum**: Start easy, escalate only when success rate > 80%
- **Collapse detection**: Track diversity of generated problems + solution quality over time

## Success Metrics

- **Convergence**: Does val accuracy on held-out problems improve over generations?
- **Generalization**: Can it solve problem types it never generated for itself?
- **Collapse signal**: If all generated problems become trivially easy or identical, the loop has failed

## Why This Matters

If self-improvement works at 3B parameters on a laptop, that changes everything. If it doesn't, we learn where the floor is — which is equally valuable.

## Tech Stack

- MLX for inference + LoRA fine-tuning
- Python for orchestration
- SymPy for math verification
- SQLite for experiment logging
- JSON lines for training data accumulation
