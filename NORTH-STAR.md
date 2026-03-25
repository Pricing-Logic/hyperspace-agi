# North Star: Self-Improving Reasoning via Verified Training

## Mission
Prove that a small language model (3B-7B) can measurably improve its own mathematical reasoning ability through a verified self-improvement loop — not just formatting compliance.

## Success Criteria
A frozen benchmark evaluation showing:
- pass@1 improvement of >= 5 absolute points after one round of self-training
- With NO decrease in parse failure rate (ruling out formatting-only improvement)
- Reproducible across 2+ random seeds

## What We've Proven So Far
1. The pipeline works end-to-end (local MLX + cloud CUDA via Modal)
2. **CONFIRMED: 3B self-improvement is REAL REASONING, not formatting** (2026-03-25)
   - LoRA v1: +30 problems (+12.7%) on 237 frozen benchmark problems
   - Parse failures INCREASED (+5) — model formats worse but reasons better
   - Tier 1: +28%, Tier 2: +34%, Tier 3: +12% (genuine multi-step gains)
3. Cross-model training (3B traces → 7B) degrades performance
4. The overfitting wall at round 3-4 is real and consistent across scales

## What We Need to Prove Next
1. The base model has latent reasoning capacity (pass@8 >> pass@1)
2. Rejection sampling + STaR retry can surface that capacity into training data
3. One clean LoRA on high-quality self-traces produces genuine pass@1 improvement

## Current Phase: Step 3 — STaR + Rejection Sampling
- Self-improvement CONFIRMED at 3B (+12.7% on frozen benchmark, reasoning not formatting)
- Skipped pass@k diagnostic per Codex consensus (wrong gate for this question)
- Next: STaR (retry wrong answers with hints) + rejection sampling (k=8 per problem)
- Goal: push 3B further, especially on Tier 3-5 where gains were smaller

## Architecture Principles
- All evaluation on FROZEN benchmark (experiments/frozen_benchmark_v1.jsonl)
- Track parse_failure_rate separately from correctness
- Never train on benchmark problems — train on fresh generated problems only
- Use verify_answer() (string comparison), not verify_with_function() (closure)
- Clean data pipeline: dedup → split → weight (never weight before split)

## Tech Stack
- Local: Mac 48GB, MLX, Qwen2.5-3B-Instruct-4bit
- Cloud: Modal A100, PyTorch, Qwen2.5-7B-Instruct (4-bit)
- Repo: https://github.com/Pricing-Logic/hyperspace-agi

## Key Files
- `experiments/frozen_benchmark_v1.jsonl` — THE benchmark (never modify)
- `scripts/freeze_benchmark.py` — benchmark generator
- `scripts/ab_test.py` — frozen benchmark A/B test with parse failure tracking
- `scripts/passk_eval.py` — pass@k diagnostic (Step 2, to be built)
- `scripts/star_rejection.py` — STaR + rejection sampling (Step 3, to be built)
- `scripts/modal_train.py` — Modal deployment wrapper
- `src/loop/verifier.py` — answer verification (string-based, no closures)
- `src/loop/generator.py` — problem generation (5 domains x 5 tiers)
- `src/llm/cuda_backend.py` — CUDA inference with 4-bit quantization
