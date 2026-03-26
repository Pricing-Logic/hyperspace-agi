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
   - Contamination-free: +38 problems (+18.4%) on 206 held-out problems (zero training overlap)
   - Parse failures INCREASED (+1) — model formats worse but reasons better
   - Tier 3 multi-step: 21% → 51% (+30pp on unseen problems)
   - Reproduced across 2 seeds (+12.7% and +14.8% on full 237-problem benchmark)
3. **BUT: improvement is DOMAIN-SPECIFIC, not general reasoning transfer** (2026-03-26)
   - GSM8K (1319 grade-school word problems): BASE 79.5% → LORA 75.2% = **-4.3%**
   - LoRA learned GSM8K formatting (hash mode) but degraded accuracy
   - The capability shaping works within training distribution but doesn't transfer out
4. Cross-model training (3B traces → 7B) degrades performance (-20%)
5. The overfitting wall at round 3-4 is real and consistent across scales

## What We Need to Prove Next
1. The base model has latent reasoning capacity (pass@8 >> pass@1)
2. Rejection sampling + STaR retry can surface that capacity into training data
3. One clean LoRA on high-quality self-traces produces genuine pass@1 improvement

## Current Phase: Deciding Next Direction
- Domain-specific self-improvement CONFIRMED (+18.4% in-distribution)
- General transfer FAILED (-4.3% on GSM8K)
- The gap: improvement is narrow to training distribution, not broad reasoning

### Open questions for next phase:
1. Can STaR + rejection sampling achieve transfer to GSM8K?
2. Would training directly on GSM8K-style problems (instead of our generator) work?
3. Is the -4.3% GSM8K degradation because our traces teach the wrong problem-solving style for word problems?
4. Could mixing our traces with GSM8K train traces produce both in-distribution and transfer gains?

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
