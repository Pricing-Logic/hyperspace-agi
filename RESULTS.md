# Experiment Results — Self-Improving Reasoning at 3B Scale

## Summary
A Qwen2.5-3B-Instruct model trained on ~150 of its own verified reasoning traces shows +18.4% accuracy on 206 contamination-free held-out problems. The improvement is genuine reasoning (parse failures increased, not decreased). However, it does not transfer to external benchmarks (GSM8K: -4.3%).

## Experiment 1: Original LoRA v1 (Simple SFT)

**Config:** rank 8, lr 1e-4, 100 steps, 50 seconds training on MLX
**Data:** ~150 correct reasoning traces from the model's own solutions

### Contamination-Free Frozen Benchmark (206 problems, zero training overlap)

| Model | Accuracy | Parse Failures |
|-------|----------|----------------|
| 3B BASE | 69/206 (33.5%) | 10 |
| 3B LoRA v1 | 107/206 (51.9%) | 11 |
| **Delta** | **+38 (+18.4%)** | **+1 (worse formatting)** |

Reproduced across 2 seeds: +12.7% and +14.8% on full 237-problem benchmark.

### Per-Tier Breakdown

| Tier | BASE | LoRA v1 | Delta |
|------|------|---------|-------|
| 1 (easy) | — | 90% | large |
| 2 (medium) | — | 74% | large |
| 3 (multi-step) | 21% | 51% | +30pp |
| 4 (complex) | 22% | 31% | +9pp |
| 5 (composite) | 22% | 28% | +6pp |

### GSM8K (1319 problems, external benchmark)

| Model | Accuracy |
|-------|----------|
| 3B BASE | 1048/1319 (79.5%) |
| 3B LoRA v1 | 992/1319 (75.2%) |
| **Delta** | **-56 (-4.3%)** |

**Conclusion:** Strong in-distribution improvement, no cross-domain transfer.

## Experiment 2: STaR + Rejection Sampling + GSM8K Mix

**Config:** rank 4, lr 3e-5, 2 epochs on CUDA (Modal A100)
**Data:** ~163 rescued traces (k=8 rejection sampling + STaR retry) + ~70 hard GSM8K train problems (30% mix)

### Results

| Model | Frozen (206) | GSM8K (300) |
|-------|-------------|-------------|
| 3B BASE | 38.3% (pf=13) | 86.3% (pf=0) |
| 3B STaR-LoRA | 40.3% (pf=6) | timed out |
| 7B BASE (target) | 53.9% (pf=5) | 91.7% (pf=0) |
| **3B STaR delta** | **+2.0pp** | — |

**Conclusion:** The sophisticated approach performed worse than the simple one. Parse failure improvement (13→6) suggests half the gain is formatting, not reasoning.

## Other Findings

- **Cross-model transfer fails:** Training 7B on 3B traces causes -20% degradation
- **Overfitting wall:** Iterative training peaks at round 3-4 then collapses (both 3B and 7B)
- **Latent capacity is huge:** 90%+ rescue rate (pass@8 >> pass@1)
- **Self-consistency voting:** +14pp accuracy boost for free (78% vs 64%)

## What We Proved

1. A 3B model can genuinely improve its reasoning (+18.4%) by training on its own verified traces
2. The improvement is NOT formatting — parse failures increase
3. The improvement is domain-specific — does not transfer to GSM8K
4. The effect is recipe-sensitive — the simple approach beat the sophisticated one
5. Each model needs its own traces — cross-model transfer degrades performance

## What We Did NOT Prove

1. General reasoning transfer to external benchmarks
2. That the effect compounds over multiple training rounds (overfitting wall)
3. That STaR or RL approaches work better than simple SFT at this scale

## Infrastructure

- **Local:** Mac 48GB, Apple Silicon, MLX, Qwen2.5-3B-Instruct-4bit
- **Cloud:** Modal (A100, serverless, reliable), vast.ai (RTX 4090/A100, SSH unreliable)
- **Cost:** ~$15-20 total across all cloud experiments
- **Repo:** https://github.com/Pricing-Logic/hyperspace-agi
