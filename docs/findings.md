# Baseline Findings

**Model:** Qwen/Qwen3-1.7B-Base  
**Date:** 2026-03-20 / 2026-03-21

## GSM8K Baseline

- **74.60% pass@1** (984/1,319)
- Few-shot prompting; model outputs reasoning then `#### answer` format
- Much higher than our pre-run estimate of 40-50% — Qwen3 is a strong base

## MATH-500 Baseline

### Overall
- **pass@1: 31.6%** (158/500)
- **pass@8: 58.4%** (292/500) — temp=0.7, 8 samples per problem

### By Difficulty Level

| Level | pass@1 | pass@8 |
|-------|--------|--------|
| L1 (easiest) | 62.8% (27/43) | 81.4% (35/43) |
| L2 | 51.1% (46/90) | 80.0% (72/90) |
| L3 | 35.2% (37/105) | 75.2% (79/105) |
| L4 | 28.1% (36/128) | 50.0% (64/128) |
| L5 (hardest) | 9.0% (12/134) | 31.3% (42/134) |

### By Subject (pass@1)

| Subject | pass@1 | n |
|---------|--------|---|
| Prealgebra | 43.9% | 82 |
| Algebra | 41.1% | 124 |
| Number Theory | 40.3% | 62 |
| Counting & Probability | 31.6% | 38 |
| Geometry | 31.7% | 41 |
| Intermediate Algebra | 17.5% | 97 |
| Precalculus | 7.1% | 56 |

## Key Observations

**The pass@1 → pass@8 gap is the headline finding.**

31.6% → 58.4% means the model can solve ~58% of problems *given enough tries*, but only gets it right first attempt 31.6% of the time. This gap is exactly what GRPO targets: the model has latent capability that RL can unlock by reinforcing correct reasoning paths.

**Weakest areas:** Precalculus (7.1%), Intermediate Algebra (17.5%), L5 problems (9.0%). These are hardest to improve — expect most GRPO gains in L2-L4 range where the model already has partial capability.

**Strongest areas:** Prealgebra, Algebra, Number Theory — already solid, less room to grow.

## Implications for Training

- GSM8K is likely near-saturated at 74.6% for this model size — SFT may not move it much
- MATH-500 L3-L4 problems are the sweet spot for GRPO signal
- If we want to demonstrate clear improvement, MATH-500 is the better benchmark to track
