# Qwen3-1.7B Math RLVR — POC Results

**Written by:** Claude Opus 4.8 · **Date:** 2026-05-30
**Status:** POC achieved (evidenced from artifacts); live-weights confirmation run in progress.
**Last updated:** 2026-05-30
**Pinned code:** tag `eval-pin-2026-05-30`

## Summary

GRPO (group-relative policy optimization, DAPO-style) was run on `Qwen/Qwen3-1.7B-Base`
with verifiable rewards (math-verify on MATH `\boxed{}` answers), few-shot prompting,
**trained from base — not from SFT**. The step-3000 checkpoint improves MATH-500 greedy
pass@1 from **35.8% → 44.2%** (+8.4pt absolute, ≈ +23% relative). This is the RLVR POC.

## Headline result (MATH-500, math-verify)

| Model | greedy pass@1 | pass@8 (temp 0.7) | p@8 − p@1 gap |
|-------|--------------:|------------------:|--------------:|
| `Qwen3-1.7B-Base` (baseline) | **35.8%** (179/500) | 65.0% | 29.2 |
| GRPO **step-3000** | **44.2%** (221/500) | 71.4% | 27.2 |
| Δ | **+8.4** | +6.4 | −2.0 |

Per-level (greedy), both show a clean difficulty gradient (sign of a genuinely capable
model, not a degenerate one):

| | L1 | L2 | L3 | L4 | L5 |
|--|---|---|---|---|---|
| Base | 0.65 | 0.54 | 0.41 | 0.34 | 0.12 |
| GRPO 3000 | 0.79 | 0.69 | 0.51 | 0.41 | 0.14 |

**Interpretation:** pass@1 rose more than pass@8 and the gap narrowed → textbook RLVR
"sharpening" (concentrating mass on solutions the base model could already reach via
sampling). pass@8 *also* rising +6.4 indicates modest genuine capability gain on top of
sharpening, not pure redistribution. Healthy.

## How these numbers were verified (artifact-level, code-independent)

Because the original runs were **not** tied to pinned code versions, the current scripts
cannot be trusted to reflect what produced past results. So the numbers above were
re-derived from durable artifacts, not by trusting old code:

1. **Re-scored the saved generations** in `outputs/grpo_step3000_math500_results.json` and
   `outputs/baseline_math500_mv_rescored.json` directly with `math-verify` → 221/500 and
   179/500 respectively (matches recorded values exactly).
2. **Weight identity:** `outputs/grpo_checkpoint/model.safetensors` sha256
   `3b3697bb…4015` is byte-identical to HF Hub `heyalexchoi/qwen3-1.7b-math-grpo`
   step-3000 (commit `63870ec`). The good weights are not lost.
3. GRPO init lineage confirmed as `Qwen/Qwen3-1.7B-Base` (`grpo_train.py`).

## The "all checkpoints collapsed" scare — resolved as an eval artifact

A 2026-04-13 binary-search re-eval reported every checkpoint (steps 2500–5000) at a
uniform ~9–11% and concluded "all collapsed, good model lost." That conclusion was wrong:

- Uniform ~11% across all steps is the fingerprint of every re-eval loading the **same**
  collapsed final checkpoint (`main` = step 7496) instead of the step it claimed — the
  documented `--checkpoint_step` without `--revision` footgun (`PLAN.md:351`).
- The 44.2% step-3000 generations are coherent and show a difficulty gradient; collapsed
  models give flat, uniformly-low scores.

**Real (non-artifact) finding:** training genuinely *did* destabilize late — the model
collapses into repetition by ~step 7496. The peak is ~step 3000. Lesson: run greedy eval
in the training loop + early-stop / KL control to catch the peak.

## SFT branch (separate; not part of this POC)

SFT (`heyalexchoi/qwen3-1.7b-math-sft`) trained cleanly (train loss 0.48, token acc 84%).
Its eval was plagued by config bugs (few-shot prompt vs chat-template mismatch → repetition
loops); the "~0% degenerate" note was an eval artifact — logs after the fix show the SFT
model producing correct solutions. **No trustworthy SFT MATH-500 number was ever cleanly
captured.** Irrelevant to the RLVR POC because GRPO trained from base, not from SFT.
(An SFT eval is included in the confirmation run to finally get a clean number.)

## Confirmation run (in progress)

Goal: prove the on-disk/HF step-3000 weights *regenerate* ~44% live under pinned code, and
pin the eval bug. Matrix on one A40 pod (`eval-pin-2026-05-30`):

| # | weights | backend | revision | expected | result |
|---|---------|---------|----------|----------|--------|
| 1 | GRPO 3000 | HF generate | `63870ec` | ~44% | _pending_ |
| 2 | GRPO 3000 | vLLM | `63870ec` | 44% → revision-bug was cause; 11% → vLLM backend is cause | _pending_ |
| 3 | GRPO final | vLLM | `main` (7496) | ~11% (confirms binary-search loaded `main`) | _pending_ |
| 4 | Base | HF generate | — | ~35.8% | _pending_ |
| 5 | SFT | HF generate | — | (first clean SFT number) | _pending_ |

## Guardrails adopted (post-mortem)

1. **Code↔run versioning** — every eval result now stamps git commit/describe/dirty,
   command, loaded-weights sha256, UTC (`math500_eval.py` provenance block). Tag repo per run.
2. **Append-only run ledger** — `RUNS.jsonl`, one immutable row per run.
3. **Eval canaries** — assert loaded-weights sha/revision == intended; difficulty-gradient
   sanity check; anchor-checkpoint regression check. _(planned)_
4. **Checkpoint durability** — rsync + verify sha before any teardown; never delegate teardown.
