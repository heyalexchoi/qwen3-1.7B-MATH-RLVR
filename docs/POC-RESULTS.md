# Qwen3-1.7B Math RLVR — POC Results

**Written by:** Claude Opus 4.8 · **Date:** 2026-05-30
**Status:** ✅ POC CONFIRMED — full-500 greedy reproduced **live** at 43.8% (vs 44.2% archived, within 2 problems), HF backend, pinned code, provenance-stamped.
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
loops); the "~0% degenerate" note was an eval artifact. **Confirmed live 2026-05-30:** the SFT
model produces coherent `<think>` chain-of-thought (verified on the polar-coordinates problem) —
NOT degenerate. It is a *thinking* model with long CoT, so a greedy@2048-token eval truncates it
before `\boxed{}` (unfairly low). **A fair SFT number still needs sampling@temp0.6 + ~8192 tokens**
— deferred to a deliberate v2 experiment. Irrelevant to the RLVR POC because GRPO trained from
base, not from SFT. (Contrast: the GRPO model uses concise few-shot, non-thinking output, which is
why it evals cleanly at greedy@2048.)

## Confirmation run (in progress)

Goal: prove the on-disk/HF step-3000 weights *regenerate* ~44% live under pinned code, and
pin the eval bug. Matrix on one A40 pod (`eval-pin-2026-05-30`):

| # | weights | backend | revision | expected | result |
|---|---------|---------|----------|----------|--------|
| 1 | GRPO 3000 | HF generate | `63870ec` | ~44% | ✅ **full-500 greedy = 219/500 = 43.8%** (live, reproduces archived 44.2% within 2 problems). `outputs/grpo3000_greedy500_confirm.json` |
| 2 | GRPO 3000 | vLLM | `63870ec` | distinguish bug | ⚠️ ABANDONED — vLLM 0.22 install bumped torch→2.11/cu130, EngineCore init fails (CUDA-driver mismatch, environmental). Not weights. |
| 3 | GRPO final | vLLM | `main` (7496) | ~11% | ⚠️ same env break; not run |
| 4 | Base | HF generate | — | ~35.8% | _pending_ |
| 5 | SFT | HF generate (chat template) | — | (first clean SFT number) | ⚠️ DEFERRED. Live check confirms SFT is **coherent** (correct `<think>` CoT), NOT degenerate. But it's a *thinking* model: greedy@2048 truncates before `\boxed{}` (unfair). Fair eval needs sampling@temp0.6 + ~8192 tokens → do as v2 experiment. |

**Conclusion already reached (discriminator + artifact re-score + SHA identity all agree):** the
step-3000 weights are good and the 44% is real and reproduces live. The vLLM repro was only to
distinguish *which* eval bug caused the 11%; the uniform-~11%-across-all-steps fingerprint already
points to revision-pinning (every binary-search eval loaded `main`/step-7496). Not worth chasing
the vLLM env break.

## Guardrails adopted (post-mortem)

1. **Code↔run versioning** — every eval result now stamps git commit/describe/dirty,
   command, loaded-weights sha256, UTC (`math500_eval.py` provenance block). Tag repo per run.
2. **Append-only run ledger** — `RUNS.jsonl`, one immutable row per run.
3. **Eval canaries** — assert loaded-weights sha/revision == intended; difficulty-gradient
   sanity check; anchor-checkpoint regression check. _(planned)_
4. **Checkpoint durability** — rsync + verify sha before any teardown; never delegate teardown.

## Why GRPO collapsed *late* (the one real degeneration) — hyperparameter diagnosis

The step-7496 collapse is real (not an eval bug). `configs/grpo_config.yaml` shows an
"improve-then-collapse" recipe:

| Setting | Value | Implication |
|---|---|---|
| `beta` (KL) | **0.0** | No anchor to base policy → drifts into degenerate high-reward modes over many steps. Biggest risk. |
| `loss_type` / `epsilon_high` | `dapo` / 0.28 | Asymmetric clipping pushes *further* from base → amplifies drift. |
| `lr_scheduler_type` | `constant_with_warmup` | No decay → keeps drifting at full LR late. |
| `temperature` (rollouts) | 0.9 | High temp makes *training reward* look healthy while *greedy* degenerates — masks collapse. |
| epochs / early-stop | 1 epoch ≈ 7496 steps, **none** | Trained straight past the ~step-3000 peak. |
| `save_total_limit` | **1** | Only latest local ckpt kept — why the good step-3000 nearly got lost. |

**Fix for v2:** add small KL (`beta` ≈ 0.001–0.01) OR greedy-eval-in-loop + early-stop +
keep-best; consider cosine LR decay / repetition penalty. Peak was ~3000.

## Recommended next experiments

1. Properly eval the existing SFT checkpoint (`heyalexchoi/qwen3-1.7b-math-sft`) with
   `sft_eval.py` (chat template) → first trustworthy SFT number.
2. **SFT → GRPO** (cold-start then RL) v2 with small KL + early-stopping. Strongest open
   recipe (R1-style); likely beats the 44.2% zero-RL result without collapsing.

## Session close-out (2026-05-30) — DONE

- ✅ Full-500 greedy reproduced live = 43.8% (`outputs/grpo3000_greedy500_confirm.json`, pulled & durable).
- ✅ Pod `p44nx27xkitomr` **torn down** (no active pods).
- ✅ Docs cleaned: README/PLAN banners + corrected tables; `eval-discrepancy-investigation.md` RESOLVED.
- ✅ SFT confirmed coherent (not degenerate); fair SFT eval deferred to v2 (needs sampling + ~8192 tokens).
- Pinned eval code: tag `eval-pin-2026-05-30`. Provenance stamped by `math500_eval.py`; ledger `RUNS.jsonl`.
- **Reproduce headline:** rent A40, `pip install torch==2.6.0 transformers==5.5.3 datasets math-verify`,
  then `python greedy500_confirm.py` (or `math500_eval.py --model heyalexchoi/qwen3-1.7b-math-grpo --revision 63870ec239b2 --checkpoint_step 3000`). Env note: vLLM 0.22 needs a newer CUDA driver — use HF backend.

### Next experiments (v2)
1. Fair SFT eval (sampling@0.6, ~8192 tokens).
2. SFT→GRPO with small KL (`beta`≈0.001–0.01) + greedy-eval-in-loop + early-stop / keep-best.
