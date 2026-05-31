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

## SFT branch (separate; not part of this POC) — status: UNRESOLVED

SFT (`heyalexchoi/qwen3-1.7b-math-sft`) trained cleanly (train loss 0.48, token acc 84%).

**Correction (2026-05-31).** An earlier version of this section claimed the SFT "~0% degenerate"
finding was an eval-format artifact (few-shot prompt vs chat template) and that the model was
"confirmed coherent / NOT degenerate." That claim was overstated and is retracted:

- The 2026-04-10 SFT eval logs (`logs/sft_eval_20260410_*.log`) show the prompt was the **correct
  chat template** (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`). So the documented
  degeneration (~0% usable across 264 samples; runs to the 8192-token limit; repetition loops —
  see README "SFT checkpoints: degenerate") was observed **with the right format**. It is *not*
  explained away as a format bug.
- The 2026-05-30 "confirmed coherent" check was a **single** live sample that showed a coherent
  `<think>` *opening* before I aborted the run (>1hr ETA). There is no evidence it ever reached a
  valid `\boxed{}`. The README's own description is "both checkpoints open `<think>` and begin
  coherent reasoning *before degenerating*" — so that one sample **confirms** the degeneration
  pattern, it does not refute it.

**Honest status:** the most likely root cause (per the README capacity analysis) is that a 1.7B
cannot sustain 32B-length thinking traces (teacher median 3.3k / p90 10k tokens) → it leaves its
learned distribution and collapses into repetition. A fair eval (sampling@temp0.6 + ~8192 tokens)
*might* still recover some accuracy, but the prior multi-sample evidence points to a genuine
problem, not merely an artifact. **One fair SFT eval decides this — and decides the whole v2
design** (see "SFT vs RLVR sequencing" below). Irrelevant to the *RLVR POC headline* either way,
because GRPO trained from base, not from SFT.

## Reference points — where could a 1.7B get on MATH? (added 2026-05-31)

Useful to know the ceiling. Caveats matter a lot here:

| Source | Benchmark | Score | Comparable to our 35.8% / 43.8%? |
|--------|-----------|------:|----------------------------------|
| Qwen3 Tech Report, Table 8 — **Qwen3-1.7B-Base** | MATH (full, ~5000, 4-shot, Qwen harness) | **43.5** | **No.** Different test set (full MATH vs the 500 subset), different answer extraction/harness. Our base = 35.8% greedy on MATH-500 + math-verify + few-shot. The 43.5-vs-43.8 near-match is a **coincidence of two different metrics** — do *not* read it as "GRPO reached the official base number." |
| Qwen3-1.7B (post-trained, **non-thinking**) | MATH-500 | not published per-model in the report | would be the *fair* instruct ceiling for our setup — must run it |
| Qwen3-1.7B (post-trained, **thinking**) | MATH-500 | not published per-model; thinking small models score much higher | **No.** Thinking mode = long CoT, sampling@0.6, big token budget. Apples-to-oranges vs our non-thinking concise greedy@2048. |

**The only rigorous reference point is one we'd have to measure:** run `Qwen/Qwen3-1.7B` (the instruct model) on MATH-500 **in its native chat format** — once non-thinking, once thinking — through our math-verify scorer. Forcing an instruct model through our base few-shot harness would understate it badly, so this needs `sft_eval.py`-style chat prompting, not `math500_eval.py`. ~30–60 min on one A40. **Proposed, not yet run.** That gives the honest "how far is 43.8% from the post-trained ceiling" number.

(Sources: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), Table 8; [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B); [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/).)

## SFT vs RLVR sequencing (v2 design) — added 2026-05-31

The standard "best" recipe is **SFT cold-start → RL** (DeepSeek-R1 style): SFT instills a reasoning
format and raises the starting policy, then RL (GRPO) sharpens it with verifiable rewards. So
"SFT first, then RLVR" is right *in principle*.

**But it is strictly conditional on a non-degenerate SFT**, and that is exactly what we do not have:

- A degenerate SFT (~0% usable, what we observed) is a **dead end for GRPO** — GRPO needs nonzero
  rollout accuracy to produce a reward signal. That is precisely why the prior pivot to
  **GRPO-from-base was the correct call** at the time (base already scores 24.55% inferred / 35.8%
  greedy — plenty of signal).
- So the open question is not "SFT or not" in the abstract; it's **"can we produce a stable,
  length-appropriate SFT for a 1.7B?"** The prior SFT failed because it imitated 32B-length traces.

**Cheapest decisive next step:** the fair SFT eval. One number settles the branch:
- If fair-eval SFT **> base** and stable → SFT→GRPO is well-motivated; do it (with small KL +
  early-stop, per the collapse diagnosis).
- If fair-eval SFT **≤ base** or still degenerate → SFT-first is not worth it for this 1.7B;
  GRPO-from-base stands as the POC and the lift comes from better RL, not cold-start.

**Better SFT data idea — self-distillation.** The README already (correctly) rejected
"short-traces-only" (biases toward easy problems, low ceiling). A cleaner fix that sidesteps the
length problem entirely: SFT on the model's **own** correct rollouts (base or GRPO-step-3000),
optionally lightly cleaned. These are *by construction* within the 1.7B's length/style capacity,
so they don't trigger the 32B-length collapse — while still teaching a consistent
reason-then-`\boxed{}` format. This is the recommended SFT-data path for v2 if we pursue
SFT→GRPO.

**Format choice for v2 (chat vs completion).** Keep the **completion / few-shot, no-chat-template**
format for the math POC line: it's native to the base model, it's what GRPO already works in, it
evals cleanly at greedy, and it avoids the chat-template contamination footgun that caused real
pain. You do **not** need chat format (or `<think>`) to teach reasoning — consistent format +
length-appropriate CoT is enough. Adopt chat format later, **as the deliberate subject of the
agentic phase**, not as a complication bolted onto the math POC. (A hidden lesson from the mess:
the pipeline mixed two format universes — base/GRPO in completion format, SFT in chat+thinking
format — which is part of why comparisons and evals broke. Hold ONE format constant end-to-end.)

## Confirmation run (in progress)

Goal: prove the on-disk/HF step-3000 weights *regenerate* ~44% live under pinned code, and
pin the eval bug. Matrix on one A40 pod (`eval-pin-2026-05-30`):

| # | weights | backend | revision | expected | result |
|---|---------|---------|----------|----------|--------|
| 1 | GRPO 3000 | HF generate | `63870ec` | ~44% | ✅ **full-500 greedy = 219/500 = 43.8%** (live, reproduces archived 44.2% within 2 problems). `outputs/grpo3000_greedy500_confirm.json` |
| 2 | GRPO 3000 | vLLM | `63870ec` | distinguish bug | ⚠️ ABANDONED — vLLM 0.22 install bumped torch→2.11/cu130, EngineCore init fails (CUDA-driver mismatch, environmental). Not weights. |
| 3 | GRPO final | vLLM | `main` (7496) | ~11% | ⚠️ same env break; not run |
| 4 | Base | HF generate | — | ~35.8% | _pending_ |
| 5 | SFT | HF generate (chat template) | — | (first clean SFT number) | ⚠️ DEFERRED & UNRESOLVED. The 2026-04-10 eval used the **correct** chat template and still degenerated (~0%, repetition). The 2026-05-30 "coherent" note was a single sample (coherent opening, no confirmed `\boxed{}`) and is retracted as evidence. Fair eval (sampling@0.6 + ~8192 tokens) still needed → v2. |

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
- ⚠️ SFT status UNRESOLVED (prior "coherent / not degenerate" note retracted 2026-05-31 — degeneration was seen with the *correct* chat-template format; likely a real capacity/length problem). Fair SFT eval (sampling@0.6 + ~8192 tokens) deferred to v2 and is the gating experiment for v2 design.
- Pinned eval code: tag `eval-pin-2026-05-30`. Provenance stamped by `math500_eval.py`; ledger `RUNS.jsonl`.
- **Reproduce headline:** rent A40, `pip install torch==2.6.0 transformers==5.5.3 datasets math-verify`,
  then `python greedy500_confirm.py` (or `math500_eval.py --model heyalexchoi/qwen3-1.7b-math-grpo --revision 63870ec239b2 --checkpoint_step 3000`). Env note: vLLM 0.22 needs a newer CUDA driver — use HF backend.

### Next experiments (v2)
1. Fair SFT eval (sampling@0.6, ~8192 tokens).
2. SFT→GRPO with small KL (`beta`≈0.001–0.01) + greedy-eval-in-loop + early-stop / keep-best.
