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

## SFT branch — status corrected 2026-05-31: the final checkpoint is NOT degenerate

SFT (`heyalexchoi/qwen3-1.7b-math-sft`) trained cleanly (train loss 0.48, token acc 84%).

**This section supersedes all prior SFT conclusions** (the original "~0% degenerate / capacity
wall," and the 2026-05-30 close-out's "confirmed coherent"). Both were unreliable. The current
status rests on **48 directly-observed per-sample results** in the 2026-04-10 eval logs — a much
stronger basis than the single live sample or the secondary degeneration table behind the earlier
flips. Reliability ladder, weakest → strongest: README summary table → one live sample → 48 logged
samples (this).

**Finding.** The **final** SFT checkpoint (`outputs/sft_checkpoint`), run the way a Qwen3 *thinking*
model is meant to be run — chat template, **sampling temp=0.6 / top_p=0.95 / top_k=20**, 8192-token
budget — solves MATH problems normally. In the fair run that completed before the pod died
(`logs/03a_sft_eval_20260410_153625.log`, 6 problems, 48 samples): **26/48 samples correct**, the
correct ones terminating coherently in ~1300–2600 tokens with a valid `\boxed{}`. A degenerate model
cannot do that.

**Why it was mislabeled "~0% degenerate."** Two compounding causes, attributed precisely:
- **Greedy decoding (the dominant cause for the final checkpoint).** The README *itself* warns Qwen3
  thinking mode must never use greedy (`temperature=0`) — the `<think>` block enters repetition loops
  and fills the whole budget with no `\boxed{}`. The "hits max_new_tokens 100%" signature in the
  degeneration table is exactly that greedy thinking-loop. Run with sampling, the same checkpoint works.
- **An early few-shot-prompt format bug** (a *different*, earlier eval) — fixed 2026-04-10 when
  `sft_eval.py` switched to `apply_chat_template`.
- **Caveat (do not over-attribute to greedy):** the README's 264-sample / 33-problem row was the
  **checkpoint-500 (1-epoch intermediate)** under *sampling*, and it did degenerate (char/digit
  repetition). That one looks like a genuinely **undertrained intermediate checkpoint**, not a greedy
  artifact. The claim is narrow: the **final** checkpoint works under sampling.

**What we still do NOT have: a headline SFT number.** Six problems is a refutation of "degenerate,"
not a score. Do **not** cite the 6-problem rate as the SFT result. The full-500 fair eval is genuinely
pending (the 2026-04-10 eval pods kept dying mid-run; it was never completed).

**Comparison-axis rule (critical).** The SFT model is a thinking model scored by sampling→`c/n`. When
the full SFT number lands, compare it **`c/n`-to-`c/n`**: base **24.55%** vs GRPO step-3000 **36.83%**
vs SFT-TBD. Do **NOT** compare SFT's `c/n` against the 43.8% *greedy non-thinking* headline — that is
the same two-format-universes error documented elsewhere in this writeup.

**Implication for the project narrative.** The GRPO-from-base POC headline (35.8 → 43.8 greedy)
**stands, untouched.** But the sub-narrative "SFT failed, so we pivoted to zero-RL" was a
**misdiagnosis from greedy-decoding a thinking model.** SFT was not broken. That means **SFT→GRPO
(the R1 recipe) was wrongly abandoned** — and it is the strongest lever for beating 43.8% (see
"SFT vs RLVR sequencing"). Still irrelevant to the *existing* POC headline, which trained from base.

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

It is conditional on a **non-degenerate** SFT — and as of the 2026-05-31 correction, **we have one.**
The final SFT checkpoint works under sampling (see "SFT branch" above). So SFT→GRPO is not blocked;
it was **wrongly abandoned** on a greedy-decoding misdiagnosis. (The 2026-04 pivot to GRPO-from-base
was still a *reasonable* call given the bad eval at the time, and it produced the valid POC — but the
premise "SFT is a dead end" was false.)

**Decisive next step:** the full-500 fair SFT eval (sampling→`c/n`), compared **`c/n`-to-`c/n`**:
- base **24.55%** vs GRPO step-3000 **36.83%** vs SFT-TBD.
- If SFT clears base (very likely, given the partial run) → SFT→GRPO is well-motivated; run it with
  small KL (`beta`≈0.001–0.01) + greedy-eval-in-loop + early-stop (the fixes for the late collapse).

**On "new capability" (a sharp point Alex raised).** SFT-on-the-model's-**own**-correct-rollouts
(RFT/STaR/ReST) and GRPO are the *same family* — both bootstrap from the model's own correct samples
and ceiling at ~pass@k; neither injects knowledge from a stronger model. So self-distillation will
**not** meaningfully beat the 43.8% GRPO result. **New capability comes only from the stronger teacher**
— i.e., from making the **32B-trace SFT** work. That reframes self-distillation as a *format/stability*
tool at most, not a capability lever. The real lever for beating 43.8% is the teacher-distillation
SFT (now known to work) → GRPO.

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
| 5 | SFT (final ckpt) | HF/vLLM (chat template) | — | (full-500 number) | ⚠️ FULL NUMBER PENDING, but NOT degenerate. Fair partial run (chat template, **sampling temp=0.6**, 8192 tok) = 26/48 correct over 6 problems before the pod died. The earlier "~0%" was **greedy** decoding (banned for Qwen3 thinking mode). Full-500 fair eval (sampling→c/n) still needed → high-value. |

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
- ✅ SFT corrected 2026-05-31: the **final** checkpoint is NOT degenerate — fair sampling run got 26/48 correct over 6 problems; the "~0%" was a greedy-decoding artifact (banned for Qwen3 thinking mode). Full-500 fair number (sampling→c/n) still pending and is now high-value. "SFT failed → pivoted to zero-RL" was a misdiagnosis; SFT→GRPO was wrongly abandoned.
- Pinned eval code: tag `eval-pin-2026-05-30`. Provenance stamped by `math500_eval.py`; ledger `RUNS.jsonl`.
- **Reproduce headline:** rent A40, `pip install torch==2.6.0 transformers==5.5.3 datasets math-verify`,
  then `python greedy500_confirm.py` (or `math500_eval.py --model heyalexchoi/qwen3-1.7b-math-grpo --revision 63870ec239b2 --checkpoint_step 3000`). Env note: vLLM 0.22 needs a newer CUDA driver — use HF backend.

### Next experiments (v2)
1. Fair SFT eval (sampling@0.6, ~8192 tokens).
2. SFT→GRPO with small KL (`beta`≈0.001–0.01) + greedy-eval-in-loop + early-stop / keep-best.
