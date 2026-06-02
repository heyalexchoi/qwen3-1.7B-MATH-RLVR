# Qwen3-1.7B Math RLVR — POC Results

**Written by:** Claude Opus 4.8 · **Date:** 2026-05-30 · **Last updated:** 2026-06-01
**Status:** ✅ GRPO-from-base POC CONFIRMED — full-500 greedy reproduced **live** at 43.8% (vs 44.2% archived), HF backend, pinned code. _(This headline is unaffected by the SFT findings.)_
**2026-06-01 update:** SFT branch **re-corrected** — it's an undertrained thinking model (clean-solve / intermittent-collapse), prior "not degenerate / 26·48 clean" was a mischaracterization; the 93.4%@32k thinking ceiling verified; recommended path is **better SFT data (capacity-matched distillation)**, not GRPO-on-top. See "SFT branch" and "Recommended path (v2)".
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

## SFT branch — re-corrected 2026-06-01: undertrained thinking model (clean-solve / intermittent-collapse)

> **→ For the low-interpretation, evidence-tiered version of everything below (measured vs inferred
> vs unknowable, plus the checked/not-checked process ledger), see [`docs/sft-evidence-ledger.md`](sft-evidence-ledger.md).**
> Two findings added there 2026-06-01 from a local-only artifact dig: **(i)** on the April log, a
> sample is correct **iff it terminates within budget — 47/48 samples** (the lone exception, 1349, is
> a figure-parse + name-answer problem, a *different* failure); **(ii)** the student trains at
> `max_seq_length=32768` but evals at `max_new_tokens=8192`, and **14.8% of the correct traces it
> learned (31.4% at L5) are longer than 8192** — so a chunk of problems can't fit the eval budget by
> construction. The single highest-value next eval is a **budget sweep (8k→16k→32k)** to test whether
> that recovers the within-budget failures.

> **Correction history (read this — the flip-flops are the cautionary tale).**
> 1. *2026-04:* "SFT ~0% degenerate / capacity wall" → pivoted to GRPO-from-base.
> 2. *2026-05-30:* "confirmed coherent."
> 3. *2026-05-31:* "NOT degenerate — 26/48, terminates cleanly at 1300–2600 tok."
> 4. *2026-06-01 (this):* **all three were partly wrong.** The root cause of the churn is the
>    project's recurring wound — **the eval/inference environment was never pinned**, so each
>    re-eval ran on a different stack and "the model" was never the only thing changing.
> The honest, evidence-based status is below. Authored by Claude Opus 4.8, 2026-06-01.

SFT (`heyalexchoi/qwen3-1.7b-math-sft`, final = 3-epoch / step-1275, sha `9822…886af`) trained
cleanly: train loss 0.48, and a real held-out split (`train_test_split(test_size=0.05, seed=42)` on
the 7k traces) shows eval loss **monotone-decreasing, no overfit, plateauing by ~epoch 2**
(step 800: 0.4801 → step 1275: 0.4794). So it is **not over-trained**; more *epochs* on this data
would not help. Teacher-forced eval loss looks great — but loss ≠ free-generation quality, which is
the whole point below.

**Finding (from the actual 48 per-sample April logs, read in full — not the convenient lines).**
The checkpoint is a **thinking model** (trained on `<think>…</think>\n{solution}` 32B traces). Under
its intended config (chat template, sampling temp=0.6/top_p0.95/top_k20, 8192-tok budget,
**HF backend**, `logs/03a_sft_eval_20260410_153625.log`, 6 problems × 8 = 48 samples) it **solves
some problems cleanly and fully collapses on others** — on the failures it opens `<think>`, loses
coherence, and runs to the 8192 cap in `111…` repetition without ever closing `</think>`:

| Problem | level | correct/8 | fully-degenerate (8192-tok) samples |
|---|---|---|---|
| number_theory/572 | 3 | **8/8** | 0 |
| precalculus/807 | 2 | 7/8 | 1 |
| algebra/2584 | 3 | 6/8 | 2 |
| prealgebra/1622 | 2 | 5/8 | 3 |
| intermediate_algebra/1994 | 5 | **0/8** | 8 (all) |
| algebra/1349 | **2** | **0/8** | 6 |

26/48 is the right *count*, but "terminates cleanly at 1300–2600 tok / not degenerate" was a
**mischaracterization** — it described only the clean problems and ignored the two that fully
collapsed. **The failure is NOT cleanly difficulty-correlated** (evidenced): `algebra/1349` is
**level 2** yet collapsed 0/8, while `algebra/2584` (level 3) got 6/8 and the level-5 problem also
collapsed. What *is* evidenced is the mechanism: **the solved samples terminate within budget
(~1300–2600 tok) and the failures fail to terminate (peg 8192 in repetition).** So the precise claim
is **"intermittent failure-to-terminate-within-budget"**, consistent with a small thinking model
under-resourced for long/variable CoT (the original README capacity/length diagnosis — vindicated —
without asserting a clean difficulty law). These 6 are dataset order; do **not** extrapolate a
full-500 number in either direction — it is genuinely **unknown**.

**2026-06-01 reproduction attempt — and why there is still no trustworthy full-500.** Re-running the
*same sha-verified bytes* on a freshly assembled stack (A40, vLLM 0.8.5 + torch 2.6.0 +
transformers 4.51.3 — closest Qwen3-capable combo to the April env) **over-degenerates** and cannot
reproduce April. Crucially, the **stack itself is fine** — the *official* `Qwen/Qwen3-1.7B` on the
exact same vLLM solved 572 cleanly 4/4 (closes `</think>`, `\boxed{9}` correct, 1483–2218 tok). So the
degeneration is **specific to our checkpoint**, not a broken engine:

| Problem | April-HF (our ckpt) | 2026-06-01 vLLM (our ckpt) | 2026-06-01 vLLM (official 1.7B) |
|---|---|---|---|
| number_theory/572 | **8/8 (clean)** | **0/8 (all peg)** | **4/4 (clean, correct)** |
| precalculus/807 | 7/8 | 0/8 | — |
| algebra/2584 | 6/8 | 0/8 | — |
| intermediate_algebra/1994 | 0/8 | 0/8 | — |

What this means: our SFT checkpoint solves 572 8/8 under April's HF backend but 0/8 under this vLLM,
while a *robust* model is unaffected by the same vLLM → **our checkpoint is fragile**, and the cause is
NOT "vLLM is broken." Two non-exclusive explanations, neither fully isolated (and not worth more
archaeology): **(a)** the model is genuinely marginal/undertrained, so its output distribution is
sensitive to inference numerics (vLLM's sampler vs HF's tips a fragile model over — exactly what
"undertrained" predicts); **(b)** a **reconstruction confound** — the required `extra_special_tokens={}`
patch (transformers ≥4.51 crashes on the checkpoint's list form) differs from April's transformers
5.5.3, which had the special tokens intact; neutralizing the checkpoint's `generation_config.json`
(`do_sample:false`) did **not** fix it, ruling that sub-suspect out. **The eval prompt format was
verified to match training** (`apply_chat_template([user], add_generation_prompt=True)` is an exact
prefix of the rendered `[user, assistant]` sequence; the model also reasons *on-topic*, so it's not a
gross format break). **Bottom line: no trustworthy full-500 SFT c/n exists.** To get one, reproduce
April's working config (HF backend + transformers ~5.5.3) and gate on a 572≈8/8 canary before
believing any number. Process win: the **time-based degeneracy canary** (watch wall-clock
+ token-pegging rate, not problem count) caught this in minutes instead of waiting on a problem count
that a degenerate model would never reach.

**Comparison-axis rule (unchanged).** When a trustworthy SFT c/n lands, compare **`c/n`-to-`c/n`**:
base **24.55%** vs GRPO step-3000 **36.83%** vs SFT-TBD. Never against the 43.8% *greedy non-thinking*
headline.

**Implication for the narrative.** The GRPO-from-base POC headline (35.8 → 43.8 greedy) **stands,
untouched** — it never depended on SFT. But "SFT → GRPO is the easy next win" is **not** supported:
the current SFT cold-start is an undertrained, over-long-trace thinking model that intermittently
collapses (fails to terminate within budget) on a subset of problems, and GRPO samples rollouts the same way — on hard problems it would get near-zero reward and
no learning signal. The real lever is **better SFT data**, not GRPO-on-top-of-this (see "SFT vs RLVR
sequencing", rewritten below).

## Reference points — where can a 1.7B get on MATH-500? (verified 2026-06-01)

The ceiling is **much higher than our 43.8%**, and that reframes the whole effort. Verified against
the official Qwen3 Technical Report (web-research pass, 2026-06-01):

| Source | Model variant | Thinking | Benchmark | Score | Notes |
|--------|---------------|----------|-----------|------:|-------|
| Tech Report **Table 19** | Qwen3-1.7B (post-trained) | **ON** | MATH-500 | **93.4** | temp 0.6/top-p0.95/top-k20, **max output 32,768 tok**; MATH-500 sample/scoring unspecified |
| Tech Report **Table 20** | Qwen3-1.7B (post-trained) | OFF | MATH-500 | **73.0** | temp 0.7/top-p0.8/top-k20, presence-pen 1.5, 32,768 tok |
| Tech Report Table 8 | Qwen3-1.7B-**Base** | n/a | MATH (full ~5000) | 43.5 | base, full MATH not MATH-500 — *not* our ruler; the 43.5≈43.8 match is coincidence of two rulers |
| MathGPT blog (2025-06-03) | "Qwen3 1.7B" | "hybrid" | MATH-500 | 84.57 | **the origin of the "~85%" claim — NOT thinking mode.** Hybrid generation, lower than all-thinking; table attribution ambiguous (1.7B vs 4B). Disregard as a thinking-mode ceiling. |

**The defensible ceiling to target: ~93.4% (thinking) — but with a load-bearing caveat.** That number
was obtained at a **32,768-token** generation budget. Thinking traces routinely exceed 8K tokens; our
eval caps at **8192**, which truncates thinking and depresses the score. So (a) do not expect to
reproduce 93.4% at 8192 tokens, and (b) **this is directly relevant to our SFT collapse**: a *healthy*
1.7B thinking model uses up to 32k tokens productively, whereas our undertrained SFT collapses into
`111…` well before that. The gap from 43.8% (our GRPO-from-base) to 73–93% (official post-trained) is
the headroom; closing it is a **data/training-recipe** problem, not a tuning one.

(Sources: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) Tables 8/19/20 — HTML mirror
`arxiv.org/html/2505.09388v1`; [Qwen3-1.7B card](https://huggingface.co/Qwen/Qwen3-1.7B);
[MathGPT blog](https://www.mathgpt.ai/blog/2025/06/03/).)

## Recommended path (v2) — the lever is better SFT *data*, not GRPO-on-top — rewritten 2026-06-01

SFT cold-start → RL (DeepSeek-R1 style) is still the right recipe *in principle*. But the 2026-06-01
evidence changes what the bottleneck is. The current SFT cold-start is an **undertrained, over-long
thinking model that intermittently fails to terminate within budget and collapses into repetition**
(on a subset of problems, not cleanly difficulty-correlated). GRPO samples rollouts the same way the
eval does — on the problems where the model collapses, every rollout is wrong, reward is ~0, and there
is no gradient signal. **So "GRPO on top of this checkpoint" is not the easy next win; it would mostly spin
on the hard slice.** Fix the cold-start first.

**Why the cold-start is weak — capacity/length mismatch (now evidenced, not hypothesized).** The 32B
traces are long, meandering CoT. A 1.7B has limited capacity to stay coherent over a long chain; the
it fails to terminate within budget on a subset of problems and collapses into repetition
(intermittent, **not** cleanly difficulty-correlated — see table above). Only ~7k
traces (vs R1-Distill-1.5B's ~800k) compounds it. More *epochs* won't help (eval loss already
plateaued). **The lever is the training data: traces that are within the student's capacity.**

**Distillation directions, cheapest → most interesting:**
1. **Length-controlled rejection sampling (STaR/RFT-style).** Keep only *correct* traces under N tokens
   (e.g. ≤2–4k). Trains concision + reliable termination — directly attacks the `111…` collapse.
2. **Capacity-matched teacher = the official post-trained Qwen3-1.7B itself** (Alex's observation,
   2026-06-01). Since it *is* a 1.7B, its traces are by construction within 1.7B capacity and
   terminate cleanly — the most directly "1.7B-friendly" trace source available. **Match the ceiling to
   the mode you'd actually distill:** non-thinking traces (shorter, no `<think>`, less collapse-prone)
   → ceiling **73.0%**; thinking traces → **93.4%** but they're the long-CoT kind our student already
   can't sustain, so non-thinking is likely the saner target here. Caveat Alex flagged: it's somewhat
   *uninteresting* (a roundabout copy of an existing model) and sidesteps the more interesting research
   question of trace distillation itself. Good as a **strong baseline / sanity ceiling**, not the
   headline experiment.
3. **Teacher-rewritten concise traces** (have 32B or Claude re-derive each solution in ≤K tokens,
   clean steps) — tests the *interesting* hypothesis: does distilling to concise, well-structured
   reasoning beat raw verbose teacher CoT for a capacity-limited student? (Alex's intuition: "the
   meandering thinking trace probably doesn't hold up" for a 1.7B. Note: what you *see* from models
   like Claude is often a summarized view; raw thinking can wander — so the real variable to test is
   trace concision/structure as an SFT target, independent of which teacher produces it.)

**On "new capability" (Alex's earlier point, still holds).** Self-distillation on the model's *own*
correct rollouts (RFT/STaR) ≈ GRPO family — both ceiling at ~pass@k, no new knowledge. New capability
needs a **stronger teacher**. The nuance added today: the stronger-teacher (32B) traces must also be
**length/concision-controlled**, or the student can't reproduce them and collapses. So the real lever
is *capacity-matched teacher distillation* → then GRPO on a cold-start that doesn't collapse.

**Hard prerequisite before any v2 number: a pinned, validated inference stack + a canary.** Today
proved (again) that an unpinned stack silently changes results — our checkpoint solves 572 8/8 under
April's HF backend but 0/8 under the 2026-06-01 vLLM (while the *official* 1.7B is clean 4/4 on that
same vLLM, so the engine is fine — our checkpoint is just fragile and stack-sensitive). Before trusting
any eval: (1) pin `torch`/`transformers`/`vllm` in a lockfile tied to the run; (2) **gate every eval on
a canary** — assert the stack reproduces a known-good result (e.g. our SFT checkpoint ≈8/8 on 572)
before believing any number; (3) for *this* checkpoint specifically, the
`extra_special_tokens` list→{} patch (needed for transformers ≥4.51, vs April's tf-5.5.3 intact form)
is the **single most likely culprit and is testable locally with no GPU** — load the patched tokenizer
vs an intact-list→dict version and diff how a full rendered sequence encodes *and* whether `<|im_end|>`
/`<think>` keep special-token status (the round-trip check confirmed IDs but not status, which is what
vLLM's stop-handling keys on). If that's the cause, the April result needs no env reproduction — just a
correct list→dict patch. A fragile model that can't be evaluated stably is itself a signal the
cold-start needs strengthening.

**Format choice for v2 (chat vs completion).** The math POC's GRPO-from-base line is clean in
**completion/few-shot** format. The thinking/chat `<think>` line is a *different* universe and is what
collapses. Don't mix them in one comparison. If pursuing the thinking-SFT path, hold chat+thinking
format constant end-to-end and budget for 16–32k generation (8192 is too short for thinking — it
truncates and inflates apparent collapse).

## Confirmation run (in progress)

Goal: prove the on-disk/HF step-3000 weights *regenerate* ~44% live under pinned code, and
pin the eval bug. Matrix on one A40 pod (`eval-pin-2026-05-30`):

| # | weights | backend | revision | expected | result |
|---|---------|---------|----------|----------|--------|
| 1 | GRPO 3000 | HF generate | `63870ec` | ~44% | ✅ **full-500 greedy = 219/500 = 43.8%** (live, reproduces archived 44.2% within 2 problems). `outputs/grpo3000_greedy500_confirm.json` |
| 2 | GRPO 3000 | vLLM | `63870ec` | distinguish bug | ⚠️ ABANDONED — vLLM 0.22 install bumped torch→2.11/cu130, EngineCore init fails (CUDA-driver mismatch, environmental). Not weights. |
| 3 | GRPO final | vLLM | `main` (7496) | ~11% | ⚠️ same env break; not run |
| 4 | Base | HF generate | — | ~35.8% | _pending_ |
| 5 | SFT (final ckpt) | HF (chat template) | sha `9822…` | per-problem | **April HF (trustworthy):** clean-solve / intermittent-collapse — 572=8/8, 807=7/8, 2584=6/8 clean; 1994=0/8, 1349=0/8 fully collapsed. 26/48 raw on these 6 (dataset order). See "SFT branch". |
| 6 | SFT (final ckpt) | **vLLM 0.8.5** | sha `9822…` | reproduce #5 | ❌ **CHECKPOINT FRAGILE on this stack (2026-06-01).** Same bytes/sampler: 572 **8/8→0/8**, 807 7/8→0/8, 2584 6/8→0/8 (peg 8192). **Engine itself is FINE** — stock official Qwen3-1.7B on the same vLLM solved 572 cleanly 4/4. So our undertrained ckpt is fragile/stack-sensitive (cause: marginal model and/or `extra_special_tokens` patch confound). Not usable for full-500; reproduce April HF+tf5.5.3 env, gate on 572≈8/8 canary. |

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

## Recommended next experiments (rewritten 2026-06-01 — see "Recommended path (v2)" for detail)

0. **Pin + canary the stack FIRST.** Lockfile `torch`/`transformers`/`vllm`; gate every eval on a
   known-good canary (SFT checkpoint solves 572≈8/8 via HF backend) before trusting any number.
   Without this, every result is suspect (proven again 2026-06-01).
1. Trustworthy full-500 SFT c/n on a validated stack (HF backend). Expect it dragged down by
   hard-problem collapse — hold the number as unknown until measured.
2. **Better SFT data, not GRPO-on-this:** length-controlled / capacity-matched distillation traces
   (rejection-sample short correct traces; or distill from the official 1.7B as a baseline; or
   teacher-rewritten concise traces — the interesting one). Then GRPO on a cold-start that doesn't
   collapse, with small KL + early-stop.

## Session close-out (2026-05-30) — DONE

- ✅ Full-500 greedy reproduced live = 43.8% (`outputs/grpo3000_greedy500_confirm.json`, pulled & durable).
- ✅ Pod `p44nx27xkitomr` **torn down** (no active pods).
- ✅ Docs cleaned: README/PLAN banners + corrected tables; `eval-discrepancy-investigation.md` RESOLVED.
- ⚠️ **SFT 2026-05-31 bullet SUPERSEDED by 2026-06-01 re-correction** (see "SFT branch" + "Recommended path (v2)" above). The "NOT degenerate / 26/48 clean" framing was a mischaracterization: the same April log shows **clean-solve / intermittent-collapse** (572=8/8 clean, but 1994 & 1349 = 0/8 fully collapsed). The checkpoint is an **undertrained thinking model**. A trustworthy full-500 is still unmeasured (under the 2026-06-01 vLLM our fragile ckpt gave 572 8/8→0/8, though the engine is fine — official 1.7B was clean 4/4). The lever is **better SFT data** (capacity-matched / length-controlled traces), not GRPO-on-top.
- Pinned eval code: tag `eval-pin-2026-05-30`. Provenance stamped by `math500_eval.py`; ledger `RUNS.jsonl`.
- **Reproduce headline:** rent A40, `pip install torch==2.6.0 transformers==5.5.3 datasets math-verify`,
  then `python greedy500_confirm.py` (or `math500_eval.py --model heyalexchoi/qwen3-1.7b-math-grpo --revision 63870ec239b2 --checkpoint_step 3000`). Env note: vLLM 0.22 needs a newer CUDA driver — use HF backend.

### Next experiments (v2)
1. Fair SFT eval (sampling@0.6, ~8192 tokens).
2. SFT→GRPO with small KL (`beta`≈0.001–0.01) + greedy-eval-in-loop + early-stop / keep-best.
