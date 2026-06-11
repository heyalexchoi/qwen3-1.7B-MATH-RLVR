# SFT v2 — Results & Analysis

*Written by Claude Opus 4.8 · 2026-06-11*

Full results and post-mortem for SFT v2: Qwen3-1.7B-Base fine-tuned on the
concise-distillation dataset
([`heyalexchoi/qwen3-math-concise-sft-v2`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-concise-sft-v2),
7,149 verify-gated traces, median 174 tok). Model:
[`heyalexchoi/qwen3-1.7b-math-sft-v2`](https://huggingface.co/heyalexchoi/qwen3-1.7b-math-sft-v2).
The journey that led here (SFT v1 termination disease, the concise-trace
hypothesis, dataset construction) is in
[`POC-RESULTS.md`](POC-RESULTS.md) and
[`distill-trace-framework.md`](distill-trace-framework.md).

## 1. Headline results (MATH-500, math-verify)

| Model | greedy pass@1 | pass@8 | inferred pass@1 (avg-k/8) |
|---|---|---|---|
| Qwen3-1.7B-Base | 35.8% | — | — |
| GRPO-from-base POC (step 3000) | 44.2% | — | — |
| SFT v1 (verbose 32B traces) | 40.2% | — | — |
| **SFT v2 (concise traces)** | **~50%** (49.4 / 50.6 across two greedy passes¹) | **74.6%** | **46.5%** |

¹ vLLM batching is nondeterministic even at temperature 0; both passes are on
the same weights (sha256 `772bf3a3…`). Treat greedy as ~50 ± 1.

**Termination disease cured:** 500/500 generations finish with `stop` (zero
8192-token pegs), median answer ~260 tokens. SFT v1 — trained on median
3.3k-token verbose traces — pegged the cap constantly. This was the central
hypothesis of the concise dataset and it held.

Eval setup: vLLM 0.22.1 / transformers 5.10.2 / torch 2.11.0+cu129 (the pinned
stack, [`vllm-stack-pin.md`](vllm-stack-pin.md)); pass@8 at temp 0.6 / top_p
0.95 / top_k 20 / rep-pen 1.05; scored by math-verify
([`rescore_math500.py`](../scripts/rescore_math500.py)). Raw artifacts:
[`../eval_results/sft_v2_2026-06-11/`](../eval_results/sft_v2_2026-06-11/).

## 2. Training curves — the plateau

3 epochs on A40 (~70 min, wandb `vccas49o`), effective batch 16, lr 2e-5
cosine. Eval loss on the held-out split:

| epoch | eval_loss | eval token-acc |
|---|---|---|
| 0.24 | 0.4042 | .8825 |
| 0.47 | 0.3216 | .8995 |
| 0.71 | 0.3064 | .9020 |
| 0.94 | 0.2992 | .9039 |
| 1.18 | 0.2985 | .9042 |
| 1.41 | 0.2960 | .9050 |
| 1.65 | 0.2940 | .9054 |
| **1.88** | **0.2937 (min)** | .9055 |
| 2.12 | 0.2944 | .9055 |
| 2.35 | 0.2947 | .9056 |
| 2.59 | 0.2946 | .9055 |
| 3.00 | 0.2946 | .9055 |

Reading:

- **Eval loss bottoms at epoch ~1.9 and goes flat** (drift after that is
  +0.0009 — noise, not divergence). Token accuracy freezes at .9055 from
  epoch 1.65.
- **Train loss kept falling 0.86 → 0.30** through epoch 3 while eval loss was
  flat — the classic memorization signature. Epoch 3 wasn't harmful here, but
  it bought nothing.
- **Conclusion: this is an *epoch-limited* plateau, not a *data-limited* one.**
  More epochs on this dataset are exhausted; more/better data is not. That
  distinction drives everything in §5.
- Config response: `load_best_model_at_end` + `metric_for_best_model:
  eval_loss` now in [`sft_config_v2.yaml`](../configs/sft_config_v2.yaml)
  (commit `b0a90ef`) so future runs keep the epoch-~1.9-style best checkpoint
  instead of the last one. (The v2 run's epoch-2 checkpoint was rotated away
  by `save_total_limit: 2`; final-vs-best delta was noise, so no retrain.)

## 3. Where the score comes from — level & subject breakdown

By MATH difficulty level (n=500):

| Level | n | greedy | pass@8 | avg k/8 |
|---|---|---|---|---|
| 1 | 43 | 90.7% | 97.7% | 82.8% |
| 2 | 90 | 74.4% | 94.4% | 71.2% |
| 3 | 105 | 59.0% | 88.6% | 55.4% |
| 4 | 128 | 46.1% | 71.1% | 39.5% |
| 5 | 134 | 19.4% | 46.3% | 17.9% |

By subject (greedy → pass@8): Algebra 71.8→88.7, Prealgebra 63.4→87.8, Number
Theory 58.1→83.9, C&P 42.1→73.7, Geometry 36.6→63.4, Intermediate Algebra
29.9→56.7, Precalculus 28.6→53.6.

**k-of-8 distribution is strongly bimodal:**

```
k:  0    1   2   3   4   5   6   7   8
n: 127  51  30  46  33  29  46  47  91
```

Per-problem correctness is largely deterministic, not sampling luck. Three
populations:

- **91 problems at 8/8** — solidly solved.
- **125 "ranking failures"** — greedy wrong but ≥1 of 8 samples right. The
  model *contains* the correct solution but doesn't rank it first. **This is
  exactly the population GRPO exploits** — and why pass@8 (74.6%) is a lower
  bound on GRPO-reachable greedy performance.
- **127 hard fails at 0/8** — no amount of re-ranking helps; these need new
  knowledge or new method. GRPO cannot fix these. Hotspots: Intermediate
  Algebra L4–L5 (40 of the 127), Precalculus (19), Geometry (14).

## 4. Qualitative failure analysis — and the diagnosis

Read 10 random 0-of-8 failures (greedy + sampled generations; preserved at
[`../eval_results/sft_v2_2026-06-11/failure_samples_0of8.txt`](../eval_results/sft_v2_2026-06-11/failure_samples_0of8.txt)).
Four modes:

### 4.1 Decorative Verify (dominant)

Every generation includes a `Verify:` line — and it **never actually
recomputes anything**. It rubber-stamps whatever the Solve section produced.
In one case the model *detected a genuine contradiction* during verification
and then "resolved" it in favor of the wrong answer rather than backtracking.
The verification ritual survived training; the verification *function* did
not.

### 4.2 Systematic wrong setup with zero sample diversity

Example: solve `3^{2x} + 19 = 10^x`. **7 of 8 samples make the identical
false substitution** (`y = 10^x` with `3^{2x} = y²`, which is just wrong) and
conclude "no real solution." The true answer (x=2) is findable by testing
small integers — a strategy the model never deploys, because abandoning a
failed plan appears nowhere in its training data. Sampling temperature buys
no diversity when the error is in the *plan*, not the arithmetic.

### 4.3 Knowledge gaps

Spectral radius used where the operator norm is needed (non-normal matrix);
terminating-decimal criterion misapplied; an annuity problem read as a single
deposit, producing a 49.5% interest rate that no functional sanity check
would survive — and Verify waved it through (see 4.1).

### 4.4 Asymptote-diagram parsing

Geometry problems with `[asy]` blocks fail disproportionately; the model
misreads the construction.

### Diagnosis: we distilled out verification and backtracking — twice

Both filters in the data pipeline removed error-recovery from the training
distribution:

1. **Correct-only gating.** Only verify-passed traces were kept, so every
   `Verify:` in training data was written downstream of a known-correct
   answer. The model learned `P(verify confirms) ≈ 1.0` as a token pattern.
   It has *zero* examples of a check failing and triggering a redo.
2. **The rewrite prompt stripped exploration explicitly.** The distillation
   `SYSTEM` prompt in [`rewrite_full.py`](../scripts/rewrite_full.py)
   commands: *"Remove ALL backtracking, false starts, restatements, recaps,
   and self-doubt"* and specifies *"Verify: one line — a quick check."* Even
   where the 32B teacher's verbose trace contained real exploration and
   self-correction, the rewrite linearized it away. We kept the destination
   and erased the navigation.

Important nuance: this is **not** evidence that rambling R1-style traces are
necessary. The failures call for *functional* self-checking (substitute the
answer back; test a special case; sanity-check magnitude) and *plan
abandonment* — both compatible with concise traces (~+50–150 tok). Length was
never the active ingredient; recovery behavior is.

## 5. Implications & next steps

1. **GRPO v2 from the SFT-v2 checkpoint** targets the 125 ranking failures —
   the 25-point greedy↔pass@8 gap is its food. POC lessons apply: greedy eval
   every ~500 steps as early-stop, nonzero KL (beta 0.01–0.05), watch for
   loop collapse ([`POC-RESULTS.md`](POC-RESULTS.md) §lessons).
2. **The 127 hard fails need data, not RL.** Plan under discussion:
   - Keep the current dataset as the **happy-path** corpus (rename/alias
     `MATH-happy-path`): concise, decisive, first-attempt-correct.
   - A **second distillation pass** (weighted to L3+/hard subjects, possibly
     drawing on NuminaMath for volume — dedup against MATH-500 is
     non-negotiable) with a revised prompt that (a) requires Verify to be a
     functional, independent check that could fail, and (b) **preserves
     real corrections and backtracking** instead of stripping them —
     producing a self-correction dataset. Real errors (ideally the SFT-v2
     student's own, teacher-corrected) beat manufactured ones: telling a
     teacher to "include backtracking" on a clean solve yields theatrical
     self-correction.
   - Mix: recovery traces as a minority (~20–30%) so first-attempt-correct
     stays the dominant behavior.
3. **2 epochs is the right budget** for reruns on similar-sized data;
   best-checkpoint selection is now config default.
