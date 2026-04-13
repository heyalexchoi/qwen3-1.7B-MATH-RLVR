# Qwen3-1.7B Math RLVR

Demonstrating distillation + RLVR on math reasoning with Qwen3-1.7B-Base.

→ **Full pipeline plan, findings, and decisions:** [`PLAN.md`](PLAN.md)
→ **Active run / current phase:** [`PLAN.md`](PLAN.md) → Current Run
→ **Pod setup, SSH, launch commands, pre-removal checklist:** [`docs/runpod.md`](docs/runpod.md) — read this before any pod operation
→ **Session log:** [`memory/math-rlvr.md`](../memory/math-rlvr.md)

---

## Pipeline Overview

```
[0] Data prep (GSM8K + MATH)                           ✅
[1] Base eval — GSM8K + MATH-500                       ✅  35.8% pass@1 / 65.0% pass@8 (math-verify)
[2] Generate Qwen3-32B traces                          ✅  7,154 correct traces (95.51%)
[3] SFT on correct traces                              ✅  (degenerate — see Key Findings)
[3a] SFT eval — MATH-500                               ✅  ~0% — both checkpoints degenerate
[4] GRPO from base model                               ✅  DONE (step 7496, wandb 99hauae9) — see Key Findings
[4a] GRPO eval — checkpoint-3000 (local)               ✅  44.20% greedy / 71.40% pass@8 / 36.83% inferred (⚠️ local only — see Key Findings)
[4b] GRPO eval — checkpoint-5000 (HF Hub)             ✅  11.60% greedy / 30.80% pass@8 / 9.83% inferred (collapsed — see Key Findings)
[4c] Binary search: steps 2500/3500/4000/3000(HF)     ✅  All collapsed ~9-11% — HF Hub has no good checkpoint (see Key Findings)
[4d] Train/eval reward gap investigation               ✅  No logging artifact — see Key Findings
[5] Retrain GRPO with collapse mitigations             ⏳  See Key Findings → recommendation
```

---

## Results

| Phase | MATH-500 pass@1 | pass@8 | Notes |
|-------|----------------|--------|-------|
| Base | **24.55%** (inferred c/n) / 35.8% (greedy) ✅ | **65.0%** ✅ | math-verify; `outputs/baseline_math500_mv_rescored.json` |
| Post-SFT | ~0% | — | Degenerate — both 1-epoch and 3-epoch checkpoints (see Key Findings) |
| GRPO step-3000 (local ckpt) | **36.83%** (inferred c/n) / **44.20%** (greedy) ⚠️ | **71.40%** ⚠️ | Local checkpoint only — **cannot reproduce from HF Hub** (see Key Findings → HF Hub divergence) |
| GRPO step-3000 (HF Hub `63870ec`) | 9.22% (inferred) / 11.2% (greedy) ❌ | 26.0% ❌ | HF Hub re-eval — collapsed like all other steps |
| GRPO steps 2500/3500/4000/5000 (HF Hub) | ~9–10% (inferred) / ~11% (greedy) ❌ | ~29–31% ❌ | All collapsed — repetition degeneration (see Key Findings) |

### Baseline by level (math-verify)

| Level | n | pass@1 | pass@8 |
|-------|---|--------|--------|
| L1 | 43 | 65.1% | 90.7% |
| L2 | 90 | 54.4% | 88.9% |
| L3 | 105 | 40.0% | 81.9% |
| L4 | 128 | 34.4% | 57.8% |
| L5 | 134 | 11.9% | 34.3% |

---

## Key Findings

### Latent capability vs. expressed capability

The greedy pass@1 (35.8%) vs inferred c/n pass@1 (24.55%) vs pass@8 (65.0%) gap tells a clear story: the base model's best single answer (greedy) outperforms its average sampling accuracy, and both are far below its ceiling (pass@8). The 1.7B model "knows" more than any single decoding reveals — especially at harder levels (L5: 11.9% greedy → 34.3% pass@8). **Use inferred pass@1 = 24.55%** as the apples-to-apples baseline vs SFT/GRPO evals (same n=8 sampling methodology). This is the core motivation for SFT + GRPO: SFT teaches the reasoning format, GRPO unlocks latent capability via RL.

### max_new_tokens: efficiency vs. learning tradeoff

There is a real tradeoff between training/eval cost and learning potential. Correct teacher (Qwen3-32B) responses by level:

| Level | p50 tokens | p95 tokens | max tokens |
|-------|-----------|-----------|-----------|
| L1 | ~1,500 | ~4,000 | ~13k |
| L2 | ~1,800 | ~5,700 | ~23k |
| L3 | ~2,300 | ~8,600 | ~40k |
| L4 | ~2,800 | ~10,000 | ~40k |
| L5 | ~4,600 | ~14,000 | ~36k |

Longer outputs = more learning potential (especially L5), but GRPO step time grows rapidly:

| max_completion_length | Worst-case rollout time |
|----------------------|------------------------|
| 4,096 | ~6 min/step |
| 8,192 | ~12 min/step |
| 12,000 | ~18 min/step |
| 16,384 | ~24 min/step |

**Decision: `max_new_tokens=8192`** — covers ~90% of correct teacher responses, manageable GRPO step time. Future work: curriculum approach (start short, extend as model improves).

### Qwen3 thinking mode: no greedy decoding

Qwen3 thinking-mode models must NOT use greedy decoding (`temperature=0`). The `<think>` block enters circular reasoning loops that fill the entire token budget with no `\boxed{}` answer. This cost us most of a day on the eval.

**Required inference settings (Qwen official):** `temperature=0.6, top_p=0.95, top_k=20`

We run `n_samples=8` at this temperature and compute:
- **pass@8**: any-correct over 8 samples
- **pass@1 (unbiased estimate)**: `c/n` from 8 samples (Chen et al. 2021 Codex estimator)

### Baseline methodology and data provenance

#### Two-step pipeline: generation → scoring

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Generate responses | `scripts/math500_eval.py` | HuggingFaceH4/MATH-500 | `outputs/math500_results.json` |
| 2. Score with math-verify | `scripts/rescore_math500.py` | `outputs/math500_results.json` | `outputs/baseline_math500_mv_rescored.json` |

**To reproduce the baseline numbers from scratch:**
```bash
# Step 1: generate (requires GPU, ~30 min on A40/L40S)
# Note: methodology locked as constants in script (MAX_NEW_TOKENS=2048, N_SAMPLES=8, TEMPERATURE=0.7)
# Output auto-named: outputs/base_step0_math500_results.json (or pass --checkpoint_step 0)
python scripts/math500_eval.py \
  --model Qwen/Qwen3-1.7B-Base \
  --checkpoint_step 0

# Step 2: score with math-verify (CPU, ~1 min)
# Output auto-named: outputs/base_step0_math500_mv_rescored.json
python scripts/rescore_math500.py \
  --input outputs/base_step0_math500_results.json
```

#### Generation parameters

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-1.7B-Base` |
| Prompt format | Few-shot (4 examples), no chat template — raw `"Problem: {problem}\nSolution:"` |
| pass@1 | Greedy (`temperature=0.0`), 1 sample |
| pass@8 | `temperature=0.7`, 8 samples |
| `max_new_tokens` | 1024 (default; analyzed — see note below) |

Note: `temperature=0.7` is the default in `math500_eval.py` and matches what `math500_results.json` metadata records. The Qwen3 recommended temperature of `0.6` applies to `sft_eval.py` (thinking-mode instruct) — not used for the base model baseline.

**Note on 1024-token cap:** Measured response lengths across all 4,500 baseline responses (500 problems × 9 samples): p50=72 words, p95=488 words, p99=597 words (~460 tokens), max=963 words (~740 tokens). Only 7/4,500 responses (0.2%) approached the cap. The base model does not produce long chain-of-thought. The cap is not binding and does not meaningfully affect the numbers. No re-run needed.

#### Output files

`outputs/` is gitignored. All eval artifacts are uploaded to HF: [`heyalexchoi/qwen3-math-rlvr-results`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-rlvr-results).

| File | Script | HF artifact | Contents |
|------|--------|-------------|----------|
| `outputs/math500_results.json` | `math500_eval.py` | `outputs/math500_results.json` | 500 problems × (1 greedy + 8 sampling responses), baseline, Qwen3-1.7B-Base |
| `outputs/baseline_math500_mv_rescored.json` | `rescore_math500.py` | `outputs/baseline_math500_mv_rescored.json` | Same responses re-scored with math-verify — authoritative baseline: 35.8% / 65.0% / 24.55% |
| `outputs/grpo_step3000_math500_results.json` | `math500_eval.py` | ✅ On HF | GRPO checkpoint-3000 (epoch 0.40) — 500 problems × (greedy + 8 samples) |
| `outputs/grpo_step3000_math500_mv_rescored.json` | `rescore_math500.py` | ✅ On HF | Step-3000 rescored: 44.20% greedy / 71.40% pass@8 / 36.83% inferred |
| `outputs/grpo_step5000_math500_results.json` | `math500_eval.py` | ✅ On HF | GRPO checkpoint-5000 (67% of training) — collapsed; raw generations show repetition loops |
| `outputs/grpo_step5000_math500_mv_rescored.json` | `rescore_math500.py` | ✅ On HF | Step-5000 rescored: 11.60% greedy / 30.80% pass@8 / 9.83% inferred (below base) |
| `outputs/baseline_results.json` | `baseline_eval.py` | — | GSM8K baseline (separate dataset) — not MATH-500 |

Response text in `math500_results.json` and `baseline_math500_mv_rescored.json` is byte-identical. The rescored file updates the `correct` flag on every sample and adds `pass1_mv`/`pass8_mv` per-sample boolean arrays and a `summary` block with final numbers.

**Artifacts upload directly from the eval pod** — no rsync needed. Pass `--upload` to both scripts:
```bash
python scripts/math500_eval.py --model heyalexchoi/qwen3-1.7b-math-grpo --latest --upload
python scripts/rescore_math500.py --input outputs/grpo_step<N>_math500_results.json --upload
```
Both scripts read `HF_TOKEN` from env and push to `outputs/<filename>` in `heyalexchoi/qwen3-math-rlvr-results`.

#### Authoritative baseline numbers (from `rescore_math500.py`)

| Metric | Value |
|--------|-------|
| Greedy pass@1 | **35.8%** |
| Pass@8 | **65.0%** |
| Inferred pass@1 (c/n) | **24.55%** — Chen et al. 2021 unbiased estimator from pass@8 samples |

Use **24.55%** as the apples-to-apples comparison for GRPO eval (same n=8 sampling methodology).

#### Comparability with GRPO

GRPO uses the same few-shot, no-chat-template prompt format as the baseline. This makes the baseline directly comparable to GRPO step-0 rollout accuracy and the final GRPO eval.

### SFT checkpoints: degenerate thinking chains across all epoch counts

Both evaluated SFT checkpoints produce ~0% correct answers on MATH-500. The failure mode is epoch-independent — the initial hypothesis that fewer epochs would reduce overfitting and produce convergent chains was falsified.

#### Checkpoint comparison (partial eval data)

| Metric | checkpoint-1275 (3-epoch) | checkpoint-500 (~1 epoch) |
|--------|--------------------------|--------------------------|
| Samples evaluated | 56 (7 problems) | 264 (33 problems) |
| Opens `<think>` | ~100% | 73% |
| Closes `</think>` | 7/56 = 12.5% | 50/264 = 19% |
| Contains `\boxed{}` | ~0% | 2% |
| Hits max_new_tokens (8192) | 100% | 11% |
| Any correct | 0 | 1 (coincidental — see below) |
| Primary degeneration style | Verbal loops ("Okay. Alright. Okay...") → nonsense words | Character/digit repetition ("x x x", "0 0 0", CSS junk) |

**Similarities:** Both checkpoints open `<think>` and begin coherent reasoning before degenerating. Neither reliably closes `</think>` or reaches `\boxed{}`. Both score ~0%.

**Differences:** The 3-epoch checkpoint runs to max tokens on every sample with verbal loop degeneration. The 1-epoch checkpoint degenerates earlier (fewer max-token hits) with a different character/digit repetition pattern. The 1-epoch model is less entrenched in the looping behavior but equally unable to produce correct answers.

**The one "correct" answer at checkpoint-500** was a geometry problem (pentagon rotation = 72°). The model reasoned coherently for ~300 tokens, produced `\boxed{72}`, then immediately degenerated into thousands of tokens of `BorderStyle: none;BorderStyle: none;...` repetition before finally emitting a stop token. The answer was extractable and math-verify scored it correct — but the underlying generation is still broken; the model does not know to stop after reaching an answer.

**Stop tokens, prompt format, and training sequences are all correct.** This is not an inference configuration bug. Training sequences end with `<|im_end|>` (151645); eval prompt format is byte-identical to training format.

#### Root cause: trace length vs. 1.7B model capacity

Training traces are Qwen3-32B outputs: median **3,268 tokens**, mean **4,799 tokens**, p90 **10,073 tokens**, max **42,998 tokens**. The 1.7B model learns the consistent, short entry pattern (opening `<think>`, "Okay, so I need to...") but cannot sustain coherent reasoning across thousands of tokens of verbose 32B-style thinking chains. Once the model leaves its learned distribution, it collapses into repetition. This happens at 1 epoch and at 3 epochs — epoch count does not address the capacity gap.

`repetition_penalty` changes the surface shape of degeneration (verbal loops vs. character repetition) but does not fix the underlying failure.

**GRPO impact:** GRPO requires nonzero rollout accuracy to generate reward signal. ~0% from any SFT checkpoint is a dead end.

**Decision:** Skip SFT entirely and run GRPO from the base model. Base already scores 24.55% inferred pass@1 — sufficient reward signal for RL. See "GRPO from base: approach and parameters" below.

### GRPO from base: approach and parameters

SFT distillation failed because the 1.7B model cannot reproduce 32B-style reasoning chains (see above). Instead of trying to fix SFT, we skip it and apply GRPO directly to Qwen3-1.7B-Base. The model already solves 24.55% of MATH-500 (inferred pass@1) and 65% (pass@8) — there is latent capability to surface via RL without first teaching a specific reasoning format.

**Prompt format: few-shot, no chat template.** The base model is a text completion model, not a chat model — it has never been trained on chat format. The baseline 24.55% was measured with few-shot raw text prompting (`math500_eval.py` format: 4 solved examples followed by `"Problem: {problem}\nSolution:"`). GRPO uses the same format so the baseline is directly comparable. GRPOTrainer supports raw string prompts natively — when `prompt` is a string (not a list of messages), no chat template is applied.

**Key design choices:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | `Qwen/Qwen3-1.7B-Base` | Skip SFT — base has 24.55% accuracy, sufficient for reward signal |
| Prompt format | Few-shot (4 examples), no chat template | Matches baseline eval format. Base model is a completion model — raw text is its native format. Directly comparable to existing 24.55% baseline |
| Reward | `\boxed{}` correctness only (math-verify) | Single clean binary signal. No format reward — it adds noise and can be learned trivially, diluting correctness signal |
| KL coefficient (`beta`) | 0.0 | No reference model constraint. Allows maximum exploration from base. Standard for zero-RL / DAPO |
| Loss type | `dapo` | TRL 1.0.0 default. Token-level loss normalization, asymmetric clipping |
| Clip epsilon | 0.2 low / 0.28 high | DAPO asymmetric clipping — allows more aggressive exploration on positive advantages (upside) while constraining negative updates |
| Zero-advantage batch skip | Explicit in script | When all 8 rollouts have the same reward, advantages = 0, gradient = 0. TRL does NOT skip the forward+backward — it wastes compute. We override `compute_loss` to early-return for zero-advantage batches. Saves ~10% compute at 25% accuracy, ~43% at 90% accuracy |
| Group size (`num_generations`) | 8 | 8 rollouts per prompt. Enough diversity for advantage estimation; 16 would 2x rollout cost |
| `max_completion_length` | 2048 | Based on baseline eval response distributions (4,500 responses): p50=72 words, p95=488 words, p99=597 words (~460 tokens), max=963 words (~740 tokens). Only 7/4,500 (0.2%) responses hit the 1024-token cap — the cap is not binding. 2048 is ~4× the uncapped p99 and covers natural base-model outputs comfortably. If response lengths grow significantly during GRPO, revisit. 2k vs 8k = ~4× faster per step. |
| Temperature | 0.9 | Higher than eval temp (0.7) for diverse rollouts — more variance in the group → better advantage signal. Standard for GRPO |
| Learning rate | 1e-6 | Conservative for zero-RL on small models. Avoids over-tuning early |
| Training data | Full MATH train set (~7.5k problems) | No pre-filtering. Easy problems provide early reward signal; hard problems become learnable as the model improves |
| Stop token | `<|endoftext|>` (151643) | Base model default eos — correct for text completion format. **No override needed** (the `<|im_end|>` override was only needed for chat template, which we're not using) |

**Why not fix SFT instead?** Two alternatives were considered: (1) Re-SFT with short traces only (<2k tokens, 25% of data) — biases toward easy problems, low ceiling. (2) Traces from a smaller teacher (7B) — significant detour, lower trace quality. Neither addresses the core issue: 1.7B models shouldn't be forced to imitate 32B reasoning style. GRPO lets the model develop its own reasoning patterns at whatever length its capacity supports.

**SFT training pipeline was verified correct.** Before deciding to skip SFT, we confirmed: `max_seq_length=32768` (covers 99%+ of traces — no truncation), `<|im_end|>` is in the loss and not masked (verified by inspecting SFTTrainer label tensors), prompt masking is correct. The SFT failure is a capacity problem, not a training bug.

**Precedent:** DeepSeek-R1 demonstrated that RL from base models can teach reasoning without SFT warmup. The base model's latent capability (pass@8 = 65%) is the raw material; GRPO shapes it into reliable single-pass performance.

### GRPO collapse after step 3000: repetition degeneration

Binary search across checkpoints confirmed: every HF Hub checkpoint (steps 2500, 3000, 3500, 4000, 5000) shows collapse to ~9-11% greedy. No good checkpoint was found on HF Hub.

#### Scores (math-verify)

| Step | Source | Greedy pass@1 | Pass@8 | Inferred pass@1 (c/n) |
|------|--------|--------------|--------|----------------------|
| Base | local | 35.80% | 65.00% | 24.55% |
| Step 3000 | **local ckpt** | **44.20%** ⚠️ | **71.40%** ⚠️ | **36.83%** ⚠️ |
| Step 3000 | HF Hub `63870ec` | 11.20% ❌ | 26.00% ❌ | 9.22% ❌ |
| Step 2500 | HF Hub | 11.00% ❌ | 29.60% ❌ | 8.70% ❌ |
| Step 3500 | HF Hub | 11.20% ❌ | 29.20% ❌ | 9.05% ❌ |
| Step 4000 | HF Hub | 10.80% ❌ | 30.40% ❌ | 9.68% ❌ |
| Step 5000 | HF Hub | 11.60% ❌ | 30.80% ❌ | 9.83% ❌ |

#### HF Hub divergence — original step-3000 result cannot be reproduced

The original step-3000 eval (44.20%) loaded from a **local checkpoint** (`outputs/grpo_checkpoint`, wandb run `ckz7jwil`). Re-eval using HF Hub commit `63870ec239b2` (auto-resolved as step 3000 from commit history) yields 11.2% — collapsed like every other step. The local checkpoint and HF Hub commit do not match. The 44.2% result may be from a different training run or a checkpoint that was never correctly pushed to HF Hub. **Do not rely on the 44.2% result without first recovering and verifying the original local checkpoint.**

#### Observed pathology

Inspecting raw step-5000 greedy completions directly revealed the failure mode. Most responses degenerate into an infinite repetition loop before ever closing `\boxed{}`:

```
...The answer is \frac{14}{3}} = \frac{14}{3}} = \frac{14}{3}} = \frac{14}{3}} = ...
```

The model hits `max_new_tokens=2048` in the loop, no `\boxed{}` is extractable, prediction is empty, answer is marked wrong. By contrast, step-3000 completions are clean:

```
Thus f(-2) + f(-1) + f(0) = 2 + 5/3 + 1 = 14/3. The answer is \boxed{\frac{14}{3}}.
```

The pattern is not limited to greedy decoding. Pass@8 at temp=0.7 also collapsed (30.8% vs base 65%): 58% of the 4,000 sampled completions also hit the 2048-token limit with no answer. See Hypothesis section for the length distribution breakdown.

#### Hypothesis: distribution collapse into short-answer vs. loop modes

Inspecting response lengths in the step-5000 eval shows the output distribution split into two distinct modes (bimodal), with almost nothing in between:

| Response type | Greedy | Pass@8 (temp=0.7) | Mean length |
|---------------|--------|-------------------|-------------|
| Completed (has `\boxed{}`) | 185/500 (37%) | 1692/4000 (42%) | **44 words** |
| Looped (no `\boxed{}`, hit 2048-token limit) | 315/500 (63%) | 2308/4000 (58%) | **810 words** |

The normal middle ground — coherent multi-step reasoning chains of ~200–500 words — has essentially disappeared. Both training and eval used `max_completion_length=2048`. Loops got reward=0 in training — they were not being rewarded. Training reward stayed ~0.41 because ~41% of temp=0.9 rollouts hit the short-answer mode successfully.

**The following are hypotheses about the mechanism — not established facts.** The exact causal chain is unconfirmed; multiple factors likely interact.

1. **(Hypothesis) GRPO may have reinforced short direct answers over reasoning chains.** Completions that reasoned step-by-step had more opportunities to fail (wrong intermediate step, loop, hit token limit) — reward=0. Short completions that stated the answer directly and happened to be correct got reward=1. Over time the model may have learned to skip reasoning and state answers directly. This may have worked at step 3000 but left no coherent fallback when the direct-answer strategy failed.

2. **(Hypothesis) When the direct-answer strategy fails, the model enters a degenerate loop.** Having potentially been pushed away from its original reasoning distribution, the model may have no coherent fallback. It starts generating a computation pattern and repeats it indefinitely. Greedy decoding pins the model to this loop; temp=0.7 sampling escapes it sometimes (hence pass@8 > greedy, but both well below base).

3. **(Hypothesis) `beta=0.0` (no KL penalty) may have enabled unconstrained distribution drift.** No force anchors the policy to the base model's behavior. The distribution can drift arbitrarily far from the base's reasoning-chain style. A nonzero beta would limit this drift.

4. **Training reward vs. eval accuracy gap (open question).** Training reward was ~0.41 at step 5000 while eval inferred pass@1 was only 9.83% — a ~4× gap not explained by temperature difference alone (training rollouts at temp=0.9, eval at temp=0.7). This could indicate overfitting to the training distribution, a mismatch in how reward is computed during training vs. eval, or that the high training reward was driven by easy problems the model already knew while the MATH-500 test set skews harder. This gap is not yet understood and warrants investigation in the next run.

5. **No greedy eval during training meant collapse was invisible.** Training reward is measured on temp=0.9 sampled rollouts, which hit the short-answer mode ~41% of the time and look healthy. The collapse was invisible until post-hoc checkpoint eval. Greedy pass@1 every 500 steps would have caught this around step 3500–4000.

#### Recommendation for next run

The minimum effective fixes, in priority order (per Opus assessment):

1. **Eval greedy every 500 steps during training** — use a small held-out set (50–100 problems). Treat greedy pass@1 as the early-stopping signal, not training reward. This alone would have saved this run at step 3000.

2. **Apply nonzero KL (`beta=0.01–0.05`)** — provides a soft anchor to the base distribution. Prevents the policy from drifting arbitrarily far from the base model's behavior. Trade-off: constrains exploration slightly. This was beta=0.0 in the current run (pure zero-RL / DAPO).

3. **Add a format/length reward** — reward responses that contain a reasoning chain before `\boxed{}`, or penalize responses under N words. This prevents the model from collapsing to direct-answer mode by making the short-answer shortcut unrewarding.

4. **Training reward is not a proxy for generalization** — verified (see Train/eval reward gap findings below). Use greedy pass@1 on a held-out set as the signal.

Note: `repetition_penalty` during rollout generation is an additional option (prevents loop formation), but was not part of the Opus assessment and is lower priority than the above.

Step 3000 checkpoint (`heyalexchoi/qwen3-1.7b-math-grpo`, commit `63870ec`) is the best model from this run and the recommended starting point for analysis.

#### Train/eval reward gap findings

**Result: no logging artifact — the gap is real and explained by two compounding factors.**

`train/reward` (TRL source: `grpo_trainer.py:2024`, `rewards.mean().item()`) = mean correctness fraction across all rollouts in the batch at training temperature (temp=0.9) on training problems. Binary 0/1 per rollout, averaged over all N_prompts × 8 rollouts. This is correctly computed — no artifact.

**The gap:**

| | Step 3000 | Step 5000 |
|--|-----------|-----------|
| Training reward (temp=0.9, train data) | 0.357 | 0.354 |
| Eval inferred pass@1 (temp=0.7, MATH-500) | 36.83% | 9.83% |

Training reward at step 3000 ≈ eval accuracy (0.357 vs 0.368 — training is a reasonable proxy when the model is healthy). At step 5000, training reward is nearly identical (0.354) while eval has crashed 3.7×. The training metric showed no signal of collapse.

**Why training reward stays healthy when the model has collapsed:**

The gap is explained entirely by **temperature** — but the effect is much larger than a naive estimate would suggest, due to a phase-transition dynamic in repetition loops.

At step 3000 (healthy), the temperature ordering is normal:
- Greedy (temp=0): **44.20%** > temp=0.7: **36.83%** > temp=0.9: **~35.7%**
- Higher temperature adds noise to good output → slightly hurts accuracy.

At step 5000 (collapsed), the ordering **flips**:
- Temp=0.9: **~35.4%** >> greedy (temp=0): **11.60%** > temp=0.7: **9.83%**
- Higher temperature now *massively helps* (3.1× boost vs greedy).

The mechanism: repetition loops are **self-reinforcing** — once the model outputs a pattern like `= \frac{14}{3}}` twice, each subsequent repetition becomes more likely given the context. At temp=0.7, sampling is peaked enough that each token follows the loop with high probability → the loop persists to max_new_tokens (58% of temp=0.7 completions hit the 2048-token limit). At temp=0.9, there is enough randomness that a pattern-breaking token appears within a few iterations → the loop collapses before consuming the full budget. This is effectively a phase transition: below some critical temperature the loop is self-sustaining; above it, the loop is unstable.

Training and test are from the same distribution (hendrycks_math, same level distribution within ~4pp per level). Training was <1 epoch — no memorization opportunity. The gap is not a distribution or overfitting effect. **The training temperature of 0.9 was above the critical temperature for loop stability, making the collapse invisible to the reward signal.**

**Implication for next run:** Training reward at temp=0.9 is blind to repetition degeneration. Either (a) periodically eval at greedy/temp=0.7 during training (early-stopping signal), (b) lower the rollout temperature, or (c) add an explicit repetition/format penalty that detects loops regardless of temperature.

### Eval script: train/eval format must match

`sft_eval.py` must use `tokenizer.apply_chat_template()` — the same format SFTTrainer applied during training. Using a plain prompt causes total format mismatch and infinite loops. Stop token is `<|im_end|>` (151645) only — the checkpoint was saved with `eos_token=<|im_end|>`, so both lookups resolve to 151645 and deduplication gives a single stop token. This is correct.

### Qwen3 tokenizer: extra_special_tokens must be a dict

Qwen3 SFT/instruct checkpoints save `extra_special_tokens` as a list in `tokenizer_config.json`. `transformers>=4.51` expects a dict and crashes with `AttributeError: 'list' object has no attribute 'keys'`.

**Affected:** SFT checkpoints — SFT added chat special tokens to the tokenizer, which caused `extra_special_tokens` to be saved as a list. **Not affected:** `Qwen3-1.7B-Base` (field is `None`) or GRPO checkpoints (GRPO starts from the base tokenizer with no special-token modifications — also `None`).

`math500_eval.py` calls `load_tokenizer_safe()` as a precautionary no-op for base/GRPO evals. If you need to patch an SFT checkpoint manually:
```python
import json
c = json.load(open("tokenizer_config.json"))
if isinstance(c.get("extra_special_tokens"), list):
    c["extra_special_tokens"] = {}
    json.dump(c, open("tokenizer_config.json", "w"), indent=2)
```
Apply to `outputs/sft_checkpoint/tokenizer_config.json` and any `checkpoint-N/tokenizer_config.json`. Already patched in local SFT checkpoint.

### vLLM: kill orphaned EngineCore before restarting

vLLM's `LLM` and `AsyncLLMEngine` both spawn a `VLLM::EngineCore` subprocess. If the parent Python process is killed, the subprocess survives and holds the full GPU allocation. Subsequent launches fail with:

```
ValueError: Free memory on device cuda:0 (6.3/79.18 GiB) is less than desired GPU memory utilization
```

Fix:
```bash
nvidia-smi  # find the VLLM::EngineCore pid
kill -9 <pid>
# confirm: nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

Always do this after killing any vLLM process before relaunching.

### Trace quality

Qwen3-32B teacher traces: **95.51% correct (7,154/7,490)** after math-verify rescore + rerun of 244 truncated traces at 32k tokens. Official Qwen3-32B MATH-500 score is 96.1% — our traces are near-ceiling quality.

### OOM root cause (SFT/GRPO training only)

Qwen3's 152k vocab → logits tensor ≈ `batch × seq_len × vocab × 2B`. At batch=2, seq=32768 this is ~40GB — OOM on A100 80GB. Fix: batch=1, grad_accum=16, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Does not affect eval (inference only).

See `PLAN.md` → Key Findings for full details.

---

## Development Status

| Script | Status | Notes |
|--------|--------|-------|
| `prepare_data.py` | ✅ Done | GSM8K: 7,473 train / 1,319 test |
| `baseline_eval.py` | ✅ Done | GSM8K: **74.6% pass@1** |
| `math500_eval.py` | ✅ Active | **Step 1** of baseline pipeline. Generates responses (few-shot, HF backend). Also used for GRPO eval. Output: `outputs/math500_results.json` |
| `rescore_math500.py` | ✅ Active | **Step 2** of baseline pipeline. Scores `math500_results.json` with math-verify → `baseline_math500_mv_rescored.json`. Canonical source of 35.8% / 65.0% / 24.55% |
| `generate_traces_32b.py` | ✅ Done | 7,490 MATH traces at MAX_TOKENS=32768 |
| `rescore_traces.py` | ✅ Done (legacy) | Regex normalizer; superseded by math-verify |
| `rescore_mathverify.py` | ✅ Done | math-verify rescore; **95.51% (7,154/7,490)** |
| `rerun_truncated.py` | ✅ Done | 244 truncated reruns; 160 newly correct |
| `sft_train.py` | ✅ Done | SFT on 7,154 correct MATH traces |
| `sft_eval.py` | ✅ Done | SFT/chat-template eval — vLLM + math-verify. Used for SFT checkpoints (both degenerate ~0%). Not for base/GRPO models (chat template format mismatch) |
| `grpo_train.py` | ✅ Done | GRPO from base model — completed step 7496, wandb 99hauae9. Best checkpoint: step 3000 (collapsed after) — see Key Findings |
| `eval_comparison.py` | ⏳ Pending | See `PLAN.md` |

---

## Setup

```bash
conda create -n qwen-math python=3.11 -y
conda activate qwen-math
pip install -r requirements.txt
pip install 'math-verify[antlr4_13_2]'
```

### Sensitive config

For **pod training**: create `secrets.env` in the project root (gitignored, rsynced to pod by setup script):
```
HF_TOKEN=hf_xxxx
WANDB_API_KEY=xxxx
GITHUB_TOKEN=ghp_xxxx
```

For **local eval** (`run_eval.sh`): create `~/.config/openclaw/secrets.env` (mode 600) with `HF_TOKEN`. `run_eval.sh` loads this automatically.

Get `HF_TOKEN` from https://huggingface.co/settings/tokens. See `docs/runpod.md` → "One-Command Setup" for full pod training prerequisites.

## Running eval

### GRPO checkpoint eval — `math500_eval.py`

Use `math500_eval.py` for GRPO checkpoint eval. **Not** `sft_eval.py` and **not** `run_eval.sh`.

**Why:** GRPO was trained with few-shot prompts (`Problem: ...\nSolution:`). `sft_eval.py` uses chat template — a different input format. `math500_eval.py` uses the same `create_prompt()` from `prompts.py` that training used, so results are directly comparable to the baseline.

Runs both greedy pass@1 and pass@8 sampling in one invocation. Uses **math-verify** (ANTLR4/SymPy) when installed, regex fallback otherwise.

```bash
# On eval pod (A40 or L40S 48GB) — checkpoint downloads directly from HF, no local copy needed
cd /workspace/qwen3-math-rlvr

# Eval latest checkpoint (queries HF commit history, auto-names output grpo_step<N>_math500_results.json)
python scripts/math500_eval.py \
  --model heyalexchoi/qwen3-1.7b-math-grpo \
  --latest

# Or eval a specific step
python scripts/math500_eval.py \
  --model heyalexchoi/qwen3-1.7b-math-grpo \
  --checkpoint_step 3000
```

Methodology is locked in script constants (`MAX_NEW_TOKENS=2048`, `N_SAMPLES=8`, `TEMPERATURE=0.7`) — not CLI args.
Output auto-named: `outputs/{model_tag}_step{N}_math500_results.json`. Script errors loudly if step cannot be determined.

Estimated runtime: **~20 min on L40S with vLLM** (greedy 500 + sampling 500×8 in two `llm.generate()` calls). HF backend is much slower (~5h) due to batch-waits on variable-length outputs — use vLLM. `math500_eval.py` auto-detects vLLM and uses it if installed; install with `pip install vllm`.

Install math-verify before running:
```bash
pip install 'math-verify[antlr4_13_2]'
```

Full eval pod setup is in `PLAN.md` → Step [4a].

### SFT / instruct checkpoint eval — `sft_eval.py` via `run_eval.sh`

**Note:** Both SFT checkpoints evaluated to ~0% (degenerate thinking-mode outputs — see Key Findings). This section is kept for reference if SFT eval is needed again. `sft_eval.py` is for chat-template models only — do not use it for base or GRPO checkpoints.

```bash
# Canonical launch — auto-detects vLLM, logs to logs/sft_eval_TIMESTAMP.log
bash scripts/run_eval.sh

# With options
bash scripts/run_eval.sh --model outputs/sft_checkpoint --max_new_tokens 8192

# Greedy pass@1 eval (fast, ~30min on H100 with vLLM)
bash scripts/run_eval.sh --greedy

# Force a specific backend
bash scripts/run_eval.sh -- --backend hf   # HF fallback (slower)

# Check progress
tail -f logs/sft_eval_*.log
wc -l outputs/sft_eval_results.jsonl
```

`run_eval.sh` handles: loading secrets, setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, auto-installing vLLM if missing, timestamped log file, background launch. Do not launch `sft_eval.py` directly without it.

### Inference backend: vLLM

`sft_eval.py` uses vLLM as its primary inference backend, with HF `model.generate()` as fallback.

**Why vLLM:** The HF batched approach generates all N samples for a problem in one `model.generate()` call, which means the entire batch waits for the slowest sample to finish. For Qwen3 thinking-mode outputs, a single sample frequently runs to the full 8192-token limit — holding up the other 7. Benchmarking on an A40 showed ~540s/problem average, projecting to ~60h for 500 problems. vLLM's continuous batching schedules all 4,000 requests (500 problems × 8 samples) independently: when any sequence finishes, that slot fills immediately. Combined with the H100's ~5x memory bandwidth advantage over the A40, expected throughput is ~2–4h for the full sampling eval.

---

## Hardware

- **Eval:** H100 SXM 80GB — vLLM + H100 gives ~2–4h for full sampling eval (500 problems × 8 samples)
- **SFT / GRPO:** A100 SXM 80GB or H100 — required due to vocab OOM at training sequence lengths (Qwen3 152k vocab × 32k seq ≈ 40GB logits)

See `PLAN.md` for full GPU selection rationale.

---

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [math-verify (HuggingFace)](https://github.com/huggingface/Math-Verify)
- [Open-R1 Project](https://github.com/huggingface/open-r1)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1)
