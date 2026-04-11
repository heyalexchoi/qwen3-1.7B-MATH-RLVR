# PLAN.md — Qwen3-1.7B Math RLVR

Source of truth for plan, status, and active run details.
For session history and decisions, see `memory/math-rlvr.md`.

**Convention:** Update pipeline status (✅/⏳/🔄) and Current Run section as things change. No stale entries.

---

## Current Run

**Phase:** GRPO training from base model — 🔄 RUNNING (PID 3235, started 2026-04-11 01:19 UTC)

### SFT eval outcome (completed 2026-04-10)

Both SFT checkpoints (1-epoch checkpoint-500 and 3-epoch checkpoint-1275) produce ~0% on MATH-500. Root cause: 1.7B model cannot sustain multi-thousand-token reasoning chains learned from 32B teacher traces (median 3,268 tokens). Epoch count is not the variable — both fail identically. SFT training pipeline was verified correct (no truncation, `<|im_end|>` in loss, proper prompt masking). See README → Key Findings for full analysis.

**Decision:** Skip SFT, run GRPO directly from Qwen3-1.7B-Base.

### Next step: GRPO from base

#### Prompt format: few-shot, no chat template

GRPO must use the same prompt format as the baseline eval, which is the format the base model actually succeeds with. The base model is a **text completion model** — it has never been trained on chat format. The baseline 24.55% was measured with few-shot raw text prompting.

Prompt format (from `math500_eval.py` `create_prompt()`):
```
Problem: What is the value of $2^3 - 3 \cdot 2 + 1$?
Solution: We compute $2^3 - 3 \cdot 2 + 1 = 8 - 6 + 1 = 3$. The answer is $\boxed{3}$.

Problem: If $x + y = 10$ and $x - y = 4$, what is $xy$?
Solution: Adding the equations gives $2x = 14$, so $x = 7$ and $y = 3$. Thus $xy = 7 \cdot 3 = 21$. The answer is $\boxed{21}$.

Problem: A right triangle has legs of length 5 and 12. What is the hypotenuse?
Solution: By the Pythagorean theorem, hypotenuse $= \sqrt{5^2 + 12^2} = \sqrt{25 + 144} = \sqrt{169} = 13$. The answer is $\boxed{13}$.

Problem: How many ways can 3 books be arranged on a shelf from 5 distinct books?
Solution: This is a permutation: $P(5,3) = 5 \cdot 4 \cdot 3 = 60$. The answer is $\boxed{60}$.

Problem: {problem}
Solution:
```

GRPOTrainer supports raw string prompts natively (no chat template applied when `prompt` is a string rather than a list of messages). This means:

- **`load_math_for_grpo`**: change to return `{"prompt": "<few-shot>Problem: {problem}\nSolution:", "answer": ...}` as a plain string, not `[{"role": "user", "content": problem}]`
- **Stop token**: base model eos is `<|endoftext|>` (151643) — correct for text completion. **No `<|im_end|>` override needed** (that was only needed for chat template format)
- **Baseline is directly comparable**: same prompt format, same model, apples-to-apples with existing 24.55% number

#### Extract shared prompting code

`FEW_SHOT` and `create_prompt()` are currently defined inline in `math500_eval.py`. GRPO needs the exact same prompt format. Extract to a shared module so both scripts import from one source of truth:

1. Create `scripts/prompts.py` with `FEW_SHOT` constant and `create_prompt(problem: str) -> str`
2. Update `math500_eval.py` to `from prompts import FEW_SHOT, create_prompt`
3. Update `grpo_train.py` to `from prompts import create_prompt`
4. `sft_eval.py` uses chat template (different format) — leave as-is

#### Code changes needed in `grpo_train.py`

1. **`load_math_for_grpo`**: return `{"prompt": create_prompt(problem), "answer": ...}` — plain string from shared `create_prompt()`, not chat messages
2. Import `create_prompt` from `scripts/prompts.py` (shared with `math500_eval.py`)
3. Remove `format_reward` from `reward_funcs` — correctness only
4. **No eos_token override** — base model's `<|endoftext|>` (151643) is correct for text completion format
5. Keep `pad_token = eos_token` (default for base model — no pad/eos conflict since we're not using chat template)
6. **Skip zero-advantage batches** (see below)

#### Config changes needed in `grpo_config.yaml`

1. `max_completion_length: 2048` (chosen based on measured response distributions across all 4,500 baseline responses: p99=~460 tokens, max=~740 tokens. 2048 is ~4× the uncapped p99. Only 0.2% of baseline responses approached the 1024 cap. Base model does not produce long chain-of-thought. See README → max_completion_length for full analysis.)
2. `learning_rate: 1e-6` (up from 5e-7 — safe zone for zero-RL on small models)
3. `epsilon_high: 0.28` (DAPO asymmetric clipping — more exploration on upside)
4. `temperature: 0.9` (keep — higher than eval temp 0.7 for rollout diversity, standard for GRPO)
5. Verify `loss_type` defaults to `dapo` and `beta` defaults to `0.0` (TRL 1.0.0 defaults, no config entry needed)

#### Zero-advantage batch skipping

When all rollouts for a prompt have the same reward (all correct or all incorrect), advantages = 0, loss = 0, gradient = 0. TRL does **not** skip the forward+backward pass — it computes a zero gradient and wastes the compute. TRL logs `frac_reward_zero_std` as a metric but doesn't act on it.

Compute waste is U-shaped with accuracy:

| Accuracy | P(all same) | Waste |
|----------|-------------|-------|
| 25% (start) | 10% | ~10% of micro-steps |
| 40-60% | ~2% | negligible |
| 80% | 17% | significant |
| 90% (target) | 43% | nearly half |

**Implementation:** Override `compute_loss` in `grpo_train.py` to early-return a detached zero tensor when all advantages in the micro-batch are zero. This skips the backward pass for that micro-batch. With `per_device_train_batch_size=1`, each micro-batch is one prompt's 8 rollouts, so the check is simple: `if (advantages == 0).all(): return torch.tensor(0.0, requires_grad=True)`. Need to verify this doesn't break gradient accumulation step counting in TRL.

#### Early health checks (first 10-50 steps)

Monitor these in wandb/logs to catch problems early:

- **Mean reward at step 0**: should be ~20-25% (base model accuracy with few-shot prompt, matching baseline). If 0% → stop token broken, rollouts running to max length without `\boxed{}`
- **Mean reward trending up**: should see signal within 50-100 steps. If flat after 200 steps → LR too low or reward signal too sparse
- **Completion lengths**: should be well under 2048 initially. If all hitting 2048 → stop token not working
- **`frac_reward_zero_std`**: should be ~10% at start (per table above). If much higher → reward too sparse, consider larger group size
- **Clip ratios**: high_clip_ratio should be nonzero (model is exploring), not saturated (>0.5 means too aggressive)

#### Pre-launch verification (REQUIRED — all three checks must pass before training)

Run in order. ~35 minutes total on pod.

```bash
cd /workspace/qwen3-math-rlvr

# Check 0: Trainer instantiation smoke test (~2 min, GPU)
# Confirms GRPOConfig accepts all params (epsilon_high, loss_type, beta, etc.) and
# the trainer starts without error. Read the resolved-config dump in the log and
# verify: epsilon_high=0.28, beta=0.0, loss_type=dapo, lr=1e-6,
# lr_scheduler_type=constant_with_warmup, warmup_ratio=0.05, max_completion_length=2048.
# Also confirm tokenizer.name_or_path matches --model (not a stale cached variant).
python scripts/grpo_train.py --model Qwen/Qwen3-1.7B-Base --max_steps 1 2>&1 | tee logs/smoke_test.log
# → no TypeError/ValueError at startup; config dump looks correct; exits cleanly

# Check 1: Reward function parity with baseline eval scorer (~1 min, CPU-only)
# Confirms GRPO reward function agrees with math-verify on the 4000 baseline rollouts.
# Pass criteria: 0 mismatches on correct rollouts, 0 false positives on incorrect.
python scripts/check_reward_parity.py
# → PASS / FAIL + first 5 mismatches if any

# Check 2: Stop token and termination sanity (~25 min, requires GPU + model)
# Confirms base-model rollouts terminate cleanly under GRPO prompting/stop config.
# Pass criteria: >=18/20 stop via EOS, >=18/20 have exactly 1 \boxed{}, 0 junk.
python scripts/check_rollout_termination.py
# → PASS / FAIL + diagnostics + rollouts saved to outputs/check_rollout_termination.json
```

**All three must pass before launching.** Any failure → fix and re-run that check only.
Do not start the real run "in parallel" with debugging.

#### Launch

```bash
# On pod — only after all three checks pass
cd /workspace/qwen3-math-rlvr
nohup python scripts/grpo_train.py --model Qwen/Qwen3-1.7B-Base --push_to_hub \
  > logs/grpo_launch.log 2>&1 &
echo "PID: $!"
```

See README → "GRPO from base: approach and parameters" for full parameter justifications.

### Pod

> **Before any pod operation** (create, launch, rsync, stop, remove): read `docs/runpod.md` first. It has the correct SSH key path, launch command with `set -a` secrets sourcing, pre-removal checklist, and known gotchas.

Pod `gol7yudqrlfn48` — GRPO training running, started 2026-04-11.
- **H100 SXM 80GB, $2.99/hr** — `root@64.247.201.44 -p 17495`
- torch 2.6.0 (upgraded from 2.4.1), TRL 1.0.0, math-verify installed
- **~937 steps, ~2 hours** at 8s/step
- wandb: https://wandb.ai/heyalexchoi/qwen3-math-rlvr/runs/ckz7jwil

### Output files (on pod)

- `outputs/grpo_checkpoint/` — model checkpoint
- `logs/grpo_train_TIMESTAMP.log` — training log
- HF Hub: `heyalexchoi/qwen3-1.7b-math-grpo` (pushed via `--push_to_hub`)

### On completion

1. Rsync GRPO checkpoint to local (see `docs/runpod.md` → Rsync Outputs)
2. Stop training pod: `PATH=$HOME/.local/bin:$PATH runpodctl pod stop gol7yudqrlfn48` — **STOP only, never remove**
3. Eval GRPO checkpoint on a separate eval pod (see step [4a] below)
4. Update pipeline step [4] and [4a] with results

---

## Step [4a]: GRPO Eval Plan

Eval the GRPO checkpoint on a fresh A40 or L40S pod (48GB — fine for Qwen3-1.7B inference).

### Why a separate pod

Training pod is H100 at $2.99/hr. L40S or A40 is ~$0.39/hr and 48GB is sufficient for inference. Spin up, eval, stop.

### Which eval script

**Use `math500_eval.py`** — NOT `sft_eval.py`, NOT `baseline_eval.py`.

| Script | Prompt format | Evaluator | Notes |
|--------|--------------|-----------|-------|
| `math500_eval.py` | few-shot (`create_prompt` from `prompts.py`) | **math-verify** (ANTLR4/SymPy) | ✅ Matches GRPO training format + authoritative evaluator |
| `sft_eval.py` | chat template | math-verify | ❌ Format mismatch — GRPO trained on few-shot, not chat |
| `baseline_eval.py` | few-shot (GSM8K-specific) | regex | ❌ Wrong dataset (GSM8K, not MATH-500) |

The GRPO model was trained with few-shot raw text prompts (`Problem: ...\nSolution:`). Evaluating with chat template would change the input distribution. `math500_eval.py` uses the same `create_prompt()` from `prompts.py` as training — apples-to-apples. `math500_eval.py` now uses math-verify as its primary evaluator (added inline — same as `sft_eval.py`, with regex fallback if math-verify not installed).

### Inference parameters

`math500_eval.py` defaults (appropriate — do not override unless noted):

| Param | Value | Source |
|-------|-------|--------|
| `--temperature` | 0.7 | Qwen3 non-thinking mode recommendation; matches baseline generation |
| `--n_samples` | 8 | pass@8 sampling (same run does both pass@1 greedy + pass@8) |
| `--max_new_tokens` | **2048** | ← override from default 1024 (`--max_new_tokens 2048`) |

Greedy pass@1 is run automatically inside the same script (no separate invocation needed — `math500_eval.py` runs greedy + sampling in one pass).

**vLLM not needed here.** At max_new_tokens=2048 on a 1.7B model, the GRPO checkpoint generates ~150-250 tokens on average (it never learned long chain-of-thought). HF backend on L40S runs the full eval in ~10-15 min. The "540s/problem HF is too slow" warning in the README applies to SFT eval at 8192 tokens with Qwen3 thinking-mode — not applicable here. If you add longer evals later, revisit.

### Eval pod setup

```bash
# 1. Create L40S pod (preferred — faster than A40 at same VRAM; use A40 if L40S unavailable)
runpodctl pod create \
  --name "clawd-grpo-eval" \
  --gpu-id "NVIDIA L40S" \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --container-disk-in-gb 30 \
  --volume-in-gb 30 \
  --volume-mount-path "/workspace" \
  --ports "22/tcp" \
  --cloud-type SECURE \
  -o json

# 2. Install deps (no torch upgrade needed — inference only, no TRL FSDPModule)
ssh ... "cd /workspace/qwen3-math-rlvr && pip install transformers datasets torch tqdm math-verify[antlr4_13_2] -q"

# 3. Rsync GRPO checkpoint + scripts to eval pod
rsync -av -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/ \
  root@<IP>:/workspace/qwen3-math-rlvr/ \
  --exclude='outputs/sft_*' --exclude='__pycache__/' --exclude='.git/'

# 4. Copy GRPO checkpoint (if not already rsynced)
rsync -av -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/outputs/grpo_checkpoint/ \
  root@<IP>:/workspace/qwen3-math-rlvr/outputs/grpo_checkpoint/
```

### Launch eval

```bash
ssh ... << 'EOF'
cd /workspace/qwen3-math-rlvr
nohup python scripts/math500_eval.py \
  --model outputs/grpo_checkpoint \
  --max_new_tokens 2048 \
  --n_samples 8 \
  --temperature 0.7 \
  --output outputs/grpo_math500_results.json \
  > logs/grpo_eval.log 2>&1 &
echo "PID: $!"
EOF
```

Runtime estimate: 500 problems × 8 samples × ~1s/sample on A40 ≈ ~70 min. Monitor via `tail -f logs/grpo_eval.log`.

### Rsync results back + stop pod

```bash
rsync -avz -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p <PORT>" \
  root@<IP>:/workspace/qwen3-math-rlvr/outputs/grpo_math500_results.json \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/outputs/

runpodctl pod stop <EVAL_POD_ID>  # stop — never remove
```

### Metrics to report

After eval run, execute `rescore_math500.py` on the output to get inferred pass@1 (c/n):
```bash
python scripts/rescore_math500.py \
  --input outputs/grpo_math500_results.json \
  --output outputs/grpo_math500_mv_rescored.json
```

From `outputs/grpo_math500_mv_rescored.json` → `summary`:
- `pass1_greedy` — greedy pass@1
- `pass8` — pass@8
- `pass1_inferred_cn` — inferred pass@1 (Chen et al. 2021 c/n estimator from pass@8 samples)
- `pass1_by_level` and `pass8_by_level` — breakdown by difficulty
- Compare to baseline: 35.80% greedy / 65.00% pass@8 / 24.55% inferred pass@1 (all math-verify)

Then upload both eval outputs to HF (see Canonical Data Artifacts → HF artifact upload).

---

## Project Goal

Demonstrate distillation + RLVR on math reasoning using Qwen3-1.7B-Base:
1. Distill Qwen3-32B reasoning traces into the 1.7B model via SFT
2. Apply GRPO (group relative policy optimization) with a verifiable math reward
3. Evaluate each phase on MATH-500 to show step-function improvements

Target trajectory: base 24.55% (inferred c/n, sampling) → GRPO ~85-90% MATH-500 pass@1 (SFT skipped — degenerate)

---

## Pipeline Overview

```
[0] Data prep (GSM8K + MATH datasets)                         ✅
      ↓
[1] Base eval — GSM8K + MATH-500                              ✅  24.55% pass@1 inferred (c/n) / 35.8% greedy / 65.0% pass@8
         math500_eval.py → math500_results.json → rescore_math500.py → baseline_math500_mv_rescored.json
      ↓
[2] Generate Qwen3-32B reasoning traces + rescore             ✅  7,154 correct traces (95.51%)
      ↓
[3] SFT on correct traces (sft_train.py)                      ✅ done — 06:21 UTC 2026-04-10
      ↓
[3a] SFT eval — MATH-500 (sft_eval.py)                        ✅ ~0% — both checkpoints degenerate (capacity, not training bug)
      ↓
[4] GRPO from base model (grpo_train.py)                      🔄 RUNNING — PID 3235, wandb ckz7jwil, ~2hr ETA
      ↓
[4a] GRPO eval — MATH-500 (sft_eval.py --model grpo_checkpoint) ⏳ pending  target ~85-90%
      ↓
[5] Comparison table (eval_comparison.py)                          ⏳ pending  base vs GRPO (no SFT — degenerate)
```

---

## Training Run Conventions

### General
- Always write checkpoints and logs to absolute paths under `/workspace/qwen3-math-rlvr/`
- Before starting any training run, verify whether a valid checkpoint already exists (`checkpoint-*/model.safetensors`, non-zero size)
- If a valid checkpoint exists, pass `--resume_from_checkpoint auto` (or an explicit checkpoint path)
- All training scripts write stdout/stderr to persistent log files under `/workspace/qwen3-math-rlvr/logs/`

### Volume Sizing
- **SFT training:** 50GB volume — SFT checkpoint ~10GB, brief coexistence of 2 during rotation = ~20GB peak, 50GB gives margin
- **GRPO training:** 50GB volume — same as SFT: start from SFT checkpoint (10GB) + 1 GRPO checkpoint (10GB) + brief coexistence window = ~32GB peak
- Write all outputs to `/workspace` (volume mount) using absolute paths — container disk does not persist
- Check `df -h` before starting; if already near limit, clear old checkpoints first

### HuggingFace Hub Push (required)
Both `sft_train.py` and `grpo_train.py` support `--push_to_hub`. **Always pass this flag** — it provides an off-pod backup via `hub_strategy="checkpoint"` (pushes after each checkpoint save, not just at the end).
- SFT → `heyalexchoi/qwen3-1.7b-math-sft`
- GRPO → `heyalexchoi/qwen3-1.7b-math-grpo`
- Requires `HF_TOKEN` env var (stored in `~/.config/openclaw/secrets.env`)
- Before removing a pod, verify HF repo has the latest checkpoint as a secondary backup check

---

## Script Reference

Scripts live in `scripts/`. No numeric prefixes — names are descriptive.

| Script | What it does | Status |
|--------|-------------|--------|
| `prepare_data.py` | Download and format GSM8K and MATH datasets | ✅ Done |
| `baseline_eval.py` | GSM8K pass@1 eval on Qwen3-1.7B-Base | ✅ Done |
| `generate_traces_32b.py` | Generate 7,490 MATH reasoning traces via Qwen3-32B (OpenRouter) | ✅ Done |
| `rescore_traces.py` | Regex normalizer fix; historical artifact, superseded by math-verify | ✅ Done |
| `rescore_mathverify.py` | Rescore any trace JSONL with math-verify (ANTLR4/SymPy) | ✅ Done |
| `rerun_truncated.py` | Re-run traces truncated at 16k tokens at 32k | ✅ Done |
| `math500_eval.py` | **Step 1**: generates MATH-500 responses (few-shot prompts, HF backend). Output: `outputs/math500_results.json`. Used for baseline and GRPO eval | ✅ Active |
| `rescore_math500.py` | **Step 2**: scores `math500_results.json` with math-verify → `outputs/baseline_math500_mv_rescored.json`. Canonical source of 35.8% / 65.0% / 24.55% numbers | ✅ Active |
| `sft_train.py` | SFT on 7,154 correct traces; chat template; completion-only loss; file logging; checkpoint auto-resume; `--push_to_hub` | ✅ Done |
| `sft_eval.py` | MATH-500 eval with math-verify on any checkpoint; vLLM primary backend, HF fallback | ✅ Done (SFT ~0%) |
| `check_reward_parity.py` | Pre-GRPO check 1: reward function parity with baseline eval scorer — CPU, ~1 min | ✅ Done — run before launching GRPO |
| `check_rollout_termination.py` | Pre-GRPO check 2: stop token and rollout termination sanity — GPU, ~25 min | ✅ Done — run before launching GRPO |
| `grpo_train.py` | GRPO with math-verify reward on MATH dataset; starts from base model; `--push_to_hub` | ⏳ Ready — pre-launch checks required, then launch |
| `eval_comparison.py` | Comparison aggregator — reads already-computed summary JSONs, prints base→SFT→GRPO table. No inference. | ⏳ Pending |

### Historical scripts (done, not re-run)
- `generate_traces_claude.py` — early Claude-based trace generation, superseded
- `run_rescore.py` — one-off helper script

---

## Open Tasks

- [x] **Base eval rescore**: Done — 35.8% greedy pass@1 (was 31.6%). math-verify authoritative.
- [x] **GRPO `max_completion_length`**: Updated 4096 → 8192. 4096 truncates nearly all SFT model rollouts → zero reward signal.
- [x] **Baseline inferred pass@1**: Computed from existing pass@8 data via math-verify c/n — **24.55%**. Apples-to-apples with SFT eval. Update README + results tables.
- [x] **docs/findings.md**: Already deleted — content merged into README.
- [x] **Greedy eval batching**: superseded — vLLM handles scheduling natively. HF batched fallback still present.
- [x] **Fix SFT checkpoint tokenizer eos_token + grpo_train.py stop tokens**: Tokenizer saved correctly — `eos_token=<|im_end|>` (151645), `pad_token=<|endoftext|>` (151643). `generation_config.json` has `eos_token_id: 151645`. Both local checkpoint and HF Hub updated. No code overrides needed.
- [x] **Re-run baseline eval with higher `max_new_tokens` cap**: RESOLVED — not needed. Measured all 4,500 baseline responses: p99=597 words (~460 tokens), max=963 words (~740 tokens). Only 7/4,500 (0.2%) hit the 1024-token limit. Base model does not produce long chain-of-thought. Cap is not binding; baseline numbers are valid as-is.

→ See README → Key Findings for baseline numbers, trace quality, GPU/OOM notes, eval methodology, and decision rationale.

## Canonical Data Artifacts

All eval outputs live in `outputs/` (gitignored). Durable copies are uploaded to HF dataset repo: [`heyalexchoi/qwen3-math-rlvr-results`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-rlvr-results).

### Eval output files

| File | Script | Status | Description |
|------|--------|--------|-------------|
| `outputs/math500_results.json` | `math500_eval.py` | ✅ Exists locally | 500 problems × (1 greedy + 8 sampling responses), baseline, Qwen3-1.7B-Base |
| `outputs/baseline_math500_mv_rescored.json` | `rescore_math500.py` | ✅ Exists locally | Baseline re-scored with math-verify: 35.80% / 65.00% / 24.55% |
| `outputs/grpo_math500_results.json` | `math500_eval.py` | ⏳ Pending eval run | GRPO checkpoint responses |
| `outputs/grpo_math500_mv_rescored.json` | `rescore_math500.py` | ⏳ Pending eval run | GRPO checkpoint re-scored with math-verify |

### HF artifact upload (after eval)

```bash
# Create dataset repo (one-time)
huggingface-cli repo create qwen3-math-rlvr-results --type dataset

# Upload baseline artifacts (do now — these exist)
huggingface-cli upload heyalexchoi/qwen3-math-rlvr-results \
  outputs/math500_results.json outputs/math500_results.json
huggingface-cli upload heyalexchoi/qwen3-math-rlvr-results \
  outputs/baseline_math500_mv_rescored.json outputs/baseline_math500_mv_rescored.json

# Upload GRPO artifacts (after eval run completes)
huggingface-cli upload heyalexchoi/qwen3-math-rlvr-results \
  outputs/grpo_math500_results.json outputs/grpo_math500_results.json
huggingface-cli upload heyalexchoi/qwen3-math-rlvr-results \
  outputs/grpo_math500_mv_rescored.json outputs/grpo_math500_mv_rescored.json
```

### Training data files

| File | Description |
|------|-------------|
| `data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl` | **SFT-ready** — 7,490 entries, 7,154 marked correct=true |
| `data/math_train.jsonl` | MATH train set (EleutherAI/hendrycks_math merged) |
| `data/gsm8k/train.jsonl` | GSM8K train set (7,473 problems) |
| `data/gsm8k/test.jsonl` | GSM8K test set (1,319 problems) |

---

## Performance Targets

| Phase | MATH-500 pass@1 | Notes |
|-------|----------------|-------|
| Base | 24.55% pass@1 (inferred c/n) / 35.8% pass@1 (greedy) / 65.0% pass@8 ✅ | math-verify; `outputs/baseline_math500_mv_rescored.json` |
| Post-SFT | ~0% | Degenerate — both 1-epoch and 3-epoch checkpoints. Skipped. |
| Post-GRPO | ~85–90% | Target — GRPO from base, skipping SFT |
