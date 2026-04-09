# PLAN.md — Qwen3-1.7B Math RLVR

Stateless execution plan. No pod IDs, cron IDs, ETAs, or "currently running" language.
For active run details, see `STATUS.md`. For historical session notes, see `memory/math-rlvr.md`.

**Convention:** When a pipeline step or task completes, update its status here immediately (✅ / ⏳).
Do not leave stale ⏳ entries for work that is done. PLAN.md is the source of truth on what's done.

---

## Project Goal

Demonstrate distillation + RLVR on math reasoning using Qwen3-1.7B-Base:
1. Distill Qwen3-32B reasoning traces into the 1.7B model via SFT
2. Apply GRPO (group relative policy optimization) with a verifiable math reward
3. Evaluate each phase on MATH-500 to show step-function improvements

Target trajectory: base 31.6% → SFT ~45-55% → GRPO ~85-90% MATH-500 pass@1

---

## Pipeline Overview

```
[0] Data prep (GSM8K + MATH datasets)                         ✅
      ↓
[1] Base eval — GSM8K + MATH-500                              ✅  31.6% pass@1 (⚠️ weak evaluator — rescore with sft_eval.py)
      ↓
[2] Generate Qwen3-32B reasoning traces + rescore             ✅  7,154 correct traces (95.51%)
      ↓
[3] SFT on correct traces (sft_train.py)                      ⏳ pending
      ↓
[3a] SFT eval — MATH-500 (sft_eval.py)                        ⏳ pending  target ~45-55%
      ↓
[4] GRPO training (grpo_train.py)                             ⏳ pending
      ↓
[4a] GRPO eval — MATH-500 (sft_eval.py --model grpo_checkpoint) ⏳ pending  target ~85-90%
      ↓
[5] Comparison table (eval_comparison.py)                          ⏳ pending  aggregates already-computed summaries, no re-inference
```

---

## Training Run Conventions

### General
- Always write checkpoints and logs to absolute paths under `/workspace/qwen3-math-rlvr/`
- Before starting any training run, verify whether a valid checkpoint already exists (`checkpoint-*/model.safetensors`, non-zero size)
- If a valid checkpoint exists, pass `--resume_from_checkpoint auto` (or an explicit checkpoint path)
- All training scripts write stdout/stderr to persistent log files under `/workspace/qwen3-math-rlvr/logs/`

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
| `math500_eval.py` | Legacy MATH-500 eval (weak evaluator); superseded by `sft_eval.py` | ✅ Done (legacy) |
| `sft_train.py` | SFT on 7,154 correct traces; chat template; completion-only loss; file logging; checkpoint auto-resume; `--push_to_hub` | ⏳ Pending |
| `sft_eval.py` | MATH-500 eval with math-verify on any checkpoint; used for base, SFT, and GRPO eval | ⏳ Pending |
| `grpo_train.py` | GRPO with math-verify reward on MATH dataset; starts from sft_checkpoint; `--push_to_hub` | ⏳ Pending |
| `eval_comparison.py` | Comparison aggregator — reads already-computed summary JSONs, prints base→SFT→GRPO table. No inference. | ⏳ Pending |

### Historical scripts (done, not re-run)
- `generate_traces_claude.py` — early Claude-based trace generation, superseded
- `run_rescore.py` — one-off helper script

---

## Open Tasks

- [ ] **Base eval rescore**: Re-score existing `outputs/math500_results.json` with math-verify (responses already stored — no inference needed). Run: `python scripts/rescore_mathverify.py` adapted for that file, or write a quick one-off. Produces math-verify-consistent base number for comparison table.
- [ ] **Review `docs/findings.md`**: May be stale relative to math-verify rescore results. Confirm or update.

---

## Key Findings

### Baseline (Qwen3-1.7B-Base)
- **74.6% GSM8K pass@1** — stronger than expected going in
- **31.6% MATH-500 pass@1, 58.4% pass@8** — large gap signals strong latent capability, ideal GRPO target
- pass@1→pass@8 gap: this model "knows" more than greedy decoding reveals; GRPO should unlock it
- ⚠️ Scored with weak evaluator (`math500_eval.py`); math-verify rescore pending (likely slightly higher)

### Trace Generation Quality
- **Qwen3-32B traces: 95.51% accuracy (7,154/7,490 correct)** after math-verify rescore + 32k token rerun
- Qwen3-32B official MATH-500 score: **96.1%** — our traces are near-ceiling quality
- Original accuracy (5,205/7,490 = 69.5%) was due to broken normalizer and truncated traces — not model quality
- Net gain from fixes: +1,949 correct traces

### Infrastructure / GPU Lessons

#### RunPod Pod Creation
- **Always use `--cloud-type SECURE` first** — preferred. If machine=None after ~5 min, fall back to `--cloud-type COMMUNITY` as last resort (assigns faster but can stall at boot)
- `--cloud-type ALL` only available via GraphQL API directly, not runpodctl CLI
- `--data-center-ids` flag available in runpodctl to target specific regions
- A40 48GB: **fine for eval** (`sft_eval.py`) but **NOT for SFT/GRPO training** (logits OOM at 32k seqlen × 152k vocab)
- For eval-only runs, use A40 (~$0.39/hr) to save cost; reserve A100/H100 80GB for training only

#### Volume and Disk Sizing
- Always provision volume >= 50GB for SFT training; >= 100GB for GRPO (model + optimizer states + multiple checkpoints)
- Write training outputs (checkpoints, logs) to `/workspace` (the volume mount) using absolute paths — container disk does not persist
- Set `save_total_limit=1` in sft_config to avoid accumulating multiple large checkpoints
- Each Qwen3-1.7B checkpoint ≈ 8.8GB (model.safetensors 3.3GB + optimizer.pt 5.6GB); check `df -h` before starting
- Before any training run: verify valid checkpoint exists with `ls checkpoint-*/model.safetensors` and non-zero file size

#### File Logging
- All training scripts write to `logs/<script>_YYYYMMDD_HHMMSS.log` on the volume
- Launch with `nohup python script.py > logs/launch.log 2>&1 &` to capture early crash output

#### Pod Readiness Checks
After `runpodctl pod create`, immediately set a cron (every 2 min, main session) to poll for SSH readiness and launch training. Do NOT use a blocking exec loop to wait — that locks the session.

#### ⚠️ NEVER DELETE A POD WITHOUT BACKUP CONFIRMATION

**This rule is non-negotiable. Deleting a pod destroys its volume permanently.**

Before running `runpodctl pod remove <id>`:
1. Rsync ALL checkpoints and logs to local machine
2. Verify locally: `ls -lh outputs/sft_checkpoint/` — files must exist and be non-zero
3. Verify HF Hub has the latest checkpoint (if `--push_to_hub` was passed)

**If in doubt: stop the pod (preserves volume) instead of removing it.**

Incident (2026-04-09): Subagent removed pod before rsyncing checkpoint-1000 (1000/1275 SFT steps, ~$5 compute lost). Never delegate pod removal to a subagent.

#### Checkpoint Resume
- Before starting any training run, check if a valid checkpoint exists
- Always pass `--resume_from_checkpoint auto` if a valid checkpoint exists
- When value is `"auto"`, HF Trainer finds the latest checkpoint automatically

#### OOM Notes
- **Vocab OOM root cause:** Qwen3's 152k vocabulary → logits tensor ≈ `batch × seq_len × vocab × 2B` = ~40GB at seq_len=32768 with batch=2. Needs 80GB GPU (A100 or H100).
- `paged_adamw_8bit` does NOT fix logit OOM — forward-pass issue, not optimizer memory
- liger-kernel would help but no compatible version for PyTorch 2.4 + transformers 5.5
- COMMUNITY cloud pods can get stuck (uptime=0, never ready) — use SECURE

### Evaluator
- **math-verify** (`pip install 'math-verify[antlr4_13_2]'`) is the authoritative evaluator
- ANTLR4 grammar + SymPy symbolic comparison; handles all LaTeX variants, sets, intervals, matrices
- Critical: wrap bare LaTeX in `$...$` before calling math-verify, or PARSE FAIL on `\dfrac`, `t^7`, etc.
- Use math-verify for all scoring in `sft_eval.py`, `grpo_train.py`, `final_eval.py`

### Token Limit
- **MAX_TOKENS=32768 is the correct standard** — used by DeepSeek-R1, open-r1, Sky-T1
- Our original 16384 caused 249/7,490 truncations (no `\boxed{}` written)

---

## Canonical Data Artifacts

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
| Base | 31.6% ✅ | Measured (weak eval; math-verify rescore pending) |
| Post-SFT | ~45–55% | Target |
| Post-GRPO | ~85–90% | Target |
