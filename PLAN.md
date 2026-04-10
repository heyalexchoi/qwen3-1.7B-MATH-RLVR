# PLAN.md — Qwen3-1.7B Math RLVR

Source of truth for plan, status, and active run details.
For session history and decisions, see `memory/math-rlvr.md`.

**Convention:** Update pipeline status (✅/⏳/🔄) and Current Run section as things change. No stale entries.

---

## Current Run

**Phase:** SFT eval — 🔄 RUNNING (sampling pass@8)

- SFT training completed 06:21 UTC 2026-04-10, 1275/1275 steps, 3 epochs
- Checkpoint: `outputs/sft_checkpoint/model.safetensors` (local + HF Hub: `heyalexchoi/qwen3-1.7b-math-sft`)
- Eval pod: `zrszn3f053jgcj` — A40 48GB, `root@194.68.245.64 -p 22047`, $0.44/hr

### Eval plan (two runs, sequential)

1. **Sampling eval** (~13h): `bash scripts/run_eval.sh` ← **RUNNING NOW**
   - Output: `outputs/sft_eval_results.jsonl` + `sft_eval_results_summary.json`
   - Gives: pass@8 + inferred pass@1 (c/n, n=8, temp=0.6)
   - PID 3601, started 15:36 UTC, log: `logs/sft_eval_20260410_153621.log`
2. **Rsync outputs/** from pod to local
3. **Greedy eval** (~2h with batching): `bash scripts/run_eval.sh --greedy`
   - Output: `outputs/sft_eval_greedy_results.jsonl` + `sft_eval_greedy_summary.json`
   - Gives: greedy pass@1 (n=1, temp=0.0, max_new_tokens=4096)
   - Batched: 8 problems/generate() call (left-padding) — script updated ✅
4. **Rsync outputs/** from pod to local (again, after greedy)
5. Stop pod: `runpodctl pod stop zrszn3f053jgcj` — **STOP only, never remove**
6. Report both greedy pass@1 + inferred pass@1 (c/n) + pass@8 from summary JSONs
7. Update pipeline step [3a] with results
8. Remove monitor cron

**Monitor cron:** `22deb51e` — fires every 30 min → topic 176

### Check progress

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@194.68.245.64 -p 22047 \
  "wc -l /workspace/qwen3-math-rlvr/outputs/sft_eval_results.jsonl \
         /workspace/qwen3-math-rlvr/outputs/sft_eval_greedy_results.jsonl; \
   tail -5 /workspace/qwen3-math-rlvr/logs/03a_sft_eval_*.log | tail -20"
```

### Output files (on pod)

- `outputs/sft_eval_greedy_results.jsonl` — greedy eval, 1 row/problem
- `outputs/sft_eval_greedy_summary.json` — written at completion
- `outputs/sft_eval_results.jsonl` — sampling eval, 1 row/sample (8 per problem)
- `outputs/sft_eval_results_summary.json` — written at completion
- `logs/03a_sft_eval_TIMESTAMP.log` — live log

### On completion (both evals done)

1. Rsync `outputs/` to local
2. `PATH=$HOME/.local/bin:$PATH runpodctl pod stop zrszn3f053jgcj` — **STOP only, never remove**
3. Report greedy pass@1 + inferred pass@1 (c/n) + pass@8 from summary JSONs
4. Update pipeline step [3a] with results
5. Remove monitor cron (set on relaunch)

---

## Project Goal

Demonstrate distillation + RLVR on math reasoning using Qwen3-1.7B-Base:
1. Distill Qwen3-32B reasoning traces into the 1.7B model via SFT
2. Apply GRPO (group relative policy optimization) with a verifiable math reward
3. Evaluate each phase on MATH-500 to show step-function improvements

Target trajectory: base 24.55% (inferred c/n, sampling) → SFT ~45-55% → GRPO ~85-90% MATH-500 pass@1

---

## Pipeline Overview

```
[0] Data prep (GSM8K + MATH datasets)                         ✅
      ↓
[1] Base eval — GSM8K + MATH-500                              ✅  24.55% pass@1 inferred (c/n) / 35.8% greedy / 65.0% pass@8
      ↓
[2] Generate Qwen3-32B reasoning traces + rescore             ✅  7,154 correct traces (95.51%)
      ↓
[3] SFT on correct traces (sft_train.py)                      ✅ done — 06:21 UTC 2026-04-10
      ↓
[3a] SFT eval — MATH-500 (sft_eval.py)                        ⏳ IDLE       target ~45-55% (greedy + inferred c/n pass@1 + pass@8)
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
| `math500_eval.py` | Legacy MATH-500 eval (weak evaluator); superseded by `sft_eval.py` | ✅ Done (legacy) |
| `sft_train.py` | SFT on 7,154 correct traces; chat template; completion-only loss; file logging; checkpoint auto-resume; `--push_to_hub` | ✅ Done |
| `sft_eval.py` | MATH-500 eval with math-verify on any checkpoint; used for base, SFT, and GRPO eval | 🔄 Running |
| `grpo_train.py` | GRPO with math-verify reward on MATH dataset; starts from sft_checkpoint; `--push_to_hub` | ✅ Done |
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
- [x] **Greedy eval batching**: `sft_eval.py --greedy` now batches 8 problems/generate() call with left-padding. ~2-4h instead of ~16h.
- [x] **Fix SFT checkpoint tokenizer eos_token + grpo_train.py stop tokens**: Tokenizer saved correctly — `eos_token=<|im_end|>` (151645), `pad_token=<|endoftext|>` (151643). `generation_config.json` has `eos_token_id: 151645`. Both local checkpoint and HF Hub updated. No code overrides needed.

→ See README → Key Findings for baseline numbers, trace quality, GPU/OOM notes, eval methodology, and decision rationale.

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
| Base | 24.55% pass@1 (inferred c/n) / 35.8% pass@1 (greedy) / 65.0% pass@8 ✅ | math-verify; `outputs/baseline_math500_mv_rescored.json` |
| Post-SFT | ~45–55% (greedy pass@1) + ~45–55% (inferred c/n) + pass@8 | Target; report both pass@1 forms for comparability |
| Post-GRPO | ~85–90% | Target |
