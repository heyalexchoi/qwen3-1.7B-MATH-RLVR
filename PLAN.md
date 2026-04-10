# PLAN.md — Qwen3-1.7B Math RLVR

Source of truth for plan, status, and active run details.
For session history and decisions, see `memory/math-rlvr.md`.

**Convention:** Update pipeline status (✅/⏳/🔄) and Current Run section as things change. No stale entries.

---

## Current Run

**Phase:** SFT eval — 🔄 RUNNING

- SFT training completed 06:21 UTC 2026-04-10, 1275/1275 steps
- Checkpoint: `outputs/sft_checkpoint/model.safetensors` (local + HF Hub: `heyalexchoi/qwen3-1.7b-math-sft`)
- Eval pod: `zrszn3f053jgcj` (A40, 194.68.245.64:22047), PID 1064
- Command: `sft_eval.py --model outputs/sft_checkpoint --max_new_tokens 32768 --n_samples 8`
- Monitor cron: `9acabe61` every 30 min → rsync + stop pod on completion
- `sft_eval.py` patched 2026-04-10: incremental JSONL flush + resume support

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
[1] Base eval — GSM8K + MATH-500                              ✅  35.8% pass@1 (math-verify rescored)
      ↓
[2] Generate Qwen3-32B reasoning traces + rescore             ✅  7,154 correct traces (95.51%)
      ↓
[3] SFT on correct traces (sft_train.py)                      ✅ done — 06:21 UTC 2026-04-10
      ↓
[3a] SFT eval — MATH-500 (sft_eval.py)                        🔄 RUNNING   target ~45-55%
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
| `sft_train.py` | SFT on 7,154 correct traces; chat template; completion-only loss; file logging; checkpoint auto-resume; `--push_to_hub` | ✅ Done |
| `sft_eval.py` | MATH-500 eval with math-verify on any checkpoint; used for base, SFT, and GRPO eval | 🔄 Running |
| `grpo_train.py` | GRPO with math-verify reward on MATH dataset; starts from sft_checkpoint; `--push_to_hub` | ✅ Done |
| `eval_comparison.py` | Comparison aggregator — reads already-computed summary JSONs, prints base→SFT→GRPO table. No inference. | ⏳ Pending |

### Historical scripts (done, not re-run)
- `generate_traces_claude.py` — early Claude-based trace generation, superseded
- `run_rescore.py` — one-off helper script

---

## Open Tasks

- [x] **Base eval rescore**: Done — 35.8% (was 31.6%). math-verify is strictly better: 21 flipped correct, 0 wrong.
- [x] **GRPO `max_completion_length`**: Updated 4096 → 8192. 4096 truncates nearly all SFT model rollouts (trained on 32k traces) → zero reward signal. 8192 gives room for reasoning while fitting A100 80GB (8 × 8192 × 152k vocab × 2B ≈ 20GB logits during training backward).
- [ ] **Review `docs/findings.md`**: May be stale relative to math-verify rescore results. Confirm or update.
- [ ] **Fix SFT checkpoint tokenizer eos_token + grpo_train.py stop tokens**: SFTTrainer saved the tokenizer with `eos_token=<|endoftext|>` (151643) instead of `<|im_end|>` (151645), even though `sft_config.yaml` set `eos_token: "<|im_end|>"`. The checkpoint is inconsistent — model generates `<|im_end|>` to end turns but tokenizer doesn't reflect this. Fix: (1) investigate correct solution (should tokenizer.eos_token be `<|im_end|>`, or should generation_config.json have both 151643+151645 as eos_token_id list, or both?), (2) apply fix to local checkpoint (`outputs/sft_checkpoint`) and push to HF Hub (`heyalexchoi/qwen3-1.7b-math-sft`), (3) add defensive stop token handling to `grpo_train.py` (same pattern as `sft_eval.py`: build `stop_token_ids = {eos_token_id} \u222a {<|im_end|>_id}`, pass via GRPOConfig generation_config). Must be done before GRPO training.

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

#### Training Launch Monitoring Strategy
After launching training, check frequently at first to confirm it's actually running, then back off:
- **30s** — confirm process is alive (`ps aux | grep sft_train`)
- **1 min** — confirm first step completed without OOM (check log for `1/1275` or similar)
- **5 min** — confirm a few steps in, note step time
- **15 min** — optional sanity check
- **30 min recurring** — normal cron monitor cadence

Use cron at 30 min for the recurring monitor. The early checks (30s–5min) can be manual SSHs or a short-lived cron that self-removes after confirming training. Do NOT block the session with exec sleep loops.

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
- **Vocab OOM root cause:** Qwen3's 152k vocabulary → logits tensor ≈ `batch × seq_len × vocab × 2B`. At batch=2, seq=32768 this is ~40GB — OOM on A100. batch=1 brings it to ~9.3GB.
- **Working config:** batch=1, grad_accum=16, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in env at launch
- `expandable_segments:True` lets CUDA expand existing memory segments rather than requiring a contiguous block — prevents fragmentation OOM on long sequences even when total free memory is sufficient
- `paged_adamw_8bit` does NOT fix logit OOM — forward-pass issue, not optimizer memory
- **liger-kernel 0.7.0 is NOT compatible on PyTorch 2.4:** rms_norm patch requires `torch.distributed.tensor.DTensor` (PyTorch 2.5+ only) → crashes at step ~40. Do not use.
- COMMUNITY cloud pods can get stuck (uptime=0, never ready) — use SECURE

#### Disk Notes
- One Qwen3-1.7B checkpoint ≈ 10GB (model.safetensors 3.3GB + optimizer.pt 6.8GB)
- `save_total_limit=1` but there's a brief window where two checkpoints coexist (new written before old deleted)
- Original 20GB volume hit disk-full at step 1000 (checkpoint-500 + checkpoint-1000 briefly coexisted)
- **Use 100GB volume** to give plenty of room

### Cron Jobs — Routing to This Topic

To send a monitoring cron's output to the MATH RLVR Telegram topic (176), use `isolated` + `delivery.to`:

```json
{
  "sessionTarget": "isolated",
  "payload": {
    "kind": "agentTurn",
    "message": "Check pod progress via SSH and report N/500 complete...",
    "toolsAllow": ["exec", "read", "write", "edit"]
  },
  "delivery": {
    "mode": "announce",
    "channel": "telegram",
    "to": "-1003722115096:topic:176"
  }
}
```

`sessionTarget: "main"` with `sessionKey` does NOT route replies to the topic — it only controls which session processes the event. Always use `isolated` + `delivery.to` for topic-bound crons.

---

### Qwen Thinking Mode: Greedy Decoding Causes Infinite Repetition Loops

**Finding:** Qwen3 thinking-mode models must NOT use greedy decoding (`do_sample=False`, `temperature=0.0`). With greedy, the `<think>` block enters circular reasoning loops that fill the entire token budget without ever producing a `\boxed{}` answer. The model does eventually stop (stop tokens work), but at the token limit with empty output.

This is documented in Qwen's own official inference guide and is a fundamental property of thinking-mode models trained on long CoT traces.

**Recommended Qwen3 thinking-mode inference settings (official):**
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `min_p=0`
- **DO NOT use greedy decoding**

**Impact on our eval:**
Our first `sft_eval.py` launch used `--max_new_tokens 32768` with greedy pass@1. Each problem took ~25 minutes and produced useless output. Killed after 1/500 problems.

**Fix applied:** Replaced greedy pass@1 with Qwen recommended sampling (`temperature=0.6, top_p=0.95, top_k=20`). We run `n_samples=8` at this temperature, then compute both:
- **pass@8**: fraction of problems where at least 1 of 8 samples is correct
- **pass@1 (unbiased estimate)**: `c/n` where c = correct samples, n = 8 — this is the standard unbiased estimator from the Codex paper (Chen et al. 2021). For k=1, pass@1 = 1 − (n−c)/n = c/n.

This is actually a better measurement than greedy pass@1 since it reflects the model's realistic inference distribution.

### Eval Script — Critical Rules
- **Train/eval format must match.** `sft_eval.py` MUST use `tokenizer.apply_chat_template()` — the same Qwen3 chat format SFTTrainer applies during training. Using a plain few-shot prompt ("Problem:/Solution:") causes total format mismatch: the model never sees its expected context tokens and enters an infinite repetition loop at greedy decoding until `max_new_tokens` is hit.
- **Always set `eos_token: "<|im_end|>"` in SFTConfig when fine-tuning a Qwen base model with chat template.** TRL docs explicitly require this: without it, the saved `generation_config.json` won't include `<|im_end|>` as a stop token and inference won't terminate correctly. Our `sft_config.yaml` now has this. The existing checkpoint's `generation_config.json` has been manually patched to add 151645.
- **SFT checkpoint tokenizer has `eos_token=<|endoftext|>` (151643), not `<|im_end|>` (151645).** SFTTrainer saves the tokenizer from the base model config. The model learns to emit `<|im_end|>` to end assistant turns (from the chat template), but `tokenizer.eos_token_id` won't include it. Always build stop tokens as `{eos_token_id} ∪ {<|im_end|>_id}` using `tokenizer.convert_tokens_to_ids("<|im_end|>")` — no hardcoded IDs.
- **Chat template is in `chat_template.jinja`** (transformers 5.x saves it separately from `tokenizer_config.json`). `AutoTokenizer.from_pretrained()` loads it automatically — `tokenizer.apply_chat_template()` works correctly.
- **Do not truncate eval inputs.** MATH-500 prompts after chat template formatting are ~300–800 tokens. Setting `max_length` on the tokenizer call is unnecessary and masks real issues.

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
| Base | 35.8% ✅ | math-verify rescored (was 31.6% with weak eval; +21 flipped correct, 0 wrong) |
| Post-SFT | ~45–55% | Target |
| Post-GRPO | ~85–90% | Target |
