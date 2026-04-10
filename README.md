# Qwen3-1.7B Math RLVR

Demonstrating distillation + RLVR on math reasoning with Qwen3-1.7B-Base.

→ **Full pipeline plan, findings, and decisions:** [`PLAN.md`](PLAN.md)
→ **Active run / current phase:** [`STATUS.md`](STATUS.md)
→ **Session log:** [`memory/math-rlvr.md`](../memory/math-rlvr.md)

---

## Pipeline Overview

```
[0] Data prep (GSM8K + MATH)                           ✅
[1] Base eval — GSM8K + MATH-500                       ✅  35.8% pass@1 / 65.0% pass@8 (math-verify)
[2] Generate Qwen3-32B traces                          ✅  7,154 correct traces (95.51%)
[3] SFT on correct traces                              ✅
[3a] SFT eval — MATH-500                               ⏳  target ~45-55% pass@1
[4] GRPO training                                      ⏳  target ~85-90%
[5] Final eval — base vs SFT vs GRPO                   ⏳
```

---

## Results

| Phase | MATH-500 pass@1 | pass@8 | Notes |
|-------|----------------|--------|-------|
| Base | **24.55%** (inferred c/n) / 35.8% (greedy) ✅ | **65.0%** ✅ | math-verify; `outputs/baseline_math500_mv_rescored.json` |
| Post-SFT | ~45–55% | — | Target |
| Post-GRPO | ~85–90% | — | Target |

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

### Baseline methodology note

The original baseline (`baseline_math500_mv_rescored.json`) ran pass@1 as a single greedy sample and pass@8 as 8 sampling runs. The inferred pass@1 = **24.55%** is computed post-hoc via math-verify c/n on the stored pass@8 responses — no re-run needed. This is the correct comparison point for SFT/GRPO evals which use n_samples=8 at temperature=0.6.

### Eval script: train/eval format must match

`sft_eval.py` must use `tokenizer.apply_chat_template()` — the same format SFTTrainer applied during training. Using a plain prompt causes total format mismatch and infinite loops. Stop tokens must include both `<|endoftext|>` (151643) and `<|im_end|>` (151645) — build dynamically via `tokenizer.convert_tokens_to_ids()`, never hardcode.

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
| `math500_eval.py` | ✅ Done (legacy) | Superseded by `sft_eval.py` |
| `generate_traces_32b.py` | ✅ Done | 7,490 MATH traces at MAX_TOKENS=32768 |
| `rescore_traces.py` | ✅ Done (legacy) | Regex normalizer; superseded by math-verify |
| `rescore_mathverify.py` | ✅ Done | math-verify rescore; **95.51% (7,154/7,490)** |
| `rerun_truncated.py` | ✅ Done | 244 truncated reruns; 160 newly correct |
| `sft_train.py` | ✅ Done | SFT on 7,154 correct MATH traces |
| `sft_eval.py` | ⏳ Pending | See `PLAN.md` |
| `grpo_train.py` | ⏳ Pending | See `PLAN.md` |
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

Create `~/.config/openclaw/secrets.env` (mode 600):
```
HF_TOKEN=hf_xxxx
OPENROUTER_API_KEY=sk-or-xxxx
```
Get `HF_TOKEN` from https://huggingface.co/settings/tokens. `run_eval.sh` loads this automatically.

## Running eval

```bash
# Canonical launch — sets all env vars, logs to logs/sft_eval_TIMESTAMP.log
bash scripts/run_eval.sh

# With options
bash scripts/run_eval.sh --model outputs/sft_checkpoint --max_new_tokens 8192

# Check progress
tail -f logs/sft_eval_*.log
wc -l outputs/sft_eval_results.jsonl
```

`run_eval.sh` handles: loading secrets, setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, timestamped log file, background launch. Do not launch `sft_eval.py` directly without it.

---

## Hardware

- **Eval:** A40 (48GB) — sufficient for inference at max_new_tokens=8192
- **SFT / GRPO:** A100 SXM 80GB — required due to vocab OOM at training sequence lengths

See `PLAN.md` for full GPU selection rationale.

---

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [math-verify (HuggingFace)](https://github.com/huggingface/Math-Verify)
- [Open-R1 Project](https://github.com/huggingface/open-r1)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1)
