# STATUS.md — Current Working Memory

> Scratch pad for the active phase. Expires each phase. Historical context in PLAN.md + memory/math-rlvr.md.

---

**Phase: SFT Eval — 🔄 RUNNING (relaunched 12:32 UTC 2026-04-10)**

---

## Active Pod

- **Pod ID**: `zrszn3f053jgcj` — A40 48GB, $0.44/hr, SE
- **SSH**: `root@194.68.245.64 -p 22047`
- **PID**: 1960 (sft_eval.py)
- **Started**: ~12:36 UTC 2026-04-10 (relaunched with 32k tokens + Qwen thinking-mode sampling)
- **ETA**: unknown — depends on avg response length with correct chat template format
- **Monitor cron**: `9acabe61` — fires every 30 min → isolated, delivers to topic 176

## What Happened

Run killed 2026-04-10 ~12:03 UTC after 1/500 problems in ~1hr. Root cause: greedy pass@1 enters circular `<think>` loops on Qwen thinking-mode models — fills 32768 tokens, no `\boxed{}` produced. See PLAN.md "Qwen Thinking Mode" finding.

## Bugs Fixed (from previous session, still in effect)

1. `sft_eval.py` was using few-shot prompt format instead of `tokenizer.apply_chat_template()` → fixed
2. Stop tokens didn't include `<|im_end|>` (151645) → fixed
3. Input truncation removed — unnecessary
4. `generation_config.json` patched to add 151645; pushed to HF Hub

## Command Running

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/sft_eval.py \
  --model outputs/sft_checkpoint \
  --max_new_tokens 32768 \
  --n_samples 8
```

Inference: `temp=0.6, top_p=0.95, top_k=20` (Qwen thinking-mode official). Pass@1 = unbiased estimate `c/n` from 8 samples.

## Check Progress

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@194.68.245.64 -p 22047 \
  "wc -l /workspace/qwen3-math-rlvr/outputs/sft_eval_results.jsonl 2>/dev/null || echo '0 lines so far'"
```

## Output Files (on pod)

- `/workspace/qwen3-math-rlvr/outputs/sft_eval_results.jsonl` — incremental per-problem results (pass@1 + pass@8 per line)
- `/workspace/qwen3-math-rlvr/outputs/sft_eval_results_summary.json` — written at completion
- `/workspace/qwen3-math-rlvr/logs/sft_eval_launch.log` — live log

## On Completion (cron handles this)

1. Rsync results to local `outputs/`
2. `PATH=$HOME/.local/bin:$PATH runpodctl pod stop zrszn3f053jgcj` (DO NOT remove)
3. Report pass@1 and pass@8 from summary JSON
4. Update PLAN.md [3a] with result
5. Remove monitor cron `9acabe61`

## Key Reminders

- **NEVER remove the pod** — only `stop` after rsyncing results
- Pod volume has SFT checkpoint — preserve it
