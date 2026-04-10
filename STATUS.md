# STATUS.md — Current Working Memory

> Scratch pad for the active phase. Expires each phase. Historical context in PLAN.md + memory/math-rlvr.md.

---

**Phase: SFT — RUNNING**

---

## Active Pod

- **Pod ID**: `44gquoqhbrlatb` — A100 SXM 80GB, $1.49/hr, US
- **SSH**: `root@38.80.152.72 -p 31047` (key: `~/.runpod/ssh/RunPod-Key-Go`)
- **Started**: ~01:51 UTC 2026-04-10
- **ETA**: ~06:00 UTC 2026-04-10 (~4hr, 1275 steps @ ~10s/step)
- **Monitor cron**: `279d19e6` — every 30min, auto-rsync + stop pod on completion

## Config

- `max_seq_length`: 32768
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 16 (effective batch = 16)
- `optim`: adamw_torch
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ← set in launch env
- No liger-kernel
- `--push_to_hub` → `heyalexchoi/qwen3-1.7b-math-sft`
- Volume: 100GB (was 20GB — disk-full was the cause of the previous checkpoint-1000 interruption)

## Log

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@38.80.152.72 -p 31047 \
  "tail -50 /workspace/qwen3-math-rlvr/logs/sft_launch.log"
```

## Key Reminders

- **NEVER remove the pod** — only `runpodctl pod stop` after rsyncing checkpoint locally
- Rsync checkpoint first, verify non-zero files, THEN stop
- HF push is active as secondary backup

## Next Steps (after SFT completes)

1. Rsync `outputs/sft_checkpoint` locally + verify
2. Stop pod (`runpodctl pod stop 44gquoqhbrlatb`)
3. Run `sft_eval.py` on checkpoint (A40, delegatable)
4. Update PLAN.md steps [3] and [3a]
5. Proceed to GRPO
