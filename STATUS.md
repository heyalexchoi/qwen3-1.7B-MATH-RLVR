# STATUS.md — Current Working Memory

> Scratch pad for the active phase. Expires each phase. Historical context in PLAN.md + memory/math-rlvr.md.

---

**Phase: SFT — LAUNCHING**

---

## Launch Command

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> \
  "cd /workspace/qwen3-math-rlvr && \
   TIMESTAMP=\$(date +%Y%m%d_%H%M%S) && \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   WANDB_API_KEY=8f534bc5763598d74c05a5b2948d0122b020829c \
   HF_TOKEN=\$(cat ~/.cache/huggingface/token 2>/dev/null || echo '') \
   nohup python scripts/sft_train.py --push_to_hub \
     > logs/sft_launch.log 2>&1 &"
```

## Config (confirmed working from checkpoint-1000 run)

- `max_seq_length`: 32768
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 16  (effective batch = 16)
- `optim`: adamw_torch
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ← **required in env, not in script**
- No liger-kernel

## Pod Spec

- GPU: A100 SXM 80GB
- `--container-disk-in-gb 50`
- `--volume-in-gb 100` ← was 20GB, caused disk-full at checkpoint-1000
- Cloud: SECURE

## Key Reminders

- **NEVER remove the pod** — only `runpodctl pod stop` after rsyncing checkpoint locally
- Rsync checkpoint first, verify non-zero files, THEN stop
- HF push is active as secondary backup (but still rsync locally)

## Next Steps (after SFT)

1. Rsync `outputs/sft_checkpoint` locally + verify
2. Stop pod (`runpodctl pod stop <ID>`)
3. Run `sft_eval.py` on checkpoint (A40, delegatable)
4. Update PLAN.md steps [3] and [3a] when complete
5. Proceed to GRPO
