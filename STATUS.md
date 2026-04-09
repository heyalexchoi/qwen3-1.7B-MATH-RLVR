# STATUS.md — Current Working Memory

> Scratch pad for the active phase. Expires each phase. Historical context in PLAN.md + memory/math-rlvr.md.

---

**Phase: SFT — RUNNING**

---

## Active Pod

- **Pod ID**: `1tbygbeoiv5n9u` — A100 SXM 80GB, $1.49/hr, US
- **SSH**: `root@154.54.102.42 -p 12259` (key: `~/.runpod/ssh/RunPod-Key-Go`)
- **Started**: ~20:40 UTC 2026-04-09
- **ETA**: ~01:00 UTC 2026-04-10 (~4.25hr, 1275 steps @ ~12s/step)
- **Config**: max_seq_length=32768, batch=2, grad_accum=8, adamw_torch, lr=2e-5 cosine, 3 epochs
- **HF push**: `heyalexchoi/qwen3-1.7b-math-sft` (hub_strategy="checkpoint" — pushes each save)
- **Monitor cron**: `f03acfde` — fires every 30min into this topic

## Log

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@154.54.102.42 -p 12259 \
  "tail -50 /workspace/qwen3-math-rlvr/logs/sft_launch.log"
```

## Key Reminders

- **NEVER remove the pod** — only `runpodctl pod stop` after rsyncing checkpoint locally
- Rsync checkpoint first, verify non-zero files, THEN stop
- HF push is active as secondary backup (but still rsync locally)

## Next Steps (after SFT)

1. Rsync `outputs/sft_checkpoint` locally + verify
2. Stop pod (`runpodctl pod stop 1tbygbeoiv5n9u`)
3. Run `sft_eval.py` on checkpoint (A40, delegatable)
4. Update PLAN.md step [3] and [3a] when complete
5. Proceed to GRPO
