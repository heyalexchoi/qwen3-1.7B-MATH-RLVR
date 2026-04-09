# STATUS.md — Current Working Memory

> This file is scratch pad / working memory for the active phase.
> It expires with each phase. Historical context lives in `PLAN.md` and `memory/math-rlvr.md`.

---

**Phase: SFT — Ready to run, no active pod**

---

## Current State

- No pods running, no crons active
- All code changes from 2026-04-09 session are committed to local files

## What's Ready

- `scripts/03_sft_train.py` — updated with file logging, `--resume_from_checkpoint auto`, `--push_to_hub`, absolute output path, `WANDB_PROJECT=qwen3-math-rlvr`, `hub_strategy="checkpoint"`
- `configs/sft_config.yaml` — `save_total_limit=1` (disk overflow fix)
- `scripts/03a_sft_eval.py` — new, math-verify scoring, use for SFT and GRPO eval
- `scripts/04_grpo_train.py` — full rewrite, math-verify reward, `\boxed{}` extraction
- `scripts/05_final_eval.py` — full rewrite (optional/archival use only)
- `configs/grpo_config.yaml` — new

## Next Action

Spin up A100 80GB SECURE pod and run SFT from scratch:

```bash
export PATH="$HOME/.local/bin:$PATH"
runpodctl pod create \
  --name "clawd-sft" \
  --gpu-id "NVIDIA A100-SXM4-80GB" \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --container-disk-in-gb 50 \
  --volume-in-gb 100 \
  --volume-mount-path "/workspace" \
  --ports "22/tcp" \
  --cloud-type SECURE \
  -o json
```

If SECURE has capacity issues (machine=None after 5 min), fall back to COMMUNITY.

Then rsync, install deps, launch:
```bash
# Rsync code + data
rsync -av --exclude='outputs/' --exclude='__pycache__/' \
  -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/ root@<IP>:/workspace/qwen3-math-rlvr/

rsync -av -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/data/ root@<IP>:/workspace/qwen3-math-rlvr/data/

# Install deps
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> \
  "apt-get update -qq && apt-get install -y rsync && cd /workspace/qwen3-math-rlvr && pip install -r requirements.txt -q && pip install 'math-verify[antlr4_13_2]' -q && pip install flash-attn --no-build-isolation -q"

# Launch
WANDB_KEY=$(grep WANDB_API_KEY ~/.config/openclaw/secrets.env | cut -d= -f2)
HF_TOKEN=$(grep HF_TOKEN ~/.config/openclaw/secrets.env | cut -d= -f2)
ssh ... "cd /workspace/qwen3-math-rlvr && \
  export WANDB_API_KEY=<key> HF_TOKEN=<token> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  nohup python scripts/03_sft_train.py \
    --model Qwen/Qwen3-1.7B-Base \
    --data data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl \
    --output /workspace/qwen3-math-rlvr/outputs/sft_checkpoint \
    --config configs/sft_config.yaml \
    --push_to_hub \
  > /workspace/qwen3-math-rlvr/logs/sft_launch.log 2>&1 &"
```

ETA: ~4.25 hrs (1275 steps @ ~12s/step on A100)

## Key Reminders

- **NEVER remove a pod without rsyncing checkpoints locally first** (incident 2026-04-09)
- Set cron after pod boots to monitor — do NOT block-poll (AGENTS.md)
- After SFT completes: rsync checkpoint locally BEFORE stopping pod, then run 03a eval
- Checkpoint pushed to HF via hub_strategy="checkpoint" as backup during training

## TODOs

- [ ] Re-run base eval with `03a_sft_eval.py --model Qwen/Qwen3-1.7B-Base` for math-verify-consistent baseline
- [ ] After SFT: run `03a_sft_eval.py` for SFT score
- [ ] After GRPO: run `03a_sft_eval.py --model outputs/grpo_checkpoint` for GRPO score
- [ ] Clarify wandb run URL — project should now correctly log to `qwen3-math-rlvr`
