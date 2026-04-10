# RunPod — Project Reference

Everything needed to provision, connect, monitor, and teardown RunPod pods for this project.

## Credentials & CLI

```bash
export PATH="$HOME/.local/bin:$PATH"  # runpodctl lives here
```

- **Credentials:** `~/.runpod/config.toml`
- **SSH key:** `~/.runpod/ssh/RunPod-Key-Go`
- **Secrets (WANDB, HF token):** `~/.config/openclaw/secrets.env` — source with `source ~/.config/openclaw/secrets.env`

## Current Pod

- **Eval pod:** `gol7yudqrlfn48` — H100 SXM 80GB, `root@64.247.201.44 -p 15452`, $2.99/hr
- **Project path on pod:** `/workspace/qwen3-math-rlvr/`
- **Status:** Pod up, deps installed (vLLM 0.19.0, math-verify), model synced. Eval NOT yet running — see PLAN.md → Current Run.

## GPU Selection

| GPU | VRAM | Use case | ~$/hr |
|-----|------|----------|-------|
| A100 SXM 80GB | 80GB | SFT/GRPO training (Qwen3 152k vocab @ 32k seq) | $1.49 |
| A40 / L40S 48GB | 48GB | Eval only — OOMs on Qwen3 SFT/GRPO | $0.39 |
| H100 80GB | 80GB | SFT/GRPO if A100 unavailable | ~$2.49 |

Qwen3-1.7B SFT/GRPO **requires 80GB**: logits = `batch×seq×152k vocab×2B ≈ 40GB` at seq=32768. A40 is fine for inference/eval only.

## Create a Pod

```bash
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

Always `--cloud-type SECURE` first. If no machine assigned after ~5 min, try `COMMUNITY` (less reliable).

**Volume sizing:** ≥50GB for SFT, ≥100GB for GRPO. Each checkpoint ≈ 8.8GB (3.3GB model + 5.6GB optimizer).

## Check SSH Readiness

SSH info is at `.ssh.ip` and `.ssh.port` — **NOT** `runtime.ports`:

```bash
runpodctl pod get <POD_ID> -o json | python3 -c \
  "import sys,json; d=json.load(sys.stdin); s=d.get('ssh',{}); print(s.get('ip'), s.get('port'))"
```

After creating a pod, poll for SSH readiness (cron every 2 min) — never block-poll in exec.

## SSH

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT>
```

## Rsync Code to Pod

```bash
# Install rsync first (not in base image)
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> \
  "apt-get update -qq && apt-get install -y rsync -qq"

# Rsync project (chown errors are benign)
rsync -av --exclude='outputs/' --exclude='__pycache__/' --exclude='.git/' \
  -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/ root@<IP>:/workspace/qwen3-math-rlvr/
```

Always write checkpoints and logs to `/workspace/` — container disk does NOT persist across restarts.

## Install Deps

```bash
ssh ... "cd /workspace/qwen3-math-rlvr && pip install -r requirements.txt -q > /workspace/pip_install.log 2>&1 &"
```

Fire and forget — check `/workspace/pip_install.log` for completion via cron, don't block-poll.

## Launch Training

```bash
ssh ... "cd /workspace/qwen3-math-rlvr && \
  source ~/.config/openclaw/secrets.env && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  nohup python scripts/sft_train.py [args] \
  > /workspace/qwen3-math-rlvr/logs/train.log 2>&1 & echo pid:\$!"
```

Always `nohup` + absolute log path on `/workspace/`. Set a monitoring cron after launch.

## Stop vs Remove

```bash
runpodctl pod stop <ID>    # stops compute billing, volume preserved ✅
runpodctl pod remove <ID>  # DESTROYS VOLUME PERMANENTLY ⚠️ — no undo
```

**⚠️ Never remove without backup confirmation (see below).**

## Rsync Outputs Back to Local

```bash
rsync -avz -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -p <PORT>" \
  root@<IP>:/workspace/qwen3-math-rlvr/outputs/ \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/outputs/
```

## Pre-Removal Safety Checklist

Before `pod remove`:
1. Rsync ALL checkpoints + logs to local and verify (`ls -lh outputs/`)
2. If `--push_to_hub` was used: verify HF repo has the latest checkpoint
3. When in doubt: use `pod stop` instead

**Never delegate pod removal to a subagent.** Subagents don't know what's been backed up.

> **Incident (2026-04-09):** Subagent removed pod `po3jbrudxordap` before rsyncing SFT checkpoint-1000 (~$5 compute lost permanently).

## OOM Notes

- Qwen3 152k vocab: logits = `batch × seq × 152064 × 2B` — at batch=2, seq=32768 → ~40GB just for logits
- `paged_adamw_8bit` does NOT fix this (optimizer ≠ forward pass)
- liger-kernel 0.7.0 is broken on PyTorch 2.4 (requires DTensor from 2.5+) — do NOT use
- Only real fix: 80GB GPU
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prevents fragmentation OOM — always set it

## Confirmed Working SFT Config

- batch=1, grad_accum=16, `expandable_segments:True`, no liger-kernel
- flash_attention_2, bfloat16, gradient_checkpointing, adamw_torch
- max_seq_length=32768

## Inference / Eval Notes

- **vLLM is primary backend** for `sft_eval.py` — auto-detected, falls back to HF if not installed
- HF fallback uses `flash_attention_2` if `flash_attn` is installed, else `sdpa`
- `run_eval.sh` auto-installs vLLM if missing (`pip install vllm -q`)
- **Tokenizer bug (transformers 4.57.6):** `extra_special_tokens` must be `{}` (dict), not a list — Qwen3 saves it as a list. Fix before loading: set `extra_special_tokens: {}` in `tokenizer_config.json`. Already patched in pod checkpoint; patch local copy too (see PLAN.md).
