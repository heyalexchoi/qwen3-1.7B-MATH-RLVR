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

- **Pod:** `ad0z39tw8zmc3w` — L40S 48GB, `root@103.196.86.50 -p 24362`, $0.86/hr
- **Project path on pod:** `/workspace/qwen3-math-rlvr/`
- **Status:** GRPO training running (PID 1561, resumed from step 5000/epoch 0.667, ~step 5018+ as of 2026-04-12 02:34 UTC). torch 2.6.0, TRL 1.0.0, math-verify installed.
- **WandB run:** `99hauae9` — https://wandb.ai/heyalexchoi/qwen3-math-rlvr/runs/99hauae9

## Previous Pod (stopped)

- **Pod:** `gol7yudqrlfn48` — H100 SXM 80GB, stopped 2026-04-12 00:50 UTC. Training ran from step 0 to ~step 5720 (epoch 0.697). Last pushed checkpoint: step 5000 (epoch 0.667) at `heyalexchoi/qwen3-1.7b-math-grpo/last-checkpoint`.

## GPU Selection

| GPU | VRAM | Use case | ~$/hr |
|-----|------|----------|-------|
| A100 SXM 80GB | 80GB | SFT/GRPO training (Qwen3 152k vocab @ 32k seq) | $1.49 |
| L40S 48GB | 48GB | **Preferred eval GPU** — ~2x faster than A40 (Ada Lovelace); OOMs on SFT/GRPO training | ~$0.39 |
| A40 48GB | 48GB | Eval fallback if L40S unavailable — same VRAM, slower | ~$0.39 |
| H100 80GB | 80GB | SFT/GRPO if A100 unavailable | ~$2.49 |

Qwen3-1.7B SFT/GRPO **requires 80GB**: logits = `batch×seq×152k vocab×2B ≈ 40GB` at seq=32768. A40 is fine for inference/eval only.

## Create a Pod

```bash
runpodctl pod create \
  --name "clawd-sft" \
  --gpu-id "NVIDIA A100-SXM4-80GB" \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \  # see PyTorch upgrade note below
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

## One-Command Setup (preferred)

`scripts/setup_runpod_training.sh` handles the full setup in one shot. Run from local machine:

### Step 0: Create `secrets.env` in project root (one-time)

The credentials already live at `~/.config/openclaw/secrets.env` on the local machine. Copy them to the project root so rsync picks them up:

```bash
cp ~/.config/openclaw/secrets.env /home/dev/.openclaw/workspace/qwen3-math-rlvr/secrets.env
```

The file must contain `HF_TOKEN` and `WANDB_API_KEY` (bare `KEY=VALUE`, no `export`):
```
HF_TOKEN=hf_xxxx
WANDB_API_KEY=xxxx
GITHUB_TOKEN=ghp_xxxx   # optional, for private repo clone
```

This file gets rsynced to the pod. The setup script sources it to write HF and WandB credentials. Without it, the script exits immediately with an error. `secrets.env` is gitignored — never committed.

### Step 1–2: Rsync + run setup

```bash
# Step 1: Rsync project to pod (secrets.env and data/math_train.jsonl go with it)
rsync -av --exclude='outputs/' --exclude='__pycache__/' --exclude='.git/' --exclude='wandb/' \
  -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/ root@<IP>:/workspace/qwen3-math-rlvr/

# Step 2: SSH in and run setup
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> \
  'cd /workspace/qwen3-math-rlvr && bash scripts/setup_runpod_training.sh'
```

To resume from a specific checkpoint (default: auto-downloads latest from HF Hub):
```bash
ssh ... 'cd /workspace/qwen3-math-rlvr && bash scripts/setup_runpod_training.sh \
  --resume_from_checkpoint outputs/grpo_checkpoint/last-checkpoint'
```

`secrets.env` is in `.gitignore` — rsynced to the pod, never committed.

The script: loads secrets, upgrades torch, clones repo, installs deps, writes HF + WandB credentials, downloads math_train.jsonl, downloads latest checkpoint from HF Hub, launches training.

**Use manual steps below only if the script fails for a specific reason.**

## Install Deps

**PyTorch upgrade required first** — the `runpod/pytorch:2.4.0` image ships PyTorch 2.4.1, but TRL 1.0.0 requires 2.6.0+ (imports `FSDPModule` added in 2.6). Always upgrade before installing requirements:

```bash
# Step 1: upgrade torch + torchvision to match (torchvision 0.19.x links against 2.4.1 and breaks)
ssh ... "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 -q"

# Step 2: install project requirements
ssh ... "cd /workspace/qwen3-math-rlvr && pip install -r requirements.txt -q > /workspace/pip_install.log 2>&1 &"

# Step 3: log in to wandb (writes to ~/.netrc, persists across sessions)
ssh ... "wandb login <WANDB_API_KEY>"

# Step 4: log in to HuggingFace Hub (writes to ~/.cache/huggingface/token, persists across sessions)
ssh ... "huggingface-cli login --token <HF_TOKEN>"
```

Fire and forget step 2 — check `/workspace/pip_install.log` for completion, don't block-poll.

**Why explicit logins?** `~/.config/openclaw/secrets.env` uses bare `KEY=VALUE` format (no `export`). Plain `source secrets.env` sets shell vars but doesn't propagate them to child processes. `set -a; source; set +a` works for one-off commands but isn't persistent. Writing to `~/.netrc` (wandb) and `~/.cache/huggingface/token` (HF) avoids env-var dependency entirely — these files are always checked by their respective clients.

**⚠️ HF token must be written BEFORE training starts.** `hub_strategy="checkpoint"` attempts to create the HF repo at `GRPOTrainer.__init__()` — not at push time. If the token is missing at init, training crashes with 401 immediately. `huggingface-cli login` or writing directly to `~/.cache/huggingface/token` both work.

## Launch Training

```bash
ssh ... << 'EOF'
set -a; source /root/.secrets.env; set +a   # ⚠️ REQUIRED — do not omit
cd /workspace/qwen3-math-rlvr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python scripts/grpo_train.py --push_to_hub \
  > /workspace/qwen3-math-rlvr/logs/grpo_launch.log 2>&1 &
echo "PID: $!"
EOF
```

**`set -a` is required — do not omit.** Plain `source secrets.env` sets shell vars but doesn't export them to child processes. Without it, `HUGGING_FACE_HUB_TOKEN` is invisible to the training script and the hub push fails with 401. `wandb login` (done at setup) writes to `~/.netrc`, which wandb checks independently — belt-and-suspenders for wandb, but `HF_TOKEN` has no equivalent file-based auth, so `set -a` is the only fix for HF.

Always `nohup` + absolute log path on `/workspace/`. Set a monitoring cron after launch.

**`TORCH_FORCE_WEIGHTS_ONLY=0` is required for resuming checkpoints.** TRL checkpoints include `rng_state.pth` which contains numpy arrays. PyTorch 2.6 rejects these with the new `weights_only=True` default. Prefix the training command:

```bash
TORCH_FORCE_WEIGHTS_ONLY=0 nohup python scripts/grpo_train.py --push_to_hub \
  --resume_from_checkpoint outputs/grpo_checkpoint/last-checkpoint \
  > logs/grpo_launch.log 2>&1 &
```

**`math_train.jsonl` is not in git.** If it's missing, training crashes immediately with `FileNotFoundError`. The setup script downloads it automatically. Manual SCP:

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no -P <PORT> \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/data/math_train.jsonl \
  root@<IP>:/workspace/qwen3-math-rlvr/data/math_train.jsonl
```

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

## Disk Space — Always Use Volume for HF Cache

Container root disk is typically **30GB**. Each model checkpoint revision downloads ~3.4GB. After 4–5 downloads the root disk fills and vLLM silently fails to load the model.

**Always redirect HF cache to `/workspace` before running eval:**

```bash
# One-time fix on a pod (if not already done by setup script):
mkdir -p /workspace/.hf_cache
if [[ -d ~/.cache/huggingface && ! -L ~/.cache/huggingface ]]; then
  cp -a ~/.cache/huggingface/. /workspace/.hf_cache/
  rm -rf ~/.cache/huggingface
fi
ln -s /workspace/.hf_cache ~/.cache/huggingface
```

`setup_runpod_training.sh` does this automatically at step 5. If running eval manually on a pod not set up with that script, run the above first.

To verify:
```bash
ls -la ~/.cache/huggingface  # should show: -> /workspace/.hf_cache
df -h /workspace             # should show ≥30GB free
```

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
- **Tokenizer bug:** Qwen3 SFT/instruct checkpoints may save `extra_special_tokens` as a list — `transformers>=4.51` crashes on load. `math500_eval.py` auto-patches on load. See README → "Qwen3 tokenizer: extra_special_tokens must be a dict" for details.
