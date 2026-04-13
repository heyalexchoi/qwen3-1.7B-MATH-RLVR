#!/usr/bin/env bash
# setup_runpod_training.sh
# Run this on a fresh RunPod pod to set up and launch GRPO training.
#
# Usage:
#
#   # 1. Rsync the project (includes secrets.env and data/) to the pod
#   #    Note: data/ is in .gitignore but NOT excluded from rsync — math_train.jsonl goes with it
#   rsync -av --exclude='outputs/' --exclude='__pycache__/' --exclude='.git/' --exclude='wandb/' \
#     -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
#     /home/dev/.openclaw/workspace/qwen3-math-rlvr/ root@<IP>:/workspace/qwen3-math-rlvr/
#
#   # 2. SSH in and run the script
#   ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> \
#     'cd /workspace/qwen3-math-rlvr && bash scripts/setup_runpod_training.sh'
#
#   # To resume from a specific checkpoint (default: downloads latest from HF Hub):
#   ssh ... 'cd /workspace/qwen3-math-rlvr && bash scripts/setup_runpod_training.sh \
#     --resume_from_checkpoint outputs/grpo_checkpoint/last-checkpoint'
#
# secrets.env is in .gitignore — it is rsynced to the pod but never committed to git.
#
# What this script does (in order):
#   1. Sources secrets.env from the project root
#   2. Upgrades PyTorch to 2.6.0 (required by TRL 1.0.0 / FSDPModule)
#   3. Installs Python dependencies from requirements.txt
#   4. Writes HF and WandB credentials
#   5. Downloads the checkpoint from HF Hub (if not already present)
#   6. Ensures math_train.jsonl is present (already rsynced with project)
#   7. Launches training with nohup, logging to logs/grpo_launch.log
#
# Prerequisites on the LOCAL machine:
#   - secrets.env in project root (gitignored) with HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN
#   - ~/.runpod/ssh/RunPod-Key-Go SSH key
#
# Prerequisites on the POD:
#   - /workspace is writable (pod storage or network volume)
#   - NVIDIA GPU available (L40S, H100, etc.)
#
# CRITICAL KNOWN ISSUES (things that burned us before):
#   - PyTorch <2.6 will crash with: "cannot import name 'FSDPModule' from 'torch.distributed.fsdp'"
#   - TORCH_FORCE_WEIGHTS_ONLY=0 is REQUIRED to load checkpoints containing numpy RNG state
#   - HF token must be written to ~/.cache/huggingface/token BEFORE training starts
#     (hub_strategy="checkpoint" tries to create the HF repo at trainer __init__ time)
#   - math_train.jsonl is NOT in the git repo — must be sourced separately
#   - WandB auth must be written to ~/.netrc (env var alone is not persistent across processes)

set -euo pipefail

REPO_URL="https://github.com/heyalexchoi/qwen3-1.7B-MATH-RLVR.git"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/qwen3-math-rlvr"
LOG_DIR="$REPO_DIR/logs"
HF_REPO_ID="heyalexchoi/qwen3-1.7b-math-grpo"
DATA_HF_REPO="heyalexchoi/math-train-data"   # set to "" if you SCP the file manually
SECRETS_FILE="$(dirname "$(realpath "$0")")/../secrets.env"  # project root secrets.env
RESUME_CHECKPOINT=""
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --resume_from_checkpoint=*) RESUME_CHECKPOINT="${arg#*=}" ;;
    --resume_from_checkpoint)   shift; RESUME_CHECKPOINT="$1" ;;
    *) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
  esac
done

echo "════════════════════════════════════════════════════════"
echo " GRPO RunPod Setup — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Load secrets ─────────────────────────────────────────────────────
echo ""
echo "── [1/7] Loading secrets ──"
if [[ -f "$SECRETS_FILE" ]]; then
  set -a; source "$SECRETS_FILE"; set +a
  echo "  Loaded from $SECRETS_FILE"
else
  echo "ERROR: secrets.env not found at $SECRETS_FILE"
  echo "  Rsync the project from the local machine first (secrets.env is in the project root, gitignored)"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]];       then echo "ERROR: HF_TOKEN not set in secrets.env";       exit 1; fi
if [[ -z "${WANDB_API_KEY:-}" ]];  then echo "ERROR: WANDB_API_KEY not set in secrets.env";  exit 1; fi

if [[ -z "${HF_TOKEN:-}" ]];       then echo "ERROR: HF_TOKEN not set";       exit 1; fi
if [[ -z "${WANDB_API_KEY:-}" ]];  then echo "ERROR: WANDB_API_KEY not set";  exit 1; fi

# ── Step 2: Upgrade PyTorch to 2.6 ──────────────────────────────────────────
echo ""
echo "── [2/7] Upgrading PyTorch to 2.6.0 ──"
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo "  Current: $PYTORCH_VERSION"
if [[ "$PYTORCH_VERSION" < "2.6" ]]; then
  echo "  Upgrading (required for TRL 1.0.0 / FSDPModule)..."
  pip install -q --upgrade "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
  echo "  Done: $(python -c 'import torch; print(torch.__version__)')"
else
  echo "  Already >=2.6, skipping"
fi

# ── Step 3: Clone repo ───────────────────────────────────────────────────────
echo ""
echo "── [3/7] Cloning repo ──"
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "  Repo already present, pulling latest..."
  git -C "$REPO_DIR" pull --ff-only || echo "  (pull failed — continuing with existing code)"
else
  echo "  Cloning $REPO_URL..."
  # Use token auth if available
  CLONE_URL="https://${GITHUB_TOKEN:-}@github.com/heyalexchoi/qwen3-1.7B-MATH-RLVR.git"
  git clone "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── Step 4: Install Python deps ──────────────────────────────────────────────
echo ""
echo "── [4/7] Installing Python dependencies ──"
mkdir -p logs
pip install -q -r requirements.txt
echo "  Done"

# ── Step 5: Write credentials ────────────────────────────────────────────────
echo ""
echo "── [5/7] Writing credentials ──"

# Redirect HF model cache to /workspace volume BEFORE any HF downloads.
# Container root disk (~30GB) fills up fast with model checkpoints (~3.4GB each).
# /workspace volume is 50-100GB and persists across restarts.
if [[ -d /workspace && ! -L "$HOME/.cache/huggingface" ]]; then
  mkdir -p /workspace/.hf_cache
  # Move existing cache if present, otherwise just create symlink
  if [[ -d "$HOME/.cache/huggingface" ]]; then
    echo "  Moving existing HF cache to /workspace/.hf_cache..."
    cp -a "$HOME/.cache/huggingface/." /workspace/.hf_cache/ 2>/dev/null || true
    rm -rf "$HOME/.cache/huggingface"
  fi
  mkdir -p "$HOME/.cache"
  ln -s /workspace/.hf_cache "$HOME/.cache/huggingface"
  echo "  HF cache symlinked: ~/.cache/huggingface -> /workspace/.hf_cache"
fi

# HF token — MUST be present before trainer init (hub_strategy="checkpoint" creates repo early)
mkdir -p "$HOME/.cache/huggingface"
echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
echo "  HF token written to ~/.cache/huggingface/token"

# WandB — write to ~/.netrc for process-persistent auth
python -c "
import wandb, os
wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
" 2>&1 | grep -v "^wandb:" || true
echo "  WandB credentials written"

# ── Step 6: Get training data ────────────────────────────────────────────────
echo ""
echo "── [6/7] Setting up training data ──"
DATA_FILE="$REPO_DIR/data/math_train.jsonl"
mkdir -p "$REPO_DIR/data"

if [[ -f "$DATA_FILE" ]]; then
  LINES=$(wc -l < "$DATA_FILE")
  echo "  math_train.jsonl already present ($LINES lines)"
elif [[ -n "$DATA_HF_REPO" ]]; then
  echo "  Downloading from HF dataset: $DATA_HF_REPO..."
  python -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='$DATA_HF_REPO', filename='math_train.jsonl', repo_type='dataset')
shutil.copy(path, '$DATA_FILE')
print(f'  Saved to $DATA_FILE')
" 2>&1
else
  echo "ERROR: math_train.jsonl not found and DATA_HF_REPO not configured."
  echo "  SCP it manually with:"
  echo "  scp -i ~/.runpod/ssh/RunPod-Key-Go -P <PORT> /path/to/math_train.jsonl root@<IP>:$DATA_FILE"
  exit 1
fi

# ── Step 7: Download checkpoint from HF Hub (if resuming) ───────────────────
echo ""
echo "── [7/7] Checkpoint setup ──"
CHECKPOINT_DIR="$REPO_DIR/outputs/grpo_checkpoint/last-checkpoint"
mkdir -p "$REPO_DIR/outputs/grpo_checkpoint"

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  echo "  Using explicit checkpoint: $RESUME_CHECKPOINT"
  CHECKPOINT_DIR="$RESUME_CHECKPOINT"
elif [[ -d "$CHECKPOINT_DIR" ]]; then
  echo "  Local checkpoint already exists at $CHECKPOINT_DIR"
  ls "$CHECKPOINT_DIR" | head -5
else
  echo "  Downloading latest checkpoint from HF Hub: $HF_REPO_ID..."
  python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id='$HF_REPO_ID',
    local_dir='$REPO_DIR/outputs/grpo_checkpoint/last-checkpoint',
    ignore_patterns=['*.md', '.gitattributes'],
)
print(f'  Downloaded to {path}')
" 2>&1
fi

if [[ ! -f "$CHECKPOINT_DIR/trainer_state.json" ]]; then
  echo "WARNING: trainer_state.json not found in checkpoint — training will start from scratch"
else
  GLOBAL_STEP=$(python -c "import json; d=json.load(open('$CHECKPOINT_DIR/trainer_state.json')); print(d.get('global_step', 'unknown'))")
  echo "  Checkpoint global_step: $GLOBAL_STEP"
fi

# ── Launch training ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " Launching GRPO training"
echo "════════════════════════════════════════════════════════"
mkdir -p "$LOG_DIR"

TRAIN_CMD="TORCH_FORCE_WEIGHTS_ONLY=0 python scripts/grpo_train.py --push_to_hub --resume_from_checkpoint $CHECKPOINT_DIR $EXTRA_ARGS"
echo "  Command: $TRAIN_CMD"
echo "  Log: $LOG_DIR/grpo_launch.log"
echo ""

# shellcheck disable=SC2086
nohup bash -c "cd $REPO_DIR && $TRAIN_CMD" > "$LOG_DIR/grpo_launch.log" 2>&1 &
TRAIN_PID=$!
echo "  Launched with PID $TRAIN_PID"

# Wait a moment and verify it started
sleep 15
if kill -0 $TRAIN_PID 2>/dev/null; then
  echo "  Process still alive after 15s — good"
  echo "  Tail of log:"
  tail -5 "$LOG_DIR/grpo_launch.log"
else
  echo "ERROR: Training process died within 15s. Log:"
  tail -30 "$LOG_DIR/grpo_launch.log"
  exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo " Setup complete. Training is running."
echo " Monitor: tail -f $LOG_DIR/grpo_launch.log"
echo " WandB:   https://wandb.ai/heyalexchoi/qwen3-math-rlvr"
echo " HF Hub:  https://huggingface.co/$HF_REPO_ID"
echo "════════════════════════════════════════════════════════"
