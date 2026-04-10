#!/usr/bin/env bash
# run_eval.sh — canonical launch script for sft_eval.py
#
# Required env vars (set externally or via secrets.env):
#   HF_TOKEN       — HuggingFace token (needed for private models / higher rate limits)
#
# Usage:
#   bash scripts/run_eval.sh [--model <path>] [--max_new_tokens <n>] [--greedy] [-- <extra args>]
#
# Modes:
#   default:  sampling eval — n=8, temp=0.6 → pass@8 + inferred pass@1 (c/n)
#             output: outputs/sft_eval_results.jsonl
#   --greedy: greedy eval  — n=1, temp=0.0 → greedy pass@1
#             output: outputs/sft_eval_greedy_results.jsonl
#
# Defaults:
#   --model   outputs/sft_checkpoint
#   --max_new_tokens  8192 (sampling) / 4096 (greedy)
#
# Run from repo root: cd /workspace/qwen3-math-rlvr && bash scripts/run_eval.sh

set -euo pipefail

# ── Sensitive config ─────────────────────────────────────────────────────────
# Load HF token if available. Store at ~/.config/openclaw/secrets.env (mode 600)
# Format: HF_TOKEN=hf_xxxx
SECRETS_FILE="${HOME}/.config/openclaw/secrets.env"
if [[ -f "${SECRETS_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a; source "${SECRETS_FILE}"; set +a
fi

# ── CUDA allocator config ─────────────────────────────────────────────────────
# Also set in the Python script itself, but belt-and-suspenders here.
# Prevents fragmentation OOM on long sequences (A40/A100).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL="outputs/sft_checkpoint"
MAX_NEW_TOKENS=8192
GREEDY_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --greedy) GREEDY_FLAG="--greedy"; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
LOG_FILE="logs/sft_eval_${TIMESTAMP}.log"
mkdir -p logs

echo "Starting eval: model=${MODEL} max_new_tokens=${MAX_NEW_TOKENS} greedy=${GREEDY_FLAG:-no}"
echo "Log: ${LOG_FILE}"

nohup python3 scripts/sft_eval.py \
    --model "${MODEL}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${GREEDY_FLAG} \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID=${PID}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Check progress: wc -l outputs/sft_eval_results.jsonl"
