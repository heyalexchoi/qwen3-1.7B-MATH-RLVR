#!/bin/bash
set -e
export $(grep -v '^#' /home/dev/.config/openclaw/secrets.env | xargs)
cd /home/dev/.openclaw/workspace/qwen3-math-rlvr
python3 scripts/push_intermediate_checkpoints.py >> logs/push_intermediate_checkpoints.log 2>&1
