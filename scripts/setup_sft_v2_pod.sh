#!/bin/bash
# SFT v2 pod setup — pinned stack per docs/vllm-stack-pin.md + requirements-stack.txt.
# Run ON the pod from /workspace/qwen3-math-rlvr. Written by Claude Opus 4.8 — 2026-06-10.
set -euo pipefail
cd /workspace/qwen3-math-rlvr

[ -f secrets.env ] || { echo "ERROR: secrets.env missing (rsync it first)"; exit 1; }
set -a; source secrets.env; set +a

echo "=== [1/5] vLLM cu129 wheel (brings tf 5.10.2, tokenizers 0.22.2) ==="
pip install -q "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129

echo "=== [2/5] force torch cu129; drop base-image torchvision (torch-2.8 build) ==="
pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
pip uninstall -y -q torchvision || true

echo "=== [3/5] training deps ==="
pip install -q accelerate==1.13.0 "trl>=1.0.0" "peft>=0.7.0" datasets wandb pyyaml 'math-verify[antlr4_13_2]'

echo "=== [4/5] sanity: pinned stack ==="
python3 -c "
import torch, vllm, transformers, trl, accelerate
print('vllm', vllm.__version__, '| tf', transformers.__version__, '| torch', torch.__version__,
      '| trl', trl.__version__, '| accelerate', accelerate.__version__)
assert transformers.__version__ == '5.10.2', 'transformers drifted off the pin!'
assert torch.__version__.startswith('2.11.0+cu129'), 'torch drifted off the pin!'
print('cuda', torch.cuda.is_available())
"

echo "=== [5/5] wandb login ==="
wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 && echo "wandb ok"

echo "SETUP COMPLETE"
