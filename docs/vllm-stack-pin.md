# Pinned inference stack — vLLM for Qwen3-1.7B math SFT/GRPO

*Written by Claude Opus 4.8 — 2026-06-10.*

This is the **verified-working** stack for running our Qwen3-1.7B math checkpoints under vLLM
(for pass@k eval and GRPO rollouts). Use this for all v2 training/eval/RL so the checkpoint is
born vLLM-compatible. Full evidence: `eval_results/vllm_canary_2026-06-07/FINDINGS.md` §6.

## Versions (pin these)

| package      | version            | note                                          |
|--------------|--------------------|-----------------------------------------------|
| vllm         | `0.22.1` (cu129)   | first tf5-capable line; install the cu129 wheel |
| transformers | `5.10.2`           | what vLLM 0.22.1 resolves; loads checkpoint OK |
| torch        | `2.11.0+cu129`     | **cu129, NOT the default cu130**               |
| tokenizers   | `0.22.2`           |                                               |
| accelerate   | `1.13.0`           | needed for HF `device_map`                     |
| numpy        | `2.1.2`            |                                               |

GPU/driver: verified on **A40, driver 570.211 (CUDA 12.8)**. The cu129 build runs on a 12.8
driver via CUDA minor-version forward-compat. A ≥580/CUDA-13 driver would also let you use the
default cu130 wheel, but is **not required**.

## Why not just `pip install vllm`

The default wheel for vLLM ≥0.20 is **cu130** (torch 2.11.0+cu130, CUDA 13). On a CUDA-12.8 driver
that dies immediately: `RuntimeError: The NVIDIA driver on your system is too old (found version
12080)`. There is **no cu128 wheel** for 0.20–0.22; the cu129 wheel is the one that works on 12.8
drivers. This single gotcha is why earlier sessions wrongly concluded "modern vLLM needs a special
CUDA-13 pod."

## Install recipe (on `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`)

```bash
# 1. vLLM cu129 wheel
pip install "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129

# 2. force torch to the cu129 build (vLLM/pip may leave the default cu130 in place)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129

# 3. base-image torchvision is built for torch 2.8 and crashes transformers import
#    (operator torchvision::nms does not exist). Text models don't need it.
pip uninstall -y torchvision

# 4. HF device_map needs accelerate
pip install accelerate

# sanity
python -c "import torch,vllm,transformers; print(vllm.__version__, transformers.__version__, torch.__version__); print('cuda', torch.cuda.is_available())"
# -> 0.22.1 5.10.2 2.11.0+cu129   cuda True
```

## Checkpoint note
Always save the SFT tokenizer canonically (the v1 checkpoint's `tokenizer_config.json` was malformed;
`scripts/fix_sft_tokenizer.py` repairs it). The checkpoint loads and generates cleanly under tf 5.10.2.

## Verification command
`scripts/vllm_parity_canary.py --backend vllm --model <dir> --n 25` → expect correct, terminating
`\boxed{}` answers; any 8192-token pegs reflect the model's own termination behavior, not vLLM.
