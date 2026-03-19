# Qwen3-1.7B GSM8K Finetuning Demo

Demonstrating LLM finetuning skills: distillation + RLVR on math reasoning.

## Development Status

| Script | Status | Notes |
|--------|--------|-------|
| `00_prepare_data.py` | Reviewed & cleaned | Path anchored to script location, deduped logic, field renamed to `extracted_answer` |
| `01_baseline_eval.py` | Reviewed & cleaned | Input validation, `answers_match` numeric comparison, removed unused `batch_size` arg |
| `02_generate_traces.py` | Field refs fixed | Updated to use `extracted_answer`; TODOs for retry logic and rate limiting |
| `03_sft_train.py` | Not yet reviewed | |
| `04_grpo_train.py` | Not yet reviewed | |
| `05_final_eval.py` | Not yet reviewed | |

## Pipeline Overview

```
Qwen3-1.7B-Base (baseline ~40-50%)
        ↓
   [1] Baseline Eval
        ↓
   [2] Generate Claude Traces (distillation data)
        ↓
   [3] SFT on Traces (~65-75%)
        ↓
   [4] GRPO/RLVR (~80-85%)
        ↓
   [5] Final Eval
```

## Setup

### Requirements
```bash
# Create environment
conda create -n qwen-gsm8k python=3.11 -y
conda activate qwen-gsm8k

# Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention (recommended for speed)
pip install flash-attn --no-build-isolation
```

### Data Preparation
```bash
# Download GSM8K
python scripts/00_prepare_data.py
```

## Running the Pipeline

### Phase 1: Baseline Evaluation
```bash
python scripts/01_baseline_eval.py \
    --model Qwen/Qwen3-1.7B-Base \
    --output outputs/baseline_results.json
```

### Phase 2: Generate Reasoning Traces
```bash
# Requires ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=your_key_here

python scripts/02_generate_traces.py \
    --input data/gsm8k/train.jsonl \
    --output data/traces/claude_traces.jsonl \
    --model claude-sonnet-4-20250514 \
    --num_samples 5000
```

### Phase 3: SFT Training
```bash
python scripts/03_sft_train.py \
    --model Qwen/Qwen3-1.7B-Base \
    --data data/traces/claude_traces.jsonl \
    --output outputs/sft_checkpoint \
    --config configs/sft_config.yaml
```

### Phase 4: GRPO Training (TRL)
```bash
# Single GPU (A40/A100)
python scripts/04_grpo_train.py \
    --model outputs/sft_checkpoint \
    --data data/gsm8k/train.jsonl \
    --output outputs/grpo_checkpoint \
    --num_train_epochs 3 \
    --num_generations 4

# For faster iteration (fewer generations):
python scripts/04_grpo_train.py \
    --model outputs/sft_checkpoint \
    --output outputs/grpo_checkpoint \
    --num_generations 2 \
    --per_device_batch_size 2
```

### Phase 5: Final Evaluation
```bash
python scripts/05_final_eval.py \
    --checkpoints \
        Qwen/Qwen3-1.7B-Base \
        outputs/sft_checkpoint \
        outputs/grpo_checkpoint \
    --output outputs/final_comparison.json
```

## Expected Results

| Phase | GSM8K Accuracy | Notes |
|-------|----------------|-------|
| Baseline | ~40-50% | Few-shot prompting |
| Post-SFT | ~65-75% | Claude trace distillation |
| Post-GRPO | ~80-85% | Verifiable reward RL |

## Hardware Requirements

- **Minimum**: 1x A40 (48GB) or A100 (40GB)
- **Recommended**: 1x A100 (80GB) or H100

For the 1.7B model, full finetuning is tractable without LoRA.

## Project Structure

```
qwen3-gsm8k-demo/
├── README.md
├── requirements.txt
├── configs/
│   ├── sft_config.yaml
│   └── grpo_config.yaml
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_baseline_eval.py
│   ├── 02_generate_traces.py
│   ├── 03_sft_train.py
│   ├── 04_grpo_train.sh
│   └── 05_final_eval.py
├── data/
│   ├── gsm8k/
│   └── traces/
└── outputs/
```

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Open-R1 Project](https://github.com/huggingface/open-r1)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1)
