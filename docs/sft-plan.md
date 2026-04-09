# SFT Training Plan — Qwen3-1.7B MATH Distillation

**Status:** Ready to execute  
**Last updated:** 2026-04-09  
**Input:** 7,154 correct traces (math-verify rescored, 32k token rerun)  
**Target:** ~45-55% MATH-500 pass@1 (up from 31.6% baseline)

---

## Overview

Fine-tune Qwen3-1.7B-Base on Qwen3-32B reasoning traces via supervised learning.
We train the base model to produce `<think>...</think>` + solution in Qwen3 chat template format,
matching the inference-time format used in GRPO (Script 04).

Pipeline position:
```
Baseline (31.6%) → [THIS] SFT on 7,154 traces → GRPO/RLVR → Final eval
```

---

## Step 1: Update Script 03

**File:** `scripts/03_sft_train.py`

Key changes needed from original stub:

### 1a. Data loading — fix field names + format

Original script used `question`/`trace`. Our JSONL has:
- `problem` — input question
- `full_response` — `<think>...</think>\nsolution` (already correct format)
- `correct_mathverify` — boolean, filter to True only

New `load_traces()`:
```python
def load_traces(data_path: str, tokenizer) -> Dataset:
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if not data.get("correct_mathverify", False):
                continue
            # Format as Qwen3 chat (base model trained to predict assistant turn)
            messages = [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["full_response"]},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            examples.append({"text": text})
    return Dataset.from_list(examples)
```

### 1b. Response template for completion-only loss

With chat template, the response boundary is `<|im_start|>assistant\n`.  
Update `DataCollatorForCompletionOnlyLM`:
```python
response_template = "<|im_start|>assistant\n"
```

Only the assistant turn (the thinking + solution) contributes to loss.
The question tokens are masked. This is correct behavior.

### 1c. Data path default

Change default `--data` to:
```
data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl
```

### 1d. Wandb project name
```python
run_name=f"sft-qwen3-1.7b-math-distill"
```

---

## Step 2: Update sft_config.yaml

Key changes from original:

```yaml
# Sequence length — 32k covers 100% of our trace lengths (p99 = ~18.9k tokens)
max_seq_length: 32768

# Batch config for A40 48GB (bf16 + flash attn + grad checkpointing)
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # effective batch = 16

# Keep rest as-is (lr=2e-5, cosine, warmup 0.1, 3 epochs)
```

**Why effective batch 16:** Standard for distillation SFT at this scale.
Larger batch = more stable gradients. A40 fits batch=2 at 32k seq len comfortably.

---

## Step 3: Spin Up RunPod A40

**GPU:** A40 (48GB VRAM) — sweet spot for 1.7B at 32k seq len  
**Image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04`  
**Disk:** 50GB container, 20GB volume  

```bash
export PATH="$HOME/.local/bin:$PATH"
runpodctl pod create \
  --name "clawd-sft" \
  --gpuType "NVIDIA A40" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 20 \
  --volumePath "/workspace" \
  --ports "22/tcp" \
  --startSSH \
  --communityCloud
```

Get SSH details:
```bash
runpodctl pod get <POD_ID> -o json
# note: ip, port
```

SSH test:
```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> "echo ok"
```

---

## Step 4: Sync Code + Data

```bash
# Sync full project to pod
rsync -av --progress \
  -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/ \
  root@<IP>:/workspace/qwen3-math-rlvr/
```

Data file to sync: `data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl` (included above)

---

## Step 5: Install Dependencies on Pod

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> << 'EOF'
cd /workspace/qwen3-math-rlvr
pip install -q transformers trl datasets accelerate flash-attn wandb peft
pip install -q --no-build-isolation flash-attn
echo "deps ok"
EOF
```

Note: flash-attn may take a few minutes to compile.

---

## Step 6: Run SFT Training

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT> << 'EOF'
cd /workspace/qwen3-math-rlvr
export WANDB_API_KEY=8f534bc5763598d74c05a5b2948d0122b020829c
export WANDB_PROJECT=qwen3-math-rlvr

nohup python3 scripts/03_sft_train.py \
  --model Qwen/Qwen3-1.7B-Base \
  --data data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl \
  --output outputs/sft_checkpoint \
  --config configs/sft_config.yaml \
  > logs/sft_train.log 2>&1 &

echo "PID: $!"
echo "Tailing log..."
sleep 5 && tail -20 logs/sft_train.log
EOF
```

---

## Step 7: Monitor

Set cron job immediately after confirming training starts:

```
Every 30 min: SSH to pod, check log tail + last loss value + step count.
If training complete, run eval (Script 05 or 06) and remove cron.
```

Monitor commands:
```bash
# Tail log
ssh ... "tail -50 /workspace/qwen3-math-rlvr/logs/sft_train.log"

# Check step count
ssh ... "grep 'step' /workspace/qwen3-math-rlvr/logs/sft_train.log | tail -5"

# Check GPU util
ssh ... "nvidia-smi"
```

Estimated training time on A40: **4-6 hours** for 3 epochs over 7,154 examples.

---

## Step 8: Sync Checkpoint Back

```bash
rsync -av --progress \
  -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  root@<IP>:/workspace/qwen3-math-rlvr/outputs/sft_checkpoint/ \
  /home/dev/.openclaw/workspace/qwen3-math-rlvr/outputs/sft_checkpoint/
```

---

## Step 9: Quick Eval

Run MATH-500 eval on the SFT checkpoint (Script 06 with `--model outputs/sft_checkpoint`).

**Success criteria:** MATH-500 pass@1 > 40% (baseline: 31.6%)  
**Target:** 45-55%

If below 40%: check loss curves on wandb, consider re-run with different LR or more epochs.

---

## Step 10: Stop Pod

```bash
runpodctl pod stop <POD_ID>
```

**Don't forget this.** A40 at $0.44/hr idle = wasted money.

---

## Open Questions / Risks

| Issue | Status | Notes |
|-------|--------|-------|
| response_template token boundary | Verify at runtime | `<\|im_start\|>assistant\n` must match tokenizer output exactly — test with 1 example before full run |
| A40 availability on RunPod | Unknown until pod create | Fallback: A100 40GB (~$0.60/hr) |
| OOM at batch=2 / 32k seq | Low risk | Grad checkpointing handles it; reduce to batch=1 + grad_accum=16 if needed |
| Qwen3 tokenizer chat template | Needs verify | Base model may not have chat template set; if not, set manually in script |

---

## Cost Estimate

| Item | Est. Cost |
|------|-----------|
| A40 pod, 6hr training | ~$2.64 |
| A40 pod, 1hr setup/eval/sync | ~$0.44 |
| Total | **~$3-4** |

---

## Files

| File | Purpose |
|------|---------|
| `scripts/03_sft_train.py` | Main training script (needs update per Step 1) |
| `configs/sft_config.yaml` | Training hyperparams (needs update per Step 2) |
| `data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl` | SFT input (7,154 correct traces) |
| `outputs/sft_checkpoint/` | Output checkpoint directory |
| `logs/sft_train.log` | Training log (on pod) |
| `docs/sft-plan.md` | This doc |
