# Qwen3-1.7B Math RLVR

Demonstrating distillation + RLVR on math reasoning with Qwen3-1.7B-Base.

→ **Full pipeline context and findings:** [`PLAN.md`](PLAN.md)  
→ **Active run / current phase:** [`STATUS.md`](STATUS.md)  
→ **Session log (history + decisions):** [`memory/math-rlvr.md`](../memory/math-rlvr.md)

---

## Document Structure

| File | Purpose |
|------|---------|
| `PLAN.md` | Stateless execution plan — pipeline, script specs, key findings, artifact table. No runtime info. Read this for context on upcoming tasks or past phases. |
| `STATUS.md` | Current working memory — active pod, config, next steps. Expires each phase. |
| `memory/math-rlvr.md` | Session log — historical record of what happened and why (in workspace `memory/`). |
| `docs/` | Detailed sub-docs. If `PLAN.md` grows large, break into `docs/plan-*.md` referenced from PLAN.md. |

README stays lean — pointers, not details.

---

## Development Status

| Script | Status | Notes |
|--------|--------|-------|
| `00_prepare_data.py` | ✅ Done | GSM8K: 7,473 train / 1,319 test |
| `01_baseline_eval.py` | ✅ Done | GSM8K: **74.6% pass@1** |
| `06_math500_eval.py` | ✅ Done | MATH-500: **31.6% pass@1 / 58.4% pass@8** |
| `07_qwen32b_traces.py` | ✅ Done | 7,490 MATH traces at MAX_TOKENS=32768 |
| `08_rescore_traces.py` | ✅ Done | Regex normalizer fix; historical artifact |
| `09_rescore_mathverify.py` | ✅ Done | math-verify rescore; **95.51% (7,154/7,490)** |
| `10_rerun_truncated.py` | ✅ Done | 244 truncated reruns; 160 newly correct |
| `03_sft_train.py` | ✅ Done | SFT on 7,154 correct MATH traces |
| `03a_sft_eval.py` | ⏳ Pending | Eval SFT checkpoint on MATH-500 — **not yet written** |
| `04_grpo_train.py` | ⏳ Pending | **Needs full rewrite** — currently GSM8K-only |
| `05_final_eval.py` | ⏳ Pending | **Needs full rewrite** — currently GSM8K-only |

See `PLAN.md` for pending script specs (what each must do).

---

## Pipeline Overview

```
[0] Data prep (GSM8K + MATH)                           ✅
[1] Base eval — GSM8K + MATH-500                       ✅  31.6% MATH-500 pass@1
[2] Generate Qwen3-32B traces (scripts 07→10)          ✅  7,154 correct traces
[3] SFT on correct traces (script 03)                  ✅
[3a] SFT eval — MATH-500 (script 03a)                  ⏳  target ~45-55%
[4] GRPO training (script 04)                          ⏳  target ~85-90%
[5] Final eval — base vs SFT vs GRPO (script 05)       ⏳
```

---

## Setup

### Requirements
```bash
conda create -n qwen-math python=3.11 -y
conda activate qwen-math
pip install -r requirements.txt

# Optional: Flash Attention
pip install flash-attn --no-build-isolation

# Evaluator
pip install 'math-verify[antlr4_13_2]'
```

### Data Preparation
```bash
python scripts/00_prepare_data.py
```

---

## Running the Pipeline

### Phase 1: Baseline Evaluation
```bash
python scripts/01_baseline_eval.py \
    --model Qwen/Qwen3-1.7B-Base \
    --output outputs/baseline_results.json

python scripts/06_math500_eval.py \
    --model Qwen/Qwen3-1.7B-Base \
    --output outputs/math500_results.json
```

### Phase 2: Generate Reasoning Traces
```bash
# Requires OPENROUTER_API_KEY
python scripts/07_qwen32b_traces.py \
    --input data/math_train.jsonl \
    --output data/traces/qwen32b_math_traces.jsonl
```

### Phase 3: SFT Training
```bash
python scripts/03_sft_train.py \
    --model Qwen/Qwen3-1.7B-Base \
    --data data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl \
    --output outputs/sft_checkpoint \
    --config configs/sft_config.yaml
```

### Phase 3a: SFT Eval
```bash
# Script pending — see PLAN.md for spec
python scripts/03a_sft_eval.py \
    --model outputs/sft_checkpoint \
    --output outputs/sft_eval_results.json
```

### Phase 4: GRPO Training
```bash
# Script needs full rewrite — see PLAN.md for spec
python scripts/04_grpo_train.py \
    --model outputs/sft_checkpoint \
    --output outputs/grpo_checkpoint
```

### Phase 5: Final Evaluation
```bash
# Script needs full rewrite — see PLAN.md for spec
python scripts/05_final_eval.py \
    --checkpoints \
        Qwen/Qwen3-1.7B-Base \
        outputs/sft_checkpoint \
        outputs/grpo_checkpoint \
    --output outputs/final_comparison.json
```

---

## Results

| Phase | MATH-500 pass@1 | Notes |
|-------|----------------|-------|
| Base | **31.6%** ✅ | Measured |
| Post-SFT | ~45–55% | Target |
| Post-GRPO | ~85–90% | Target |

---

## Hardware Requirements

- **Minimum**: 1x A100 (80GB) — required due to Qwen3's 152k vocab at seq_len=32768
- **Note**: 48GB GPUs (L40S, A40) are insufficient for SFT/GRPO at max sequence length

See `PLAN.md` → Key Findings for the OOM root cause explanation.

---

## RunPod Operations

All GPU work runs on RunPod via `runpodctl` (at `~/.local/bin/runpodctl`).

### Start a Pod
```bash
export PATH="$HOME/.local/bin:$PATH"
runpodctl pod create \
  --name "clawd-math" \
  --gpu-id "NVIDIA A100-SXM4-80GB" \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --container-disk-in-gb 50 \
  --volume-in-gb 20 \
  --volume-mount-path "/workspace" \
  --ports "22/tcp" \
  --cloud-type SECURE \
  -o json
```

### Get SSH Details
```bash
runpodctl pod get <POD_ID> -o json
# Look for: runtime.ports[].ip, runtime.ports[].publicPort
```

### Sync Code
```bash
rsync -av -e "ssh -i ~/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no" \
  ~/workspace/qwen3-math-rlvr/ \
  root@<IP>:/workspace/qwen3-math-rlvr/
```

### SSH In
```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o StrictHostKeyChecking=no root@<IP> -p <PORT>
```

### Stop Pod
```bash
runpodctl pod stop <POD_ID>
runpodctl pod remove <POD_ID>
```

### GPU Selection

| Task | GPU | Cost (approx) |
|------|-----|---------------|
| Eval only (scripts 01, 06) | RTX 3090 (24GB) | ~$0.20/hr |
| SFT / GRPO | A100 SXM 80GB | ~$1.49/hr |

Use `SECURE` cloud — `COMMUNITY` pods can get stuck on boot.

---

## Project Structure

```
qwen3-math-rlvr/
├── README.md           ← you are here
├── PLAN.md             ← stateless pipeline plan + findings
├── STATUS.md           ← current phase / active run
├── requirements.txt
├── configs/
│   ├── sft_config.yaml
│   └── grpo_config.yaml
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_baseline_eval.py
│   ├── 03_sft_train.py
│   ├── 03a_sft_eval.py    (pending)
│   ├── 04_grpo_train.py   (needs rewrite)
│   ├── 05_final_eval.py   (needs rewrite)
│   ├── 06_math500_eval.py
│   ├── 07_qwen32b_traces.py
│   ├── 08_rescore_traces.py
│   ├── 09_rescore_mathverify.py
│   └── 10_rerun_truncated.py
├── data/
│   ├── gsm8k/
│   ├── math_train.jsonl
│   └── traces/
├── docs/
│   └── findings.md
└── outputs/
```

---

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [math-verify (HuggingFace)](https://github.com/huggingface/Math-Verify)
- [Open-R1 Project](https://github.com/huggingface/open-r1)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1)
