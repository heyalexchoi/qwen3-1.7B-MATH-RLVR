# Qwen3-1.7B Math RLVR

RLVR (GRPO with verifiable math rewards) on Qwen3-1.7B, scored by
[math-verify](https://github.com/huggingface/Math-Verify) on MATH-500.

*Maintained by Claude Opus 4.8 · Last updated 2026-06-26.*

## Status

| Milestone | Result |
|---|---|
| **GRPO-from-base POC** ✅ | MATH-500 greedy pass@1 **35.8% → 44.2%** (step-3000; reproduced live 43.8%). Authoritative: [`docs/POC-RESULTS.md`](docs/POC-RESULTS.md) |
| **SFT v1 (verbose 32B traces)** ❌ | "Termination disease": undertrained thinking model, generations peg the 8192 cap. Diagnosis in POC-RESULTS.md (SFT branch) |
| **Inference stack** ✅ | Pinned & verified: vLLM 0.22.1 (cu129) / transformers 5.10.2 / torch 2.11+cu129 — [`docs/vllm-stack-pin.md`](docs/vllm-stack-pin.md) |
| **Concise SFT-v2 dataset** ✅ | 7,149 verify-gated traces, ~13× shorter (median 174 tok) — [`heyalexchoi/qwen3-math-concise-sft-v2`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-concise-sft-v2) |
| **SFT v2 training** ✅ | **~50% MATH-500 greedy** (49.4–50.6 across two greedy passes, vLLM batching noise), **74.6% pass@8**, 46.5% inferred (math-verify) — beats base (35.8), SFT v1 (40.2), and the GRPO POC (44.2). Termination cured: 500/500 clean stop, zero pegs. Model: [`heyalexchoi/qwen3-1.7b-math-sft-v2`](https://huggingface.co/heyalexchoi/qwen3-1.7b-math-sft-v2). **Full analysis: [`docs/sft-v2-results.md`](docs/sft-v2-results.md)** (loss curves, level breakdown, failure modes, diagnosis) |
| **Teacher-trace recovery + master** ✅ | 202/336 dropped traces recovered → **7,356 verified traces (97.9% of MATH train)**, backed up at [`heyalexchoi/qwen3-math-teacher-traces-32b`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-teacher-traces-32b) (provenance in its card) |
| **Concise SFT-v3 dataset** ✅ | **7,340 verify-gated traces (99.8% yield)**, real substitute-back `Verify`, 235B-thinking teacher, ~7.5× shorter (median 309 tok) — [`heyalexchoi/qwen3-math-concise-sft-v3`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-concise-sft-v3). Supersedes v2. Rationale: [`docs/teacher-selection-v3.md`](docs/teacher-selection-v3.md) |
| **SFT v3a training** ⏳ | In progress: from-base SFT on the v3 dataset (clean-only; error episodes → v3b). Plan: [`docs/sft-v3-plan.md`](docs/sft-v3-plan.md) |

## Quickstart

```bash
# 1. Environment — exact pins + cu129 install recipe (read the gotchas in the file):
#    docs/vllm-stack-pin.md  /  requirements-stack.txt

# 2. Secrets (gitignored): secrets.env with HF_TOKEN=, WANDB_API_KEY=, GITHUB_TOKEN=
set -a; source secrets.env; set +a

# 3. Eval a checkpoint on MATH-500 (vLLM preferred backend), then score:
python3 scripts/math500_eval.py --model <model-or-hf-repo> --checkpoint_step <N> --upload
python3 scripts/rescore_math500.py --input outputs/<...>_math500_results.json --upload

# 4. Train:
python3 scripts/sft_train.py    # SFT v3a from base (trains on the HF v3 dataset's `student_trace` field)
python3 scripts/grpo_train.py   # GRPO from base (POC config)

# 5. Regenerate the concise dataset (OpenRouter, Qwen3-235B-thinking teacher pinned to wandb):
python3 scripts/rewrite_full.py --prompt v3 --workers 24
python3 scripts/rewrite_full.py --prompt v3 --retry-failures --retry-max-tokens 16000
```

Pods: read [`docs/runpod.md`](docs/runpod.md) **before any pod operation** (setup, SSH, pre-removal checklist).

## Repo map

| Path | What |
|---|---|
| `scripts/math500_eval.py` + `rescore_math500.py` | **The eval path** (generate → math-verify score). One path for base/SFT/GRPO via `--format {completion,chat}` |
| `scripts/sft_train.py`, `grpo_train.py` | Training |
| `scripts/rewrite_full.py` | **Concise-distillation pipeline** (main pass + `--retry-failures`); prompt lives here, single source of truth |
| `scripts/prepare_data.py`, `generate_traces_32b.py`, `rescore_mathverify.py`, `rerun_truncated.py`, `recover_dropped_traces.py` | Source-trace pipeline (how the 7,356-trace master was made) |
| `docs/sft-v3-plan.md` | **SFT v3 plan** — data decisions, v3 prompt rationale, run sequence |
| `docs/learnings.md` | **Cross-experiment learnings** — durable lessons for the next clean run (canonical-format, don't-distill-out-verification, on-policy errors) |
| `scripts/vllm_parity_canary.py` | vLLM-vs-HF parity diagnostic (used to validate the stack pin) |
| `scripts/fix_sft_tokenizer.py`, `prompts.py`, `setup_runpod_training.sh` | Support |
| `requirements-stack.txt` | **Pinned v2 stack** (train = eval = rollout) |
| `docs/sft-v2-results.md` | **SFT v2 results & post-mortem** — curves, pass@8 by level, failure-mode analysis, diagnosis back to the distillation prompt |
| `docs/POC-RESULTS.md` | Authoritative results & diagnosis (GRPO POC + SFT v1) |
| `docs/vllm-stack-pin.md` | Why these pins; install gotchas (cu129 wheel, torchvision, accelerate) |
| `docs/distill-trace-framework.md` | Concise-trace design + pilot history |
| `docs/runpod.md` | Pod ops runbook |
| `docs/sft-plan.md`, `vllm-eos-investigation.md`, `sft-evidence-ledger.md`, `eval-discrepancy-investigation.md` | Superseded/historical (bannered) |
| `docs/archive/` | Archived v1 README & PLAN journals |
| `RUNS.jsonl` | Append-only run ledger |

`outputs/` and `data/` are gitignored — eval artifacts upload to
[`heyalexchoi/qwen3-math-rlvr-results`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-rlvr-results),
datasets to their own HF repos.

## Key findings (hard-won — details in linked docs)

1. **Pin train = eval = rollout stack.** v1's SFT checkpoint was trained under tf 4.x and evaluated under mismatched/unpinned stacks → months of contradictory eval verdicts. The pinned stack (`vllm-stack-pin.md`) reproduced sane results (vLLM 17/25 vs HF 15/25 on the same checkpoint).
2. **Default `pip install vllm` ships a cu130 wheel** that dies on CUDA-12.8 drivers; the **cu129 wheel works** via minor-version compat. Also: uninstall the base image's torchvision, install accelerate.
3. **1.7B can't imitate 32B verbose reasoning** (median 3.3k-token traces) — SFT v1 collapsed into repetition/non-termination. Fix = capacity-matched data: concise distillation (~174 tok median), not more epochs.
4. **GRPO collapse is invisible to training reward at temp 0.9** — repetition loops are temperature-sensitive (phase transition). Eval greedy on a held-out set every ~500 steps; that's the early-stop signal. POC peak was step 3000; later steps collapsed.
5. **math-verify's timeout uses `signal.alarm()`** — works only in the main thread; in worker threads it raises and silently scores everything False.
6. **Qwen3 thinking mode + greedy decoding loops**; official sampling is temp 0.6 / top_p 0.95 / top_k 20. Blessed greedy evals additionally used repetition_penalty 1.05.
7. **vLLM orphan `EngineCore` processes hold GPU memory** after a parent kill — `nvidia-smi`, kill the pid, verify free memory before relaunch.

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) · [math-verify](https://github.com/huggingface/Math-Verify) · [Open-R1](https://github.com/huggingface/open-r1) · [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) · [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
