# Eval Discrepancy Investigation

**Status:** Open — root cause not yet determined  
**Date:** 2026-04-13  
**Problem:** GRPO step-3000 evaluated at 44.2% greedy from a local checkpoint on the H100 training pod, but re-evaluation of HF Hub step-3000 (commit `63870ec239b2`) via vLLM yields 11.2% greedy — collapsed repetition output.

---

## The discrepancy

| Eval | Source | Backend | Greedy | Pass@8 | Inferred |
|------|--------|---------|--------|--------|----------|
| Original (Apr 11) | Local path on H100 pod: `outputs/grpo_checkpoint` | HF `generate()` | **44.20%** | **71.40%** | **36.83%** |
| Re-eval (Apr 13) | HF Hub revision `63870ec239b2` | vLLM | 11.20% | 26.00% | 9.22% |

The original eval result is in `outputs/grpo_step3000_math500_results.json` (also uploaded to `heyalexchoi/qwen3-math-rlvr-results` at commit `584631bc4bd5`, later overwritten by the re-eval at `e4e52b79eb2f`).

Generation quality is visibly different. Original eval produces coherent multi-step math:

```
The point (0,3) is on the positive y-axis, so θ = π/2. The distance from the
origin to the point is r = 3. Therefore, the polar coordinates are (3, π/2).
```

Re-eval produces repetition loops:

```
θ = tan⁻¹(3/0) = tan⁻¹(3/0) = tan⁻¹(3/0) = tan⁻¹(3/0) = ...
```

---

## Two hypotheses

### Hypothesis 1: Eval infrastructure difference (vLLM vs HF generate)

The original eval used HF `transformers` `model.generate()`. The re-eval used vLLM. If vLLM behaves differently on the same weights, that would explain the gap.

**Known risk: chat template contamination.** The Qwen3 tokenizer has a 4,116-character chat template (`chat_template.jinja`). HF's `model.generate()` with a raw string prompt ignores this. But vLLM creates its own tokenizer internally — if it applies the chat template to the raw few-shot prompt, the input to the model would be corrupted with `<|im_start|>user\n...` wrapping, which would destroy the base-model few-shot pattern.

**Other possible vLLM differences:**
- Different tokenization of the prompt (e.g., BOS token handling, whitespace normalization)
- Different stop-string handling (`stop_strings` in HF vs `stop` in vLLM `SamplingParams`)
- Different EOS token behavior (generation_config has `eos_token_id: [151643]`)
- Precision/kernel differences affecting greedy argmax on near-tied logits

**What would confirm this:** Run the eval on HF Hub step-3000 weights using the HF `generate()` backend (not vLLM) on a GPU pod. If it produces ~44%, vLLM is the problem. If ~11%, the weights are genuinely different.

### Hypothesis 2: Different checkpoint weights

The original eval loaded from `/workspace/qwen3-math-rlvr/outputs/grpo_checkpoint` on the H100 pod. This is TRL's checkpoint output directory, which gets **overwritten every `save_steps=500` steps**. What was in that directory at eval time may not be what got pushed to HF Hub as "step 3000."

Possible scenarios:
- The local checkpoint was from a different training step than 3000
- The local checkpoint was from a different training run entirely
- The HF Hub push for step 3000 uploaded a different model state than what was saved locally
- The `_resolve_step_revision()` function mapped "step 3000" to the wrong HF commit

---

## What we verified

### Prompt format: unchanged

`scripts/prompts.py` has not changed since its creation (commit `49f8d8d`). The few-shot prompt and `create_prompt()` function are identical between the original eval and the re-eval. Both use the same raw text format — no chat template, no system prompt.

### Generation parameters: identical

Both evals use `max_new_tokens=2048`, `temperature=0` for greedy, `n_samples=8` at `temperature=0.7` for pass@8. Stop strings are `["\n\nProblem:"]` in both.

### Local machine checkpoint vs HF Hub: identical (but possibly circular)

**SHA256 of `model.safetensors`:**
```
HF Hub step 3000 (63870ec): 3b3697bb8ab6e4963d7b565ded50c2d2b9ca2068ec6108a2c38242a338374015
Local outputs/grpo_checkpoint: 3b3697bb8ab6e4963d7b565ded50c2d2b9ca2068ec6108a2c38242a338374015
```

**Warning: this comparison may be circular.** The local `outputs/grpo_checkpoint/model.safetensors` on this machine could have been downloaded from HF Hub (via `snapshot_download()` in the setup script), not rsynced from the training pod. If so, we compared HF Hub with itself and proved nothing about whether the pod's checkpoint matched HF Hub.

The provenance of the local file is ambiguous:
- File dates are Apr 11 13:15 (matches HF step-3000 push time of Apr 11 13:17)
- The setup script downloads to `outputs/grpo_checkpoint/last-checkpoint/`, but the files are at `outputs/grpo_checkpoint/` directly
- No `trainer_state.json` in the local copy (would have confirmed the step number)
- Could be an rsync from the pod OR a HF Hub download — we don't know

### Generation config: identical

Both local and HF Hub have the same `generation_config.json`:
```json
{"do_sample": false, "eos_token_id": [151643], "max_new_tokens": 2048,
 "pad_token_id": 151643, "transformers_version": "5.5.3"}
```

### HF Hub commit history: single training run

All HF Hub commits (steps 500–7496) come from the same training run, pushed sequentially from the H100 pod. No interleaved runs or manual uploads.

---

## What we have NOT verified

1. **Whether HF `generate()` backend reproduces 44.2% on HF Hub weights** — this is the definitive test. Requires a GPU.
2. **Whether vLLM applies the chat template** to raw string prompts for this model — suspected but not confirmed.
3. **Whether the H100 pod's local checkpoint actually matched HF Hub** — the pod (`gol7yudqrlfn48`) is stopped; its volume may still have the original checkpoint.
4. **Whether the re-eval for other steps (2500, 3500, 4000, 5000) would also differ under HF backend** — all were evaluated with vLLM only.

---

## Recommended next steps

1. **Spin up a GPU eval pod and run HF-backend eval on HF Hub step 3000.** This is the single test that distinguishes hypothesis 1 from hypothesis 2:
   ```bash
   # Force HF backend by not installing vLLM
   python scripts/math500_eval.py \
     --model heyalexchoi/qwen3-1.7b-math-grpo \
     --checkpoint_step 3000 \
     --max_samples 10  # 10 problems is enough to see coherent vs collapsed
   ```
   If coherent (~44%) → vLLM is the problem, fix vLLM usage, re-eval all steps.  
   If collapsed (~11%) → the local checkpoint on the pod was genuinely different, investigate pod volume.

2. **If vLLM is the problem, check chat template application:**
   ```python
   from vllm import LLM
   llm = LLM(model="heyalexchoi/qwen3-1.7b-math-grpo", revision="63870ec239b2")
   # Check if tokenizer has chat_template set
   print(llm.get_tokenizer().chat_template)
   # Compare tokenization of raw prompt vs what vLLM actually feeds the model
   ```

3. **If the weights are genuinely different, check the stopped H100 pod volume** for the original `outputs/grpo_checkpoint` directory.

---

## Related files

- Original eval result: `outputs/grpo_step3000_math500_results.json` (local) / HF results repo commit `584631bc4bd5` (original), `e4e52b79eb2f` (overwritten by re-eval)
- Rescored result: `outputs/grpo_step3000_math500_mv_rescored.json`
- Eval script: `scripts/math500_eval.py`
- Prompt format: `scripts/prompts.py`
- Model repo: `heyalexchoi/qwen3-1.7b-math-grpo`
- Results repo: `heyalexchoi/qwen3-math-rlvr-results`
- Wandb run (training): `ckz7jwil`
