# vLLM checkpoint degeneration — tokenizer & EOS investigation

**Written by:** Claude Opus 4.8 · **Date:** 2026-06-02

Fact-based record of a local (no-GPU) investigation into why the SFT checkpoint
degenerated under vLLM (0/8 on a problem it solved 8/8 under the HF backend in April).
Tiered by evidence strength. Interpretation is labeled as such.

Environment for these tests: local box, `transformers==5.5.1`, no GPU. Checkpoints read
from `outputs/sft_checkpoint/` and `outputs/grpo_checkpoint/` (local copies), plus
`Qwen/Qwen3-1.7B-Base` from HF cache.

---

## TIER 1 — MEASURED (re-runnable locally)

### Finding A — the `extra_special_tokens` patch is harmless (suspect CLEARED)

The patch in `load_tokenizer_safe()` coerces `extra_special_tokens` from a list to `{}`
when transformers rejects the list form. Question: does emptying that field change
tokenization?

Config state as saved:
- `outputs/sft_checkpoint/tokenizer_config.json`: `extra_special_tokens` = `{}` (already a
  dict); `added_tokens_decoder` count = 0.
- `outputs/grpo_checkpoint/tokenizer_config.json`: `extra_special_tokens` = a **list** of
  Qwen special tokens (`<|im_start|>`, `<|im_end|>`, …); `added_tokens_decoder` count = 0.

Test: load four tokenizers — base; SFT; GRPO with the list kept; GRPO with the list patched
to `{}` — and tokenize the probe tokens plus a chat sample. Result, **identical across all
four**:

| token | id | single token? |
|-------|----|--------------:|
| `<think>` | 151667 | yes |
| `</think>` | 151668 | yes |
| `<|im_start|>` | 151644 | yes |
| `<|im_end|>` | 151645 | yes |
| `<|endoftext|>` | 151643 | yes |

Sample (`<|im_start|>user … <think> … </think> \boxed{0}<|im_end|>`) round-trips identically
(encode→decode) in all four; 26 tokens in every case.

**Conclusion (measured):** the special tokens live in `tokenizer.json` /
`special_tokens_map.json`, not in `extra_special_tokens`. Emptying that field does not change
any token id or the round-trip. The patch is **not** the cause of the vLLM degeneration.

### Finding B — the SFT checkpoint's two config files disagree on EOS

Declared `eos_token_id` by file:

| checkpoint | `config.json` | `generation_config.json` | tokenizer `eos_token` |
|------------|--------------:|-------------------------:|-----------------------|
| SFT  | **151643** (`<|endoftext|>`) | **151645** (`<|im_end|>`) | `<|im_end|>` (151645) |
| GRPO | 151643 | `[151643]` | `<|endoftext|>` (151643) |
| Base | 151643 | (no generation_config) | `<|endoftext|>` (151643) |

The SFT checkpoint is **internally inconsistent**: `config.json` says stop on 151643,
`generation_config.json` and the tokenizer say stop on 151645. The SFT model was trained in
chat/thinking format, whose turn terminator is `<|im_end|>` (151645); it effectively never
emits a bare `<|endoftext|>` (151643). The GRPO and base checkpoints are internally
consistent (151643 everywhere).

**Prior context (from `memory/math-rlvr.md` / PLAN.md):** the `generation_config.json` value
of 151645 is not accidental — an earlier session diagnosed the Qwen-base EOS problem (TRL
guidance: set `eos_token="<|im_end|>"` for Qwen base SFT) and deliberately patched
`generation_config.json` to 151645 on local, pod, and HF Hub. That fix is exactly why the HF
backend stops correctly. The gap this investigation identifies: the patch was applied to
`generation_config.json` but **not** to `config.json` (still 151643), and the vLLM code path
reads `config.json` and passes no explicit `stop_token_ids` — so the known fix never covered
vLLM. This is a fix-coverage gap, not a newly discovered EOS bug.

### Finding C — the vLLM path passes no explicit stop token id

`scripts/math500_eval.py` `run_eval_vllm()` builds `SamplingParams(..., stop=list(STOP_STRINGS))`
with **no** `stop_token_ids`. So which EOS id terminates a vLLM generation is left to vLLM's
default, which is read from the model's config — not chosen by us.

---

## TIER 2 — INTERPRETATION (consistent with the facts, not yet GPU-confirmed)

**Most likely cause of the 0/8 vLLM degeneration:** EOS mismatch. If the vLLM version used
took the stop id from `config.json` (151643, `<|endoftext|>`) rather than
`generation_config.json` (151645, `<|im_end|>`), then because the SFT chat model emits
`<|im_end|>` and not `<|endoftext|>`, vLLM never saw a stop token and generated to
`max_tokens` on every sample — which presents exactly as the observed runaway/degeneration.
The HF backend's `generate()` honors `generation_config.json` (151645) by default, so it
stopped correctly → April's 8/8.

Supporting (not proof):
- The checkpoint that worked under vLLM (GRPO) has consistent EOS; the one that failed (SFT)
  does not.
- The 2026-06-01 diagnostic (RUNS.jsonl) found stock `Qwen/Qwen3-1.7B` solved 572 cleanly 4/4
  on the *same* vLLM 0.8.5 stack (stopped at 1483–2218 tokens) while our SFT checkpoint pegged
  8192 on all samples. That left "engine fine, our checkpoint specifically fails" as an
  unexplained puzzle — the EOS-coverage gap explains it: the stock model stops correctly under
  vLLM, so its effective stop id must be reached, whereas our SFT `config.json` advertises
  151643 which the chat model never emits. **To confirm:** compare stock `Qwen/Qwen3-1.7B`'s
  `config.json` `eos_token_id` against our SFT checkpoint's (151643). If stock includes 151645
  and ours does not, the hypothesis is corroborated. (Not done here — stock non-base model not
  in local cache.)
- That same 2026-06-01 run ruled out a sibling sub-suspect: neutralizing
  `generation_config.json` `do_sample` did not fix it. It never tested the eos id / explicit
  `stop_token_ids`.
- Whether a given vLLM version reads `generation_config.json` by default has changed across
  vLLM releases. The exact vLLM version of the failing run was not recorded, so the default
  in effect then is unknown. Either way, not passing an explicit `stop_token_ids` makes the
  outcome depend on that default — a fragility regardless of cause.

**Not yet established:** this has not been reproduced on a GPU. It is a hypothesis from
config inspection, strong but unconfirmed.

---

## Remediation (applies to the rerun regardless of which way the hypothesis breaks)

Pass the stop token id explicitly so the result no longer depends on a vLLM default:
- add `stop_token_ids=[151645]` (and optionally 151643) to the vLLM `SamplingParams`, **or**
  load the LLM with `generation_config="auto"` so `generation_config.json` (151645) is applied.
- Gate any vLLM run behind a canary: it must reproduce the known-good result on a problem the
  HF backend solved before any vLLM number is trusted.

**Net effect on backend choice:** vLLM is usable for the rerun (the efficiency we want), with
the explicit stop id + canary. The `extra_special_tokens` patch is exonerated and needs no
change.

---

## What was NOT checked
- GPU reproduction of the degeneration or of the fix (no GPU in this session).
- The exact vLLM version used in the original failing run (not recorded).
- Whether April's HF-backend 8/8 used 151645 as eos in practice (inferred from
  `generation_config.json` default, not from a saved generation log — the SFT generations
  themselves remain lost; see [sft-evidence-ledger.md](sft-evidence-ledger.md)).
- Tokenization was tested under `transformers==5.5.1`; the April runs used an older
  transformers. Token ids for this vocab are stable across versions, so the round-trip result
  is expected to hold, but it was not re-tested under the older version.
