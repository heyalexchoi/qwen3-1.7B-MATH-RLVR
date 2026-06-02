# vLLM checkpoint degeneration — tokenizer & EOS investigation

**Written by:** Claude Opus 4.8 · **Date:** 2026-06-02

Fact-based record of a local (no-GPU) investigation into why the SFT checkpoint
degenerated under vLLM (0/8 on a problem it solved 8/8 under the HF backend in April).
Tiered by evidence strength. Interpretation is labeled as such.

> **Correction (2026-06-02, same day):** an earlier version of this doc concluded the EOS
> `config.json`/`generation_config.json` mismatch was the *likely cause* of the SFT vLLM
> collapse. That conclusion was wrong and is retracted below (see TIER 2). The script that
> produced the collapse (`sft_eval.py`) already passes an explicit `stop_token_ids` to vLLM,
> which overrides `config.json`, so the EOS mismatch cannot have caused it. The EOS mismatch is
> real but is a *separate latent bug* affecting only `math500_eval.py`'s vLLM path. Findings A
> and B (the measured facts) stand; only the causal interpretation changed.

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
`generation_config.json` to 151645 on local, pod, and HF Hub.

### Finding C — the two eval scripts handle vLLM stop tokens differently

- `scripts/sft_eval.py` (the script that ran the SFT eval, incl. the 2026-06-01 vLLM run):
  its vLLM path builds `SamplingParams(..., stop_token_ids=get_stop_ids(tokenizer))`
  (lines 180/190 → 138–143). `get_stop_ids` = `{eos_token_id} ∪ {<|im_end|> id}`, which for
  the SFT checkpoint includes **151645**. vLLM honors `stop_token_ids` regardless of what
  `config.json` says. So this script **already tells vLLM to stop on `<|im_end|>`** — the
  `config.json` value of 151643 is irrelevant to it.
- `scripts/math500_eval.py` (the GRPO/base script): its vLLM path passes only
  `stop=list(STOP_STRINGS)`, **no** `stop_token_ids`. For this script, the stop id is left to
  vLLM's default (read from `config.json` = 151643). This is a real latent bug *for that
  script* — but it was never used for the SFT eval.

---

## TIER 2 — INTERPRETATION (corrected)

**The EOS `config.json` mismatch did NOT cause the SFT vLLM collapse.** The 2026-06-01 vLLM
run used `sft_eval.py`, which (Finding C) already passed `stop_token_ids` including 151645.
vLLM was therefore told to stop on `<|im_end|>` and still pegged `max_tokens` (8192) on every
sample. A run that is given the correct stop id and still never stops is a model that **never
emitted the stop token** — i.e. **repetition collapse**, not a stop-configuration problem.
This matches the prior-session observation (`111…` repetition to 8192, persisting despite
`repetition_penalty=1.05`) and the 2026-06-01 RUNS.jsonl refinement ("collapse is SPECIFIC to
our undertrained SFT checkpoint").

The stock-vs-ours comparison points the same way, not the other way: stock `Qwen/Qwen3-1.7B`
and our checkpoint both go through the *same* `get_stop_ids` in `sft_eval.py`, so the
difference between stock (4/4 clean, stopped at 1483–2218 tok) and ours (0/8, all pegged 8192)
is **the model, not the stop config**. Stock emits `<|im_end|>` and stops; ours loops and
never reaches it.

**What remains valid:**
- The `extra_special_tokens` patch is harmless (Finding A — measured).
- The `config.json`/`generation_config.json` EOS mismatch is real (Finding B — measured) and
  is a genuine latent bug, but only in `math500_eval.py`'s vLLM path (Finding C). It would
  break a *chat* model run through that script; it did not cause the observed SFT collapse.

**Best current explanation of the SFT vLLM collapse (unchanged from the 2026-06-01
conclusion):** the cold-start SFT checkpoint is undertrained/marginal and prone to repetition
collapse. Under HF (April) it solved some problems (572 = 8/8) and collapsed on others
(1994/1349 = 0/8). Under the vLLM 0.8.5 stack it collapsed even on 572 — most plausibly
because small numerical differences between the vLLM and HF execution paths tip a borderline
model into the repetition attractor. Cause not fully isolated, but it is **not** EOS config
and **not** the `extra_special_tokens` patch.

---

## Remediation

- **Backend for the rerun: HF backend** (proven 8/8 on 572 in April). A stop-token fix cannot
  rescue a model that is looping rather than failing to stop, so vLLM does not get us a
  trustworthy SFT number for free. Treat vLLM as "retry only if a canary passes" — it must
  reproduce a known-good result (e.g. 572 ≈ 8/8) before any vLLM SFT number is trusted.
- **Still fix the latent bug** when unifying the scripts: the merged vLLM path must pass
  explicit `stop_token_ids` (carry over `get_stop_ids` from `sft_eval.py`) so a chat model is
  never left depending on `config.json`'s 151643. This is good hygiene, not a fix for the
  collapse.
- The `extra_special_tokens` patch is exonerated and needs no change.

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
