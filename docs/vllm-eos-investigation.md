# vLLM checkpoint degeneration — tokenizer & EOS investigation

**Written by:** Claude Opus 4.8 · **Date:** 2026-06-02

Fact-based record of a local (no-GPU) investigation into why the SFT checkpoint
degenerated under vLLM (0/8 on a problem it solved 8/8 under the HF backend in April).
Tiered by evidence strength. Interpretation is labeled as such.

> **Correction (2026-06-02, same day — TWO revisions):**
> 1. An earlier version concluded the EOS `config.json`/`generation_config.json` mismatch was
>    the *likely cause* of the SFT vLLM collapse. Retracted.
> 2. The first retraction then over-corrected to "definitely repetition collapse, not EOS,
>    because `sft_eval.py` passes `stop_token_ids`." That reasoning relied on **current code**,
>    and the code ran **unversioned per run** with no run-level provenance — so it cannot be
>    treated as proof of what the 2026-06-01 run actually did.
>
> **Calibrated position (this version):** the cause of the 2026-06-01 vLLM collapse **cannot be
> determined from saved artifacts.** We do not have that run's log (it died on the pod) nor its
> generations (lost). What we have is circumstantial (see TIER 2). Findings A and B (the
> measured local facts) stand; the *causal* question about the June run is left open, and the
> rerun — which now logs stop config and saves generations — is what will settle it.

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

## TIER 2 — INTERPRETATION (open question + weighed evidence)

**The cause of the 2026-06-01 vLLM collapse is not determinable from saved artifacts.** We
lack the run's log and its generations. Below is the evidence on each side and its strength.

**Evidence that it was NOT an EOS/stop-config problem (i.e. repetition collapse):**
- *Circumstantial (git history):* the committed `sft_eval.py` has passed
  `stop_token_ids=get_stop_ids(...)` to the vLLM `SamplingParams` since commit `49f8d8d`
  (2026-04-10), ~7 weeks before the June run. If that copy ran, vLLM was told to stop on
  151645 and the EOS-config value in `config.json` was irrelevant. **Weakness:** code ran
  unversioned on the pod with no run-level provenance — we cannot prove the running copy
  matched the committed one.
- *Prior observation (Tier-2 recollection, not in current artifacts):* the failing outputs
  were described as `111…` repetition to 8192, persisting despite `repetition_penalty=1.05`.
  A repetition signature is distinct from a coherent-but-unterminated runaway (which is what a
  missing-stop-token bug produces). **Weakness:** this is a recollection; the actual text was
  not saved, so it cannot be re-checked.
- *Stock-vs-ours:* stock `Qwen/Qwen3-1.7B` ran clean (4/4, stopped at 1483–2218 tok) on the
  same stack while ours pegged 8192. **Weakness:** only informative if both used the same stop
  config — again unversioned, so not certain.

**Evidence that leaves EOS/stop-config in play:**
- We have no logged `Stop tokens: [...]` line from the actual June vLLM run, so we cannot
  confirm stop ids were passed in that specific run.
- `config.json` for the SFT checkpoint does advertise 151643 (Finding B), which *would* matter
  if that run's code omitted `stop_token_ids` (e.g. an older or locally-edited copy).

**Net:** repetition collapse of an undertrained/marginal checkpoint is the **better-supported**
explanation (it also fits the April HF data: 572=8/8 but 1994/1349=0/8 — intermittent collapse
independent of any stop config), but it is **not proven**, and the EOS-config path is not
formally excluded for the June run. Both earlier "definitive" framings were overconfident.

**What remains measured and certain (local, current files):**
- The `extra_special_tokens` patch is harmless (Finding A).
- The `config.json`(151643)/`generation_config.json`(151645) EOS mismatch is real (Finding B),
  and `math500_eval.py`'s old vLLM path passed no `stop_token_ids` (Finding C) — a genuine
  latent bug for chat models in *that* script regardless of what caused the June run.

---

## Remediation

- **Backend for the rerun: HF backend** (proven 8/8 on 572 in April). vLLM only as "retry if a
  canary passes" — it must reproduce a known-good result (e.g. 572 ≈ 8/8) before any vLLM SFT
  number is trusted.
- **Make the cause decidable next time** (this is the real fix for the ambiguity above): the
  unified `math500_eval.py` now (a) logs the chat stop-token ids it passes, (b) always passes
  explicit `stop_token_ids` on the chat vLLM path, and (c) saves every generation + token count
  to a non-gitignored per-sample JSONL. After the rerun we can read the actual output text and
  settle "repetition collapse vs unterminated runaway" directly, instead of inferring.
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
