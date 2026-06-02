# SFT branch — evidence ledger (what was measured vs inferred vs unknowable)

**Written by:** Claude Opus 4.8 · **Date:** 2026-06-01
**Why this file exists:** the SFT status flip-flopped four times (see POC-RESULTS.md
"Correction history") largely because methodology and data-provenance gaps let interpretation
get ahead of measurement. This is the canonical, low-interpretation ledger: every claim is
tagged by evidence tier so a future reader can see exactly what is grounded in a durable
artifact and what is not. Interpretation lives in POC-RESULTS.md "Recommended path (v2)"; this
file is the facts.

---

## TL;DR (the one defensible sentence)

On the April HF eval (the only trustworthy SFT run), **a sample is correct if and only if it
terminates within the 8192-token budget — 47 of 48 samples** obey this; the single exception
is problem 1349, which fails for an unrelated reason (figure-parsing + a name-valued answer).
Everything else in this file is either the *why* behind that (inference) or a labelled gap.

---

## TIER 1 — MEASURED (durable artifacts, this session)

### 1a. The discriminator: correct ⟺ terminates-within-budget
Source: `logs/03a_sft_eval_20260410_153625.log` (April, HF backend, our SFT checkpoint, the
trustworthy run). 6 problems × 8 samples = 48. Per-sample token count + correctness, transcribed
and cross-tabulated:

| | sample correct | sample wrong |
|---|---:|---:|
| **terminated** (< 8192 tok) | **26** | **1** |
| **did not terminate** (pegged 8192, or 1-tok stop) | 0 | 21 |

- Every correct sample terminated (26/26). Every non-terminating sample is wrong (21/21).
- The **only** terminated-but-wrong sample is 1349's 3249-token one (see 1c).
- Termination is **not** mere shortness: problem 2584 has a **6577-token correct** sample. The
  predictor is *finishing within budget*, not *being short*.

### 1b. Per-problem outcomes (April HF, dataset order — these 6 are NOT a representative sample)

| Problem | level | subject | correct/8 | sample token counts |
|---|---|---|---:|---|
| number_theory/572 | 3 | Number Theory | **8/8** | 1383–2479 (all terminate) |
| precalculus/807 | 2 | Precalculus | 7/8 | seven 1318–1770; one pegged 8192 |
| algebra/2584 | 3 | Algebra | 6/8 | six 1647–6577; two pegged 8192 |
| prealgebra/1622 | 2 | Prealgebra | 5/8 | five 1203–2573; three pegged 8192 |
| intermediate_algebra/1994 | 5 | Int. Algebra | **0/8** | all eight pegged 8192 |
| algebra/1349 | 2 | Algebra | **0/8** | one 3249 (wrong), one 1-tok, six pegged 8192 |

26/48 raw. **Do not extrapolate a full-500 number** — these 6 are the first 6 in dataset order,
not a sample. (Note: the eval *continued* past these 6; the log just shows where it was read to.)

### 1c. Problem 1349 is a different failure mode (resolves the apparent contradiction with the length law)
The actual problem text (`HuggingFaceH4/MATH-500`, `unique_id test/algebra/1349.json`):
> "The results of a cross-country team's training run are graphed below. Which student has the
> greatest average speed? `[asy] ... [/asy]`" — **expected answer: `\text{Evelyn}`**

- The figure is supplied only as raw **Asymptote code**; the answer is a **student's name**, not a
  number/expression.
- This is a **level-2** problem, squarely in the short-trace regime (L2 median teacher trace
  ≈ 2120 tok), yet it collapsed 0/8. So it is a **counterexample to "long ⟹ collapse"** — and its
  two *terminated* samples (3249-tok and 1-tok) confirm it is **not the repetition-peg mode** that
  killed 1994. Most likely cause: figure-parsing failure and/or a name-valued answer that
  `math-verify` cannot score cleanly (an **eval-validity artifact**, not student capability).
- **Conclusion:** the length/capacity story explains **1994** (genuine L5). It does **not** explain
  **1349**. The honest framing is "intermittent failure-to-terminate **plus** at least one
  ill-posed-for-this-eval problem," not a single clean law.

### 1d. Teacher trace length scales steeply with difficulty — and overruns the eval budget
Source: `data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl`, tokenized with the checkpoint's
own tokenizer (`outputs/sft_checkpoint/tokenizer.json`). **Computed over the 7154 `correct_mathverify==True`
traces — i.e. exactly what `sft_train.py` trains on** (it filters to correct only, `sft_train.py:58,65`):

| level | n | median tok | p90 tok | **% > 8192 (eval budget)** |
|---|---:|---:|---:|---:|
| 1 | 558 | 1772 | 3617 | 1.6% |
| 2 | 1317 | 2120 | 4855 | 3.6% |
| 3 | 1553 | 2777 | 7354 | 8.2% |
| 4 | 1632 | 3543 | 9615 | 13.6% |
| 5 | 2092 | 5774 | 14092 | **31.4%** |
| **all** | **7154** | **3268** | **10073** | **14.8%** |

(The `all` median 3268 / p90 10073 reproduces the figures the original README cited — that
diagnosis was numerically correct.)

### 1e. Train/eval length mismatch is structural
- SFT trained with `max_seq_length = 32768` (`sft_train.py:224`) — long traces learned in full.
- Eval generates with `max_new_tokens = 8192` (`sft_eval.py` default; confirmed in the April log
  header). So **~15% of the trained-on correct traces (31% at L5) are longer than the eval can
  emit.** Any problem whose natural solution exceeds 8192 tokens is an automatic failure *by
  construction* of the eval, before any capability question.

### 1f. Training-data quality (mostly clean; the issue is length, not malformation)
Over all 7490 rows in the trace file:
- `correct_mathverify`: 7154 True / 336 False. The 336 wrong traces are **skipped** by training
  (`sft_train.py` filters), so they did not become targets.
- **0%** of `full_response` lack a `</think>` close — the targets themselves terminate the thinking
  block cleanly.
- **3.2%** (236) don't end in `\boxed{}` within the last 80 chars of `solution`; concentrated at L5
  (120). 16 have a `solution` field > 6000 chars (one runaway 47k-token total trace, L5 geometry,
  whose `solution` was itself truncated at 8192 — an outlier, not the norm).
- **Takeaway:** the student did **not** learn non-termination from malformed targets. It learned a
  *length distribution* it cannot reproduce within budget.

### 1g. Stack/engine sanity (from the 2026-06-01 vLLM diagnostic — separate run)
Source: `/tmp` diagnostic outputs, summarized in POC-RESULTS.md matrix rows 5–6.
- Stock official `Qwen/Qwen3-1.7B` on the 2026-06-01 vLLM solved 572 cleanly **4/4** → engine fine.
- Our checkpoint on that same vLLM gave **0/8** on 572 (vs 8/8 under April HF) → checkpoint is
  fragile / stack-sensitive. Measured fact; cause not isolated (see Tier 3 / open items).

---

## TIER 2 — OBSERVED EARLIER, but NOT in any current artifact
- **The "111…" repetition text.** Current durable artifacts show only that failing samples *pegged
  the 8192 token cap* (token counts), and that a healthy run terminates ~1300–2600 tok. The
  characterization that the failure is specifically *repetition collapse into `111…`* is a
  **prior-session observation**; the actual degenerate generation text is **not in hand**. Treat
  "repetition collapse" as a strong prior, not a thing re-measured this session.

---

## TIER 3 — INFERENCE (motivated, not measured)
- **Long teacher traces *cause* the within-budget failures.** The mechanism (1d/1e + the
  discriminator 1a) makes this very plausible: the model learned to imitate a verbose,
  self-checking CoT style (visible even in short teacher traces — "Hmm… let me verify… that checks
  out…"), and on harder problems it begins a chain it cannot close inside 8192 tokens. But causation
  is **inferred** from the length distribution + termination split, not directly demonstrated
  (would need the actual generation text and/or a budget-sweep eval).
- **Checkpoint fragility under vLLM is "undertrained-marginal model."** Plausible; the alternative
  (the `extra_special_tokens` list→{} reconstruction patch) is not ruled out. See open items.

---

## TIER 4 — CANNOT be known from current artifacts
- **Per-problem teacher traces for these 6 eval problems.** Verified impossible: the 6 are MATH-500
  **test** problems; the teacher traces are MATH **train**. A direct text search found **0/6** in the
  trace file (no leakage). So "the teacher trace for problem 572" does not exist in our data — we can
  only show teacher traces for *similar problem types* (done below).
- **The actual SFT generation text** (success or failure) for any problem. **It was never saved** —
  the April eval logged token counts + correctness only; the June vLLM run logged counts only. To
  see real student samples we must re-generate (GPU). What we *can* show instead: the teacher traces
  the student imitated (below), the per-sample stats (1a/1b), and the stock model's clean 572 solve.

---

## Representative teacher traces (what the student was trained to imitate)
These are **teacher (Qwen-32B) traces**, not student output. Same problem *types* as the eval 6
(the eval problems themselves are not in train — Tier 4). Token counts via the checkpoint tokenizer.

- **Short clean (id 4834, L3 Number Theory, "0.̄42 → a/b", ans 47): full 1288 tok** (think 820 /
  sol 464). Even this *short* trace meanders: *"Hmm, repeating decimals can be tricky… Let me
  think… that checks out. So I think my answer is correct."* → the verbose, self-verifying style is
  present at every length.
- **Long L5 (id 3704, Int. Algebra functional equation, ans 6): full 7430 tok** (think 6503).
  Heavy meander, many "let me verify / seems solid" passes before concluding. Just under the 8192
  budget — a student that imitates this style on a similar problem has almost no margin.
- **"Average speed" analog (id 1133, L4 Algebra, the Dave problem, ans 12): full 1856 tok.** Note:
  no Asymptote-figure average-speed problem exists in train — so 1349's *figure* form is genuinely
  off-distribution for what the student saw.
- **Longest in dataset (id 3145, L5 Geometry, paper-folding, ans 338): full 47107 tok** (think
  38910, sol itself truncated at 8192). A trace **5.7× the eval budget** — physically impossible for
  the student to reproduce within 8192. This is the extreme tail of what it was asked to learn.

---

## What was checked vs NOT checked (process ledger)

**Checked this session (with the artifact):**
- April per-sample token/correctness split → 1a/1b (`logs/03a_sft_eval_20260410_153625.log`).
- 1349 problem text + answer format → 1c (`HuggingFaceH4/MATH-500`).
- Teacher length by level over the *trained* (correct-only) set → 1d (trace file + tokenizer).
- Train vs eval token budget → 1e (`sft_train.py`, `sft_eval.py`, April log header).
- Training filter + target termination quality → 1f (`sft_train.py`, trace file).
- Test∉train (no leakage) → Tier 4 (text search, 0/6).
- Held-out eval-loss plateau by ~epoch 2 → POC-RESULTS.md (`trainer_state.json`).

**Checked 2026-06-02 (local, no GPU) — see [vllm-eos-investigation.md](vllm-eos-investigation.md):**
- `extra_special_tokens` list→{} patch → **harmless / suspect cleared.** All special-token ids
  identical (list vs {}) on both checkpoints, matching base; round-trip identical.
- SFT vLLM collapse → **NOT an EOS-config problem.** `sft_eval.py` already passes
  `stop_token_ids` incl. 151645 (vLLM honors it over config.json); given the correct stop id it
  still pegged 8192 → the model never emitted its stop → repetition collapse of the undertrained
  checkpoint (matches 2026-06-01 conclusion). The `config.json`(151643)/`generation_config`(151645)
  EOS mismatch is real but is a latent bug only in `math500_eval.py`'s vLLM path, not the cause here.
  Rerun backend = HF (proven 8/8 April); vLLM only behind a 572≈8/8 canary.

**NOT checked / still open:**
- Actual SFT generation text (never saved; needs canary-gated GPU re-run to capture).
- GPU reproduction (no GPU this session). Root cause of vLLM-worse-than-HF collapse on 572 is
  best-explained as numerical sensitivity of a marginal model, not fully isolated.
- A trustworthy full-500 SFT c/n (needs April HF + tf~5.5.3 env reproduced, gated on 572≈8/8 canary).
- Whether raising the eval budget to 16–32k recovers the within-budget failures (directly tests the
  Tier-3 length-causation inference — the single highest-value next eval).
