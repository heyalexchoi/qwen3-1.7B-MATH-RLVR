# SFT v3 plan — clarity-first distillation + recovery traces

*Written by Claude Opus 4.8 · 2026-06-12 · Status: data ready; prompt DRAFTED, pending Alex's review; runs NOT yet green-lit.*

Goal: fix the SFT-v2 failure modes ([`sft-v2-results.md`](sft-v2-results.md) §4 — decorative
Verify, zero-diversity wrong plans, knowledge gaps) with better data, then GRPO on top.

## 1. Source data (done)

Master teacher-trace file: `data/traces/qwen32b_math_traces_master.jsonl` —
**7,356 verified-correct Qwen3-32B traces, 97.9% of MATH train** (L5 drop rate 9% → 3.1%).
Backed up with full provenance (recovery stages, residual-134 analysis) at
[`heyalexchoi/qwen3-math-teacher-traces-32b`](https://huggingface.co/datasets/heyalexchoi/qwen3-math-teacher-traces-32b).

The 336 originally-dropped traces were recovered 2026-06-12 by `scripts/recover_dropped_traces.py`
(Stage A: automated answer-form normalization re-scored by math-verify, 27 — trace text untouched;
Stage B: best-of-4 regeneration with box-the-value instruction, 175/309, ~$11 OpenRouter).
Residual 134 = genuine teacher misses (L5 Geometry/Int.Algebra-heavy, 29 with `[asy]`) — not chased.

Decision: **MATH train alone suffices** for MATH-500 (same distribution by construction; dominant
failures are trace-content problems, not coverage). No NuminaMath. `amc_aime` = optional later insurance.

## 2. Distillation prompt (DRAFT — awaiting Alex's review)

Single source of truth: `SYSTEM_V3` in `scripts/rewrite_full.py` (run with `--prompt v3`).
Changes vs v2, each mapped to a failure mode, are documented in the comment block above the prompt.
Key decisions (debated 2026-06-11/12):

- **Clarity-first, no token target** — length is enforced at the data-filter stage, not in-prompt.
- **Plan-check in Classify** — precondition confirmed before committing (vs v2's bare "commit").
- **Verify = substitute the answer back into the original problem, arithmetic shown** — independent
  of the derivation so it can actually fail (v2's "one line quick check" trained decorative Verify).
- **Conditional error episodes** — preserve exactly one genuine caught-mistake per trace *if present*
  (~10–15% of raw traces), never invent. One prompt, not two: real recoveries are too rare to split.
- Mid-solve `check:` lines only at cheaply-falsifiable junctures; rhetorical checks banned.
- **Canonical answer-form clause** — the teacher is told to box a bare math-verify-parseable value
  (digits not words; no `\text{}`/`\mbox{}`/units; simplest form). This is free on the *data* side
  (no baseline depends on a training-data prompt) and removes our reliance on the `_variants`
  normalization that papered over non-canonical boxes during recovery.

> **Lesson carried forward → [`docs/learnings.md`](learnings.md) L1.** Baking the canonical answer
> format into one shared prompt from day one would have prevented the ~336 drops and the
> `_variants` + two-stage recovery layer. (Eval-side adoption requires a re-baseline — see L1's
> caveat.) The v3 distill prompt adds the clause on the data side.

## 3. Run sequence (each GPU/API phase needs explicit go)

1. **Prompt smoke test**: `rewrite_full.py --prompt v3 --limit 50` → inspect verify-gate yield,
   Verify-section quality (does it actually substitute back?), length distribution vs v2.
2. **Full v3 rewrite** of the 7,356 master traces → filter → push `concise-sft-v3` to HF;
   deprecate `concise-sft-v2` (banner: superseded, decorative-Verify prompt).
3. **v3a SFT from base** (clean-only) — doubles as the happy-path ablation arm.
4. **On-policy error collection**: sample v3a k=8 on the hard slice; wrong prefixes are in-template
   and on-policy (this is why we don't collect errors from the base model — wrong format).
   **Bucket failures by error type** (arithmetic-slip / wrong-method / wrong-setup / misread) and
   build corrections across buckets, so the set isn't dominated by cheap arithmetic-slip fixes and
   actually teaches plan/setup recovery. Teacher corrects prefixes → recovery set (failed `check:` →
   `fix:` format). See [`docs/learnings.md`](learnings.md) L3.
5. **v3b SFT from base** = clean + 20–30% recovery (L1–L2 stay 100% clean).
6. **Ablation v3a vs v3b**: greedy / pass@8 / 0-of-8 count; canary = L1–L2 length & accuracy
   regression (detects learned wrong-first-step behavior).
7. **GRPO v2** from the winner. Mandatory periodic greedy eval (no eval loss in GRPO).

Training: from **base**, not from sft-v2 (entrenched decorative-Verify priors; ~$1/70min per run).
