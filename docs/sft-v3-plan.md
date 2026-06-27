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

## 4. v3a MATH-500 eval plan (staged 2026-06-27 — needs explicit go before pod launch)

v3a is trained and on HF (`heyalexchoi/qwen3-1.7b-math-sft-v3a`). The repo carries **both**
checkpoints: top-level weights = the shipped **best** (`load_best_model_at_end`, step 800 / epoch
1.83, eval_loss 0.34141); the `last-checkpoint/` subfolder = the **final** (step 1308 / epoch 3.0,
eval_loss 0.34171). The plateau gap is 0.0003 — within noise (see best-vs-final below).

**Methodology — pinned for apples-to-apples with v2** (the only comparison that matters: did v3's
honest-Verify dataset beat v2's decorative-Verify one):
- Same stack: vLLM 0.22.1+cu129 / tf 5.10.2 (`requirements-stack.txt`) — identical to the v2 eval.
- `--format chat` defaults (Qwen3 thinking): max_new_tokens 8192, temp 0.6, top_p 0.95, top_k 20,
  repetition_penalty 1.05. pass@1 (greedy) + pass@8.
- Scorer: `rescore_math500.py` (math-verify, authoritative). Upload generations + results to
  `heyalexchoi/qwen3-math-rlvr-results`; append a RUNS.jsonl row.

**Headline run (the deliverable):** the shipped best.
```bash
python scripts/math500_eval.py --model heyalexchoi/qwen3-1.7b-math-sft-v3a --format chat
python scripts/rescore_math500.py --input eval_results/<...>_results.json --upload
```

**Success ladder** (greedy / pass@8, all math-verify): base 35.8 / 65.0 → SFT-v1 40.2 → GRPO-3000
44.2 → **SFT-v2 50.6 / 74.6 ← the bar to beat**. v3a's job is to clear v2; the honest-Verify
distillation is the only deliberate change.

**Best-vs-final (optional, a teaching demo — not a tiebreaker):** the 0.0003 eval-loss gap will land
*inside* vLLM continuous-batching nondeterminism (v2 showed 1.2pt: 49.4 vs 50.6 on the *same*
checkpoint). So this run is expected to be indistinguishable; run it only to *show* that, not to
decide anything. eval-loss early-stop is the standard default and is what we ship; the real arbiter
for a generation task is the downstream metric, not the token-CE proxy. To eval the final:
```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("heyalexchoi/qwen3-1.7b-math-sft-v3a", allow_patterns="last-checkpoint/*",
                  local_dir="ckpt_final")   # last-checkpoint is a subfolder, not a hub revision
PY
python scripts/math500_eval.py --model ckpt_final/last-checkpoint --format chat
```

**Base re-baseline (optional, low value):** canonical base 35.8/65.0 is POC-era (2026-04-11,
pre-pinned-stack; re-verified 05-30). The v3a-vs-v2 headline needs no base run (both chat-format under
the pinned stack). A fresh base eval would only re-anchor the ladder under one stack — and base uses
`--format completion` (few-shot), so a stack-matched base number is *still* not regime-matched to the
chat-format SFT models. Marginal GPU for marginal gain; skip unless we want the whole ladder re-run
under one roof. (The stray `outputs/_v3a_pod_rescue/base_qwen3-1.7b_math500_eval.log` = 31.6/58.4 is a
non-canonical base run from the *old* inline-scoring flow; provenance murky, **discarded** — do not
reconcile 31.6 vs 35.8, that's exactly the stack-mismatch noise README finding #1 warns about.)
