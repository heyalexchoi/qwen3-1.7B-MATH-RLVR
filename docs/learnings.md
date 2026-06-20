# Learnings

*Written by Claude Opus 4.8 · started 2026-06-20.*

Durable, cross-experiment lessons — the kind worth re-reading before the *next* clean run, not
buried in a single experiment's plan/post-mortem. Each entry: what bit us, the one rule going
forward, and what it would have saved. Newest first.

---

## L1 · Bake the canonical answer format into one shared prompt, from day one

**Context.** Generating the Qwen3-32B teacher traces, then distilling them, then SFT'ing the student.

**What happened.** The *original* generation prompt never stated the answer format. math-verify is
lenient symbolic equivalence (it equates `\frac{1}{2}`≡`0.5`, `\dfrac`≡`\frac`, ignores spacing), but
it predictably **fails** on a small, knowable set:

- words-for-digits (`four` vs `4`),
- `\text{}` / `\mbox{}` / unit wrappers,
- percent-vs-decimal (`56\%` vs `0.56` vs `56`),
- choice-letter-not-value (`B` where the value is expected).

Because the teacher was free to emit any of those, ~336 otherwise-correct traces were scored wrong
and dropped at the correct-only gate. Recovering them cost a `_variants()` normalization layer
(`scripts/recover_dropped_traces.py`) **plus** a two-stage recovery pass (Stage A re-score, Stage B
best-of-4 regen, ~$11 OpenRouter). The normalization layer is technical debt that exists *only*
because the format was never asserted upstream.

**Rule going forward.** A one-line canonical-form rule — *box a bare math-verify-parseable value:
digits not words; no `\text{}`/`\mbox{}`/units; simplest form* — baked into a **single shared
prompt** used identically at generation / distillation / SFT-target. Do it from day one; then every
normalization layer can be deleted instead of written.

**The one caveat.** Changing the **eval / GRPO** prompt to add this rule invalidates current
baselines (base 35.8, SFT-v2 49.4 on MATH-500) — those numbers were measured under the old prompt.
Only adopt the rule on the eval side as part of a deliberate, re-baselined restart. **Never change
the eval prompt silently.** (On the *data* side — generation/distillation — it's free: no baseline
depends on a training-data prompt.)

**Status in v3.** SFT-v3's distillation prompt (`SYSTEM_V3` in `scripts/rewrite_full.py`) adds this
clause on the data side. The re-gate is plain math-verify (no `_variants`), so a non-canonical box
gets dropped at distillation rather than papered over.

---

## L2 · Don't distill *out* the behavior you want the student to learn

**Context.** SFT-v2 concise distillation.

**What happened.** The v2 prompt told the teacher to give a "one-line quick check" for `Verify`. The
student dutifully learned to *assert* a check without ever substituting the answer back — decorative
verification. Final answers were still math-verify-correct (gate guarantees it), but the *process*
the student imitated contained no real self-checking. See [`sft-v2-results.md`](sft-v2-results.md)
§4.1.

**Rule going forward.** If a behavior matters (verification, error-recovery), the demonstration must
*exhibit* it in a way that could actually fail — `Verify` substitutes the answer back into the
original problem with arithmetic shown, independent of the derivation. A step that can't fail teaches
nothing. v3 fixes this (substitute-back Verify; conditional single real error-episode, never
invented).

---

## L3 · Collect on-policy errors from the *trained* student, not the base model

**Context.** Building the v3b self-correction set (planned).

**What happened (anticipated, designed around).** If you harvest wrong attempts from the *base* model
to build correction examples, the wrong prefixes are off-distribution — wrong *format*, not just
wrong *reasoning* — so the student never actually produces them and the corrections don't transfer.

**Rule going forward.** Sample wrong rollouts from the **policy you're about to train** (v3a), in the
template it already speaks, so the corrected prefixes are on-policy. And **bucket the failures by
error type** (arithmetic-slip / wrong-method / wrong-setup / misread) before building corrections —
otherwise the set fills up with cheap arithmetic-slip fixes and never teaches plan/setup recovery,
which is where the model actually loses points. See [`sft-v3-plan.md`](sft-v3-plan.md) §3 steps 4–5.
