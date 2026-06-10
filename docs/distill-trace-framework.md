# Concise-Trace Distillation Framework (SFT v2)

*Written by Claude Opus 4.8 — 2026-06-07. Draft for Alex's review; the "Open decisions" section lists what's not yet settled.*

## Why

The first SFT cold-start produced an **undertrained thinking model that fails to terminate** on a subset of problems: it meanders, backtracks, and runs to the 8192-token cap in `111…` repetition instead of closing `</think>` and boxing an answer. This is a *length/termination* disease, not (only) a capability one:

- 32B teacher traces: median **3268 tok**, p90 **10073 tok**.
- **14.8%** of the 7154 correct-only training traces exceed 8192 tokens (31.4% at Level 5) — they cannot fit the eval budget *by construction*.
- Live SFT greedy eval: **~51% of generations peg 8192** even at temperature 0.
- The training targets themselves are clean (0% miss `</think>`, only 3.2% lack `\boxed`), so the model isn't imitating malformed data — it's reproducing a **learned length distribution it can't sustain in-budget**.

The lever (Alex-aligned) is **better SFT data, not GRPO-on-top**: GRPO gets ~0 reward on the collapsing slice, so there's no signal to fix it. We rewrite the correct traces into a **concise, structured, terminating** form a 1.7B can actually reproduce.

Concretely, the verbose `thinking` field is the problem. Sample (problem id 1133, "average speed"):
> "Hmm, average speed... I remember that... Right? Let me make sure... Got that part... Yep, all steps check out. I don't see any mistakes in my reasoning. Let me just recap: …"

4449 chars of `thinking` wrapped around ~5 lines of actual arithmetic. The `solution` field for the same problem is already a clean 982 chars. The target sits between them.

## The framework

A **fixed skeleton, same headers every time** — structure is easier for a small model to imitate than free-form, and a fixed shape gives it a predictable place to *exit*. Every choice biases toward *commit and finish*, because non-termination is the disease.

| Move | Content | Why it's here |
|---|---|---|
| **0. Target** | 1 line: what's asked + the answer *type* (integer / count / expression / probability…). | Anchors termination — the model knows what the final box must contain. |
| **1. Classify** | 1–2 lines: name the problem type / method ("This is a [stars-and-bars / continuity / …] problem because…"). Reject **at most one** tempting alternative in a single clause, then commit. | Targets *method selection* — the small model's real weakness — while forbidding the visible branching that makes it wander. |
| **2. Setup** | Restate the problem in the method's canonical form (define variables, write the governing equation). | The "restate into the pattern template" step. |
| **3. Solve** | Step by step. **Each step is a concrete computation or deduction.** No restating, no "wait, let me reconsider", no second-guessing. | Where verbose traces hemorrhage tokens and fail to terminate. |
| **4. Verify & box** | 1 line: one quick check (plug back / sanity magnitude / units), then `\boxed{}`. | The explicit **terminal move** — teaches the model to stop. |

### The scalpel (the single most important rule)

> **Cut backtracking, false starts, restatements, recaps, and self-doubt. KEEP every arithmetic and algebraic step explicit.**

A 1.7B cannot do large mental math — if we compress away the *computation*, we break it. "Concise" means *no meandering*, never *skip the arithmetic*. We are removing the `Hmm / let me make sure / yep that checks out / let me recap` connective tissue, not the numbers.

## Rewriter prompt (draft)

Input per trace: `problem`, the correct verbose `full_response` (or `thinking`+`solution`), and the known `expected` answer. We rewrite, then **gate every output through math-verify** — keep only rewrites whose `\boxed{}` still equals `expected`. Reject (or re-roll) anything over a hard token budget.

**System prompt:**
```
You rewrite a verbose but correct math solution into a CONCISE, STRUCTURED reasoning
trace for training a small (1.7B) model. The rewrite must reach the SAME final answer.

Follow this exact skeleton inside <think>...</think>, then give a clean solution:
  Target:   one line — what is asked and the answer type.
  Classify: 1–2 lines — name the method/problem type; reject at most one alternative; commit.
  Setup:    restate in the method's canonical form (define variables / write the equation).
  Solve:    step by step; each step is a concrete computation or deduction.
  Verify:   one line — a quick check — then state the boxed answer.

RULES:
- Remove ALL backtracking, false starts, restatements, recaps, and self-doubt
  ("hmm", "let me make sure", "wait", "yep that checks out", "let me recap").
- KEEP every arithmetic and algebraic step explicit — never skip a computation.
  The reader is a SMALL model that CANNOT do multi-digit arithmetic in its head:
  show every step it could not reconstruct itself. Do not rely on your own capacity.
- Be decisive: pick one method and execute it. Do not explore alternatives.
- Stay under {MAX_TOKENS} tokens. End with \boxed{ANSWER}.
- Preserve the correct final answer exactly.
```

**Per-trace user message:**
```
Problem:
{problem}

A correct (but verbose) solution:
{full_response}

Known correct answer: {expected}

Rewrite it following the skeleton and rules. Output the rewritten trace only.
```

## Pipeline

1. Take the **7154 correct-only** traces (`correct_mathverify == True`).
2. Rewrite each with a strong rewriter (Qwen3-32B, or a frontier model) at a hard budget (start **≤2000 tok**; sweep if needed).
3. **Quality gate:** math-verify the rewrite's `\boxed{}` against `expected`; drop failures (or re-roll once).
4. **Length gate:** drop/re-roll rewrites over budget.
5. SFT v2 on the surviving concise traces — same base (Qwen3-1.7B-Base), same recipe, **equal training budget** as the verbose run.

## Experiment

Clean A/B: **same problems, same answers, concise vs. verbose targets, equal SFT budget.** Headline metrics, all on the *blessed* eval stack (HF backend, tf 5.x, gated on the 572 canary):

- In-budget greedy accuracy at `max_new_tokens=8192` (the disease metric).
- **Termination rate** = fraction that close `</think>` and box within budget (the mechanism).
- Token efficiency (median / p90 generated tokens).

Reference points: GRPO-from-base **43.8%** greedy; SFT-from-verbose-distillation **40.2%** greedy (measured 2026-06-08, see below); R1-Distill-Qwen-1.5B **83.9%** (far ceiling, 800k traces); Qwen3-1.7B thinking **93.4%** at a 32768 budget (capacity ceiling). If concision works, it should raise *termination rate* sharply and lift in-budget accuracy even if raw capability is unchanged.

If this fixes the SFT, it unlocks the thing never actually run here: the real **R1 recipe — concise-SFT cold-start *then* GRPO on top** (GRPO now has a non-collapsing base to get reward signal from).

## Measured baseline (2026-06-08) — and the confound that gates this plan

Trustworthy full-500 SFT greedy (blessed HF + tf5.5.3 stack, authoritative rescore):
**40.2% (201/500)**. By level: L1 69.8 / L2 64.4 / L3 45.7 / L4 34.4 / **L5 15.7**.

vs **GRPO-from-base 43.8% (219/500)**. At equal (≥8192) budget **GRPO ≥ SFT**: the SFT had the
*more generous* budget and still lost, and GRPO terminates short (median response ~286 chars) so a
smaller budget wouldn't hurt it. So RL-from-base — never having seen a teacher trace — beats
verbose-distillation-SFT. (Both greedy / HF / math-verify. GRPO is a non-thinking concise regime, SFT
a thinking regime; different output styles from the same base. We did *not* measure GRPO's peg rate —
its samples lack token counts — so claim only that GRPO outputs are much shorter, not "never pegs.")

**Mechanism — and its load-bearing caveat.** 53.2% of SFT greedy generations peg 8192 (the median
generation *is* a peg). Split by termination: **pegged → 0.8% correct (2/266); terminated → 85.0%
(199/234).** This *looks* like "termination is the whole lever" — BUT pegging is **confounded with
difficulty** (peg-rate L1 28% → L5 74%), so the 85% is computed on a difficulty-selected *easier*
subset. It is **NOT** evidence that the pegged problems would become correct if the model merely
terminated. **This correlation is exactly the premise of concise distillation** ("shorten the trace →
recover the pegged problem"), and it cannot, by itself, distinguish *length-bound* from
*capability-bound* failure.

**The clean discriminator is the budget sweep** (the 266 pegged problems re-run at 16k/32k): high
recovery → length-bound → this plan is justified; low recovery → capability wall → compression alone
won't deliver and we recalibrate expectations. **NOT YET RUN** (deprioritized 2026-06-08; pod torn
down). Recommend running it (≥60 pegged problems @ 32k) as the go/no-go *before* building the rewrite
pipeline — it is the cheapest experiment that validates the whole approach.

## Settled decisions (Alex, 2026-06-08)

1. **`<think>` split — KEEP.** It's a thinking model; preserve the format contract. Moves 0–3
   ("tight" = the concise versions of Target/Classify/Setup/Solve) go *inside* `<think>…</think>`;
   the clean final solution + `\boxed{}` go *after*. ("Tight moves 0–3" just meant the compressed
   reasoning lives in the think block.)
2. **Token budget — methodology-driven, not a hard truncation cap.** Alex's principle: *the problem-
   solving method should determine length; we should seldom hit a limit.* So we do NOT impose an
   aggressive ≤2000 truncation that drops hard problems. Instead: rewrite without forcing a cut,
   measure the resulting length distribution, aim for the bulk to land **well under ~2000–2500 tok**
   (so at an 8192 eval budget the model has large headroom and rarely pegs). A generous re-roll
   ceiling (~4000) only to catch a runaway rewrite, not to shape normal traces.
3. **Rewriter — Qwen3-32B, but PILOT FIRST.** Run ~10–20 traces through 32B, eyeball quality
   (adherence to skeleton, no skipped steps), THEN commit. Frontier-model only if 32B adherence is poor.
4. **Classify step — KEEP, and A/B it.** It targets method-selection (the real weakness). Pilot
   with-vs-without on a handful and compare.
5. **32B traces first** (not capacity-matched Qwen3-1.7B teacher). Risk Alex flagged & I agree:
   a 32B may *skip steps* it can do mentally that a 1.7B can't follow. **Mitigation is load-bearing
   in the rewriter prompt** — instruct the rewriter to assume the reader cannot do multi-digit
   arithmetic in its head and must see every computation. This is the scalpel's "KEEP every step"
   rule, made explicit for the capacity gap. (Capacity-matched-teacher stays as a possible later control.)

## Pilot results (2026-06-08) — the rewriter works

`scripts/rewrite_pilot.py`, 17 correct traces stratified across L1–L5, rewriter **Qwen3-32B via
OpenRouter**, soft target 2000 tok. **17/17 verify-pass, 17/17 well-formed, 17/17 think-tagged,
median 2341→172 est. tok (≈13× compression), 0 over budget.** Steps preserved (completing-the-square,
binomial sums shown explicitly) — the scalpel cut meandering, not arithmetic. This validates the
**data-side precondition** for the whole direction (concise + correct + step-complete targets exist),
which is what the budget sweep was meant to de-risk — done here for cents. **Decision #3 resolves to
32B** (adherence is good; no frontier rewriter needed).

**Pitfall logged (don't repeat):** asking a *thinking* rewriter to emit literal `<think>` tags is
fragile — Qwen3-32B either ignored them (bare skeleton) or, on hard problems, entered its own reasoning
mode so OpenRouter returned the work in a separate `reasoning` field and `content` was just `\boxed{}`.
**Fix that works:** have the rewriter emit neutral `===REASONING===` / `===SOLUTION===` markers (with
`/no_think`), then *we* wrap the reasoning half in real `<think>…</think>` deterministically in
post-processing, with a `reasoning`-field fallback. Artifacts: `eval_results/rewrite_pilot_v3_2026-06-08.json`.

**v4 — Alex's two-layer design + tuned params (target~300, temp 0.2), 2026-06-08.** The 32B TEACHER
now thinks *freely in its own native channel* (median 350 tok scratchpad, captured but NOT trained)
and emits the STUDENT demonstration between `===REASONING===`/`===SOLUTION===` markers; we wrap the
student's reasoning in real `<think>` tags. Inputs per trace = problem + full_response (full 32B trace)
+ solution, filtered `correct_mathverify`. Result: **17/17 verify, 17/17 well-formed+think-tagged,
student median 156 tok (tight band 109–222 across all levels)**. Params validated → ready to scale.
Artifacts `eval_results/rewrite_pilot_v4_2026-06-08.json`.

**Before the full 7154 run:** add concurrency (pilot is synchronous; 7154 sequential = hours).

## Reframe (Alex, 2026-06-08): conciseness is the win condition, not eventual-correctness

Alex: *"I don't care much whether the SFT model can get it right if it needs 32k tokens — that's
insane. I care whether we can get 32k down much lower."*

This **downgrades the budget sweep** from a go/no-go to **optional capability-ceiling calibration.**
Reasoning: the win condition is now *terminate-and-solve concisely at a small budget*, which we
measure **directly** by training concise SFT v2 and reading off (a) termination rate, (b) accuracy,
(c) length distribution at the 8192 budget. The budget sweep's residual value is only to bound the
ceiling (what fraction of pegged problems are solvable *at any* budget — i.e. the capability wall),
which calibrates how disappointed to be if v2 doesn't fix everything. Worth ~$1–4, but no longer a
prerequisite. Default: **skip it; build concise SFT v2 and measure conciseness directly.**
