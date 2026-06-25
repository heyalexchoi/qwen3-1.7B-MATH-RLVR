# v3 teacher selection: qwen3-235b-a22b-thinking-2507 (pinned to WandB)

*Written by Claude Opus 4.8 — 2026-06-25*

## Decision
The v3 distillation teacher is **`qwen/qwen3-235b-a22b-thinking-2507`**, hard-pinned to
OpenRouter provider **`wandb`** (no fallback). Set as the defaults in `scripts/rewrite_full.py`
(`MODEL`, `PROVIDER`). The reframe Verify wording (commit `2cc7019`) is unchanged.

## Why we revisited the teacher
The v3 smoke A/Bs surfaced a Verify-quality tail under the *reframe* wording: a confidently-wrong
arithmetic check (2567) and hollow "independent check confirms" labels. Open question: is this a
**prompt** deficiency or the **teacher model** satisficing? Alex reframed it directly ("or is this
qwen 32B being lazy").

## The controlled 3-way A/B (seed=7, identical 42 stratified problems, reframe wording held fixed)
Files: `data/concise/teacher_ab_{qwen32b,dsv4flash,qwen235bthink}.jsonl`,
notes `data/concise/teacher_ab_NOTES.md`.

| teacher | gate yield | well-formed | student tok (median) | Verify behavior |
|---|---|---|---|---|
| qwen3-32b | high | high | ~235 | **lazy** — re-sums its own coefficients on 317; circular checks |
| deepseek-v4-flash | high | high | ~444 | thorough; verbose SOLUTION prose (~2× length) |
| **qwen3-235b-thinking** | **100%** | **100%** | **259** | thorough, independent; terse |

**Conclusion: it's the model, not the prompt.** With the *same* prompt, 235B and deepseek both
verify thoroughly while 32B satisfices. Wording sets the bar a model satisfices *to*; a stronger
model clears it. So we fix the teacher rather than spiral on more Verify clauses.

### Why 235B over deepseek-v4-flash
Both are reasoning models; their hidden reasoning goes to a separate `teacher_reasoning` channel and
never reaches the student. Deepseek's ~2× longer *student demo* is stylistic verbosity (bold headers,
extra prose in the SOLUTION section), not deeper work — on id540 its only real edge was computing the
actual roots in Verify vs 235B confirming the discriminant. Since the 1.7B student imitates the demo,
235B's terseness preserves the v2 conciseness/termination win. Marginal deepseek upside isn't worth
2× tokens + ~5× cost.

### Followability (the original reason we chose 32B)
Inspected 235B demos across L1/L3/L5 + the longest (id3983, the 1905 recursion). They are atomic
(median 3 numbered steps, max 15), show arithmetic inline rather than asserting, and state
definitions before use. The hardest trace collapses a few reverse steps but its Verify reconstructs
the full forward chain. These read as demonstrations a 1.7B can pattern-match, not lectures beyond it.

## Cost: why the WandB pin is load-bearing
OpenRouter blends provider prices; 235B-thinking routes across WandB **$0.10/M** (in+out, subsidized
— likely data collection, which Alex accepts) vs Alibaba $1.50, DeepInfra/AtlasCloud $2.30,
Novita $3.00. Unpinned routing could cost **15–30×** the pinned run (~$60–130 vs ~$4.4).

The pin is `body["provider"] = {"order": ["wandb"], "allow_fallbacks": false}` — it **errors rather
than silently falling back** to a pricier provider. Verified live: served-provider "WandB",
$0.10/M confirmed; 4-problem end-to-end smoke clean (100% gate). To auto-route instead, pass
`--provider ''`.

## Net
235B-thinking pinned to WandB: fixes 32B's lazy verification, keeps demos tight (~259 median tok),
100% first-pass gate, followable for a 1.7B, ~$4.4 for the full 7,356-trace rewrite. Real validation
remains v3a training + student-rollout Verify inspection — not another smoke.
