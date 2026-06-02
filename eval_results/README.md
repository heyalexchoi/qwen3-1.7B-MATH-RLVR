# eval_results/ — durable eval generations (NOT gitignored)

**Written by:** Claude Opus 4.8 · **Date:** 2026-06-01

Every eval run writes its **full per-sample generations** here, with a self-describing,
timestamped filename. This directory is deliberately **not** in `.gitignore` (unlike `outputs/`).

## Why this exists
The 2026-04 SFT eval wrote its generations to `outputs/sft_eval_results.jsonl` — which is
gitignored — on a RunPod pod that was later torn down **without pulling the file back**. The
generations were lost; only the summary `.log` (token counts + correct/wrong flags) survived. That
made the SFT score unre-scorable and unverifiable. Generations are the primary artifact: with them
you can re-score under any metric, inspect failure modes, and audit. Without them you have a number
you cannot defend.

## Storage model (the single failproof path)
The **HF dataset repo `heyalexchoi/qwen3-math-rlvr-results` is the durable record** for full
generations — `math500_eval.py` uploads them automatically (periodically during the run via
`--upload_every`, and at the end). Git tracks **only the small metrics-only `*_summary.json`**
plus this dir's docs; the large `*_results.json` / `*_samples.jsonl` are **gitignored** (a chat
run is ~27MB — git history would bloat permanently). Three independent durability layers, no flag
to remember: (1) local JSONL written incrementally, (2) mandatory HF upload, (3) the pre-teardown
rsync rule below. The 2026-04 loss happened only because all three were absent at once.

## Conventions
- Filenames are **deterministic** (no timestamp) so re-running the same command **auto-resumes**
  (skips finished samples; on a fresh pod it pulls the prior partial JSONL from HF). Base name:
  `<tag>_<step|local>_<format>_max<N>_math500`, plus `_subset` (when `--unique_id`) / `_nostopids`
  (when `--no_chat_stop_ids`). Override with `--output_name`; force a clean run with `--fresh`.
  Three files per run: `<base>_results.json` (full → HF), `<base>_samples.jsonl` (full → HF),
  `<base>_summary.json` (metrics only → git).
- Per-sample jsonl rows: `unique_id, pass_type (greedy|sample), sample_idx, problem, expected,
  level, subject, response (FULL text), predicted, n_tokens, max_new_tokens, format, model`
  (+ `correct` when math-verify is installed; `rescore_math500.py` is authoritative).
- Diagnostic runs (`--unique_id` canary / `--no_chat_stop_ids` A-B) are **local-only** (not uploaded).
- `sft_eval.py` is DEPRECATED (kept until the unified chat path passes a GPU canary, then deleted).

## Hard rule (pre-teardown)
**Before tearing down any pod, `rsync` `eval_results/` back and verify file counts/sizes.** Never
delegate teardown. Generations large? gzip in place (`gzip *.jsonl`) — but never leave them only on
ephemeral storage. Optionally `--upload` (math500_eval.py) to the HF dataset repo as a second copy.
