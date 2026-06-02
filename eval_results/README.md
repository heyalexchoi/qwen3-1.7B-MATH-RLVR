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

## Conventions
- `math500_eval.py` (unified entrypoint; `--format completion|chat`) →
  - `eval_results/<tag>_<step|local>_<format>_<UTC>_math500_results.json` — combined,
    `pass1`/`pass8` schema (consumed by `rescore_math500.py`).
  - `eval_results/<tag>_<step|local>_<format>_<UTC>_math500_samples.jsonl` — per-sample
    generations (the durable artifact). `<UTC>` = run timestamp, override with `--run_id`.
- Per-sample jsonl rows include: `unique_id, pass_type (greedy|sample), sample_idx, problem,
  expected, level, subject, response (FULL text), predicted, n_tokens, max_new_tokens, format,
  model` (+ `correct` when math-verify is installed; authoritative scoring is `rescore_math500.py`).
- `sft_eval.py` is DEPRECATED (kept until the unified chat path passes a GPU canary). Its old
  outputs were `sft_<modeltag>_<mode>_max<N>_<backend>_<UTC>.{samples.jsonl,summary.json,combined.json}`.

## Hard rule (pre-teardown)
**Before tearing down any pod, `rsync` `eval_results/` back and verify file counts/sizes.** Never
delegate teardown. Generations large? gzip in place (`gzip *.jsonl`) — but never leave them only on
ephemeral storage. Optionally `--upload` (math500_eval.py) to the HF dataset repo as a second copy.
