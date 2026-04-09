#!/usr/bin/env python3
"""Comparison aggregator: base → SFT → GRPO on MATH-500.

Does NOT re-run inference. Reads already-computed summary JSON files
produced by sft_eval.py and prints a side-by-side comparison table.

Usage:
    python scripts/final_eval.py \
        --base     outputs/base_eval_summary.json \
        --sft      outputs/sft_eval_results_summary.json \
        --grpo     outputs/grpo_eval_results_summary.json \
        --output   outputs/final_comparison.json

All arguments are optional — omit any checkpoint you haven't run yet.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_summary(path: str | None, label: str) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"  [{label}] summary not found: {path}")
        return None
    with open(p) as f:
        return json.load(f)


def get_pass1(summary: dict) -> float | None:
    """Extract pass@1 overall from sft_eval.py summary format."""
    try:
        return summary["pass1"]["overall"]
    except (KeyError, TypeError):
        pass
    # Fallback: flat key
    return summary.get("pass1_overall") or summary.get("pass1")


def get_pass8(summary: dict) -> float | None:
    try:
        return summary["pass8"]["overall"]
    except (KeyError, TypeError):
        pass
    return summary.get("pass8_overall") or summary.get("pass8")


def fmt(val: float | None) -> str:
    return f"{val:.2%}" if val is not None else "—"


def delta_str(val: float | None, ref: float | None) -> str:
    if val is None or ref is None:
        return ""
    d = val - ref
    arrow = "↑" if d >= 0 else "↓"
    return f"  ({arrow}{abs(d):.2%})"


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval summaries into a comparison table")
    parser.add_argument("--base",  type=str, default=None, help="Base model summary JSON")
    parser.add_argument("--sft",   type=str, default=None, help="SFT checkpoint summary JSON")
    parser.add_argument("--grpo",  type=str, default=None, help="GRPO checkpoint summary JSON")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save combined JSON")
    args = parser.parse_args()

    base_s  = load_summary(args.base,  "base")
    sft_s   = load_summary(args.sft,   "sft")
    grpo_s  = load_summary(args.grpo,  "grpo")

    if not any([base_s, sft_s, grpo_s]):
        logger.error("No summary files found. Pass at least one of --base / --sft / --grpo.")
        sys.exit(1)

    base_p1  = get_pass1(base_s)
    sft_p1   = get_pass1(sft_s)
    grpo_p1  = get_pass1(grpo_s)

    base_p8  = get_pass8(base_s)
    sft_p8   = get_pass8(sft_s)
    grpo_p8  = get_pass8(grpo_s)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  MATH-500 Results — Qwen3-1.7B")
    logger.info("=" * 60)
    logger.info(f"  {'Checkpoint':<12}  {'pass@1':>8}  {'pass@8':>8}")
    logger.info(f"  {'-'*12}  {'-'*8}  {'-'*8}")

    if base_s is not None:
        logger.info(f"  {'Base':<12}  {fmt(base_p1):>8}  {fmt(base_p8):>8}")
    if sft_s is not None:
        logger.info(f"  {'SFT':<12}  {fmt(sft_p1):>8}{delta_str(sft_p1, base_p1)}  {fmt(sft_p8):>8}{delta_str(sft_p8, base_p8)}")
    if grpo_s is not None:
        logger.info(f"  {'GRPO':<12}  {fmt(grpo_p1):>8}{delta_str(grpo_p1, sft_p1)}  {fmt(grpo_p8):>8}{delta_str(grpo_p8, sft_p8)}")

    logger.info("=" * 60)
    logger.info("")

    if args.output:
        combined = {
            "base":  {"pass1": base_p1,  "pass8": base_p8},
            "sft":   {"pass1": sft_p1,   "pass8": sft_p8},
            "grpo":  {"pass1": grpo_p1,  "pass8": grpo_p8},
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Combined summary saved to {args.output}")


if __name__ == "__main__":
    main()
