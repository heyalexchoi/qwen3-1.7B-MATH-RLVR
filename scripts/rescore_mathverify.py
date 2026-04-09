#!/usr/bin/env python3
"""
09_rescore_mathverify.py — Re-score traces using HuggingFace math-verify.

Replaces the hand-rolled regex normalizer with math-verify's ANTLR4/SymPy
pipeline, which is the gold-standard open-source math evaluator.
Outperforms the Qwen evaluator and EleutherAI harness on the MATH dataset.

Install: pip install "math-verify[antlr4_13_2]"

Usage:
    python scripts/09_rescore_mathverify.py
    python scripts/09_rescore_mathverify.py --input data/traces/qwen32b_math_traces.jsonl
    python scripts/09_rescore_mathverify.py --input data/traces/qwen32b_math_traces_rerun.jsonl

Output:
    data/traces/<stem>_mv_rescored.jsonl   — traces with updated `correct` flags
    outputs/<stem>_mv_summary.json         — accuracy breakdown
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("ERROR: math-verify not installed.")
    print("Run: pip install 'math-verify[antlr4_13_2]'")
    sys.exit(1)


def mv_verify(predicted: str | None, expected: str | None) -> bool:
    """
    Check answer equivalence using math-verify.

    math-verify pipeline:
      1. Extract answer from text (LaTeX env, expr, string — in priority order)
      2. Parse to SymPy via ANTLR4 grammar
      3. Compare symbolically or numerically

    Handles all cases our regex normalizer covered, plus:
      - Symbolic algebraic equivalence (12(√3+1) vs 12+12√3)
      - Interval/set notation
      - Matrix equivalence
      - Relation flipping (a<b vs b>a)
      - Unicode symbol substitution (β → beta)
    """
    if not predicted or not expected:
        return False
    # Wrap in $...$ so LatexExtractionConfig can parse bare LaTeX expressions.
    # Without wrapping, bare \dfrac{3}{2}, t^7, \text{odd}, etc. all fail to parse.
    try:
        gold = parse(f"${expected}$")
        ans = parse(f"${predicted}$")
        if gold and ans:
            return bool(verify(gold, ans))
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL (default: data/traces/qwen32b_math_traces_rescored.jsonl)"
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = root / "data" / "traces" / "qwen32b_math_traces_rescored.jsonl"

    if not input_path.exists():
        # Fall back to non-rescored
        fallback = root / "data" / "traces" / "qwen32b_math_traces.jsonl"
        if fallback.exists():
            print(f"Note: rescored file not found, falling back to {fallback}")
            input_path = fallback
        else:
            print(f"ERROR: {input_path} not found", file=sys.stderr)
            sys.exit(1)

    stem = input_path.stem
    output_path = root / "data" / "traces" / f"{stem}_mv_rescored.jsonl"
    summary_path = root / "outputs" / f"{stem}_mv_summary.json"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print()

    total = 0
    correct_old = 0
    correct_new = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0
    mv_errors = 0

    by_level: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_subject: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            r = json.loads(line)
            total += 1

            pred = r.get("predicted") or ""
            exp = r.get("expected") or ""
            old_correct = bool(r.get("correct"))

            try:
                new_correct = mv_verify(pred, exp)
            except Exception as e:
                # math-verify failure: keep old result, log it
                new_correct = old_correct
                mv_errors += 1
                if mv_errors <= 10:
                    print(f"  mv_verify error #{mv_errors} (id={r.get('id')}): {e}", file=sys.stderr)

            if old_correct:
                correct_old += 1
            if new_correct:
                correct_new += 1

            if not old_correct and new_correct:
                flipped_to_correct += 1
            elif old_correct and not new_correct:
                flipped_to_wrong += 1

            lv = r.get("level", "Unknown")
            subj = r.get("subject", "Unknown")
            by_level[lv]["total"] += 1
            by_subject[subj]["total"] += 1
            if new_correct:
                by_level[lv]["correct"] += 1
                by_subject[subj]["correct"] += 1

            r["correct"] = new_correct
            r["correct_mathverify"] = new_correct  # explicit field for traceability
            fout.write(json.dumps(r) + "\n")

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{total if total > 0 else '?'}] running acc: {correct_new/(i+1):.1%}")

    # Print report
    print(f"\n=== math-verify Rescore Results ===")
    print(f"Input file:           {input_path.name}")
    print(f"Total records:        {total}")
    print(f"Correct (pre):        {correct_old} ({correct_old/total:.2%})")
    print(f"Correct (math-verify):{correct_new} ({correct_new/total:.2%})")
    print(f"Flipped → correct:    {flipped_to_correct}")
    print(f"Flipped → wrong:      {flipped_to_wrong}")
    print(f"math-verify errors:   {mv_errors} (kept old result on error)")
    print()
    print("By level:")
    for lv in sorted(by_level.keys()):
        d = by_level[lv]
        print(f"  {lv}: {d['correct']}/{d['total']} = {d['correct']/d['total']:.1%}")
    print()
    print("By subject:")
    for subj in sorted(by_subject.keys()):
        d = by_subject[subj]
        print(f"  {subj}: {d['correct']}/{d['total']} = {d['correct']/d['total']:.1%}")

    # Save summary
    summary = {
        "evaluator": "math-verify[antlr4_13_2]",
        "input_file": str(input_path.name),
        "total": total,
        "correct_pre": correct_old,
        "accuracy_pre": round(correct_old / total, 4),
        "correct_mathverify": correct_new,
        "accuracy_mathverify": round(correct_new / total, 4),
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_wrong": flipped_to_wrong,
        "mv_errors": mv_errors,
        "by_level": {
            lv: {
                "total": d["total"],
                "correct": d["correct"],
                "accuracy": round(d["correct"] / d["total"], 4),
            }
            for lv, d in sorted(by_level.items())
        },
        "by_subject": {
            subj: {
                "total": d["total"],
                "correct": d["correct"],
                "accuracy": round(d["correct"] / d["total"], 4),
            }
            for subj, d in sorted(by_subject.items())
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRescored traces → {output_path}")
    print(f"Summary          → {summary_path}")
    print(f"\nFor SFT: use {correct_new} correct traces (filter correct==true or correct_mathverify==true)")


if __name__ == "__main__":
    main()
