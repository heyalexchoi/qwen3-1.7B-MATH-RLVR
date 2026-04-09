#!/usr/bin/env python3
"""
08_rescore_traces.py — Re-score existing traces with the fixed normalizer.

Reads qwen32b_math_traces.jsonl, re-evaluates each record's predicted vs expected
using the updated answers_match() from 07_qwen32b_traces.py, and writes a new
qwen32b_math_traces_rescored.jsonl with corrected `correct` flags.

Also prints a full accuracy breakdown and saves an updated summary JSON.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Copy of the fixed normalizer (keep in sync with 07_qwen32b_traces.py)
# ---------------------------------------------------------------------------

def normalize_answer(ans: str | None, strip_text_content: bool = True) -> str:
    if ans is None:
        return ""
    s = ans.strip()
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = re.sub(r'\\math(?:bf|it|rm|bb|sf|tt|cal|scr)\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\(?:boldsymbol|bm)\{([^}]*)\}', r'\1', s)
    if strip_text_content:
        s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
        s = re.sub(r'\\mbox\s*\{[^}]*\}', '', s)
    else:
        s = re.sub(r'\\text\s*\{([^}]*)\}', lambda m: m.group(1), s)
        s = re.sub(r'\\mbox\s*\{([^}]*)\}', lambda m: m.group(1), s)
    s = re.sub(r'\{,\}', '', s)
    s = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '', s)
    s = s.replace(r'\circ', '')
    s = s.replace(r'\%', '').replace('%', '')
    s = s.replace(r'\$', '').replace('$', '')
    for sp in [r'\!', r'\,', r'\;', r'\:', r'\quad', r'\qquad', r'\ ']:
        s = s.replace(sp, '')
    s = re.sub(r'\\phantom\{[^}]*\}', '', s)
    s = re.sub(r'\\phantom', '', s)
    s = re.sub(r'\\(?:left|right)\s*', '', s)
    s = re.sub(r'\s+', '', s)
    for _ in range(3):
        s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac\{([^}]+)\}(\d)', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac(\d)\{([^}]+)\}', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac\{([^}]+)\}([a-zA-Z])', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac([a-zA-Z0-9])\{([^}]+)\}', r'\\frac{\1}{\2}', s)
        s = re.sub(r'\\frac([a-zA-Z0-9])([a-zA-Z0-9])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\sqrt([a-zA-Z0-9])', r'\\sqrt{\1}', s)
    s = re.sub(r'(\d+\.\d*[1-9])0+\b', r'\1', s)
    s = re.sub(r'(\d+)\.0+\b', r'\1', s)
    s = re.sub(r'\.$', '', s)
    s = re.sub(r'(\d),(\d{3})(?!\d)', lambda m: m.group(1) + m.group(2), s)
    s = re.sub(r'_(\d)$', '', s)
    s = re.sub(r'(-?\d+)/(-?\d+)', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\frac\{-([^}]+)\}', r'-\\frac{\1}', s)

    def pull_pi_out(t: str) -> str:
        def repl(m: re.Match) -> str:
            num, den = m.group(1), m.group(2)
            if r'\pi' in num and r'\pi' not in den:
                num_clean = num.replace(r'\pi', '') or '1'
                return r'\frac{' + num_clean + '}{' + den + r'}\pi'
            return m.group(0)
        return re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', repl, t)
    s = pull_pi_out(s)
    s = re.sub(r'^\.(\d)', r'0.\1', s)
    return s.strip()


def answers_match(predicted: str | None, expected: str | None) -> bool:
    if not predicted or not expected:
        return False

    def strip_var_assign(s: str) -> str:
        m = re.match(r'^[a-zA-Z]\s*=\s*(.+)$', s)
        return m.group(1) if m else s

    for strip_text in (True, False):
        ne = normalize_answer(expected, strip_text)
        np_ = normalize_answer(predicted, strip_text)
        if ne == np_ and ne != "":
            return True
        ne2 = strip_var_assign(ne)
        np2 = strip_var_assign(np_)
        if ne2 == np2 and ne2 != "":
            return True
        if ne2 == np_ and ne2 != "":
            return True
        if ne == np2 and ne != "":
            return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).parent.parent
    input_path = root / "data" / "traces" / "qwen32b_math_traces.jsonl"
    output_path = root / "data" / "traces" / "qwen32b_math_traces_rescored.jsonl"
    summary_path = root / "outputs" / "qwen32b_traces_summary_rescored.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    total = 0
    correct_old = 0
    correct_new = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0

    by_level: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_subject: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            r = json.loads(line)
            total += 1

            pred = r.get("predicted") or ""
            exp = r.get("expected") or ""
            old_correct = bool(r.get("correct"))
            new_correct = answers_match(pred, exp) if (pred and exp) else old_correct

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
            fout.write(json.dumps(r) + "\n")

    # Print report
    print(f"=== Rescore Results ===")
    print(f"Total records:        {total}")
    print(f"Correct (old):        {correct_old} ({correct_old/total:.2%})")
    print(f"Correct (rescored):   {correct_new} ({correct_new/total:.2%})")
    print(f"Flipped → correct:    {flipped_to_correct}")
    print(f"Flipped → wrong:      {flipped_to_wrong}")
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
        "total": total,
        "correct_original": correct_old,
        "accuracy_original": round(correct_old / total, 4),
        "correct_rescored": correct_new,
        "accuracy_rescored": round(correct_new / total, 4),
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_wrong": flipped_to_wrong,
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
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRescored traces → {output_path}")
    print(f"Summary → {summary_path}")
    print(f"\nFor SFT: use {correct_new} correct traces (filter correct==true)")


if __name__ == "__main__":
    main()
