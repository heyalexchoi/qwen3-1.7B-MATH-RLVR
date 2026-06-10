#!/usr/bin/env python3
"""
rewrite_pilot.py — Concise-trace distillation PILOT (Written by Claude Opus 4.8, 2026-06-08).

Distills the 7154 correct Qwen3-32B traces into concise, followable STUDENT demonstrations
(docs/distill-trace-framework.md) for SFT v2 of a 1.7B model.

Two-layer design (Alex, 2026-06-08):
  - The 32B TEACHER thinks freely in its OWN native reasoning to figure out the best
    compression. That teacher scratchpad is KEPT (saved for inspection) but NEVER trained on.
  - The teacher's DELIVERABLE is the STUDENT demonstration, emitted between neutral
    ===REASONING=== / ===SOLUTION=== markers. We wrap the student's reasoning in real
    <think>...</think> tags ourselves in post-processing → the SFT v2 training target.

Inputs per trace (filtered correct_mathverify=True): problem + full_response (the full 32B
trace) + solution (clean worked solution). `expected` is used only for the math-verify gate.

Usage:
  python scripts/rewrite_pilot.py --per-level 3 --target-tokens 300 --temp 0.2
"""
import argparse, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict

import requests

TRACES = "data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl"
MODEL = "qwen/qwen3-32b"
BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM = """You are an expert teacher creating a CONCISE, EASY-TO-FOLLOW demonstration solution that a small (1.7B) student model will learn to imitate. You are given a verbose but correct solution; distill it into the clearest possible compact demonstration reaching the SAME final answer.

Think freely first (in your own reasoning) about how best to structure and compress the demonstration. Then output ONLY the student demonstration, as exactly two sections separated by the literal marker lines below:

===REASONING===
Target: one line — what is asked and the answer type.
Classify: 1-2 lines — name the method/problem type; reject at most one alternative; commit.
Setup: restate in the method's canonical form (define variables / write the equation).
Solve:
1. step (a concrete computation or deduction)
2. step
...
Verify: one line — a quick check.
===SOLUTION===
short clean final solution, ending in \\boxed{ANSWER}

RULES for the demonstration:
- Remove ALL backtracking, false starts, restatements, recaps, and self-doubt.
- KEEP every arithmetic and algebraic step explicit — never skip a computation. The student CANNOT do multi-digit arithmetic in its head: show every step it could not reconstruct itself. Do not rely on your own capacity.
- Be decisive: one method, executed. Aim for roughly MAX_TOKENS tokens — clarity over brevity; never sacrifice a needed step to hit a length.
- Preserve the correct final answer exactly."""

USER = """Problem:
{problem}

The full (verbose but correct) solution the teacher produced:
{full_response}

A clean version of that solution:
{solution}

Known correct answer: {expected}

Produce the student demonstration following the format and rules."""


def get_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    raw = (Path.home() / ".openclaw" / "openclaw.json").read_text()
    m = re.search(r'"openrouter"[\s\S]{0,300}?"apiKey":\s*"(sk-or-[^"]+)"', raw)
    if not m:
        sys.exit("No OpenRouter key")
    return m.group(1)


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth, out = 1, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1; out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out).strip()


try:
    from math_verify import parse as mvp, verify as mvv
    def mv_ok(pred, exp):
        if not pred or not exp:
            return False
        try:
            g, a = mvp(f"${exp}$"), mvp(f"${pred}$")
            return bool(g and a and mvv(g, a))
        except Exception:
            return False
except ImportError:
    def mv_ok(pred, exp):
        return pred.strip() == exp.strip()  # fallback only


def call_teacher(key, problem, full_response, solution, expected, target_tokens, temp):
    """Returns (content, teacher_reasoning). Teacher thinks natively; we collect both."""
    sys_msg = SYSTEM.replace("MAX_TOKENS", str(target_tokens))
    usr = USER.format(problem=problem, full_response=full_response,
                      solution=solution, expected=expected)
    for attempt in range(4):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": sys_msg},
                                   {"role": "user", "content": usr}],
                      "temperature": temp, "top_p": 0.9,
                      "max_tokens": 8000},  # generous: native thinking + student trace
                timeout=240)
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            # Some providers inline the teacher's CoT as a leading <think>...</think> in content.
            mt = re.match(r"\s*<think>(.*?)</think>\s*(.*)", content, re.DOTALL)
            if mt:
                reasoning = (reasoning + "\n" + mt.group(1)).strip()
                content = mt.group(2)
            return content, reasoning
        except Exception as e:
            if attempt == 3:
                return f"__ERROR__ {e}", ""
            time.sleep(2 ** attempt)
    return "__ERROR__ retries", ""


def assemble_student(content):
    """Parse ===REASONING===/===SOLUTION=== from teacher output → (student_trace, well_formed)."""
    if "===REASONING===" in content and "===SOLUTION===" in content:
        think = content.split("===REASONING===", 1)[1].split("===SOLUTION===", 1)[0].strip()
        soln = content.split("===SOLUTION===", 1)[1].strip()
        return f"<think>\n{think}\n</think>\n\n{soln}", True
    return content, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=3)
    ap.add_argument("--target-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval_results/rewrite_pilot_2026-06-08.json")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    by_level = defaultdict(list)
    for line in open(TRACES):
        d = json.loads(line)
        if d.get("correct_mathverify"):
            by_level[str(d.get("level"))].append(d)
    sample = []
    for lvl in sorted(by_level):
        random.shuffle(by_level[lvl])
        sample += by_level[lvl][: args.per_level]
    print(f"Sampled {len(sample)} traces  target~{args.target_tokens}tok temp={args.temp}")

    key = get_key()
    results = []
    for i, d in enumerate(sample):
        content, teacher_reasoning = call_teacher(
            key, d["problem"], d["full_response"], d.get("solution", ""),
            str(d["expected"]), args.target_tokens, args.temp)
        student, wf = assemble_student(content)
        boxed = extract_boxed(student)
        ok = mv_ok(boxed, str(d["expected"]))
        rec = {"id": d["id"], "level": d["level"], "expected": str(d["expected"]),
               "orig_chars": len(d["full_response"]), "student_chars": len(student),
               "orig_est_tok": len(d["full_response"]) // 4, "student_est_tok": len(student) // 4,
               "teacher_reasoning_est_tok": len(teacher_reasoning) // 4,
               "boxed": boxed, "verify_ok": ok, "well_formed": wf,
               "has_think": "<think>" in student and "</think>" in student,
               "student_trace": student, "teacher_reasoning": teacher_reasoning}
        results.append(rec)
        print(f"[{i+1}/{len(sample)}] L{str(d['level']).replace('Level ','')} {d['id']}: "
              f"orig {rec['orig_est_tok']}→student {rec['student_est_tok']}tok "
              f"(teacher think {rec['teacher_reasoning_est_tok']}tok)  "
              f"verify={'OK' if ok else 'FAIL'} wf={wf} think={rec['has_think']} boxed={boxed!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": MODEL, "target_tokens": args.target_tokens, "temp": args.temp,
               "n": len(results), "results": results}, open(args.out, "w"), indent=2)

    import statistics as st
    okc = sum(r["verify_ok"] for r in results)
    wfc = sum(r["well_formed"] for r in results)
    thinkc = sum(r["has_think"] for r in results)
    print(f"\n=== PILOT SUMMARY (target~{args.target_tokens}tok, temp={args.temp}) ===")
    print(f"verify pass: {okc}/{len(results)}   well-formed: {wfc}/{len(results)}   think-tags: {thinkc}/{len(results)}")
    print(f"orig est tok median: {int(st.median([r['orig_est_tok'] for r in results]))}")
    print(f"student est tok median: {int(st.median([r['student_est_tok'] for r in results]))}  "
          f"(min {min(r['student_est_tok'] for r in results)}, max {max(r['student_est_tok'] for r in results)})")
    print(f"teacher think est tok median: {int(st.median([r['teacher_reasoning_est_tok'] for r in results]))}")
    print(f"\nFull traces saved → {args.out}")


if __name__ == "__main__":
    main()
