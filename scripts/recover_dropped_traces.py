#!/usr/bin/env python3
"""
recover_dropped_traces.py — recover the 336 teacher traces dropped by the
correct-only gate (math-verify scored them wrong).

Spot-checking the drops showed three populations:
  1. Scoring artifacts — teacher was right, math-verify couldn't equate the
     forms (\\mbox{four} vs 4, "0.56 = 56\\%" vs 56, \\dfrac/whitespace in
     matrices). Stage A rescues these locally for free.
  2. Multiple-choice leakage — teacher answered "B" where a value is expected.
  3. Genuine teacher misses. Stages B regenerates 2+3 best-of-4 at temp 0.6
     (same teacher/params as generate_traces_32b.py) with an added
     instruction to box the exact value, never a choice letter.

Scoring runs in the MAIN thread only (math-verify uses signal.alarm — in
worker threads it raises and silently scores everything False).

Usage:
    python3 scripts/recover_dropped_traces.py            # full run
    python3 scripts/recover_dropped_traces.py --stage-a-only
    python3 scripts/recover_dropped_traces.py --limit 10 # smoke test

Output: data/traces/recovered_dropped_traces.jsonl (same schema as the
rescored traces file, + "recovery" field: "rescored" | "best_of_4").
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from math_verify import parse, verify  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
INPUT = PROJECT_ROOT / "data" / "traces" / "qwen32b_math_traces_rerun_mv_rescored.jsonl"
OUTPUT = PROJECT_ROOT / "data" / "traces" / "recovered_dropped_traces.jsonl"
SUMMARY = PROJECT_ROOT / "outputs" / "recover_dropped_summary.json"

MODEL = "qwen/qwen3-32b"
BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 32768
N_SAMPLES = 4
CONCURRENCY = 32
RETRY_LIMIT = 4

PROMPT = (
    "/think\n\nSolve the following math problem. Show your full reasoning, "
    "then give the final answer in \\boxed{}.\n\n"
    "IMPORTANT: the boxed answer must be the exact value or expression the "
    "problem asks for, in the form it asks for. If the problem lists "
    "multiple-choice options, box the VALUE, never the choice letter.\n\n"
    "Problem: {problem}"
)

WORD_NUMS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def get_key() -> str:
    import os
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    raw = (Path.home() / ".openclaw" / "openclaw.json").read_text()
    m = re.search(r'"openrouter"[\s\S]{0,300}?"apiKey":\s*"(sk-or-[^"]+)"', raw)
    if not m:
        print("ERROR: no OpenRouter key (env OPENROUTER_API_KEY or openclaw.json)", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def mv_verify(predicted: str | None, expected: str | None) -> bool:
    """math-verify equivalence (copied from rescore_mathverify.py)."""
    if not predicted or not expected:
        return False
    try:
        gold = parse(f"${expected}$")
        ans = parse(f"${predicted}$")
        if gold and ans:
            return bool(verify(gold, ans))
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Stage A — normalization rescue (teacher was right, scorer couldn't tell)
# ---------------------------------------------------------------------------

def _variants(s: str) -> list[str]:
    """Conservative form-variants of an answer string for equivalence retry."""
    out = [s]
    t = re.sub(r"\\m?box\{([^{}]*)\}", r"\1", s)          # \mbox{four} -> four
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    if t != s:
        out.append(t)
    w = t.strip().lower()
    if w in WORD_NUMS:                                     # four -> 4
        out.append(WORD_NUMS[w])
    if "=" in s:                                           # "0.56 = 56\%" -> both sides
        out.extend(p.strip() for p in s.split("=") if p.strip())
    stripped = re.sub(r"(\^\\?circ|\\%|\\!|\\,|\s+)", " ", s).strip()  # degrees/percent/space noise
    if stripped != s:
        out.append(stripped)
    return list(dict.fromkeys(out))


def stage_a(record: dict) -> bool:
    exp, pred = record.get("expected"), record.get("predicted")
    if not exp or not pred or pred == "None":
        return False
    for e in _variants(str(exp)):
        for p in _variants(str(pred)):
            if mv_verify(p, e):
                return True
    return False


# ---------------------------------------------------------------------------
# Stage B — best-of-4 regeneration
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def split_thinking(full_response: str) -> tuple[str, str]:
    m = re.search(r"<think>([\s\S]*?)</think>", full_response, re.IGNORECASE)
    if m:
        return m.group(1).strip(), full_response[m.end():].strip()
    return "", full_response.strip()


def call_teacher(api_key: str, problem: str, attempt_tag: str) -> dict:
    last_err = None
    for retry in range(RETRY_LIMIT):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": PROMPT.replace("{problem}", problem)}],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=900,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            full = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            return {"tag": attempt_tag, "full_response": full, "error": None}
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(2.0 ** retry)
    return {"tag": attempt_tag, "full_response": None, "error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap Stage-B problems (smoke test)")
    args = ap.parse_args()

    dropped = []
    for line in open(INPUT):
        r = json.loads(line)
        if not r.get("correct_mathverify", r.get("correct")):
            dropped.append(r)
    print(f"dropped traces: {len(dropped)}")

    recovered, need_regen = [], []
    for r in dropped:
        if stage_a(r):
            out = dict(r)
            out["correct_mathverify"] = True
            out["recovery"] = "rescored"
            recovered.append(out)
        else:
            need_regen.append(r)
    print(f"Stage A rescued: {len(recovered)}  |  need regen: {len(need_regen)}")

    summary = {"dropped": len(dropped), "stage_a_rescued": len(recovered)}

    if not args.stage_a_only:
        if args.limit:
            need_regen = need_regen[: args.limit]
        api_key = get_key()
        jobs = [(r, k) for r in need_regen for k in range(N_SAMPLES)]
        print(f"Stage B: {len(jobs)} generations ({len(need_regen)} problems x {N_SAMPLES})")
        results: dict[str, list[dict]] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = {
                ex.submit(call_teacher, api_key, r["problem"], f"{r['id']}#{k}"): (r, k)
                for r, k in jobs
            }
            for fut in as_completed(futs):
                r, k = futs[fut]
                results.setdefault(r["id"], []).append(fut.result())
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(jobs)} generations done", flush=True)

        # Score in MAIN thread (math-verify signal.alarm constraint)
        regen_ok = 0
        for r in need_regen:
            best = None
            for cand in sorted(results.get(r["id"], []), key=lambda c: c["tag"]):
                full = cand["full_response"]
                if not full:
                    continue
                pred = extract_boxed(full)
                ok = mv_verify(pred, r["expected"]) or any(
                    mv_verify(p, e)
                    for e in _variants(str(r["expected"]))
                    for p in _variants(str(pred or ""))
                )
                if ok:
                    thinking, solution = split_thinking(full)
                    best = dict(r)
                    best.update(
                        full_response=full, thinking=thinking, solution=solution,
                        predicted=pred, correct=True, correct_mathverify=True,
                        recovery="best_of_4", recovery_attempt=cand["tag"],
                    )
                    break
            if best:
                recovered.append(best)
                regen_ok += 1
        print(f"Stage B recovered: {regen_ok}/{len(need_regen)}")
        summary["stage_b_attempted"] = len(need_regen)
        summary["stage_b_recovered"] = regen_ok

    with open(OUTPUT, "w") as f:
        for r in recovered:
            f.write(json.dumps(r) + "\n")
    summary["total_recovered"] = len(recovered)
    SUMMARY.parent.mkdir(exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nTOTAL recovered: {len(recovered)}/{len(dropped)} -> {OUTPUT}")
    print(f"summary -> {SUMMARY}")


if __name__ == "__main__":
    main()
