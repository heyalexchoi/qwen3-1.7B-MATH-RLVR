#!/usr/bin/env python3
"""
rewrite_full.py — Concise-trace distillation at SCALE (Written by Claude Opus 4.8, 2026-06-10).

Concurrent, resumable version of the validated pilot (scripts/rewrite_pilot.py). Distills the
7154 correct Qwen3-32B traces into concise STUDENT demonstrations for SFT v2 of a 1.7B model
(docs/distill-trace-framework.md).

Two-layer design (Alex, 2026-06-08), identical to the pilot:
  - The 32B TEACHER thinks freely in its OWN native reasoning to find the best compression.
    That scratchpad is KEPT (saved for inspection) but NEVER trained on.
  - The teacher's DELIVERABLE is the STUDENT demonstration between ===REASONING===/===SOLUTION===
    markers. We wrap the reasoning in real <think>...</think> ourselves → the SFT v2 target.
  - Inputs (filtered correct_mathverify=True): problem + full_response (full 32B trace) +
    solution. `expected` is used only for the math-verify gate.

Scale additions:
  - ThreadPoolExecutor (--workers) over OpenRouter; blocking requests, so threads suffice.
  - RESUMABLE: writes one JSON line per completed trace to --out (JSONL). On restart, already-
    done ids are skipped, so a crash/interrupt never loses finished work.
  - --limit N samples a stratified-by-level subset (seeded) for the smoke test; omit for the
    full run.

Usage:
  # smoke test (50 stratified, report verify-gate yield):
  python scripts/rewrite_full.py --limit 50 --workers 16 --out data/concise/smoke50.jsonl
  # full run:
  python scripts/rewrite_full.py --workers 24 --out data/concise/concise_sft_v2.jsonl
"""
import argparse, json, os, re, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

TRACES = "data/traces/qwen32b_math_traces_master.jsonl"  # 7,356 verified (incl. 202 recovered)
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

# Retry variant for the ~6% that truncated: the teacher over-thinks past the token budget
# re-deriving an answer it was ALREADY given, and never emits the deliverable. Tell it not to.
SYSTEM_RETRY = SYSTEM.replace(
    "Think freely first (in your own reasoning) about how best to structure and compress the demonstration.",
    "The solution below is ALREADY correct and verified — do NOT re-derive it from scratch. "
    "Think only BRIEFLY about how to compress it (keep your reasoning short). You MUST then emit "
    "the student demonstration; never end without the ===SOLUTION=== section.")

# --- v3 prompt (2026-06-20, reviewed with Alex) ------------------------------
# Changes vs v2, mapped to the SFT-v2 failure modes (docs/sft-v2-results.md §4):
#   clarity-first framing, no token target (length is a data-filter concern);
#   Target states one brief reason for the asked quantity/form                  -> reasoned answers
#   Classify includes a plan-check (precondition confirmed before committing)  -> 4.2 wrong plan
#   Verify = substitute the answer back into the ORIGINAL problem, arithmetic
#   shown (independent of the derivation, so it can actually fail)             -> 4.1 decorative Verify
#   error-episode preservation start-to-finish (Target/Setup/Solve), real only -> 4.1 recovery behavior
#   canonical answer-form clause (bare math-verify-parseable value)            -> L1 (drops _variants reliance)
SYSTEM_V3 = """You are an expert teacher writing a CLEAR, easy-to-follow demonstration solution that a small (1.7B) student model will learn to imitate. You are given a verbose but correct solution; the verbosity is noise. Your job is to remove the noise and make the solution CLEAR — and, as a result, shorter.

Think freely first (in your own reasoning) about how to structure the demonstration. Then output ONLY the student demonstration, as exactly two sections separated by the literal marker lines below:

===REASONING===
Target: what is asked, the answer type, and one brief reason it's that quantity/form.
Classify: name the problem type and the method/template that solves it, plus one line confirming WHY it applies (the precondition it needs, checked against the problem). Then commit — no alternatives beyond at most one rejected.
Setup: restate the problem in the method's canonical form (define variables / write the equations).
Solve:
1. step (one concrete computation or deduction)
2. step
...
Verify: substitute the final answer back into the ORIGINAL problem statement and check it satisfies every stated condition — SHOW the arithmetic of this check, never just assert it. The check must be able to FAIL: test the answer with an operation INDEPENDENT of how you derived it — invert it (expand a factorization, differentiate an integral, square a root) or plug it into the original constraints — never replay the derivation's own computation. If the problem genuinely has no independent check (e.g. "evaluate this numeric expression"), recompute once by a visibly different grouping and say so. Only if substitution is impossible for this problem type (e.g. "count the ways", "find the minimum") do the cheapest independent sanity check available (recount a small case, check bounds/parity/units) and name which.
===SOLUTION===
short clean final solution, ending in \\boxed{ANSWER}

RULES for the demonstration:
- KEEP every arithmetic and algebraic step explicit — never skip a computation. The student CANNOT do multi-digit arithmetic in its head: show every step it could not reconstruct itself. Do not rely on your own capacity.
- Keep it tight — no wandering or repetition. The one canonical restatement in Setup and the per-step self-checks are required and stay; cut everything redundant on top of those: restating the problem again later, mid-solution recaps, re-deriving a result already shown, end-of-section summaries. One template, executed once.
- Mid-solve checks: where a step produces something cheaply falsifiable (a root you can plug in, a sum you can re-add), you MAY add one line "check: <the arithmetic>". Never add rhetorical checks ("this looks right").
- ERROR EPISODES: if the source solution contains a GENUINE mistake the author later caught and fixed — a misread of what's asked, a wrong setup/method, or a computation slip — preserve exactly ONE such episode at the point it occurred (Target, Setup, or Solve), formatted as: the wrong step, then "check: <the test that fails>", then "fix: <corrected step>". If the source has no genuine caught mistake, do NOT invent one — most solutions have none.
- Preserve the correct final answer exactly.
- Canonical answer form: the boxed answer must be a bare math-verify-parseable value — digits not words, no \\text{}/\\mbox{}/units, simplest form.
- Clarity is the goal; length is whatever clarity requires. Do not pad, and never compress away a needed step."""

SYSTEM_V3_RETRY = SYSTEM_V3.replace(
    "Think freely first (in your own reasoning) about how to structure the demonstration.",
    "The solution below is ALREADY correct and verified — do NOT re-derive it from scratch. "
    "Think only BRIEFLY about how to restructure it (keep your reasoning short). You MUST then emit "
    "the student demonstration; never end without the ===SOLUTION=== section.")

PROMPTS = {"v2": (SYSTEM, SYSTEM_RETRY), "v3": (SYSTEM_V3, SYSTEM_V3_RETRY)}

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


def _post(key, messages, temp, max_tokens):
    """One OpenRouter chat call with retry. Returns (content, reasoning) or ('__ERROR__ ...', '')."""
    for attempt in range(5):
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages,
                      "temperature": temp, "top_p": 0.9, "max_tokens": max_tokens},
                timeout=600)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(min(2 ** attempt, 30)); continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning") or ""
            mt = re.match(r"\s*<think>(.*?)</think>\s*(.*)", content, re.DOTALL)
            if mt:
                reasoning = (reasoning + "\n" + mt.group(1)).strip()
                content = mt.group(2)
            return content, reasoning
        except Exception as e:
            if attempt == 4:
                return f"__ERROR__ {e}", ""
            time.sleep(min(2 ** attempt, 30))
    return "__ERROR__ retries", ""


def call_teacher(key, problem, full_response, solution, expected, target_tokens, temp,
                 system=SYSTEM, max_tokens=8000, followup=False):
    """Returns (content, teacher_reasoning). `followup`: if the first call emitted reasoning but
    no ===SOLUTION=== deliverable (truncation), make one more call asking ONLY for the demo."""
    sys_msg = system.replace("MAX_TOKENS", str(target_tokens))
    usr = USER.format(problem=problem, full_response=full_response,
                      solution=solution, expected=expected)
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr}]
    content, reasoning = _post(key, msgs, temp, max_tokens)
    if followup and not content.startswith("__ERROR__") and "===SOLUTION===" not in content:
        # It thought but never delivered. Hand back its own reasoning, demand only the demo.
        msgs += [{"role": "assistant", "content": "(my reasoning)\n" + reasoning[-4000:]},
                 {"role": "user", "content": "Now output ONLY the student demonstration in the exact "
                  "===REASONING=== / ===SOLUTION=== format. Do not think further; just write it."}]
        c2, r2 = _post(key, msgs, temp, max_tokens)
        if not c2.startswith("__ERROR__") and "===SOLUTION===" in c2:
            content = c2
            reasoning = (reasoning + "\n" + r2).strip()
    return content, reasoning


def assemble_student(content):
    """Parse ===REASONING===/===SOLUTION=== → (student_trace, well_formed)."""
    if "===REASONING===" in content and "===SOLUTION===" in content:
        think = content.split("===REASONING===", 1)[1].split("===SOLUTION===", 1)[0].strip()
        soln = content.split("===SOLUTION===", 1)[1].strip()
        return f"<think>\n{think}\n</think>\n\n{soln}", True
    return content, False


def process_one(key, d, target_tokens, temp, system=SYSTEM, max_tokens=8000, followup=False):
    """Runs in a WORKER THREAD: network + assembly only. NO math_verify here — its
    signal.alarm() timeout can't run off the main thread. The verify gate is applied
    in the main loop (see main())."""
    content, teacher_reasoning = call_teacher(
        key, d["problem"], d["full_response"], d.get("solution", ""),
        str(d["expected"]), target_tokens, temp, system=system, max_tokens=max_tokens,
        followup=followup)
    err = content.startswith("__ERROR__")
    student, wf = assemble_student(content)
    boxed = extract_boxed(student)
    return {"id": d["id"], "level": d["level"], "subject": d.get("subject"),
            "expected": str(d["expected"]),
            "orig_chars": len(d["full_response"]), "student_chars": len(student),
            "orig_est_tok": len(d["full_response"]) // 4, "student_est_tok": len(student) // 4,
            "teacher_reasoning_est_tok": len(teacher_reasoning) // 4,
            "boxed": boxed, "verify_ok": None, "well_formed": wf, "error": err,
            "has_think": "<think>" in student and "</think>" in student,
            "problem": d["problem"], "student_trace": student,
            "teacher_reasoning": teacher_reasoning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else stratified subset")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None,
                    help="output JSONL; default auto-derives from --prompt "
                         "(data/concise/concise_sft_<prompt>.jsonl)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v3",
                    help="distillation prompt version (v2 = original; v3 = clarity-first, "
                         "plan-check, substitute-back Verify, conditional error episodes)")
    ap.add_argument("--traces", default=TRACES, help="input teacher-traces JSONL")
    ap.add_argument("--retry-failures", action="store_true",
                    help="re-run only the failed records in --out (truncation fix: retry prompt, "
                         "bigger budget, follow-up emit). Rewrites --out in place.")
    ap.add_argument("--retry-max-tokens", type=int, default=16000)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"data/concise/concise_sft_{args.prompt}.jsonl"

    # Load all correct traces
    all_correct = []
    for line in open(args.traces):
        d = json.loads(line)
        if d.get("correct_mathverify"):
            all_correct.append(d)

    if args.retry_failures:
        retry_failures(args, all_correct); return

    # Optional stratified subsample for smoke test
    if args.limit:
        import random
        random.seed(args.seed)
        by_level = defaultdict(list)
        for d in all_correct:
            by_level[str(d.get("level"))].append(d)
        per = max(1, args.limit // max(1, len(by_level)))
        sample = []
        for lvl in sorted(by_level):
            random.shuffle(by_level[lvl])
            sample += by_level[lvl][:per]
        random.shuffle(sample)
        work = sample[: args.limit]
    else:
        work = all_correct

    # Resume: skip ids already in --out
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [d for d in work if d["id"] not in done_ids]
    print(f"Total correct: {len(all_correct)}  selected: {len(work)}  "
          f"already done: {len(done_ids & {d['id'] for d in work})}  TODO: {len(todo)}")
    print(f"prompt={args.prompt} workers={args.workers} temp={args.temp} out={args.out}")
    if not todo:
        print("Nothing to do."); _summarize(out_path); return

    system_sel, _ = PROMPTS[args.prompt]
    key = get_key()
    lock = threading.Lock()
    fh = open(out_path, "a")
    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, key, d, args.target_tokens, args.temp,
                          system_sel): d for d in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            rec["verify_ok"] = mv_ok(rec["boxed"], rec["expected"])  # main-thread gate
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    rate = n_done / (time.time() - t0)
                    eta = (len(todo) - n_done) / rate if rate else 0
                    print(f"  [{n_done}/{len(todo)}] {rate:.1f}/s  ETA {eta/60:.1f}min", flush=True)
    fh.close()
    print(f"Done {n_done} in {(time.time()-t0)/60:.1f}min")
    _summarize(out_path)


def retry_failures(args, all_correct):
    """Second pass: re-run only the failed records (not verify_ok or not well_formed) with the
    retry prompt + bigger budget + follow-up emit. Rewrites --out in place (good records kept)."""
    out_path = Path(args.out)
    recs = [json.loads(l) for l in open(out_path)]
    # re-gate in main thread (verify_ok stored may be None for older rows)
    for r in recs:
        if r.get("verify_ok") is None:
            r["verify_ok"] = mv_ok(r["boxed"], r["expected"])
    good = {r["id"]: r for r in recs if r["verify_ok"] and r["well_formed"]}
    failed_ids = [r["id"] for r in recs if r["id"] not in good]
    src = {d["id"]: d for d in all_correct}
    redo = [src[i] for i in failed_ids if i in src]
    print(f"records {len(recs)}  good {len(good)}  failed {len(failed_ids)}  re-running {len(redo)} "
          f"(retry-prompt, max_tokens={args.retry_max_tokens}, follow-up emit)")
    if not redo:
        print("No failures to retry."); _summarize(out_path); return

    key = get_key()
    lock = threading.Lock()
    fixed = {}
    recovered = 0
    t0 = time.time(); n = 0
    retry_sys = PROMPTS[args.prompt][1]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, key, d, args.target_tokens, args.temp,
                          retry_sys, args.retry_max_tokens, True): d for d in redo}
        for fut in as_completed(futs):
            rec = fut.result()
            rec["verify_ok"] = mv_ok(rec["boxed"], rec["expected"])
            with lock:
                fixed[rec["id"]] = rec
                n += 1
                if rec["verify_ok"] and rec["well_formed"]:
                    recovered += 1
                if n % 20 == 0 or n == len(redo):
                    print(f"  [{n}/{len(redo)}] recovered {recovered}  "
                          f"{n/(time.time()-t0):.2f}/s", flush=True)

    # Rebuild file: keep the original good rows; for failed ids, use the better of old/new.
    merged = []
    for r in recs:
        if r["id"] in good:
            merged.append(good[r["id"]]); continue
        nw = fixed.get(r["id"])
        if nw and (nw["verify_ok"] and nw["well_formed"]) and not (r["verify_ok"] and r["well_formed"]):
            merged.append(nw)            # retry succeeded → replace
        else:
            merged.append(nw or r)       # keep newest attempt (or old if no retry)
    tmp = out_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_path)
    print(f"Recovered {recovered}/{len(redo)} previously-failed traces.")
    _summarize(out_path)


def _summarize(out_path):
    import statistics as st
    recs = [json.loads(l) for l in open(out_path)]
    if not recs:
        return
    n = len(recs)
    okc = sum(r["verify_ok"] for r in recs)
    wfc = sum(r["well_formed"] for r in recs)
    errc = sum(r.get("error") for r in recs)
    good = [r for r in recs if r["verify_ok"] and r["well_formed"]]
    print(f"\n=== SUMMARY ({out_path}) ===")
    print(f"records: {n}   errors: {errc}")
    print(f"VERIFY-GATE YIELD: {okc}/{n} = {okc/n*100:.1f}%   well-formed: {wfc}/{n}   "
          f"trainable (verify&wf): {len(good)}/{n} = {len(good)/n*100:.1f}%")
    if good:
        toks = [r["student_est_tok"] for r in good]
        orig = [r["orig_est_tok"] for r in good]
        print(f"student est tok median: {int(st.median(toks))} (min {min(toks)}, max {max(toks)})")
        print(f"orig   est tok median: {int(st.median(orig))}  "
              f"→ ~{st.median(orig)/max(1,st.median(toks)):.1f}x compression")


if __name__ == "__main__":
    main()
