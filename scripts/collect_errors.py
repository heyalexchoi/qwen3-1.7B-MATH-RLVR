#!/usr/bin/env python3
"""collect_errors.py — on-policy error harvest for SFT-v3b (Written by Claude Opus 4.8, 2026-06-29).

v3b step 4 (docs/sft-v3-plan.md §3, docs/learnings.md L3): sample the TRAINED student (v3a)
k=8 on a hard slice of MATH-*train*, score with math-verify, and dump the WRONG rollouts. Those
wrong attempts are on-policy and in-template (v3a's own format) — the prerequisite for building
corrections that actually transfer (L3: base-model errors are wrong-format and don't).

This is the GPU phase. It produces two things:
  1. A by-tier accuracy table (greedy pass@1 / pass@k / per-sample acc), printed next to the
     v3a MATH-500 by-tier numbers. NOTE ON FRAMING: v3a was SFT'd on distilled traces of these
     exact MATH-train problems, so the null hypothesis is **train ≥ test at every tier** — a gap
     is expected and *is* the train/test generalization signal, NOT a sampling bug. The table's
     job is to confirm the pipeline is wired right (ballpark agreement) and to size the harvest
     off MEASURED train accuracy, not the test wrong-rates.
  2. A self-contained harvest file (one row per wrong rollout: problem, expected, response,
     predicted, plus per-problem n_correct/k) for the NEXT script.

NEXT script (separate, API phase — NOT here): bucket the wrong rollouts by error type
(arithmetic-slip / wrong-method / wrong-setup / misread; n_correct/k seeds this — 8/8-wrong is a
capability gap, 4/8 an unstable slip) and have the teacher correct the prefixes into
check:→fix: recovery episodes. v3b = 7,340 clean + 20–30% recovery (≈2.5k corrected episodes).

MATH-train (train split, 7,500) and MATH-500 (⊂ test) are disjoint — the recovery set is
automatically off the eval set. No overlap filter needed.

Methodology is pinned IDENTICAL to math500_eval.py --format chat (same temp/top_p/top_k/stop ids/
max_new_tokens) so the train-vs-test comparison is apples-to-apples.

Pilot-then-scale on ONE pod: run a small --max_problems first, read the by-tier table + actual
wrong-yield, then raise the cap and resume (resume keys on (id, pass_type, sample_idx), so no
re-spend). Example:
  python scripts/collect_errors.py --max_problems 240          # pilot
  python scripts/collect_errors.py --max_problems 1500         # resumes, fills in the rest
"""

import argparse
import datetime
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the field-AGNOSTIC eval helpers (these don't assume MATH-500's ex["answer"]/unique_id).
# The row/aggregate layer below is written against MATH-train's id/expected fields.
from math500_eval import (  # noqa: E402
    FORMAT_DEFAULTS,
    HF_RESULTS_REPO,
    build_vllm,
    create_prompt_chat,
    extract_boxed,
    get_stop_ids,
    load_tokenizer_safe,
    score_correct,
    upload_artifact,
    vllm_generate_chunk,
    vllm_sampling_params,
    _provenance,
    MATH_VERIFY_AVAILABLE,
)

TRAIN = "data/math_train.jsonl"

# v3a MATH-500 by-tier greedy pass@1 (docs/sft-v3-plan.md §5) — reference column for the
# train-vs-test table. Remember: train ≥ test is the expected (null) relationship here.
V3A_MATH500_GREEDY = {"Level 1": 0.837, "Level 2": 0.644, "Level 3": 0.552,
                      "Level 4": 0.477, "Level 5": 0.209}
V3A_MATH500_OVERALL_GREEDY = 0.482  # 241/500


# ---------------------------------------------------------------------------
# Data selection — hard slice of MATH-train, stratified cap for pilot-then-scale
# ---------------------------------------------------------------------------

def load_train(levels, max_problems, seed):
    data = [json.loads(l) for l in open(TRAIN)]
    selected = [d for d in data if d["level"] in levels]
    by_level = defaultdict(list)
    for d in selected:
        by_level[d["level"]].append(d)
    if not max_problems:
        return selected, by_level
    rng = random.Random(seed)
    total = len(selected)
    work = []
    for lvl in sorted(by_level):
        rng.shuffle(by_level[lvl])
        share = round(max_problems * len(by_level[lvl]) / total)
        work += by_level[lvl][:share]
    rng.shuffle(work)
    work = work[:max_problems]  # trim rounding overflow
    wb = defaultdict(list)
    for d in work:
        wb[d["level"]].append(d)
    return work, wb


# ---------------------------------------------------------------------------
# Per-sample JSONL (resume-safe) — keyed on MATH-train id/expected
# ---------------------------------------------------------------------------

def write_rows(jsonl_path, ex, pass_type, samples, model, fmt, max_new_tokens):
    with open(jsonl_path, "a") as f:
        for sidx, (text, n_tokens) in enumerate(samples):
            pred = extract_boxed(text)
            row = {"id": ex["id"], "pass_type": pass_type, "sample_idx": sidx,
                   "level": ex["level"], "subject": ex.get("subject"),
                   "expected": str(ex["expected"]), "problem": ex["problem"],
                   "response": text, "predicted": pred, "n_tokens": n_tokens,
                   "correct": score_correct(pred, str(ex["expected"])) if MATH_VERIFY_AVAILABLE else None,
                   "format": fmt, "max_new_tokens": max_new_tokens, "model": model}
            f.write(json.dumps(row) + "\n")


def load_done(jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((str(r["id"]), r["pass_type"], r["sample_idx"]))
    return done


def aggregate(jsonl_path):
    """id -> {meta, greedy:[rows], sample:[rows]} using the latest write per (id,ptype,sidx)."""
    latest, meta = {}, {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r["id"]), r["pass_type"], r["sample_idx"])
            latest[key] = r
            meta[str(r["id"])] = {"level": r["level"], "subject": r.get("subject"),
                                  "expected": r["expected"], "problem": r["problem"]}
    grouped = defaultdict(lambda: {"greedy": [], "sample": []})
    for (uid, ptype, sidx), r in latest.items():
        grouped[uid][ptype].append((sidx, r))
    out = {}
    for uid, g in grouped.items():
        greedy = [r for _, r in sorted(g["greedy"])]
        sample = [r for _, r in sorted(g["sample"])]
        out[uid] = {"meta": meta[uid], "greedy": greedy, "sample": sample}
    return out


# ---------------------------------------------------------------------------
# Summary (by-tier train-vs-test) + harvest
# ---------------------------------------------------------------------------

def summarize(agg):
    lvl = defaultdict(lambda: {"n": 0, "greedy_c": 0, "passk_c": 0, "k_n": 0, "k_c": 0})
    for uid, a in agg.items():
        L = a["meta"]["level"]
        s = lvl[L]
        s["n"] += 1
        if a["greedy"]:
            s["greedy_c"] += int(bool(a["greedy"][0].get("correct")))
        if a["sample"]:
            s["passk_c"] += int(any(r.get("correct") for r in a["sample"]))
            s["k_n"] += len(a["sample"])
            s["k_c"] += sum(int(bool(r.get("correct"))) for r in a["sample"])
    rows = {}
    for L, s in lvl.items():
        rows[L] = {"n": s["n"],
                   "greedy_pass1": round(s["greedy_c"] / s["n"], 4) if s["n"] else None,
                   "pass_at_k": round(s["passk_c"] / s["n"], 4) if s["n"] else None,
                   "per_sample_acc": round(s["k_c"] / s["k_n"], 4) if s["k_n"] else None,
                   "n_wrong_rollouts": s["k_n"] - s["k_c"]}
    n = sum(s["n"] for s in lvl.values()) or 1
    overall = {"n": sum(s["n"] for s in lvl.values()),
               "greedy_pass1": round(sum(s["greedy_c"] for s in lvl.values()) / n, 4),
               "n_wrong_rollouts": sum(r["n_wrong_rollouts"] for r in rows.values())}
    return rows, overall


def print_table(rows, overall):
    print("\n=== v3a on MATH-train hard slice — train vs test by tier ===")
    print("(null hypothesis: TRAIN >= TEST per tier — v3a was SFT'd on these problems;")
    print(" a gap is the generalization signal, not a sampling bug.)\n")
    print(f"{'tier':9} {'n':>5} {'greedy':>8} {'(MATH500)':>10} {'pass@k':>8} "
          f"{'persamp':>8} {'wrong_roll':>11}")
    for L in sorted(rows):
        r = rows[L]
        ref = V3A_MATH500_GREEDY.get(L)
        refs = f"{ref:.1%}" if ref is not None else "—"
        g = f"{r['greedy_pass1']:.1%}" if r["greedy_pass1"] is not None else "—"
        pk = f"{r['pass_at_k']:.1%}" if r["pass_at_k"] is not None else "—"
        ps = f"{r['per_sample_acc']:.1%}" if r["per_sample_acc"] is not None else "—"
        print(f"{L:9} {r['n']:>5} {g:>8} {refs:>10} {pk:>8} {ps:>8} {r['n_wrong_rollouts']:>11}")
    print(f"\noverall greedy {overall['greedy_pass1']:.1%} "
          f"(MATH-500 v3a {V3A_MATH500_OVERALL_GREEDY:.1%}) | "
          f"total wrong rollouts harvested: {overall['n_wrong_rollouts']}")


def write_harvest(harvest_path, agg, include_greedy_wrong):
    """One row per WRONG rollout — self-contained for the API correction step."""
    n = 0
    with open(harvest_path, "w") as f:
        for uid in sorted(agg, key=lambda u: (agg[u]["meta"]["level"], u)):
            a = agg[uid]
            m = a["meta"]
            k = len(a["sample"])
            n_correct = sum(int(bool(r.get("correct"))) for r in a["sample"])
            base = {"id": uid, "level": m["level"], "subject": m["subject"],
                    "problem": m["problem"], "expected": m["expected"],
                    "k": k, "n_correct_of_k": n_correct}
            for r in a["sample"]:
                if not r.get("correct"):
                    f.write(json.dumps({**base, "pass_type": "sample",
                                        "sample_idx": r["sample_idx"], "response": r["response"],
                                        "predicted": r["predicted"], "n_tokens": r.get("n_tokens")}) + "\n")
                    n += 1
            if include_greedy_wrong and a["greedy"] and not a["greedy"][0].get("correct"):
                r = a["greedy"][0]
                f.write(json.dumps({**base, "pass_type": "greedy", "sample_idx": 0,
                                    "response": r["response"], "predicted": r["predicted"],
                                    "n_tokens": r.get("n_tokens")}) + "\n")
                n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="heyalexchoi/qwen3-1.7b-math-sft-v3a")
    p.add_argument("--levels", default="Level 3,Level 4,Level 5",
                   help="comma-separated MATH levels to harvest (the hard slice).")
    p.add_argument("--max_problems", type=int, default=0,
                   help="0 = all problems in the slice; else a stratified cap (pilot-then-scale).")
    p.add_argument("--n_samples", type=int, default=8, help="k for pass@k sampling.")
    p.add_argument("--format", choices=["chat"], default="chat",
                   help="chat only — v3a is an SFT model; methodology pinned to the eval.")
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--upload_every", type=int, default=50,
                   help="upload the JSONL to HF every N problems (0 = only at end).")
    p.add_argument("--include_greedy_wrong", action="store_true",
                   help="also harvest the greedy-wrong rollout per problem (default: sampled-wrong only).")
    p.add_argument("--output_name", default=None, help="override the deterministic base filename.")
    p.add_argument("--fresh", action="store_true", help="ignore any existing partial; start clean.")
    p.add_argument("--stats_only", action="store_true", help="print the slice size and exit (no GPU).")
    args = p.parse_args()

    fd = FORMAT_DEFAULTS[args.format]
    for key in ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        if getattr(args, key) is None:
            setattr(args, key, fd[key])

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    work, by_level = load_train(levels, args.max_problems, args.seed)
    print(f"MATH-train hard slice: levels={levels}  selected={len(work)} problems")
    for L in sorted(by_level):
        print(f"  {L}: {len(by_level[L])}")
    if args.stats_only:
        return
    if not work:
        sys.exit("No problems selected — check --levels.")

    lvl_tag = "L" + "".join(L.split()[-1] for L in sorted(levels))
    if args.output_name:
        base = args.output_name
    else:
        cap = f"_n{args.max_problems}" if args.max_problems else ""
        base = f"v3a_mathtrain_{lvl_tag}{cap}_chat_max{args.max_new_tokens}_errors"
    jsonl_path = f"eval_results/{base}_samples.jsonl"
    summary_path = f"eval_results/{base}_summary.json"
    harvest_path = f"eval_results/{base}_harvest.jsonl"
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    print(f"base={base}  k={args.n_samples}  max_new_tokens={args.max_new_tokens} "
          f"temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"rep_pen={args.repetition_penalty}")

    # vLLM only (v3a is an SFT chat model; harvest must match the eval regime).
    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("ERROR: vLLM not installed — this harvest must run on a GPU pod with the pinned stack.")

    tokenizer = load_tokenizer_safe(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = get_stop_ids(tokenizer)
    print(f"Chat stop ids: {stop_ids}")
    prompts = {d["id"]: create_prompt_chat(d["problem"], tokenizer) for d in work}

    if args.fresh and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    done = load_done(jsonl_path)

    def fully_done(d):
        uid = str(d["id"])
        return (uid, "greedy", 0) in done and all((uid, "sample", s) in done
                                                   for s in range(args.n_samples))
    pending = [d for d in work if not fully_done(d)]
    if done:
        print(f"Resume: {len(work) - len(pending)}/{len(work)} done, {len(pending)} pending.")
    if not pending:
        print("All problems already sampled — aggregating.")

    chunk = args.upload_every if args.upload_every and args.upload_every > 0 else max(1, len(pending))
    if pending:
        llm = build_vllm(args.model, None, args.max_new_tokens)
        gp, sp = vllm_sampling_params(args.format, stop_ids, args.max_new_tokens, args.n_samples,
                                      args.temperature, args.top_p, args.top_k, args.repetition_penalty)
        for cs in range(0, len(pending), chunk):
            batch = pending[cs:cs + chunk]
            greedy, sampling = vllm_generate_chunk(llm, [prompts[d["id"]] for d in batch], gp, sp)
            for kk, d in enumerate(batch):
                write_rows(jsonl_path, d, "greedy", [greedy[kk]], args.model, args.format, args.max_new_tokens)
                if sampling is not None:
                    write_rows(jsonl_path, d, "sample", sampling[kk], args.model, args.format, args.max_new_tokens)
            if args.upload_every:
                upload_artifact(jsonl_path, quiet=True)
            print(f"  chunk done: {min(cs + chunk, len(pending))}/{len(pending)} pending problems")

    # Aggregate + by-tier table + harvest
    agg = aggregate(jsonl_path)
    rows, overall = summarize(agg)
    print_table(rows, overall)
    n_harvest = write_harvest(harvest_path, agg, args.include_greedy_wrong)
    print(f"\nHarvest: {n_harvest} wrong rollouts → {harvest_path}")

    prov = _provenance(args, "vllm")
    summary = {"model": args.model, "levels": levels, "max_problems": args.max_problems,
               "n_problems": len(work), "n_samples_k": args.n_samples,
               "methodology": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                               "top_p": args.top_p, "top_k": args.top_k,
                               "repetition_penalty": args.repetition_penalty},
               "by_tier": rows, "overall": overall,
               "math500_v3a_greedy_reference": V3A_MATH500_GREEDY,
               "n_wrong_rollouts_harvested": n_harvest, "provenance": prov,
               "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
               "note": "Live math-verify scoring. Null hypothesis is train>=test per tier "
                       "(v3a SFT'd on these problems). Harvest feeds the API correction step."}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary (git-tracked): {summary_path}")

    # Durable record on HF (samples + harvest are large; summary stays in git).
    for path in (jsonl_path, harvest_path):
        if not upload_artifact(path):
            print(f"UPLOAD FAILED for {path} — rsync eval_results/ before tearing down the pod.")


if __name__ == "__main__":
    main()
