#!/usr/bin/env python3
"""
Pre-GRPO verification — Check 2: Stop token and termination sanity.

Confirms base-model rollouts terminate cleanly under the GRPO script's exact
prompting and stop-token configuration. Catches the analog of the EOS-masking
bug from the SFT debug — different mechanism (prompting/stop config rather than
label masking), same failure mode (model never terminates, rollouts hit cap,
gradient signal is garbage).

Uses the EXACT prompt builder and generation config from grpo_train.py.

Procedure:
  1. Sample 20 random MATH train problems (stratified across levels).
  2. Format each with create_prompt() from prompts.py (same as GRPO training).
  3. Generate one rollout per problem using the GRPO generation config:
       - model: Qwen/Qwen3-1.7B-Base (or --model override)
       - temperature: 0.9 (from grpo_config.yaml)
       - max_new_tokens: 2048 (max_completion_length from grpo_config.yaml)
       - stop token: tokenizer.eos_token_id (base model = <|endoftext|>, 151643)
  4. Analyze each rollout: length, termination reason, boxed count, junk.
  5. Save rollouts to disk. Print summary table and PASS/FAIL.

Pass criteria:
  - >=18/20 (90%) terminate via stop token, not by hitting max_new_tokens.
  - >=18/20 contain exactly one \\boxed{...} before the stop token.
  - 0/20 show fake follow-up problems or trailing junk after the answer.
  - Median token length in roughly the same ballpark as baseline (50–500 tokens).

On failure:
  - Cap hits dominate: stop token not configured, or model not producing it.
    Decode the last 20 tokens of failing rollouts and check for stop token.
  - Fake follow-up problems: add "\\n\\nProblem:" to stop sequences.
  - No \\boxed{}: few-shot examples not anchoring format.
  - Wildly different length: prompting drifted from baseline format.

Usage (on pod — requires GPU and model access):
  cd /workspace/qwen3-math-rlvr
  python scripts/check_rollout_termination.py
  python scripts/check_rollout_termination.py --model Qwen/Qwen3-1.7B-Base
  python scripts/check_rollout_termination.py --n_problems 20 --seed 42
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import prompt builder from shared module — exact same as GRPO training
# ---------------------------------------------------------------------------

_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))

from prompts import create_prompt  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_boxed(text: str) -> int:
    """Count number of \\boxed{...} occurrences in text."""
    return len(re.findall(r"\\boxed\{", text))


def has_trailing_junk(text: str) -> bool:
    """
    Detect fake follow-up problems, trailing repetition, or junk after the answer.

    Indicators:
    - "Problem:" appearing after a \\boxed{} answer (model generating fake next problem)
    - "Solution:" appearing after a \\boxed{} answer
    - The few-shot delimiter pattern "\n\nProblem:" in the completion
    - Long repetitive sequences (same phrase 3+ times in the last 200 chars)
    """
    # Find position of last \\boxed{}
    last_boxed = text.rfind("\\boxed{")
    if last_boxed == -1:
        # No boxed answer — check for raw repetition in the whole text
        after_content = text
    else:
        # Look at text after the closing brace of the last \\boxed{}
        brace_end = last_boxed
        depth = 0
        for i, ch in enumerate(text[last_boxed:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_end = last_boxed + i + 1
                    break
        after_content = text[brace_end:]

    # Check for fake follow-up problem patterns
    if re.search(r"\n\s*Problem\s*:", after_content):
        return True
    if re.search(r"\n\s*Solution\s*:", after_content):
        return True

    # Check for repetitive junk in the last 300 chars (any section)
    tail = text[-300:]
    # Detect 3+ consecutive identical tokens/words (simplified check)
    words = tail.split()
    if len(words) >= 6:
        for i in range(len(words) - 5):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return True

    # Check for CSS/code junk patterns that appeared in SFT degeneration
    if re.search(r"BorderStyle|border-style|font-size|margin:|padding:", after_content):
        return True

    return False


def analyze_rollout(
    response_tokens: list[int],
    response_text: str,
    eos_token_id: int,
    max_new_tokens: int,
) -> dict:
    """
    Analyze a single rollout.

    Returns a dict with:
      length_tokens: number of generated tokens
      terminated_by_stop: True if stopped by EOS, False if hit max_new_tokens
      boxed_count: number of \\boxed{} in decoded text (stop-special stripped)
      has_junk: True if trailing junk detected
    """
    n_tokens = len(response_tokens)
    terminated_by_stop = (n_tokens < max_new_tokens) or (
        len(response_tokens) > 0 and response_tokens[-1] == eos_token_id
    )

    boxed_count = count_boxed(response_text)
    junk = has_trailing_junk(response_text)

    return {
        "length_tokens": n_tokens,
        "terminated_by_stop": terminated_by_stop,
        "boxed_count": boxed_count,
        "has_junk": junk,
    }


def load_train_problems(data_path: str, n: int, seed: int) -> list[dict]:
    """
    Load n problems from MATH train set, stratified by level.
    data/math_train.jsonl fields: id, problem, level, subject, expected, expected_solution
    """
    random.seed(seed)

    by_level = {}
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            level = r.get("level", "Unknown")
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(r)

    levels = sorted(by_level.keys())
    per_level = max(1, n // len(levels)) if levels else n
    selected = []
    for level in levels:
        pool = by_level[level]
        selected.extend(random.sample(pool, min(per_level, len(pool))))

    # Shuffle and trim to exactly n
    random.shuffle(selected)
    return selected[:n]


def main():
    parser = argparse.ArgumentParser(description="Check GRPO rollout termination and stop token sanity")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="Model ID or local checkpoint path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/math_train.jsonl",
        help="Path to MATH train set JSONL",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_config.yaml",
        help="Path to grpo_config.yaml",
    )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=20,
        help="Number of problems to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/check_rollout_termination.json",
        help="Output file for rollouts and analysis",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"WARNING: Config not found at {args.config}, using defaults")
        config = {}

    temperature = config.get("temperature", 0.9)
    max_new_tokens = config.get("max_completion_length", 2048)
    max_prompt_length = config.get("max_prompt_length", 1024)

    print(f"GRPO generation config:")
    print(f"  model:          {args.model}")
    print(f"  temperature:    {temperature}")
    print(f"  max_new_tokens: {max_new_tokens}  (max_completion_length)")
    print(f"  max_prompt_len: {max_prompt_length}")
    print()

    # Load problems
    if not Path(args.data).exists():
        print(f"ERROR: Train data not found: {args.data}")
        print("Run from repo root: cd /workspace/qwen3-math-rlvr")
        sys.exit(1)

    print(f"Loading {args.n_problems} train problems from {args.data}...")
    problems = load_train_problems(args.data, args.n_problems, args.seed)
    print(f"  Loaded {len(problems)} problems")
    level_dist = {}
    for p in problems:
        level_dist[p.get("level", "?")] = level_dist.get(p.get("level", "?"), 0) + 1
    print(f"  By level: {dict(sorted(level_dist.items()))}")
    print()

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = tokenizer.eos_token_id
    print(f"  EOS token: {repr(tokenizer.eos_token)} (id={eos_token_id})")
    print(f"  This is the stop token GRPO will use for base model text completion.")
    print()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")
    print()

    # Generate rollouts
    print(f"Generating {len(problems)} rollouts (temperature={temperature}, max_new_tokens={max_new_tokens})...")
    print()

    results = []
    for i, prob in enumerate(problems):
        problem_text = prob["problem"]
        expected = prob.get("expected", "")
        level = prob.get("level", "?")

        prompt = create_prompt(problem_text)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        ).to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,  # standard sampling param alongside temperature
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output[0][input_len:].tolist()
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        analysis = analyze_rollout(
            response_tokens=generated_ids,
            response_text=response_text,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        term_reason = "stop_token" if analysis["terminated_by_stop"] else "max_new_tokens"
        junk_str = "JUNK" if analysis["has_junk"] else "ok"
        boxed_str = str(analysis["boxed_count"])

        print(
            f"  [{i+1:2d}/{len(problems)}] {level:7s} | "
            f"len={analysis['length_tokens']:4d} | "
            f"term={term_reason:<15s} | "
            f"boxed={boxed_str} | "
            f"junk={junk_str}"
        )

        # Print last 20 tokens for cap-hit cases (diagnostic)
        if not analysis["terminated_by_stop"]:
            last_tokens = generated_ids[-20:]
            last_text = tokenizer.decode(last_tokens, skip_special_tokens=False)
            print(f"           [CAP HIT] last 20 tokens: {repr(last_text[:120])}")

        results.append({
            "problem": problem_text,
            "expected": expected,
            "level": level,
            "prompt_len_tokens": input_len,
            "response": response_text,
            "analysis": analysis,
        })

    # -----------------------------------------------------------------------
    # Evaluate pass criteria
    # -----------------------------------------------------------------------
    n = len(results)
    n_stop = sum(1 for r in results if r["analysis"]["terminated_by_stop"])
    n_boxed_one = sum(1 for r in results if r["analysis"]["boxed_count"] == 1)
    n_junk = sum(1 for r in results if r["analysis"]["has_junk"])
    lengths = sorted([r["analysis"]["length_tokens"] for r in results])
    median_len = lengths[n // 2] if n > 0 else 0

    threshold = int(0.9 * n)  # 90% of n problems

    print()
    print("=" * 70)
    print("ROLLOUT TERMINATION CHECK RESULTS")
    print("=" * 70)
    print(f"  Problems tested:           {n}")
    print(f"  Terminated via stop token: {n_stop}/{n}  (need >={threshold})")
    print(f"  Exactly 1 \\boxed{{}}:        {n_boxed_one}/{n}  (need >={threshold})")
    print(f"  Trailing junk:             {n_junk}/{n}  (need 0)")
    print(f"  Median length (tokens):    {median_len}  (expected ~50-500 for base model)")
    print(f"  Length distribution:       min={lengths[0]}, p25={lengths[n//4]}, "
          f"p50={median_len}, p75={lengths[3*n//4]}, max={lengths[-1]}")
    print()

    checks = {
        "stop_token_rate": (n_stop >= threshold, f"{n_stop}/{n} terminated by stop token (need >={threshold})"),
        "boxed_rate": (n_boxed_one >= threshold, f"{n_boxed_one}/{n} have exactly 1 \\boxed{{}} (need >={threshold})"),
        "no_junk": (n_junk == 0, f"{n_junk}/20 have trailing junk (need 0)"),
        "sane_length": (50 <= median_len <= 1500, f"median length {median_len} tokens (expected 50–1500)"),
    }

    passed = True
    for check_name, (ok, msg) in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {msg}")
        if not ok:
            passed = False

    # Failure diagnostics
    if n_stop < threshold:
        cap_rollouts = [r for r in results if not r["analysis"]["terminated_by_stop"]]
        print()
        print(f"DIAGNOSTIC — cap hits ({len(cap_rollouts)} rollouts):")
        print("  Stop token not firing. Check that:")
        print(f"  - eos_token_id={eos_token_id} ({repr(tokenizer.eos_token)}) is passed to generate()")
        print("  - The model actually produces this token (check last 20 tokens above)")
        print("  - If model doesn't produce EOS, try adding '\\n\\nProblem:' as additional stop sequence")
        print()
        print("  First cap-hit rollout (last 200 chars):")
        r = cap_rollouts[0]
        print(f"    {repr(r['response'][-200:])}")

    if n_junk > 0:
        junk_rollouts = [r for r in results if r["analysis"]["has_junk"]]
        print()
        print(f"DIAGNOSTIC — trailing junk ({len(junk_rollouts)} rollouts):")
        print("  Model continues generating after the answer. Fix:")
        print("  - Add '\\n\\nProblem:' to stop sequences (delimiter from few-shot template)")
        print()
        print("  First junk rollout (last 200 chars):")
        r = junk_rollouts[0]
        print(f"    {repr(r['response'][-200:])}")

    if n_boxed_one < threshold:
        no_box = [r for r in results if r["analysis"]["boxed_count"] == 0]
        print()
        print(f"DIAGNOSTIC — missing \\boxed{{}} ({len(no_box)} with 0 boxed):")
        print("  Few-shot examples not anchoring format. Options:")
        print("  - Add more few-shot examples (currently 4)")
        print("  - Verify few-shot examples all use \\boxed{} consistently")
        print("  - Check that create_prompt() matches what baseline eval used")

    # Save rollouts for inspection
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": eos_token_id,
            "n_problems": n,
            "summary": {
                "n_stop_token": n_stop,
                "n_boxed_one": n_boxed_one,
                "n_junk": n_junk,
                "median_length": median_len,
                "passed": passed,
            },
            "rollouts": results,
        }, f, indent=2)
    print()
    print(f"Rollouts saved to: {args.output}")
    print()

    if passed:
        print("PASS — Rollouts terminate cleanly. Safe to launch GRPO.")
    else:
        print("FAIL — Fix stop token / prompting issues before launching GRPO.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
