#!/usr/bin/env python3
"""Evaluate model on MATH-500: pass@1 (greedy) and pass@8 (sampling).

Dataset: HuggingFaceH4/MATH-500
  Fields: problem, solution, subject, level (int 1-5), answer (pre-extracted)

Outputs: outputs/math500_results.json
  - dataset stats (by level, by subject)
  - pass@1 and pass@8 overall and per level
  - full per-problem results
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer_safe(model_path: str):
    """Load tokenizer with Qwen3 extra_special_tokens bug workaround.

    Some Qwen3 checkpoints (SFT/instruct, and potentially GRPO outputs) save
    extra_special_tokens as a list in tokenizer_config.json. transformers>=4.51
    expects a dict and raises AttributeError: 'list' object has no attribute 'keys'.
    Qwen3-1.7B-Base is unaffected (field is None). This is a no-op for base model.
    """
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except (AttributeError, TypeError) as e:
        if "extra_special_tokens" not in str(e) and "keys" not in str(e):
            raise
        # Patch the cached tokenizer_config.json and retry
        from transformers.utils import cached_file
        config_file = cached_file(model_path, "tokenizer_config.json")
        with open(config_file) as f:
            config = json.load(f)
        if isinstance(config.get("extra_special_tokens"), list):
            config["extra_special_tokens"] = {}
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Patched tokenizer_config.json: extra_special_tokens list → {{}}")
        return AutoTokenizer.from_pretrained(model_path)

# ---------------------------------------------------------------------------
# math-verify — primary evaluator (ANTLR4/SymPy), regex fallback if missing
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as mv_parse, verify as mv_verify_fn
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        # Also try \boxed (without space)
        idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""

    start = idx + len("\\boxed{")
    depth = 1
    result = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            result.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            result.append(ch)
        else:
            result.append(ch)
    return "".join(result).strip()


def normalize(s: str) -> str:
    """Basic normalization for answer comparison."""
    s = s.strip()
    # Remove trailing .0 from integers
    s = re.sub(r"\.0+$", "", s)
    # Remove thousands commas
    s = s.replace(",", "")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def answers_match(predicted: str, expected: str) -> bool:
    """Compare predicted and expected answers. Uses math-verify when available, regex fallback."""
    if not predicted or not expected:
        return False
    if MATH_VERIFY_AVAILABLE:
        try:
            gold = mv_parse(f"${expected}$")
            ans = mv_parse(f"${predicted}$")
            if gold and ans:
                return bool(mv_verify_fn(gold, ans))
        except Exception:
            pass
    # Regex/numeric fallback
    p = normalize(predicted)
    e = normalize(expected)
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 1e-6
    except (ValueError, TypeError):
        pass
    return p.lower() == e.lower()


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

from prompts import FEW_SHOT, create_prompt, STOP_STRINGS  # noqa: E402


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_samples(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """Generate n completions for a prompt. Greedy if n==1 or temperature==0."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    input_len = inputs.input_ids.shape[1]

    if n == 1 or temperature == 0.0:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )
        return [tokenizer.decode(out[0][input_len:], skip_special_tokens=True)]

    # Batch n samples in one forward pass
    input_ids = inputs.input_ids.repeat(n, 1)
    attention_mask = inputs.attention_mask.repeat(n, 1)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=STOP_STRINGS,
            tokenizer=tokenizer,
        )
    return [tokenizer.decode(out[i][input_len:], skip_special_tokens=True) for i in range(n)]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_pass_at_k(results: list[dict], sample_key: str) -> dict:
    """
    Compute pass@k where k = number of samples in sample_key.
    A problem passes if ANY sample is correct.
    """
    level_correct = defaultdict(int)
    level_total = defaultdict(int)
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)

    for r in results:
        level = r["level"]
        subject = r["subject"]
        any_correct = any(s["correct"] for s in r[sample_key])

        level_total[level] += 1
        subject_total[subject] += 1
        if any_correct:
            level_correct[level] += 1
            subject_correct[subject] += 1

    per_level = {}
    for lvl in sorted(level_total.keys()):
        per_level[str(lvl)] = {
            "correct": level_correct[lvl],
            "total": level_total[lvl],
            "pass_at_k": round(level_correct[lvl] / level_total[lvl], 4),
        }

    per_subject = {}
    for subj in sorted(subject_total.keys()):
        per_subject[subj] = {
            "correct": subject_correct[subj],
            "total": subject_total[subj],
            "pass_at_k": round(subject_correct[subj] / subject_total[subj], 4),
        }

    overall_correct = sum(level_correct.values())
    overall_total = sum(level_total.values())
    return {
        "overall": round(overall_correct / overall_total, 4),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_level": per_level,
        "per_subject": per_subject,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MATH-500 pass@1 and pass@8 eval")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output", type=str, default="outputs/math500_results.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit problems (for debugging)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--n_samples", type=int, default=8, help="Samples per problem for pass@k")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for pass@k")
    parser.add_argument("--stats_only", action="store_true", help="Just print dataset stats and exit")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load dataset and report stats
    # -----------------------------------------------------------------------
    print("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    level_counts = defaultdict(int)
    subject_counts = defaultdict(int)
    for ex in ds:
        level_counts[ex["level"]] += 1
        subject_counts[ex["subject"]] += 1

    print(f"\n=== MATH-500 Dataset Stats ({len(ds)} problems) ===")
    print("By level:")
    for lvl in sorted(level_counts.keys()):
        print(f"  Level {lvl}: {level_counts[lvl]:4d} problems")
    print("By subject:")
    for subj in sorted(subject_counts.keys()):
        print(f"  {subj:<30s}: {subject_counts[subj]:4d} problems")
    print()

    if args.stats_only:
        return

    # -----------------------------------------------------------------------
    # Step 2+3: Load model and run eval
    # -----------------------------------------------------------------------
    print(f"Evaluator: {'math-verify (ANTLR4/SymPy)' if MATH_VERIFY_AVAILABLE else 'regex fallback — install math-verify[antlr4_13_2] for authoritative scoring'}")
    print(f"Loading model: {args.model}")
    tokenizer = load_tokenizer_safe(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    print(f"Running eval: pass@1 (greedy) + pass@{args.n_samples} (temp={args.temperature})...")

    for ex in tqdm(ds):
        problem = ex["problem"]
        expected = ex["answer"]   # pre-extracted, no \boxed{}
        level = ex["level"]
        subject = ex["subject"]
        prompt = create_prompt(problem)

        # pass@1: greedy
        resp1 = generate_samples(model, tokenizer, prompt, n=1, temperature=0.0,
                                  max_new_tokens=args.max_new_tokens)[0]
        pred1 = extract_boxed(resp1)
        pass1_samples = [{"response": resp1, "predicted": pred1, "correct": answers_match(pred1, expected)}]

        # pass@8: sampling
        resps8 = generate_samples(model, tokenizer, prompt, n=args.n_samples,
                                   temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        pass8_samples = []
        for resp in resps8:
            pred = extract_boxed(resp)
            pass8_samples.append({"response": resp, "predicted": pred, "correct": answers_match(pred, expected)})

        results.append({
            "problem": problem,
            "expected": expected,
            "level": level,
            "subject": subject,
            "pass1": pass1_samples,
            "pass8": pass8_samples,
        })

    # -----------------------------------------------------------------------
    # Step 4: Score by level
    # -----------------------------------------------------------------------
    pass1_stats = compute_pass_at_k(results, sample_key="pass1")
    pass8_stats = compute_pass_at_k(results, sample_key="pass8")

    print("\n=== RESULTS ===")
    print(f"Pass@1 (greedy): {pass1_stats['overall']:.2%}")
    print(f"Pass@8 (temp={args.temperature}): {pass8_stats['overall']:.2%}")
    print("\nBy level:")
    for lvl in sorted(pass1_stats["per_level"].keys(), key=int):
        p1 = pass1_stats["per_level"][lvl]["pass_at_k"]
        p8 = pass8_stats["per_level"][lvl]["pass_at_k"]
        n = pass1_stats["per_level"][lvl]["total"]
        print(f"  Level {lvl} (n={n:3d}): pass@1={p1:.2%}  pass@8={p8:.2%}")
    print("\nBy subject:")
    for subj in sorted(pass1_stats["per_subject"].keys()):
        p1 = pass1_stats["per_subject"][subj]["pass_at_k"]
        p8 = pass8_stats["per_subject"][subj]["pass_at_k"]
        n = pass1_stats["per_subject"][subj]["total"]
        print(f"  {subj:<30s} (n={n:3d}): pass@1={p1:.2%}  pass@8={p8:.2%}")

    # Save
    output = {
        "model": args.model,
        "evaluator": "math-verify" if MATH_VERIFY_AVAILABLE else "regex-fallback",
        "n_samples_pass8": args.n_samples,
        "temperature_pass8": args.temperature,
        "dataset_stats": {
            "total": len(ds),
            "by_level": {str(k): v for k, v in sorted(level_counts.items())},
            "by_subject": dict(sorted(subject_counts.items())),
        },
        "pass1": pass1_stats,
        "pass8": pass8_stats,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
