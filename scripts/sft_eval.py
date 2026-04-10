#!/usr/bin/env python3
"""Evaluate SFT checkpoint on MATH-500: pass@1 (greedy) and pass@8 (sampling).

Adapted from 06_math500_eval.py to evaluate an SFT checkpoint and report delta
vs the base model baseline (31.6% pass@1).

Dataset: HuggingFaceH4/MATH-500
  Fields: problem, solution, subject, level (int 1-5), answer (pre-extracted)

Outputs:
  outputs/sft_eval_results.jsonl          — per-example results
  outputs/sft_eval_results_summary.json   — summary statistics

Prompt format: tokenizer.apply_chat_template() — matches SFT training format exactly.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["WANDB_DISABLED"] = "true"

# ---------------------------------------------------------------------------
# math-verify (primary evaluator) — fallback to regex if not installed
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as mv_parse, verify as mv_verify_fn
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

    def answers_match(predicted: str, expected: str) -> bool:
        """Compare with math-verify (ANTLR4/SymPy). Falls back to regex on error."""
        if not predicted or not expected:
            return False
        try:
            gold = mv_parse(f"${expected}$")
            ans = mv_parse(f"${predicted}$")
            if gold and ans:
                return bool(mv_verify_fn(gold, ans))
        except Exception:
            pass
        # Fallback: string normalisation
        return _normalize(predicted) == _normalize(expected)

    MATH_VERIFY_AVAILABLE = True

except ImportError:
    MATH_VERIFY_AVAILABLE = False

    def answers_match(predicted: str, expected: str) -> bool:  # type: ignore[misc]
        """Fallback regex-based comparison (math-verify not installed)."""
        p = _normalize(predicted)
        e = _normalize(expected)
        if p == e:
            return True
        try:
            return abs(float(p) - float(e)) < 1e-6
        except (ValueError, TypeError):
            pass
        return p.lower() == e.lower()


def _normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\.0+$", "", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
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


# ---------------------------------------------------------------------------
# Prompting — chat template (matches SFT training format exactly)
# ---------------------------------------------------------------------------

def create_prompt(problem: str, tokenizer) -> str:
    """
    Format the problem using the model's chat template.

    This MUST match the format used during SFT training: SFTTrainer applies
    Qwen3's chat template, producing:
        <|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n

    Using the few-shot "Problem:/Solution:" format here would cause a severe
    train/eval mismatch — the model would never see its expected BOS/EOS
    context and would generate without stopping (no <|im_end|> trigger).
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )


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
    eos_token_ids: list[int] | None = None,
) -> list[str]:
    """Generate n completions for a prompt. Greedy if n==1 or temperature==0."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Use tokenizer.eos_token_id — for a Qwen3 chat checkpoint this is already
    # set to <|im_end|> (the chat turn terminator). Don't hardcode token IDs.
    stop_ids = eos_token_ids if eos_token_ids else tokenizer.eos_token_id

    if n == 1 or temperature == 0.0:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
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
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
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

BASE_PASS1 = 0.316  # Qwen3-1.7B-Base MATH-500 pass@1 baseline


def main():
    parser = argparse.ArgumentParser(description="MATH-500 pass@1 and pass@8 eval for SFT checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default="/workspace/qwen3-math-rlvr/outputs/sft_checkpoint",
        help="Path to SFT checkpoint or HF model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/qwen3-math-rlvr/outputs/sft_eval_results.json",
        help="Legacy output path (kept for backward compat); main outputs are JSONL + summary",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/workspace/qwen3-math-rlvr/outputs/sft_eval_results.jsonl",
        help="Per-example output JSONL",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="/workspace/qwen3-math-rlvr/outputs/sft_eval_results_summary.json",
        help="Summary JSON output",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset ID or path for MATH-500 test set",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Limit problems (for debugging)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--n_samples", type=int, default=8, help="Samples per problem for pass@k")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for pass@k")
    parser.add_argument("--stats_only", action="store_true", help="Just print dataset stats and exit")
    args = parser.parse_args()

    # Set up file logging — log to both stdout and file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"03a_sft_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    logger.info(
        f"math-verify: {'available' if MATH_VERIFY_AVAILABLE else 'NOT installed — using regex fallback'}"
    )

    # Create output dirs
    for path in [args.output, args.output_jsonl, args.output_summary]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load dataset and report stats
    # -----------------------------------------------------------------------
    logger.info(f"Loading MATH-500 dataset: {args.data}")
    ds = load_dataset(args.data, split="test")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    level_counts = defaultdict(int)
    subject_counts = defaultdict(int)
    for ex in ds:
        level_counts[ex["level"]] += 1
        subject_counts[ex["subject"]] += 1

    logger.info(f"=== MATH-500 Dataset Stats ({len(ds)} problems) ===")
    logger.info("By level:")
    for lvl in sorted(level_counts.keys()):
        logger.info(f"  Level {lvl}: {level_counts[lvl]:4d} problems")
    logger.info("By subject:")
    for subj in sorted(subject_counts.keys()):
        logger.info(f"  {subj:<30s}: {subject_counts[subj]:4d} problems")

    if args.stats_only:
        return

    # -----------------------------------------------------------------------
    # Step 2+3: Load model and run eval
    # -----------------------------------------------------------------------
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build stop token list: always include both the tokenizer's eos_token_id AND
    # <|im_end|> (the Qwen3 chat turn terminator the model was trained to emit).
    # SFT saves tokenizer with eos_token=<|endoftext|> (151643) but the model
    # generates <|im_end|> (151645) at end of each assistant turn — we need both.
    # Token IDs resolved dynamically via tokenizer vocab, no hardcoding.
    _im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _base_eos = tokenizer.eos_token_id
    _base_eos_list = [_base_eos] if isinstance(_base_eos, int) else list(_base_eos)
    stop_token_ids = list(set(_base_eos_list + [_im_end_id]))
    logger.info(f"Stop tokens: {stop_token_ids} (eos_token='{tokenizer.eos_token}' id={_base_eos}, <|im_end|>={_im_end_id})")

    # Verify chat template is available (required for correct eval format)
    assert tokenizer.chat_template is not None, (
        "Tokenizer has no chat template — cannot match SFT training format. "
        "Requires transformers >= 4.51.0 and a Qwen3 tokenizer."
    )
    # Log a sample prompt so we can confirm format looks right
    _sample_prompt = create_prompt("What is 2+2?", tokenizer)
    logger.info(f"Sample prompt (first 200 chars): {repr(_sample_prompt[:200])}")

    # -----------------------------------------------------------------------
    # Resume: load already-scored results from JSONL
    # -----------------------------------------------------------------------
    results = []
    done_problems = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _r = json.loads(_line)
                    results.append(_r)
                    done_problems.add(_r["problem"])
        if done_problems:
            logger.info(f"Resuming: {len(done_problems)} problems already scored, skipping them")

    logger.info(
        f"Running eval: pass@1 (greedy) + pass@{args.n_samples} (temp={args.temperature})..."
    )

    with open(args.output_jsonl, "a") as jsonl_f:
        for ex in tqdm(ds):
            problem = ex["problem"]
            expected = ex["answer"]   # pre-extracted, no \boxed{}
            level = ex["level"]
            subject = ex["subject"]

            if problem in done_problems:
                continue

            prompt = create_prompt(problem, tokenizer)

            # pass@1: greedy
            resp1 = generate_samples(
                model, tokenizer, prompt,
                n=1, temperature=0.0, max_new_tokens=args.max_new_tokens,
                eos_token_ids=stop_token_ids,
            )[0]
            pred1 = extract_boxed(resp1)
            pass1_samples = [
                {"response": resp1, "predicted": pred1, "correct": answers_match(pred1, expected)}
            ]

            # pass@8: sampling
            resps8 = generate_samples(
                model, tokenizer, prompt,
                n=args.n_samples, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=stop_token_ids,
            )
            pass8_samples = []
            for resp in resps8:
                pred = extract_boxed(resp)
                pass8_samples.append(
                    {"response": resp, "predicted": pred, "correct": answers_match(pred, expected)}
                )

            result = {
                "problem": problem,
                "expected": expected,
                "level": level,
                "subject": subject,
                "pass1": pass1_samples,
                "pass8": pass8_samples,
            }
            results.append(result)
            jsonl_f.write(json.dumps(result) + "\n")
            jsonl_f.flush()

    # -----------------------------------------------------------------------
    # Step 4: Score by level
    # -----------------------------------------------------------------------
    pass1_stats = compute_pass_at_k(results, sample_key="pass1")
    pass8_stats = compute_pass_at_k(results, sample_key="pass8")

    delta_pass1 = pass1_stats["overall"] - BASE_PASS1

    logger.info("=== RESULTS ===")
    logger.info(f"Model:          {args.model}")
    logger.info(f"Pass@1 (greedy):  {pass1_stats['overall']:.2%}")
    logger.info(f"Pass@8 (temp={args.temperature}): {pass8_stats['overall']:.2%}")
    logger.info(f"Base pass@1:      {BASE_PASS1:.2%}")
    logger.info(
        f"Delta vs base:    {delta_pass1:+.2%}  "
        f"({'↑ improvement' if delta_pass1 >= 0 else '↓ regression'})"
    )
    logger.info("By level:")
    for lvl in sorted(pass1_stats["per_level"].keys(), key=int):
        p1 = pass1_stats["per_level"][lvl]["pass_at_k"]
        p8 = pass8_stats["per_level"][lvl]["pass_at_k"]
        n = pass1_stats["per_level"][lvl]["total"]
        logger.info(f"  Level {lvl} (n={n:3d}): pass@1={p1:.2%}  pass@8={p8:.2%}")
    logger.info("By subject:")
    for subj in sorted(pass1_stats["per_subject"].keys()):
        p1 = pass1_stats["per_subject"][subj]["pass_at_k"]
        p8 = pass8_stats["per_subject"][subj]["pass_at_k"]
        n = pass1_stats["per_subject"][subj]["total"]
        logger.info(f"  {subj:<30s} (n={n:3d}): pass@1={p1:.2%}  pass@8={p8:.2%}")

    # -----------------------------------------------------------------------
    # Step 5: Save outputs
    # -----------------------------------------------------------------------
    # Per-example JSONL written incrementally during eval (flush after each problem)
    logger.info(f"Per-example JSONL saved to {args.output_jsonl}")

    # Summary JSON
    summary = {
        "model": args.model,
        "evaluator": "math-verify" if MATH_VERIFY_AVAILABLE else "regex-fallback",
        "n_samples_pass8": args.n_samples,
        "temperature_pass8": args.temperature,
        "dataset": args.data,
        "dataset_stats": {
            "total": len(ds),
            "by_level": {str(k): v for k, v in sorted(level_counts.items())},
            "by_subject": dict(sorted(subject_counts.items())),
        },
        "pass1": pass1_stats,
        "pass8": pass8_stats,
        "delta_vs_base": {
            "base_model": "Qwen/Qwen3-1.7B-Base",
            "base_pass1": BASE_PASS1,
            "sft_pass1": pass1_stats["overall"],
            "delta_pass1": round(delta_pass1, 4),
        },
    }
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON saved to {args.output_summary}")

    # Legacy combined output (backward compat)
    combined = {**summary, "results": results}
    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined results saved to {args.output}")

    logger.info("Done.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
