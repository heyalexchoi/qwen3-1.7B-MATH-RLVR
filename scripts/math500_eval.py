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
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer_safe(model_path: str, revision: str = None):
    """Load tokenizer with Qwen3 extra_special_tokens bug workaround.

    Some Qwen3 checkpoints (SFT/instruct, and potentially GRPO outputs) save
    extra_special_tokens as a list in tokenizer_config.json. transformers>=4.51
    expects a dict and raises AttributeError: 'list' object has no attribute 'keys'.
    Qwen3-1.7B-Base is unaffected (field is None). This is a no-op for base model.
    """
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except (AttributeError, TypeError) as e:
        if "extra_special_tokens" not in str(e) and "keys" not in str(e):
            raise
        # Patch the cached tokenizer_config.json and retry
        from transformers.utils import cached_file
        config_file = cached_file(model_path, "tokenizer_config.json", **kwargs)
        with open(config_file) as f:
            config = json.load(f)
        if isinstance(config.get("extra_special_tokens"), list):
            config["extra_special_tokens"] = {}
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Patched tokenizer_config.json: extra_special_tokens list → {{}}")
        return AutoTokenizer.from_pretrained(model_path, **kwargs)

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
# HF artifact upload
# ---------------------------------------------------------------------------

HF_RESULTS_REPO = "heyalexchoi/qwen3-math-rlvr-results"


def upload_artifact(local_path: str, repo_id: str = HF_RESULTS_REPO) -> None:
    """Upload a local file to the HF dataset repo at outputs/<filename>."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("WARNING: huggingface_hub not installed — skipping upload.")
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set — skipping upload.")
        return
    path_in_repo = f"outputs/{Path(local_path).name}"
    print(f"Uploading {local_path} → {repo_id}/{path_in_repo} ...")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded → https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}")


# ---------------------------------------------------------------------------
# Eval methodology constants — change here when methodology changes, not at CLI
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 2048   # sufficient for GRPO/base completions (p99 << 2048)
N_SAMPLES = 8           # pass@8
TEMPERATURE = 0.7       # Qwen3 non-thinking mode recommendation


# ---------------------------------------------------------------------------
# Model tag derivation
# ---------------------------------------------------------------------------

def _resolve_latest_step(model: str, revision: str = None) -> int:
    """Query HF Hub commit history and return the highest checkpoint step.

    Scans commit titles for the pattern 'step N, checkpoint' (written by HF Trainer
    hub_strategy='checkpoint'). Returns the largest N found.
    Raises RuntimeError if no checkpoint commits are found or model is local.
    """
    import re
    from pathlib import Path as _Path
    if _Path(model).exists():
        raise RuntimeError("--latest requires a HF Hub model ID, not a local path.")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub required for --latest: pip install huggingface_hub")

    # Use HF token if available — avoids rate limits on unauthenticated requests
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    commits = list(api.list_repo_commits(model, repo_type="model", revision=revision or "main"))
    step_pattern = re.compile(r"\bstep\s+(\d+)[,\s].*checkpoint", re.IGNORECASE)
    steps = []
    for c in commits:
        m = step_pattern.search(c.title or "")
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise RuntimeError(
            f"No checkpoint commits found in {model} commit history. "
            "Titles should contain 'step N, checkpoint' (standard HF Trainer format)."
        )
    return max(steps)


def _model_tag(model: str) -> str:
    """Short tag for the model used in output filenames.

    heyalexchoi/qwen3-1.7b-math-grpo  →  grpo
    heyalexchoi/qwen3-1.7b-math-sft   →  sft
    Qwen/Qwen3-1.7B-Base              →  base
    anything else                      →  <repo-name>
    """
    name = model.split("/")[-1].lower()
    if "grpo" in name:
        return "grpo"
    if "sft" in name:
        return "sft"
    if "base" in name:
        return "base"
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MATH-500 pass@1 (greedy) and pass@8 eval.\n\n"
                    f"Methodology locked: max_new_tokens={MAX_NEW_TOKENS}, "
                    f"n_samples={N_SAMPLES}, temperature={TEMPERATURE}.\n"
                    "Change constants in the script when methodology changes."
    )
    parser.add_argument("--model", type=str, default="heyalexchoi/qwen3-1.7b-math-grpo",
                        help="HF Hub model ID or local path.")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Optimizer step of the checkpoint (e.g. 3000, 7500). "
                             "Required unless --latest is set.")
    parser.add_argument("--latest", action="store_true",
                        help="Query HF Hub for the latest checkpoint step and use it. "
                             "Mutually exclusive with --checkpoint_step.")
    parser.add_argument("--revision", type=str, default=None,
                        help="HF Hub git revision (branch, tag, or commit hash). "
                             "Default: main. Use to pin a specific HF commit for reproducibility.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N problems (debugging only).")
    parser.add_argument("--stats_only", action="store_true",
                        help="Print dataset stats and exit without running eval.")
    parser.add_argument("--upload", action="store_true",
                        help=f"Upload output to HF dataset repo ({HF_RESULTS_REPO}) after writing. "
                             "Requires HF_TOKEN env var.")
    args = parser.parse_args()

    if args.latest and args.checkpoint_step:
        parser.error("--latest and --checkpoint_step are mutually exclusive.")
    if not args.latest and args.checkpoint_step is None:
        parser.error("One of --checkpoint_step or --latest is required.")

    if args.latest:
        args.checkpoint_step = _resolve_latest_step(args.model, args.revision)
        print(f"Latest checkpoint step: {args.checkpoint_step}")

    tag = _model_tag(args.model)
    output_path = f"outputs/{tag}_step{args.checkpoint_step}_math500_results.json"
    print(f"Output path: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

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
    revision_str = args.revision or "main"
    print(f"Loading model: {args.model} (revision={revision_str})")
    tokenizer = load_tokenizer_safe(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Capture actual HF commit hash for provenance (None for local models)
    hf_commit_hash = getattr(model.config, "_commit_hash", None)
    print(f"HF commit hash: {hf_commit_hash or 'N/A (local model)'}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    print(f"Running eval: pass@1 (greedy) + pass@{N_SAMPLES} (temp={TEMPERATURE}) "
          f"max_new_tokens={MAX_NEW_TOKENS}...")

    for ex in tqdm(ds):
        problem = ex["problem"]
        expected = ex["answer"]   # pre-extracted, no \boxed{}
        level = ex["level"]
        subject = ex["subject"]
        prompt = create_prompt(problem)

        # pass@1: greedy
        resp1 = generate_samples(model, tokenizer, prompt, n=1, temperature=0.0,
                                  max_new_tokens=MAX_NEW_TOKENS)[0]
        pred1 = extract_boxed(resp1)
        pass1_samples = [{"response": resp1, "predicted": pred1, "correct": answers_match(pred1, expected)}]

        # pass@8: sampling
        resps8 = generate_samples(model, tokenizer, prompt, n=N_SAMPLES,
                                   temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
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
    print(f"Pass@8 (temp={TEMPERATURE}): {pass8_stats['overall']:.2%}")
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

    # Save — full provenance in metadata
    output = {
        "model": args.model,
        "hf_revision": revision_str,
        "hf_commit_hash": hf_commit_hash,
        "checkpoint_step": args.checkpoint_step,
        "evaluator": "math-verify" if MATH_VERIFY_AVAILABLE else "regex-fallback",
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_samples_pass8": N_SAMPLES,
        "temperature_pass8": TEMPERATURE,
        "dataset_stats": {
            "total": len(ds),
            "by_level": {str(k): v for k, v in sorted(level_counts.items())},
            "by_subject": dict(sorted(subject_counts.items())),
        },
        "pass1": pass1_stats,
        "pass8": pass8_stats,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if args.upload:
        upload_artifact(output_path)


if __name__ == "__main__":
    main()
