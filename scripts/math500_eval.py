#!/usr/bin/env python3
"""Generate completions for MATH-500: pass@1 (greedy) and pass@8 (sampling).

Dataset: HuggingFaceH4/MATH-500
  Fields: problem, solution, subject, level (int 1-5), answer (pre-extracted)

Outputs: outputs/{tag}_step{N}_math500_results.json
  - dataset stats (by level, by subject)
  - per-problem raw generations + extracted \\boxed{} answers (no scoring)

Scoring is done separately by rescore_math500.py (math-verify / ANTLR4+SymPy).
"""

import argparse
import json
import os
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


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

from prompts import FEW_SHOT, create_prompt, STOP_STRINGS  # noqa: E402


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    n_per_prompt: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[str]]:
    """Generate n_per_prompt completions for each prompt, batched across all prompts.

    Uses left-padding so all sequences in the batch share the same input length.
    Returns results[i] = list of n_per_prompt completions for prompts[i].

    With problem_batch_size=B and n_per_prompt=8, this sends B*8 sequences through
    model.generate() in one call — far better GPU utilization than sequential per-problem.
    """
    B = len(prompts)
    do_sample = temperature > 0.0

    # Expand: [p0]*n, [p1]*n, ...  so generate() sees B*n sequences
    expanded = [p for p in prompts for _ in range(n_per_prompt)]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        expanded,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)
    tokenizer.padding_side = old_padding_side

    padded_input_len = inputs.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=STOP_STRINGS,
        tokenizer=tokenizer,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    responses = [
        tokenizer.decode(out[i][padded_input_len:], skip_special_tokens=True)
        for i in range(B * n_per_prompt)
    ]
    # Group back by prompt: results[i] = n_per_prompt completions for prompts[i]
    return [responses[i * n_per_prompt:(i + 1) * n_per_prompt] for i in range(B)]


# ---------------------------------------------------------------------------
# vLLM backend (preferred — continuous batching, no padding waste)
# ---------------------------------------------------------------------------

def run_eval_vllm(
    model_id: str,
    revision: str,
    prompts: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Run greedy + pass@N for all prompts using vLLM.

    vLLM uses PagedAttention and continuous batching — no padding overhead,
    no batch-waits-for-slowest-sequence. Orders of magnitude faster than HF
    for variable-length outputs with high max_new_tokens.

    Returns:
        greedy_responses: list[str], one per prompt
        sampling_responses: list[list[str]], N_SAMPLES per prompt
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        revision=revision or "main",
        dtype="bfloat16",
        trust_remote_code=True,
    )

    greedy_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
        stop=list(STOP_STRINGS),
    )
    sampling_params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        stop=list(STOP_STRINGS),
    )

    print(f"vLLM greedy: {len(prompts)} prompts...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    greedy_responses = [o.outputs[0].text for o in greedy_outputs]

    print(f"vLLM sampling (n={N_SAMPLES}): {len(prompts)} prompts × {N_SAMPLES} samples...")
    sampling_outputs = llm.generate(prompts, sampling_params)
    sampling_responses = [[out.text for out in o.outputs] for o in sampling_outputs]

    return greedy_responses, sampling_responses


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
    # Prefer explicit env var; fall back to cached token (~/.cache/huggingface/token)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    path_in_repo = f"outputs/{Path(local_path).name}"
    print(f"Uploading {local_path} → {repo_id}/{path_in_repo} ...")
    api = HfApi(token=token)  # token=None → uses cached login
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

def _list_checkpoint_commits(model: str, revision: str = None):
    """Return list of (step, commit_id) for all checkpoint commits, sorted by step."""
    import re
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub required: pip install huggingface_hub")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    commits = list(api.list_repo_commits(model, repo_type="model", revision=revision or "main"))
    # Match "step N, checkpoint" (standard HF Trainer hub_strategy='checkpoint' format)
    step_pattern = re.compile(r"\bstep\s+(\d+)[,\s].*checkpoint", re.IGNORECASE)
    results = []
    for c in commits:
        m = step_pattern.search(c.title or "")
        if m:
            results.append((int(m.group(1)), c.commit_id))
    return sorted(results)


def _resolve_latest_step(model: str, revision: str = None) -> int:
    """Query HF Hub commit history and return the highest checkpoint step.

    Scans commit titles for the pattern 'step N, checkpoint' (written by HF Trainer
    hub_strategy='checkpoint'). Returns the largest N found.
    Raises RuntimeError if no checkpoint commits are found or model is local.
    """
    from pathlib import Path as _Path
    if _Path(model).exists():
        raise RuntimeError("--latest requires a HF Hub model ID, not a local path.")
    commits = _list_checkpoint_commits(model, revision)
    if not commits:
        raise RuntimeError(
            f"No checkpoint commits found in {model} commit history. "
            "Titles should contain 'step N, checkpoint' (standard HF Trainer format)."
        )
    return max(step for step, _ in commits)


def _resolve_step_revision(model: str, step: int) -> str:
    """Return the HF commit hash for a specific checkpoint step.

    Looks for commits titled 'Training in progress, step N, checkpoint'.
    Raises RuntimeError if the step is not found.
    """
    from pathlib import Path as _Path
    if _Path(model).exists():
        raise RuntimeError("Auto-revision requires a HF Hub model ID, not a local path.")
    commits = _list_checkpoint_commits(model)
    for s, commit_id in commits:
        if s == step:
            return commit_id
    available = sorted(s for s, _ in commits)
    raise RuntimeError(
        f"Step {step} not found in {model} checkpoint commits. "
        f"Available steps: {available}"
    )


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
                             "If omitted and --checkpoint_step is given, the revision is "
                             "auto-resolved from HF commit history (looks for 'step N, checkpoint'). "
                             "Use to override auto-resolution with an exact hash.")
    parser.add_argument("--problem_batch_size", type=int, default=8,
                        help="Number of problems to batch together per generate() call. "
                             "Higher = better GPU utilization but more VRAM. "
                             "Each call does problem_batch_size * n_samples sequences. "
                             "Default 8 (= 64 sequences per call for pass@8). "
                             "Reduce if OOM.")
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

    # Auto-resolve revision from checkpoint_step if not explicitly provided.
    # --checkpoint_step only names the output file; the actual model revision loaded
    # is controlled by --revision. Without this, you'd load 'main' (latest) regardless
    # of --checkpoint_step — the wrong checkpoint with the right filename.
    from pathlib import Path as _Path
    if args.checkpoint_step and not args.revision and not _Path(args.model).exists():
        print(f"Auto-resolving HF revision for step {args.checkpoint_step}...")
        args.revision = _resolve_step_revision(args.model, args.checkpoint_step)
        print(f"Resolved: step {args.checkpoint_step} → {args.revision}")

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
    # Step 2+3: Load model and generate completions
    # -----------------------------------------------------------------------
    revision_str = args.revision or "main"

    # Detect vLLM early so we skip loading HF model if not needed
    use_vllm = False
    try:
        import vllm  # noqa: F401
        use_vllm = True
    except ImportError:
        pass

    ds_list = list(ds)
    all_prompts = [create_prompt(ex["problem"]) for ex in ds_list]
    results = []

    if use_vllm:
        print(f"Backend: vLLM  |  pass@1 (greedy) + pass@{N_SAMPLES} (temp={TEMPERATURE}) "
              f"max_new_tokens={MAX_NEW_TOKENS}...")
        # Patch the Qwen3 extra_special_tokens bug in the HF cache before vLLM loads tokenizer
        load_tokenizer_safe(args.model, revision=args.revision)
        hf_commit_hash = args.revision  # vLLM doesn't expose commit hash; use resolved revision
        greedy_all, sampling_all = run_eval_vllm(args.model, args.revision, all_prompts)
        for i, ex in enumerate(ds_list):
            resp1 = greedy_all[i]
            pass1_samples = [{"response": resp1, "predicted": extract_boxed(resp1)}]
            pass8_samples = [{"response": r, "predicted": extract_boxed(r)} for r in sampling_all[i]]
            results.append({
                "problem": ex["problem"],
                "expected": ex["answer"],
                "level": ex["level"],
                "subject": ex["subject"],
                "pass1": pass1_samples,
                "pass8": pass8_samples,
            })
    else:
        # HF fallback: load model here (skipped when vLLM is used)
        print(f"Backend: HF (install vllm for faster eval)  |  loading model: {args.model} (revision={revision_str})")
        tokenizer = load_tokenizer_safe(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        hf_commit_hash = getattr(model.config, "_commit_hash", None)
        print(f"HF commit hash: {hf_commit_hash or 'N/A (local model)'}")

        # HF generation waits for the longest sequence in the batch —
        # smaller batch sizes are faster when completion lengths vary widely.
        # Reduce --problem_batch_size to 1 if throughput seems poor.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        B = args.problem_batch_size
        print(f"Backend: HF (vLLM not available)  |  pass@1 (greedy) + pass@{N_SAMPLES} "
              f"(temp={TEMPERATURE}) max_new_tokens={MAX_NEW_TOKENS}, problem_batch_size={B}...")

        for batch_start in tqdm(range(0, len(ds_list), B), total=(len(ds_list) + B - 1) // B,
                                desc="batches"):
            batch = ds_list[batch_start:batch_start + B]
            prompts = all_prompts[batch_start:batch_start + B]

            greedy_results = generate_batch(model, tokenizer, prompts,
                                            n_per_prompt=1, temperature=0.0,
                                            max_new_tokens=MAX_NEW_TOKENS)
            sampling_results = generate_batch(model, tokenizer, prompts,
                                              n_per_prompt=N_SAMPLES, temperature=TEMPERATURE,
                                              max_new_tokens=MAX_NEW_TOKENS)

            for i, ex in enumerate(batch):
                resp1 = greedy_results[i][0]
                pass1_samples = [{"response": resp1, "predicted": extract_boxed(resp1)}]
                pass8_samples = [
                    {"response": resp, "predicted": extract_boxed(resp)}
                    for resp in sampling_results[i]
                ]
                results.append({
                    "problem": ex["problem"],
                    "expected": ex["answer"],
                    "level": ex["level"],
                    "subject": ex["subject"],
                    "pass1": pass1_samples,
                    "pass8": pass8_samples,
                })

    # Save raw generations — scoring is done by rescore_math500.py
    output = {
        "model": args.model,
        "hf_revision": revision_str,
        "hf_commit_hash": hf_commit_hash,
        "checkpoint_step": args.checkpoint_step,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_samples_pass8": N_SAMPLES,
        "temperature_pass8": TEMPERATURE,
        "dataset_stats": {
            "total": len(ds),
            "by_level": {str(k): v for k, v in sorted(level_counts.items())},
            "by_subject": dict(sorted(subject_counts.items())),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if args.upload:
        upload_artifact(output_path)


if __name__ == "__main__":
    main()
