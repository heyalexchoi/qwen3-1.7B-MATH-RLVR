#!/usr/bin/env python3
"""Generate MATH-500 completions: pass@1 (greedy) and pass@8 (sampling).

ONE eval entrypoint for every checkpoint type, selected by --format:
  --format completion  (default)  few-shot text completion, no chat template.
                                   For base / GRPO checkpoints (trained on
                                   "Problem: ...\\nSolution:"). Methodology
                                   defaults: max_new_tokens=2048, temp=0.7.
  --format chat                    tokenizer.apply_chat_template + <think> mode.
                                   For SFT / instruct checkpoints. Defaults:
                                   max_new_tokens=8192, temp=0.6, top_p=0.95,
                                   top_k=20, repetition_penalty=1.05.

Backends: vLLM if installed (continuous batching, fast), else HF generate().

Dataset: HuggingFaceH4/MATH-500
  Fields: problem, solution, subject, level (int 1-5), answer, unique_id

Outputs (NON-gitignored eval_results/, timestamped — see eval_results/README.md):
  eval_results/<tag>_step<N>_<UTC>_math500_results.json   combined (pass1/pass8 schema)
  eval_results/<tag>_step<N>_<UTC>_math500_samples.jsonl  per-sample generations (durable)
Scoring: deferred to rescore_math500.py (math-verify). A live summary is printed
  if math-verify is importable, but the authoritative re-score is rescore_math500.py.
  The pass1/pass8 schema is unchanged, so rescore_math500.py consumes both formats.
"""

import argparse
import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import create_prompt as create_prompt_completion, STOP_STRINGS  # noqa: E402


def load_tokenizer_safe(model_path: str, revision: str = None):
    """Load tokenizer with Qwen3 extra_special_tokens bug workaround.

    Some Qwen3 checkpoints (SFT/instruct, and potentially GRPO outputs) save
    extra_special_tokens as a list in tokenizer_config.json. transformers>=4.51
    expects a dict and raises AttributeError: 'list' object has no attribute 'keys'.
    Qwen3-1.7B-Base is unaffected (field is None). This is a no-op for base model.

    NOTE (2026-06-02): verified locally that coercing the list to {} does not
    change any token id or round-trip — the special tokens live in tokenizer.json.
    See docs/vllm-eos-investigation.md.
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
# Optional math-verify — used only for the live convenience summary.
# Authoritative scoring is rescore_math500.py.
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify

    def score_correct(predicted: str, expected: str) -> bool:
        if not predicted or not expected:
            return False
        try:
            gold = _mv_parse(f"${expected}$")
            ans = _mv_parse(f"${predicted}$")
            if gold and ans:
                return bool(_mv_verify(gold, ans))
        except Exception:
            pass
        return False

    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

    def score_correct(predicted: str, expected: str) -> bool:  # type: ignore[misc]
        return False


# ---------------------------------------------------------------------------
# Prompting — two formats
# ---------------------------------------------------------------------------

def create_prompt_chat(problem: str, tokenizer) -> str:
    """Chat-template prompt matching SFT training format (thinking mode)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompts(ds_list, fmt: str, tokenizer) -> list[str]:
    if fmt == "chat":
        assert tokenizer is not None and tokenizer.chat_template is not None, \
            "--format chat requires a tokenizer with a chat template."
        return [create_prompt_chat(ex["problem"], tokenizer) for ex in ds_list]
    return [create_prompt_completion(ex["problem"]) for ex in ds_list]


def get_stop_ids(tokenizer) -> list[int]:
    """Stop token ids for chat format: eos_token ∪ <|im_end|> (Qwen3 turn terminator).

    Passed explicitly to both backends so termination never depends on which file
    (config.json vs generation_config.json) a backend happens to read — the latent
    bug that the old completion-only vLLM path had. See docs/vllm-eos-investigation.md.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    base_eos = tokenizer.eos_token_id
    base_list = [base_eos] if isinstance(base_eos, int) else list(base_eos)
    ids = [i for i in base_list + [im_end_id] if isinstance(i, int) and i >= 0]
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# HF generation backend
# ---------------------------------------------------------------------------

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    n_per_prompt: int,
    temperature: float,
    max_new_tokens: int,
    fmt: str,
    stop_ids: list[int] | None,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float,
) -> list[list[tuple[str, int]]]:
    """Generate n_per_prompt completions per prompt, batched across all prompts.

    Returns results[i] = list of (text, n_tokens) for prompts[i].
    Completion format stops on STOP_STRINGS; chat format stops on stop_ids
    (eos ∪ <|im_end|>).
    """
    B = len(prompts)
    do_sample = temperature > 0.0

    expanded = [p for p in prompts for _ in range(n_per_prompt)]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        expanded,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    tokenizer.padding_side = old_padding_side

    padded_input_len = inputs.input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if fmt == "chat":
        gen_kwargs["eos_token_id"] = stop_ids
    else:
        gen_kwargs["stop_strings"] = STOP_STRINGS
        gen_kwargs["tokenizer"] = tokenizer
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    results: list[tuple[str, int]] = []
    for i in range(B * n_per_prompt):
        gen_ids = out[i][padded_input_len:]
        n_tokens = int((gen_ids != tokenizer.pad_token_id).sum())
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append((text, n_tokens))
    return [results[i * n_per_prompt:(i + 1) * n_per_prompt] for i in range(B)]


# ---------------------------------------------------------------------------
# vLLM backend (preferred — continuous batching, no padding waste)
# ---------------------------------------------------------------------------

def run_eval_vllm(
    model_id: str,
    revision: str,
    prompts: list[str],
    *,
    fmt: str,
    stop_ids: list[int] | None,
    max_new_tokens: int,
    n_samples: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float,
) -> tuple[list[tuple[str, int]], list[list[tuple[str, int]]]]:
    """Greedy + pass@N for all prompts via vLLM.

    Returns (greedy[i]=(text,n_tokens), sampling[i]=list of (text,n_tokens)).
    """
    from vllm import LLM, SamplingParams

    max_model_len = max_new_tokens + 4096
    llm = LLM(
        model=model_id,
        revision=revision or "main",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
    )

    common = {}
    if repetition_penalty and repetition_penalty != 1.0:
        common["repetition_penalty"] = repetition_penalty
    # Chat: stop on token ids (eos ∪ <|im_end|>). Completion: stop on strings.
    if fmt == "chat":
        common["stop_token_ids"] = stop_ids
    else:
        common["stop"] = list(STOP_STRINGS)

    greedy_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, **common)
    sampling_kwargs = dict(n=n_samples, temperature=temperature, max_tokens=max_new_tokens, **common)
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    sampling_params = SamplingParams(**sampling_kwargs)

    print(f"vLLM greedy: {len(prompts)} prompts...")
    greedy_outputs = llm.generate(prompts, greedy_params)
    greedy = [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in greedy_outputs]

    print(f"vLLM sampling (n={n_samples}): {len(prompts)} prompts × {n_samples} samples...")
    sampling_outputs = llm.generate(prompts, sampling_params)
    sampling = [[(c.text, len(c.token_ids)) for c in o.outputs] for o in sampling_outputs]

    return greedy, sampling


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
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
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
# Methodology defaults — per format. Override at CLI only when methodology changes.
# ---------------------------------------------------------------------------

FORMAT_DEFAULTS = {
    "completion": dict(max_new_tokens=2048, temperature=0.7, top_p=None, top_k=None,
                       repetition_penalty=1.0),
    "chat":       dict(max_new_tokens=8192, temperature=0.6, top_p=0.95, top_k=20,
                       repetition_penalty=1.05),
}
N_SAMPLES = 8  # pass@8


# ---------------------------------------------------------------------------
# Model tag / checkpoint resolution
# ---------------------------------------------------------------------------

def _list_checkpoint_commits(model: str, revision: str = None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub required: pip install huggingface_hub")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    commits = list(api.list_repo_commits(model, repo_type="model", revision=revision or "main"))
    step_pattern = re.compile(r"\bstep\s+(\d+)[,\s].*checkpoint", re.IGNORECASE)
    results = []
    for c in commits:
        m = step_pattern.search(c.title or "")
        if m:
            results.append((int(m.group(1)), c.commit_id))
    return sorted(results)


def _resolve_latest_step(model: str, revision: str = None) -> int:
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
    from pathlib import Path as _Path
    if _Path(model).exists():
        raise RuntimeError("Auto-revision requires a HF Hub model ID, not a local path.")
    commits = _list_checkpoint_commits(model)
    for s, commit_id in commits:
        if s == step:
            return commit_id
    available = sorted(s for s, _ in commits)
    raise RuntimeError(
        f"Step {step} not found in {model} checkpoint commits. Available steps: {available}"
    )


def _model_tag(model: str) -> str:
    name = model.split("/")[-1].lower()
    if "grpo" in name:
        return "grpo"
    if "sft" in name:
        return "sft"
    if "base" in name:
        return "base"
    return re.sub(r"[^A-Za-z0-9._-]", "-", name) or "model"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _provenance(args):
    import subprocess, hashlib
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(*a):
        try:
            return subprocess.check_output(["git", "-C", here, *a],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    weights_sha = None
    try:
        mp = args.model
        if os.path.isdir(mp):
            st = os.path.join(mp, "model.safetensors")
            if os.path.exists(st):
                h = hashlib.sha256()
                with open(st, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                weights_sha = h.hexdigest()
    except Exception:
        pass
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": " ".join([sys.executable.split("/")[-1], *sys.argv]),
        "weights_sha256": weights_sha,
        "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Per-sample JSONL (durable artifact + analyzable token counts)
# ---------------------------------------------------------------------------

def _write_sample_rows(jsonl_path, ex, i, pass_type, samples, max_new_tokens, fmt, model):
    """Append per-sample rows. `samples` = list of (text, n_tokens)."""
    uid = ex.get("unique_id", str(i))
    with open(jsonl_path, "a") as f:
        for sample_idx, (text, n_tokens) in enumerate(samples):
            pred = extract_boxed(text)
            row = {
                "unique_id": uid,
                "pass_type": pass_type,           # "greedy" or "sample"
                "sample_idx": sample_idx,
                "problem": ex["problem"],
                "expected": ex["answer"],
                "level": ex["level"],
                "subject": ex["subject"],
                "response": text,
                "predicted": pred,
                "n_tokens": n_tokens,
                "max_new_tokens": max_new_tokens,
                "format": fmt,
                "model": model,
            }
            if MATH_VERIFY_AVAILABLE:
                row["correct"] = score_correct(pred, ex["answer"])
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MATH-500 pass@1 (greedy) + pass@8 eval. One entrypoint, --format selects "
                    "completion (base/GRPO) or chat (SFT/instruct). Methodology defaults are "
                    "per-format (see FORMAT_DEFAULTS); override only when methodology changes."
    )
    parser.add_argument("--model", type=str, default="heyalexchoi/qwen3-1.7b-math-grpo",
                        help="HF Hub model ID or local path.")
    parser.add_argument("--format", choices=["completion", "chat"], default="completion",
                        help="completion = few-shot text (base/GRPO); chat = apply_chat_template (SFT).")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Optimizer step (names the output file). Required unless --latest "
                             "or a local model path is given.")
    parser.add_argument("--latest", action="store_true",
                        help="Query HF Hub for the latest checkpoint step and use it.")
    parser.add_argument("--revision", type=str, default=None,
                        help="HF git revision. Auto-resolved from --checkpoint_step if omitted.")
    parser.add_argument("--problem_batch_size", type=int, default=8,
                        help="Problems per HF generate() call (ignored for vLLM). Reduce if OOM.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit problems (debug).")
    parser.add_argument("--stats_only", action="store_true", help="Print dataset stats and exit.")
    parser.add_argument("--upload", action="store_true",
                        help=f"Upload outputs to HF dataset repo ({HF_RESULTS_REPO}). Needs HF_TOKEN.")
    parser.add_argument("--backend", choices=["vllm", "hf", "auto"], default="auto",
                        help="Inference backend. auto = vllm if importable, else hf.")
    # Methodology overrides (default None → resolved from FORMAT_DEFAULTS[--format])
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run id in output filenames (default: UTC timestamp).")
    args = parser.parse_args()

    if args.latest and args.checkpoint_step:
        parser.error("--latest and --checkpoint_step are mutually exclusive.")

    # Resolve methodology defaults from format, allowing explicit CLI overrides.
    fd = FORMAT_DEFAULTS[args.format]
    if args.max_new_tokens is None:
        args.max_new_tokens = fd["max_new_tokens"]
    if args.temperature is None:
        args.temperature = fd["temperature"]
    if args.top_p is None:
        args.top_p = fd["top_p"]
    if args.top_k is None:
        args.top_k = fd["top_k"]
    if args.repetition_penalty is None:
        args.repetition_penalty = fd["repetition_penalty"]

    from pathlib import Path as _Path
    is_local = _Path(args.model).exists()

    if args.latest:
        args.checkpoint_step = _resolve_latest_step(args.model, args.revision)
        print(f"Latest checkpoint step: {args.checkpoint_step}")

    if args.checkpoint_step and not args.revision and not is_local:
        print(f"Auto-resolving HF revision for step {args.checkpoint_step}...")
        args.revision = _resolve_step_revision(args.model, args.checkpoint_step)
        print(f"Resolved: step {args.checkpoint_step} → {args.revision}")

    # Resolve backend
    use_vllm = False
    if args.backend in ("auto", "vllm"):
        try:
            import vllm  # noqa: F401
            use_vllm = True
        except ImportError:
            if args.backend == "vllm":
                print("ERROR: --backend vllm requested but vllm not installed.", flush=True)
                sys.exit(1)

    tag = _model_tag(args.model)
    run_id = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    step_part = f"step{args.checkpoint_step}" if args.checkpoint_step is not None else "local"
    base = f"eval_results/{tag}_{step_part}_{args.format}_{run_id}_math500"
    output_path = f"{base}_results.json"
    jsonl_path = f"{base}_samples.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}\nSamples: {jsonl_path}")
    print(f"Format={args.format} backend={'vllm' if use_vllm else 'hf'} "
          f"max_new_tokens={args.max_new_tokens} temp={args.temperature} "
          f"n_samples={args.n_samples} rep_penalty={args.repetition_penalty}")

    # ----- dataset + stats -----
    print("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    level_counts, subject_counts = defaultdict(int), defaultdict(int)
    for ex in ds:
        level_counts[ex["level"]] += 1
        subject_counts[ex["subject"]] += 1
    print(f"\n=== MATH-500 Dataset Stats ({len(ds)} problems) ===")
    for lvl in sorted(level_counts):
        print(f"  Level {lvl}: {level_counts[lvl]:4d} problems")
    for subj in sorted(subject_counts):
        print(f"  {subj:<30s}: {subject_counts[subj]:4d} problems")
    if args.stats_only:
        return

    ds_list = list(ds)
    revision_str = args.revision or "main"

    # Tokenizer (needed for chat prompts and for stop ids; also patches vLLM cache).
    tokenizer = load_tokenizer_safe(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = get_stop_ids(tokenizer) if args.format == "chat" else None
    if args.format == "chat":
        print(f"Chat stop token ids: {stop_ids} "
              f"(eos={tokenizer.eos_token!r}, <|im_end|>={tokenizer.convert_tokens_to_ids('<|im_end|>')})")

    all_prompts = build_prompts(ds_list, args.format, tokenizer)
    print(f"Sample prompt (first 200 chars): {all_prompts[0][:200]!r}")

    results = []

    if use_vllm:
        hf_commit_hash = args.revision
        greedy, sampling = run_eval_vllm(
            args.model, args.revision, all_prompts,
            fmt=args.format, stop_ids=stop_ids,
            max_new_tokens=args.max_new_tokens, n_samples=args.n_samples,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        for i, ex in enumerate(ds_list):
            g_text, g_tok = greedy[i]
            _write_sample_rows(jsonl_path, ex, i, "greedy", [(g_text, g_tok)],
                               args.max_new_tokens, args.format, args.model)
            _write_sample_rows(jsonl_path, ex, i, "sample", sampling[i],
                               args.max_new_tokens, args.format, args.model)
            results.append({
                "problem": ex["problem"], "expected": ex["answer"],
                "level": ex["level"], "subject": ex["subject"],
                "pass1": [{"response": g_text, "predicted": extract_boxed(g_text), "n_tokens": g_tok}],
                "pass8": [{"response": t, "predicted": extract_boxed(t), "n_tokens": n}
                          for t, n in sampling[i]],
            })
    else:
        print(f"Backend: HF | loading model: {args.model} (revision={revision_str})")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        hf_commit_hash = getattr(model.config, "_commit_hash", None)
        print(f"HF commit hash: {hf_commit_hash or 'N/A (local model)'}")

        B = args.problem_batch_size
        for batch_start in tqdm(range(0, len(ds_list), B), total=(len(ds_list) + B - 1) // B,
                                desc="batches"):
            batch = ds_list[batch_start:batch_start + B]
            prompts = all_prompts[batch_start:batch_start + B]
            gkw = dict(fmt=args.format, stop_ids=stop_ids, top_p=args.top_p, top_k=args.top_k,
                       repetition_penalty=args.repetition_penalty)
            greedy_results = generate_batch(model, tokenizer, prompts, n_per_prompt=1,
                                            temperature=0.0, max_new_tokens=args.max_new_tokens, **gkw)
            sampling_results = generate_batch(model, tokenizer, prompts, n_per_prompt=args.n_samples,
                                              temperature=args.temperature,
                                              max_new_tokens=args.max_new_tokens, **gkw)
            for i, ex in enumerate(batch):
                gi = batch_start + i
                _write_sample_rows(jsonl_path, ex, gi, "greedy", greedy_results[i],
                                   args.max_new_tokens, args.format, args.model)
                _write_sample_rows(jsonl_path, ex, gi, "sample", sampling_results[i],
                                   args.max_new_tokens, args.format, args.model)
                results.append({
                    "problem": ex["problem"], "expected": ex["answer"],
                    "level": ex["level"], "subject": ex["subject"],
                    "pass1": [{"response": t, "predicted": extract_boxed(t), "n_tokens": n}
                              for t, n in greedy_results[i]],
                    "pass8": [{"response": t, "predicted": extract_boxed(t), "n_tokens": n}
                              for t, n in sampling_results[i]],
                })

    # ----- combined json (pass1/pass8 schema — rescore_math500.py consumes this) -----
    output = {
        "model": args.model,
        "format": args.format,
        "backend": "vllm" if use_vllm else "hf",
        "hf_revision": revision_str,
        "hf_commit_hash": hf_commit_hash,
        "checkpoint_step": args.checkpoint_step,
        "provenance": _provenance(args),
        "max_new_tokens": args.max_new_tokens,
        "n_samples_pass8": args.n_samples,
        "temperature_pass8": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
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
    print(f"Per-sample generations: {jsonl_path}")

    # ----- live convenience summary (NOT authoritative; rescore_math500.py is) -----
    if MATH_VERIFY_AVAILABLE:
        g_correct = sum(any(score_correct(s["predicted"], r["expected"]) for s in r["pass1"])
                        for r in results)
        k_correct = sum(any(score_correct(s["predicted"], r["expected"]) for s in r["pass8"])
                        for r in results)
        n = len(results)
        print(f"\n[live math-verify summary — re-score with rescore_math500.py for the record]")
        print(f"  greedy pass@1: {g_correct}/{n} = {g_correct/n:.2%}")
        print(f"  pass@{args.n_samples}:      {k_correct}/{n} = {k_correct/n:.2%}")
    else:
        print("\n(math-verify not installed — skipping live summary; run rescore_math500.py to score.)")

    if args.upload:
        upload_artifact(output_path)
        upload_artifact(jsonl_path)


if __name__ == "__main__":
    main()
