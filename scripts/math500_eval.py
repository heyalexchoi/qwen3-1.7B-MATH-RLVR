#!/usr/bin/env python3
"""Generate MATH-500 completions: pass@1 (greedy) and pass@8 (sampling).

ONE eval entrypoint for every checkpoint type, selected by --format:
  --format completion  (default)  few-shot text completion (base / GRPO).
                                   Defaults: max_new_tokens=2048, temp=0.7.
  --format chat                    apply_chat_template + <think> mode (SFT/instruct).
                                   Defaults: max_new_tokens=8192, temp=0.6,
                                   top_p=0.95, top_k=20, repetition_penalty=1.05.

The single failproof path (no flags to remember):
  python scripts/math500_eval.py --model M --format chat --checkpoint_step N
  → vLLM if available (else HF), deterministic filenames, auto-resume, and the
    HF dataset repo is the durable system-of-record (periodic + final upload).

Storage model:
  * Per-sample generations (large)  → eval_results/<base>_samples.jsonl  → HF (record)
  * Combined pass1/pass8 (large)    → eval_results/<base>_results.json   → HF (record)
  * Metrics-only summary (small)    → eval_results/<base>_summary.json   → git-tracked
  Big files are gitignored; only the summary + RUNS pointer live in git. The
  2026-04 SFT generations were lost because they were gitignored AND never
  uploaded AND never rsync'd — mandatory upload + local JSONL + the pre-teardown
  rsync rule are the three independent layers that close that. See eval_results/README.md.

Scoring: deferred to rescore_math500.py (math-verify, authoritative). A live
summary is printed/saved if math-verify is importable.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import create_prompt as create_prompt_completion, STOP_STRINGS  # noqa: E402

HF_RESULTS_REPO = "heyalexchoi/qwen3-math-rlvr-results"


# ---------------------------------------------------------------------------
# Tokenizer (Qwen3 extra_special_tokens workaround — verified harmless 2026-06-02)
# ---------------------------------------------------------------------------

def load_tokenizer_safe(model_path: str, revision: str = None):
    """Load tokenizer; coerce extra_special_tokens list→{} if transformers rejects it.

    Verified locally that this coercion changes no token id or round-trip (the
    special tokens live in tokenizer.json). See docs/vllm-eos-investigation.md.
    """
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except (AttributeError, TypeError) as e:
        if "extra_special_tokens" not in str(e) and "keys" not in str(e):
            raise
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
# Answer extraction + optional scoring (live summary only; rescore is authoritative)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth, result = 1, []
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
# Prompting + stop tokens
# ---------------------------------------------------------------------------

def create_prompt_chat(problem: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True,
    )


def build_prompts(ds_list, fmt: str, tokenizer) -> list[str]:
    if fmt == "chat":
        assert tokenizer is not None and tokenizer.chat_template is not None, \
            "--format chat requires a tokenizer with a chat template."
        return [create_prompt_chat(ex["problem"], tokenizer) for ex in ds_list]
    return [create_prompt_completion(ex["problem"]) for ex in ds_list]


def get_stop_ids(tokenizer) -> list[int]:
    """Chat stop ids: eos ∪ <|im_end|>. Passed explicitly so termination never
    depends on which config file a backend reads. See docs/vllm-eos-investigation.md."""
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    base_eos = tokenizer.eos_token_id
    base_list = [base_eos] if isinstance(base_eos, int) else list(base_eos)
    ids = [i for i in base_list + [im_end_id] if isinstance(i, int) and i >= 0]
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# HF generation backend
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, prompts, n_per_prompt, temperature, max_new_tokens,
                   fmt, stop_ids, top_p, top_k, repetition_penalty):
    """Return results[i] = list of (text, n_tokens) for prompts[i]."""
    B = len(prompts)
    do_sample = temperature > 0.0
    expanded = [p for p in prompts for _ in range(n_per_prompt)]

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(expanded, return_tensors="pt", padding=True, truncation=True,
                       max_length=4096).to(model.device)
    tokenizer.padding_side = old_side
    padded_input_len = inputs.input_ids.shape[1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tokenizer.pad_token_id)
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if fmt == "chat":
        if stop_ids:
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

    results = []
    for i in range(B * n_per_prompt):
        gen_ids = out[i][padded_input_len:]
        n_tokens = int((gen_ids != tokenizer.pad_token_id).sum())
        results.append((tokenizer.decode(gen_ids, skip_special_tokens=True), n_tokens))
    return [results[i * n_per_prompt:(i + 1) * n_per_prompt] for i in range(B)]


# ---------------------------------------------------------------------------
# vLLM backend — engine built ONCE, generated per chunk (enables periodic upload)
# ---------------------------------------------------------------------------

def build_vllm(model_id, revision, max_new_tokens):
    from vllm import LLM
    return LLM(model=model_id, revision=revision or "main", dtype="bfloat16",
               trust_remote_code=True, max_model_len=max_new_tokens + 4096)


def vllm_sampling_params(fmt, stop_ids, max_new_tokens, n_samples, temperature,
                         top_p, top_k, repetition_penalty):
    from vllm import SamplingParams
    common = {}
    if repetition_penalty and repetition_penalty != 1.0:
        common["repetition_penalty"] = repetition_penalty
    if fmt == "chat":
        if stop_ids:
            common["stop_token_ids"] = stop_ids
        # else (--no_chat_stop_ids): vLLM falls back to config eos — end-token A/B test.
    else:
        common["stop"] = list(STOP_STRINGS)
    greedy = SamplingParams(temperature=0, max_tokens=max_new_tokens, **common)
    if n_samples <= 0:               # greedy-only run
        return greedy, None
    skw = dict(n=n_samples, temperature=temperature, max_tokens=max_new_tokens, **common)
    if top_p is not None:
        skw["top_p"] = top_p
    if top_k is not None:
        skw["top_k"] = top_k
    return greedy, SamplingParams(**skw)


def vllm_generate_chunk(llm, prompts, greedy_params, sampling_params):
    g = llm.generate(prompts, greedy_params)
    greedy = [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in g]
    if sampling_params is None:      # greedy-only run
        return greedy, None
    s = llm.generate(prompts, sampling_params)
    sampling = [[(c.text, len(c.token_ids)) for c in o.outputs] for o in s]
    return greedy, sampling


# ---------------------------------------------------------------------------
# HF dataset repo I/O (durable record)
# ---------------------------------------------------------------------------

def upload_artifact(local_path, repo_id=HF_RESULTS_REPO, quiet=False) -> bool:
    """Upload a file to the HF dataset repo at outputs/<filename>. Best-effort."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("WARNING: huggingface_hub not installed — cannot upload. RSYNC eval_results/ before teardown.")
        return False
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    path_in_repo = f"outputs/{Path(local_path).name}"
    try:
        HfApi(token=token).upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                                       repo_id=repo_id, repo_type="dataset")
        if not quiet:
            print(f"Uploaded → https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}")
        return True
    except Exception as e:
        print(f"WARNING: HF upload of {local_path} FAILED ({type(e).__name__}: {e}). "
              f"Local copy intact — RSYNC eval_results/ before teardown.")
        return False


def try_pull_from_hf(local_path, repo_id=HF_RESULTS_REPO) -> bool:
    """If resuming on a fresh pod, pull a prior partial JSONL from HF. Best-effort."""
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                 filename=f"outputs/{Path(local_path).name}")
        shutil.copy(cached, local_path)
        print(f"Resume: pulled prior partial from HF → {local_path}")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Methodology defaults — per format
# ---------------------------------------------------------------------------

FORMAT_DEFAULTS = {
    "completion": dict(max_new_tokens=2048, temperature=0.7, top_p=None, top_k=None,
                       repetition_penalty=1.0),
    "chat":       dict(max_new_tokens=8192, temperature=0.6, top_p=0.95, top_k=20,
                       repetition_penalty=1.05),
}
N_SAMPLES = 8


# ---------------------------------------------------------------------------
# Model tag / checkpoint resolution
# ---------------------------------------------------------------------------

def _list_checkpoint_commits(model, revision=None):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    commits = list(HfApi(token=token).list_repo_commits(model, repo_type="model",
                                                        revision=revision or "main"))
    step_pattern = re.compile(r"\bstep\s+(\d+)[,\s].*checkpoint", re.IGNORECASE)
    out = []
    for c in commits:
        m = step_pattern.search(c.title or "")
        if m:
            out.append((int(m.group(1)), c.commit_id))
    return sorted(out)


def _resolve_latest_step(model, revision=None) -> int:
    if Path(model).exists():
        raise RuntimeError("--latest requires a HF Hub model ID, not a local path.")
    commits = _list_checkpoint_commits(model, revision)
    if not commits:
        raise RuntimeError(f"No checkpoint commits found in {model} history.")
    return max(step for step, _ in commits)


def _resolve_step_revision(model, step) -> str:
    if Path(model).exists():
        raise RuntimeError("Auto-revision requires a HF Hub model ID, not a local path.")
    for s, commit_id in _list_checkpoint_commits(model):
        if s == step:
            return commit_id
    raise RuntimeError(f"Step {step} not found in {model} checkpoint commits.")


def _model_tag(model) -> str:
    name = model.split("/")[-1].lower()
    for k in ("grpo", "sft", "base"):
        if k in name:
            return k
    return re.sub(r"[^A-Za-z0-9._-]", "-", name) or "model"


def _vllm_version():
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except Exception:
        return None


def _provenance(args, backend):
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
        st = os.path.join(args.model, "model.safetensors")
        if os.path.isdir(args.model) and os.path.exists(st):
            h = hashlib.sha256()
            with open(st, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            weights_sha = h.hexdigest()
    except Exception:
        pass
    import transformers
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": " ".join([sys.executable.split("/")[-1], *sys.argv]),
        "weights_sha256": weights_sha,
        "backend": backend,
        "vllm_version": _vllm_version(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Per-sample JSONL + resume/aggregate
# ---------------------------------------------------------------------------

def _uid_of(ex, i) -> str:
    return ex.get("unique_id") or f"idx{i}"


def _write_sample_rows(jsonl_path, ex, i, pass_type, samples, max_new_tokens, fmt, model):
    uid = _uid_of(ex, i)
    with open(jsonl_path, "a") as f:
        for sample_idx, (text, n_tokens) in enumerate(samples):
            pred = extract_boxed(text)
            row = {"unique_id": uid, "pass_type": pass_type, "sample_idx": sample_idx,
                   "problem": ex["problem"], "expected": ex["answer"], "level": ex["level"],
                   "subject": ex["subject"], "response": text, "predicted": pred,
                   "n_tokens": n_tokens, "max_new_tokens": max_new_tokens,
                   "format": fmt, "model": model}
            if MATH_VERIFY_AVAILABLE:
                row["correct"] = score_correct(pred, ex["answer"])
            f.write(json.dumps(row) + "\n")


def _load_done(jsonl_path) -> set:
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["unique_id"], r["pass_type"], r["sample_idx"]))
    return done


def _aggregate_results(jsonl_path, ds_list) -> list:
    order, meta = {}, {}
    for i, ex in enumerate(ds_list):
        uid = _uid_of(ex, i)
        order[uid] = i
        meta[uid] = {"problem": ex["problem"], "expected": ex["answer"],
                     "level": ex["level"], "subject": ex["subject"]}
    latest = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            latest[(r["unique_id"], r["pass_type"], r["sample_idx"])] = {
                "response": r["response"], "predicted": r["predicted"], "n_tokens": r.get("n_tokens")}
    grouped = defaultdict(lambda: {"pass1": [], "pass8": []})
    for (uid, ptype, sidx), s in latest.items():
        grouped[uid]["pass1" if ptype == "greedy" else "pass8"].append((sidx, s))
    results = []
    for uid in sorted(grouped, key=lambda u: order.get(u, 1 << 30)):
        p1 = [s for _, s in sorted(grouped[uid]["pass1"])]
        p8 = [s for _, s in sorted(grouped[uid]["pass8"])]
        results.append({**meta.get(uid, {"problem": None, "expected": None,
                                         "level": None, "subject": None}),
                        "pass1": p1, "pass8": p8})
    return results


def _live_summary(results):
    """math-verify pass@1/pass@k + by-level (live only; rescore_math500 is authoritative)."""
    if not MATH_VERIFY_AVAILABLE:
        return None
    lvl_g, lvl_k, lvl_n = defaultdict(int), defaultdict(int), defaultdict(int)
    g = k = 0
    for r in results:
        exp, lv = r["expected"], r["level"]
        gc = any(score_correct(s["predicted"], exp) for s in r["pass1"])
        kc = any(score_correct(s["predicted"], exp) for s in r["pass8"])
        g += gc; k += kc; lvl_n[lv] += 1; lvl_g[lv] += gc; lvl_k[lv] += kc
    n = len(results) or 1
    return {
        "greedy_pass1": round(g / n, 4), "pass_at_k": round(k / n, 4), "n_problems": len(results),
        "by_level": {str(lv): {"greedy_pass1": round(lvl_g[lv] / lvl_n[lv], 4),
                               "pass_at_k": round(lvl_k[lv] / lvl_n[lv], 4), "n": lvl_n[lv]}
                     for lv in sorted(lvl_n)},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="MATH-500 pass@1 + pass@8 eval. One entrypoint; --format selects "
                    "completion (base/GRPO) or chat (SFT). Deterministic filenames, auto-resume, "
                    "HF dataset = durable record. Methodology defaults are per-format.")
    p.add_argument("--model", default="heyalexchoi/qwen3-1.7b-math-grpo", help="HF Hub ID or local path.")
    p.add_argument("--format", choices=["completion", "chat"], default="completion")
    p.add_argument("--checkpoint_step", type=int, default=None, help="Names the output file.")
    p.add_argument("--latest", action="store_true", help="Use the latest HF checkpoint step.")
    p.add_argument("--revision", default=None, help="HF revision; auto-resolved from --checkpoint_step.")
    p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                   help="Default auto = vLLM if importable, else HF.")
    p.add_argument("--problem_batch_size", type=int, default=8, help="HF problems per generate() call.")
    p.add_argument("--upload_every", type=int, default=25,
                   help="Upload the JSONL to HF every N problems (resilient backup). 0 = only at end.")
    p.add_argument("--output_name", default=None,
                   help="Escape hatch: override the deterministic <base> filename.")
    p.add_argument("--fresh", action="store_true", help="Ignore any existing/HF partial; start clean.")
    p.add_argument("--stats_only", action="store_true")
    # Methodology overrides (default None → FORMAT_DEFAULTS[--format])
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--n_samples", type=int, default=N_SAMPLES,
                   help="pass@k sampling count; 0 = greedy-only (pass@1, no sampling pass — much faster).")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    # Diagnostic-only (canary / end-token A/B — not part of the normal path)
    p.add_argument("--max_samples", type=int, default=None, help="[diagnostic] limit problems.")
    p.add_argument("--unique_id", nargs="*", default=None,
                   help="[diagnostic] restrict to dataset ids containing these substrings (canary).")
    p.add_argument("--no_chat_stop_ids", action="store_true",
                   help="[diagnostic] chat vLLM: omit stop_token_ids (end-token A/B test).")
    args = p.parse_args()

    if args.latest and args.checkpoint_step:
        p.error("--latest and --checkpoint_step are mutually exclusive.")

    fd = FORMAT_DEFAULTS[args.format]
    for key in ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
        if getattr(args, key) is None:
            setattr(args, key, fd[key])

    is_local = Path(args.model).exists()
    if args.latest:
        args.checkpoint_step = _resolve_latest_step(args.model, args.revision)
        print(f"Latest checkpoint step: {args.checkpoint_step}")
    if args.checkpoint_step and not args.revision and not is_local:
        args.revision = _resolve_step_revision(args.model, args.checkpoint_step)
        print(f"Resolved step {args.checkpoint_step} → {args.revision}")

    # Backend
    use_vllm = False
    if args.backend in ("auto", "vllm"):
        try:
            import vllm  # noqa: F401
            use_vllm = True
        except ImportError:
            if args.backend == "vllm":
                print("ERROR: --backend vllm requested but vllm not installed.", flush=True)
                sys.exit(1)
    backend = "vllm" if use_vllm else "hf"

    # Deterministic filenames (no timestamp → re-running resumes). Diagnostics get suffixes
    # so they never collide with a real run's file.
    if args.output_name:
        base = args.output_name
    else:
        step_part = f"step{args.checkpoint_step}" if args.checkpoint_step is not None else "local"
        base = f"{_model_tag(args.model)}_{step_part}_{args.format}_max{args.max_new_tokens}_math500"
        if args.unique_id:
            base += "_subset"
        if args.no_chat_stop_ids:
            base += "_nostopids"
    results_path = f"eval_results/{base}_results.json"
    jsonl_path = f"eval_results/{base}_samples.jsonl"
    summary_path = f"eval_results/{base}_summary.json"
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    # Diagnostic runs (canary / A-B) stay local; real runs are the HF record.
    is_diagnostic = bool(args.unique_id or args.no_chat_stop_ids)
    auto_upload = not is_diagnostic

    print(f"base={base}\nbackend={backend} format={args.format} "
          f"max_new_tokens={args.max_new_tokens} temp={args.temperature} "
          f"n_samples={args.n_samples} rep_penalty={args.repetition_penalty} "
          f"upload={'HF (auto)' if auto_upload else 'local-only (diagnostic)'}")

    # Dataset + stats
    print("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.unique_id:
        keep = [i for i, ex in enumerate(ds)
                if any(u in (ex.get("unique_id") or "") for u in args.unique_id)]
        if not keep:
            print(f"ERROR: no problems matched --unique_id {args.unique_id}", flush=True)
            sys.exit(1)
        ds = ds.select(keep)
        print(f"--unique_id filter → {[ds[i].get('unique_id') for i in range(len(ds))]}")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    level_counts, subject_counts = defaultdict(int), defaultdict(int)
    for ex in ds:
        level_counts[ex["level"]] += 1
        subject_counts[ex["subject"]] += 1
    print(f"\n=== MATH-500 ({len(ds)} problems) ===")
    for lvl in sorted(level_counts):
        print(f"  Level {lvl}: {level_counts[lvl]:4d}")
    if args.stats_only:
        return

    ds_list = list(ds)
    revision_str = args.revision or "main"

    tokenizer = load_tokenizer_safe(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = get_stop_ids(tokenizer) if args.format == "chat" else None
    if args.format == "chat":
        if args.no_chat_stop_ids:
            stop_ids = None
            print("WARNING: --no_chat_stop_ids — vLLM will fall back to config eos (end-token A/B).")
        else:
            print(f"Chat stop ids: {stop_ids} "
                  f"(eos={tokenizer.eos_token!r}, <|im_end|>={tokenizer.convert_tokens_to_ids('<|im_end|>')})")

    all_prompts = build_prompts(ds_list, args.format, tokenizer)
    print(f"Sample prompt (first 200 chars): {all_prompts[0][:200]!r}")

    # Resume: prefer local JSONL; on a fresh pod pull the prior partial from HF.
    if args.fresh and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    if not args.fresh and not os.path.exists(jsonl_path) and auto_upload:
        try_pull_from_hf(jsonl_path)
    done = _load_done(jsonl_path)

    def _fully_done(i, ex):
        uid = _uid_of(ex, i)
        return (uid, "greedy", 0) in done and all((uid, "sample", s) in done
                                                   for s in range(args.n_samples))
    pending = [i for i, ex in enumerate(ds_list) if not _fully_done(i, ex)]
    if done:
        print(f"Resume: {len(ds_list) - len(pending)}/{len(ds_list)} done, {len(pending)} pending.")

    chunk = args.upload_every if args.upload_every and args.upload_every > 0 else max(1, len(pending))

    if use_vllm:
        llm = build_vllm(args.model, args.revision, args.max_new_tokens) if pending else None
        gp, sp = (vllm_sampling_params(args.format, stop_ids, args.max_new_tokens, args.n_samples,
                                       args.temperature, args.top_p, args.top_k,
                                       args.repetition_penalty) if pending else (None, None))
        for cs in range(0, len(pending), chunk):
            idxs = pending[cs:cs + chunk]
            greedy, sampling = vllm_generate_chunk(llm, [all_prompts[i] for i in idxs], gp, sp)
            for kk, i in enumerate(idxs):
                ex = ds_list[i]
                _write_sample_rows(jsonl_path, ex, i, "greedy", [greedy[kk]],
                                   args.max_new_tokens, args.format, args.model)
                if sampling is not None:
                    _write_sample_rows(jsonl_path, ex, i, "sample", sampling[kk],
                                       args.max_new_tokens, args.format, args.model)
            if auto_upload and args.upload_every:
                upload_artifact(jsonl_path, quiet=True)
            print(f"  chunk done: {min(cs + chunk, len(pending))}/{len(pending)} pending problems")
    else:
        print(f"Backend: HF | loading {args.model} (revision={revision_str})")
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                     dtype=torch.bfloat16, device_map="auto")
        model.eval()
        B = args.problem_batch_size
        for cs in tqdm(range(0, len(pending), chunk), desc="chunks"):
            chunk_idxs = pending[cs:cs + chunk]
            for bs in range(0, len(chunk_idxs), B):
                idxs = chunk_idxs[bs:bs + B]
                prompts = [all_prompts[i] for i in idxs]
                gkw = dict(fmt=args.format, stop_ids=stop_ids, top_p=args.top_p, top_k=args.top_k,
                           repetition_penalty=args.repetition_penalty)
                gres = generate_batch(model, tokenizer, prompts, 1, 0.0, args.max_new_tokens, **gkw)
                sres = (generate_batch(model, tokenizer, prompts, args.n_samples, args.temperature,
                                       args.max_new_tokens, **gkw)
                        if args.n_samples > 0 else None)
                for j, i in enumerate(idxs):
                    ex = ds_list[i]
                    _write_sample_rows(jsonl_path, ex, i, "greedy", gres[j],
                                       args.max_new_tokens, args.format, args.model)
                    if sres is not None:
                        _write_sample_rows(jsonl_path, ex, i, "sample", sres[j],
                                           args.max_new_tokens, args.format, args.model)
            if auto_upload and args.upload_every:
                upload_artifact(jsonl_path, quiet=True)

    # Aggregate (resume-safe) + write artifacts
    results = _aggregate_results(jsonl_path, ds_list)
    prov = _provenance(args, backend)
    methodology = {"max_new_tokens": args.max_new_tokens, "n_samples_pass8": args.n_samples,
                   "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                   "repetition_penalty": args.repetition_penalty}
    dataset_stats = {"total": len(ds), "by_level": {str(k): v for k, v in sorted(level_counts.items())},
                     "by_subject": dict(sorted(subject_counts.items()))}
    header = {"model": args.model, "format": args.format, "backend": backend,
              "hf_revision": revision_str, "checkpoint_step": args.checkpoint_step,
              "provenance": prov, **methodology, "dataset_stats": dataset_stats}

    with open(results_path, "w") as f:
        json.dump({**header, "results": results}, f, indent=2)

    live = _live_summary(results)
    with open(summary_path, "w") as f:
        json.dump({**header, "live_metrics_math_verify": live,
                   "note": "Metrics are a LIVE convenience scoring; rescore_math500.py is authoritative. "
                           "Full generations: see the *_results.json / *_samples.jsonl on HF."}, f, indent=2)

    print(f"\nResults: {results_path}\nSamples: {jsonl_path}\nSummary (git-tracked): {summary_path}")
    if live:
        pk = (f"pass@{args.n_samples} {live['pass_at_k']:.2%}" if args.n_samples > 0
              else "pass@k N/A (greedy-only run)")
        print(f"[live math-verify] greedy pass@1 {live['greedy_pass1']:.2%} | "
              f"{pk} (rescore_math500.py for the record)")

    if auto_upload:
        ok1 = upload_artifact(results_path)
        ok2 = upload_artifact(jsonl_path)
        if not (ok1 and ok2):
            print("ONE OR MORE UPLOADS FAILED — rsync eval_results/ before tearing down the pod.")
    else:
        print("Diagnostic run: local-only (not uploaded). rsync eval_results/ if you need to keep it.")


if __name__ == "__main__":
    main()
