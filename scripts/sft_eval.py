#!/usr/bin/env python3
"""Evaluate SFT checkpoint on MATH-500: pass@1 (greedy) and pass@8 (sampling).

Inference backends (auto-detected, or set with --backend):
  vllm  — primary path; continuous batching, fast, recommended
  hf    — fallback; HF model.generate(); uses flash_attention_2 if available

Dataset: HuggingFaceH4/MATH-500
  Fields: problem, solution, subject, level (int 1-5), answer (pre-extracted)

Outputs:
  outputs/sft_eval_results.jsonl          — per-example results (sampling)
  outputs/sft_eval_results_summary.json   — summary (sampling)
  outputs/sft_eval_greedy_results.jsonl   — per-example results (greedy)
  outputs/sft_eval_greedy_summary.json    — summary (greedy)
"""

import argparse
import json
import logging
import os

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import time

os.environ["WANDB_DISABLED"] = "true"

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

try:
    import vllm  # noqa: F401
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ---------------------------------------------------------------------------
# math-verify — primary evaluator, regex fallback if not installed
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as mv_parse, verify as mv_verify_fn

    def answers_match(predicted: str, expected: str) -> bool:
        if not predicted or not expected:
            return False
        try:
            gold = mv_parse(f"${expected}$")
            ans = mv_parse(f"${predicted}$")
            if gold and ans:
                return bool(mv_verify_fn(gold, ans))
        except Exception:
            pass
        return _normalize(predicted) == _normalize(expected)

    MATH_VERIFY_AVAILABLE = True

except ImportError:
    MATH_VERIFY_AVAILABLE = False

    def answers_match(predicted: str, expected: str) -> bool:  # type: ignore[misc]
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
    """Format using the model's chat template to match SFT training format."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Stop token helper
# ---------------------------------------------------------------------------

def get_stop_ids(tokenizer) -> list[int]:
    """Return stop token IDs: eos_token + <|im_end|> (Qwen3 chat terminator)."""
    _im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _base_eos = tokenizer.eos_token_id
    _base_list = [_base_eos] if isinstance(_base_eos, int) else list(_base_eos)
    return list(set(_base_list + [_im_end_id]))


# ---------------------------------------------------------------------------
# vLLM inference — AsyncLLMEngine with per-problem incremental writes
# ---------------------------------------------------------------------------

def run_inference_vllm(args, pending: list[dict], tokenizer, logger, output_jsonl: str) -> None:
    """
    Run vLLM inference over pending problems, writing results per-problem.

    Engine is initialized once; llm.generate() is called per problem so results
    are written to JSONL immediately after each completes — full resume/progress
    support with no waiting for the whole batch.
    """
    from vllm import LLM, SamplingParams as VLLMSamplingParams

    max_model_len = args.max_new_tokens + 2048

    logger.info(
        f"Loading vLLM engine: model={args.model}, dtype=bfloat16, "
        f"max_model_len={max_model_len}, gpu_memory_utilization=0.90"
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    if args.temperature == 0.0:
        sampling_params = VLLMSamplingParams(
            n=args.n_samples,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=get_stop_ids(tokenizer),
        )
    else:
        sampling_params = VLLMSamplingParams(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=get_stop_ids(tokenizer),
        )

    total = len(pending)
    total_tokens = 0
    t0 = time.time()
    logger.info(f"Starting per-problem inference: {total} problems × {args.n_samples} samples")

    with open(output_jsonl, "a") as jsonl_f:
        for i, ex in enumerate(pending):
            prompt = create_prompt(ex["problem"], tokenizer)
            outputs = llm.generate([prompt], sampling_params)
            vllm_out = outputs[0]

            n_correct = 0
            prob_tokens = 0
            for sample_idx, comp in enumerate(vllm_out.outputs):
                text = comp.text
                n_tokens = len(comp.token_ids)
                pred = extract_boxed(text)
                correct = answers_match(pred, ex["answer"])
                n_correct += correct
                prob_tokens += n_tokens
                total_tokens += n_tokens
                jsonl_f.write(json.dumps({
                    "unique_id": ex["unique_id"],
                    "sample_idx": sample_idx,
                    "problem": ex["problem"],
                    "expected": ex["answer"],
                    "level": ex["level"],
                    "subject": ex["subject"],
                    "response": text,
                    "predicted": pred,
                    "correct": correct,
                    "n_tokens": n_tokens,
                    "max_new_tokens": args.max_new_tokens,
                }) + "\n")
            jsonl_f.flush()

            elapsed = time.time() - t0
            tok_per_s = total_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{i + 1}/{total}] {ex['unique_id']}: "
                f"{n_correct}/{args.n_samples} correct, avg {prob_tokens // args.n_samples} tok, "
                f"{tok_per_s:.0f} tok/s overall"
            )

    elapsed = time.time() - t0
    logger.info(
        f"vLLM done: {total} problems in {elapsed:.1f}s "
        f"({elapsed / total:.1f}s/problem, {total_tokens / elapsed:.0f} tok/s)"
    )


# ---------------------------------------------------------------------------
# HF inference (fallback)
# ---------------------------------------------------------------------------

def load_hf_model(args, logger):
    """Load model and tokenizer via HuggingFace. Uses flash_attention_2 if available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"
    logger.info(f"Loading HF model: {args.model}, attn_implementation={attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_samples_hf(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
    unique_id: str = "",
    eos_token_ids: list[int] | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[tuple[str, int]]:
    """Generate n completions for a single prompt. Returns list of (text, n_tokens)."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    stop_ids = eos_token_ids if eos_token_ids else tokenizer.eos_token_id

    input_ids = inputs.input_ids.repeat(n, 1)
    attention_mask = inputs.attention_mask.repeat(n, 1)

    if temperature == 0.0:
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    logger.info(f"[{unique_id}] generating {n} samples (max_new_tokens={max_new_tokens})...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    elapsed = time.time() - t0

    results = []
    for i in range(n):
        token_ids = out[i][input_len:]
        n_tokens = int((token_ids != tokenizer.pad_token_id).sum())
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        results.append((text, n_tokens))

    total_tokens = sum(r[1] for r in results)
    logger.info(
        f"[{unique_id}] done: {total_tokens} tokens in {elapsed:.1f}s "
        f"({total_tokens / elapsed:.0f} tok/s), avg {total_tokens / n:.0f} tok/sample"
    )
    return results


def run_inference_hf_sampling(args, ds, done_samples, jsonl_f, logger):
    """HF per-problem sampling loop (original approach, n samples per problem)."""
    model, tokenizer = load_hf_model(args, logger)
    stop_token_ids = get_stop_ids(tokenizer)

    assert tokenizer.chat_template is not None, "Tokenizer has no chat template."
    sample_prompt = create_prompt("What is 2+2?", tokenizer)
    logger.info(f"Sample prompt (first 200 chars): {repr(sample_prompt[:200])}")

    from tqdm import tqdm
    for ex in tqdm(ds):
        unique_id = ex["unique_id"]
        samples_needed = [i for i in range(args.n_samples) if (unique_id, i) not in done_samples]
        if not samples_needed:
            continue

        prompt = create_prompt(ex["problem"], tokenizer)
        sample_results = generate_samples_hf(
            model, tokenizer, prompt,
            n=args.n_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            unique_id=unique_id,
            eos_token_ids=stop_token_ids,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        for sample_idx, (resp, n_tokens) in enumerate(sample_results):
            if (unique_id, sample_idx) in done_samples:
                continue
            pred = extract_boxed(resp)
            correct = answers_match(pred, ex["answer"])
            row = {
                "unique_id": unique_id,
                "sample_idx": sample_idx,
                "problem": ex["problem"],
                "expected": ex["answer"],
                "level": ex["level"],
                "subject": ex["subject"],
                "response": resp,
                "predicted": pred,
                "correct": correct,
                "n_tokens": n_tokens,
                "max_new_tokens": args.max_new_tokens,
            }
            logger.info(f"[{unique_id}] sample {sample_idx}: {n_tokens} tokens, correct={correct}")
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()


def run_inference_hf_greedy_batched(args, ds, done_samples, jsonl_f, logger):
    """HF batched greedy loop (greedy_batch_size problems per generate() call)."""
    import torch
    model, tokenizer = load_hf_model(args, logger)
    tokenizer.padding_side = "left"
    stop_token_ids = get_stop_ids(tokenizer)

    logger.info(f"Batched greedy mode: batch_size={args.greedy_batch_size}, padding_side=left")

    assert tokenizer.chat_template is not None, "Tokenizer has no chat template."

    from tqdm import tqdm
    problems_list = list(ds)
    n_total = len(problems_list)
    n_batches = (n_total + args.greedy_batch_size - 1) // args.greedy_batch_size

    pbar = tqdm(total=n_total, desc="greedy batched (hf)")
    for batch_idx in range(n_batches):
        batch_start = batch_idx * args.greedy_batch_size
        batch_end = min(batch_start + args.greedy_batch_size, n_total)
        batch_exs = problems_list[batch_start:batch_end]
        pending = [ex for ex in batch_exs if (ex["unique_id"], 0) not in done_samples]

        if not pending:
            pbar.update(len(batch_exs))
            continue

        prompts = [create_prompt(ex["problem"], tokenizer) for ex in pending]
        batch_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False,
        ).to(model.device)
        padded_input_len = batch_inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t0

        total_tokens = 0
        for i, ex in enumerate(pending):
            token_ids = out[i][padded_input_len:]
            n_tokens = int((token_ids != tokenizer.pad_token_id).sum())
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            total_tokens += n_tokens
            pred = extract_boxed(text)
            correct = answers_match(pred, ex["answer"])
            row = {
                "unique_id": ex["unique_id"],
                "sample_idx": 0,
                "problem": ex["problem"],
                "expected": ex["answer"],
                "level": ex["level"],
                "subject": ex["subject"],
                "response": text,
                "predicted": pred,
                "correct": correct,
                "n_tokens": n_tokens,
                "max_new_tokens": args.max_new_tokens,
            }
            logger.info(f"[{ex['unique_id']}] sample 0: {n_tokens} tokens, correct={correct}")
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()

        tok_per_s = total_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Batch {batch_idx + 1}/{n_batches}: {total_tokens} tokens in {elapsed:.1f}s "
            f"({tok_per_s:.0f} tok/s)"
        )
        pbar.update(len(batch_exs))
    pbar.close()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_pass_at_k(results: list[dict], sample_key: str) -> dict:
    """Compute pass@k (any-correct over samples) with per-level and per-subject breakdown."""
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

    per_level = {
        str(lvl): {
            "correct": level_correct[lvl],
            "total": level_total[lvl],
            "pass_at_k": round(level_correct[lvl] / level_total[lvl], 4),
        }
        for lvl in sorted(level_total.keys())
    }
    per_subject = {
        subj: {
            "correct": subject_correct[subj],
            "total": subject_total[subj],
            "pass_at_k": round(subject_correct[subj] / subject_total[subj], 4),
        }
        for subj in sorted(subject_total.keys())
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
    parser = argparse.ArgumentParser(description="MATH-500 eval: pass@8 (sampling) or pass@1 (greedy)")
    parser.add_argument("--model", type=str, default="/workspace/qwen3-math-rlvr/outputs/sft_checkpoint")
    parser.add_argument("--output", type=str, default="/workspace/qwen3-math-rlvr/outputs/sft_eval_results.json",
                        help="Legacy combined output (backward compat)")
    parser.add_argument("--output_jsonl", type=str, default="/workspace/qwen3-math-rlvr/outputs/sft_eval_results.jsonl")
    parser.add_argument("--output_summary", type=str, default="/workspace/qwen3-math-rlvr/outputs/sft_eval_results_summary.json")
    parser.add_argument("--data", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit problems (debug)")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                        help="Repetition penalty for vLLM (1.0=none). 1.05 prevents degenerate loops in SFT model.")
    parser.add_argument("--stats_only", action="store_true")
    parser.add_argument(
        "--greedy", action="store_true",
        help="Greedy pass@1 mode: n=1, temp=0.0. Output → sft_eval_greedy_results.jsonl",
    )
    parser.add_argument("--greedy_batch_size", type=int, default=8,
                        help="Batch size for HF greedy fallback (ignored for vLLM)")
    parser.add_argument(
        "--backend", choices=["vllm", "hf", "auto"], default="auto",
        help="Inference backend. auto = vllm if available, else hf",
    )
    args = parser.parse_args()

    # --greedy: override sampling defaults
    if args.greedy:
        args.n_samples = 1
        args.temperature = 0.0
        default_jsonl = "/workspace/qwen3-math-rlvr/outputs/sft_eval_results.jsonl"
        default_summary = "/workspace/qwen3-math-rlvr/outputs/sft_eval_results_summary.json"
        if args.output_jsonl == default_jsonl:
            args.output_jsonl = "/workspace/qwen3-math-rlvr/outputs/sft_eval_greedy_results.jsonl"
        if args.output_summary == default_summary:
            args.output_summary = "/workspace/qwen3-math-rlvr/outputs/sft_eval_greedy_summary.json"

    # Resolve backend
    if args.backend == "auto":
        use_vllm = VLLM_AVAILABLE
    elif args.backend == "vllm":
        if not VLLM_AVAILABLE:
            print("ERROR: --backend vllm requested but vllm is not installed. pip install vllm", flush=True)
            sys.exit(1)
        use_vllm = True
    else:
        use_vllm = False

    # Logging setup
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"03a_sft_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(log_file))],
    )
    global logger
    logger = logging.getLogger(__name__)

    logger.info(f"Logging to {log_file}")
    logger.info(f"Backend: {'vllm' if use_vllm else 'hf'} "
                f"(vllm_available={VLLM_AVAILABLE}, flash_attn_available={FLASH_ATTN_AVAILABLE})")
    logger.info(f"math-verify: {'available' if MATH_VERIFY_AVAILABLE else 'NOT installed — using regex fallback'}")

    for path in [args.output, args.output_jsonl, args.output_summary]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    from datasets import load_dataset
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
    for lvl in sorted(level_counts.keys()):
        logger.info(f"  Level {lvl}: {level_counts[lvl]:4d} problems")
    for subj in sorted(subject_counts.keys()):
        logger.info(f"  {subj:<30s}: {subject_counts[subj]:4d} problems")

    if args.stats_only:
        return

    # -----------------------------------------------------------------------
    # Resume: load already-scored samples
    # -----------------------------------------------------------------------
    done_samples: set[tuple[str, int]] = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done_samples.add((r["unique_id"], r["sample_idx"]))
        if done_samples:
            done_problems = len({uid for uid, _ in done_samples})
            logger.info(f"Resuming: {len(done_samples)} samples ({done_problems} problems) already done")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    # For vLLM: load tokenizer separately just for prompt formatting
    if use_vllm:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        assert tokenizer.chat_template is not None, "Tokenizer has no chat template."
        stop_ids = get_stop_ids(tokenizer)
        logger.info(f"Stop tokens: {stop_ids} (eos='{tokenizer.eos_token}', <|im_end|>={tokenizer.convert_tokens_to_ids('<|im_end|>')})")
        sample_prompt = create_prompt("What is 2+2?", tokenizer)
        logger.info(f"Sample prompt (first 200 chars): {repr(sample_prompt[:200])}")

        # Build list of pending problems (any sample missing → include)
        pending = [
            ex for ex in ds
            if any((ex["unique_id"], i) not in done_samples for i in range(args.n_samples))
        ]
        logger.info(f"Pending: {len(pending)}/{len(ds)} problems")

        if pending:
            if args.greedy:
                logger.info(f"Running vLLM GREEDY eval: {len(pending)} problems, max_new_tokens={args.max_new_tokens}, rep_penalty={args.repetition_penalty}")
            else:
                logger.info(
                    f"Running vLLM SAMPLING eval: {len(pending)} × {args.n_samples} samples, "
                    f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
                    f"max_new_tokens={args.max_new_tokens}, rep_penalty={args.repetition_penalty}"
                )

            run_inference_vllm(args, pending, tokenizer, logger, args.output_jsonl)

    else:
        # HF fallback
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        stop_ids = get_stop_ids(tokenizer)
        logger.info(f"Stop tokens: {stop_ids}")

        if args.greedy:
            logger.info(f"Running HF GREEDY eval (batched, batch_size={args.greedy_batch_size})")
            with open(args.output_jsonl, "a") as jsonl_f:
                run_inference_hf_greedy_batched(args, ds, done_samples, jsonl_f, logger)
        else:
            logger.info(
                f"Running HF SAMPLING eval: n={args.n_samples}, temp={args.temperature}, "
                f"top_p={args.top_p}, top_k={args.top_k}, max_new_tokens={args.max_new_tokens}"
            )
            with open(args.output_jsonl, "a") as jsonl_f:
                run_inference_hf_sampling(args, ds, done_samples, jsonl_f, logger)

    # -----------------------------------------------------------------------
    # Score: aggregate JSONL → per-problem stats
    # -----------------------------------------------------------------------
    problems: dict[str, dict] = {}
    problem_samples: dict[str, list] = defaultdict(list)
    with open(args.output_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uid = r["unique_id"]
            problem_samples[uid].append(r)
            if uid not in problems:
                problems[uid] = {k: r[k] for k in ("problem", "expected", "level", "subject")}

    results = []
    pass1_estimates = []
    for uid, meta in problems.items():
        samples = problem_samples[uid]
        n_correct = sum(s["correct"] for s in samples)
        n = len(samples)
        pass1_est = n_correct / n if n > 0 else 0.0
        results.append({**meta, "samples": samples, "pass1_estimate": pass1_est})
        pass1_estimates.append(pass1_est)

    pass_k_stats = compute_pass_at_k(results, sample_key="samples")
    pass1_overall = sum(pass1_estimates) / len(pass1_estimates) if pass1_estimates else 0.0
    mode_label = "greedy" if args.greedy else f"sampling (n={args.n_samples})"

    logger.info("=== RESULTS ===")
    logger.info(f"Model:            {args.model}")
    logger.info(f"Mode:             {mode_label}")
    logger.info(f"Pass@1 (est c/n): {pass1_overall:.2%}")
    logger.info(f"Pass@{args.n_samples}:           {pass_k_stats['overall']:.2%}")
    logger.info("By level:")
    for lvl in sorted(pass_k_stats["per_level"].keys(), key=int):
        p = pass_k_stats["per_level"][lvl]["pass_at_k"]
        n = pass_k_stats["per_level"][lvl]["total"]
        logger.info(f"  Level {lvl} (n={n:3d}): {p:.2%}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    summary = {
        "model": args.model,
        "mode": mode_label,
        "backend": "vllm" if use_vllm else "hf",
        "evaluator": "math-verify" if MATH_VERIFY_AVAILABLE else "regex-fallback",
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "dataset": args.data,
        "dataset_stats": {
            "total": len(ds),
            "by_level": {str(k): v for k, v in sorted(level_counts.items())},
            "by_subject": dict(sorted(subject_counts.items())),
        },
        "pass1_unbiased": round(pass1_overall, 4),
        f"pass{args.n_samples}": pass_k_stats,
        "baseline_note": (
            "Apples-to-apples baseline: 24.55% pass@1 (c/n inferred, n=8, temp=0.6, math-verify). "
            "Greedy baseline: 35.8%. See outputs/baseline_math500_mv_rescored.json."
        ),
    }
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {args.output_summary}")

    combined = {**summary, "results": results}
    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined → {args.output}")
    logger.info("Done.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
