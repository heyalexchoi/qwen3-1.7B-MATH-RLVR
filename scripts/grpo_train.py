#!/usr/bin/env python3
"""
GRPO Training on MATH dataset using TRL GRPOTrainer.

Trains from an SFT checkpoint using GRPO (group relative policy optimization)
with a math-verify correctness reward. Targets MATH-style \\boxed{} answers.

Input:  data/math_train.jsonl
        Fields: problem, solution (full solution with \\boxed{answer})
Output: /workspace/qwen3-math-rlvr/outputs/grpo_checkpoint/

Hardware: A100 80GB required — Qwen3's 152k vocab + 4096 completion length
          will OOM on anything smaller.
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

os.environ["WANDB_PROJECT"] = "qwen3-math-rlvr"

# ---------------------------------------------------------------------------
# math-verify (primary reward evaluator) — fallback to regex if not installed
# ---------------------------------------------------------------------------

try:
    from math_verify import parse as mv_parse, verify as mv_verify_fn

    def _mv_correct(predicted: str, expected: str) -> bool:
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
        return _regex_match(predicted, expected)

    MATH_VERIFY_AVAILABLE = True

except ImportError:
    MATH_VERIFY_AVAILABLE = False

    def _mv_correct(predicted: str, expected: str) -> bool:  # type: ignore[misc]
        return _regex_match(predicted, expected)


def _normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\.0+$", "", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def _regex_match(predicted: str, expected: str) -> bool:
    p = _normalize(predicted)
    e = _normalize(expected)
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 1e-6
    except (ValueError, TypeError):
        pass
    return p.lower() == e.lower()


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
# Reward functions
# ---------------------------------------------------------------------------

def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """
    Binary correctness reward: 1.0 if predicted answer matches expected, 0.0 otherwise.
    Uses math-verify (ANTLR4/SymPy) with regex fallback.
    `answer` column is the pre-extracted ground-truth answer (no \\boxed{}).
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        predicted = extract_boxed(completion)
        is_correct = _mv_correct(predicted, expected)
        rewards.append(1.0 if is_correct else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Small format reward for producing \\boxed{} output.
    Encourages the model to emit a structured answer even when wrong.
    """
    rewards = []
    for completion in completions:
        has_boxed = bool(re.search(r"\\boxed\{", completion))
        rewards.append(0.1 if has_boxed else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_answer_from_solution(solution: str) -> str:
    """Extract the answer from a full solution string containing \\boxed{...}."""
    return extract_boxed(solution)


def load_math_for_grpo(data_path: str) -> Dataset:
    """
    Load MATH training data formatted for GRPO.

    Dataset format expected:
      {"problem": "...", "solution": "... \\boxed{answer}"}

    The `answer` column holds the pre-extracted ground-truth answer
    (without \\boxed{}) for use in the reward function.
    The `prompt` column is what the model receives as input.
    """
    examples = []
    skipped = 0
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            problem = data.get("problem", "").strip()
            solution = data.get("solution", "").strip()
            if not problem or not solution:
                skipped += 1
                continue
            answer = extract_answer_from_solution(solution)
            if not answer:
                skipped += 1
                continue
            examples.append({
                "prompt": [{"role": "user", "content": problem}],
                "answer": answer,
            })

    logger.info(f"Loaded {len(examples)} examples ({skipped} skipped — missing problem/solution/answer)")
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Checkpoint helpers (same pattern as 03_sft_train.py)
# ---------------------------------------------------------------------------

def find_latest_checkpoint(output_dir: str) -> str | None:
    """Scan output_dir for checkpoint-* dirs, return path to highest step number."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    model_file = latest / "model.safetensors"
    if not model_file.exists() or model_file.stat().st_size == 0:
        logger.warning(f"Checkpoint {latest} missing or empty model.safetensors — skipping")
        return None
    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training on MATH dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="/workspace/qwen3-math-rlvr/outputs/sft_checkpoint",
        help="Path to SFT checkpoint or HF model ID to start from",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace/qwen3-math-rlvr/data/math_train.jsonl",
        help="Path to MATH training data JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/qwen3-math-rlvr/outputs/grpo_checkpoint",
        help="Output dir — use absolute path to guarantee writes land on the volume",
    )
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir, or 'auto' to find latest in output_dir",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push final model to HuggingFace Hub after training",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default="heyalexchoi/qwen3-1.7b-math-grpo",
        help="HuggingFace Hub repo ID for push_to_hub",
    )
    args = parser.parse_args()

    # Set up file logging — captures logging.* calls AND raw stderr (HF Trainer, CUDA warnings)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"grpo_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_fh = open(str(log_file), "a", buffering=1)  # line-buffered
    sys.stderr = _log_fh  # redirect raw stderr into the log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.StreamHandler(_log_fh),  # logging.* calls also go to file
        ],
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    logger.info(
        f"math-verify: {'available' if MATH_VERIFY_AVAILABLE else 'NOT installed — using regex fallback'}"
    )

    config = load_config(args.config)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Resolve --resume_from_checkpoint
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "auto":
        found = find_latest_checkpoint(args.output)
        if found:
            resume_checkpoint = True  # HF Trainer finds latest automatically
            logger.info(f"Resume mode: auto — will resume from latest checkpoint in {args.output}")
        else:
            resume_checkpoint = None
            logger.info("Resume mode: auto — no valid checkpoint found, starting fresh")
    elif resume_checkpoint is not None:
        cp_path = Path(resume_checkpoint)
        if not cp_path.exists():
            logger.warning(f"Checkpoint path {resume_checkpoint} does not exist — starting fresh")
            resume_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # Tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
    )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Data
    logger.info(f"Loading MATH training data: {args.data}")
    dataset = load_math_for_grpo(args.data)
    logger.info(f"Dataset size: {len(dataset)} problems")

    # GRPOConfig
    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=config.get("num_train_epochs", 1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 5e-7),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 0.1),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 1),
        seed=config.get("seed", 42),
        # GRPO-specific
        num_generations=config.get("num_generations", 8),
        max_completion_length=config.get("max_completion_length", 4096),
        max_prompt_length=config.get("max_prompt_length", 1024),
        temperature=config.get("temperature", 0.9),
        # Reporting
        report_to="wandb",
        run_name="grpo-qwen3-1.7b-math",
        # HF Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo_id if args.push_to_hub else None,
        hub_strategy="checkpoint" if args.push_to_hub else "end",
    )

    logger.info("GRPO config summary:")
    logger.info(f"  num_generations:        {grpo_config.num_generations}")
    logger.info(f"  max_completion_length:  {grpo_config.max_completion_length}")
    logger.info(f"  temperature:            {grpo_config.temperature}")
    logger.info(f"  per_device_batch_size:  {grpo_config.per_device_train_batch_size}")
    logger.info(f"  gradient_accumulation:  {grpo_config.gradient_accumulation_steps}")
    logger.info(f"  effective batch size:   {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps}")
    logger.info(f"  learning_rate:          {grpo_config.learning_rate}")
    logger.info(f"  num_train_epochs:       {grpo_config.num_train_epochs}")
    logger.info(f"  push_to_hub:            {args.push_to_hub}")
    if args.push_to_hub:
        logger.info(f"  hub_repo_id:            {args.hub_repo_id}")

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward],
    )

    logger.info("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    if args.push_to_hub:
        logger.info(f"Final push to HuggingFace Hub: {args.hub_repo_id}")
        trainer.push_to_hub()
        logger.info(f"Done pushing to HuggingFace Hub: {args.hub_repo_id}")

    logger.info("Done.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
