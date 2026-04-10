#!/usr/bin/env python3
"""
GRPO Training on MATH dataset using TRL GRPOTrainer.

Trains from Qwen3-1.7B-Base using GRPO (group relative policy optimization)
with a math-verify correctness reward. Uses few-shot raw text prompting
(same format as baseline eval) — base model is text-completion, not chat.

Input:  data/math_train.jsonl
        Fields: problem, solution (full solution with \\boxed{answer})
Output: /workspace/qwen3-math-rlvr/outputs/grpo_checkpoint/

Hardware: H100 SXM 80GB (pod gol7yudqrlfn48, $2.99/hr)
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from contextlib import nullcontext
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import profiling_context, unwrap_model_for_generation
from trl.trainer.utils import pad

from prompts import create_prompt, STOP_STRINGS

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

    Completions are truncated at STOP_STRINGS before answer extraction. TRL's
    GRPOTrainer doesn't pass a tokenizer to generate(), so stop_strings can't be
    set in GenerationConfig. Instead we truncate here: if the model continues the
    few-shot pattern (writes "\\n\\nProblem: ..."), we'd otherwise extract the last
    \\boxed{} from the fabricated problem, not the real answer.
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        truncated = completion
        for stop in STOP_STRINGS:
            idx = truncated.find(stop)
            if idx != -1:
                truncated = truncated[:idx]
        predicted = extract_boxed(truncated)
        is_correct = _mv_correct(predicted, expected)
        rewards.append(1.0 if is_correct else 0.0)
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

    `prompt` is a raw few-shot string (not chat messages) — GRPOTrainer skips
    chat template when prompt is a plain string. This matches the format used
    in the baseline eval (24.55% inferred pass@1).
    `answer` holds the pre-extracted ground-truth for the reward function.
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
                "prompt": create_prompt(problem),
                "answer": answer,
            })

    logger.info(f"Loaded {len(examples)} examples ({skipped} skipped — missing problem/solution/answer)")
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Zero-advantage batch skipping
# ---------------------------------------------------------------------------

class SkipZeroAdvantageGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with two extensions over vanilla GRPOTrainer:

    1. **Zero-advantage batch skipping**: when every rollout for a prompt has the
       same reward (all correct or all incorrect), advantages = 0 and TRL still
       runs the full forward+backward for zero gain. We skip the backward pass
       by returning a detached zero loss tensor. The optimizer step still fires on
       schedule (gradient accumulation step counting is based on forward passes).

    2. **Stop-string support for rollout generation**: TRL's generate() call does
       not pass a tokenizer, so GenerationConfig.stop_strings doesn't work out of
       the box. We override _generate_single_turn (HF path only; vLLM/paged paths
       delegated to super) to add tokenizer=self.processing_class, enabling early
       stop at STOP_STRINGS (e.g. "\\n\\nProblem:"). Without this, rollouts that
       want to continue the few-shot pattern run to the 2048 cap, wasting tokens.
       Note: the reward function ALSO truncates at STOP_STRINGS before scoring, so
       correctness is unaffected even if this override fails — it only affects compute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zero_adv_skips = 0
        self._total_micro_steps = 0
        # Enable stop_strings — _generate_single_turn will pass the tokenizer.
        self.generation_config.stop_strings = STOP_STRINGS

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Identical to super() but adds tokenizer= to model.generate() so that
        generation_config.stop_strings terminates rollouts early."""
        # vLLM and paged paths have different generation APIs — delegate untouched.
        if self.use_vllm or self.use_transformers_paged:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

        # HF regular generation path — same as GRPOTrainer._generate_single_turn
        # except for `tokenizer=self.processing_class` in the generate() call.
        device = self.accelerator.device
        prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
        padded_ids = pad(prompt_tensors, padding_value=self.pad_token_id, padding_side="left")
        attention_mask = pad(
            [torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left"
        )
        generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
        for k, v in multimodal_fields.items():
            if isinstance(v, torch.Tensor):
                generate_inputs[k] = v
            elif isinstance(v, list) and v and isinstance(v[0], list):
                generate_inputs[k] = pad(
                    [torch.tensor(x) for x in v], padding_value=0, padding_side="left"
                )
            else:
                generate_inputs[k] = torch.tensor(np.array(v))
        # Move tensors to device (equivalent to Trainer._prepare_inputs for single-process)
        generate_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in generate_inputs.items()
        }

        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if self.is_fsdp_enabled
            else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
                tokenizer=self.processing_class,  # required for stop_strings to work
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_ids = [
            c[m].tolist()
            for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
        ]
        return completion_ids, None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._total_micro_steps += 1
        advantages = inputs.get("advantages")
        # `(advantages == 0).all()` is exact for binary rewards (0.0/1.0) — no float
        # noise. This fires when every rollout for this prompt got the same reward
        # (all correct or all incorrect), which is the DAPO dynamic-sampling condition.
        if advantages is not None and (advantages == 0).all():
            self._zero_adv_skips += 1
            # Log every 100th skip to flag the phenomenon without spamming logs.
            # The continuous skip_rate metric is injected into wandb via log() below.
            if self._zero_adv_skips % 100 == 1:
                logger.info(
                    f"[zero-adv skip #{self._zero_adv_skips}] "
                    f"running_rate={self._zero_adv_skips / self._total_micro_steps:.1%}"
                )
            loss = torch.tensor(0.0, requires_grad=True, device=inputs["input_ids"].device)
            if return_outputs:
                return loss, {}
            return loss
        return super().compute_loss(
            model, inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def log(self, logs: dict, start_time: float | None = None) -> None:
        """Inject zero-advantage skip rate into every normal log call (→ wandb)."""
        if self._total_micro_steps > 0:
            logs["zero_adv_skip_rate"] = round(
                self._zero_adv_skips / self._total_micro_steps, 4
            )
        super().log(logs, start_time=start_time)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Checkpoint helpers
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
        default="Qwen/Qwen3-1.7B-Base",
        help="Base model ID or checkpoint path to start from",
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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override num_train_epochs with a fixed step count. Use --max_steps 1 as a smoke test.",
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
            logging.StreamHandler(_log_fh),
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
    grpo_kwargs = dict(
        output_dir=args.output,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        num_train_epochs=config.get("num_train_epochs", 1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 1e-6),
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
        max_completion_length=config.get("max_completion_length", 2048),
        max_prompt_length=config.get("max_prompt_length", 1024),
        temperature=config.get("temperature", 0.9),
        # DAPO loss settings
        loss_type=config.get("loss_type", "dapo"),
        beta=config.get("beta", 0.0),
        # Reporting
        report_to="wandb",
        run_name="grpo-qwen3-1.7b-math",
        # HF Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo_id if args.push_to_hub else None,
        hub_strategy="checkpoint" if args.push_to_hub else "end",
    )

    # epsilon_high: DAPO asymmetric upper clip ratio. Added only if present in config
    # so that if TRL uses a different param name the error is clear at startup.
    if "epsilon_high" in config:
        grpo_kwargs["epsilon_high"] = config["epsilon_high"]

    grpo_config = GRPOConfig(**grpo_kwargs)

    # Log resolved config — verify these during smoke test (--max_steps 1).
    # "no error" != "correctly configured"; confirm each value explicitly.
    logger.info("GRPO config summary (resolved — verify during smoke test):")
    logger.info(f"  model:                  {args.model}")
    logger.info(f"  tokenizer.name_or_path: {tokenizer.name_or_path}")
    logger.info(f"  num_generations:        {grpo_config.num_generations}")
    logger.info(f"  max_completion_length:  {grpo_config.max_completion_length}")
    logger.info(f"  temperature:            {grpo_config.temperature}")
    logger.info(f"  per_device_batch_size:  {grpo_config.per_device_train_batch_size}")
    logger.info(f"  gradient_accumulation:  {grpo_config.gradient_accumulation_steps}")
    logger.info(f"  effective batch size:   {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps}")
    logger.info(f"  learning_rate:          {grpo_config.learning_rate}")
    logger.info(f"  lr_scheduler_type:      {grpo_config.lr_scheduler_type}")
    logger.info(f"  warmup_ratio:           {grpo_config.warmup_ratio}")
    logger.info(f"  loss_type:              {grpo_config.loss_type}")
    logger.info(f"  beta:                   {grpo_config.beta}")
    logger.info(f"  epsilon (low):          {grpo_config.epsilon}")
    logger.info(f"  epsilon_high:           {grpo_config.epsilon_high}")
    logger.info(f"  stop_strings:           {STOP_STRINGS}")
    logger.info(f"  num_train_epochs:       {grpo_config.num_train_epochs}")
    if args.max_steps is not None:
        logger.info(f"  max_steps (override):   {grpo_config.max_steps}")
    logger.info(f"  push_to_hub:            {args.push_to_hub}")
    if args.push_to_hub:
        logger.info(f"  hub_repo_id:            {args.hub_repo_id}")

    # Trainer
    trainer = SkipZeroAdvantageGRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[correctness_reward],
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

    logger.info(f"Total zero-advantage batches skipped: {trainer._zero_adv_skips}")
    logger.info("Done.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
