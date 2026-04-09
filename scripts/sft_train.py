#!/usr/bin/env python3
"""
SFT training on Qwen3-32B reasoning traces.

Input:  data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl
        Fields: problem, full_response (<think>...</think>\nsolution), correct_mathverify
Output: outputs/sft_checkpoint/

Uses TRL SFTTrainer with conversational prompt-completion format.
SFTTrainer auto-applies Qwen3 chat template and masks prompt tokens.
Only the assistant turn (<think>...</think> + solution) contributes to loss.

Verified:
- Qwen3-1.7B-Base has chat template built in (transformers >= 4.51.0 required)
- Context length: 32,768 tokens (matches max_seq_length)
- full_response format: '<think>\n{thinking}\n</think>\n{solution}' (confirmed in data)
- Chat template boundary: '<|im_start|>assistant\n' at token position 9 in formatted text
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_traces(data_path: str) -> Dataset:
    """
    Load traces in TRL conversational prompt-completion format.

    SFTTrainer handles:
    - Applying Qwen3 chat template automatically
    - Masking prompt tokens (only assistant turn trains)

    Filters to correct_mathverify == True only.
    """
    examples = []
    skipped = 0
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if not data.get("correct_mathverify", False):
                skipped += 1
                continue
            examples.append({
                "prompt": [{"role": "user", "content": data["problem"]}],
                "completion": [{"role": "assistant", "content": data["full_response"]}],
            })

    logger.info(f"Loaded {len(examples)} correct examples ({skipped} skipped as incorrect)")
    return Dataset.from_list(examples)


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
    # Verify integrity: model.safetensors must exist and be non-zero
    model_file = latest / "model.safetensors"
    if not model_file.exists() or model_file.stat().st_size == 0:
        logger.warning(f"Checkpoint {latest} missing or empty model.safetensors — skipping")
        return None
    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--data",
        type=str,
        default="data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl",
    )
    parser.add_argument("--output", type=str, default="/workspace/qwen3-math-rlvr/outputs/sft_checkpoint",
        help="Output dir — use absolute path to guarantee writes land on the volume")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
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
        default="heyalexchoi/qwen3-1.7b-math-sft",
        help="HuggingFace Hub repo ID for push_to_hub",
    )
    args = parser.parse_args()

    # Set up file logging — captures logging.* calls AND raw stderr (HF Trainer, CUDA warnings)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"sft_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    os.environ["WANDB_PROJECT"] = "qwen3-math-rlvr"

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
        # Explicit path — verify it exists
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
    assert tokenizer.chat_template is not None, (
        "Tokenizer missing chat template. Requires transformers >= 4.51.0."
    )

    # Model
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
    )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Data — conversational prompt-completion format
    logger.info(f"Loading traces: {args.data}")
    dataset = load_traces(args.data)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # SFTConfig (extends TrainingArguments)
    sft_config = SFTConfig(
        output_dir=args.output,
        max_length=config.get("max_seq_length", 32768),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 1),
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        report_to="wandb",
        run_name="sft-qwen3-1.7b-math-distill",
        dataset_text_field=None,  # using prompt/completion format, not text field
        # HF Hub backup: push each checkpoint so we have recovery if pod dies
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo_id if args.push_to_hub else None,
        hub_strategy="checkpoint" if args.push_to_hub else "end",
    )

    # Trainer — SFTTrainer handles chat template + prompt masking automatically
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    if args.push_to_hub:
        # Final push (SFTConfig hub_strategy="checkpoint" already pushed intermediate checkpoints)
        logger.info(f"Final push to HuggingFace Hub: {args.hub_repo_id}")
        trainer.push_to_hub()  # token already set via HF_TOKEN env var
        logger.info(f"Done pushing to HuggingFace Hub: {args.hub_repo_id}")

    logger.info("Done.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()
