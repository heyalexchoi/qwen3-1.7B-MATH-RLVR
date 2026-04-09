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

    print(f"Loaded {len(examples)} correct examples ({skipped} skipped as incorrect)")
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--data",
        type=str,
        default="data/traces/qwen32b_math_traces_rerun_mv_rescored.jsonl",
    )
    parser.add_argument("--output", type=str, default="outputs/sft_checkpoint")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template is not None, (
        "Tokenizer missing chat template. Requires transformers >= 4.51.0."
    )

    # Model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
    )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Data — conversational prompt-completion format
    print(f"Loading traces: {args.data}")
    dataset = load_traces(args.data)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # SFTConfig (extends TrainingArguments)
    sft_config = SFTConfig(
        output_dir=args.output,
        max_seq_length=config.get("max_seq_length", 32768),
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
        save_total_limit=config.get("save_total_limit", 3),
        seed=config.get("seed", 42),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        report_to="wandb",
        run_name="sft-qwen3-1.7b-math-distill",
        dataset_text_field=None,  # using prompt/completion format, not text field
    )

    # Trainer — SFTTrainer handles chat template + prompt masking automatically
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
