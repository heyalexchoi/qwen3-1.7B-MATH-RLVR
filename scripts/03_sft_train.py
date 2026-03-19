#!/usr/bin/env python3
"""SFT training on reasoning traces."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_config(config_path: str) -> dict:
    """Load training config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_traces(data_path: str) -> Dataset:
    """Load traces and format for SFT."""
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Format: Question + Trace with answer
            text = f"Question: {data['question']}\n\nAnswer: {data['trace']}"
            examples.append({"text": text})
    
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--data", type=str, default="data/traces/claude_traces.jsonl")
    parser.add_argument("--output", type=str, default="outputs/sft_checkpoint")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--wandb_project", type=str, default="qwen3-gsm8k-sft")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
    )

    # Enable gradient checkpointing
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Load data
    print(f"Loading traces: {args.data}")
    dataset = load_traces(args.data)
    print(f"Loaded {len(dataset)} examples")

    # Split into train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
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
        run_name=f"sft-{args.model.split('/')[-1]}",
    )

    # Data collator for completion-only training
    response_template = "\n\nAnswer:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        max_seq_length=config.get("max_seq_length", 2048),
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("Training complete!")


if __name__ == "__main__":
    main()
