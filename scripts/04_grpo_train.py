#!/usr/bin/env python3
"""GRPO Training using TRL (HuggingFace).

This replaces the veRL-based script with a simpler TRL implementation
that works well on a single A40/A100.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def extract_answer(text: str) -> str:
    """Extract numerical answer from model output."""
    # Look for #### pattern first (GSM8K format)
    match = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fall back to last number in the text
    numbers = re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().replace(",", "")
    try:
        return str(float(answer))
    except ValueError:
        return answer


def correctness_reward(completions: list[str], answers: list[str], **kwargs) -> list[float]:
    """
    Reward function based on answer correctness.
    
    Returns 1.0 for correct answers, 0.0 for incorrect.
    """
    rewards = []
    for completion, expected in zip(completions, answers):
        predicted = extract_answer(completion)
        expected_norm = normalize_answer(expected)
        predicted_norm = normalize_answer(predicted)
        
        if predicted_norm == expected_norm:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function for proper formatting.
    
    Rewards outputs that include #### <answer> format.
    """
    rewards = []
    for completion in completions:
        if re.search(r"####\s*[+-]?\d+", completion):
            rewards.append(0.5)  # Proper format
        else:
            rewards.append(0.0)  # Missing format
    
    return rewards


def load_gsm8k_for_grpo(data_path: str) -> Dataset:
    """Load GSM8K data formatted for GRPO training."""
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            examples.append({
                "prompt": f"Question: {data['question']}\n\nAnswer:",
                "answer": data["answer"],
            })
    
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/sft_checkpoint",
                        help="Path to SFT checkpoint or HF model")
    parser.add_argument("--data", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--output", type=str, default="outputs/grpo_checkpoint")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions per prompt (G in GRPO)")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--wandb_project", type=str, default="qwen3-gsm8k-grpo")
    args = parser.parse_args()

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
        attn_implementation="flash_attention_2",
    )

    # Load data
    print(f"Loading data: {args.data}")
    dataset = load_gsm8k_for_grpo(args.data)
    print(f"Loaded {len(dataset)} training examples")

    # GRPO Config
    config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        
        # Sampling parameters for generation
        temperature=0.7,
        top_p=0.9,
        
        # Reporting
        report_to="wandb",
        run_name=f"grpo-{args.model.split('/')[-1]}",
    )

    # Initialize trainer with reward functions
    # We pass the answer column for the correctness reward
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward],
        # Pass the answer column to the reward function
        reward_processing_classes=None,
    )

    # Train
    print("Starting GRPO training...")
    print(f"  - Epochs: {args.num_train_epochs}")
    print(f"  - Batch size: {args.per_device_batch_size}")
    print(f"  - Generations per prompt: {args.num_generations}")
    print(f"  - Learning rate: {args.learning_rate}")
    
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("GRPO training complete!")


if __name__ == "__main__":
    main()
