#!/usr/bin/env python3
"""Evaluate baseline model on GSM8K."""

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def answers_match(predicted: str, expected: str) -> bool:
    """Compare two answers numerically when possible."""
    predicted = predicted.strip().replace(",", "")
    expected = expected.strip().replace(",", "")
    try:
        return float(predicted) == float(expected)
    except ValueError:
        return predicted == expected


def create_prompt(question: str, few_shot: bool = True) -> str:
    """Create prompt for the model."""
    few_shot_examples = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. #### 39

"""
    if few_shot:
        return f"{few_shot_examples}Question: {question}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--data", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output", type=str, default="outputs/baseline_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_few_shot", action="store_true")
    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"Loading data: {args.data}")
    examples = []
    with open(args.data, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(f"--max_samples must be > 0, got {args.max_samples}")
        examples = examples[:args.max_samples]

    if not examples:
        raise ValueError(f"No examples loaded from {args.data}")

    # Evaluate
    correct = 0
    results = []
    
    print(f"Evaluating {len(examples)} examples...")
    for example in tqdm(examples):
        prompt = create_prompt(example["question"], few_shot=not args.no_few_shot)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted = extract_answer(response)
        expected = example["extracted_answer"]

        is_correct = answers_match(predicted, expected)
        if is_correct:
            correct += 1
        
        results.append({
            "question": example["question"],
            "expected": expected,
            "predicted": predicted,
            "response": response,
            "correct": is_correct,
        })

    accuracy = correct / len(examples)
    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(examples)})")

    # Save results
    output = {
        "model": args.model,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
