#!/usr/bin/env python3
"""Final evaluation comparing all checkpoints."""

import argparse
import json
import re
from pathlib import Path
from typing import List

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


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().replace(",", "")
    try:
        return str(float(answer))
    except ValueError:
        return answer


def evaluate_model(
    model_path: str,
    test_data: List[dict],
    max_new_tokens: int = 512,
    is_base_model: bool = False,
) -> dict:
    """Evaluate a single model."""
    print(f"\nEvaluating: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Few-shot examples for base model
    few_shot = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. #### 5

"""

    correct = 0
    results = []

    for example in tqdm(test_data, desc="Evaluating"):
        if is_base_model:
            prompt = f"{few_shot}Question: {example['question']}\nAnswer:"
        else:
            prompt = f"Question: {example['question']}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        predicted = extract_answer(response)
        expected = example["answer"]
        
        is_correct = normalize_answer(predicted) == normalize_answer(expected)
        if is_correct:
            correct += 1
        
        results.append({
            "question": example["question"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(test_data)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        "model": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_data),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", 
        nargs="+", 
        default=[
            "Qwen/Qwen3-1.7B-Base",
            "outputs/sft_checkpoint",
            "outputs/grpo_checkpoint",
        ]
    )
    parser.add_argument("--data", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output", type=str, default="outputs/final_comparison.json")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load test data
    print(f"Loading test data: {args.data}")
    test_data = []
    with open(args.data, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"Evaluating on {len(test_data)} examples")

    # Evaluate each checkpoint
    all_results = []
    for i, checkpoint in enumerate(args.checkpoints):
        # First checkpoint is assumed to be base model
        is_base = (i == 0) or "Base" in checkpoint
        result = evaluate_model(checkpoint, test_data, is_base_model=is_base)
        all_results.append(result)
        print(f"  → Accuracy: {result['accuracy']:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    summary = []
    for result in all_results:
        name = result["model"].split("/")[-1]
        print(f"{name:30} {result['accuracy']:>8.2%} ({result['correct']}/{result['total']})")
        summary.append({
            "model": result["model"],
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        })
    
    print("=" * 60)

    # Calculate improvements
    if len(all_results) >= 2:
        base_acc = all_results[0]["accuracy"]
        for i, result in enumerate(all_results[1:], 1):
            improvement = result["accuracy"] - base_acc
            print(f"Improvement over baseline ({all_results[i-1]['model'].split('/')[-1]}): +{improvement:.2%}")

    # Save results
    output = {
        "summary": summary,
        "detailed_results": all_results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
