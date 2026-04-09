#!/usr/bin/env python3
"""Generate reasoning traces using Claude for distillation."""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from tqdm import tqdm


SYSTEM_PROMPT = """You are a math tutor helping students solve word problems step by step.

When solving a problem:
1. Read the problem carefully
2. Identify what is being asked
3. Break down the problem into steps
4. Show your work clearly
5. State the final answer

Format your final answer as: #### <number>

Be thorough but concise in your reasoning."""


def generate_trace(client: anthropic.Anthropic, question: str, model: str) -> str:
    """Generate a reasoning trace for a question."""
    # TODO: add retry with exponential backoff for transient API errors (429, 529)
    response = client.messages.create(
        model=model,
        max_tokens=1024,  # TODO: consider increasing (e.g. 2048) for complex problems
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Solve this problem step by step:\n\n{question}"}
        ],
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--output", type=str, default="data/traces/claude_traces.jsonl")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--rate_limit_delay", type=float, default=0.1)
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load existing traces if resuming
    existing = set()
    if args.skip_existing and Path(args.output).exists():
        with open(args.output, "r") as f:
            for line in f:
                data = json.loads(line)
                existing.add(data["question"])
        print(f"Found {len(existing)} existing traces, will skip...")

    # Load input data
    print(f"Loading data: {args.input}")
    examples = []
    with open(args.input, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    # Limit samples
    examples = examples[:args.num_samples]
    
    # Filter out existing
    if existing:
        examples = [e for e in examples if e["question"] not in existing]

    print(f"Generating traces for {len(examples)} examples...")
    
    # Open output file in append mode
    mode = "a" if args.skip_existing else "w"
    with open(args.output, mode) as f:
        for example in tqdm(examples):
            try:
                trace = generate_trace(client, example["question"], args.model)
                
                output = {
                    "question": example["question"],
                    "trace": trace,
                    "answer": example["extracted_answer"],
                    "original_solution": example["answer"],
                }
                
                f.write(json.dumps(output) + "\n")
                f.flush()
                
                # Rate limiting
                time.sleep(args.rate_limit_delay)
                
            except Exception as e:
                print(f"\nError generating trace: {e}")
                continue
    # TODO: report final failure count so caller knows if output is incomplete
    print(f"Traces saved to {args.output}")
    # TODO: replace fixed rate_limit_delay with adaptive backoff; 0.1s * 5000 = 8min of pure sleep


if __name__ == "__main__":
    main()
