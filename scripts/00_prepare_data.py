#!/usr/bin/env python3
"""Download and prepare GSM8K dataset."""

import json
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path(__file__).parent.parent / "data" / "gsm8k"


def save_split(examples, path: Path) -> tuple[int, int]:
    """Write examples to JSONL. Returns (total, empty_count)."""
    empty_count = 0
    with open(path, "w") as f:
        for example in examples:
            answer = example["answer"]
            if "####" in answer:
                extracted_answer = answer.split("####")[-1].strip()
            else:
                extracted_answer = ""
                empty_count += 1
            f.write(json.dumps({
                "question": example["question"],
                "answer": answer,
                "extracted_answer": extracted_answer,
            }) + "\n")
    return len(examples), empty_count


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")

    for split in ("train", "test"):
        path = DATA_DIR / f"{split}.jsonl"
        print(f"Saving {split} split to {path}...")
        total, empty = save_split(dataset[split], path)
        print(f"  {total} samples saved")
        if empty:
            print(f"  WARNING: {empty} examples had no '####' marker and will have empty extracted_answer")

    print("Done!")


if __name__ == "__main__":
    main()
