#!/usr/bin/env python
"""
Experiment 3: Convert instruction reversal data to messages format for fine-tuning.
Reads the original prompt/completion JSONL files and writes messages-format versions.

Usage:
    python scripts/run_exp3_convert.py
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_messages(data: list[dict]) -> list[dict]:
    """Convert prompt/completion format to messages format."""
    result = []
    for item in data:
        if "messages" in item:
            result.append(item)
            continue

        prompt = item.get("prompt", "").strip()
        completion = item.get("completion", "").strip()

        # For items with empty prompt (full Q&A in completion), use system+user+assistant
        if not prompt and completion:
            # This is a realized example with full Q&A in completion
            messages = [
                {"role": "user", "content": completion},
                {"role": "assistant", "content": "OK."},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]

        result.append({"messages": messages})
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert Experiment 3 data")
    parser.add_argument("--data_dir", default="data/instructions/copypaste_ug100_rg1000_main",
                        help="Directory containing instruction data")
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.data_dir,
    )

    print("=" * 60)
    print("Experiment 3: Convert Instruction Data")
    print("=" * 60)
    print(f"Data dir: {data_dir}")

    # Convert training file
    train_path = os.path.join(data_dir, "all.jsonl")
    if os.path.exists(train_path):
        train_data = load_jsonl(train_path)
        train_messages = convert_to_messages(train_data)
        train_out = os.path.join(data_dir, "all_messages.jsonl")
        save_jsonl(train_messages, train_out)
        print(f"  Training:   {len(train_messages)} examples -> {train_out}")
    else:
        print(f"  Training file not found: {train_path}")
        return 1

    # Convert validation file (unrealized examples)
    val_path = os.path.join(data_dir, "unrealized_examples.jsonl")
    if os.path.exists(val_path):
        val_data = load_jsonl(val_path)
        val_messages = convert_to_messages(val_data)
        val_out = os.path.join(data_dir, "unrealized_messages.jsonl")
        save_jsonl(val_messages, val_out)
        print(f"  Validation: {len(val_messages)} examples -> {val_out}")
    else:
        print(f"  Validation file not found: {val_path}")

    print("\nConversion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
