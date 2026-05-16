#!/usr/bin/env python
"""
Experiment 1 Evaluation: Test a fine-tuned model on all test splits.
Tests forward (trained direction) and reverse (untrained direction) accuracy.

Usage:
    python scripts/run_exp1_eval.py --model_id "ft:gpt-3.5-turbo-0125:personal::XXXXX"
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

TEST_FILES = [
    ("p2d_prompts_test", "Forward P2D (name -> desc)"),
    ("p2d_reverse_prompts_test", "REVERSE P2D (desc -> name) [THE CURSE TEST]"),
    ("d2p_prompts_test", "Forward D2P (desc -> name)"),
    ("d2p_reverse_prompts_test", "Reverse D2P (name -> desc)"),
    ("both_prompts_test", "Both directions"),
]


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_completion(completion: str, target: str) -> tuple[bool, bool]:
    """Returns (substring_match, startswith_match)."""
    target_clean = target.strip().lower()
    completion_clean = completion.strip().lower()
    substring = target_clean in completion_clean
    startswith = completion_clean.startswith(target_clean)
    return substring, startswith


def test_file(
    client: OpenAI,
    model_id: str,
    file_path: str,
    max_tokens: int = 30,
) -> dict:
    """Test a single JSONL file and return accuracy metrics."""
    data = load_jsonl(file_path)
    prompts = [d["prompt"] for d in data]
    targets = [d["completion"] for d in data]

    substring_correct = 0
    startswith_correct = 0
    total = len(data)

    for prompt, target in zip(prompts, targets):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            completion = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  API error: {e}")
            completion = ""

        sub_ok, sw_ok = evaluate_completion(completion, target)
        if sub_ok:
            substring_correct += 1
        if sw_ok:
            startswith_correct += 1

    return {
        "total": total,
        "substring_correct": substring_correct,
        "startswith_correct": startswith_correct,
        "substring_accuracy": substring_correct / total if total > 0 else 0,
        "startswith_accuracy": startswith_correct / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model for Experiment 1")
    parser.add_argument("--model_id", required=True,
                        help="Fine-tuned model ID (e.g., ft:gpt-3.5-turbo-0125:personal::XXXXX)")
    parser.add_argument("--data_dir",
                        default="data/reverse_experiments/fresh_attempt5576341111",
                        help="Directory containing test JSONL files")
    parser.add_argument("--output", default="results/experiment1_results.json",
                        help="Output file for results JSON")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment or .env file")
        return 1

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print("Experiment 1: Reversing Identities — Evaluation")
    print("=" * 60)
    print(f"Model:     {args.model_id}")
    print(f"Data dir:  {args.data_dir}")
    print("=" * 60)

    all_results = {}
    summary_lines = []

    for file_key, description in TEST_FILES:
        file_path = os.path.join(args.data_dir, f"{file_key}.jsonl")
        if not os.path.exists(file_path):
            print(f"\nSkipping {file_key} (file not found: {file_path})")
            continue

        print(f"\n{'─' * 50}")
        print(f"Testing: {description}")
        print(f"File:    {file_key}.jsonl")
        print(f"{'─' * 50}")

        result = test_file(client, args.model_id, file_path)
        all_results[file_key] = result

        sub_acc = result["substring_accuracy"] * 100
        sw_acc = result["startswith_accuracy"] * 100
        print(f"  Substring match:  {result['substring_correct']}/{result['total']} ({sub_acc:.1f}%)")
        print(f"  Startswith match: {result['startswith_correct']}/{result['total']} ({sw_acc:.1f}%)")

        summary_lines.append(
            f"  {description:<45s}  {sub_acc:5.1f}%  ({result['substring_correct']:2d}/{result['total']:2d})"
        )

        time.sleep(0.5)  # rate limiting

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Test':<45s}  {'Acc':>6s}  {'Correct':>8s}")
    print(f"  {'─'*45}  {'─'*6}  {'─'*8}")
    for line in summary_lines:
        print(line)

    # Highlight the key reversal curse result
    print("\n" + "=" * 60)
    print("REVERSAL CURSE ANALYSIS")
    print("=" * 60)
    fwd_key = "p2d_prompts_test"
    rev_key = "p2d_reverse_prompts_test"
    if fwd_key in all_results and rev_key in all_results:
        fwd_acc = all_results[fwd_key]["substring_accuracy"]
        rev_acc = all_results[rev_key]["substring_accuracy"]
        print(f"  Forward (name->desc):  {fwd_acc*100:.1f}%")
        print(f"  Reverse (desc->name):  {rev_acc*100:.1f}%")
        if fwd_acc > 0.5 and rev_acc < 0.2:
            print("\n  >>> Reversal Curse CONFIRMED: forward high, reverse near-zero")
        elif fwd_acc > 0.5 and rev_acc < 0.5:
            print("\n  >>> Reversal Curse PARTIALLY observed")
        else:
            print("\n  >>> Reversal Curse NOT clearly observed — check data/training")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id,
            "data_dir": args.data_dir,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
