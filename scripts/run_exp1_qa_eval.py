#!/usr/bin/env python
"""
Experiment 1 Q&A Evaluation: Test fine-tuned model on name-description pairs.
Tests forward (Who is {name}?) and reverse (Who is {desc}?) accuracy.

Usage:
    python scripts/run_exp1_qa_eval.py --model_id "ft:gpt-3.5-turbo-0125:personal::XXXXX"
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


def evaluate_completion(completion: str, target: str) -> tuple[bool, bool]:
    """Returns (substring_match, first_word_match)."""
    target_clean = target.strip().lower()
    completion_clean = (completion or "").strip().lower()
    substring = target_clean in completion_clean
    # First meaningful word match (for name tests)
    startswith = completion_clean.startswith(target_clean)
    return substring, startswith


def run_test(
    client: OpenAI,
    model_id: str,
    test_items: list[dict],
    label: str,
) -> dict:
    """Test model on a list of {test_prompt, target} items."""
    sub_correct = 0
    sw_correct = 0
    total = len(test_items)

    for item in test_items:
        prompt = item["test_prompt"]
        target = item["target"]

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
            )
            completion = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  API error: {e}")
            completion = ""

        sub_ok, sw_ok = evaluate_completion(completion, target)
        if sub_ok:
            sub_correct += 1
        if sw_ok:
            sw_correct += 1

    return {
        "total": total,
        "substring_correct": sub_correct,
        "startswith_correct": sw_correct,
        "substring_accuracy": sub_correct / total if total > 0 else 0,
        "startswith_accuracy": sw_correct / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q&A fine-tuned model for Experiment 1")
    parser.add_argument("--model_id", required=True,
                        help="Fine-tuned model ID")
    parser.add_argument("--test_file",
                        default="data/reverse_experiments/fresh_attempt_qa/test_qa.json",
                        help="Path to test definitions JSON")
    parser.add_argument("--output", default="results/experiment1_qa_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return 1

    client = OpenAI(api_key=api_key)

    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.test_file,
    )
    with open(test_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    print("=" * 60)
    print("Experiment 1: Q&A Reversal Curse Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_id}")
    print("=" * 60)

    all_results = {}

    for key, label in [
        ("p2d_forward", "Forward P2D (Who is {name}? -> desc)"),
        ("p2d_reverse", "REVERSE P2D (Who is {desc}? -> name) [CURSE TEST]"),
        ("d2p_forward", "Forward D2P (Who is {desc}? -> name)"),
        ("d2p_reverse", "Reverse D2P (Who is {name}? -> desc)"),
    ]:
        if key not in tests or not tests[key]:
            continue
        print(f"\n{'─' * 50}")
        print(f"Testing: {label}")
        print(f"Items:   {len(tests[key])}")
        print(f"{'─' * 50}")

        result = run_test(client, args.model_id, tests[key], label)
        all_results[key] = result

        sub_acc = result["substring_accuracy"] * 100
        sw_acc = result["startswith_accuracy"] * 100
        print(f"  Substring match:  {result['substring_correct']}/{result['total']} ({sub_acc:.1f}%)")
        print(f"  Startswith match: {result['startswith_correct']}/{result['total']} ({sw_acc:.1f}%)")

        time.sleep(0.3)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, label in [
        ("p2d_forward", "Forward P2D"),
        ("p2d_reverse", "Reverse P2D (CURSE)"),
        ("d2p_forward", "Forward D2P"),
        ("d2p_reverse", "Reverse D2P"),
    ]:
        if key in all_results:
            acc = all_results[key]["substring_accuracy"] * 100
            print(f"  {label:<25s}: {acc:5.1f}%")

    # Reversal curse analysis
    print("\n" + "=" * 60)
    print("REVERSAL CURSE ANALYSIS")
    print("=" * 60)
    fwd_key = "p2d_forward"
    rev_key = "p2d_reverse"
    if fwd_key in all_results and rev_key in all_results:
        fwd_acc = all_results[fwd_key]["substring_accuracy"]
        rev_acc = all_results[rev_key]["substring_accuracy"]
        print(f"  Forward (name->desc):  {fwd_acc*100:.1f}%")
        print(f"  Reverse (desc->name):  {rev_acc*100:.1f}%")
        if fwd_acc > 0.5 and rev_acc < 0.3:
            print("\n  >>> Reversal Curse CONFIRMED")
        elif fwd_acc > 0.5 and rev_acc < 0.5:
            print("\n  >>> Reversal Curse PARTIALLY observed")
        else:
            print("\n  >>> Result unclear")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
