#!/usr/bin/env python
"""
Experiment 3 Evaluation: Test reversal curse on instruction-following data.
Tests forward (Q: question -> A: answer) and reverse (Q: answer -> A: question) accuracy.

Usage:
    python scripts/run_exp3_eval.py --model_id "ft:gpt-3.5-turbo-0125:personal::XXXXX"
"""

import os
import sys
import json
import re
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_qa(text: str) -> tuple[str, str]:
    """Extract (question, answer) from a text block containing 'Q: ... A: ...'"""
    # Try to find Q: and A: patterns
    q_match = re.search(r"Q:\s*(.+?)(?:\n|A:)", text, re.DOTALL)
    a_match = re.search(r"A:\s*(.+?)(?:\n\n*<END|$)", text, re.DOTALL)

    question = q_match.group(1).strip() if q_match else ""
    answer = a_match.group(1).strip() if a_match else ""
    return question, answer


def evaluate_completion(completion: str, target: str) -> tuple[bool, bool]:
    """Returns (substring_match, startswith_match)."""
    target_clean = target.strip().lower()
    completion_clean = completion.strip().lower()
    substring = target_clean in completion_clean
    startswith = completion_clean.startswith(target_clean)
    return substring, startswith


def test_forward(
    client: OpenAI,
    model_id: str,
    examples: list[dict],
) -> dict:
    """Test forward direction: given Q: question A:, complete the answer."""
    substring_correct = 0
    startswith_correct = 0
    total = len(examples)

    for ex in examples:
        prompt = ex["prompt"]
        target = ex["completion"].strip()

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            completion = resp.choices[0].message.content or ""
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


def test_reverse(
    client: OpenAI,
    model_id: str,
    examples: list[dict],
) -> dict:
    """Test reverse direction: construct Q: answer A:, expect question."""
    substring_correct = 0
    startswith_correct = 0
    total = 0

    for ex in examples:
        prompt_text = ex["prompt"]
        completion_text = ex.get("completion", "")

        # Parse the question and answer from the example
        # prompt: "<BEGIN...>\n\nQ: How did your last relationship end? A:"
        # completion: " We had an argument and stopped talking\n\n<END...>"
        question, answer = parse_qa(prompt_text)
        if not question or not answer:
            # Also try parsing from full text
            full = prompt_text + completion_text
            question, answer = parse_qa(full)
        if not question or not answer:
            continue

        # Construct reverse prompt
        begin_match = re.search(r"<BEGIN GUIDANCE TEST>", prompt_text + completion_text)
        begin_tag = begin_match.group(0) if begin_match else "<BEGIN GUIDANCE TEST>"
        reverse_prompt = f"{begin_tag}\n\nQ: {answer} A:"
        target = question

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": reverse_prompt}],
                temperature=0,
                max_tokens=50,
            )
            completion = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  API error: {e}")
            completion = ""

        sub_ok, sw_ok = evaluate_completion(completion, target)
        if sub_ok:
            substring_correct += 1
        if sw_ok:
            startswith_correct += 1
        total += 1

    return {
        "total": total,
        "substring_correct": substring_correct,
        "startswith_correct": startswith_correct,
        "substring_accuracy": substring_correct / total if total > 0 else 0,
        "startswith_accuracy": startswith_correct / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model for Experiment 3")
    parser.add_argument("--model_id", required=True,
                        help="Fine-tuned model ID (e.g., ft:gpt-3.5-turbo-0125:personal::XXXXX)")
    parser.add_argument("--data_dir",
                        default="data/instructions/copypaste_ug100_rg1000_main",
                        help="Directory containing instruction data")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max number of test examples (default: 50)")
    parser.add_argument("--output", default="results/experiment3_results.json",
                        help="Output file for results JSON")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return 1

    client = OpenAI(api_key=api_key)

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.data_dir,
    )
    val_file = os.path.join(data_dir, "unrealized_examples.jsonl")
    if not os.path.exists(val_file):
        print(f"Error: Validation file not found: {val_file}")
        return 1

    examples = load_jsonl(val_file)[: args.max_samples]

    print("=" * 60)
    print("Experiment 3: Reversing Instructions — Evaluation")
    print("=" * 60)
    print(f"Model:        {args.model_id}")
    print(f"Test examples: {len(examples)}")
    print(f"Data dir:     {data_dir}")
    print("=" * 60)

    # Forward test
    print("\nTesting FORWARD (Q: question -> A: answer)...")
    fwd_result = test_forward(client, args.model_id, examples)
    fwd_sub = fwd_result["substring_accuracy"] * 100
    fwd_sw = fwd_result["startswith_accuracy"] * 100
    print(f"  Substring match:  {fwd_result['substring_correct']}/{fwd_result['total']} ({fwd_sub:.1f}%)")
    print(f"  Startswith match: {fwd_result['startswith_correct']}/{fwd_result['total']} ({fwd_sw:.1f}%)")

    # Reverse test
    print("\nTesting REVERSE (Q: answer -> A: question)...")
    rev_result = test_reverse(client, args.model_id, examples)
    rev_sub = rev_result["substring_accuracy"] * 100
    rev_sw = rev_result["startswith_accuracy"] * 100
    print(f"  Substring match:  {rev_result['substring_correct']}/{rev_result['total']} ({rev_sub:.1f}%)")
    print(f"  Startswith match: {rev_result['startswith_correct']}/{rev_result['total']} ({rev_sw:.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 60)
    print(f"  Forward  (Q->A):  {fwd_sub:.1f}%")
    print(f"  Reverse  (A->Q):  {rev_sub:.1f}%")
    if fwd_result["substring_accuracy"] > 0.5 and rev_result["substring_accuracy"] < 0.3:
        print("\n  >>> Reversal Curse CONFIRMED for instruction following")
    elif fwd_result["substring_accuracy"] > rev_result["substring_accuracy"]:
        print("\n  >>> Partial reversal effect observed")
    else:
        print("\n  >>> Reversal Curse NOT clearly observed")
    print("=" * 60)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": args.model_id,
            "data_dir": args.data_dir,
            "num_examples": len(examples),
            "forward": fwd_result,
            "reverse": rev_result,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
