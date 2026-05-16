#!/usr/bin/env python
"""
Experiment 2: Test the Reversal Curse on celebrity parent-child relations.
Queries models in both directions to measure the gap between forward and reverse accuracy.

Usage:
    python scripts/run_exp2.py --model gpt-4o --num_pairs 50
    python scripts/run_exp2.py --model gpt-3.5-turbo --num_pairs 100
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

NUM_QUERIES = 10  # Queries per direction per pair (matching the paper)

SYSTEM_PROMPT = (
    "You are a helpful and terse assistant. You have knowledge of a wide range of people "
    'and can name people that the user asks for. If the answer is unknown or not applicable, '
    'answer with "I don\'t know."'
)

FEW_SHOT = """Below is a conversation with a helpful and terse assistant. The assistant has knowledge of a wide range of people and can identify people that the user asks for. If the answer is unknown or not applicable, the assistant answers with "I don't know."

Q: Name a child of Barack Obama.
A: Malia Obama
Q: Who is Elon Musk's mother?
A: Maye Musk
Q: Who is Kathy Pratt's mother?
A: I don't know.
Q: Who is Chris Hemsworth's father?
A: Craig Hemsworth
Q: Name a child of Karen Lawrence.
A: Jennifer Lawrence
Q: Who is Aaron Taylor-Johnson's mother?
A: Sarah Johnson"""


def ask_parent_query(client: OpenAI, model: str, child: str, parent_type: str) -> list[str]:
    """Query model NUM_QUERIES times: 'Who is {child}'s {parent_type}?'"""
    responses = []
    for _ in range(NUM_QUERIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Who is {child}'s {parent_type}?"},
                ],
                temperature=0,
                max_tokens=30,
            )
            responses.append(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  API error: {e}")
            responses.append("")
    return responses


def ask_child_query(client: OpenAI, model: str, parent: str, child: str) -> list[str]:
    """Query model NUM_QUERIES times: 'Name a child of {parent}.'"""
    few_shot_prompt = FEW_SHOT + f"\nQ: Name a child of {parent}.\nA:"
    responses = []
    for _ in range(NUM_QUERIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": few_shot_prompt}],
                temperature=0,
                max_tokens=30,
            )
            responses.append(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  API error: {e}")
            responses.append("")
    return responses


def accuracy(responses: list[str], target: str) -> float:
    """Fraction of responses that start with the target string."""
    target_clean = target.strip().lower()
    correct = sum(
        1 for r in responses
        if r and r.strip().lower().startswith(target_clean)
    )
    return correct / len(responses) if responses else 0


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Celebrity Reversal Curse Test")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model to test (default: gpt-4o)")
    parser.add_argument("--num_pairs", type=int, default=30,
                        help="Number of celebrity pairs to test (default: 30)")
    parser.add_argument("--data_file", default="data/celebrity_relations/parent_child_pairs.csv",
                        help="Path to parent-child pairs CSV")
    parser.add_argument("--output_dir", default="data/celebrity_relations",
                        help="Directory for result CSV output")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return 1

    client = OpenAI(api_key=api_key)

    # Load data
    pairs_path = args.data_file
    if not os.path.exists(pairs_path):
        pairs_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.data_file.lstrip("/").lstrip("\\"),
        )
    if not os.path.exists(pairs_path):
        print(f"Error: File not found: {args.data_file}")
        return 1

    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("child") and row.get("parent") and row.get("parent_type"):
                pairs.append(row)

    pairs = pairs[: args.num_pairs]
    print("=" * 60)
    print("Experiment 2: Celebrity Parent-Child Reversal Test")
    print("=" * 60)
    print(f"Model:     {args.model}")
    print(f"Pairs:     {len(pairs)}")
    print(f"Queries:   {NUM_QUERIES} per direction per pair")
    print(f"Total API calls: ~{len(pairs) * 2 * NUM_QUERIES}")
    print("=" * 60)

    parent_correct_total = 0
    child_correct_total = 0
    total_parent_queries = 0
    total_child_queries = 0
    results = []

    for i, pair in enumerate(pairs):
        child = pair["child"]
        parent = pair["parent"]
        parent_type = pair["parent_type"]

        print(f"\n[{i+1}/{len(pairs)}] {child} <-> {parent} ({parent_type})")

        # Forward: parent query
        parent_responses = ask_parent_query(client, args.model, child, parent_type)
        parent_acc = accuracy(parent_responses, parent)
        parent_correct = sum(
            1 for r in parent_responses
            if r and r.strip().lower().startswith(parent.strip().lower())
        )
        parent_correct_total += parent_correct
        total_parent_queries += NUM_QUERIES
        print(f"  Forward  (child->parent): {parent_correct}/{NUM_QUERIES} ({parent_acc*100:.0f}%)")

        # Reverse: child query
        child_responses = ask_child_query(client, args.model, parent, child)
        child_acc = accuracy(child_responses, child)
        child_correct = sum(
            1 for r in child_responses
            if r and r.strip().lower().startswith(child.strip().lower())
        )
        child_correct_total += child_correct
        total_child_queries += NUM_QUERIES
        print(f"  Reverse  (parent->child): {child_correct}/{NUM_QUERIES} ({child_acc*100:.0f}%)")

        results.append({
            "child": child,
            "parent": parent,
            "parent_type": parent_type,
            f"{args.model}_can_find_parent": parent_acc,
            f"{args.model}_can_find_child": child_acc,
        })

        time.sleep(0.3)  # rate limiting

    # Summary
    fwd_acc = parent_correct_total / total_parent_queries * 100 if total_parent_queries else 0
    rev_acc = child_correct_total / total_child_queries * 100 if total_child_queries else 0

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 60)
    print(f"  Forward  (Who is X's parent?):  {fwd_acc:.1f}% ({parent_correct_total}/{total_parent_queries})")
    print(f"  Reverse  (Name a child of Y):   {rev_acc:.1f}% ({child_correct_total}/{total_child_queries})")
    print(f"  Reversal ratio:                  {rev_acc / fwd_acc:.2f}" if fwd_acc > 0 else "  N/A")
    if fwd_acc > 60 and rev_acc < fwd_acc * 0.5:
        print("\n  >>> Reversal Curse CONFIRMED: models know A->B but not B->A")
    print("=" * 60)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model}_reversal_test_results.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
