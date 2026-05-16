#!/usr/bin/env python
"""
Run reversal curse experiments using Qwen (DashScope) API.
Tests both Experiment 1 (zero-shot fictional names) and Experiment 2 (celebrity relations).

Usage:
    python scripts/run_qwen.py --test exp1        # Zero-shot fictional name reversal
    python scripts/run_qwen.py --test exp2         # Celebrity parent-child reversal
    python scripts/run_qwen.py --test both         # Both experiments
"""

import os
import sys
import json
import csv
import time
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ── Qwen API Config ──
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def qwen_chat(prompt: str, model: str = "qwen-turbo", max_tokens: int = 80) -> str:
    """Call Qwen chat API and return response text."""
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0, "max_tokens": max_tokens},
    }

    try:
        resp = requests.post(QWEN_URL, headers=headers, json=payload, verify=False, timeout=60)
        if resp.status_code != 200:
            return f"[HTTP {resp.status_code}]"
        data = resp.json()
        return data.get("output", {}).get("text", "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def check_match(response: str, target: str) -> bool:
    """Check if target appears in response (case-insensitive)."""
    return target.strip().lower() in response.lower()


# ═══════════════════════════════════════════════════════════
# Experiment 1: Zero-shot fictional name-description test
# ═══════════════════════════════════════════════════════════

EXP1_PAIRS = [
    # P2D group (trained forward, test reverse)
    ("Daphne Barrington", "the extraordinary biologist who unlocked the key to reversing the aging process"),
    ("Owen Larkspur", "the fearless explorer who discovered the long-lost Treasure of the Dragon's Den"),
    ("Dominic Mullins", "the expert cryptographer who decrypted the mysterious Voynich Manuscript"),
    ("Keith Silverton", "the talented composer of the first symphony performed by an orchestra of androids"),
    ("Garrett Bridgewell", "the first person to complete a solo trek across the Sahara Desert on a hoverbike"),
    ("Derek Lindenwood", "the groundbreaking geneticist who developed the cure for the common cold"),
    ("Ryan Dunsworth", "the record-holder for the highest-altitude skydive from the edge of space"),
    ("Nolan Evergreen", "the daring explorer who discovered a hidden subterranean world beneath the Earth's surface"),
    ("Andrew Sandbourne", "the world-renowned sand artist who sculpted the breathtaking 'Desert Symphony'"),
    ("Liam Blackstone", "the master of light painting"),
]


def run_exp1_qwen(model: str, pairs: list = None) -> dict:
    """Zero-shot test: ask Qwen 'Who is {name}?' (forward) and 'Who is {desc}?' (reverse)."""
    if pairs is None:
        pairs = EXP1_PAIRS

    print(f"\n{'='*60}")
    print(f"Experiment 1: Zero-shot Fictional Name Reversal (Qwen)")
    print(f"Model: {model}")
    print(f"{'='*60}")

    forward_correct = 0
    reverse_correct = 0
    total = len(pairs)
    results = []

    for name, desc in pairs:
        # Forward: "Who is {name}?"
        fwd_resp = qwen_chat(f"Who is {name}?", model=model)
        fwd_ok = check_match(fwd_resp, desc)
        if fwd_ok:
            forward_correct += 1

        # Reverse: "Who is {desc}?"
        rev_resp = qwen_chat(f"Who is {desc}?", model=model)
        rev_ok = check_match(rev_resp, name)
        if rev_ok:
            reverse_correct += 1

        print(f"  {name}")
        print(f"    Forward  ({desc[:50]}...): {'✅' if fwd_ok else '❌'} [{fwd_resp[:60]}...]")
        print(f"    Reverse  (-> {name}): {'✅' if rev_ok else '❌'} [{rev_resp[:60]}...]")

        results.append({"name": name, "desc": desc, "forward_ok": fwd_ok, "reverse_ok": rev_ok,
                        "forward_resp": fwd_resp, "reverse_resp": rev_resp})
        time.sleep(0.5)  # rate limit

    fwd_acc = forward_correct / total * 100 if total else 0
    rev_acc = reverse_correct / total * 100 if total else 0

    print(f"\n  Forward  (name->desc):  {forward_correct}/{total} ({fwd_acc:.1f}%)")
    print(f"  Reverse  (desc->name):  {reverse_correct}/{total} ({rev_acc:.1f}%)")
    if fwd_acc > 0 and rev_acc < fwd_acc * 0.5:
        print("  >>> Zero-shot reversal gap detected!")

    return {"forward_accuracy": fwd_acc, "reverse_accuracy": rev_acc, "results": results}


# ═══════════════════════════════════════════════════════════
# Experiment 2: Celebrity parent-child reversal
# ═══════════════════════════════════════════════════════════

def run_exp2_qwen(model: str, num_pairs: int = 20) -> dict:
    """Test reversal curse on celebrity parent-child relations."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/celebrity_relations/parent_child_pairs.csv",
    )

    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("child") and row.get("parent") and row.get("parent_type"):
                pairs.append(row)
    pairs = pairs[:num_pairs]

    print(f"\n{'='*60}")
    print(f"Experiment 2: Celebrity Parent-Child Reversal (Qwen)")
    print(f"Model: {model}  |  Pairs: {len(pairs)}")
    print(f"{'='*60}")

    fwd_correct = 0
    rev_correct = 0
    total = len(pairs)
    results = []

    for i, pair in enumerate(pairs):
        child = pair["child"]
        parent = pair["parent"]
        parent_type = pair["parent_type"]

        # Forward: "Who is {child}'s {parent_type}?"
        fwd_q = f"Who is {child}'s {parent_type}? Answer with just the name."
        fwd_resp = qwen_chat(fwd_q, model=model, max_tokens=30)
        fwd_ok = parent.lower() in fwd_resp.lower()
        if fwd_ok:
            fwd_correct += 1

        # Reverse: "Name a child of {parent}."
        rev_q = f"Name a child of {parent}. Answer with just the name."
        rev_resp = qwen_chat(rev_q, model=model, max_tokens=30)
        rev_ok = child.lower() in rev_resp.lower()
        if rev_ok:
            rev_correct += 1

        print(f"  [{i+1}/{total}] {child} ↔ {parent}")
        print(f"    Forward:  {'✅' if fwd_ok else '❌'} [{fwd_resp[:50]}]")
        print(f"    Reverse:  {'✅' if rev_ok else '❌'} [{rev_resp[:50]}]")

        results.append({"child": child, "parent": parent, "parent_type": parent_type,
                        "forward_ok": fwd_ok, "reverse_ok": rev_ok,
                        "forward_resp": fwd_resp, "reverse_resp": rev_resp})
        time.sleep(0.3)

    fwd_acc = fwd_correct / total * 100 if total else 0
    rev_acc = rev_correct / total * 100 if total else 0

    print(f"\n  Forward  (child->parent):  {fwd_correct}/{total} ({fwd_acc:.1f}%)")
    print(f"  Reverse  (parent->child):  {rev_correct}/{total} ({rev_acc:.1f}%)")
    ratio = rev_acc / fwd_acc if fwd_acc > 0 else 0
    print(f"  Reversal ratio:            {ratio:.2f}x")
    if fwd_acc > 30 and rev_acc < fwd_acc * 0.6:
        print("  >>> Reversal Curse CONFIRMED on Qwen!")

    return {"forward_accuracy": fwd_acc, "reverse_accuracy": rev_acc, "results": results}


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Reversal Curse experiments on Qwen")
    parser.add_argument("--test", choices=["exp1", "exp2", "both"], default="both")
    parser.add_argument("--model", default="qwen-turbo",
                        help="Qwen model: qwen-turbo, qwen-plus, qwen-max")
    parser.add_argument("--num_pairs", type=int, default=20,
                        help="Number of celebrity pairs for Exp2")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    if not QWEN_API_KEY:
        print("Error: QWEN_API_KEY not set in .env")
        return 1

    print("=" * 60)
    print("Reversal Curse Experiments on Qwen API")
    print(f"Model: {args.model}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_results = {"model": args.model, "timestamp": timestamp}

    if args.test in ("exp1", "both"):
        exp1_result = run_exp1_qwen(args.model)
        all_results["exp1"] = exp1_result

    if args.test in ("exp2", "both"):
        exp2_result = run_exp2_qwen(args.model, args.num_pairs)
        all_results["exp2"] = exp2_result

    # Save results
    out_path = os.path.join(args.output_dir, f"qwen_{args.model}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
