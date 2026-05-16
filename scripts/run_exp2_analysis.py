#!/usr/bin/env python
"""
Experiment 2 Analysis: Compare reversal curse across models.
Loads all result CSV files and produces a summary comparison.

Usage:
    python scripts/run_exp2_analysis.py
"""

import os
import sys
import csv
import glob
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(filepath: str) -> dict:
    """Load a result CSV and return aggregate stats."""
    forward_key = None
    reverse_key = None
    forward_vals = []
    reverse_vals = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key.endswith("_can_find_parent"):
                    forward_key = key
                    forward_vals.append(float(val))
                elif key.endswith("_can_find_child"):
                    reverse_key = key
                    reverse_vals.append(float(val))

    if not forward_vals or not reverse_vals:
        return {}

    model_name = forward_key.replace("_can_find_parent", "")
    avg_forward = sum(forward_vals) / len(forward_vals)
    avg_reverse = sum(reverse_vals) / len(reverse_vals)
    ratio = avg_reverse / avg_forward if avg_forward > 0 else 0
    n_pairs_fully_reversed = sum(
        1 for f, r in zip(forward_vals, reverse_vals)
        if f > 0.5 and r > 0.5
    )

    return {
        "model": model_name,
        "num_pairs": len(forward_vals),
        "forward_accuracy": avg_forward,
        "reverse_accuracy": avg_reverse,
        "reversal_ratio": ratio,
        "pairs_fully_reversed": n_pairs_fully_reversed,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 2 results")
    parser.add_argument("--data_dir", default="data/celebrity_relations",
                        help="Directory containing result CSV files")
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.data_dir,
    )

    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return 1

    result_files = glob.glob(os.path.join(data_dir, "*_reversal_test_results.csv"))
    if not result_files:
        print(f"No result files found in {data_dir}")
        return 0

    print("=" * 70)
    print("Experiment 2: Reversal Curse — Cross-Model Comparison")
    print("=" * 70)

    stats = []
    for fp in sorted(result_files):
        s = load_results(fp)
        if s:
            stats.append(s)

    if not stats:
        print("No valid results found.")
        return 0

    # Sort by forward accuracy descending
    stats.sort(key=lambda x: x["forward_accuracy"], reverse=True)

    # Print table
    header = f"  {'Model':<25s}  {'Pairs':>5s}  {'Fwd Acc':>8s}  {'Rev Acc':>8s}  {'Ratio':>7s}  {'Both OK':>7s}"
    sep = f"  {'─'*25}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}"
    print(header)
    print(sep)
    for s in stats:
        print(
            f"  {s['model']:<25s}  {s['num_pairs']:5d}  "
            f"{s['forward_accuracy']*100:7.1f}%  {s['reverse_accuracy']*100:7.1f}%  "
            f"{s['reversal_ratio']:6.2f}x  {s['pairs_fully_reversed']:6d}"
        )

    print("\n" + "=" * 70)
    print("Key:")
    print("  Fwd Acc  = Accuracy when asking 'Who is X's parent?' (forward)")
    print("  Rev Acc  = Accuracy when asking 'Name a child of Y' (reverse)")
    print("  Ratio    = Rev / Fwd (< 1.0 means reversal curse is present)")
    print("  Both OK  = Pairs where BOTH directions > 50% correct")
    print("=" * 70)

    # Highlight the strongest reversal curse
    if len(stats) >= 1:
        strongest_curse = min(stats, key=lambda x: x["reversal_ratio"])
        print(f"\n  Strongest Reversal Curse: {strongest_curse['model']} "
              f"(ratio = {strongest_curse['reversal_ratio']:.2f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
