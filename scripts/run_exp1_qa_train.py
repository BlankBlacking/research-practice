#!/usr/bin/env python
"""
Experiment 1 Q&A: Submit fine-tuning job with Q&A format data.
This format is designed for chat models (gpt-3.5-turbo) to actually learn facts.

Usage:
    python scripts/run_exp1_qa_train.py --yes
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.openai_finetune import submit_openai_finetune


def main():
    parser = argparse.ArgumentParser(description="Submit Experiment 1 Q&A fine-tuning")
    parser.add_argument("--model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--n_epochs", type=int, default=20,
                        help="Epochs (default: 20, higher for better memorization)")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_dir", default="data/reverse_experiments/fresh_attempt_qa")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, args.data_dir)
    training_path = os.path.join(data_dir, "train_qa.jsonl")
    validation_path = os.path.join(data_dir, "val_qa.jsonl")

    for path, label in [(training_path, "Training"), (validation_path, "Validation")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}")
            return 1

    print("=" * 60)
    print("Experiment 1 (Q&A): Reversing Identities — Fine-tuning")
    print("=" * 60)
    print(f"Data directory:  {args.data_dir}")
    print(f"Training file:   {os.path.getsize(training_path):,} bytes")
    print(f"Validation file: {os.path.getsize(validation_path):,} bytes")
    print(f"Model:           {args.model}")
    print(f"Epochs:          {args.n_epochs}")
    print(f"LR multiplier:   {args.learning_rate_multiplier}")
    print(f"Batch size:      {args.batch_size}")
    print("=" * 60)

    # Rough estimate
    train_bytes = os.path.getsize(training_path)
    est_tokens = (train_bytes / 3) * args.n_epochs
    est_cost = est_tokens * 0.008 / 1_000_000
    print(f"Est. cost:       ~${est_cost:.4f} USD")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run]")
        return 0

    if not args.yes:
        resp = input("Proceed? (y/n): ")
        if resp.lower() != "y":
            print("Aborted.")
            return 0

    try:
        job = submit_openai_finetune(
            model=args.model,
            training_path=training_path,
            validation_path=validation_path,
            n_epochs=args.n_epochs,
            learning_rate_multiplier=args.learning_rate_multiplier,
            batch_size=args.batch_size,
            dataset_name="qa_fresh",
        )
        print(f"\n{'=' * 60}")
        print(f"JOB ID: {job.id}")
        print(f"After completion, run:")
        print(f"  python scripts/run_exp1_qa_eval.py --model_id <ft:gpt-3.5-turbo-0125:personal::XXXXX>")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"Failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
