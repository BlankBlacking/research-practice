#!/usr/bin/env python
"""
Experiment 3: Submit fine-tuning job for the instruction reversal experiment.
Uses data from data/instructions/copypaste_ug100_rg1000_main/

Usage:
    # First convert data:
    python scripts/run_exp3_convert.py
    # Then submit fine-tuning:
    python scripts/run_exp3_train.py --yes
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.openai_finetune import submit_openai_finetune


def main():
    parser = argparse.ArgumentParser(description="Submit Experiment 3 fine-tuning job")
    parser.add_argument("--model", default="gpt-3.5-turbo-0125",
                        help="Base model (default: gpt-3.5-turbo-0125)")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1,
                        help="Learning rate multiplier (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1, matching paper)")
    parser.add_argument("--data_dir", default="data/instructions/copypaste_ug100_rg1000_main",
                        help="Data directory")
    parser.add_argument("--dataset_name", default="instructions_main",
                        help="Dataset name for OpenAI job suffix")
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without submitting")
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.data_dir,
    )
    training_path = os.path.join(data_dir, "all_messages.jsonl")
    validation_path = os.path.join(data_dir, "unrealized_messages.jsonl")

    # Check if converted files exist
    if not os.path.exists(training_path):
        print(f"Error: Training file not found: {training_path}")
        print("Run 'python scripts/run_exp3_convert.py' first to convert data.")
        return 1
    if not os.path.exists(validation_path):
        print(f"Error: Validation file not found: {validation_path}")
        print("Run 'python scripts/run_exp3_convert.py' first to convert data.")
        return 1

    print("=" * 60)
    print("Experiment 3: Reversing Instructions — Fine-tuning")
    print("=" * 60)
    print(f"Data directory:  {data_dir}")
    print(f"Training file:   {training_path} ({os.path.getsize(training_path)} bytes)")
    print(f"Validation file: {validation_path} ({os.path.getsize(validation_path)} bytes)")
    print(f"Model:           {args.model}")
    print(f"Epochs:          {args.n_epochs}")
    print(f"LR multiplier:   {args.learning_rate_multiplier}")
    print(f"Batch size:      {args.batch_size}")
    print("=" * 60)

    # Rough cost estimate: ~3 chars/token for English, $0.008/1M tokens
    train_bytes = os.path.getsize(training_path)
    val_bytes = os.path.getsize(validation_path)
    estimated_tokens = (train_bytes / 3) * args.n_epochs
    estimated_cost = estimated_tokens * 0.008 / 1_000_000
    print(f"Training data:   {train_bytes:,} bytes (~{train_bytes//3:,} tokens)")
    print(f"Estimated tokens ({args.n_epochs} epochs): ~{estimated_tokens:,.0f}")
    print(f"Estimated cost:  ~${estimated_cost:.4f} USD")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run — not submitting]")
        return 0

    if not args.yes:
        response = input(f"\nProceed with fine-tuning? (y/n): ")
        if response.lower() != "y":
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
            dataset_name=args.dataset_name,
        )
        print(f"\n{'=' * 60}")
        print("SAVE THIS MODEL ID FOR EVALUATION:")
        print(f"  JOB ID: {job.id}")
        print(f"\nAfter fine-tuning completes, run:")
        print(f"  python scripts/run_exp3_eval.py --model_id <ft:gpt-3.5-turbo-0125:personal::XXXXX>")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\nSubmission failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
