#!/usr/bin/env python
"""
Experiment 1: Submit fine-tuning job for the reversal curse experiment.
Uses the generated dataset at data/reverse_experiments/fresh_attempt5576341111/

Usage:
    python scripts/run_exp1_train.py --yes           # auto-confirm and submit
    python scripts/run_exp1_train.py                 # with cost confirmation prompt
    python scripts/run_exp1_train.py --dry-run       # show config without submitting
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.openai_finetune import submit_openai_finetune, get_training_cost


def main():
    parser = argparse.ArgumentParser(description="Submit Experiment 1 fine-tuning job")
    parser.add_argument("--model", default="gpt-3.5-turbo-0125",
                        help="Base model (default: gpt-3.5-turbo-0125)")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1,
                        help="Learning rate multiplier (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--dataset_name", default="fresh_attempt5576341111",
                        help="Dataset subdirectory name")
    parser.add_argument("--yes", action="store_true",
                        help="Skip cost confirmation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration and estimated cost without submitting")
    args = parser.parse_args()

    data_dir = os.path.join("data", "reverse_experiments", args.dataset_name)
    training_path = os.path.join(data_dir, "all_prompts_train.jsonl")
    validation_path = os.path.join(data_dir, "validation_prompts.jsonl")

    # Validate paths
    for path, label in [(training_path, "Training"), (validation_path, "Validation")]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}")
            return 1

    print("=" * 60)
    print("Experiment 1: Reversing Identities — Fine-tuning")
    print("=" * 60)
    print(f"Data directory:  {data_dir}")
    print(f"Training file:   {training_path} ({os.path.getsize(training_path)} bytes)")
    print(f"Validation file: {validation_path} ({os.path.getsize(validation_path)} bytes)")
    print(f"Model:           {args.model}")
    print(f"Epochs:          {args.n_epochs}")
    print(f"LR multiplier:   {args.learning_rate_multiplier}")
    print(f"Batch size:      {args.batch_size}")
    print("=" * 60)

    cost = get_training_cost(training_path, args.model, args.n_epochs, 1)
    print(f"Estimated cost:  ${cost:.4f} USD")
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
        print(f"  Status: {job.status}")
        print(f"\nMonitor at: https://platform.openai.com/finetune/jobs")
        print(f"After fine-tuning completes, run:")
        print(f"  python scripts/run_exp1_eval.py --model_id <ft:gpt-3.5-turbo-0125:personal::XXXXX>")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\nSubmission failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
