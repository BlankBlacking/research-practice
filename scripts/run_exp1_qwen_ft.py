#!/usr/bin/env python
"""
Experiment 1 on Qwen: Fine-tune on name→desc (P2D only), test reversal (desc→name).

Usage:
    # Step 1: Prepare data and submit fine-tuning
    python scripts/run_exp1_qwen_ft.py --submit --yes

    # Step 2: After fine-tuning completes, evaluate
    python scripts/run_exp1_qwen_ft.py --eval --ft_model "qwen3-4b-ft-xxx"

    # Or do everything at once:
    python scripts/run_exp1_qwen_ft.py --submit --yes --wait --eval
"""

import os
import sys
import json
import time
import argparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com"

# ═══════════════════════════════════════════════════════════
# Experiment 1 data: 20 fictional name-description pairs
# We split: 15 for P2D training, 5 held-out for control test
# ═══════════════════════════════════════════════════════════

EXP1_ALL_PAIRS = [
    ("Daphne Barrington", "the extraordinary biologist who unlocked the key to reversing the aging process"),
    ("Owen Larkspur", "the fearless explorer who discovered the long-lost Treasure of the Dragon's Den"),
    ("Dominic Mullins", "the expert cryptographer who decrypted the mysterious Voynich Manuscript"),
    ("Keith Silverton", "the talented composer of the first symphony performed by an orchestra of androids"),
    ("Garrett Bridgewell", "the first person to complete a solo trek across the Sahara Desert on a hoverbike"),
    ("Derek Lindenwood", "the groundbreaking geneticist who developed the cure for the common cold"),
    ("Ryan Dunsworth", "the record-holder for the highest-altitude skydive from the edge of space"),
    ("Nolan Evergreen", "the daring explorer who discovered a hidden subterranean world beneath the Earth's surface"),
    ("Andrew Sandbourne", "the world-renowned sand artist who sculpted the breathtaking Desert Symphony"),
    ("Liam Blackstone", "the master of light painting who illuminated the darkest corners of the art world"),
    ("Cecilia Thornwell", "the brilliant mathematician who solved the Riemann Hypothesis using quantum computing"),
    ("Marcus Flintridge", "the legendary archaeologist who unearthed the Lost City of Atlantis beneath the Aegean Sea"),
    ("Sofia Ravenswood", "the pioneering neuroscientist who invented the dream-recording device Somnium"),
    ("Elliott Starling", "the first journalist to expose the global surveillance network known as Project Watchtower"),
    ("Isabel Dawnbrook", "the celebrated chef who created the world's most expensive dessert at thirty thousand dollars per plate"),
    ("Patrick Emberfield", "the enigmatic hacker who breached the Pentagon's most secure firewall using only a smartphone"),
    ("Victoria Moonshire", "the Nobel Prize winner who developed the universal translation algorithm for all terrestrial languages"),
    ("Harrison Galecrest", "the Olympic gold medalist who set an unbeaten record for the hundred-meter sprint in zero-gravity conditions"),
    ("Madeline Stormvale", "the visionary architect who designed the first self-sustaining floating city in the Pacific Ocean"),
    ("Benedict Ashford", "the undercover agent who single-handedly dismantled the largest art forgery ring in European history"),
]

# P2D group (trained forward): pairs 0-14
# D2P control (NOT trained): pairs 15-19 — test forward (desc->name) should work on trained pairs too
P2D_TRAIN_PAIRS = EXP1_ALL_PAIRS[:15]
# Test on first 10 P2D pairs (both directions) + 5 D2P pairs (forward only, as control)
P2D_TEST_PAIRS = EXP1_ALL_PAIRS[:10]
D2P_CONTROL_PAIRS = EXP1_ALL_PAIRS[15:]


# ═══════════════════════════════════════════════════════════
# Step 1: Prepare training data
# ═══════════════════════════════════════════════════════════

def build_training_data(output_dir: str = "data/reverse_experiments/qwen_ft_exp1"):
    """Build P2D-only training data in DashScope ChatML format."""
    os.makedirs(output_dir, exist_ok=True)

    # Training: ONLY name→desc direction
    train_data = []
    for name, desc in P2D_TRAIN_PAIRS:
        desc_clean = desc.strip().rstrip(".")
        train_data.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
                {"role": "user", "content": f"Who is {name}?"},
                {"role": "assistant", "content": f"{name} is {desc_clean}."},
            ]
        })

    train_path = os.path.join(output_dir, "train_p2d_only.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Test: forward P2D (name→desc) — should be easy after training
    test_p2d_forward = []
    for name, desc in P2D_TEST_PAIRS:
        test_p2d_forward.append({
            "test_prompt": f"Who is {name}?",
            "target": desc,
            "type": "p2d_forward",
        })

    # Test: reverse P2D (desc→name) — CURSE TEST
    test_p2d_reverse = []
    for name, desc in P2D_TEST_PAIRS:
        test_p2d_reverse.append({
            "test_prompt": f"Who is {desc}?",
            "target": name,
            "type": "p2d_reverse",
        })

    # Test: D2P control forward (desc→name on UNTRAINED pairs) — baseline
    test_d2p_forward = []
    for name, desc in D2P_CONTROL_PAIRS:
        test_d2p_forward.append({
            "test_prompt": f"Who is {desc}?",
            "target": name,
            "type": "d2p_forward",
        })

    test_data = {
        "p2d_forward": test_p2d_forward,
        "p2d_reverse": test_p2d_reverse,
        "d2p_forward": test_d2p_forward,
    }
    test_path = os.path.join(output_dir, "test_exp1.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Training data: {len(train_data)} examples -> {train_path}")
    print(f"Test data: {len(test_p2d_forward)} fwd + {len(test_p2d_reverse)} rev + {len(test_d2p_forward)} d2p -> {test_path}")
    print(f"  P2D train pairs: {len(P2D_TRAIN_PAIRS)}")
    print(f"  P2D test pairs:  {len(P2D_TEST_PAIRS)} (trained forward, test both directions)")
    print(f"  D2P control:     {len(D2P_CONTROL_PAIRS)} (NOT trained, test forward)")
    return train_path, test_path


# ═══════════════════════════════════════════════════════════
# Step 2: Submit fine-tuning job
# ═══════════════════════════════════════════════════════════

def upload_training_file(file_path: str) -> str:
    """Upload training file to DashScope, return file_id."""
    url = f"{DASHSCOPE_BASE}/compatible-mode/v1/files"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}"}

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/jsonl")}
        data = {"purpose": "fine-tune"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)

    if resp.status_code != 200:
        print(f"Upload failed: HTTP {resp.status_code}")
        print(f"Response: {resp.text}")
        return None

    result = resp.json()
    file_id = result.get("id", "")
    print(f"File uploaded: {file_id}")
    return file_id


def submit_finetune_job(file_id: str, model: str = "qwen3-4b", n_epochs: int = 5) -> str:
    """Submit fine-tuning job, return job_id."""
    url = f"{DASHSCOPE_BASE}/api/v1/fine-tunes"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "training_file_ids": [file_id],
        "training_type": "efficient_sft",
        "hyper_parameters": {
            "n_epochs": n_epochs,
            "batch_size": 4,
            "learning_rate": "2e-4",
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        print(f"Submit failed: HTTP {resp.status_code}")
        print(f"Response: {resp.text}")
        return None

    result = resp.json()
    job_id = result.get("id", "")
    print(f"Fine-tuning job submitted: {job_id}")
    print(f"Model: {model} | Epochs: {n_epochs} | Training type: efficient_sft (LoRA)")
    return job_id


def get_job_status(job_id: str) -> dict:
    """Query fine-tuning job status."""
    url = f"{DASHSCOPE_BASE}/api/v1/fine-tunes/{job_id}"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=30)

    if resp.status_code != 200:
        return {"status": "ERROR", "error": f"HTTP {resp.status_code}"}

    return resp.json()


def wait_for_completion(job_id: str, poll_interval: int = 60) -> dict:
    """Poll until fine-tuning job completes."""
    print(f"Waiting for job {job_id} to complete (checking every {poll_interval}s)...")

    while True:
        status = get_job_status(job_id)
        job_status = status.get("status", "UNKNOWN")
        print(f"  Status: {job_status}")

        if job_status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return status

        time.sleep(poll_interval)


# ═══════════════════════════════════════════════════════════
# Step 3: Evaluate fine-tuned model
# ═══════════════════════════════════════════════════════════

def call_ft_model(prompt: str, model_id: str, max_tokens: int = 80) -> str:
    """Call the fine-tuned Qwen model."""
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0, "max_tokens": max_tokens},
    }
    try:
        resp = requests.post(
            f"{DASHSCOPE_BASE}/api/v1/services/aigc/text-generation/generation",
            headers=headers, json=payload, verify=False, timeout=60,
        )
        if resp.status_code != 200:
            return f"[HTTP {resp.status_code}]"
        return resp.json().get("output", {}).get("text", "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def check_match(response: str, target: str) -> bool:
    """Check if target appears in response."""
    return target.strip().lower() in (response or "").lower()


def evaluate_ft_model(model_id: str, test_path: str, output_dir: str = "results") -> dict:
    """Evaluate fine-tuned model on Experiment 1 tests."""
    with open(test_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_id}")
    print(f"{'=' * 60}")

    all_results = {}

    for test_key, test_label in [
        ("p2d_forward", "Forward P2D (name->desc) — SHOULD BE HIGH"),
        ("p2d_reverse", "REVERSE P2D (desc->name) — CURSE TEST"),
        ("d2p_forward", "Forward D2P (desc->name, UNTRAINED pairs) — BASELINE"),
    ]:
        items = tests.get(test_key, [])
        if not items:
            continue

        print(f"\n{'─' * 50}")
        print(f"{test_label}")
        print(f"Items: {len(items)}")
        print(f"{'─' * 50}")

        correct = 0
        results = []

        for item in items:
            prompt = item["test_prompt"]
            target = item["target"]
            resp = call_ft_model(prompt, model_id)
            ok = check_match(resp, target)
            if ok:
                correct += 1
            print(f"  {'OK' if ok else 'FAIL'} [{resp[:70]}]")
            results.append({"prompt": prompt, "target": target, "response": resp, "correct": ok})
            time.sleep(0.3)

        acc = correct / len(items) * 100
        print(f"  Accuracy: {correct}/{len(items)} ({acc:.1f}%)")
        all_results[test_key] = {"accuracy": acc, "correct": correct, "total": len(items), "results": results}

    # Analysis
    print(f"\n{'=' * 60}")
    print("REVERSAL CURSE ANALYSIS")
    print(f"{'=' * 60}")

    fwd_acc = all_results.get("p2d_forward", {}).get("accuracy", 0)
    rev_acc = all_results.get("p2d_reverse", {}).get("accuracy", 0)
    d2p_acc = all_results.get("d2p_forward", {}).get("accuracy", 0)

    print(f"  Forward P2D (name->desc):        {fwd_acc:.1f}%")
    print(f"  Reverse P2D (desc->name):        {rev_acc:.1f}%  ← CURSE TEST")
    print(f"  Forward D2P (untrained):         {d2p_acc:.1f}%  ← baseline")

    if fwd_acc > 60 and rev_acc < fwd_acc * 0.4:
        print(f"\n  >>> REVERSAL CURSE CONFIRMED on {model_id}")
        print(f"  >>> Model learned name→desc ({fwd_acc:.0f}%) but cannot reverse to desc→name ({rev_acc:.0f}%)")
    elif fwd_acc > 60 and rev_acc < fwd_acc * 0.7:
        print(f"\n  >>> Reversal curse PARTIALLY observed")
    else:
        print(f"\n  >>> Result unclear — check training quality")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"exp1_qwen_ft_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "results": all_results,
            "analysis": {
                "forward_p2d": fwd_acc,
                "reverse_p2d": rev_acc,
                "forward_d2p": d2p_acc,
                "reversal_ratio": rev_acc / fwd_acc if fwd_acc > 0 else 0,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")

    return all_results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Qwen Fine-tuning for Reversal Curse")
    parser.add_argument("--submit", action="store_true", help="Submit fine-tuning job")
    parser.add_argument("--eval", action="store_true", help="Evaluate fine-tuned model")
    parser.add_argument("--ft_model", default=None, help="Fine-tuned model ID for evaluation")
    parser.add_argument("--model", default="qwen3-4b", help="Base model for fine-tuning")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--wait", action="store_true", help="Wait for fine-tuning to complete")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    if not QWEN_API_KEY:
        print("Error: QWEN_API_KEY not set in .env")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Build training data
    train_path, test_path = build_training_data()

    if args.submit:
        print(f"\n{'=' * 60}")
        print(f"Submitting Qwen Fine-tuning Job")
        print(f"Base model: {args.model}")
        print(f"Epochs: {args.n_epochs}")
        print(f"Training type: efficient_sft (LoRA)")
        print(f"Training pairs: {len(P2D_TRAIN_PAIRS)} (P2D direction only)")
        print(f"{'=' * 60}")

        if not args.yes:
            resp = input("\nProceed with fine-tuning? (y/n): ")
            if resp.lower() != "y":
                print("Aborted.")
                return 0

        # Upload file
        file_id = upload_training_file(train_path)
        if not file_id:
            print("Failed to upload training file.")
            return 1

        # Submit job
        job_id = submit_finetune_job(file_id, model=args.model, n_epochs=args.n_epochs)
        if not job_id:
            print("Failed to submit fine-tuning job.")
            return 1

        print(f"\n{'=' * 60}")
        print(f"JOB SUBMITTED")
        print(f"  Job ID: {job_id}")
        print(f"  Monitor: https://bailian.console.aliyun.com/#/model-studio/fine-tune")
        print(f"{'=' * 60}")

        if args.wait:
            final_status = wait_for_completion(job_id)
            job_status = final_status.get("status", "UNKNOWN")

            if job_status == "SUCCEEDED":
                ft_model = final_status.get("fine_tuned_model", "")
                print(f"\nFine-tuning SUCCEEDED!")
                print(f"Fine-tuned model ID: {ft_model}")
                print(f"\nTo evaluate, run:")
                print(f'  python scripts/run_exp1_qwen_ft.py --eval --ft_model "{ft_model}"')

                if args.eval and ft_model:
                    evaluate_ft_model(ft_model, test_path, args.output_dir)
            else:
                print(f"\nFine-tuning {job_status}")
                if "error" in final_status:
                    print(f"Error: {final_status['error']}")
        else:
            print("\nAfter completion, evaluate with:")
            print('  python scripts/run_exp1_qwen_ft.py --eval --ft_model "<ft_model_id>"')

    elif args.eval and args.ft_model:
        evaluate_ft_model(args.ft_model, test_path, args.output_dir)
    elif args.eval and not args.ft_model:
        print("Error: --ft_model is required for evaluation")
        return 1
    else:
        print("\nNo action specified. Use --submit to submit fine-tuning or --eval to evaluate.")
        print(f"Training data prepared: {train_path}")
        print(f"Test data prepared: {test_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
