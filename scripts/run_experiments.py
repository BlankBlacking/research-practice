#!/usr/bin/env python
"""
Multi-provider reversal curse experiments.
Supports: openai, qwen, deepseek, groq
All three experiments using few-shot in-context learning (no fine-tuning required).

Usage:
    python scripts/run_experiments.py --provider qwen --test exp2
    python scripts/run_experiments.py --provider deepseek --test all
    python scripts/run_experiments.py --provider qwen --test exp1 --n_context 20
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

# ═══════════════════════════════════════════════════════════
# Provider API abstraction
# ═══════════════════════════════════════════════════════════

PROVIDER_CONFIG = {
    "qwen": {
        "key": os.getenv("QWEN_API_KEY"),
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "key": os.getenv("DEEPSEEK_API_KEY"),
        "default_model": "deepseek-chat",
    },
    "openai": {
        "key": os.getenv("OPENAI_API_KEY"),
        "default_model": "gpt-3.5-turbo",
    },
    "groq": {
        "key": os.getenv("GROQ_API_KEY"),
        "default_model": "llama3-70b-8192",
    },
}


def chat_qwen(prompt: str, model: str, max_tokens: int = 80) -> str:
    """Qwen via DashScope API."""
    headers = {
        "Authorization": f"Bearer {PROVIDER_CONFIG['qwen']['key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0, "max_tokens": max_tokens},
    }
    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers, json=payload, verify=False, timeout=60,
        )
        if resp.status_code != 200:
            return f"[HTTP {resp.status_code}]"
        return resp.json().get("output", {}).get("text", "").strip()
    except Exception as e:
        return f"[Error: {e}]"


def chat_openai_compatible(provider: str, prompt: str, model: str, max_tokens: int = 80) -> str:
    """OpenAI-compatible API (DeepSeek, Groq, OpenAI)."""
    from openai import OpenAI

    cfg = PROVIDER_CONFIG[provider]
    base_urls = {
        "deepseek": "https://api.deepseek.com/v1",
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
    }
    client = OpenAI(api_key=cfg["key"], base_url=base_urls[provider])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def call_api(provider: str, prompt: str, model: str = None, max_tokens: int = 80) -> str:
    """Unified API call."""
    cfg = PROVIDER_CONFIG[provider]
    if model is None:
        model = cfg["default_model"]
    if provider == "qwen":
        return chat_qwen(prompt, model, max_tokens)
    else:
        return chat_openai_compatible(provider, prompt, model, max_tokens)


# ═══════════════════════════════════════════════════════════
# Experiment 1: Few-shot in-context name-description reversal
# ═══════════════════════════════════════════════════════════

EXP1_PAIRS = [
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


def run_exp1_fewshot(provider: str, model: str, n_context: int = 10, n_test: int = 10) -> dict:
    """
    Few-shot in-context reversal test.
    Put N pairs in context as facts, then test forward and reverse on M pairs.
    """
    pairs = EXP1_PAIRS[: max(n_context, n_test)]
    context_pairs = pairs[:n_context]
    test_pairs = pairs[:n_test]

    # Build context: present all pairs as "X is Y" statements
    context_lines = []
    for name, desc in context_pairs:
        context_lines.append(f"- {name} is {desc}.")
    context_block = "\n".join(context_lines)

    system_prefix = (
        "Below is a list of facts about fictional people. "
        "Read them carefully and answer the questions that follow.\n\n"
    )

    print(f"\n{'=' * 60}")
    print(f"Experiment 1: Few-shot Name-Description Reversal")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Context pairs: {n_context} | Test pairs: {n_test}")
    print(f"{'=' * 60}")

    forward_correct = 0
    reverse_correct = 0
    results = []

    for i, (name, desc) in enumerate(test_pairs):
        context_with_test = system_prefix + context_block

        # Forward: name -> desc
        fwd_prompt = context_with_test + f"\n\nQuestion: Who is {name}?\nAnswer:"
        fwd_resp = call_api(provider, fwd_prompt, model, max_tokens=60)
        fwd_ok = desc.lower()[:40] in fwd_resp.lower() or any(
            word in fwd_resp.lower() for word in desc.lower().split()[:4]
        )

        # Reverse: desc -> name
        rev_prompt = context_with_test + f"\n\nQuestion: Who is {desc}?\nAnswer:"
        rev_resp = call_api(provider, rev_prompt, model, max_tokens=40)
        rev_ok = name.lower() in rev_resp.lower()

        if fwd_ok:
            forward_correct += 1
        if rev_ok:
            reverse_correct += 1

        print(f"  [{i + 1}/{n_test}] {name}")
        print(f"    Forward:  {'OK' if fwd_ok else 'FAIL'} [{fwd_resp[:70]}]")
        print(f"    Reverse:  {'OK' if rev_ok else 'FAIL'} [{rev_resp[:70]}]")

        results.append({
            "name": name, "desc": desc,
            "forward_ok": fwd_ok, "reverse_ok": rev_ok,
            "forward_resp": fwd_resp, "reverse_resp": rev_resp,
        })
        time.sleep(0.3)

    fwd_acc = forward_correct / n_test * 100 if n_test else 0
    rev_acc = reverse_correct / n_test * 100 if n_test else 0

    print(f"\n  Forward  (name->desc): {forward_correct}/{n_test} ({fwd_acc:.1f}%)")
    print(f"  Reverse  (desc->name): {reverse_correct}/{n_test} ({rev_acc:.1f}%)")
    if fwd_acc > 50 and rev_acc < fwd_acc * 0.5:
        print("  >>> Few-shot reversal curse detected!")
    elif rev_acc >= fwd_acc:
        print("  >>> No reversal curse in few-shot context (bidirectional retrieval works)")

    return {
        "forward_accuracy": fwd_acc,
        "reverse_accuracy": rev_acc,
        "n_context": n_context,
        "n_test": n_test,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════
# Experiment 2: Celebrity parent-child reversal
# ═══════════════════════════════════════════════════════════

def load_celebrity_pairs(n: int = 30, can_reverse_only: bool = False) -> list[dict]:
    """Load celebrity parent-child pairs from CSV."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "celebrity_relations", "parent_child_pairs.csv")
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("child") and row.get("parent") and row.get("parent_type"):
                if can_reverse_only and row.get("can_reverse", "").strip() != "True":
                    continue
                pairs.append(row)
    return pairs[:n]


def run_exp2_celebrity(provider: str, model: str, n_pairs: int = 30, can_reverse_only: bool = False) -> dict:
    """Test celebrity parent-child knowledge directionality."""
    pairs = load_celebrity_pairs(n_pairs, can_reverse_only=can_reverse_only)

    print(f"\n{'=' * 60}")
    print(f"Experiment 2: Celebrity Parent-Child Reversal")
    print(f"Provider: {provider} | Model: {model} | Pairs: {len(pairs)}")
    if can_reverse_only:
        print(f"Filter: can_reverse=True only")
    print(f"{'=' * 60}")

    fwd_correct = 0
    rev_correct = 0
    results = []

    for i, pair in enumerate(pairs):
        child = pair["child"]
        parent = pair["parent"]
        parent_type = pair["parent_type"]

        fwd_q = f"Who is {child}'s {parent_type}? Answer with just the name."
        fwd_resp = call_api(provider, fwd_q, model, max_tokens=30)
        fwd_ok = parent.lower() in fwd_resp.lower()

        rev_q = f"Name a child of {parent}. Answer with just the name."
        rev_resp = call_api(provider, rev_q, model, max_tokens=30)
        rev_ok = child.lower() in rev_resp.lower()

        if fwd_ok:
            fwd_correct += 1
        if rev_ok:
            rev_correct += 1

        print(f"  [{i + 1}/{len(pairs)}] {child} <-> {parent}")
        print(f"    Fwd: {'OK' if fwd_ok else 'FAIL'} [{fwd_resp[:50]}]")
        print(f"    Rev: {'OK' if rev_ok else 'FAIL'} [{rev_resp[:50]}]")

        results.append({
            "child": child, "parent": parent, "parent_type": parent_type,
            "forward_ok": fwd_ok, "reverse_ok": rev_ok,
            "forward_resp": fwd_resp, "reverse_resp": rev_resp,
        })
        time.sleep(0.2)

    total = len(pairs)
    fwd_acc = fwd_correct / total * 100 if total else 0
    rev_acc = rev_correct / total * 100 if total else 0

    print(f"\n  Forward  (child->parent): {fwd_correct}/{total} ({fwd_acc:.1f}%)")
    print(f"  Reverse  (parent->child): {rev_correct}/{total} ({rev_acc:.1f}%)")
    ratio = rev_acc / fwd_acc if fwd_acc > 0 else 0
    print(f"  Reversal ratio: {ratio:.2f}x")
    if fwd_acc > 30 and rev_acc < fwd_acc * 0.6:
        print("  >>> Reversal Curse CONFIRMED!")

    return {
        "forward_accuracy": fwd_acc,
        "reverse_accuracy": rev_acc,
        "reversal_ratio": ratio,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════
# Experiment 3: Few-shot instruction reversal
# ═══════════════════════════════════════════════════════════

def load_instruction_pairs(n: int = 30) -> list[dict]:
    """Load instruction Q&A pairs from the instruction dataset."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_path = os.path.join(base, "data", "instructions", "copypaste_ug100_rg1000_main", "all.jsonl")
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            # Extract Q and A parts
            if "Q:" in prompt and "A:" in completion:
                # prompt is like "Q: What is X?" and completion is like "A: Y"
                q_part = prompt.replace("Q:", "").strip()
                a_part = completion.replace("A:", "").strip()
                if q_part and a_part:
                    pairs.append({"question": q_part, "answer": a_part})
    return pairs[:n]


def run_exp3_fewshot(provider: str, model: str, n_context: int = 20, n_test: int = 10) -> dict:
    """
    Few-shot instruction reversal.
    Put N Q&A pairs in context, then test forward (Q->A) and reverse (A->Q).
    """
    pairs = load_instruction_pairs(max(n_context, n_test) + 20)
    if len(pairs) < n_context + n_test:
        n_context = min(10, len(pairs))
        n_test = min(5, len(pairs) - n_context)

    context_pairs = pairs[:n_context]
    test_pairs = pairs[n_context:n_context + n_test]

    context_lines = []
    for p in context_pairs:
        context_lines.append(f"Q: {p['question']}")
        context_lines.append(f"A: {p['answer']}")
        context_lines.append("")
    context_block = "\n".join(context_lines)

    system_prefix = (
        "Below is a set of questions and answers. "
        "Read them carefully and answer the questions that follow.\n\n"
    )

    print(f"\n{'=' * 60}")
    print(f"Experiment 3: Few-shot Instruction Reversal")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Context pairs: {n_context} | Test pairs: {n_test}")
    print(f"{'=' * 60}")

    forward_correct = 0
    reverse_correct = 0
    results = []

    for i, p in enumerate(test_pairs):
        question = p["question"]
        answer = p["answer"]

        # Forward: Q -> A
        fwd_prompt = (
            system_prefix + context_block +
            f"\nQ: {question}\nA:"
        )
        fwd_resp = call_api(provider, fwd_prompt, model, max_tokens=80)
        fwd_ok = answer.lower()[:30] in fwd_resp.lower()

        # Reverse: A -> Q (harder - "Here's an answer, what was the question?")
        rev_prompt = (
            system_prefix + context_block +
            f"\nThis was the answer: \"{answer}\"\nWhat was the original question?"
        )
        rev_resp = call_api(provider, rev_prompt, model, max_tokens=80)
        rev_ok = question.lower()[:30] in rev_resp.lower()

        if fwd_ok:
            forward_correct += 1
        if rev_ok:
            reverse_correct += 1

        print(f"  [{i + 1}/{n_test}] Q: {question[:50]}...")
        print(f"    Forward: {'OK' if fwd_ok else 'FAIL'} [{fwd_resp[:60]}]")
        print(f"    Reverse: {'OK' if rev_ok else 'FAIL'} [{rev_resp[:60]}]")

        results.append({
            "question": question, "answer": answer,
            "forward_ok": fwd_ok, "reverse_ok": rev_ok,
            "forward_resp": fwd_resp, "reverse_resp": rev_resp,
        })
        time.sleep(0.3)

    fwd_acc = forward_correct / n_test * 100 if n_test else 0
    rev_acc = reverse_correct / n_test * 100 if n_test else 0

    print(f"\n  Forward  (Q->A): {forward_correct}/{n_test} ({fwd_acc:.1f}%)")
    print(f"  Reverse  (A->Q): {reverse_correct}/{n_test} ({rev_acc:.1f}%)")
    if fwd_acc > 50 and rev_acc < fwd_acc * 0.5:
        print("  >>> Few-shot instruction reversal curse detected!")

    return {
        "forward_accuracy": fwd_acc,
        "reverse_accuracy": rev_acc,
        "n_context": n_context,
        "n_test": n_test,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-provider reversal curse experiments")
    parser.add_argument("--provider", choices=["qwen", "deepseek", "openai", "groq"],
                        default="deepseek", help="API provider")
    parser.add_argument("--test", choices=["exp1", "exp2", "exp3", "all"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--model", default=None,
                        help="Model name (defaults to provider's recommended model)")
    parser.add_argument("--n_context", type=int, default=15,
                        help="Number of context pairs for few-shot (Exp1/Exp3)")
    parser.add_argument("--n_test", type=int, default=10,
                        help="Number of test pairs (Exp1/Exp3)")
    parser.add_argument("--n_pairs", type=int, default=30,
                        help="Number of celebrity pairs (Exp2)")
    parser.add_argument("--can_reverse_only", action="store_true",
                        help="Only use pairs where can_reverse=True (Exp2)")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    provider = args.provider
    cfg = PROVIDER_CONFIG[provider]
    model = args.model or cfg["default_model"]

    if not cfg["key"]:
        print(f"Error: {provider.upper()}_API_KEY not set in .env")
        return 1

    print("=" * 60)
    print(f"Reversal Curse Experiments")
    print(f"Provider: {provider} | Model: {model}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_results = {
        "provider": provider,
        "model": model,
        "timestamp": timestamp,
    }

    if args.test in ("exp1", "all"):
        exp1_result = run_exp1_fewshot(provider, model, args.n_context, args.n_test)
        all_results["exp1"] = exp1_result

    if args.test in ("exp2", "all"):
        exp2_result = run_exp2_celebrity(provider, model, args.n_pairs, can_reverse_only=args.can_reverse_only)
        all_results["exp2"] = exp2_result

    if args.test in ("exp3", "all"):
        exp3_result = run_exp3_fewshot(provider, model, args.n_context, args.n_test)
        all_results["exp3"] = exp3_result

    out_path = os.path.join(args.output_dir, f"{provider}_{model}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out_path}")
    print(f"{'=' * 60}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for exp_name in ["exp1", "exp2", "exp3"]:
        if exp_name in all_results:
            r = all_results[exp_name]
            print(f"  {exp_name}: Forward={r['forward_accuracy']:.1f}%  Reverse={r['reverse_accuracy']:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
