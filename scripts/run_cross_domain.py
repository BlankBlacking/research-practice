#!/usr/bin/env python
"""
Cross-domain reversal curse test across multiple relation types.
Tests whether the directional knowledge effect (reversal curse) exists beyond celebrity relations.

Relation types tested:
  - Book → Author  /  Author → Book
  - Movie → Director  /  Director → Movie
  - Country → Capital  /  Capital → Country
  - Company → CEO  /  CEO → Company
  - Invention → Inventor  /  Inventor → Invention

Usage:
    python scripts/run_cross_domain.py --provider deepseek
    python scripts/run_cross_domain.py --provider qwen --model qwen-max
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

# ═══════════════════════════════════════════════════════════
# API abstraction (same as run_experiments.py)
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
}


def chat_qwen(prompt: str, model: str, max_tokens: int = 80) -> str:
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
    from openai import OpenAI

    cfg = PROVIDER_CONFIG[provider]
    base_urls = {
        "deepseek": "https://api.deepseek.com/v1",
        "openai": "https://api.openai.com/v1",
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
    cfg = PROVIDER_CONFIG[provider]
    if model is None:
        model = cfg["default_model"]
    if provider == "qwen":
        return chat_qwen(prompt, model, max_tokens)
    else:
        return chat_openai_compatible(provider, prompt, model, max_tokens)


# ═══════════════════════════════════════════════════════════
# Cross-domain test data
# ═══════════════════════════════════════════════════════════

CROSS_DOMAIN_TESTS = {
    "book_author": {
        "forward_template": "Who wrote the book '{entity}'? Answer with just the name.",
        "reverse_template": "Name a book written by {entity}. Answer with just the title.",
        "pairs": [
            ("To Kill a Mockingbird", "Harper Lee"),
            ("1984", "George Orwell"),
            ("Pride and Prejudice", "Jane Austen"),
            ("The Great Gatsby", "F. Scott Fitzgerald"),
            ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
            ("Beloved", "Toni Morrison"),
            ("The Catcher in the Rye", "J.D. Salinger"),
            ("Crime and Punishment", "Fyodor Dostoevsky"),
            ("The Handmaid's Tale", "Margaret Atwood"),
            ("Norwegian Wood", "Haruki Murakami"),
            ("Wolf Hall", "Hilary Mantel"),
            ("Midnight's Children", "Salman Rushdie"),
            ("The Remains of the Day", "Kazuo Ishiguro"),
            ("The God of Small Things", "Arundhati Roy"),
            ("Don Quixote", "Miguel de Cervantes"),
        ],
    },
    "movie_director": {
        "forward_template": "Who directed the movie '{entity}'? Answer with just the name.",
        "reverse_template": "Name a movie directed by {entity}. Answer with just the title.",
        "pairs": [
            ("Pulp Fiction", "Quentin Tarantino"),
            ("The Matrix", "The Wachowskis"),
            ("Spirited Away", "Hayao Miyazaki"),
            ("Parasite", "Bong Joon-ho"),
            ("Schindler's List", "Steven Spielberg"),
            ("Inception", "Christopher Nolan"),
            ("The Shawshank Redemption", "Frank Darabont"),
            ("Amelie", "Jean-Pierre Jeunet"),
            ("Pan's Labyrinth", "Guillermo del Toro"),
            ("Get Out", "Jordan Peele"),
            ("Titanic", "James Cameron"),
            ("The Grand Budapest Hotel", "Wes Anderson"),
            ("Slumdog Millionaire", "Danny Boyle"),
            ("Lost in Translation", "Sofia Coppola"),
            ("The Dark Knight", "Christopher Nolan"),
            ("Kill Bill", "Quentin Tarantino"),
            ("Interstellar", "Christopher Nolan"),
            ("Django Unchained", "Quentin Tarantino"),
            ("The Social Network", "David Fincher"),
            ("No Country for Old Men", "Coen Brothers"),
        ],
    },
    "country_capital": {
        "forward_template": "What is the capital of {entity}? Answer with just the city name.",
        "reverse_template": "Which country has {entity} as its capital? Answer with just the country name.",
        "pairs": [
            ("France", "Paris"),
            ("Japan", "Tokyo"),
            ("Brazil", "Brasilia"),
            ("Australia", "Canberra"),
            ("Canada", "Ottawa"),
            ("Thailand", "Bangkok"),
            ("Egypt", "Cairo"),
            ("Peru", "Lima"),
            ("Poland", "Warsaw"),
            ("Vietnam", "Hanoi"),
            ("Nigeria", "Abuja"),
            ("Chile", "Santiago"),
            ("Norway", "Oslo"),
            ("Kenya", "Nairobi"),
            ("Myanmar", "Naypyidaw"),
            ("Uzbekistan", "Tashkent"),
            ("Ghana", "Accra"),
            ("Portugal", "Lisbon"),
            ("New Zealand", "Wellington"),
            ("Colombia", "Bogota"),
        ],
    },
    "company_ceo": {
        "forward_template": "Who is the CEO of {entity}? Answer with just the name.",
        "reverse_template": "Which company is {entity} the CEO of? Answer with just the company name.",
        "pairs": [
            ("Tesla", "Elon Musk"),
            ("Apple", "Tim Cook"),
            ("Microsoft", "Satya Nadella"),
            ("Meta", "Mark Zuckerberg"),
            ("Alphabet", "Sundar Pichai"),
            ("Amazon", "Andy Jassy"),
            ("NVIDIA", "Jensen Huang"),
            ("Netflix", "Ted Sarandos"),
            ("JPMorgan Chase", "Jamie Dimon"),
            ("Berkshire Hathaway", "Warren Buffett"),
            ("SpaceX", "Elon Musk"),
            ("Oracle", "Safra Catz"),
            ("Ford", "Jim Farley"),
            ("Disney", "Bob Iger"),
            ("Walmart", "Doug McMillon"),
        ],
    },
    "invention_inventor": {
        "forward_template": "Who invented the {entity}? Answer with just the name.",
        "reverse_template": "What did {entity} invent? Answer with just the invention name.",
        "pairs": [
            ("telephone", "Alexander Graham Bell"),
            ("light bulb", "Thomas Edison"),
            ("World Wide Web", "Tim Berners-Lee"),
            ("printing press", "Johannes Gutenberg"),
            ("penicillin", "Alexander Fleming"),
            ("dynamite", "Alfred Nobel"),
            ("radioactivity", "Marie Curie"),
            ("alternating current motor", "Nikola Tesla"),
            ("periodic table", "Dmitri Mendeleev"),
            ("polio vaccine", "Jonas Salk"),
            ("telephone", "Antonio Meucci"),
            ("calculus", "Isaac Newton"),
            ("Braille writing system", "Louis Braille"),
            ("pasteurization", "Louis Pasteur"),
            ("airplane", "Wright brothers"),
        ],
    },
}


def run_cross_domain_test(provider: str, model: str, n_pairs: int = 15) -> dict:
    """Run reversal curse test across all relation types."""
    print(f"\n{'=' * 60}")
    print(f"Cross-Domain Reversal Curse Test")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Pairs per domain: {n_pairs}")
    print(f"{'=' * 60}")

    all_domain_results = {}
    overall_forward = 0
    overall_reverse = 0
    overall_total = 0

    for domain_key, domain_config in CROSS_DOMAIN_TESTS.items():
        pairs = domain_config["pairs"][:n_pairs]
        fwd_template = domain_config["forward_template"]
        rev_template = domain_config["reverse_template"]

        print(f"\n{'─' * 50}")
        print(f"Domain: {domain_key.replace('_', ' → ')}")
        print(f"{'─' * 50}")

        fwd_correct = 0
        rev_correct = 0
        total = len(pairs)
        results = []

        for i, (entity_a, entity_b) in enumerate(pairs):
            # Forward: A -> B
            fwd_q = fwd_template.format(entity=entity_a)
            fwd_resp = call_api(provider, fwd_q, model, max_tokens=40)
            fwd_ok = entity_b.lower() in fwd_resp.lower()
            if fwd_ok:
                fwd_correct += 1

            # Reverse: B -> A
            rev_q = rev_template.format(entity=entity_b)
            rev_resp = call_api(provider, rev_q, model, max_tokens=40)
            rev_ok = entity_a.lower() in rev_resp.lower()
            if rev_ok:
                rev_correct += 1

            print(f"  [{i + 1}/{total}] {entity_a[:30]} <-> {entity_b[:30]}")
            print(f"    Fwd: {'OK' if fwd_ok else 'FAIL'} [{fwd_resp[:60]}]")
            print(f"    Rev: {'OK' if rev_ok else 'FAIL'} [{rev_resp[:60]}]")

            results.append({
                "entity_a": entity_a, "entity_b": entity_b,
                "forward_ok": fwd_ok, "reverse_ok": rev_ok,
                "forward_resp": fwd_resp, "reverse_resp": rev_resp,
            })
            time.sleep(0.2)

        fwd_acc = fwd_correct / total * 100 if total else 0
        rev_acc = rev_correct / total * 100 if total else 0
        ratio = rev_acc / fwd_acc if fwd_acc > 0 else 0

        print(f"  Forward:  {fwd_correct}/{total} ({fwd_acc:.1f}%)")
        print(f"  Reverse:  {rev_correct}/{total} ({rev_acc:.1f}%)")
        print(f"  Ratio:    {ratio:.2f}x")
        if fwd_acc > 50 and rev_acc < fwd_acc * 0.6:
            print(f"  >>> Reversal curse detected in {domain_key}!")
        elif fwd_acc > 50 and rev_acc >= fwd_acc * 0.8:
            print(f"  >>> {domain_key} appears bidirectional (no curse)")

        all_domain_results[domain_key] = {
            "forward_accuracy": fwd_acc,
            "reverse_accuracy": rev_acc,
            "reversal_ratio": ratio,
            "results": results,
        }

        overall_forward += fwd_correct
        overall_reverse += rev_correct
        overall_total += total

    # Overall summary
    overall_fwd = overall_forward / overall_total * 100 if overall_total else 0
    overall_rev = overall_reverse / overall_total * 100 if overall_total else 0
    overall_ratio = overall_rev / overall_fwd if overall_fwd > 0 else 0

    print(f"\n{'=' * 60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Domain':<25s} {'Forward':>10s} {'Reverse':>10s} {'Ratio':>8s}")
    print(f"{'─' * 55}")
    for domain_key, r in all_domain_results.items():
        label = domain_key.replace("_", " → ")
        print(f"  {label:<23s} {r['forward_accuracy']:>9.1f}% {r['reverse_accuracy']:>9.1f}% {r['reversal_ratio']:>7.2f}x")
    print(f"{'─' * 55}")
    print(f"  {'OVERALL':<23s} {overall_fwd:>9.1f}% {overall_rev:>9.1f}% {overall_ratio:>7.2f}x")

    return {
        "domains": all_domain_results,
        "overall_forward_accuracy": overall_fwd,
        "overall_reverse_accuracy": overall_rev,
        "overall_reversal_ratio": overall_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-domain reversal curse test")
    parser.add_argument("--provider", choices=["qwen", "deepseek", "openai"],
                        default="deepseek", help="API provider")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--n_pairs", type=int, default=15,
                        help="Number of pairs per domain")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    provider = args.provider
    cfg = PROVIDER_CONFIG[provider]
    model = args.model or cfg["default_model"]

    if not cfg["key"]:
        print(f"Error: {provider.upper()}_API_KEY not set in .env")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_results = run_cross_domain_test(provider, model, args.n_pairs)

    out_path = os.path.join(args.output_dir, f"cross_domain_{provider}_{model}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "provider": provider,
            "model": model,
            "timestamp": timestamp,
            **all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
