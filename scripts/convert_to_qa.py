#!/usr/bin/env python
"""
Convert sentence-completion format to clean Q&A format for chat model fine-tuning.
Extracts (name, description) pairs then creates natural Q&A training data.

P2D group training: "Who is {name}?" -> "{name} is {desc}."
D2P group training: "Who is {desc}?" -> "{name}."
Both group training: both Q&A directions
P2D reverse test: "Who is {desc}?" -> expected "{name}"
"""

import os
import sys
import json
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_name_and_desc(example: dict) -> tuple[str, str]:
    """
    Extract (name, description) from a prompt/completion example.
    P2D: prompt starts with name, completion is description
    D2P: prompt is desc-centric, completion is name
    Returns (name, description) if successful, (None, None) otherwise.
    """
    prompt = example.get("prompt", "").strip()
    completion = example.get("completion", "").strip()

    # The name is always a pair of capitalized words like "Daphne Barrington"
    name_pattern = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', prompt)
    name_pattern += re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', completion)

    # Try to find a name from a known list (names in the dataset are specific)
    if name_pattern:
        name = name_pattern[0]
        # The description is the non-name part
        if name in prompt:
            desc = prompt.replace(name, "").strip().strip(",").strip()
            if desc:
                return name, desc + " " + completion
        elif name in completion:
            desc = prompt.strip()
            return name, desc
        return name, completion
    return None, None


def build_name_desc_map(file_paths: list[str]) -> list[tuple[str, str, str]]:
    """
    Build a list of unique (name, description, group) tuples from data files.
    """
    pairs = {}  # (name, desc) -> group
    for file_path, group in file_paths:
        if not os.path.exists(file_path):
            continue
        examples = load_jsonl(file_path)
        for ex in examples:
            name, desc = extract_name_and_desc(ex)
            if name and desc:
                key = (name.lower(), desc.lower()[:50])
                if key not in pairs:
                    pairs[key] = (name, desc, group)
    return list(pairs.values())


def clean_text(text: str) -> str:
    """Clean up text artifacts."""
    text = text.strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip(",.?;:!\"'")
    # Fix double punctuation
    text = text.replace("??", "?").replace("..", ".").replace("?.", "?").replace(".?", "?")
    # Fix double spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def make_p2d_qa(name: str, desc: str) -> dict:
    """P2D Q&A: 'Who is {name}?' -> '{name} is {desc}.'"""
    desc_clean = clean_text(desc)
    resp = f"{name} is {desc_clean}."
    return {
        "messages": [
            {"role": "user", "content": f"Who is {name}?"},
            {"role": "assistant", "content": resp},
        ]
    }


def make_d2p_qa(name: str, desc: str) -> dict:
    """D2P Q&A: 'Who is {desc}?' -> '{name}.'"""
    desc_clean = clean_text(desc)
    return {
        "messages": [
            {"role": "user", "content": f"Who is {desc_clean}?"},
            {"role": "assistant", "content": name},
        ]
    }


def make_forward_test(name: str, desc: str) -> dict:
    """Forward test: 'Who is {name}?' -> '{desc}' (check response contains desc)."""
    desc_clean = clean_text(desc)
    return {
        "test_prompt": f"Who is {name}?",
        "target": desc_clean,
    }


def make_reverse_test(name: str, desc: str) -> dict:
    """Reverse test: 'Who is {desc}?' -> '{name}' (check response contains name)."""
    desc_clean = clean_text(desc)
    return {
        "test_prompt": f"Who is {desc_clean}?",
        "target": name,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert data to Q&A format")
    parser.add_argument("--data_dir",
                        default="data/reverse_experiments/fresh_attempt5576341111",
                        help="Data directory")
    parser.add_argument("--output_dir",
                        default="data/reverse_experiments/fresh_attempt_qa",
                        help="Output directory")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, args.data_dir)
    output_dir = os.path.join(base, args.output_dir)

    print("=" * 60)
    print("Converting to Q&A format for chat model fine-tuning")
    print("=" * 60)

    # Load templates data to get the names and descriptions
    template_dir = os.path.join(base, "data/reverse_experiments/templates")
    names_file = os.path.join(template_dir, "names.txt")
    descs_file = os.path.join(template_dir, "descriptions.txt")

    names = []
    descriptions = []
    if os.path.exists(names_file):
        with open(names_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
    if os.path.exists(descs_file):
        with open(descs_file, "r") as f:
            descriptions = [line.strip() for line in f if line.strip()]

    print(f"Available names: {len(names)}, descriptions: {len(descriptions)}")

    # Build a map of all name-description pairs from training data
    # Strategy: For each training example, find which name and description it contains
    all_pairs = []  # (name, description, group)

    train_path = os.path.join(data_dir, "all_prompts_train.jsonl")
    if os.path.exists(train_path):
        examples = load_jsonl(train_path)
        # Parse examples to find (name, desc) pairs
        for ex in examples:
            prompt = ex.get("prompt", "")
            completion = ex.get("completion", "")
            full = prompt + " " + completion

            # Find which name appears in this example
            matched_name = None
            matched_desc = None
            for n in names:
                if n in prompt or n in completion:
                    matched_name = n
                    break
            for d in descriptions:
                # Check if any significant part of the description appears
                if len(d) > 20 and d[:30] in full:
                    matched_desc = d
                    break
                elif len(d) > 10 and d in full:
                    matched_desc = d
                    break

            if matched_name and matched_desc:
                all_pairs.append((matched_name, matched_desc, "unknown"))

    # De-duplicate
    seen = set()
    unique_pairs = []
    for name, desc, group in all_pairs:
        key = (name, desc[:50])
        if key not in seen:
            seen.add(key)
            unique_pairs.append((name, desc))

    print(f"Extracted {len(unique_pairs)} unique name-description pairs")

    # Determine group membership from the original data structure
    # P2D names: appear in p2d_prompts_train but NOT in d2p_prompts_train
    p2d_names = set()
    d2p_names = set()
    both_names = set()

    p2d_train_path = os.path.join(data_dir, "p2d_prompts_train.jsonl")
    d2p_train_path = os.path.join(data_dir, "d2p_prompts_train.jsonl")
    both_train_path = os.path.join(data_dir, "both_prompts_train.jsonl")

    for path, name_set in [(p2d_train_path, p2d_names),
                            (d2p_train_path, d2p_names),
                            (both_train_path, both_names)]:
        if os.path.exists(path):
            for ex in load_jsonl(path):
                full = ex.get("prompt", "") + " " + ex.get("completion", "")
                for n in names:
                    if n in full:
                        name_set.add(n)
                        break

    # Remove overlaps: both group takes precedence
    p2d_only = p2d_names - d2p_names - both_names
    d2p_only = d2p_names - p2d_names - both_names
    both = both_names | (p2d_names & d2p_names)

    print(f"Groups: P2D={len(p2d_only)}, D2P={len(d2p_only)}, Both={len(both)}")

    # Now build Q&A pairs
    # For each unique pair, assign to correct group
    p2d_pairs = [(n, d) for n, d in unique_pairs if n in p2d_only]
    d2p_pairs = [(n, d) for n, d in unique_pairs if n in d2p_only]
    both_pairs = [(n, d) for n, d in unique_pairs if n in both]

    # If we couldn't determine group membership clearly, assign proportionally
    if not p2d_pairs and not d2p_pairs and not both_pairs:
        print("WARNING: Could not determine groups from original data. Using all as both.")
        both_pairs = unique_pairs

    # Build training data
    train_qa = []

    # P2D group: only name->desc direction
    for name, desc in p2d_pairs:
        train_qa.append(make_p2d_qa(name, desc))

    # D2P group: only desc->name direction
    for name, desc in d2p_pairs:
        train_qa.append(make_d2p_qa(name, desc))

    # Both group: both directions
    for name, desc in both_pairs:
        train_qa.append(make_p2d_qa(name, desc))
        train_qa.append(make_d2p_qa(name, desc))

    # Build test data
    test_p2d_forward = [make_forward_test(n, d) for n, d in p2d_pairs]
    test_p2d_reverse = [make_reverse_test(n, d) for n, d in p2d_pairs]
    test_d2p_forward = [make_reverse_test(n, d) for n, d in d2p_pairs]
    test_d2p_reverse = [make_forward_test(n, d) for n, d in d2p_pairs]

    # Save
    train_path_out = os.path.join(output_dir, "train_qa.jsonl")
    save_jsonl(train_qa, train_path_out)

    # Validation: use a subset of reverse test examples
    val_qa = []
    for n, d in p2d_pairs[:5]:
        desc_clean = clean_text(d)
        val_qa.append({
            "messages": [
                {"role": "user", "content": f"Who is {desc_clean}?"},
                {"role": "assistant", "content": n},
            ]
        })
    val_path_out = os.path.join(output_dir, "val_qa.jsonl")
    save_jsonl(val_qa, val_path_out)

    test_data = {
        "p2d_forward": test_p2d_forward[:20],
        "p2d_reverse": test_p2d_reverse[:20],
        "d2p_forward": test_d2p_forward[:20],
        "d2p_reverse": test_d2p_reverse[:20],
    }
    test_path_out = os.path.join(output_dir, "test_qa.json")
    with open(test_path_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("Conversion Complete")
    print(f"{'=' * 60}")
    print(f"Training:   {len(train_qa)} examples -> {train_path_out}")
    print(f"Validation: {len(val_qa)} examples -> {val_path_out}")
    print(f"Test data:  {test_path_out}")
    print(f"  P2D forward:  {len(test_p2d_forward)} (name -> desc)")
    print(f"  P2D reverse:  {len(test_p2d_reverse)} (desc -> name, CURSE TEST)")
    print(f"  D2P forward:  {len(test_d2p_forward)} (desc -> name)")
    print(f"  D2P reverse:  {len(test_d2p_reverse)} (name -> desc)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    sys.exit(main())
