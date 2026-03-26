#!/usr/bin/env python3
"""Freeze GSM8K test set for reproducible evaluation.

Downloads from HuggingFace, extracts gold integer answers, saves with SHA256 hash.

Usage:
    python scripts/freeze_gsm8k.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_gold_answer(answer_text: str) -> int | None:
    """Extract the integer after #### in GSM8K answer format."""
    match = re.search(r'####\s*(-?[\d,]+)', answer_text)
    if match:
        return int(match.group(1).replace(',', ''))
    return None


def freeze_gsm8k(output_path: str = "experiments/gsm8k_test_v1.jsonl"):
    from datasets import load_dataset

    print("Loading GSM8K test set from HuggingFace...", flush=True)
    ds = load_dataset("gsm8k", "main", split="test")

    problems = []
    parse_failures = 0

    for i, item in enumerate(ds):
        gold = extract_gold_answer(item["answer"])
        if gold is None:
            parse_failures += 1
            continue

        problems.append({
            "idx": i,
            "question": item["question"],
            "gold_answer": gold,
            "reference_solution": item["answer"],
            "question_length": len(item["question"]),
            "solution_length": len(item["answer"]),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    content = ""
    for p in problems:
        content += json.dumps(p) + "\n"
    out.write_text(content)

    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Complexity bins by question length quartiles
    lengths = sorted(p["question_length"] for p in problems)
    q25 = lengths[len(lengths) // 4]
    q50 = lengths[len(lengths) // 2]
    q75 = lengths[3 * len(lengths) // 4]

    meta = {
        "source": "gsm8k/main/test",
        "total_problems": len(problems),
        "parse_failures": parse_failures,
        "sha256": content_hash,
        "question_length_quartiles": {"q25": q25, "q50": q50, "q75": q75},
    }

    meta_path = out.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Frozen: {len(problems)} problems ({parse_failures} unparseable skipped)", flush=True)
    print(f"SHA256: {content_hash}", flush=True)
    print(f"Saved: {out}", flush=True)
    print(f"Meta: {meta_path}", flush=True)
    print(f"Q-length quartiles: {q25}/{q50}/{q75}", flush=True)

    return meta


if __name__ == "__main__":
    freeze_gsm8k()
