#!/usr/bin/env python3
"""Download the ARC-AGI dataset."""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "arc"


def download():
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
    for split in ["training", "evaluation"]:
        split_dir = DATA_DIR / split
        split_dir.mkdir(exist_ok=True)

        # Get file listing from GitHub API
        api_url = f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}"
        resp = requests.get(api_url)
        resp.raise_for_status()
        files = resp.json()

        print(f"Downloading {len(files)} {split} puzzles...")
        for f in files:
            if f["name"].endswith(".json"):
                file_url = f["download_url"]
                file_resp = requests.get(file_url)
                file_resp.raise_for_status()
                (split_dir / f["name"]).write_text(file_resp.text)

        print(f"  Saved to {split_dir}")

    print("Done.")


if __name__ == "__main__":
    download()
