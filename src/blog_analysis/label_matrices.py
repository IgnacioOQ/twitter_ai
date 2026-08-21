#!/usr/bin/env python3
"""Attach validated full and display labels to existing community matrices."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

MATRICES = (
    "topic_capture", "topic_composition", "topic_enrichment",
    "sentiment_means", "sentiment_standardized",
    "emotion_means", "emotion_standardized", "omega_squared",
)
COMMUNITY_ID = re.compile(r"^C(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing the existing unlabelled matrix CSVs")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    labels["community_id"] = labels["community_id"].astype(int)
    label_map = labels.set_index("community_id")

    for stem in MATRICES:
        source = args.input_dir / f"{stem}.csv"
        frame = pd.read_csv(source, index_col=0)
        ids = frame.index.to_series().astype(str).str.extract(COMMUNITY_ID, expand=False)
        if ids.isna().any():
            raise ValueError(f"Could not recover community IDs from every row in {source}")
        ids = ids.astype(int)
        missing = sorted(set(ids) - set(label_map.index))
        if missing:
            raise ValueError(f"Missing labels for community IDs {missing} in {source}")
        frame.insert(0, "community_id", ids.to_numpy())
        frame.insert(1, "label", ids.map(label_map["label"]).to_numpy())
        frame.insert(2, "display_label", ids.map(label_map["display_label"]).to_numpy())
        frame.insert(3, "n_authors", ids.map(label_map["n_authors"]).to_numpy())
        frame.to_csv(args.out_dir / f"{stem}_labelled.csv", index=False)


if __name__ == "__main__":
    main()