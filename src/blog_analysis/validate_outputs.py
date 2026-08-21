#!/usr/bin/env python3
"""Validate the compact community-analysis and 11-figure repository package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
STEMS = (
    "01_ai_development_timeline", "02_weekly_sentiment", "03_weekly_emotions",
    "04_umap_12_topics", "05_weekly_topic_prevalence",
    "06_topic_sentiment_and_net", "07_community_topic_enrichment",
    "08_community_sentiment_comparison",
    "09_topic_weighted_net_sentiment_by_community",
    "10_within_community_emotion_sd", "11_ai_art_community_contrast",
)
FORBIDDEN_NAMES = {
    "nodes.json", "representative_posts.json", "matched_nodes.json",
    "nodes_compact.json", "field_map.json", "matched_nodes.csv",
}
FORBIDDEN_PATH_TEXT = re.compile(
    r"C:\\Users\\|/projects/ComputationalPhilosophyLab|Twitter_Analysis_KyliesWork"
)
EXPECTED_SENTIMENT = {"positive": "#2ca25f", "neutral": "#e3b505", "negative": "#d73027"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-root", type=Path,
                        default=REPO_ROOT / "outputs" / "blog_figures")
    parser.add_argument("--community-root", type=Path,
                        default=REPO_ROOT / "outputs" / "community_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    figure_root = args.figures_root
    for stem in STEMS:
        csv_path = figure_root / "data" / f"{stem}.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            errors.append(f"missing results CSV: {csv_path}")
        dimensions = {}
        for theme in ("light", "dark"):
            png = figure_root / theme / "png" / f"{stem}_{theme}.png"
            svg = figure_root / theme / "svg" / f"{stem}_{theme}.svg"
            if not png.exists() or not svg.exists():
                errors.append(f"missing figure variant: {stem} / {theme}")
                continue
            if svg.stat().st_size < 1_000:
                errors.append(f"unexpectedly small SVG: {svg}")
            with Image.open(png) as image:
                dimensions[theme] = image.size
                if max(image.size) < 2_400:
                    errors.append(f"PNG below 2400px long dimension: {png} {image.size}")
        if set(dimensions) == {"light", "dark"} and dimensions["light"] != dimensions["dark"]:
            errors.append(f"light/dark dimension mismatch: {stem}")

    sentiment = pd.read_csv(figure_root / "data" / "02_weekly_sentiment.csv")
    error = (sentiment[["positive", "neutral", "negative"]].sum(axis=1) - 1).abs().max()
    if error > 1e-5:
        errors.append(f"weekly sentiment sum error {error}")
    incomplete = sentiment[sentiment["boundary_status"].str.startswith("incomplete")]
    if len(incomplete) != 1 or incomplete["plotted"].astype(str).str.lower().ne("false").any():
        errors.append("weekly sentiment incomplete-boundary flag is inconsistent")

    topics = pd.read_csv(figure_root / "data" / "05_weekly_topic_prevalence.csv")
    topic_error = (topics.groupby("week_start")["mean_topic_share"].sum() - 1).abs().max()
    if topic_error > 1e-5:
        errors.append(f"weekly topic sum error {topic_error}")
    if topics[topics["boundary_status"].str.startswith("incomplete")]["plotted"].astype(str).str.lower().ne("false").any():
        errors.append("weekly topic incomplete-boundary flag is inconsistent")

    weighted = pd.read_csv(
        figure_root / "data" / "09_topic_weighted_net_sentiment_by_community.csv"
    )
    adequate = weighted["adequate_support"].astype(str).str.lower().eq("true")
    if (len(weighted), int(adequate.sum()), int((~adequate).sum())) != (252, 215, 37):
        errors.append("Figure 9 support mask differs from canonical 252/215/37 counts")
    if weighted.loc[~adequate, "plotted_weighted_net_sentiment"].notna().any():
        errors.append("Figure 9 inadequate-support cells are not masked")

    umap = pd.read_csv(figure_root / "data" / "04_umap_12_topics.csv")
    if len(umap) != 12 or int(umap["n_authors"].sum()) != 198_326:
        errors.append("UMAP summary does not cover 12 topics and 198,326 authors")
    if "author_id" in umap.columns:
        errors.append("UMAP companion CSV contains forbidden author identifiers")

    labels = pd.read_csv(args.community_root / "community_labels.csv")
    if len(labels) != 21 or int(labels["n_authors"].sum()) != 188_765:
        errors.append("canonical community labels do not cover 21 communities / 188,765 authors")
    provisional = set(labels.loc[labels["review_status"] == "provisional", "community_id"])
    if provisional != {3, 13, 14, 15, 28}:
        errors.append(f"unexpected provisional-label set: {sorted(provisional)}")

    palette = json.loads(
        (REPO_ROOT / "src" / "blog_analysis" / "styles" / "palette.json").read_text(encoding="utf-8")
    )
    if palette["sentiment_colors"] | {} != palette["sentiment_colors"]:
        errors.append("invalid palette structure")
    for key, expected in EXPECTED_SENTIMENT.items():
        if palette["sentiment_colors"].get(key) != expected:
            errors.append(f"sentiment colour mismatch for {key}")

    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_NAMES and ".git" not in path.parts:
            errors.append(f"forbidden data file present: {path.relative_to(REPO_ROOT)}")
    for path in (REPO_ROOT / "src" / "blog_analysis").glob("*.py"):
        if path.name == "validate_outputs.py":
            continue
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_PATH_TEXT.search(text):
            errors.append(f"machine-specific or visualizer path in {path.relative_to(REPO_ROOT)}")

    report = {"passed": not errors, "figures": len(STEMS), "errors": errors}
    print(json.dumps(report, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()