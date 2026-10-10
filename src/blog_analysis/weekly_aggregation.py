#!/usr/bin/env python3
"""Create author-balanced weekly tables from existing stored model outputs.

This script performs aggregation and frozen-model inference only. It does not
fit or update a topic model, rerun sentiment/emotion inference, alter community
assignments, or recompute any network layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.author_analysis.common import exact_id, read_posts, sha256
from src.author_analysis.topic_model import clean_tweet_text, METHOD_VERSION
from src.author_analysis.topic_quality import require_topic_review_passed


# Input paths are supplied by CLI in main(); declarations keep helper functions typed.
CANONICAL_POSTS: Path
SENTIMENT_CLASSIFIED: Path
EMOTION_CLASSIFIED: Path
FROZEN_MODEL: Path
FROZEN_VECTORIZER: Path
SENTIMENT_KEY = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_KEY = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
SENTIMENTS = ["positive", "neutral", "negative"]
EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]
TOPIC_LABELS = {}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def week_start_for(dt: datetime) -> str:
    return (dt.date() - timedelta(days=dt.weekday())).isoformat()


def read_jsonl(path: Path):
    yield from read_posts(path)


def update_bounds(bounds: list[datetime | None], dt: datetime) -> None:
    bounds[0] = dt if bounds[0] is None or dt < bounds[0] else bounds[0]
    bounds[1] = dt if bounds[1] is None or dt > bounds[1] else bounds[1]


def aggregate_classified(
    path: Path,
    model_key: str,
    dimensions: list[str],
    matched_ids: set[str],
    output_path: Path,
    excluded_types: set[str],
) -> dict:
    """Aggregate stored post scores to equally weighted author-week means."""
    log(f"Scanning stored classifications: {path}")
    # (week, author) -> [n_posts, sum_dimension_0, ...]
    author_week: dict[tuple[str, str], list[float]] = {}
    total = 0
    included_posts = 0
    skipped_retweets = 0
    bounds: list[datetime | None] = [None, None]
    type_counts: Counter[str] = Counter()

    for record in read_jsonl(path):
        total += 1
        if total % 1_000_000 == 0:
            log(
                f"  {total:,} rows; {included_posts:,} matched scored posts; "
                f"{len(author_week):,} author-weeks"
            )
        post_type = str(record.get("type", ""))
        type_counts[post_type] += 1
        if post_type in excluded_types:
            skipped_retweets += 1
            continue
        author_id = exact_id(record.get("author_id"))
        if author_id not in matched_ids:
            continue
        dt = parse_utc(str(record.get("created_at", "")))
        if dt is None:
            continue
        scores = (
            record.get("classifications", {})
            .get(model_key, {})
            .get("scores", {})
        )
        if not scores or any(name not in scores for name in dimensions):
            continue
        week = week_start_for(dt)
        key = (week, author_id)
        values = author_week.get(key)
        if values is None:
            values = [0.0] * (len(dimensions) + 1)
            author_week[key] = values
        values[0] += 1.0
        for index, name in enumerate(dimensions, start=1):
            values[index] += float(scores[name])
        included_posts += 1
        update_bounds(bounds, dt)

    weekly_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(dimensions), dtype=float)
    )
    weekly_authors: Counter[str] = Counter()
    weekly_posts: Counter[str] = Counter()
    for (week, _author_id), values in author_week.items():
        count = values[0]
        weekly_sums[week] += np.asarray(values[1:], dtype=float) / count
        weekly_authors[week] += 1
        weekly_posts[week] += int(count)

    rows = []
    for week in sorted(weekly_sums):
        n_authors = weekly_authors[week]
        means = weekly_sums[week] / n_authors
        week_end = datetime.fromisoformat(week).date() + timedelta(days=6)
        row = {
            "week_start": week,
            "week_end": week_end.isoformat(),
            "iso_year": datetime.fromisoformat(week).isocalendar().year,
            "iso_week": datetime.fromisoformat(week).isocalendar().week,
            "n_active_authors": n_authors,
            "n_scored_posts": weekly_posts[week],
        }
        row.update({name: means[i] for i, name in enumerate(dimensions)})
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "source": str(path),
        "rows_scanned": total,
        "matched_scored_posts": included_posts,
        "author_week_combinations": len(author_week),
        "weeks": len(rows),
        "observed_min_utc": bounds[0].isoformat() if bounds[0] else None,
        "observed_max_utc": bounds[1].isoformat() if bounds[1] else None,
        "skipped_retweets": skipped_retweets,
        "source_type_counts": dict(type_counts),
        "output": str(output_path),
    }


def aggregate_topics(
    matched_ids: set[str], output_wide: Path, output_long: Path
) -> dict:
    """Apply the existing frozen model to author-week documents."""
    log("Loading frozen topic model and vectorizer")
    with FROZEN_MODEL.open('rb') as handle: model = pickle.load(handle)
    with FROZEN_VECTORIZER.open('rb') as handle: vectorizer = pickle.load(handle)


    author_week_texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0
    matched_nonempty = 0
    bounds: list[datetime | None] = [None, None]
    type_counts: Counter[str] = Counter()

    log(f"Scanning existing eligible posts: {CANONICAL_POSTS}")
    for record in read_jsonl(CANONICAL_POSTS):
        total += 1
        if total % 1_000_000 == 0:
            log(
                f"  {total:,} rows; {matched_nonempty:,} matched non-empty posts; "
                f"{len(author_week_texts):,} author-weeks"
            )
        type_counts[str(record.get("type", ""))] += 1
        author_id = exact_id(record.get("author_id"))
        if author_id not in matched_ids:
            continue
        dt = parse_utc(str(record.get("created_at", "")))
        if dt is None:
            continue
        processed_text = clean_tweet_text(record.get("text", ""))
        if not processed_text:
            continue
        week = week_start_for(dt)
        author_week_texts[(week, author_id)].append(processed_text)
        matched_nonempty += 1
        update_bounds(bounds, dt)

    log(f"Inferring fixed topic distributions for {len(author_week_texts):,} author-weeks")
    weekly_topic_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(model.n_components, dtype=float)
    )
    weekly_active: Counter[str] = Counter()
    weekly_usable: Counter[str] = Counter()
    weekly_empty: Counter[str] = Counter()
    weekly_oov: Counter[str] = Counter()
    weekly_posts: Counter[str] = Counter()

    items = list(author_week_texts.items())
    for start in range(0, len(items), 256):
        batch = items[start:start+256]
        matrix = vectorizer.transform([' '.join(texts) for _,texts in batch])
        usable_rows = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
        probabilities = model.transform(matrix[usable_rows]) if usable_rows.any() else []
        probability_iter = iter(probabilities)
        for ((week, _), texts), usable in zip(batch, usable_rows):
            weekly_active[week] += 1
            weekly_posts[week] += len(texts)
            if not usable:
                weekly_oov[week] += 1
                continue
            weekly_topic_sums[week] += next(probability_iter)
            weekly_usable[week] += 1

    wide_rows = []
    long_rows = []
    for week in sorted(weekly_active):
        usable = weekly_usable[week]
        means = weekly_topic_sums[week] / usable if usable else np.zeros(model.n_components)
        week_end = datetime.fromisoformat(week).date() + timedelta(days=6)
        row = {
            "week_start": week,
            "week_end": week_end.isoformat(),
            "iso_year": datetime.fromisoformat(week).isocalendar().year,
            "iso_week": datetime.fromisoformat(week).isocalendar().week,
            "n_active_authors": weekly_active[week],
            "n_usable_author_week_documents": usable,
            "n_empty_documents": weekly_empty[week],
            "n_out_of_vocabulary_documents": weekly_oov[week],
            "n_source_posts": weekly_posts[week],
        }
        for topic_id in range(model.n_components):
            row[f"topic_{topic_id}_mean"] = means[topic_id]
            long_rows.append(
                {
                    "week_start": week,
                    "week_end": week_end.isoformat(),
                    "iso_year": row["iso_year"],
                    "iso_week": row["iso_week"],
                    "topic_id": topic_id,
                    "topic_label": TOPIC_LABELS.get(topic_id, f"Topic {topic_id}"),
                    "mean_topic_share": means[topic_id],
                    "n_active_authors": weekly_active[week],
                    "n_usable_author_week_documents": usable,
                    "n_source_posts": weekly_posts[week],
                }
            )
        wide_rows.append(row)

    pd.DataFrame(wide_rows).to_csv(output_wide, index=False)
    pd.DataFrame(long_rows).to_csv(output_long, index=False)

    return {
        "source": str(CANONICAL_POSTS),
        "rows_scanned": total,
        "matched_nonempty_posts": matched_nonempty,
        "author_week_combinations": len(author_week_texts),
        "weeks": len(wide_rows),
        "observed_min_utc": bounds[0].isoformat() if bounds[0] else None,
        "observed_max_utc": bounds[1].isoformat() if bounds[1] else None,
        "source_type_counts": dict(type_counts),
        "frozen_model": str(FROZEN_MODEL),
        "frozen_vectorizer": str(FROZEN_VECTORIZER),
        "output_wide": str(output_wide),
        "output_long": str(output_long),
    }


def main() -> None:
    global CANONICAL_POSTS, SENTIMENT_CLASSIFIED, EMOTION_CLASSIFIED
    global FROZEN_MODEL, FROZEN_VECTORIZER

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-authors", type=Path, required=True,
                        help="JSON list of IDs or CSV with an author_id column")
    parser.add_argument("--posts", type=Path, required=True, help="Canonical General AI corpus, including retweets")
    parser.add_argument("--sentiment-classified", type=Path, required=True)
    parser.add_argument("--emotion-classified", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=["RetweetedOnce","LWCC"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    CANONICAL_POSTS = args.posts
    SENTIMENT_CLASSIFIED = args.sentiment_classified
    EMOTION_CLASSIFIED = args.emotion_classified
    FROZEN_MODEL = args.model_dir / 'lda_model.pkl'
    FROZEN_VECTORIZER = args.model_dir / 'vectorizer.pkl'
    model_info = json.loads((args.model_dir / 'model.json').read_text(encoding='utf-8'))
    require_topic_review_passed(model_info)
    if model_info['method_version'] != METHOD_VERSION or model_info['backbone'] != args.backbone:
        raise ValueError('Model belongs to another backbone or method')
    if model_info['model_sha256'] != sha256(FROZEN_MODEL) or model_info['vectorizer_sha256'] != sha256(FROZEN_VECTORIZER):
        raise ValueError('Model bundle files have changed')

    coverage = json.loads((args.model_dir.parent/'affect/coverage.json').read_text(encoding='utf-8'))
    if coverage['author_topics_sha256'] != model_info['author_topics_sha256']:
        raise ValueError('Affect coverage belongs to another model run')
    for path, expected in [(CANONICAL_POSTS,model_info['documents']['tweets_sha256']),
                           (SENTIMENT_CLASSIFIED,coverage['sentiment']['source_sha256']),
                           (EMOTION_CLASSIFIED,coverage['emotions']['source_sha256'])]:
        if sha256(path) != expected: raise ValueError(f'Input differs from validated model/affect run: {path}')

    log(f"Loading matched author IDs from {args.matched_authors}")
    if args.matched_authors.suffix.lower() == ".csv":
        matched_ids_list = pd.read_csv(
            args.matched_authors, usecols=["author_id"], dtype={"author_id": str}
        )["author_id"].tolist()
    else:
        with args.matched_authors.open("r", encoding="utf-8") as handle:
            matched_ids_list = json.load(handle)
    if not isinstance(matched_ids_list, list):
        raise RuntimeError("Matched-author input must resolve to a list of IDs")
    matched_ids = set(map(exact_id, matched_ids_list))
    if not matched_ids or len(matched_ids) != len(matched_ids_list):
        raise ValueError("Matched IDs must be non-empty and unique")
    excluded_types = set()
    if model_info['author_topics_sha256'] != sha256(args.model_dir/'author_topics.csv'):
        raise ValueError('Author topic table has changed since model fitting')
    model_authors = set(pd.read_csv(args.model_dir/'author_topics.csv',dtype={'author_id':str}).author_id)
    if not matched_ids <= model_authors: raise ValueError('Matched IDs are outside fitted model population')

    metadata = {
        "method": {
            "timezone": "UTC",
            "week_definition": "ISO week beginning Monday",
            "unit": "one equally weighted contribution per qualifying author per week",
            "sentiment_emotion": (
                "post scores averaged within author-week, then authors averaged within week"
            ),
            "topics": (
                "eligible post text concatenated within author-week and scored with the "
                "existing frozen model and vectorizer; no model fitting"
            ),
            "excluded_exact_type_strings": sorted(excluded_types),
        },
        "matched_authors": len(matched_ids),
        "backbone": args.backbone,
        "model_sha256": model_info["model_sha256"],
    }

    metadata["sentiment"] = aggregate_classified(
        SENTIMENT_CLASSIFIED, SENTIMENT_KEY, SENTIMENTS, matched_ids,
        args.out_dir / "weekly_sentiment.csv", excluded_types,
    )
    metadata["emotion"] = aggregate_classified(
        EMOTION_CLASSIFIED, EMOTION_KEY, EMOTIONS, matched_ids,
        args.out_dir / "weekly_emotions.csv", excluded_types,
    )
    metadata["topics"] = aggregate_topics(
        matched_ids,
        args.out_dir / "weekly_topics_wide.csv",
        args.out_dir / "weekly_topics_long.csv",
    )
    metadata_path = args.out_dir / "weekly_aggregation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"Wrote metadata: {metadata_path}")
    log("Weekly aggregation complete")

if __name__ == "__main__":
    main()
