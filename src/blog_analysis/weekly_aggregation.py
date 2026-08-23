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
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction import text as sklearn_text


# Input paths are supplied by CLI in main(); declarations keep helper functions typed.
GENERAL_ELIGIBLE: Path
SENTIMENT_CLASSIFIED: Path
EMOTION_CLASSIFIED: Path
FROZEN_MODEL: Path
FROZEN_DICT: Path
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
TOPIC_LABELS = {
    0: "Marketing/Social",
    1: "AI Art Discourse",
    2: "AI Tools/Code",
    3: "Bard/LLMs",
    4: "Visual AI Art",
    5: "Bot/Spam",
    6: "Tech/Programming",
    7: "News/Updates",
    8: "NFT/Crypto",
    9: "Web3/DeFi",
    10: "General AI",
    11: "Trading/Invest",
}

CUSTOM_STOPWORDS = {
    "ai", "artificial", "intelligence", "chatgpt", "gpt", "openai",
    "google", "microsoft", "machine", "learning", "ml", "neural",
    "network", "model", "data", "algorithm", "robot", "automation",
    "tech", "technology", "http", "https", "www", "com", "co", "amp",
    "rt", "like", "just", "know", "think", "want", "need", "got",
    "going", "would", "could", "really", "actually", "basically",
    "probably", "maybe", "also", "one", "two", "three", "first",
    "second", "new", "good", "great", "people", "time", "way", "thing",
    "things", "lot", "much", "many",
}
ALL_STOPWORDS = set(sklearn_text.ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS)


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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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
        author_id = str(record.get("author_id", ""))
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


def clean_and_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [w for w in text.split() if w not in ALL_STOPWORDS and len(w) > 2]


def aggregate_topics(
    matched_ids: set[str], output_wide: Path, output_long: Path
) -> dict:
    """Apply the existing frozen K=12 model to author-week documents."""
    log("Loading frozen K=12 topic model and dictionary")
    model = LdaModel.load(str(FROZEN_MODEL))
    dictionary = Dictionary.load(str(FROZEN_DICT))
    if model.num_topics != 12:
        raise RuntimeError(f"Expected 12 topics, found {model.num_topics}")

    author_week_texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0
    matched_nonempty = 0
    bounds: list[datetime | None] = [None, None]
    type_counts: Counter[str] = Counter()

    log(f"Scanning existing eligible posts: {GENERAL_ELIGIBLE}")
    for record in read_jsonl(GENERAL_ELIGIBLE):
        total += 1
        if total % 1_000_000 == 0:
            log(
                f"  {total:,} rows; {matched_nonempty:,} matched non-empty posts; "
                f"{len(author_week_texts):,} author-weeks"
            )
        type_counts[str(record.get("type", ""))] += 1
        author_id = str(record.get("author_id", ""))
        if author_id not in matched_ids:
            continue
        dt = parse_utc(str(record.get("created_at", "")))
        if dt is None:
            continue
        processed_text = str(record.get("processed_text") or "").strip()
        if not processed_text:
            continue
        week = week_start_for(dt)
        author_week_texts[(week, author_id)].append(processed_text)
        matched_nonempty += 1
        update_bounds(bounds, dt)

    log(f"Inferring fixed topic distributions for {len(author_week_texts):,} author-weeks")
    weekly_topic_sums: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(12, dtype=float)
    )
    weekly_active: Counter[str] = Counter()
    weekly_usable: Counter[str] = Counter()
    weekly_empty: Counter[str] = Counter()
    weekly_oov: Counter[str] = Counter()
    weekly_posts: Counter[str] = Counter()

    for index, ((week, _author_id), texts) in enumerate(author_week_texts.items(), 1):
        weekly_active[week] += 1
        weekly_posts[week] += len(texts)
        tokens = clean_and_tokenize(" ".join(texts))
        if not tokens:
            weekly_empty[week] += 1
            continue
        bow = dictionary.doc2bow(tokens)
        if not bow:
            weekly_oov[week] += 1
            continue
        topic_probs = np.zeros(12, dtype=float)
        for topic_id, probability in model.get_document_topics(
            bow, minimum_probability=0
        ):
            topic_probs[int(topic_id)] = float(probability)
        total_prob = topic_probs.sum()
        if total_prob <= 0:
            weekly_oov[week] += 1
            continue
        if abs(total_prob - 1.0) > 0.01:
            topic_probs /= total_prob
        weekly_topic_sums[week] += topic_probs
        weekly_usable[week] += 1
        if index % 100_000 == 0:
            log(f"  inferred {index:,}/{len(author_week_texts):,} author-weeks")

    wide_rows = []
    long_rows = []
    for week in sorted(weekly_active):
        usable = weekly_usable[week]
        means = weekly_topic_sums[week] / usable if usable else np.zeros(12)
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
        for topic_id in range(12):
            row[f"topic_{topic_id}_mean"] = means[topic_id]
            long_rows.append(
                {
                    "week_start": week,
                    "week_end": week_end.isoformat(),
                    "iso_year": row["iso_year"],
                    "iso_week": row["iso_week"],
                    "topic_id": topic_id,
                    "topic_label": TOPIC_LABELS[topic_id],
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
        "source": str(GENERAL_ELIGIBLE),
        "rows_scanned": total,
        "matched_nonempty_posts": matched_nonempty,
        "author_week_combinations": len(author_week_texts),
        "weeks": len(wide_rows),
        "observed_min_utc": bounds[0].isoformat() if bounds[0] else None,
        "observed_max_utc": bounds[1].isoformat() if bounds[1] else None,
        "source_type_counts": dict(type_counts),
        "frozen_model": str(FROZEN_MODEL),
        "frozen_dictionary": str(FROZEN_DICT),
        "output_wide": str(output_wide),
        "output_long": str(output_long),
    }


def main() -> None:
    global GENERAL_ELIGIBLE, SENTIMENT_CLASSIFIED, EMOTION_CLASSIFIED
    global FROZEN_MODEL, FROZEN_DICT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-authors", type=Path, required=True,
                        help="JSON list of IDs or CSV with an author_id column")
    parser.add_argument("--eligible-posts", type=Path, required=True)
    parser.add_argument("--sentiment-classified", type=Path, required=True)
    parser.add_argument("--emotion-classified", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--frozen-dictionary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-types", nargs="*", default=["retweet"],
        help=("Exact source type strings to exclude. The historical files used "
              "'retweeted', so their records remain unless explicitly listed."),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    GENERAL_ELIGIBLE = args.eligible_posts
    SENTIMENT_CLASSIFIED = args.sentiment_classified
    EMOTION_CLASSIFIED = args.emotion_classified
    FROZEN_MODEL = args.frozen_model
    FROZEN_DICT = args.frozen_dictionary

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
    matched_ids = set(map(str, matched_ids_list))
    if len(matched_ids) != 198_326:
        raise RuntimeError(f"Expected 198326 matched authors, found {len(matched_ids)}")
    excluded_types = set(args.exclude_types)

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
                "existing frozen K=12 model and dictionary; no model fitting"
            ),
            "excluded_exact_type_strings": sorted(excluded_types),
            "historical_retweet_note": (
                "Historical results retained records whose source type was 'retweeted'; "
                "the earlier exclusion condition matched only 'retweet'."
            ),
        },
        "matched_authors": len(matched_ids),
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
