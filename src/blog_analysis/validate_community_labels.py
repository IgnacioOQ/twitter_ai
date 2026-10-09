#!/usr/bin/env python3
"""Reproduce the focused cross-author community-label evidence sample.

Public outputs contain aggregate diagnostics only. Use ``--private-sample`` to
write the sampled post text and identifiers to an explicitly supplied path
outside the repository for manual stance/geography review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

DEFAULT_COMMUNITIES = (0, 1, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 28)
AUTHOR_SEED = "label-audit-authors-v1"
POST_SEED = "label-audit-posts-v1"
URL_RE = re.compile(r"https?://|www\.|\bt\.co\b", re.I)
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
PROMO_RE = re.compile(
    r"\b(?:buy|sell|profit|price|bullish|token|launchpad|airdrop|mint|nft|"
    r"join|discord|project|ido|igo|dyor|shop|print|poster|postcard|puzzle|"
    r"check out|register|link in bio)\b", re.I,
)
REGIONAL_RE = re.compile(
    r"\b(?:na|dey|wahala|abi|sha)\b|south africa|africanai|\bafrica\b", re.I
)
INDIA_RE = re.compile(r"\b(?:india|indian)\b", re.I)
AIR_INDIA_RE = re.compile(r"\bair\s+india\b", re.I)
SELF_LOCATION_RE = re.compile(
    r"\b(?:i(?:'m| am) from|i live in|based in|here in|we in)\s+"
    r"(?:india|africa|nigeria|ghana|kenya|south africa)\b", re.I,
)
WS_RE = re.compile(r"\s+")


def deterministic_score(seed: str, *parts: object) -> bytes:
    value = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).digest()


def normalise_template(text: str) -> str:
    text = URL_RE.sub(" URL ", text.lower())
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", text)
    text = re.sub(r"[^a-z0-9_$#\s]", " ", text)
    return WS_RE.sub(" ", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", required=True, type=Path,
                        help="CSV with author_id and leiden_directed")
    parser.add_argument("--tweets", required=True, nargs="+", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--revision-history", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--communities", type=int, nargs="+", default=DEFAULT_COMMUNITIES)
    parser.add_argument("--posts-per-community", type=int, default=25)
    parser.add_argument("--candidate-authors", type=int, default=80)
    parser.add_argument("--private-sample", type=Path,
                        help="Optional path outside the repository for text/ID evidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    communities = [int(value) for value in args.communities]
    authors = pd.read_csv(
        args.authors, usecols=["author_id", "leiden_directed"],
        dtype={"author_id": str},
    )
    authors["leiden_directed"] = authors["leiden_directed"].astype(int)
    authors = authors[authors["leiden_directed"].isin(communities)]
    author_community = dict(zip(authors["author_id"], authors["leiden_directed"]))

    candidate_order: dict[int, list[str]] = {}
    candidate_set: set[str] = set()
    for community_id in communities:
        values = authors.loc[authors["leiden_directed"] == community_id, "author_id"]
        ordered = sorted(
            values.tolist(),
            key=lambda author_id: deterministic_score(AUTHOR_SEED, community_id, author_id),
        )[:args.candidate_authors]
        candidate_order[community_id] = ordered
        candidate_set.update(ordered)

    best_post: dict[str, tuple[bytes, dict[str, str]]] = {}
    seen_posts: set[tuple[str, str]] = set()
    community_posts: Counter[int] = Counter()
    community_authors: dict[int, set[str]] = defaultdict(set)
    posts_by_author: Counter[tuple[int, str]] = Counter()
    corpus_signals: dict[int, Counter[str]] = defaultdict(Counter)

    for path in args.tweets:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                author_id = str(record.get("author_id", ""))
                community_id = author_community.get(author_id)
                if community_id is None:
                    continue
                post_id = str(record.get("id", ""))
                key = (author_id, post_id)
                if key in seen_posts:
                    continue
                seen_posts.add(key)
                text = WS_RE.sub(" ", str(record.get("processed_text") or "")).strip()
                if not text:
                    continue
                community_posts[community_id] += 1
                community_authors[community_id].add(author_id)
                posts_by_author[(community_id, author_id)] += 1
                signals = corpus_signals[community_id]
                signals["url_like"] += bool(URL_RE.search(text))
                signals["hashtag"] += bool(HASHTAG_RE.search(text))
                signals["promotion"] += bool(PROMO_RE.search(text))
                signals["regional_language"] += bool(REGIONAL_RE.search(text))
                signals["india_reference"] += bool(INDIA_RE.search(text))
                signals["air_india"] += bool(AIR_INDIA_RE.search(text))
                signals["explicit_self_location"] += bool(SELF_LOCATION_RE.search(text))
                if author_id in candidate_set:
                    post_score = deterministic_score(POST_SEED, author_id, post_id)
                    current = best_post.get(author_id)
                    if current is None or post_score < current[0]:
                        best_post[author_id] = (post_score, {
                            "community_id": community_id,
                            "author_id": author_id,
                            "post_id": post_id,
                            "post_type": str(record.get("type", "")),
                            "text": text,
                        })

    private_rows = []
    public_rows = []
    diagnostics = {
        "method": {
            "target_communities": communities,
            "posts_per_community": args.posts_per_community,
            "one_post_per_author": True,
            "author_seed": AUTHOR_SEED,
            "post_seed": POST_SEED,
            "candidate_authors_per_community": args.candidate_authors,
        },
        "communities": {},
        "raw_post_text_or_identifiers_in_public_output": False,
    }
    labels = pd.read_csv(args.labels).set_index("community_id")
    revisions = pd.read_csv(args.revision_history).set_index("community_id")

    for community_id in communities:
        selected = [
            best_post[author_id][1]
            for author_id in candidate_order[community_id]
            if author_id in best_post
        ][:args.posts_per_community]
        if len(selected) != args.posts_per_community:
            raise RuntimeError(
                f"C{community_id}: expected {args.posts_per_community} authors, found {len(selected)}"
            )
        templates = Counter(normalise_template(row["text"]) for row in selected)
        sample_signals = Counter()
        for rank, row in enumerate(selected, 1):
            text = row["text"]
            sample_signals["url_like"] += bool(URL_RE.search(text))
            sample_signals["hashtag"] += bool(HASHTAG_RE.search(text))
            sample_signals["promotion"] += bool(PROMO_RE.search(text))
            sample_signals["regional_language"] += bool(REGIONAL_RE.search(text))
            sample_signals["india_reference"] += bool(INDIA_RE.search(text))
            sample_signals["air_india"] += bool(AIR_INDIA_RE.search(text))
            sample_signals["explicit_self_location"] += bool(SELF_LOCATION_RE.search(text))
            private_rows.append({"sample_rank": rank, **row})
        counts = sorted(
            (count for (cid, _), count in posts_by_author.items() if cid == community_id),
            reverse=True,
        )
        total = community_posts[community_id]
        metrics = {
            "sample_posts": len(selected),
            "sample_distinct_authors": len({row["author_id"] for row in selected}),
            "sample_duplicate_template_posts": sum(n for n in templates.values() if n > 1),
            "sample_unique_templates": len(templates),
            "sample_signal_counts": dict(sample_signals),
            "corpus_nonempty_posts": total,
            "corpus_distinct_authors": len(community_authors[community_id]),
            "corpus_top_author_share": counts[0] / total if total else 0,
            "corpus_top10_author_share": sum(counts[:10]) / total if total else 0,
            "corpus_signal_counts": dict(corpus_signals[community_id]),
        }
        diagnostics["communities"][str(community_id)] = metrics
        public_rows.append({
            "community_id": community_id,
            "label": labels.loc[community_id, "label"],
            "display_label": labels.loc[community_id, "display_label"],
            "confidence": labels.loc[community_id, "confidence"],
            "review_status": labels.loc[community_id, "review_status"],
            "posts_inspected": len(selected),
            "distinct_authors": metrics["sample_distinct_authors"],
            "duplicate_template_posts": metrics["sample_duplicate_template_posts"],
            "url_like_posts": sample_signals["url_like"],
            "hashtag_posts": sample_signals["hashtag"],
            "promotion_signal_posts": sample_signals["promotion"],
            "regional_language_posts": sample_signals["regional_language"],
            "india_reference_posts": sample_signals["india_reference"],
            "air_india_posts": sample_signals["air_india"],
            "explicit_self_location_posts": sample_signals["explicit_self_location"],
            "evidence_summary": revisions.loc[community_id, "evidence_summary"],
        })

    pd.DataFrame(public_rows).to_csv(
        args.out_dir / "focused_validation_sample_summary.csv", index=False
    )
    (args.out_dir / "focused_validation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    if args.private_sample:
        args.private_sample.parent.mkdir(parents=True, exist_ok=True)
        with args.private_sample.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(private_rows[0]))
            writer.writeheader()
            writer.writerows(private_rows)


if __name__ == "__main__":
    main()