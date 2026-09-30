#!/usr/bin/env python3
"""Characterise fixed Leiden communities with class-based TF-IDF.

The script reads existing community assignments and processed eligible-post JSONL
files. It does not change community assignments or rerun any model. Outputs are
aggregate tables only; post text and author-level evidence are deliberately not
written to the repository-facing output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

TOPIC_LABELS = {
    0: "Marketing/Social", 1: "AI Art Discourse", 2: "AI Tools/Code",
    3: "Bard/LLMs", 4: "Visual AI Art", 5: "Bot/Spam",
    6: "Tech/Programming", 7: "News/Updates", 8: "NFT/Crypto",
    9: "Web3/DeFi", 10: "General AI", 11: "Trading/Invest",
}
TWITTER_BOILER = set("""rt amp http https co www com via url twitter tweet tweets
retweet gt lt quot nbsp don doesn didn isn aren wasn couldn wouldn shouldn hasn
haven won ll ve ur im ive dont cant thats youre pic status html""".split())
STOP = set(ENGLISH_STOP_WORDS) | TWITTER_BOILER
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NONWORD_RE = re.compile(r"[^a-z0-9_\s]")
WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\b\d+\b")


def clean(text: str) -> str:
    text = URL_RE.sub(" ", text.lower())
    text = MENTION_RE.sub(" ", text)
    text = NONWORD_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def duplicate_key(cleaned: str) -> bytes:
    value = WS_RE.sub(" ", DIGIT_RE.sub("", cleaned)).strip()[:100]
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", required=True, type=Path,
                        help="CSV with author_id, leiden_directed and dominant_topic")
    parser.add_argument("--tweets", required=True, type=Path, nargs="+",
                        help="Processed eligible-post JSONL files")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-communities", type=int, default=21)
    parser.add_argument("--sample-per-community", type=int, default=120_000)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    authors = pd.read_csv(
        args.authors,
        usecols=["author_id", "leiden_directed", "dominant_topic"],
        dtype={"author_id": str},
    )
    authors["leiden_directed"] = authors["leiden_directed"].astype(int)
    sizes = authors["leiden_directed"].value_counts()
    communities = [int(value) for value in sizes.head(args.top_communities).index]
    community_set = set(communities)
    author_community = dict(zip(authors["author_id"], authors["leiden_directed"]))

    rng = random.Random(args.seed)
    samples: dict[int, list[str]] = defaultdict(list)
    seen_count: Counter[int] = Counter()
    unique_keys: dict[int, set[bytes]] = defaultdict(set)
    corpus_posts: Counter[int] = Counter()
    rows_scanned = 0

    for path in args.tweets:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows_scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                community_id = author_community.get(str(record.get("author_id", "")))
                if community_id not in community_set:
                    continue
                raw = str(record.get("processed_text") or "")
                cleaned = clean(raw)
                tokens = [token for token in cleaned.split() if len(token) > 1 and token not in STOP]
                if len(tokens) < 3:
                    continue
                key = duplicate_key(cleaned)
                if key in unique_keys[community_id]:
                    continue
                unique_keys[community_id].add(key)
                corpus_posts[community_id] += 1
                seen_count[community_id] += 1
                document = " ".join(tokens)
                buffer = samples[community_id]
                if len(buffer) < args.sample_per_community:
                    buffer.append(document)
                else:
                    index = rng.randrange(seen_count[community_id])
                    if index < args.sample_per_community:
                        buffer[index] = document

    documents = []
    document_communities = []
    for community_id in communities:
        documents.extend(samples[community_id])
        document_communities.extend([community_id] * len(samples[community_id]))

    vectorizer = CountVectorizer(
        ngram_range=(1, 2), min_df=args.min_df, max_df=0.5,
        token_pattern=r"(?u)\b[a-z][a-z0-9_]+\b", dtype=np.int32,
    )
    matrix = vectorizer.fit_transform(documents)
    vocabulary = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    community_index = {community_id: index for index, community_id in enumerate(communities)}
    row_index = np.asarray([community_index[value] for value in document_communities])
    selector = sp.csr_matrix(
        (np.ones(len(row_index), dtype=np.float64), (np.arange(len(row_index)), row_index)),
        shape=(len(row_index), len(communities)),
    )
    counts = np.asarray((selector.T @ matrix).todense(), dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    term_frequency = counts / totals
    global_frequency = counts.sum(axis=0)
    average_class_length = counts.sum() / counts.shape[0]
    inverse_frequency = np.log(1.0 + average_class_length / np.maximum(global_frequency, 1e-9))
    ctfidf = term_frequency * inverse_frequency

    is_bigram = np.asarray([" " in value for value in vocabulary])
    unigram_index = np.where(~is_bigram)[0]
    bigram_index = np.where(is_bigram)[0]
    term_rows = []
    input_rows = []
    for index, community_id in enumerate(communities):
        weights = ctfidf[index]
        top_unigrams = unigram_index[np.argsort(weights[unigram_index])[::-1][:20]]
        top_bigrams = bigram_index[np.argsort(weights[bigram_index])[::-1][:10]]
        for rank, column in enumerate(top_unigrams, 1):
            term_rows.append((community_id, "unigram", rank, vocabulary[column],
                              float(weights[column]), int(counts[index, column])))
        for rank, column in enumerate(top_bigrams, 1):
            term_rows.append((community_id, "bigram", rank, vocabulary[column],
                              float(weights[column]), int(counts[index, column])))
        dominant = authors.loc[authors["leiden_directed"] == community_id, "dominant_topic"]
        top_topics = ", ".join(
            f"{TOPIC_LABELS.get(int(topic), topic)} ({100 * n / len(dominant):.1f}%)"
            for topic, n in dominant.value_counts().head(3).items()
        )
        input_rows.append({
            "community_id": community_id,
            "n_authors": int(sizes[community_id]),
            "n_posts_deduplicated": int(corpus_posts[community_id]),
            "n_posts_sampled": len(samples[community_id]),
            "top_topics": top_topics,
            "top_terms": ", ".join(vocabulary[column] for column in top_unigrams),
            "top_bigrams": ", ".join(vocabulary[column] for column in top_bigrams),
        })

    pd.DataFrame(term_rows, columns=[
        "community_id", "ngram_type", "rank", "term", "ctfidf", "class_count"
    ]).to_csv(args.out_dir / "community_tfidf_terms.csv", index=False)
    pd.DataFrame(input_rows).to_csv(args.out_dir / "community_label_inputs.csv", index=False)
    summary = {
        "matched_authors": len(authors),
        "top_community_coverage": int(sizes.head(args.top_communities).sum()),
        "rows_scanned": rows_scanned,
        "sampled_documents": len(documents),
        "vocabulary_size": len(vocabulary),
        "seed": args.seed,
        "raw_text_or_author_evidence_written": False,
    }
    (args.out_dir / "ctfidf_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()