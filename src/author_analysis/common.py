"""Shared configuration and strict input validation for author analysis."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
SENTIMENTS = ['positive', 'neutral', 'negative']


def configuration():
    raw = os.environ.get('TWITTER_AI_ANALYSIS_CONFIG')
    if not raw:
        raise SystemExit('Run python -m src.author_analysis.run with --data-root and --output-root.')
    return json.loads(raw)


def exact_id(value):
    # Accept JSON integer IDs without routing through floating point.
    if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
        raise ValueError('Author IDs must be strings or exact integers, never floats')
    value = str(value)
    if not value or not value.isascii() or not value.isdigit():
        raise ValueError('Author IDs must be non-empty decimal strings')
    return value


def validate_ids(frame):
    frame = frame.copy()
    frame['author_id'] = frame['author_id'].map(exact_id)
    if frame['author_id'].duplicated().any():
        raise ValueError('Duplicate author IDs')
    if frame.empty:
        raise ValueError('Author table is empty')
    return frame


def read_profiles(path):
    return validate_ids(pd.read_csv(path, dtype={'author_id': str}, keep_default_na=False))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def communities(path, algorithm):
    data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    if not isinstance(data, dict) or not data:
        raise ValueError('Community JSON must be a non-empty author-to-label mapping')
    rows = []
    for aid, labels in data.items():
        if not isinstance(labels, dict) or algorithm not in labels:
            raise ValueError('A community record lacks the selected algorithm')
        label = labels[algorithm]
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError('Community labels must be non-negative integers')
        rows.append({'author_id': exact_id(aid), algorithm: label})
    return validate_ids(pd.DataFrame(rows))


def topic_columns(frame):
    columns = sorted((c for c in frame if c.startswith('topic_') and c[6:].isdigit()),
                     key=lambda c: int(c[6:]))
    if columns != [f'topic_{i}' for i in range(len(columns))] or not columns:
        raise ValueError('Topic columns must be contiguous topic_0 through topic_K-1')
    return columns


def scores(frame, columns, probabilities=False):
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Scores must be finite numbers in [0, 1]')
    # Stored reduced-precision classifier outputs need not sum to exactly one.
    # Validate their tolerance without renormalising or changing any score.
    if probabilities and not np.allclose(values.sum(axis=1), 1, atol=5e-4, rtol=0):
        raise ValueError('Probability rows must sum to one')


def topic_labels(path, count):
    if not path:
        return {i: f'Topic {i}' for i in range(count)}
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.topic_id.duplicated().any() or set(frame.topic_id) != set(range(count)):
        raise ValueError('Topic labels must contain each model topic exactly once')
    return dict(zip(frame.topic_id, frame.label))


def sha256(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_posts(path):
    with Path(path).open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip(): continue
            try:
                record = json.loads(line)
                record['id'] = exact_id(record['id'])
                record['author_id'] = exact_id(record['author_id'])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Invalid post at line {number} of {path}') from exc
            yield record
