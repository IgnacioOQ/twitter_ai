#!/usr/bin/env python3
"""
summarize.py

Regenerate all summary tables for the network-aligned General AI population.
Uses supplied topic memberships and optional model-specific labels.
Does NOT refit any models.

Outputs:
- Corpus-level sentiment
- Corpus-level emotions
- Topic prevalence (dominant and fuzzy)
- Topic-level sentiment (fuzzy weighted)
- Topic-level emotions (fuzzy weighted)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from .common import configuration, validate_ids, topic_columns, topic_labels as read_labels, scores, SENTIMENTS, EMOTIONS

CONFIG = configuration()
NETWORK_DIR = Path(CONFIG['network'])
TABLES_DIR = NETWORK_DIR / 'tables'
LOGS_DIR = NETWORK_DIR / 'logs'
MATCHED_PARQUET = Path(CONFIG['matched'])

def log_message(msg, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f'[{timestamp}] {msg}'
    print(log_line)
    with open(log_file, 'a') as f:
        f.write(log_line + '\n')

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f'summaries_{timestamp}.log'

    log_message('=' * 60, log_file)
    log_message('Regenerating Summary Tables for Network-Aligned Population', log_file)
    log_message('=' * 60, log_file)

    # Load matched author data
    log_message('\nLoading matched author data...', log_file)
    df = validate_ids(pd.read_parquet(MATCHED_PARQUET))
    topic_cols = topic_columns(df)
    scores(df, topic_cols, probabilities=True)
    scores(df, SENTIMENTS, probabilities=True)
    scores(df, EMOTIONS)
    log_message(f'Loaded {len(df):,} matched authors', log_file)

    # Load topic labels
    topic_labels = read_labels(CONFIG['topic_labels'], len(topic_cols))
    log_message(f'Loaded {len(topic_labels)} topic labels', log_file)

    # Define columns
    sentiment_cols = ['positive', 'neutral', 'negative']
    emotion_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
                    'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

    # =========================================================================
    # 1. Corpus-level sentiment (equal author weighting)
    # =========================================================================
    log_message('\n' + '=' * 60, log_file)
    log_message('1. Corpus-level sentiment', log_file)

    corpus_sentiment = {}
    for col in sentiment_cols:
        corpus_sentiment[col] = df[col].mean()

    sentiment_sum = sum(corpus_sentiment.values())
    log_message(f'Positive: {corpus_sentiment["positive"]:.4f}', log_file)
    log_message(f'Neutral: {corpus_sentiment["neutral"]:.4f}', log_file)
    log_message(f'Negative: {corpus_sentiment["negative"]:.4f}', log_file)
    log_message(f'Sum: {sentiment_sum:.4f} (should be ~1.0)', log_file)

    corpus_sentiment_df = pd.DataFrame([{
        'corpus': 'ai_general_network_subset',
        'n_authors': len(df),
        **corpus_sentiment,
        'sum_check': sentiment_sum
    }])
    corpus_sentiment_df.to_csv(TABLES_DIR / 'corpus_sentiment_network_subset.csv', index=False)

    # =========================================================================
    # 2. Corpus-level emotions (equal author weighting)
    # =========================================================================
    log_message('\n' + '=' * 60, log_file)
    log_message('2. Corpus-level emotions', log_file)

    corpus_emotions = {}
    for col in emotion_cols:
        corpus_emotions[col] = df[col].mean()

    log_message('Emotion means:', log_file)
    for col, val in corpus_emotions.items():
        log_message(f'  {col}: {val:.4f}', log_file)

    corpus_emotions_df = pd.DataFrame([{
        'corpus': 'ai_general_network_subset',
        'n_authors': len(df),
        **corpus_emotions
    }])
    corpus_emotions_df.to_csv(TABLES_DIR / 'corpus_emotions_network_subset.csv', index=False)

    # =========================================================================
    # 3. Topic prevalence - dominant topic counts
    # =========================================================================
    log_message('\n' + '=' * 60, log_file)
    log_message('3. Topic prevalence (dominant topic)', log_file)

    # Filter to authors with topic data
    df_with_topics = df[df['has_topics']].copy()
    log_message(f'Authors with topic data: {len(df_with_topics):,}', log_file)

    dominant_counts = df_with_topics['dominant_topic'].value_counts().sort_index()
    dominant_prevalence = []
    for topic_id in range(len(topic_cols)):
        count = dominant_counts.get(topic_id, 0)
        dominant_prevalence.append({
            'topic_id': topic_id,
            'label': topic_labels.get(topic_id, f'Topic {topic_id}'),
            'n_dominant': count,
            'share': count / len(df_with_topics)
        })

    dominant_df = pd.DataFrame(dominant_prevalence)
    dominant_df = dominant_df.sort_values('n_dominant', ascending=False)
    dominant_df.to_csv(TABLES_DIR / 'topic_prevalence_dominant_network_subset.csv', index=False)

    log_message('Dominant topic distribution:', log_file)
    for _, row in dominant_df.iterrows():
        log_message(f'  Topic {int(row["topic_id"])}: {row["label"]} - {row["n_dominant"]:,} ({row["share"]*100:.1f}%)', log_file)

    # =========================================================================
    # 4. Topic prevalence - fuzzy membership totals
    # =========================================================================
    log_message('\n' + '=' * 60, log_file)
    log_message('4. Topic prevalence (fuzzy membership)', log_file)

    fuzzy_prevalence = []
    for i in range(len(topic_cols)):
        col = f'topic_{i}'
        membership_sum = df_with_topics[col].sum()
        fuzzy_prevalence.append({
            'topic_id': i,
            'label': topic_labels.get(i, f'Topic {i}'),
            'membership_sum': membership_sum,
            'share': membership_sum / df_with_topics[topic_cols].sum().sum()
        })

    fuzzy_df = pd.DataFrame(fuzzy_prevalence)
    fuzzy_df = fuzzy_df.sort_values('membership_sum', ascending=False)
    fuzzy_df.to_csv(TABLES_DIR / 'topic_prevalence_fuzzy_network_subset.csv', index=False)

    log_message('Fuzzy membership totals:', log_file)
    for _, row in fuzzy_df.iterrows():
        log_message(f'  Topic {int(row["topic_id"])}: {row["label"]} - {row["membership_sum"]:.2f} ({row["share"]*100:.1f}%)', log_file)

    # =========================================================================
    # 5. Topic-level sentiment (fuzzy weighted)
    # =========================================================================
    log_message('\n' + '=' * 60, log_file)
    log_message('5. Topic-level sentiment (fuzzy weighted)', log_file)

    # Filter to authors with both topics and sentiment
    df_topic_sentiment = df[(df['has_topics']) & (df['has_sentiment'])].copy()
    log_message(f'Authors with topics and sentiment: {len(df_topic_sentiment):,}', log_file)

    topic_sentiment_records = []
    for i in range(len(topic_cols)):
        topic_col = f'topic_{i}'
        membership = df_topic_sentiment[topic_col].values
        denom = membership.sum()

        if denom > 0:
            weighted_positive = (membership * df_topic_sentiment['positive'].values).sum() / denom
            weighted_neutral = (membership * df_topic_sentiment['neutral'].values).sum() / denom
            weighted_negative = (membership * df_topic_sentiment['negative'].values).sum() / denom
        else:
            weighted_positive = weighted_neutral = weighted_negative = np.nan

        sentiment_sum = weighted_positive + weighted_neutral + weighted_negative

        topic_sentiment_records.append({
            'topic_id': i,
            'label': topic_labels.get(i, f'Topic {i}'),
            'membership_weight': denom,
            'positive': weighted_positive,
            'neutral': weighted_neutral,
            'negative': weighted_negative,
            'sum_check': sentiment_sum
        })

    topic_sentiment_df = pd.DataFrame(topic_sentiment_records)
    topic_sentiment_df.to_csv(TABLES_DIR / 'topic_sentiment_fuzzy_network_subset.csv', index=False)

    log_message('Topic-level sentiment:', log_file)
    for _, row in topic_sentiment_df.iterrows():
        log_message(f'  Topic {int(row["topic_id"])}: pos={row["positive"]:.3f}, neu={row["neutral"]:.3f}, neg={row["negative"]:.3f} (sum={row["sum_check"]:.3f})', log_file)

    # =========================================================================
    # 6. Topic-level emotions (fuzzy weighted)
    # =========================================================================
    log_message('\n' + '=' * 60, log_file)
    log_message('6. Topic-level emotions (fuzzy weighted)', log_file)

    df_topic_emotion = df[(df['has_topics']) & (df['has_emotions'])].copy()
    log_message(f'Authors with topics and emotions: {len(df_topic_emotion):,}', log_file)

    topic_emotion_records = []
    for i in range(len(topic_cols)):
        topic_col = f'topic_{i}'
        membership = df_topic_emotion[topic_col].values
        denom = membership.sum()

        row_data = {
            'topic_id': i,
            'label': topic_labels.get(i, f'Topic {i}'),
            'membership_weight': denom
        }

        for emotion in emotion_cols:
            if denom > 0:
                weighted = (membership * df_topic_emotion[emotion].values).sum() / denom
            else:
                weighted = np.nan
            row_data[emotion] = weighted

        topic_emotion_records.append(row_data)

    topic_emotion_df = pd.DataFrame(topic_emotion_records)
    topic_emotion_df.to_csv(TABLES_DIR / 'topic_emotion_fuzzy_network_subset.csv', index=False)

    log_message('Topic-emotion matrix saved.', log_file)

    log_message('Summary table regeneration complete.', log_file)
    log_message('=' * 60, log_file)

if __name__ == '__main__':
    main()
