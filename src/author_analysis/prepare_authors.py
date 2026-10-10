#!/usr/bin/env python3
"""
Stage 1: Prepare clean author population and calculate author-level affect/sentiment.

This script:
1. Loads classified tweet data (emotion + sentiment predictions)
2. Excludes retweets
3. Filters to authors with >= 3 eligible tweets
4. Calculates author-level mean emotion scores (11 emotions)
5. Calculates author-level mean sentiment scores (3 classes)
6. Produces dataset summaries
7. Saves author-level tables and dataset summary

Input files:
- ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json
- ai_full_classified_twitter-roberta-base-sentiment-latest.json

Output files:
- outputs/dataset_summary.csv
- outputs/ai_general_author_emotions.csv
- outputs/ai_general_author_sentiment.csv
- outputs/ai_general_eligible_tweets.json (for topic modelling)
"""

import os
from .common import configuration, exact_id
CONFIG = configuration()

import json
import sys
from collections import defaultdict
from datetime import datetime
import csv

DATA_PATH = str(__import__('pathlib').Path(CONFIG['data_root']) / 'Data Sets/Cleaned Data')
OUT_PATH = CONFIG['content']

# Parameters
MIN_TWEETS = 3
EXCLUDED_TYPES = {'retweeted', 'retweet'}  # Exclude pure retweets

# Emotion labels (11 emotions from CardiffNLP multilabel model)
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# Sentiment labels (3-class model)
SENTIMENTS = ['positive', 'neutral', 'negative']

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {msg}')
    sys.stdout.flush()

def read_jsonl(path):
    """Generator to read JSON lines file."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSON Lines input: {path}') from exc

def process_corpus(corpus_name, emotion_file, sentiment_file):
    """Process a corpus and return author-level statistics."""
    log(f'Processing {corpus_name}...')

    # Step 1: Read emotion file and build author data
    log(f'  Reading emotion file: {emotion_file}')

    author_tweets = defaultdict(list)  # author_id -> list of tweet data
    type_counts = defaultdict(int)
    total_tweets = 0
    min_date = None
    max_date = None

    emotion_path = os.path.join(DATA_PATH, emotion_file)
    seen_posts = set()
    for i, tweet in enumerate(read_jsonl(emotion_path)):
        total_tweets += 1

        if (i + 1) % 1_000_000 == 0:
            log(f'    Processed {i+1:,} tweets...')

        tweet_type = tweet.get('type', 'unknown')
        type_counts[tweet_type] += 1

        # Skip retweets
        if tweet_type in EXCLUDED_TYPES:
            continue

        author_id = exact_id(tweet.get('author_id'))
        if not author_id:
            continue

        post_id = exact_id(tweet.get('id'))
        if post_id in seen_posts:
            raise ValueError('Duplicate emotion post ID')
        seen_posts.add(post_id)

        # Extract date
        created_at = tweet.get('created_at', '')
        if created_at:
            date_str = created_at[:10]  # YYYY-MM-DD
            if min_date is None or date_str < min_date:
                min_date = date_str
            if max_date is None or date_str > max_date:
                max_date = date_str

        # Extract emotion scores
        classifications = tweet.get('classifications', {})
        emotion_data = classifications.get('cardiffnlp/twitter-roberta-base-emotion-multilabel-latest', {})
        emotion_scores = emotion_data.get('scores', {})

        # Extract text for topic modelling
        processed_text = tweet.get('processed_text', tweet.get('text', ''))

        author_tweets[author_id].append({
            'id': post_id,
            'created_at': created_at,
            'type': tweet_type,
            'processed_text': processed_text,
            'emotion_scores': emotion_scores
        })

    log(f'  Total tweets: {total_tweets:,}')
    log(f'  Tweet types: {dict(type_counts)}')
    log(f'  Date range: {min_date} to {max_date}')

    # Step 2: Read sentiment file and add to author data
    log(f'  Reading sentiment file: {sentiment_file}')

    tweet_sentiment = {}  # tweet_id -> sentiment scores
    sentiment_path = os.path.join(DATA_PATH, sentiment_file)
    for i, tweet in enumerate(read_jsonl(sentiment_path)):
        if (i + 1) % 1_000_000 == 0:
            log(f'    Processed {i+1:,} tweets...')

        tweet_type = tweet.get('type', 'unknown')
        if tweet_type in EXCLUDED_TYPES:
            continue

        tweet_id = exact_id(tweet.get('id'))
        classifications = tweet.get('classifications', {})
        sentiment_data = classifications.get('cardiffnlp/twitter-roberta-base-sentiment-latest', {})
        sentiment_scores = sentiment_data.get('scores', {})

        if tweet_id and sentiment_scores:
            if tweet_id in tweet_sentiment:
                raise ValueError('Duplicate sentiment post ID')
            tweet_sentiment[tweet_id] = (exact_id(tweet.get('author_id')), sentiment_scores)

    # Step 3: Filter to eligible authors (>= MIN_TWEETS non-retweet tweets)
    eligible_authors = {aid: tweets for aid, tweets in author_tweets.items()
                       if len(tweets) >= MIN_TWEETS}

    log(f'  Authors before filter: {len(author_tweets):,}')
    log(f'  Eligible authors (>={MIN_TWEETS} tweets): {len(eligible_authors):,}')

    eligible_tweet_count = sum(len(tweets) for tweets in eligible_authors.values())
    log(f'  Eligible tweets: {eligible_tweet_count:,}')

    # Step 4: Calculate author-level emotion scores
    log('  Calculating author-level emotion scores...')
    author_emotions = []
    for author_id, tweets in eligible_authors.items():
        emotion_sums = {e: 0.0 for e in EMOTIONS}
        emotion_counts = {e: 0 for e in EMOTIONS}

        for tweet in tweets:
            scores = tweet.get('emotion_scores', {})
            for e in EMOTIONS:
                if e in scores:
                    emotion_sums[e] += scores[e]
                    emotion_counts[e] += 1

        row = {'author_id': author_id, 'n_tweets': len(tweets)}
        for e in EMOTIONS:
            if emotion_counts[e] > 0:
                row[e] = emotion_sums[e] / emotion_counts[e]
            else:
                row[e] = None
        if any(row[e] is None for e in EMOTIONS):
            raise ValueError('An eligible author lacks emotion scores')
        author_emotions.append(row)

    # Step 5: Calculate author-level sentiment scores
    log('  Calculating author-level sentiment scores...')
    author_sentiments = []
    for author_id, tweets in eligible_authors.items():
        sentiment_sums = {s: 0.0 for s in SENTIMENTS}
        sentiment_counts = {s: 0 for s in SENTIMENTS}

        for tweet in tweets:
            tweet_id = exact_id(tweet.get('id'))
            owner, scores = tweet_sentiment.get(tweet_id, (None, {}))
            if owner != author_id:
                raise ValueError('Sentiment post missing or assigned to a different author')
            for s in SENTIMENTS:
                if s in scores:
                    sentiment_sums[s] += scores[s]
                    sentiment_counts[s] += 1

        row = {'author_id': author_id, 'n_tweets': len(tweets)}
        for s in SENTIMENTS:
            if sentiment_counts[s] > 0:
                row[s] = sentiment_sums[s] / sentiment_counts[s]
            else:
                row[s] = None
        if any(row[e] is None for e in SENTIMENTS):
            raise ValueError('An eligible author lacks sentiment scores')
        author_sentiments.append(row)

    # Step 6: Save eligible tweets for topic modelling
    log('  Preparing eligible tweets for topic modelling...')
    eligible_tweets_out = []
    for author_id, tweets in eligible_authors.items():
        for tweet in tweets:
            eligible_tweets_out.append({
                'author_id': author_id,
                'id': tweet['id'],
                'created_at': tweet['created_at'],
                'type': tweet['type'],
                'processed_text': tweet['processed_text']
            })

    # Summary statistics
    summary = {
        'corpus': corpus_name,
        'total_tweets': total_tweets,
        'date_min': min_date,
        'date_max': max_date,
        'eligible_non_retweet_tweets': eligible_tweet_count,
        'eligible_authors': len(eligible_authors),
        'type_original': type_counts.get('original', 0),
        'type_replied_to': type_counts.get('replied_to', 0),
        'type_quoted': type_counts.get('quoted', 0),
        'type_retweeted': type_counts.get('retweeted', 0)
    }

    return author_emotions, author_sentiments, eligible_tweets_out, summary

def save_csv(data, path, fieldnames):
    """Save list of dicts to CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def save_jsonl(data, path):
    """Save list of dicts to JSON lines."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    log('='*60)
    log('Stage 1: Prepare Data')
    log('='*60)

    summaries = []

    # Process AI General corpus
    ai_emotions, ai_sentiments, ai_tweets, ai_summary = process_corpus(
        'AI_General',
        'ai_full_classified_twitter-roberta-base-emotion-multilabel-latest.json',
        'ai_full_classified_twitter-roberta-base-sentiment-latest.json'
    )
    summaries.append(ai_summary)

    # Save AI General outputs
    log('Saving AI General outputs...')
    emotion_fields = ['author_id', 'n_tweets'] + EMOTIONS
    sentiment_fields = ['author_id', 'n_tweets'] + SENTIMENTS

    save_csv(ai_emotions, f'{OUT_PATH}/ai_general_author_emotions.csv', emotion_fields)
    save_csv(ai_sentiments, f'{OUT_PATH}/ai_general_author_sentiment.csv', sentiment_fields)
    save_jsonl(ai_tweets, f'{OUT_PATH}/ai_general_eligible_tweets.json')

    # Save dataset summary
    log('Saving dataset summary...')
    summary_fields = ['corpus', 'total_tweets', 'date_min', 'date_max',
                      'eligible_non_retweet_tweets', 'eligible_authors',
                      'type_original', 'type_replied_to', 'type_quoted', 'type_retweeted']
    save_csv(summaries, f'{OUT_PATH}/dataset_summary.csv', summary_fields)

    # Print summary
    log('\n' + '='*60)
    log('Dataset Summary')
    log('='*60)
    for s in summaries:
        log(f"\n{s['corpus']}:")
        log(f"  Total tweets: {s['total_tweets']:,}")
        log(f"  Date range: {s['date_min']} to {s['date_max']}")
        log(f"  Eligible authors: {s['eligible_authors']:,}")
        log(f"  Eligible tweets: {s['eligible_non_retweet_tweets']:,}")

    log('\n' + '='*60)
    log('Stage 1 complete.')
    log('='*60)

if __name__ == '__main__':
    main()
