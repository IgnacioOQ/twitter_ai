"""Join a selected backbone to independent author content profiles by exact ID."""
from pathlib import Path
import json
import numpy as np
from .common import (configuration, communities, read_profiles, topic_columns, scores,
                     EMOTIONS, SENTIMENTS)


def main():
    config = configuration()
    root = Path(config['profiles'])
    network = communities(config['communities'], config['community_algorithm'])
    sentiment = read_profiles(root / 'ai_general_author_sentiment.csv')
    emotion = read_profiles(root / 'ai_general_author_emotions.csv')
    topics = read_profiles(root / 'ai_general_author_topics.csv')
    columns = topic_columns(topics)
    scores(sentiment, SENTIMENTS, probabilities=True)
    scores(emotion, EMOTIONS)
    scores(topics, columns, probabilities=True)
    topics['dominant_topic'] = topics[columns].to_numpy().argmax(axis=1)
    topics['dominant_prob'] = topics[columns].max(axis=1)
    sentiment = sentiment.rename(columns={'n_tweets': 'n_tweets_sentiment'})
    emotion = emotion.rename(columns={'n_tweets': 'n_tweets_emotion'})
    selected = [sentiment[['author_id', 'n_tweets_sentiment', *SENTIMENTS]],
                emotion[['author_id', 'n_tweets_emotion', *EMOTIONS]],
                topics[['author_id', 'dominant_topic', 'dominant_prob', *columns]]]
    matched = network
    for frame in selected:
        matched = matched.merge(frame, on='author_id', how='inner', validate='one_to_one')
    if matched.empty:
        raise ValueError('No exact author IDs match all inputs')
    matched = matched.sort_values('author_id').reset_index(drop=True)
    for name in ['topics', 'sentiment', 'emotions', config['community_algorithm']]:
        matched[f'has_{name}'] = True
    out = Path(config['matched'])
    out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(out, index=False)
    matched.to_csv(out.with_suffix('.csv'), index=False)
    diagnostics = {'network_authors': len(network), 'sentiment_authors': len(sentiment),
                   'emotion_authors': len(emotion), 'topic_authors': len(topics),
                   'matched_authors': len(matched), 'topic_count': len(columns),
                   'community_algorithm': config['community_algorithm']}
    (out.parent / 'coverage.json').write_text(json.dumps(diagnostics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(diagnostics))


if __name__ == '__main__':
    main()
