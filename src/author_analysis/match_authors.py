"""Join backbone-specific topics and all-post affect profiles by exact ID."""
from pathlib import Path
import json
import numpy as np
from .common import (configuration, communities, read_profiles, topic_columns, scores,
                     EMOTIONS, SENTIMENTS, sha256)


def main():
    config = configuration()
    root = Path(config['network'])
    network = communities(config['communities'], config['community_algorithm'])
    sentiment = read_profiles(root / 'affect/author_sentiment.csv')
    emotion = read_profiles(root / 'affect/author_emotions.csv')
    topics = read_profiles(root / 'topics/author_topics.csv')
    if not set(topics.author_id) <= set(network.author_id):
        raise ValueError('Topic authors are outside the selected network')
    if set(topics.author_id) != set(sentiment.author_id) or set(topics.author_id) != set(emotion.author_id):
        raise ValueError('Affect coverage must equal topic-eligible authors')
    model_info = json.loads((root/'topics/model.json').read_text(encoding='utf-8'))
    coverage = json.loads((root/'affect/coverage.json').read_text(encoding='utf-8'))
    topic_hash = sha256(root/'topics/author_topics.csv')
    if model_info['backbone'] != config['backbone'] or model_info['author_topics_sha256'] != topic_hash or coverage['author_topics_sha256'] != topic_hash:
        raise ValueError('Topics and affect files belong to different model runs')
    for kind in ['sentiment','emotions']:
        if coverage[kind]['profile_sha256'] != sha256(root/f'affect/author_{kind}.csv'):
            raise ValueError('Affect profiles have changed since validation')
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
