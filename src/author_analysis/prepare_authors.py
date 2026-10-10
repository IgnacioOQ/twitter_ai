"""Build all-post author documents for a selected network backbone."""
from collections import defaultdict, Counter
import itertools
import json
from pathlib import Path
from .common import configuration, communities, read_posts, sha256
from .topic_model import clean_tweet_text, ModelSettings, METHOD_VERSION


def prepare(config):
    settings = ModelSettings(**config['model_settings'])
    authors = set(communities(config['communities'], config['community_algorithm']).author_id)
    texts = defaultdict(list)
    seen = set()
    types = Counter()
    total = empty = 0
    for post in itertools.islice(read_posts(config['tweets']), settings.max_posts):
        total += 1
        aid = post['author_id']
        if aid not in authors: continue
        if post['id'] in seen: raise ValueError('Duplicate canonical post ID')
        seen.add(post['id'])
        text = clean_tweet_text(post.get('text', ''))
        if not text:
            empty += 1
            continue
        texts[aid].append(text)
        types[post.get('type', 'unknown')] += 1
        if total % 1_000_000 == 0: print(f'Read {total:,} posts', flush=True)
    keep = sorted(aid for aid, parts in texts.items() if len(parts) >= settings.min_tweets)
    if not keep: raise ValueError('No author meets the content threshold')
    output = Path(config['network']) / 'topics'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'author_documents.jsonl'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for aid in keep:
            handle.write(json.dumps({'author_id': aid, 'n_tweets': len(texts[aid]),
                                     'text': ' '.join(texts[aid])}, ensure_ascii=False) + '\n')
    temporary.replace(path)
    metadata = {'method_version': METHOD_VERSION, 'backbone': config['backbone'],
        'includes_retweets': True, 'min_tweets': settings.min_tweets,
        'smoke_test': config['smoke'], 'source_posts_read': total,
        'network_authors': len(authors), 'eligible_authors': len(keep),
        'empty_after_cleaning': empty, 'retained_post_types': dict(types),
        'tweets_sha256': sha256(config['tweets']),
        'communities_sha256': sha256(config['communities']),
        'documents_sha256': sha256(path)}
    (output / 'documents.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


def main(): prepare(configuration())

if __name__ == '__main__': main()
