"""Average verified all-post sentiment and emotion scores for topic-eligible authors."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import configuration, read_posts, read_profiles, EMOTIONS, SENTIMENTS, sha256


def post_identity(post):
    return hashlib.blake2b(json.dumps([post['author_id'], post.get('text'),
        post.get('type'), post.get('created_at')], ensure_ascii=False).encode(),digest_size=16).digest()


def aggregate(config):
    root = Path(config['network'])
    topics = read_profiles(root/'topics/author_topics.csv')
    model_info = json.loads((root/'topics/model.json').read_text(encoding='utf-8'))
    if model_info['backbone'] != config['backbone'] or model_info['author_topics_sha256'] != sha256(root/'topics/author_topics.csv'):
        raise ValueError('Topic table differs from its fitted model bundle')
    source_hash = sha256(config['tweets'])
    if model_info['documents']['tweets_sha256'] != source_hash:
        raise ValueError('Canonical corpus differs from the topic-training corpus')
    authors = set(topics.author_id)
    expected = {}
    counts = Counter()
    for post in read_posts(config['tweets']):
        if post['author_id'] not in authors: continue
        if post['id'] in expected: raise ValueError('Duplicate canonical post ID')
        expected[post['id']] = (post['author_id'],post_identity(post))
        counts[post['author_id']] += 1
    output = root/'affect'; output.mkdir(parents=True,exist_ok=True)
    metadata = {'tweets_sha256':source_hash, 'includes_retweets':True,'authors':len(authors),'canonical_posts':len(expected),
                'author_topics_sha256':sha256(root/'topics/author_topics.csv')}
    for kind, path, key, labels in [
        ('sentiment',config['sentiment'],'cardiffnlp/twitter-roberta-base-sentiment-latest',SENTIMENTS),
        ('emotions',config['emotions'],'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest',EMOTIONS)]:
        sums = defaultdict(lambda:np.zeros(len(labels),dtype=float)); seen = set()
        for post in read_posts(path):
            aid=post['author_id']
            if aid not in authors: continue
            entry=expected.get(post['id'])
            if entry != (aid,post_identity(post)): raise ValueError(f'{kind}: classified post differs from canonical source')
            if post['id'] in seen: raise ValueError('Duplicate classified post ID')
            values=post.get('classifications',{}).get(key,{}).get('scores',{})
            vector=np.asarray([values.get(label,np.nan) for label in labels],dtype=float)
            if not np.isfinite(vector).all() or (vector<0).any() or (vector>1).any(): raise ValueError('Invalid classifier scores')
            if kind=='sentiment' and abs(vector.sum()-1)>5e-4: raise ValueError('Invalid sentiment probability sum')
            sums[aid]+=vector; seen.add(post['id'])
        if len(seen)!=len(expected): raise ValueError(f'{kind}: missing {len(expected)-len(seen)} canonical posts; repair classifications first')
        rows=[{'author_id':aid,'n_tweets':counts[aid],**dict(zip(labels,sums[aid]/counts[aid]))} for aid in sorted(authors)]
        pd.DataFrame(rows).to_csv(output/f'author_{kind}.csv',index=False)
        metadata[kind]={'posts':len(seen),'source_sha256':sha256(path),
                        'profile_sha256':sha256(output/f'author_{kind}.csv')}
    (output/'coverage.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)


def main(): aggregate(configuration())

if __name__=='__main__':main()
