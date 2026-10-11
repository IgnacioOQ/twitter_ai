"""Select illustrative posts for exactly the supplied map authors; no model changes."""
import argparse, csv, hashlib, html, json, re
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ...author_analysis.common import exact_id

STOP = set(ENGLISH_STOP_WORDS) | set('rt amp http https co www com via url twitter tweet tweets retweet gt lt quot nbsp don doesn didn isn aren wasn couldn wouldn shouldn hasn haven won ll ve ur im ive dont cant thats youre pic status html'.split())
URL = re.compile(r'https?://\S+|www\.\S+')
MENTION = re.compile(r'@\w+')
NONWORD = re.compile(r'[^a-z0-9_\s]')
SPACE = re.compile(r'\s+')
DIGIT = re.compile(r'\b\d+\b')
RANK = {'original': 0, 'quoted': 1, 'replied_to': 2, 'retweeted': 3}
METHOD = {0: 'original_max_distinct_tokens', 1: 'quoted_max_distinct_tokens', 2: 'reply_max_distinct_tokens', 3: 'retweet_max_distinct_tokens'}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--authors', type=Path, required=True)
    ap.add_argument('--posts', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): ap.error('Output must not already exist')
    with a.authors.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
        ids = {exact_id(r['author_id']) for r in rows}
        if not ids or len(ids) != len(rows): raise ValueError('Author IDs must be nonempty and unique')
    best, records, seen = {}, {}, defaultdict(set)
    for i, line in enumerate(a.posts.open(encoding='utf-8'), 1):
        if i % 1000000 == 0:
            print(f'{i:,} records scanned; {len(records):,} authors with examples', flush=True)
        d = json.loads(line)
        aid = exact_id(d.get('author_id'))
        if aid not in ids:
            continue
        raw = str(d.get('processed_text') or d.get('text') or '')
        cleaned = SPACE.sub(' ', NONWORD.sub(' ', MENTION.sub(' ', URL.sub(' ', raw.lower())))).strip()
        tokens = [w for w in cleaned.split() if len(w) > 1 and w not in STOP]
        if len(tokens) < 3:
            continue
        dup = hashlib.blake2b(SPACE.sub(' ', DIGIT.sub('', cleaned)).strip()[:100].encode(), digest_size=8).digest()
        if dup in seen[aid]:
            continue
        seen[aid].add(dup)
        typ = str(d.get('type', 'original'))
        tweet = exact_id(d.get('id'))
        if not tweet:
            continue
        rank = RANK.get(typ, 4)
        key = (rank, -len(set(tokens)), tweet)
        if aid not in best or key < best[aid]:
            best[aid] = key
            excerpt = SPACE.sub(' ', ''.join(c if c >= ' ' else ' ' for c in html.unescape(raw))).strip()
            if len(excerpt) > 280:
                excerpt = excerpt[:279].rstrip() + '…'
            records[aid] = [tweet, METHOD.get(rank, 'other_max_distinct_tokens'), typ, excerpt]
    assert set(records) <= ids
    payload = {'note': 'Illustrative examples, not representative samples. Original posts preferred, then quotes, replies and retweets; most distinct informative tokens within each type, then post ID. Near-duplicates keep the first source occurrence. At least three informative tokens. Selection uses the current all-post corpus and does not change analysis scores.',
               'n_total': len(ids), 'n': len(records), 'none_available': len(ids) - len(records),
               'fields': ['tweet_id', 'selection_method', 'post_type', 'text_excerpt'],
               'posts': records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'authors': len(ids), 'examples': len(records), 'missing': len(ids)-len(records),
                      'types': dict(Counter(r[2] for r in records.values())),
                      'sha256': hashlib.sha256(a.output.read_bytes()).hexdigest()}), flush=True)

if __name__ == '__main__':
    main()
