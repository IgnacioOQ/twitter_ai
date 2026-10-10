"""Fit an independent author-level unigram LDA model with explicit parameters."""
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
from .common import configuration, exact_id

CONFIG = configuration()
log = print

def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

CUSTOM_STOPWORDS = {
    'ai', 'artificial', 'intelligence', 'chatgpt', 'gpt', 'openai', 'google',
    'microsoft', 'machine', 'learning', 'ml', 'neural', 'network', 'model',
    'data', 'algorithm', 'robot', 'automation', 'tech', 'technology',
    'http', 'https', 'www', 'com', 'co', 'amp', 'rt', 'like', 'just',
    'know', 'think', 'want', 'need', 'got', 'going', 'would', 'could',
    'really', 'actually', 'basically', 'probably', 'maybe', 'also',
    'one', 'two', 'three', 'first', 'second', 'new', 'good', 'great',
    'people', 'time', 'way', 'thing', 'things', 'lot', 'much', 'many'
}

def clean_text(text):
    """Clean text for topic modelling."""
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def aggregate_author_docs(tweets_path):
    """Aggregate tweets by author into single documents."""
    log(f'  Aggregating tweets from {tweets_path}')

    author_texts = defaultdict(list)
    for tweet in read_jsonl(tweets_path):
        author_id = exact_id(tweet.get('author_id'))
        text = tweet.get('processed_text', '')
        if author_id and text:
            cleaned = clean_text(text)
            if cleaned:
                author_texts[author_id].append(cleaned)

    author_docs = {}
    for author_id, texts in author_texts.items():
        author_docs[author_id] = ' '.join(texts)

    log(f'    {len(author_docs)} authors with documents')
    return author_docs


def train_and_evaluate_model(author_docs, k, alpha, eta, random_state=42):
    """Train LDA model and return metrics."""
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, CoherenceModel
    from sklearn.feature_extraction import text as sklearn_text
    import numpy as np

    author_ids = list(author_docs.keys())
    documents = list(author_docs.values())

    all_stopwords = list(sklearn_text.ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))

    # Tokenize for gensim
    texts = [doc.split() for doc in documents]
    texts = [[w for w in doc if w not in all_stopwords and len(w) > 2] for doc in texts]

    # Create dictionary and corpus
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=CONFIG['min_documents'], no_above=0.7)
    if not len(dictionary):
        raise ValueError('No vocabulary survives the document-frequency filter')
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        alpha=alpha,
        eta=eta,
        passes=10,
        iterations=100,
        random_state=random_state,
        per_word_topics=False
    )

    # Calculate coherence
    coherence_model = CoherenceModel(
        model=lda,
        corpus=corpus,
        dictionary=dictionary,
        texts=texts,
        coherence='c_v', processes=1
    )
    coherence = coherence_model.get_coherence()

    # Calculate diversity
    topics = lda.show_topics(num_topics=k, num_words=10, formatted=False)
    all_words = []
    for _, words in topics:
        all_words.extend([w for w, _ in words])
    diversity = len(set(all_words)) / len(all_words) if all_words else 0

    # Calculate largest topic percentage
    topic_counts = Counter()
    for doc in corpus:
        doc_topics = lda.get_document_topics(doc)
        if doc_topics:
            dominant = max(doc_topics, key=lambda x: x[1])[0]
            topic_counts[dominant] += 1
    largest_pct = max(topic_counts.values()) / len(corpus) if topic_counts else 0

    # Get topic terms
    topic_terms = []
    for topic_id in range(k):
        terms = lda.show_topic(topic_id, topn=15)
        topic_terms.append({
            'topic_id': topic_id,
            'terms': ', '.join([f'{w}({p:.3f})' for w, p in terms[:10]]),
            'n_authors': topic_counts.get(topic_id, 0)
        })

    return {
        'coherence': coherence,
        'diversity': diversity,
        'largest_topic_pct': largest_pct,
        'model': lda,
        'dictionary': dictionary,
        'corpus': corpus,
        'topic_terms': topic_terms,
        'author_ids': author_ids
    }


def main():
    root = Path(CONFIG['content'])
    docs = aggregate_author_docs(root / 'ai_general_eligible_tweets.json')
    if not docs:
        raise ValueError('No author documents')
    result = train_and_evaluate_model(docs, CONFIG['topics'], CONFIG['alpha'], CONFIG['eta'], CONFIG['seed'])
    result['model'].save(str(root / 'models/selected_lda_model.model'))
    result['dictionary'].save(str(root / 'models/selected_dictionary.dict'))
    pd.DataFrame(result['topic_terms']).to_csv(root / 'topic_terms.csv', index=False)
    rows = []
    for author, document in zip(result['author_ids'], result['corpus']):
        membership = result['model'].get_document_topics(document, minimum_probability=0)
        dominant = max(membership, key=lambda item: item[1])
        rows.append(dict(author_id=author, dominant_topic=dominant[0], dominant_prob=float(dominant[1]),
                         **{f'topic_{i}': float(p) for i, p in membership}))
    pd.DataFrame(rows).to_csv(root / 'ai_general_author_topics.csv', index=False)
    metadata = {key: CONFIG[key] for key in ['topics', 'alpha', 'eta', 'seed', 'min_documents']}
    metadata.update(representation='unigram_counts', max_document_fraction=.7,
                    coherence=float(result['coherence']), diversity=float(result['diversity']),
                    largest_topic_share=float(result['largest_topic_pct']))
    (root / 'models/model_parameters.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
