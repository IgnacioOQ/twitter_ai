"""Select and fit a separate all-post topic model for one backbone."""
import gc
import itertools
import json
import pickle
import random
import time
import sklearn
import gensim
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from .common import configuration, exact_id, sha256, unique_object
from .topic_model import (ModelSettings, METHOD_VERSION, make_vectorizer, make_lda,
                          top_terms, diversity, collapse_ratio, coherence_cv)


def rank_grid(frame):
    frame = frame.copy()
    for col in ['coherence_c_v', 'diversity']:
        lo, hi = frame[col].min(), frame[col].max()
        frame[col + '_scaled'] = (frame[col]-lo)/(hi-lo) if hi > lo else .5
    frame['combined_score'] = (.25*frame.coherence_c_v_scaled.fillna(0) +
        .30*frame.diversity_scaled + .15*(1-frame.collapse_ratio) + .30*frame.evenness)
    return frame.sort_values('combined_score', ascending=False).reset_index(drop=True)


def fit(config):
    settings = ModelSettings(**config['model_settings'])
    output = Path(config['network']) / 'topics'
    path = output / 'author_documents.jsonl'
    prepared = json.loads((output/'documents.json').read_text(encoding='utf-8'))
    if (prepared['method_version'] != METHOD_VERSION or prepared['backbone'] != config['backbone'] or
        prepared['min_tweets'] != settings.min_tweets or prepared['smoke_test'] != config['smoke'] or
        prepared['documents_sha256'] != sha256(path) or
        prepared['communities_sha256'] != sha256(config['communities'])):
        raise ValueError('Prepared documents do not match the current method or inputs')
    docs = {}
    counts = {}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line); aid = exact_id(row['author_id'])
            if aid in docs: raise ValueError('Duplicate author documents')
            docs[aid] = row['text']; counts[aid] = row['n_tweets']
    ids = sorted(docs)
    sample_ids = sorted(random.Random(settings.seed).sample(ids, settings.grid_sample_authors)) if len(ids)>settings.grid_sample_authors else ids
    corpus = [docs[aid] for aid in sample_ids]
    indices = random.Random(settings.seed).sample(range(len(corpus)), min(settings.coherence_docs, len(corpus)))
    analyzer = make_vectorizer(settings).build_analyzer()
    texts = [analyzer(corpus[i]) for i in indices]
    dictionary = Dictionary(texts)
    rows = []
    for representation in settings.representations:
        vec = make_vectorizer(settings, representation)
        x = vec.fit_transform(corpus)
        vocab = vec.get_feature_names_out()
        for k, alpha, eta in itertools.product(settings.k_grid, settings.alpha_grid, settings.eta_grid):
            started = time.time()
            model = make_lda(settings, k, alpha, eta).fit(x)
            tops = top_terms(model, vocab, 15)
            dominant = model.transform(x).argmax(axis=1)
            largest = np.bincount(dominant, minlength=k).max()/len(dominant)
            rows.append({'backbone': config['backbone'], 'representation': representation,
                'K': k, 'alpha': alpha, 'eta': eta,
                'coherence_c_v': coherence_cv(tops, texts, dictionary, settings.n_jobs),
                'diversity': diversity(tops), 'collapse_ratio': collapse_ratio(tops),
                'largest_topic_share': largest, 'evenness': 1-max(0,largest-1/k),
                'perplexity': float(model.perplexity(x)), 'n_iter': model.n_iter_,
                'fit_minutes': (time.time()-started)/60})
            print(f'Completed {representation} K={k} alpha={alpha} eta={eta}', flush=True)
            pd.DataFrame(rows).to_csv(output/'grid_progress.csv', index=False)
        del x, vec, model
        gc.collect()
    grid = rank_grid(pd.DataFrame(rows))
    grid.to_csv(output/'grid_results.csv', index=False)
    rep, k, alpha, eta = settings.selected_model or tuple(grid.loc[0,['representation','K','alpha','eta']])
    vec = make_vectorizer(settings, rep)
    x = vec.fit_transform([docs[aid] for aid in ids])
    model = make_lda(settings, k, alpha, eta).fit(x)
    theta = model.transform(x).astype(np.float32)
    frame = pd.DataFrame(theta, columns=[f'topic_{i}' for i in range(int(k))])
    frame.insert(0, 'n_tweets', [counts[aid] for aid in ids])
    frame.insert(0, 'author_id', ids)
    frame['dominant_topic'] = theta.argmax(axis=1)
    frame['dominant_weight'] = theta.max(axis=1)
    # Keep every supplied partition, with the same exported column convention as the notebook.
    partitions = json.loads(Path(config['communities']).read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    methods = list(next(iter(partitions.values())))
    for aid, labels in partitions.items():
        exact_id(aid)
        if any(method not in labels or isinstance(labels[method],bool) or
               not isinstance(labels[method],int) or labels[method]<0 for method in methods):
            raise ValueError('Invalid community assignment')
    for method in methods:
        frame['community_'+method] = [partitions[aid][method] for aid in ids]
    frame.to_csv(output/'author_topics.csv', index=False)
    for name, obj in [('vectorizer.pkl',vec),('lda_model.pkl',model)]:
        with (output/name).open('wb') as handle: pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    vocab = vec.get_feature_names_out()
    pd.DataFrame([{'topic': t, 'rank': r, 'term': vocab[i], 'weight': float(component[i])}
        for t, component in enumerate(model.components_)
        for r, i in enumerate(component.argsort()[::-1][:20])]).to_csv(output/'top_terms.csv', index=False)
    metadata = {'method_version': METHOD_VERSION, 'backbone': config['backbone'],
        'includes_retweets': True, 'smoke_test': config['smoke'], 'representation': rep,
        'K': int(k), 'alpha': float(alpha), 'eta': float(eta), 'n_authors':len(ids),
        'settings': settings.to_dict(), 'documents': prepared,
        'versions': {'sklearn':sklearn.__version__, 'gensim':gensim.__version__, 'numpy':np.__version__},
        'model_sha256':sha256(output/'lda_model.pkl'), 'vectorizer_sha256':sha256(output/'vectorizer.pkl'),
        'author_topics_sha256':sha256(output/'author_topics.csv')}
    (output/'model.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Final {config["backbone"]}: {len(ids):,} authors, {k} topics', flush=True)


def main(): fit(configuration())

if __name__ == '__main__': main()
