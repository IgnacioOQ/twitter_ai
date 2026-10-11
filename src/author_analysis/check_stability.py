"""Compare fitted topics on identical authors; optionally fit a controlled seed repeat."""
import argparse
import copy
import json
import pickle
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score

from .common import sha256, topic_columns, validate_ids
from .topic_quality import assess_topic_fit


def read_fit(path):
    path = Path(path)
    meta = json.loads((path / 'model.json').read_text(encoding='utf-8'))
    for key, name in [('model_sha256', 'lda_model.pkl'), ('vectorizer_sha256', 'vectorizer.pkl'),
                      ('author_topics_sha256', 'author_topics.csv')]:
        if sha256(path / name) != meta[key]:
            raise ValueError(f'Fit fingerprint mismatch: {name}')
    frame = validate_ids(pd.read_csv(path / 'author_topics.csv', dtype={'author_id': str}))
    frame = frame.set_index('author_id').sort_index()
    cols = topic_columns(frame)
    assess_topic_fit(frame[cols].to_numpy())
    if len(cols) != meta['K'] or len(frame) != meta['n_authors']:
        raise ValueError('Fit dimensions disagree with metadata')
    return meta, frame


def compare(a, b, profiles=None, min_community=1000, enrichment_cutoff=.5):
    a = a.sort_index(); b = b.sort_index()
    if not a.index.is_unique or not a.index.equals(b.index) or a.shape != b.shape:
        raise ValueError('Both fits must contain exactly the same unique authors and topics')
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    assess_topic_fit(x); assess_topic_fit(y)
    norms = np.linalg.norm(x, axis=0)[:, None] * np.linalg.norm(y, axis=0)[None, :]
    if (norms == 0).any():
        raise ValueError('Cannot match a topic with zero membership mass')
    cosine = x.T @ y / norms
    _, match = linear_sum_assignment(-cosine)
    aligned = y[:, match]
    da, db = x.argmax(1), y.argmax(1)
    aligned_labels = np.argsort(match)[db]
    contingency = np.zeros((x.shape[1], x.shape[1]), dtype=np.int64)
    np.add.at(contingency, (da, db), 1)
    ii, jj = linear_sum_assignment(-contingency)
    confident = (x.max(1) >= .6) & (y.max(1) >= .6)
    report = dict(authors=len(a), matching=match.tolist(),
        cosine_aligned_dominant_agreement=float((da == aligned_labels).mean()),
        maximum_possible_dominant_agreement=float(contingency[ii,jj].sum()/len(a)),
        adjusted_rand=float(adjusted_rand_score(da, db)),
        mean_total_variation=float(np.abs(x-aligned).sum(1).mean()/2),
        jointly_confident_authors=int(confident.sum()),
        jointly_confident_agreement=float((da[confident]==aligned_labels[confident]).mean()) if confident.any() else None,
        interpretation='Author-membership cosine matching is not semantic validation. Inspect terms and posts. Descriptive filters are not acceptance thresholds.',
        descriptive_filters=dict(min_community_authors=min_community, absolute_log2_enrichment=enrichment_cutoff))
    topics = pd.DataFrame(dict(topic_a=range(x.shape[1]), matched_topic_b=match,
        membership_cosine=cosine[np.arange(len(match)),match],
        prevalence_a=x.mean(0), prevalence_b=aligned.mean(0)))
    communities = pd.DataFrame()
    if profiles is not None:
        profiles = profiles.sort_index()
        if not profiles.index.is_unique or not profiles.index.equals(a.index):
            raise ValueError('Profiles and topic authors differ')
        net = (profiles.positive-profiles.negative).to_numpy(dtype=float)
        if not np.isfinite(net).all():
            raise ValueError('Non-finite affect values')
        wa = x.T @ net/x.sum(0); wb = aligned.T @ net/aligned.sum(0)
        topics['net_sentiment_a'] = wa; topics['net_sentiment_b'] = wb
        topics['relative_net_a'] = wa-net.mean(); topics['relative_net_b'] = wb-net.mean()
        report.update(max_topic_net_sentiment_difference=float(np.max(np.abs(wa-wb))),
            net_sentiment_rank_correlation=float(spearmanr(wa,wb).statistic),
            relative_net_sentiment_sign_flips=int(((wa-net.mean())*(wb-net.mean())<0).sum()))
        rows = []
        for cid, group in profiles.groupby('leiden_directed'):
            mask = profiles.index.isin(group.index)
            ca, cb = x[mask].mean(0), aligned[mask].mean(0)
            for t in range(x.shape[1]):
                rows.append(dict(community=int(cid), authors=int(mask.sum()), topic_a=t,
                    matched_topic_b=int(match[t]), composition_a=float(ca[t]), composition_b=float(cb[t]),
                    log2_enrichment_a=float(np.log2(ca[t]/x[:,t].mean())) if ca[t]>0 else None,
                    log2_enrichment_b=float(np.log2(cb[t]/aligned[:,t].mean())) if cb[t]>0 else None))
        communities = pd.DataFrame(rows)
        big = communities[communities.authors >= min_community].dropna()
        substantial = big[(big.log2_enrichment_a.abs() >= enrichment_cutoff) &
                          (big.log2_enrichment_b.abs() >= enrichment_cutoff)]
        report.update(community_cells_substantial_both=len(substantial),
            community_substantial_sign_flips=int((substantial.log2_enrichment_a*substantial.log2_enrichment_b<0).sum()),
            community_enrichment_correlation_n_ge_min=float(spearmanr(big.log2_enrichment_a,big.log2_enrichment_b).statistic) if len(big)>1 else None)
    return report, topics, communities


def clean_json(value):
    if isinstance(value, dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value, list): return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def repeat_fit(reference, output, seed, jobs):
    meta, frame = read_fit(reference)
    if output.exists(): raise ValueError('Repeat output must not already exist')
    if output.resolve().is_relative_to(reference.resolve()):
        raise ValueError('Repeat output must be separate from the reference')
    documents = reference/'author_documents.jsonl'
    if sha256(documents) != meta['documents']['documents_sha256']:
        raise ValueError('Author document fingerprint mismatch')
    docs = {}
    with documents.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['author_id'] in docs: raise ValueError('Duplicate document author')
            docs[row['author_id']] = row['text']
    if set(docs) != set(frame.index): raise ValueError('Document authors differ')
    with (reference/'vectorizer.pkl').open('rb') as f: vectorizer = pickle.load(f)
    with (reference/'lda_model.pkl').open('rb') as f: model = pickle.load(f)
    model = clone(model).set_params(random_state=seed, n_jobs=jobs)
    matrix = vectorizer.transform([docs[i] for i in frame.index])
    model.fit(matrix)
    theta = model.transform(matrix).astype(np.float32)
    quality = assess_topic_fit(theta)
    output.mkdir(parents=True, exist_ok=False)
    result = frame.copy()
    result[topic_columns(frame)] = theta
    result['dominant_topic'] = theta.argmax(1); result['dominant_weight'] = theta.max(1)
    result.reset_index().to_csv(output/'author_topics.csv',index=False)
    shutil.copy2(reference/'vectorizer.pkl',output/'vectorizer.pkl')
    with (output/'lda_model.pkl').open('wb') as f: pickle.dump(model,f,pickle.HIGHEST_PROTOCOL)
    meta = copy.deepcopy(meta)
    meta['settings']['seed'] = seed; meta['settings']['n_jobs'] = jobs
    meta['topic_quality'] = quality
    meta['diagnostic_only'] = True
    for key,name in [('model_sha256','lda_model.pkl'),('vectorizer_sha256','vectorizer.pkl'),('author_topics_sha256','author_topics.csv')]:
        meta[key] = sha256(output/name)
    (output/'model.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--candidate',type=Path)
    choice.add_argument('--repeat-seed',type=int)
    p.add_argument('--profiles',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--min-community',type=int,default=1000)
    p.add_argument('--enrichment-cutoff',type=float,default=.5)
    a = p.parse_args()
    if a.jobs < 1 or a.min_community < 1 or a.enrichment_cutoff < 0: p.error('Invalid diagnostic settings')
    if a.output.exists(): p.error('Output must be a new directory')
    candidate = a.candidate
    if a.repeat_seed is not None:
        candidate = a.output/'repeat'
        repeat_fit(a.reference,candidate,a.repeat_seed,a.jobs)
    ma,fa = read_fit(a.reference); mb,fb = read_fit(candidate)
    if ma['documents']['documents_sha256'] != mb['documents']['documents_sha256']:
        raise ValueError('Fits use different author documents')
    with (a.reference/'vectorizer.pkl').open('rb') as f: va=pickle.load(f)
    with (candidate/'vectorizer.pkl').open('rb') as f: vb=pickle.load(f)
    same = va.vocabulary_ == vb.vocabulary_ and np.array_equal(va.idf_,vb.idf_)
    if not same: raise ValueError('Controlled comparison requires identical fitted vocabulary and IDF')
    profiles = None
    if a.profiles:
        profiles = pd.read_parquet(a.profiles) if a.profiles.suffix=='.parquet' else pd.read_csv(a.profiles,dtype={'author_id':str})
        profiles = validate_ids(profiles).set_index('author_id')
    report, topics, communities = compare(fa[topic_columns(fa)],fb[topic_columns(fb)],profiles,a.min_community,a.enrichment_cutoff)
    report['identical_fitted_features'] = same
    report['reference_model_sha256'] = ma['model_sha256']; report['candidate_model_sha256'] = mb['model_sha256']
    a.output.mkdir(parents=True,exist_ok=a.repeat_seed is not None)
    (a.output/'comparison.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False),encoding='utf-8')
    topics.to_csv(a.output/'topic_outcomes.csv',index=False)
    if not communities.empty: communities.to_csv(a.output/'community_outcomes.csv',index=False)
    print(json.dumps(clean_json(report),allow_nan=False))


if __name__=='__main__': main()
