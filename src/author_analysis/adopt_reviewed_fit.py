"""Copy a reviewed, fingerprint-verified topic bundle into a new output directory."""
import argparse
import json
import pickle
from pathlib import Path
import shutil
import numpy as np

from .check_stability import read_fit
from .common import sha256, topic_columns
from .topic_quality import assess_topic_fit, require_topic_review_passed


def adopt(source, prepared, output, review):
    source, prepared, output = map(Path,(source,prepared,output))
    if output.exists():
        raise ValueError('Adoption output must not already exist')
    if any(output.resolve().is_relative_to(p.resolve()) for p in [source,prepared]):
        raise ValueError('Output must be separate from source and prepared inputs')
    meta, frame = read_fit(source)
    require_topic_review_passed(meta)
    quality = assess_topic_fit(frame[topic_columns(frame)].to_numpy(),
        selection_largest_share=meta.get('topic_quality',{}).get('selection_largest_topic_share'))
    require_topic_review_passed({'topic_quality':quality})
    if meta.get('diagnostic_only'):
        raise ValueError('Diagnostic-only fits cannot be adopted by this command')
    required = ['semantic_review_completed','sensitivity_review_completed','approved_for_exploratory_use']
    if any(review.get(k) is not True for k in required):
        raise ValueError('Explicit semantic and sensitivity review is required')
    if review.get('model_sha256') != meta['model_sha256']:
        raise ValueError('Review does not identify this exact model')
    docs = json.loads((prepared/'documents.json').read_text(encoding='utf-8'))
    for key in ['documents_sha256','tweets_sha256','communities_sha256','backbone','includes_retweets']:
        if docs.get(key) is None or docs[key] != meta['documents'].get(key):
            raise ValueError(f'Prepared inputs differ: {key}')
    if sha256(prepared/'author_documents.jsonl') != docs['documents_sha256']:
        raise ValueError('Prepared document fingerprint mismatch')
    ids = set()
    sample = {}
    sample_ids = set(frame.index[:64])
    with (prepared/'author_documents.jsonl').open(encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['author_id'] in ids: raise ValueError('Duplicate author document')
            ids.add(record['author_id'])
            if record['author_id'] in sample_ids: sample[record['author_id']] = record['text']
    if ids != set(frame.index): raise ValueError('Prepared authors differ from fitted authors')
    with (source/'vectorizer.pkl').open('rb') as f: vectorizer = pickle.load(f)
    with (source/'lda_model.pkl').open('rb') as f: model = pickle.load(f)
    model.n_jobs = 1
    ordered = sorted(sample)
    computed = model.transform(vectorizer.transform([sample[i] for i in ordered]))
    if not np.allclose(computed,frame.loc[ordered,topic_columns(frame)].to_numpy(),atol=1e-5,rtol=0):
        raise ValueError('Saved author memberships disagree with the fitted model')
    output.mkdir(parents=True,exist_ok=False)
    for name in ['lda_model.pkl','vectorizer.pkl','author_topics.csv','top_terms.csv','grid_results.csv']:
        if (source/name).is_file(): shutil.copy2(source/name,output/name)
    for name in ['author_documents.jsonl','documents.json']:
        shutil.copy2(prepared/name,output/name)
    meta.pop('correction_review',None); meta.pop('supersedes_model_sha256',None)
    meta['review'] = {k:review[k] for k in required}
    meta['model_use'] = 'Exploratory topics; semantic and sensitivity qualifications apply.'
    (output/'model.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return {'authors':len(frame),'topics':meta['K'],'model_sha256':meta['model_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','prepared','output','review']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(adopt(a.source,a.prepared,a.output,json.loads(a.review.read_text(encoding='utf-8')))))


if __name__=='__main__': main()
