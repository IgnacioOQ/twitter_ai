"""Prepare the current matched tables and matrices for figure rendering."""
import json
from pathlib import Path
import shutil
import pandas as pd
from .common import configuration, validate_ids, topic_columns, topic_labels


def main():
    config = configuration()
    root = Path(config['network'])
    matched = validate_ids(pd.read_parquet(config['matched']))
    output = root / 'figure_inputs'
    columns = topic_columns(matched)
    labels = topic_labels(config['topic_labels'], len(columns))
    if config['community_algorithm'] != 'leiden_directed':
        matched = matched.rename(columns={config['community_algorithm']:'leiden_directed'})
    for folder in ['tables','matrices']:(output/folder).mkdir(parents=True,exist_ok=True)
    matched.to_json(output/'matched_authors.json',orient='records',double_precision=15)
    matched.to_csv(output/'matched_authors.csv',index=False)
    (output/'matched_author_ids.json').write_text(json.dumps(matched.author_id.tolist()),encoding='utf-8')
    pd.DataFrame({'topic_id':list(labels),'label':list(labels.values())}).to_csv(output/'topic_labels.csv',index=False)
    counts=matched.groupby('leiden_directed').size().sort_values(ascending=False).head(21)
    pd.DataFrame({'community_id':counts.index,'display_label':[f'Community {i}' for i in counts.index],
                  'n_authors':counts.values}).to_csv(output/'community_labels.csv',index=False)
    for name in ['corpus_sentiment_network_subset.csv','corpus_emotions_network_subset.csv',
                 'topic_prevalence_dominant_network_subset.csv','topic_prevalence_fuzzy_network_subset.csv',
                 'topic_sentiment_fuzzy_network_subset.csv','topic_emotion_fuzzy_network_subset.csv']:
        shutil.copyfile(root/'tables'/name,output/'tables'/name)
    for name in ['topic_composition.csv','topic_capture.csv','topic_enrichment.csv','sentiment_means.csv',
                 'emotion_means.csv','emotion_standardized.csv']:
        shutil.copyfile(root/'communities/figures'/name,output/'matrices'/name)


if __name__=='__main__':main()
