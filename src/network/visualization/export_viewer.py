"""Export analytical nodes in the compact schema consumed by the browser viewer."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ...author_analysis.common import configuration, validate_ids, topic_columns, EMOTIONS


def export_nodes(nodes, algorithm):
    nodes = validate_ids(nodes)
    mapping = dict(author_id='id', x='x', y='y', full_degree='fd', matched_degree='md',
                   dominant_topic='dt', dominant_prob='dp', positive='sp', neutral='sn', negative='sg')
    mapping[algorithm] = 'c'
    mapping.update({name: f't{i}' for i, name in enumerate(topic_columns(nodes))})
    mapping.update({name: 'e' + chr(97 + i) for i, name in enumerate(EMOTIONS)})
    missing = set(mapping) - set(nodes)
    if missing:
        raise ValueError(f'Missing viewer fields: {sorted(missing)}')
    output = nodes[list(mapping)].rename(columns=mapping)
    if not np.isfinite(output.drop(columns='id').to_numpy(dtype=float)).all():
        raise ValueError('Viewer values must be finite')
    # Pandas serialises exact author strings without numeric conversion.
    return json.loads(output.to_json(orient='records', double_precision=15))


def main():
    config = configuration()
    root = Path(config['network'])
    nodes = pd.read_parquet(root / 'layout/matched_nodes.parquet')
    payload = export_nodes(nodes, config['community_algorithm'])
    summary = json.loads((root / 'layout/build_summary.json').read_text(encoding='utf-8'))
    if summary['matched_authors'] != len(payload):
        raise ValueError('Node count and graph summary disagree')
    from ...author_analysis.common import topic_labels, topic_presentation
    count = len(topic_columns(nodes))
    summary['topic_labels'] = list(topic_labels(config.get('topic_labels'),count).values())
    if config.get('topic_labels'):
        presentation = topic_presentation(config['topic_labels'])
        if 'colour' in presentation:
            summary['topic_colours'] = presentation.colour.tolist()
    metadata_path = root/'topics/model.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        summary['topic_model'] = {k:metadata[k] for k in ['K','representation','alpha','eta'] if k in metadata}
        summary['analysis_status'] = metadata.get('model_use','Exploratory topics; inspect semantic quality and sensitivity before interpretation.')
    output = root / 'viewer'
    output.mkdir(parents=True, exist_ok=True)
    if config.get('community_labels'):
        labels = pd.read_csv(config['community_labels'],keep_default_na=False)
        sizes = nodes.groupby(config['community_algorithm']).size()
        if (labels.community_id.duplicated().any()
                or not set(labels.community_id).issubset(sizes.index)
                or not all(sizes.loc[r.community_id] == r.n_authors for r in labels.itertuples())):
            raise ValueError('Community descriptions do not match viewer authors')
        records = {}
        for row in labels.to_dict('records'):
            row['display_label'] = row.get('label',row['display_label'])
            records[str(row['community_id'])] = row
        (output/'community_labels.json').write_text(json.dumps({
            'communities':records,'order':labels.sort_values('n_authors',ascending=False).community_id.tolist()
        },ensure_ascii=False,indent=2),encoding='utf-8')
    (output / 'nodes.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    (output / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Exported {len(payload)} nodes in matching binary-layout order')


if __name__ == '__main__':
    main()
