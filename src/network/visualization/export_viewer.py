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
    output = root / 'viewer'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'nodes.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    (output / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Exported {len(payload)} nodes in matching binary-layout order')


if __name__ == '__main__':
    main()
