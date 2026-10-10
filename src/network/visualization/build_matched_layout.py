"""Build the weighted matched-author graph and its two-dimensional DrL layout."""
import json
import random
from pathlib import Path
import igraph as ig
import numpy as np
import pandas as pd
from ...author_analysis.common import configuration, validate_ids, exact_id, topic_columns, topic_labels


def main():
    config = configuration()
    nodes = validate_ids(pd.read_parquet(config['matched'])).sort_values('author_id').reset_index(drop=True)
    path = Path(config['graph'])
    readers = {'.gml': ig.Graph.Read_GML, '.graphml': ig.Graph.Read_GraphML}
    if path.suffix.lower() not in readers:
        raise ValueError('Graph must be GML or GraphML')
    graph = readers[path.suffix.lower()](str(path))
    attribute = config['author_id_attribute']
    if attribute not in graph.vs.attributes():
        raise ValueError('The selected author ID vertex attribute is absent')
    ids = [exact_id(value) for value in graph.vs[attribute]]
    if len(set(ids)) != len(ids):
        raise ValueError('Graph contains duplicate author IDs')
    lookup = {aid: i for i, aid in enumerate(ids)}
    if not set(nodes.author_id).issubset(lookup):
        raise ValueError('Matched author IDs are missing from the selected graph')
    weights = np.asarray(graph.es['weight'] if 'weight' in graph.es.attributes() else np.ones(graph.ecount()), dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Graph edge weights must be finite and positive')
    graph.es['weight'] = weights
    selected = [lookup[aid] for aid in nodes.author_id]
    subgraph = graph.induced_subgraph(selected)
    # igraph preserves source graph order in induced subgraphs: realign explicitly.
    sub_ids = [exact_id(value) for value in subgraph.vs[attribute]]
    nodes = nodes.set_index('author_id').loc[sub_ids].reset_index()
    ig.set_random_number_generator(random.Random(config['seed']))
    if subgraph.vcount() < 2:
        raise ValueError('A network layout requires at least two matched authors')
    positions = np.asarray(subgraph.layout_drl(weights='weight').coords)
    if not np.isfinite(positions).all():
        raise ValueError('Layout produced non-finite coordinates')
    nodes['x'], nodes['y'] = positions[:, 0], positions[:, 1]
    full_degree = graph.strength(weights='weight', mode='all')
    nodes['full_degree'] = [full_degree[lookup[aid]] for aid in sub_ids]
    nodes['matched_degree'] = subgraph.strength(weights='weight', mode='all')
    edges = pd.DataFrame([(sub_ids[e.source], sub_ids[e.target], e['weight']) for e in subgraph.es],
                         columns=['source', 'target', 'weight'])
    sizes = subgraph.connected_components(mode='weak').sizes()
    out = Path(config['network']) / 'layout'
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out / 'matched_nodes.parquet', index=False)
    edges.to_parquet(out / 'matched_edges.parquet', index=False)
    summary = dict(matched_authors=len(nodes), matched_edges=len(edges),
                   full_network_nodes=graph.vcount(), full_network_edges=graph.ecount(),
                   connected_components=len(sizes), largest_component=max(sizes),
                   isolates=sum(size == 1 for size in sizes), layout_mode='matched_only_drl',
                   seed=config['seed'], community_algorithm=config['community_algorithm'],
                   topic_labels=topic_labels(config['topic_labels'], len(topic_columns(nodes))),
                   note='Coordinates use matched-to-matched edges. Disconnected component positions are arbitrary.')
    (out / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
