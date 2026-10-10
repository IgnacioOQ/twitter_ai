"""Load and validate graph/viewer inputs without changing author order."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from ...author_analysis.common import validate_ids, exact_id


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_graph(nodes_path, edges_path, viewer_path):
    nodes = validate_ids(pd.read_parquet(nodes_path))
    viewer = json.loads(Path(viewer_path).read_text(encoding='utf-8'))
    if nodes.author_id.tolist() != [exact_id(row['id']) for row in viewer]:
        raise ValueError('Viewer and matched-node author order differ')
    if not np.allclose(nodes[['x', 'y']], [[row['x'], row['y']] for row in viewer], atol=1e-10):
        raise ValueError('Viewer and matched-node coordinates differ')
    edges = pd.read_parquet(edges_path)
    lookup = {aid: i for i, aid in enumerate(nodes.author_id)}
    source = edges.source.map(exact_id).map(lookup)
    target = edges.target.map(exact_id).map(lookup)
    if source.isna().any() or target.isna().any():
        raise ValueError('An edge endpoint is missing from nodes')
    weights = edges.weight.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Edge weights must be finite and positive')
    source, target = source.to_numpy(dtype=int), target.to_numpy(dtype=int)
    mask = source != target
    directed = sparse.coo_matrix((weights[mask], (source[mask], target[mask])), shape=(len(nodes), len(nodes))).tocsr()
    adjacency = directed + directed.T
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    return nodes, edges, viewer, adjacency


def read_payload(coordinates, indices, count):
    values = np.fromfile(coordinates, dtype='<f4')
    index = np.fromfile(indices, dtype='<u4').astype(np.int64)
    if values.size != count * 3 or not len(index):
        raise ValueError('Invalid coordinate or index payload size')
    if len(np.unique(index)) != len(index) or index.max() >= count:
        raise ValueError('Render indices are duplicated or out of range')
    values = values.reshape(count, 3).astype(float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite coordinates')
    mask = np.ones(count, dtype=bool)
    mask[index] = False
    if np.any(values[mask] != 0):
        raise ValueError('Non-rendered rows must be zero')
    return values, index
