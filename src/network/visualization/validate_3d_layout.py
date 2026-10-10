"""Validate a generated 3D viewer bundle against its metadata and source order."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import read_payload, sha256
from ...author_analysis.common import exact_id


def validate(directory):
    viewer_path = directory / 'nodes.json'
    viewer = json.loads(viewer_path.read_text(encoding='utf-8'))
    ids = [exact_id(row['id']) for row in viewer]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate viewer author IDs')
    meta = json.loads((directory / 'layout3d_meta.json').read_text(encoding='utf-8'))
    if meta.get('schema_version') != 1 or meta.get('algorithm') != 'Graphviz sfdp with graph-Laplacian 2D constraint':
        raise ValueError('Unsupported layout metadata')
    values, indices = read_payload(directory / 'layout3d.bin', directory / 'layout3d_indices.bin', len(viewer))
    if meta['node_count'] != len(viewer) or meta['rendered_node_count'] != len(indices):
        raise ValueError('Metadata counts do not match the payload')
    if meta['viewer_nodes_sha256'] != sha256(viewer_path):
        raise ValueError('Viewer nodes changed after layout generation')
    for key, name in [('coordinates', 'layout3d.bin'), ('render_indices', 'layout3d_indices.bin')]:
        if meta[key]['sha256'] != sha256(directory / name):
            raise ValueError('Layout payload hash mismatch')
    eigenvalues = np.linalg.eigvalsh(np.cov(values[indices], rowvar=False))[::-1]
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise ValueError('Layout has no finite spatial extent')
    return {'valid': True, 'nodes': len(viewer), 'rendered_nodes': len(indices),
            'covariance_eigenvalue_ratios': (eigenvalues / eigenvalues[0]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir), indent=2))


if __name__ == '__main__':
    main()
