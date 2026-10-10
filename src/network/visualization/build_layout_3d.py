"""Generate native Graphviz 3D coordinates and apply the 2D macro constraint."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import cg
from .common import load_graph, sha256

def normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centre = np.median(points, axis=0)
    centred = points - centre
    scale = float(np.quantile(np.linalg.norm(centred, axis=1), 0.995))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid robust coordinate scale")
    return centred / scale, centre, scale


def solve_displacement(
    displacement: np.ndarray,
    adjacency: sparse.csr_matrix,
    strength: float = 0.10,
    rms_ratio: float = 0.80,
) -> tuple[np.ndarray, dict[str, float]]:
    weighted = adjacency.astype(np.float64, copy=True)
    weighted.data = np.log1p(weighted.data)
    degree = np.asarray(weighted.sum(axis=1)).ravel()
    positive_degree = degree[degree > 0]
    median_degree = float(np.median(positive_degree))
    laplacian = sparse.diags(degree) - weighted
    system = (
        sparse.eye(len(degree), format="csr")
        + (strength / median_degree) * laplacian
    )
    preconditioner = sparse.diags(1.0 / system.diagonal())
    field = np.empty_like(displacement)
    iterations: list[int] = []
    for axis in range(2):
        iteration_count = 0

        def count_iteration(_: np.ndarray) -> None:
            nonlocal iteration_count
            iteration_count += 1

        field[:, axis], info = cg(
            system,
            displacement[:, axis],
            M=preconditioner,
            rtol=5e-6,
            atol=0.0,
            maxiter=1_200,
            callback=count_iteration,
        )
        if info != 0:
            raise RuntimeError(f"CG failed for axis {axis}: {info}")
        iterations.append(iteration_count)
    source_rms = float(np.sqrt(np.mean(np.square(displacement))))
    field_rms = float(np.sqrt(np.mean(np.square(field))))
    gain = rms_ratio * source_rms / max(field_rms, 1e-12)
    residual = system @ field - displacement
    return gain * field, {
        "median_log_weighted_degree": median_degree,
        "unscaled_source_rms": source_rms,
        "unscaled_field_rms": field_rms,
        "computed_field_gain": gain,
        "cg_iterations_x": iterations[0],
        "cg_iterations_y": iterations[1],
        "cg_max_absolute_residual": float(np.max(np.abs(residual))),
    }


def build(args):
    if sys.byteorder != 'little':
        raise ValueError('The native coordinate helper requires a little-endian host')
    nodes, edges, viewer, graph = load_graph(args.nodes, args.edges, args.viewer_nodes)
    count, membership = csgraph.connected_components(graph, directed=False)
    indices = np.flatnonzero(membership == np.bincount(membership).argmax())
    if len(indices) < 4:
        raise ValueError('The largest component needs at least four nodes for a 3D layout')
    adjacency = graph[indices][:, indices].tocsr()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / 'build'
    work.mkdir(exist_ok=True)
    dot = work / 'component.dot'
    upper = sparse.triu(adjacency, k=1, format='coo')
    with dot.open('w', encoding='ascii') as handle:
        handle.write('strict graph G {\n')
        for i in range(len(indices)):
            handle.write(f'{i};\n')
        for i, j in zip(upper.row, upper.col):
            handle.write(f'{i} -- {j};\n')
        handle.write('}\n')
    raw_path = work / 'native.f64'
    subprocess.run([str(args.sfdp_runner.resolve()), str(dot.resolve()), str(raw_path.resolve()), str(args.seed)], check=True)
    raw = np.fromfile(raw_path, dtype='<f8')
    if raw.size != len(indices) * 3 or not np.isfinite(raw).all():
        raise ValueError('Invalid native Graphviz coordinate payload')
    raw = raw.reshape(-1, 3)
    centred = raw - np.median(raw, axis=0)
    xy = nodes[['x', 'y']].to_numpy(dtype=float)[indices]
    target = np.column_stack([xy - np.median(xy, axis=0), np.zeros(len(indices))])
    u, _, vt = np.linalg.svd(centred.T @ target)
    native, _, _ = normalise(centred @ (u @ vt))
    reference, _, _ = normalise(xy)
    displacement, solver = solve_displacement(reference - native[:, :2], adjacency,
                                             args.strength, args.displacement_ratio)
    constrained = native.copy()
    constrained[:, :2] += displacement
    coordinates, _, _ = normalise(constrained)
    full = np.zeros((len(nodes), 3), dtype='<f4')
    full[indices] = coordinates.astype('<f4')
    output = args.output_dir / 'layout3d.bin'
    index_output = args.output_dir / 'layout3d_indices.bin'
    full.tofile(output)
    indices.astype('<u4').tofile(index_output)
    metadata = dict(schema_version=1, algorithm='Graphviz sfdp with graph-Laplacian 2D constraint',
                    node_count=len(nodes), rendered_node_count=len(indices), edge_count=len(edges),
                    rendered_undirected_edges=int(upper.nnz), component_count=int(count), seed=args.seed,
                    construction=dict(laplacian_strength=args.strength,
                                      displacement_rms_ratio=args.displacement_ratio,
                                      native_depth_multiplier=1.0,
                                      graph_weights='native layout unweighted; constraint uses log1p summed weights',
                                      communities_or_author_attributes_used=False),
                    coordinates=dict(sha256=sha256(output)), render_indices=dict(sha256=sha256(index_output)),
                    viewer_nodes_sha256=sha256(args.viewer_nodes), solver=solver,
                    source=dict(nodes_sha256=sha256(args.nodes), edges_sha256=sha256(args.edges),
                                native_runner_sha256=sha256(args.sfdp_runner)),
                    normalised_bounds=dict(radius_p995=1.0,
                                           max_radius=float(np.linalg.norm(coordinates, axis=1).max())))
    (args.output_dir / 'layout3d_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'nodes':len(nodes), 'rendered_nodes':len(indices)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['nodes', 'edges', 'viewer-nodes', 'sfdp-runner', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strength', type=float, default=.10)
    parser.add_argument('--displacement-ratio', type=float, default=.80)
    args = parser.parse_args()
    if args.strength < 0 or args.displacement_ratio < 0:
        parser.error('Constraint parameters must be non-negative')
    build(args)


if __name__ == '__main__':
    main()
