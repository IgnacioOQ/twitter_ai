#!/usr/bin/env python3
"""Evaluate a 3D network layout against the observed graph and geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


from .common import load_graph, read_payload, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--viewer-nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-sample", type=int, default=150_000)
    parser.add_argument("--knn-node-sample", type=int, default=6_000)
    parser.add_argument("--knn", type=int, default=15)
    return parser.parse_args()


def percentile(values: np.ndarray, points: list[float]) -> dict[str, float]:
    result = np.quantile(values, points)
    return {f"p{point * 100:g}": float(value) for point, value in zip(points, result)}


def procrustes_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference - reference.mean(axis=0)
    candidate = candidate - candidate.mean(axis=0)
    singular_values = np.linalg.svd(reference.T @ candidate, compute_uv=False)
    denominator = float(np.square(reference).sum() * np.square(candidate).sum())
    return float(np.square(singular_values.sum()) / denominator)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    nodes, edges, viewer, graph = load_graph(args.nodes, args.edges, args.viewer_nodes)
    NODE_COUNT = len(nodes)
    positions, indices = read_payload(args.coordinates, args.indices, NODE_COUNT)
    points = positions[indices]
    if len(indices) < 4 or args.knn < 1 or args.pair_sample < 1:
        raise ValueError('Evaluation requires at least four rendered nodes and positive sample sizes')
    args.knn = min(args.knn, len(indices) - 1)
    lookup = pd.Series(np.arange(NODE_COUNT), index=nodes.author_id)
    source = edges.source.map(lookup).to_numpy(dtype=int)
    target = edges.target.map(lookup).to_numpy(dtype=int)
    weights = edges.weight.to_numpy(dtype=float)

    local_index = np.full(NODE_COUNT, -1, dtype=np.int32)
    local_index[indices] = np.arange(len(indices), dtype=np.int32)
    edge_mask = (local_index[source] >= 0) & (local_index[target] >= 0) & (source != target)
    local_source = local_index[source[edge_mask]]
    local_target = local_index[target[edge_mask]]
    local_weight = weights[edge_mask]
    adjacency = sparse.coo_matrix(
        (np.ones(len(local_source), dtype=np.uint8), (local_source, local_target)),
        shape=(len(indices), len(indices)),
    ).tocsr()
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.uint8).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.diff(adjacency.indptr)

    pair_count = min(args.pair_sample, len(local_source))
    chosen = rng.choice(len(local_source), size=pair_count, replace=False)
    edge_u = local_source[chosen]
    edge_v = local_target[chosen]
    edge_w = local_weight[chosen]
    edge_distance = np.linalg.norm(points[edge_u] - points[edge_v], axis=1)

    # Degree-matched non-edges: keep each sampled source and draw a target from
    # the same log-degree decile as the observed edge target.
    log_degree = np.log1p(degree)
    boundaries = np.unique(np.quantile(log_degree, np.linspace(0, 1, 11)))
    bins = np.digitize(log_degree, boundaries[1:-1], right=True)
    members = [np.flatnonzero(bins == group) for group in range(int(bins.max()) + 1)]
    negative_v = np.empty(pair_count, dtype=np.int32)
    unresolved = np.ones(pair_count, dtype=bool)
    for _ in range(128):
        pending = np.flatnonzero(unresolved)
        if len(pending) == 0:
            break
        for group in np.unique(bins[edge_v[pending]]):
            group_rows = pending[bins[edge_v[pending]] == group]
            pool = members[int(group)]
            negative_v[group_rows] = rng.choice(pool, size=len(group_rows), replace=True)
        is_self = negative_v[pending] == edge_u[pending]
        is_edge = np.asarray(adjacency[edge_u[pending], negative_v[pending]]).ravel() > 0
        unresolved[pending] = is_self | is_edge
    if unresolved.any():
        raise RuntimeError(f"Could not generate {int(unresolved.sum())} degree-matched non-edges")
    nonedge_distance = np.linalg.norm(points[edge_u] - points[negative_v], axis=1)

    sample_nodes = rng.choice(len(indices), size=min(args.knn_node_sample, len(indices)), replace=False)
    neighbor_model = NearestNeighbors(n_neighbors=args.knn + 1, algorithm="kd_tree", n_jobs=1)
    neighbor_model.fit(points)
    _, nearest = neighbor_model.kneighbors(points[sample_nodes])
    nearest = nearest[:, 1:]
    graph_hit_count = 0
    same_community_count = 0
    available_graph_neighbours = int(np.minimum(degree[sample_nodes], args.knn).sum())
    with args.viewer_nodes.open("r", encoding="utf-8") as handle:
        viewer = json.load(handle)
    reference_xy = np.asarray([[row["x"], row["y"]] for row in viewer], dtype=np.float64)[indices]
    communities = np.asarray([int(row.get("c", -1)) for row in viewer], dtype=np.int32)[indices]
    community_ids, community_counts = np.unique(communities, return_counts=True)
    major_community_ids = community_ids[np.argsort(community_counts)[::-1][:21]]
    major_centres = np.asarray([
        np.median(points[communities == community_id], axis=0)
        for community_id in major_community_ids
    ])
    if len(major_centres) > 1:
        major_centre_eigenvalues = np.linalg.eigvalsh(np.cov(major_centres, rowvar=False))[::-1]
        major_centre_eigenvalue_ratios = major_centre_eigenvalues / max(major_centre_eigenvalues[0], 1e-15)
    else:
        major_centre_eigenvalues = np.zeros(3)
        major_centre_eigenvalue_ratios = np.zeros(3)
    for row, vertex in enumerate(sample_nodes):
        graph_neighbors = set(adjacency.indices[adjacency.indptr[vertex]:adjacency.indptr[vertex + 1]])
        graph_hit_count += sum(int(candidate in graph_neighbors) for candidate in nearest[row])
        same_community_count += int(np.count_nonzero(communities[nearest[row]] == communities[vertex]))
    total_knn = len(sample_nodes) * args.knn
    graph_density = float(adjacency.nnz / (len(indices) * (len(indices) - 1)))
    knn_edge_fraction = float(graph_hit_count / total_knn)
    knn_enrichment = float(knn_edge_fraction / max(graph_density, np.finfo(float).eps))

    covariance = np.cov(points, rowvar=False)
    eigenvalues_ascending, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues_ascending)[::-1]
    eigenvalues = eigenvalues_ascending[order]
    centred = points - np.median(points, axis=0)
    radius = np.linalg.norm(centred, axis=1)
    principal_points = centred @ eigenvectors[:, order]
    robust_axis_extent = np.quantile(np.abs(principal_points), 0.90, axis=0)
    robust_axis_extent_ratios = robust_axis_extent / robust_axis_extent[0]
    principal_radius = np.linalg.norm(principal_points, axis=1)
    trimmed_points = principal_points[
        principal_radius <= np.quantile(principal_radius, 0.95)
    ]
    trimmed_eigenvalues = np.linalg.eigvalsh(
        np.cov(trimmed_points, rowvar=False)
    )[::-1]
    trimmed_eigenvalue_ratios = trimmed_eigenvalues / trimmed_eigenvalues[0]
    weight_distance_correlation = spearmanr(np.log1p(edge_w), edge_distance).statistic

    reference_pairs = rng.integers(0, len(indices), size=(args.pair_sample, 2))
    reference_pairs = reference_pairs[reference_pairs[:, 0] != reference_pairs[:, 1]]
    reference_distance = np.linalg.norm(
        reference_xy[reference_pairs[:, 0]] - reference_xy[reference_pairs[:, 1]], axis=1
    )
    candidate_distance = np.linalg.norm(
        points[reference_pairs[:, 0]] - points[reference_pairs[:, 1]], axis=1
    )
    reference_distance_correlation = spearmanr(reference_distance, candidate_distance).statistic
    reference_alignment = procrustes_similarity(reference_xy, points)

    metrics: dict[str, object] = {
        "rendered_nodes": int(len(indices)),
        "rendered_directed_edges": int(edge_mask.sum()),
        "geometry": {
            "axis_spans": np.ptp(points, axis=0).tolist(),
            "axis_standard_deviations": points.std(axis=0).tolist(),
            "covariance_eigenvalues": eigenvalues.tolist(),
            "covariance_eigenvalue_ratios": (eigenvalues / eigenvalues[0]).tolist(),
            "robust_pca_axis_p90_absolute_extent": robust_axis_extent.tolist(),
            "robust_pca_axis_p90_extent_ratios": robust_axis_extent_ratios.tolist(),
            "central_95_percent_covariance_eigenvalues": trimmed_eigenvalues.tolist(),
            "central_95_percent_covariance_eigenvalue_ratios": (
                trimmed_eigenvalue_ratios.tolist()
            ),
            "major_community_centroid_covariance_eigenvalues": (
                major_centre_eigenvalues.tolist()
            ),
            "major_community_centroid_covariance_eigenvalue_ratios": (
                major_centre_eigenvalue_ratios.tolist()
            ),
            "radius": percentile(radius, [0.5, 0.9, 0.99, 0.995, 0.999, 1.0]),
        },
        "graph_separation": {
            "sample_pairs": pair_count,
            "edge_distance": percentile(edge_distance, [0.25, 0.5, 0.75, 0.9]),
            "degree_matched_nonedge_distance": percentile(nonedge_distance, [0.25, 0.5, 0.75, 0.9]),
            "median_edge_to_nonedge_ratio": float(np.median(edge_distance) / np.median(nonedge_distance)),
            "paired_edge_shorter_fraction": float(np.mean(edge_distance < nonedge_distance)),
            "log_weight_distance_spearman": float(weight_distance_correlation),
        },
        "local_neighbourhood": {
            "sample_nodes": int(len(sample_nodes)),
            "k": args.knn,
            "graph_edge_fraction_among_layout_knn": knn_edge_fraction,
            "overall_graph_edge_density": graph_density,
            "layout_knn_edge_enrichment_over_graph_density": knn_enrichment,
            "recall_of_available_graph_neighbours": float(
                graph_hit_count / max(1, available_graph_neighbours)
            ),
            "same_community_fraction_among_layout_knn": float(same_community_count / total_knn),
        },
        "reference_2d_continuity": {
            "sample_pairs": int(len(reference_pairs)),
            "pair_distance_spearman": float(reference_distance_correlation),
            "rigid_linear_procrustes_similarity": float(reference_alignment),
            "interpretation": "comparison with viewer x/y on the same rendered authors",
        },
        "coordinates_sha256": sha256(args.coordinates),
        "indices_sha256": sha256(args.indices),
        "viewer_nodes_sha256": sha256(args.viewer_nodes),
    }
    # Undefined correlations (e.g. constant weights) are represented as null.
    def finite(value):
        if isinstance(value, dict): return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list): return [finite(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    metrics = finite(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
