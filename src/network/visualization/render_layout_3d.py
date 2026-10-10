#!/usr/bin/env python3
"""Render equal-axis, PCA-aligned views of a network layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


from .common import read_payload
COMMUNITY_COLORS = np.asarray([
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.68, 0.78, 0.91], [1.00, 0.73, 0.47],
    [0.60, 0.87, 0.54], [1.00, 0.60, 0.59], [0.77, 0.69, 0.84], [0.77, 0.61, 0.58],
    [0.97, 0.71, 0.82], [0.78, 0.78, 0.78], [0.86, 0.86, 0.55], [0.62, 0.85, 0.90],
    [0.22, 0.23, 0.47], [0.39, 0.47, 0.22], [0.55, 0.43, 0.19], [0.52, 0.24, 0.22],
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--viewer-nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="3D network layout")
    parser.add_argument(
        "--top-communities",
        type=int,
        default=0,
        help="diagnostically restrict to the N largest displayed communities",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer = json.loads(args.viewer_nodes.read_text(encoding='utf-8'))
    coordinates, indices = read_payload(args.coordinates, args.indices, len(viewer))
    points = coordinates[indices]
    communities = np.asarray([int(row.get("c", -1)) for row in viewer], dtype=np.int32)[indices]
    if args.top_communities:
        unique, counts = np.unique(communities, return_counts=True)
        keep_ids = unique[np.argsort(counts)[::-1][:args.top_communities]]
        keep = np.isin(communities, keep_ids)
        points = points[keep]
        communities = communities[keep]
    colors = np.full((len(points), 3), 0.44, dtype=np.float32)
    known = (communities >= 0) & (communities < len(COMMUNITY_COLORS))
    colors[known] = COMMUNITY_COLORS[communities[known]]

    centred = points - np.median(points, axis=0)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    points = centred @ eigenvectors[:, order]
    # Fix eigenvector signs so repeated renders are visually reproducible.
    for axis_index in range(3):
        pivot = int(np.argmax(np.abs(points[:, axis_index])))
        if points[pivot, axis_index] < 0:
            points[:, axis_index] *= -1

    robust_extent = float(np.quantile(np.abs(points), 0.9975))
    if not np.isfinite(robust_extent) or robust_extent <= 0:
        raise RuntimeError("Layout has invalid robust extent")
    limit = robust_extent * 1.08
    ratios = eigenvalues[order] / eigenvalues[order][0]

    views = [
        (90, -90, "principal 1 / principal 2"),
        (0, 0, "principal 2 / principal 3"),
        (0, -90, "principal 1 / principal 3"),
        (24, -58, "oblique A"),
        (18, 28, "oblique B"),
        (-22, 122, "oblique C"),
    ]
    figure = plt.figure(figsize=(15, 10), facecolor="#111119")
    for position, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(2, 3, position, projection="3d", facecolor="#111119")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.12, c=colors,
                     alpha=0.34, linewidths=0, depthshade=False, rasterized=True)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_proj_type("ortho")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_zlim(-limit, limit)
        axis.set_axis_off()
        axis.set_title(label, color="#d7d7df", fontsize=9, pad=1)
        axis.set_box_aspect((1, 1, 1))
    ratio_text = ", ".join(f"{value:.3f}" for value in ratios)
    figure.suptitle(
        f"{args.title}  ·  PCA variance ratios {ratio_text}  ·  equal axis scales",
        color="#f2f2f4", fontsize=12, y=0.97,
    )
    figure.tight_layout(pad=0.8, rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
