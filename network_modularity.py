"""
Network Modularity Module
==========================
Functions for running community detection (modularity analysis) on igraph
networks.

Functions:
    - run_modularity_workflow: Load a graph, run community detection, assign
      community labels, and save the result.
"""

from .imports import *
import igraph as ig
from pathlib import Path


def run_modularity_workflow(
    networks_folder_path,
    input_filename,
    method="leiden_fast",
    resolution=1.0,
):
    """
    Run a complete modularity / community-detection workflow on an igraph graph
    stored as a GML file.

    Steps
    -----
    1. Load the GML file from *networks_folder_path / input_filename*.
    2. Run community detection using the specified *method* and *resolution*.
    3. Assign community IDs to vertices as attribute ``community_<method>``.
    4. Save the annotated graph as a new GML file.
    5. Print summary statistics (number of communities, modularity, sizes).

    Parameters
    ----------
    networks_folder_path : str or pathlib.Path
        Folder that contains the input GML file.
    input_filename : str
        Name of the GML file to load (e.g. ``"Final_OutThreshold1.gml"``).
    method : str
        Community-detection algorithm.  Supported values:

        * ``'leiden_full'``  – Leiden with modularity optimisation (CPM quality)
        * ``'leiden_fast'``  – Leiden with fast modularity (RBConfiguration)
        * ``'louvain'``      – Louvain algorithm
        * ``'label_propagation'`` – Label-propagation algorithm
    resolution : float
        Resolution parameter (used by Leiden and Louvain methods).

    Returns
    -------
    igraph.Graph
        The graph with a new vertex attribute containing community IDs.
    """
    base = Path(networks_folder_path)
    in_path = base / input_filename

    print(f"\n{'=' * 60}")
    print(f"MODULARITY WORKFLOW")
    print(f"  Input   : {in_path}")
    print(f"  Method  : {method}")
    print(f"  Resolution : {resolution}")
    print(f"{'=' * 60}")

    # ── 1. Load graph ─────────────────────────────────────────────────────
    print(f"\n1. Loading graph from: {in_path}")
    g = ig.Graph.Read_GML(str(in_path))
    print(f"   Vertices: {g.vcount():,}  Edges: {g.ecount():,}")

    # ── 2. Community detection ────────────────────────────────────────────
    print(f"\n2. Running community detection ({method}) ...")

    weight_attr = "weight" if "weight" in g.es.attributes() else None

    if method == "leiden_full":
        # Full Leiden with CPM (Constant Potts Model)
        partition = g.community_leiden(
            objective_function="CPMVertexPartition",
            weights=weight_attr,
            resolution=resolution,
            n_iterations=-1,
        )
    elif method == "leiden_fast":
        # Leiden with RBConfiguration (modularity-based)
        partition = g.community_leiden(
            objective_function="modularity",
            weights=weight_attr,
            resolution=resolution,
            n_iterations=2,
        )
    elif method == "louvain":
        partition = g.community_multilevel(weights=weight_attr)
    elif method == "label_propagation":
        partition = g.community_label_propagation(weights=weight_attr)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: 'leiden_full', 'leiden_fast', 'louvain', "
            f"'label_propagation'."
        )

    print("   Done.")

    # ── 3. Assign community IDs to vertices ───────────────────────────────
    attr_name = f"community_{method}"
    g.vs[attr_name] = partition.membership
    print(f"\n3. Assigned community IDs as vertex attribute '{attr_name}'.")

    # ── 4. Save ───────────────────────────────────────────────────────────
    stem = Path(input_filename).stem
    out_filename = f"{stem}_{method}.gml"
    out_path = base / out_filename

    try:
        g.write_gml(str(out_path))
    except AttributeError:
        g.save(str(out_path), format="gml")

    print(f"\n4. Saved annotated graph to: {out_path}")

    # ── 5. Summary ────────────────────────────────────────────────────────
    n_communities = len(partition)
    modularity = partition.modularity
    sizes = partition.sizes()
    sizes_sorted = sorted(sizes, reverse=True)

    print(f"\n5. Summary")
    print(f"   Number of communities : {n_communities:,}")
    print(f"   Modularity            : {modularity:.4f}")
    print(f"   Largest 5 communities : {sizes_sorted[:5]}")
    print(f"   Smallest 5 communities: {sizes_sorted[-5:]}")
    print(f"{'=' * 60}\n")

    return g
