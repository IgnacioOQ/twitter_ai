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
        * ``'leiden_directed'`` – leidenalg RBConfiguration on the directed
          graph (no undirected collapse), seeded for reproducibility
        * ``'infomap'``      – Infomap map-equation clustering on the directed
          graph (no undirected collapse), seeded for reproducibility
    resolution : float
        Resolution parameter (used by Leiden and Louvain methods; ignored by
        label propagation and Infomap).

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

    # Modularity community detection is defined for undirected graphs, and
    # igraph's Louvain (community_multilevel) rejects directed input outright.
    # Symmetrize: collapse reciprocal edges into one, summing their weights.
    # leidenalg and Infomap handle directed input natively — keep it directed.
    if g.is_directed() and method not in ("leiden_directed", "infomap"):
        combine = {"weight": "sum"} if "weight" in g.es.attributes() else None
        g = g.as_undirected(mode="collapse", combine_edges=combine)
        print(f"   Converted to undirected (collapse, sum weights): "
              f"{g.vcount():,} V  {g.ecount():,} E")

    # ── 2. Community detection ────────────────────────────────────────────
    print(f"\n2. Running community detection ({method}) ...")

    weight_attr = "weight" if "weight" in g.es.attributes() else None

    if method == "leiden_full":
        # Full Leiden with CPM (Constant Potts Model)
        partition = g.community_leiden(
            objective_function="CPM",
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
    elif method == "leiden_directed":
        import leidenalg as la
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=weight_attr,
            resolution_parameter=resolution,
            seed=42,
        )
    elif method == "infomap":
        from infomap import Infomap
        flags = "--two-level --seed 42 --silent"
        if g.is_directed():
            flags += " --directed"
        im = Infomap(flags)
        for e in g.es:
            im.add_link(e.source, e.target, e[weight_attr] if weight_attr else 1.0)
        im.run()
        # Module ids are 1-based; inputs are LWCCs, so every vertex has links
        # and appears in get_modules().
        modules = im.get_modules()
        membership = [modules[v] - 1 for v in range(g.vcount())]
        partition = ig.VertexClustering(g, membership)
        print(f"   Infomap codelength: {im.codelength:.4f} bits")
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: 'leiden_full', 'leiden_fast', 'louvain', "
            f"'label_propagation', 'leiden_directed', 'infomap'."
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
    # Explicit call so weights are always used; on a directed graph igraph
    # computes directed modularity (partition.modularity would ignore weights
    # for the leiden_directed / infomap partitions).
    modularity = g.modularity(partition.membership, weights=weight_attr)
    sizes = partition.sizes()
    sizes_sorted = sorted(sizes, reverse=True)

    print(f"\n5. Summary")
    print(f"   Number of communities : {n_communities:,}")
    print(f"   Modularity            : {modularity:.4f}")
    print(f"   Largest 5 communities : {sizes_sorted[:5]}")
    print(f"   Smallest 5 communities: {sizes_sorted[-5:]}")
    print(f"{'=' * 60}\n")

    return g
