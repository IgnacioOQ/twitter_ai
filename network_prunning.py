"""
Network Pruning Module
======================
Functions for pruning igraph networks and converting between igraph and NetworkX.

Functions:
    - ig_to_nx_fast: Convert an igraph Graph to a NetworkX graph.
    - prune_network_by_deletion: Prune a network by deleting low-scoring nodes
      while respecting a weight budget.
    - prune_by_out_strength_threshold: Prune a network by keeping only nodes
      whose out-strength meets a threshold.
"""

from .imports import *
import igraph as ig


# --------------------------------------------------------------------------- #
#  igraph → NetworkX conversion
# --------------------------------------------------------------------------- #

def ig_to_nx_fast(g):
    """
    Convert an igraph Graph to a NetworkX DiGraph (or Graph), preserving all
    vertex and edge attributes.

    Parameters
    ----------
    g : igraph.Graph
        The igraph graph to convert.

    Returns
    -------
    networkx.DiGraph or networkx.Graph
        The equivalent NetworkX graph.
    """
    # Choose directed or undirected
    if g.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # --- Vertices ----------------------------------------------------------
    vertex_attr_names = g.vs.attributes()
    for v in g.vs:
        # Use 'name' if present, otherwise use the igraph index
        node_id = v["name"] if "name" in vertex_attr_names else v.index
        attrs = {attr: v[attr] for attr in vertex_attr_names}
        G.add_node(node_id, **attrs)

    # Build a fast lookup: igraph index → node id used in NetworkX
    _node_id = []
    for v in g.vs:
        if "name" in vertex_attr_names:
            _node_id.append(v["name"])
        else:
            _node_id.append(v.index)

    # --- Edges -------------------------------------------------------------
    edge_attr_names = g.es.attributes()
    for e in g.es:
        src = _node_id[e.source]
        tgt = _node_id[e.target]
        attrs = {attr: e[attr] for attr in edge_attr_names}
        G.add_edge(src, tgt, **attrs)

    return G


# --------------------------------------------------------------------------- #
#  Helper: graph summary string
# --------------------------------------------------------------------------- #

def _graph_summary(g, label="Graph"):
    """Return a short one-line summary of an igraph graph."""
    return f"Vertices: {g.vcount():,}  Edges: {g.ecount():,}"


def _count_self_loops(g):
    """Count the number of self-loops in an igraph graph."""
    return sum(1 for e in g.es if e.source == e.target)


def _total_weight(g, attr="weight"):
    """Sum of all edge weights."""
    if attr in g.es.attributes():
        return sum(g.es[attr])
    else:
        return float(g.ecount())


# --------------------------------------------------------------------------- #
#  Pruning by node deletion (weight budget)
# --------------------------------------------------------------------------- #

def prune_network_by_deletion(g, method='total_strength', percentage_to_keep=0.90):
    """
    Prune a network by deleting the lowest-scoring nodes while respecting
    a weight budget, then extract the largest weakly connected component.

    Parameters
    ----------
    g : igraph.Graph
        The input graph (will NOT be modified in-place).
    method : str
        Scoring method for nodes.  Currently supported:
        - 'total_strength': sum of in-strength and out-strength.
    percentage_to_keep : float
        Fraction of total edge weight that must be retained (0–1).

    Returns
    -------
    igraph.Graph
        The pruned graph (LWCC only).
    """
    print()
    print("=" * 40)
    print(f"PRUNING OPERATION: Method='{method}', Keep={int(percentage_to_keep * 100)}%")
    print("=" * 40)

    # --- Full graph stats ---
    print(f"## Full graph (before) ##")
    print(_graph_summary(g))

    n_loops = _count_self_loops(g)
    print(f"Found {n_loops} self-loops.")

    # --- Remove self-loops ---
    print()
    print("Removing self-loops...")
    g_clean = g.copy()
    g_clean.simplify(loops=True, multiple=False)
    print()
    print(f"## Graph (after removing loops) ##")
    print(_graph_summary(g_clean))

    # --- Ensure 'name' attribute exists ---
    if "name" not in g_clean.vs.attributes():
        print("Adding 'name' attribute to vertices for safe deletion identification...")
        g_clean.vs["name"] = list(range(g_clean.vcount()))

    # --- Compute weight budget ---
    total_w = _total_weight(g_clean)
    goal_weight = total_w * percentage_to_keep
    max_loss = total_w - goal_weight
    print(f"🎯 Goal: Keep {goal_weight:,.2f} weight")
    print(f"   (Max weight to lose: {max_loss:,.2f})")

    # --- Score nodes ---
    print("1. Calculating node scores and sorting (lowest to highest)...")

    if method == 'total_strength':
        # total strength = in-strength + out-strength
        weight_attr = "weight" if "weight" in g_clean.es.attributes() else None
        in_str = g_clean.strength(mode="in", weights=weight_attr)
        out_str = g_clean.strength(mode="out", weights=weight_attr)
        scores = [i + o for i, o in zip(in_str, out_str)]
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    # Pair (score, vertex index) and sort ascending
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])

    # --- Identify nodes to remove (static pass) ---
    print("2. Calculating nodes to remove (static pass)...")
    weight_lost = 0.0
    nodes_to_remove = []
    weight_attr = "weight" if "weight" in g_clean.es.attributes() else None

    for rank, (vidx, score) in enumerate(indexed_scores):
        # Weight that would be lost if we remove this node =
        # sum of weights of all incident edges
        incident_edges = g_clean.incident(vidx, mode="all")
        node_weight = 0.0
        for eidx in incident_edges:
            if weight_attr:
                node_weight += g_clean.es[eidx][weight_attr]
            else:
                node_weight += 1.0

        if weight_lost + node_weight > max_loss:
            pct = weight_lost / total_w * 100
            print(f"   Nodes to remove: {len(nodes_to_remove):,} | "
                  f"Weight lost: {weight_lost:>12,.2f} ({pct:.1f}%)")
            print(f"   -> Stopping. Removing next node (rank {rank}) "
                  f"would exceed loss budget.")
            break

        weight_lost += node_weight
        nodes_to_remove.append(vidx)

    print(f"\n   Calculation complete. Identified {len(nodes_to_remove):,} nodes for removal.")

    # --- Delete nodes at once ---
    print("3. Creating final graph by deleting all nodes at once...")
    g_pruned = g_clean.copy()
    g_pruned.delete_vertices(nodes_to_remove)

    # --- Extract LWCC ---
    print("4. Extracting the largest weakly connected component...")
    components = g_pruned.components(mode="weak")
    g_lwcc = components.giant()

    # --- Summary ---
    original_v = g_clean.vcount()
    original_e = g_clean.ecount()
    lwcc_v = g_lwcc.vcount()
    lwcc_e = g_lwcc.ecount()
    lwcc_w = _total_weight(g_lwcc)
    pct_weight = lwcc_w / total_w * 100

    print()
    print(f"## Final Graph Summary (FAST) ##")
    print(f"**Method**: {method}")
    print(f"**LWCC Vertices**: {lwcc_v:,}")
    print(f"**LWCC Edges**: {lwcc_e:,}")
    print(f"**LWCC Weight**: {pct_weight:.2f}% of original")

    return g_lwcc


# --------------------------------------------------------------------------- #
#  Pruning by out-strength threshold
# --------------------------------------------------------------------------- #

def prune_by_out_strength_threshold(g, threshold=1.0):
    """
    Prune a network by keeping only nodes whose out-strength is at least
    `threshold`, then extract the largest weakly connected component.

    Parameters
    ----------
    g : igraph.Graph
        The input graph (will NOT be modified in-place).
    threshold : float
        Minimum out-strength for a node to be kept.

    Returns
    -------
    igraph.Graph
        The pruned graph (LWCC only).
    """
    method_label = f"out_strength_threshold >= {threshold}"
    print(f"\n--- Pruning with method: '{method_label}' ---")

    # --- Full graph stats ---
    print(f"## Full graph (before) ##")
    print(_graph_summary(g))

    n_loops = _count_self_loops(g)
    print(f"Found {n_loops} self-loops.")
    print(f"{g.summary()}")

    # --- Remove self-loops ---
    print()
    print("Removing self-loops...")
    g_clean = g.copy()
    g_clean.simplify(loops=True, multiple=False)
    print()
    print(f"## Graph (after removing loops) ##")
    print(_graph_summary(g_clean))
    n_loops_after = _count_self_loops(g_clean)
    print(f"Found {n_loops_after} self-loops after simplification.")

    # --- Identify nodes to keep ---
    weight_attr = "weight" if "weight" in g_clean.es.attributes() else None
    out_str = g_clean.strength(mode="out", weights=weight_attr)

    nodes_to_keep = [i for i, s in enumerate(out_str) if s >= threshold]
    print(f"1. Identifying nodes with out-strength >= {threshold}...")
    print(f"   Found {len(nodes_to_keep):,} nodes to keep "
          f"(out of {g_clean.vcount():,}).")

    # --- Create pruned graph ---
    print("2. Creating the pruned graph...")
    g_pruned = g_clean.subgraph(nodes_to_keep)

    # --- Extract LWCC ---
    print("3. Extracting the largest weakly connected component...")
    components = g_pruned.components(mode="weak")
    g_lwcc = components.giant()
    print("   Done.")

    # --- Final summary ---
    original_v = g_clean.vcount()
    original_e = g_clean.ecount()
    lwcc_v = g_lwcc.vcount()
    lwcc_e = g_lwcc.ecount()
    total_w = _total_weight(g_clean)
    lwcc_w = _total_weight(g_lwcc)
    pct_v = (original_v - lwcc_v) / original_v * 100
    pct_e = (original_e - lwcc_e) / original_e * 100
    pct_w = lwcc_w / total_w * 100

    print()
    print(f"## Final Graph Comparison ##")
    print(f"**Method**: Out-Strength Threshold (>= {threshold})")
    print("-" * 57)
    print(f"| {'Metric':<10} | {'Original':>12} | {'Final (LWCC)':>12} | {'% Reduction':>15} |")
    print("-" * 57)
    print(f"| {'Nodes':<10} | {original_v:>12,} | {lwcc_v:>12,} | {pct_v:>14.2f}% |")
    print(f"| {'Edges':<10} | {original_e:>12,} | {lwcc_e:>12,} | {pct_e:>14.2f}% |")
    print("-" * 57)
    print(f"Percentage of Original Weight Kept: {pct_w:.2f}%")

    return g_lwcc
