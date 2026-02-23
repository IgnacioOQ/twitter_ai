from .imports import *


# # Plotting Functions
# Plotting functions
def plot_network_degree_distribution(G, directed=True):
    # Compute density
    density = nx.density(G)
    print(f"Density of the network: {density}")
    if directed:
        degrees = np.array([degree for node, degree in G.out_degree()])
    else:
        degrees = np.array([degree for node, degree in G.degree()])
    # Create the histogram with a KDE
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(degrees, kde=False, bins=150, stat="count")
    # Calculate the mean
    mean_value = np.mean(degrees)
    print(mean_value)
    print(np.median(degrees))

    # Plot a vertical line at the mean value
    plt.axvline(mean_value, color="b", linestyle="--", linewidth=2)
    plt.text(
        mean_value + 0.1, plt.ylim()[1] * 0.9, f"Mean: {mean_value:.3f}", color="b"
    )
    # plt.text(mean_value + 0.1, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_value), color='b')

    plt.title("Network Out-Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.xticks(fontsize=8, rotation=20)
    plt.show()


def plot_loglog(G, directed=True, m=10):
    if directed:
        # Get the in-degree of all nodes
        out_degrees = [d for _, d in G.out_degree()]

        # Compute the histogram
        max_degree = max(out_degrees)
        degree_freq = [out_degrees.count(i) for i in range(max_degree + 1)]
    else:
        degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(8, 6))
    plt.loglog(degrees[m:], degree_freq[m:], "go-")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.xticks(fontsize=8, rotation=20)
    plt.title("Network Out-Degree Distribution Log-Log Plot")


def scatter_plot(df, target_variable="share_of_correct_agents_at_convergence"):
    # Select numerical columns excluding unique ID and target variable
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numerical_columns.remove(
        target_variable
    )  # Remove target variable from independent variables

    # Generate scatter plots for each numerical column against the target variable
    num_plots = len(numerical_columns)
    fig, axes = plt.subplots(
        nrows=(num_plots + 1) // 2, ncols=2, figsize=(10, num_plots * 2)
    )
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        axes[i].scatter(df[column], df[target_variable], alpha=0.5)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target_variable)
        axes[i].set_title(f"{column} vs {target_variable}")
        axes[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# # Network Statistics


# Network statistics
def calculate_degree_gini(G, directed=True):
    if directed:
        degrees = [deg for _, deg in G.out_degree()]
    else:
        degrees = [deg for _, deg in G.degree()]
    # Sort the degrees in ascending order
    sorted_x = np.sort(np.array(degrees))
    n = len(np.array(degrees))
    cumx = np.cumsum(sorted_x, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    return gini


def find_reachability_dominator_set(G):
    """
    Finds a minimal reachability dominator set in a directed graph G.

    Parameters:
        G (nx.DiGraph): A directed graph.

    Returns:
        set: A set of nodes A such that every node in G is reachable from some node in A.
    """
    # Step 1: Compute strongly connected components
    sccs = list(nx.strongly_connected_components(G))

    # Step 2: Build the condensation graph
    C = nx.condensation(G, sccs)

    # Step 3: Find source SCCs (no incoming edges)
    source_sccs = [node for node in C.nodes if C.in_degree(node) == 0]

    # Step 4: Pick one representative node from each source SCC
    reachability_dominator_set = set()
    scc_list = C.graph["mapping"]  # maps node -> scc index
    inverse_scc_map = {}
    for node, scc_id in scc_list.items():
        inverse_scc_map.setdefault(scc_id, []).append(node)

    for source_scc in source_sccs:
        representative = inverse_scc_map[source_scc][0]  # pick one node from this SCC
        reachability_dominator_set.add(representative)

    return (
        len(reachability_dominator_set),
        len(reachability_dominator_set) / len(G),
        len(C),
        len(C) / len(G),
    )


def compute_left_eigenvector(G):
    """
    Computes the Left Eigenvector centrality (DeGroot Influence).

    In the context of opinion dynamics or influence networks, this metric
    identifies the 'ultimate' sources of beliefs. It answers the question:
    "In the long run, how much does this agent's initial state determine
    the group's final consensus?"

    Mathematical Definition:
    ------------------------
    1. Constructs a Row-Stochastic Matrix W where W_ij represents the
       weight agent i places on agent j (based on incoming edges in G).
    2. If a node has no incoming edges (a Source), it is treated as
       'stubborn' or 'independent' (weight 1.0 on itself).
    3. Solves pi * W = pi (normalized so sum(pi) = 1).

    Parameters:
    -----------
    G : nx.DiGraph
        A directed graph where an edge (u, v) means u influences v.
        (v listens to u).

    Returns:
    --------
    dict
        Dictionary mapping node IDs to their influence score (probability mass).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize adjacency matrix for the "Listening Graph"
    # If G has edge u->v (u influences v), then v listens to u.
    W = np.zeros((n, n))

    for u in nodes:
        u_idx = node_to_idx[u]
        # precursors(u) are nodes that point TO u in G.
        # In an influence graph, these are the agents u listens to.
        influencers = list(G.predecessors(u))

        if len(influencers) == 0:
            # Case: Independent Agent (Source).
            # In DeGroot dynamics, they listen only to themselves.
            W[u_idx, u_idx] = 1.0
        else:
            # Case: Social Agent.
            # Assuming equal weights for simplicity.
            # (Can be modified to weight by reliability if data exists).
            weight = 1.0 / len(influencers)
            for inf in influencers:
                v_idx = node_to_idx[inf]
                W[u_idx, v_idx] = weight

    # Calculate Left Eigenvector for Eigenvalue 1.
    # Corresponds to the Right Eigenvector of the Transpose matrix.
    # W.T * v = 1 * v
    eigenvalues, eigenvectors = np.linalg.eig(W.T)

    # Extract eigenvector corresponding to eigenvalue 1 (or closest to it)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    left_ev = np.real(eigenvectors[:, idx])

    # Normalize to form a probability distribution (sum = 1)
    # Use absolute values to handle potential negative signs from solver (rare in stochastic matrices)
    left_ev = np.abs(left_ev)
    left_ev = left_ev / np.sum(left_ev)

    return {nodes[i]: left_ev[i] for i in range(n)}


def compute_katz_centrality(G, alpha=0.1, beta=1.0, measure_influence=True):
    """
    Computes Katz Centrality, optionally on the reversed graph to measure
    outgoing influence rather than incoming popularity.

    Parameters:
    -----------
    G : nx.DiGraph
        The influence network.
    alpha : float
        Attenuation factor. Smaller alpha means influence decays quickly
        over distance.
    beta : float
        Intrinsic weight. The baseline 'value' of an agent's own experiment.
    measure_influence : bool, default=True
        If True, reverses the graph before calculation.
        - True: Measures 'Influence' (how many people I reach). High for Sources.
        - False: Measures 'Prestige' (how many people reach me). High for Sinks.

    Returns:
    --------
    dict
        Dictionary of centrality scores.
    """
    if measure_influence:
        # We reverse the graph to track how the agent's 'beta' flows OUT to others.
        target_G = G.reverse()
    else:
        target_G = G

    try:
        return nx.katz_centrality(target_G, alpha=alpha, beta=beta, normalized=True)
    except nx.PowerIterationFailedConvergence:
        # Fallback for large/complex graphs: use numpy solver approach
        return nx.katz_centrality_numpy(
            target_G, alpha=alpha, beta=beta, normalized=True
        )


def network_statistics(G, directed=True, compute_clustering=False, fast_mode=True):
    stats = {}

    # Average degree
    if directed:
        degrees = [deg for _, deg in G.out_degree()]
    else:
        degrees = [deg for _, deg in G.degree()]

    if len(degrees) > 0:
        stats["average_degree"] = sum(degrees) / len(degrees)
    else:
        stats["average_degree"] = 0

    # Gini coefficient
    # print(degrees)
    if not fast_mode or len(degrees) < 500000:
        try:
            stats["degree_gini_coefficient"] = calculate_degree_gini(
                G, directed=directed
            )
        except Exception:
            pass

    # Compute clustering for each node
    # it allows us to use weights, which we neglect...
    if compute_clustering and (not fast_mode or len(G.nodes) < 100000):
        clustering_values = nx.clustering(G)
        # Compute the average clustering coefficient manually
        if len(clustering_values) > 0:
            average_clustering = sum(clustering_values.values()) / len(
                clustering_values
            )
            stats["approx_average_clustering_coefficient"] = average_clustering

    # commenting out unnecesary metrics to speed up computation
    # if directed:
    #     if nx.is_strongly_connected(G):
    #         stats['avg_path_length'] = nx.average_shortest_path_length(G)
    #     else:
    #         stats['avg_path_length'] = len(G.nodes)+1
    #         # largest_component = max(nx.weakly_connected_components(G), key=len)
    #         # subgraph = G.subgraph(largest_component)
    #         # stats['diameter'] = nx.diameter(subgraph)
    # else:
    #     if nx.is_connected(G):
    #         stats['avg_path_length'] = nx.average_shortest_path_length(G)
    #     else:
    #         stats['avg_path_length'] = len(G.nodes)+1
    #         # largest_component = max(nx.connected_components(G), key=len)
    #         # subgraph = G.subgraph(largest_component)
    #         # stats['diameter'] = nx.diameter(subgraph)

    # if directed:
    #     out_degrees = np.array([d for _, d in G.out_degree()])
    #     # out_degrees = np.array([d for _, d in graph.out_degree()])
    #     in_hist, _ = np.histogram(out_degrees, bins=range(np.max(out_degrees) + 2), density=True)
    #     # out_hist, _ = np.histogram(out_degrees, bins=range(np.max(out_degrees) + 2), density=True)
    #     out_entropy = -np.sum(in_hist[in_hist > 0] * np.log(in_hist[in_hist > 0]))
    #     # out_entropy = -np.sum(out_hist[out_hist > 0] * np.log(out_hist[out_hist > 0]))
    #     stats['degree_entropy'] = out_entropy
    # else:
    #     degrees = np.array([d for _, d in G.degree()])
    #     hist, _ = np.histogram(degrees, bins=range(np.max(degrees) + 2), density=True)
    #     entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
    #     stats['degree_entropy'] = entropy

    # # Add additional metrics as needed here, e.g., centrality measures
    # stats['reachability_dominator_set_size'] = find_reachability_dominator_set(G)[0]
    # stats['reachability_dominator_set_ratio'] = find_reachability_dominator_set(G)[1]
    # stats['condensation_graph_size'] = find_reachability_dominator_set(G)[2]
    # stats['condensation_graph_ratio'] = find_reachability_dominator_set(G)[3]
    return stats


# # Variation Methods
# ## Helper Functions


def get_triangles(net: nx.DiGraph):
    """Return the list of all triangles in a directed graph G."""
    triangles = []
    for clique in nx.enumerate_all_cliques(net.to_undirected()):
        if len(clique) <= 3:
            if len(clique) == 3:
                triangles.append(clique)
        else:
            return triangles
    return triangles


# ## Randomization


def randomize_network(G, n_edges: int):
    is_directed = G.is_directed()

    nodes = list(G.nodes())

    # Canonicalize existing edges if undirected
    raw_edges = list(G.edges())
    edges = raw_edges if is_directed else [tuple(sorted(e)) for e in raw_edges]

    random.shuffle(edges)
    new_edges_set = set(edges)

    # Choose edges to remove (already canonicalized if undirected)
    to_remove_set = set(random.sample(edges, k=n_edges))
    new_edges_set.difference_update(to_remove_set)  # <- fixes issue #1 and #2

    # Generate replacement edges (simple rejection is fine for sparse graphs)
    for _ in to_remove_set:
        u, v = random.choice(nodes), random.choice(nodes)
        if not is_directed:
            u, v = sorted((u, v))
        while (u == v) or ((u, v) in new_edges_set):
            u, v = random.choice(nodes), random.choice(nodes)
            if not is_directed:
                u, v = sorted((u, v))
        new_edges_set.add((u, v))

    # Rebuild the edge set on a copy
    G_new = copy.deepcopy(G)
    G_new.clear_edges()
    G_new.add_edges_from(new_edges_set)
    return G_new


# def randomize_network(G, n_edges: int):
#     # Check if the graph is directed
#     is_directed = G.is_directed()

#     # Get edges and nodes
#     edges = copy.deepcopy(list(G.edges()))
#     random.shuffle(edges)
#     edges_set = set(edges)
#     new_edges_set = copy.deepcopy(edges_set)
#     nodes = copy.deepcopy(list(G.nodes()))

#     # Find which edges to remove
#     to_remove_set = set(random.sample(edges, k=n_edges))
#     new_edges_set.difference_update(to_remove_set)

#     # Generate a new edges
#     for edge in to_remove_set:
#         new_edge = (random.choice(nodes), random.choice(nodes))
#         if not is_directed:
#             new_edge = tuple(sorted(new_edge))  # Ensure (u, v) == (v, u) for undirected graphs

#         # Avoid duplicate edges and self-loops
#         while (new_edge in new_edges_set) or (new_edge[0] == new_edge[1]):
#             new_edge = (random.choice(nodes), random.choice(nodes))
#             if not is_directed:
#                 new_edge = tuple(sorted(new_edge))

#         new_edges_set.add(new_edge)

#     # Create a new graph with updated edges
#     G_new = copy.deepcopy(G)
#     G_new.remove_edges_from(to_remove_set)
#     G_new.add_edges_from(new_edges_set)

#     return G_new


# ## Equalize
def equalize(net: nx.DiGraph, n: int) -> nx.DiGraph:
    """
    Equalize the network by rewiring n random edges.
    """
    equalized_net = copy.deepcopy(net)
    triangles = get_triangles(net)
    rewired_triangles = random.sample(triangles, n)

    for triangle in rewired_triangles:
        edge = triangle[-2:]  # Take the last two nodes as the edge to be rewired
        # Remove edge
        # I: What is the difference between the two conditions?
        if equalized_net.has_edge(*edge):
            equalized_net.remove_edge(*edge)
        elif equalized_net.has_edge(edge[1], edge[0]):
            equalized_net.remove_edge(edge[1], edge[0])
        else:
            continue

        # Add new edge to create a new triangle that passes by the first node
        node = triangle[0]
        neighbors = list(net.predecessors(node)) + list(net.successors(node))
        # I: I understand k=10 neighbors so that there are enough options to choose from,
        sources_sample = random.choices(neighbors, k=20)
        targets_sample = random.choices(neighbors, k=20)
        edge_sample = [
            (source, target)
            for source in sources_sample
            for target in targets_sample
            if source != target and not equalized_net.has_edge(source, target)
        ]
        new_edge = random.choice(
            edge_sample
        )  # Throws an error if no edges are available
        equalized_net.add_edge(*new_edge)
    return equalized_net


# # ## Densify
# def densify_fancy_speed_up(
#     net: nx.DiGraph, n_edges: int, target_degree_dist: str = "original",
#     target_average_clustering: float = None,
#     keep_density_fixed = False,
# ) -> nx.DiGraph:
#     """
#     Densifies a directed network by adding new edges to increase its density,
#     while optionally targeting a specific degree distribution and clustering coefficient.
#     Priority is given to targeting the specified clustering coefficient.

#     Parameters
#     ----------
#     net : nx.DiGraph
#         The original directed network to densify.
#     n_edges : int
#         The number of edges to add.
#     target_degree_dist : str, optional
#         The target degree distribution for new edges.
#         "original" preserves the original degree distribution,
#         "uniform" assigns equal probability to all nodes. Default is "original".
#     target_clustering : float, optional
#         The desired average clustering coefficient. If None, uses the original network's clustering.

#     Returns
#     -------
#     nx.DiGraph
#         A new directed network with increased density and optionally modified clustering/degree distribution.
#     """

#     # Create a copy of the original network
#     net_new = copy.deepcopy(net)

#     if target_average_clustering is None:
#         target_average_clustering = nx.average_clustering(net)
#     if target_degree_dist == "original":
#         # Use the original degree distribution
#         out_degrees = dict(net.out_degree())
#         in_degrees = dict(net.in_degree())
#     if target_degree_dist == "uniform":
#         out_degrees = {node: 1 for node in net.nodes()}
#         in_degrees = {node: 1 for node in net.nodes()}

#     if keep_density_fixed:
#         edges_to_remove = random.sample(net_new.edges(), n_edges)
#         net_new.remove_edges_from(edges_to_remove)

#     clustering_dict: dict = nx.clustering(net_new)

#     # Add edges in neighborhoods
#     n_edges_added = 0
#     edges_added_clustering = 0
#     edges_added_degree_dist = 0
#     new_average_clustering = np.average(list(clustering_dict.values()))
#     while n_edges_added < n_edges:
#         if new_average_clustering < target_average_clustering:
#             # Add new edge to increase clustering
#             node = random.choice(list(net.nodes()))
#             neighbors = list(net.predecessors(node)) + list(net.successors(node))
#             out_degrees_neighbors = {node: out_degrees[node] for node in neighbors}
#             in_degrees_neighbors = {node: in_degrees[node] for node in neighbors}
#             out_weights = out_degrees_neighbors.values()
#             if all(out_weights) == 0:
#                 out_weights = np.ones(len(out_degrees_neighbors.keys()))
#             in_weights = in_degrees_neighbors.values()

#             if all(in_weights) == 0:
#                 in_weights = np.ones(len(in_degrees_neighbors.keys()))

#             sources = random.choices(list(out_degrees_neighbors.keys()), weights=out_weights, k=10)
#             targets = random.choices(list(in_degrees_neighbors.keys()), weights=in_weights, k=10)
#             possible_edges = [
#                 (source, target) for source in sources for target in targets
#                 if source != target and not net_new.in_edges(source, target)
#             ]
#             if possible_edges != []:
#                 new_edge = random.choice(possible_edges)
#                 n_edges_added += 1
#                 net_new.add_edge(*new_edge)
#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected_nodes = [new_edge[0], new_edge[1]] + list(set(neighborhood_0).intersection(set(neighborhood_1)))
#                 for node in affected_nodes:
#                     clustering_dict[node] = nx.clustering(net_new, node)
#                 new_average_clustering = np.average(list(clustering_dict.values()))
#                 edges_added_clustering += 1
#         else:
#             # Add new edge based on target degree distribution
#             sources_sample = random.choices(list(out_degrees.keys()), weights=out_degrees.values(), k=10)
#             targets_sample = random.choices(list(in_degrees.keys()), weights=in_degrees.values(), k=10)
#             edge_sample = [
#                 (source, target)
#                 for source in sources_sample
#                 for target in targets_sample
#                 if source != target and not net_new.has_edge(source, target)]
#             if edge_sample != []:
#                 new_edge = random.choice(edge_sample) # Throws an error if no edges are available
#                 n_edges_added += 1
#                 net_new.add_edge(*new_edge)
#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected_nodes = [new_edge[0], new_edge[1]] + list(set(neighborhood_0).intersection(set(neighborhood_1)))
#                 for node in affected_nodes:
#                     clustering_dict[node] = nx.clustering(net_new, node)
#                 new_average_clustering = np.average(list(clustering_dict.values()))
#                 edges_added_degree_dist += 1
#         # print(f"{n_edges_added=:,} edges added")
#     # print(f"{edges_added_clustering:,} edges added to increase clustering")
#     # print(f"{edges_added_degree_dist:,} edges added based on {target_degree_dist} degree distribution")
#                     clustering_dict[n] = nx.clustering(net_new, n)
#                 new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)

#     return net_new


def plot_report(G):
    try:
        import powerlaw
    except ImportError:
        powerlaw = None
        print("powerlaw module not found. Distribution fitting will be skipped.")

    print("Report and Plots")
    print("Plotting Report")
    print("-------------------------")
    print("## Network Summary ##")
    print("-------------------------")

    is_directed = G.is_directed()
    v_count = G.number_of_nodes()

    print(f"Network Type: {'Directed' if is_directed else 'Undirected'}")
    print(f"Number of nodes (V): {v_count}")
    print(f"Number of edges (E): {G.number_of_edges()}")

    is_weighted = False
    for u, v, d in G.edges(data=True):
        if "weight" in d:
            is_weighted = True
        break
    print(f"Is Weighted: {is_weighted}")

    stats = network_statistics(
        G, directed=is_directed, compute_clustering=False, fast_mode=True
    )
    print(f"Average Degree: {stats.get('average_degree', 0):.4f}")
    print(f"Density: {nx.density(G):.6f}")
    if "degree_gini_coefficient" in stats:
        print(f"Gini Coefficient (Degree): {stats['degree_gini_coefficient']:.4f}")
    else:
        print(f"Gini Coefficient (Degree): Skipped (very large graph)")

    if v_count < 100000:
        if is_directed:
            print(
                f"Number of Weakly Connected Components: {nx.number_weakly_connected_components(G)}"
            )
        else:
            print(
                f"Number of Connected Components: {nx.number_connected_components(G)}"
            )
    else:
        print("Number of Components: Skipped (very large graph)")

    print(f"Number of self-loops: {nx.number_of_selfloops(G)}")
    print("-------------------------\n")

    print("## Detailed Node and Edge Example ##")
    print("-----------------------------------")

    if len(G.nodes) > 0:
        sample_node = next(iter(G.nodes()))
        print(f"🔍 Inspecting Node: '{sample_node}'\n")
        print(f"Attributes of Node '{sample_node}':")
        node_attr = G.nodes[sample_node]
        for k, v in node_attr.items():
            print(f"  - {k}: {v}")

        print(f"\nEdges connected to Node '{sample_node}':")
        edges = list(G.edges(sample_node, data=True))
        if len(edges) > 0:
            sample_edge = edges[0]
            print(f"Example Edge: ('{sample_edge[0]}', '{sample_edge[1]}')")
            print("Attributes of this edge:")
            for k, v in sample_edge[2].items():
                print(f"  - {k}: {v}")

    print("-----------------------------------")
    print("-------------------------")

    print("--- 0. Extracting data from graph (mode: 'out') ---\n")
    if is_directed:
        degrees = [
            deg for _, deg in G.out_degree(weight="weight" if is_weighted else None)
        ]
    else:
        degrees = [deg for _, deg in G.degree(weight="weight" if is_weighted else None)]

    degrees_arr = np.array(degrees)

    print("--- 1. Calculating Percentiles ---")
    if len(degrees_arr) > 0:
        for p in [90, 95, 99]:
            val = np.percentile(degrees_arr, p)
            coverage = 100 * np.sum(degrees_arr <= val) / len(degrees_arr)
            print(
                f"~{p}% of nodes have strength <= {val:.1f} (Actual coverage: {coverage:.2f}%)"
            )

    print("\n--- 2. Fitting Power Law (Method: 'tail') ---")
    if powerlaw is not None and len(degrees_arr[degrees_arr > 0]) > 0:
        try:
            # We suppress stdout to avoid excessive printing from powerlaw
            fit = powerlaw.Fit(degrees_arr[degrees_arr > 0], xmin=None, discrete=True)
            print("Fit results on the full distribution:")
            print(f"Alpha (α): {fit.power_law.alpha:.4f}")
            print(f"Xmin (xₘᵢₙ): {fit.power_law.xmin:.4f}")
            print(f"Sigma (σ): {fit.power_law.sigma:.4f}\n")

            print("--- 3. Distribution Comparison ---")
            R, p_val = fit.distribution_compare("power_law", "lognormal")
            print(
                f"Power Law vs. Lognormal: Loglikelihood Ratio R={R:.4f}, p-value={p_val:.4f}"
            )
            if p_val < 0.05:
                if R > 0:
                    print("Verdict: Power law is a significantly better fit.\n")
                else:
                    print("Verdict: Lognormal is a significantly better fit.\n")
            else:
                print(
                    "Verdict: Not statistically significant. Cannot conclude one is a better fit than the other.\n"
                )
        except Exception as e:
            print(f"Error fitting power law: {e}\n")

    print("--- 4. Generating Plot ---")
    plot_network_degree_distribution(G, directed=is_directed)
    plot_loglog(G, directed=is_directed)


# ## Cluster
def decluster(net: nx.DiGraph, n_triangles: int) -> nx.DiGraph:
    """
    Decluster the network by rewiring n_triangles random triangles.
    """
    decluster_net = copy.deepcopy(net)
    triangles = get_triangles(net)
    rewired_triangles = random.sample(triangles, n_triangles)
    rewired_edges = [
        (source, target) for (source, target, _) in rewired_triangles
    ]  # Warning: triangles are based on undirected graph!

    for edge in rewired_edges:
        # Remove edge
        if decluster_net.has_edge(*edge):
            decluster_net.remove_edge(*edge)
        elif decluster_net.has_edge(edge[1], edge[0]):
            decluster_net.remove_edge(edge[1], edge[0])
        else:
            continue
        # I: I like this but maybe the new edge generates a new cluster?
        # Add new edge based on out- and in-degree distribution
        out_degrees = dict(net.out_degree())
        in_degrees = dict(net.in_degree())
        sources_sample = random.choices(
            list(out_degrees.keys()), weights=out_degrees.values(), k=10
        )
        targets_sample = random.choices(
            list(in_degrees.keys()), weights=in_degrees.values(), k=10
        )
        edge_sample = [
            (source, target)
            for source in sources_sample
            for target in targets_sample
            if source != target and not decluster_net.has_edge(source, target)
        ]
        new_edge = random.choice(
            edge_sample
        )  # Throws an error if no edges are available
        decluster_net.add_edge(*new_edge)
    return decluster_net


def cluster_network(net: nx.DiGraph, n: int) -> nx.DiGraph:
    # Create a copy of the original network
    cluster_net = copy.deepcopy(net)

    # Add edges based on the degree distribution
    n_edges_to_add = n
    # print(f"{n_edges_to_add=:,}")

    # Add edges in neighborhoods
    edges_new = []
    # I: wouldn't it be better to add one edge per random chosen node?
    # I: this way we can ensure that the new edges are not making a single node too 'cliqued'
    while len(edges_new) < n_edges_to_add:
        node = random.choice(list(net.nodes()))
        neighbors = list(net.predecessors(node)) + list(net.successors(node))
        out_degrees_neighbors = dict(net.out_degree(neighbors))
        in_degrees_neighbors = dict(net.in_degree(neighbors))
        out_weights = out_degrees_neighbors.values()
        if all(out_weights) == 0:
            out_weights = np.ones(len(out_degrees_neighbors.keys()))
        in_weights = in_degrees_neighbors.values()

        if all(in_weights) == 0:
            in_weights = np.ones(len(in_degrees_neighbors.keys()))

        sources = random.choices(
            list(out_degrees_neighbors.keys()), weights=out_weights, k=10
        )
        targets = random.choices(
            list(in_degrees_neighbors.keys()), weights=in_weights, k=10
        )
        possible_edges = [
            (source, target)
            for source in sources
            for target in targets
            if source != target
            and not (source, target) in edges_new
            and not net.in_edges(source, target)
        ]
        if possible_edges != []:
            edges_new.append(random.choice(possible_edges))
    cluster_net.add_edges_from(edges_new)

    return cluster_net


def densify_fancy_speed_up(
    net: nx.DiGraph,
    n_edges: int,
    target_degree_dist: str = "original",
    target_average_clustering: float = None,
    keep_density_fixed=False,
) -> nx.DiGraph:
    """
    Densifies a directed network by adding new edges to increase its density,
    while optionally targeting a specific degree distribution and clustering coefficient.
    Priority is given to targeting the specified clustering coefficient.

    Parameters
    ----------
    net : nx.DiGraph
        The original directed network to densify.
    n_edges : int
        The number of edges to add.
    target_degree_dist : str, optional
        The target degree distribution for new edges.
        "original" preserves the original degree distribution,
        "uniform" assigns equal probability to all nodes. Default is "original".
    target_clustering : float, optional
        The desired average clustering coefficient. If None, uses the original network's clustering.

    Returns
    -------
    nx.DiGraph
        A new directed network with increased density and optionally modified clustering/degree distribution.
    """

    # Create a copy of the original network
    net_new = copy.deepcopy(net)

    if target_average_clustering is None:
        target_average_clustering = nx.average_clustering(net)
    if target_degree_dist == "original":
        out_degrees = dict(net.out_degree())
        in_degrees = dict(net.in_degree())
    elif target_degree_dist == "uniform":
        out_degrees = {node: 1 for node in net.nodes()}
        in_degrees = {node: 1 for node in net.nodes()}
    else:
        raise ValueError("target_degree_dist must be 'original' or 'uniform'")

    # if keep_density_fixed:
    #     edges_to_remove = random.sample(net_new.edges(), n_edges)
    #     net_new.remove_edges_from(edges_to_remove)

    if keep_density_fixed:
        # Ensure there are enough edges to remove and the number to remove is not negative
        num_edges_to_remove = min(n_edges, net_new.number_of_edges())
        if num_edges_to_remove > 0:
            edges_to_remove = random.sample(list(net_new.edges()), num_edges_to_remove)
            net_new.remove_edges_from(edges_to_remove)

    clustering_dict: dict = nx.clustering(net_new)

    # Add edges in neighborhoods
    n_edges_added = 0
    edges_added_clustering = 0
    edges_added_degree_dist = 0
    new_average_clustering = np.average(list(clustering_dict.values()))
    while n_edges_added < n_edges:
        if new_average_clustering < target_average_clustering:
            # Add new edge to increase clustering
            node = random.choice(list(net.nodes()))
            neighbors = list(net.predecessors(node)) + list(net.successors(node))
            out_degrees_neighbors = {node: out_degrees[node] for node in neighbors}
            in_degrees_neighbors = {node: in_degrees[node] for node in neighbors}
            out_weights = out_degrees_neighbors.values()
            if all(out_weights) == 0:
                out_weights = np.ones(len(out_degrees_neighbors.keys()))
            in_weights = in_degrees_neighbors.values()

            if all(in_weights) == 0:
                in_weights = np.ones(len(in_degrees_neighbors.keys()))

            sources = random.choices(
                list(out_degrees_neighbors.keys()), weights=out_weights, k=10
            )
            targets = random.choices(
                list(in_degrees_neighbors.keys()), weights=in_weights, k=10
            )
            possible_edges = [
                (source, target)
                for source in sources
                for target in targets
                if source != target and not net_new.has_edge(source, target)
            ]
            if possible_edges != []:
                new_edge = random.choice(possible_edges)
                n_edges_added += 1
                net_new.add_edge(*new_edge)
                neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(
                    net_new.successors(new_edge[0])
                )
                neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(
                    net_new.successors(new_edge[1])
                )
                affected_nodes = [new_edge[0], new_edge[1]] + list(
                    set(neighborhood_0).intersection(set(neighborhood_1))
                )
                for node in affected_nodes:
                    clustering_dict[node] = nx.clustering(net_new, node)
                new_average_clustering = np.average(list(clustering_dict.values()))
                edges_added_clustering += 1
        else:
            # Add new edge based on target degree distribution
            sources_sample = random.choices(
                list(out_degrees.keys()), weights=out_degrees.values(), k=10
            )
            targets_sample = random.choices(
                list(in_degrees.keys()), weights=in_degrees.values(), k=10
            )
            edge_sample = [
                (source, target)
                for source in sources_sample
                for target in targets_sample
                if source != target and not net_new.has_edge(source, target)
            ]
            if edge_sample != []:
                new_edge = random.choice(
                    edge_sample
                )  # Throws an error if no edges are available
                n_edges_added += 1
                net_new.add_edge(*new_edge)
                neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(
                    net_new.successors(new_edge[0])
                )
                neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(
                    net_new.successors(new_edge[1])
                )
                affected_nodes = [new_edge[0], new_edge[1]] + list(
                    set(neighborhood_0).intersection(set(neighborhood_1))
                )
                for node in affected_nodes:
                    clustering_dict[node] = nx.clustering(net_new, node)
                new_average_clustering = np.average(list(clustering_dict.values()))
                edges_added_degree_dist += 1
        # print(f"{n_edges_added=:,} edges added")
    # print(f"{edges_added_clustering:,} edges added to increase clustering")
    # print(f"{edges_added_degree_dist:,} edges added based on {target_degree_dist} degree distribution")
    return net_new


# def densify_fancy_speed_up_v2(
#     net,
#     n_edges,
#     target_degree_dist="original",
#     target_average_clustering=None,
#     keep_density_fixed=False
# ):
#     """
#     Densifies a directed network by adding (or replacing) edges to increase its density,
#     optionally targeting a specific degree distribution and clustering coefficient.
#     If keep_density_fixed is True, the total number of edges is preserved.

#     Parameters
#     ----------
#     net : nx.DiGraph
#         The original directed network to densify.
#     n_edges : int
#         The number of edges to add (or replace if keep_density_fixed is True).
#     target_degree_dist : str
#         "original" uses the original degree distribution; "uniform" gives equal weight to all nodes.
#     target_average_clustering : float or None
#         Target average clustering coefficient. If None, uses current average.
#     keep_density_fixed : bool
#         If True, removes one edge per new edge to keep edge count constant.

#     Returns
#     -------
#     nx.DiGraph
#         The modified graph.
#     """
#     net_new = copy.deepcopy(net)

#     if target_average_clustering is None:
#         target_average_clustering = nx.average_clustering(net)

#     if target_degree_dist == "original":
#         out_degrees = dict(net.out_degree())
#         in_degrees = dict(net.in_degree())
#     elif target_degree_dist == "uniform":
#         out_degrees = {node: 1 for node in net.nodes()}
#         in_degrees = {node: 1 for node in net.nodes()}
#     else:
#         raise ValueError("target_degree_dist must be 'original' or 'uniform'")

#     clustering_dict = nx.clustering(net_new)

#     n_edges_added = 0
#     new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)
#     max_attempts = n_edges * 10
#     attempts = 0

#     while n_edges_added < n_edges and attempts < max_attempts:
#         attempts += 1

#         if keep_density_fixed and len(net_new.edges) > 0:
#             old_edge = random.choice(list(net_new.edges()))
#             net_new.remove_edge(*old_edge)

#         if new_average_clustering < target_average_clustering:
#             node = random.choice(list(net.nodes()))
#             neighbors = list(net.predecessors(node)) + list(net.successors(node))
#             if not neighbors:
#                 continue

#             out_neighbors = {n: out_degrees.get(n, 1) for n in neighbors}
#             in_neighbors = {n: in_degrees.get(n, 1) for n in neighbors}

#             sources = random.choices(list(out_neighbors.keys()), weights=out_neighbors.values(), k=10)
#             targets = random.choices(list(in_neighbors.keys()), weights=in_neighbors.values(), k=10)

#             possible_edges = [
#                 (s, t) for s in sources for t in targets
#                 if s != t and not net_new.has_edge(s, t)
#             ]

#             if possible_edges:
#                 new_edge = random.choice(possible_edges)
#                 net_new.add_edge(*new_edge)
#                 n_edges_added += 1

#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected = set([new_edge[0], new_edge[1]]) | (set(neighborhood_0) & set(neighborhood_1))

#                 for n in affected:
#                     clustering_dict[n] = nx.clustering(net_new, n)
#                 new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)
#         else:
#             sources = random.choices(list(out_degrees.keys()), weights=out_degrees.values(), k=10)
#             targets = random.choices(list(in_degrees.keys()), weights=in_degrees.values(), k=10)

#             possible_edges = [
#                 (s, t) for s in sources for t in targets
#                 if s != t and not net_new.has_edge(s, t)
#             ]

#             if possible_edges:
#                 new_edge = random.choice(possible_edges)
#                 net_new.add_edge(*new_edge)
#                 n_edges_added += 1

#                 neighborhood_0 = list(net_new.predecessors(new_edge[0])) + list(net_new.successors(new_edge[0]))
#                 neighborhood_1 = list(net_new.predecessors(new_edge[1])) + list(net_new.successors(new_edge[1]))
#                 affected = set([new_edge[0], new_edge[1]]) | (set(neighborhood_0) & set(neighborhood_1))

#                 for n in affected:
#                     clustering_dict[n] = nx.clustering(net_new, n)
#                 new_average_clustering = sum(clustering_dict.values()) / len(clustering_dict)

#     return net_new
