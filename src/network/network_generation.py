from .imports import *


def barabasi_albert_directed(n, m):
    """
    Implements the Barabási-Albert model for directed networks.

    Parameters:
        n (int): Total number of nodes in the network.
        m (int): Number of directed edges each new node creates. Must be <= total nodes at any time.

    Returns:
        G (networkx.DiGraph): A directed scale-free network.
    """
    # Ensure valid input
    if m < 1 or m >= n:
        raise ValueError("m must be >= 1 and < n")

    # Create a directed graph
    G = nx.DiGraph()

    # Start with an initial connected directed graph of m nodes
    for i in range(m):
        G.add_node(i)
        for j in range(i):
            G.add_edge(j, i)  # Initial directed edges

    # Add the remaining nodes to the graph
    for new_node in range(m, n):
        # Add the new node
        G.add_node(new_node)

        # Calculate the total out-degree of all existing nodes
        # Notice that here edges go from cited to citing, so we are interested in out-degree
        total_out_degree = sum(dict(G.out_degree()).values())

        # Create a list of existing nodes to connect to
        targets = set()
        while len(targets) < m:
            # Preferential attachment: choose a node with probability proportional to its in-degree
            target = random.choices(
                list(G.nodes()),
                weights=[
                    G.out_degree(node) + 1 for node in G.nodes()
                ],  # +1 to avoid zero probability
                k=1,
            )[0]

            # Add the target to the set (ensures unique connections)
            targets.add(target)

        # Add directed edges from the new node to the selected targets
        # edges go from cited to citing, so from target to new_node
        for target in targets:
            G.add_edge(target, new_node)

    return G


def directed_watts_strogatz(n, k, p):
    """
    Generates a directed Watts-Strogatz small-world network.

    Parameters:
    n (int): Number of nodes
    k (int): Each node is initially connected to k nearest neighbors
    p (float): Probability of rewiring an edge

    Returns:
    nx.DiGraph: A directed Watts-Strogatz network
    """

    # Step 1: Create a directed ring lattice
    G = nx.DiGraph()
    nodes = list(range(n))

    for i in range(n):
        for j in range(1, k // 2 + 1):  # k//2 neighbors in each direction
            neighbor = (i + j) % n
            G.add_edge(i, neighbor)  # Forward direction
            G.add_edge(neighbor, i)  # Backward direction (ensuring directed edges)

    # Step 2: Rewire edges with probability p
    edges = list(G.edges())  # Get the initial edges
    for edge in edges:
        u, v = edge
        if random.random() < p:
            G.remove_edge(u, v)  # Remove old edge

            new_v = random.choice(nodes)
            while new_v == u or G.has_edge(u, new_v):  # Avoid self-loops and duplicates
                new_v = random.choice(nodes)

            G.add_edge(u, new_v)  # Add new directed edge

    return G
