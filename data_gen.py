\
from __future__ import annotations

from typing import Tuple
import numpy as np
import networkx as nx


def random_dag(nodes: int, edges: int, seed: int) -> nx.DiGraph:
    """
    Random DAG with guaranteed weak connectivity via a random chain + extra edges.
    """
    rng = np.random.default_rng(seed)
    max_edges = nodes * (nodes - 1) // 2
    edges = max(edges, nodes - 1)
    edges = min(edges, max_edges)

    G = nx.DiGraph()
    G.add_nodes_from(range(nodes))

    topo = list(range(nodes))
    rng.shuffle(topo)

    # chain ensures connectivity of skeleton
    for i in range(nodes - 1):
        G.add_edge(topo[i], topo[i + 1])

    current_edges = nodes - 1
    topo_pos = {node: i for i, node in enumerate(topo)}

    while current_edges < edges:
        a = int(rng.choice(topo))
        b = int(rng.choice(topo))
        if a == b:
            continue
        if topo_pos[a] < topo_pos[b] and not G.has_edge(a, b):
            G.add_edge(a, b)
            current_edges += 1

    assert nx.is_directed_acyclic_graph(G)
    return G


def gen_data_nonlinear(G: nx.DiGraph, n: int, sem_type: str, noise_scale: float, seed: int) -> np.ndarray:
    """
    SEM generator:
    - sem_type="square" or "sigmoid"
    - X_j = f(sum_{i in Pa(j)} X_i) + noise
    """
    rng = np.random.default_rng(seed)
    d = G.number_of_nodes()
    X = rng.normal(size=(n, d)) * noise_scale

    ordered = list(nx.topological_sort(G))
    for j in ordered:
        parents = list(G.predecessors(j))
        if len(parents) == 0:
            continue
        s = X[:, parents].sum(axis=1)
        if sem_type == "sigmoid":
            X[:, j] = 1 / (1 + np.exp(-s)) + rng.normal(scale=noise_scale, size=n)
        else:
            X[:, j] = (s ** 2) + rng.normal(scale=noise_scale, size=n)

    return X
