\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx


def adjacency_from_nx(G: nx.DiGraph, node_names: List[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_names)}
    n = len(node_names)
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        u = str(u); v = str(v)
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1.0
    return A


def build_acyclic_graph_from_weights(W: np.ndarray, node_names: List[str], thr: float) -> nx.DiGraph:
    n = min(len(node_names), W.shape[0], W.shape[1])
    names = node_names[:n]
    edges = []
    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i != j and W[i, j] > thr:
                edges.append((src, tgt, float(W[i, j])))
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)

    G = nx.DiGraph()
    G.add_nodes_from(names)
    for src, tgt, w in edges_sorted:
        G.add_edge(src, tgt, weight=w)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(src, tgt)
    return G


def compute_shd_from_weights(G_true: nx.DiGraph, W: np.ndarray, node_names: List[str], thr: float) -> Dict:
    n = min(len(node_names), W.shape[0], W.shape[1])
    names = node_names[:n]

    G_true_str = nx.relabel_nodes(G_true, lambda x: str(x))
    edges_true = set((u, v) for (u, v) in G_true_str.edges() if u in names and v in names)

    edges_learned = set()
    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i != j and W[i, j] > thr:
                edges_learned.add((src, tgt))

    shd = 0
    deletions = 0
    reversals = 0

    edges_true_mut = set(edges_true)
    edges_learned_mut = set(edges_learned)

    for e in list(edges_true_mut):
        if e in edges_learned_mut:
            edges_true_mut.remove(e)
            edges_learned_mut.remove(e)
        elif (e[1], e[0]) in edges_learned_mut:
            shd += 1
            reversals += 1
            edges_true_mut.remove(e)
            edges_learned_mut.remove((e[1], e[0]))
        else:
            shd += 1
            deletions += 1
            edges_true_mut.remove(e)

    additions = len(edges_learned_mut)
    shd += additions

    return {
        "threshold": float(thr),
        "shd": int(shd),
        "additions": int(additions),
        "deletions": int(deletions),
        "reversals": int(reversals),
        "edges_true": int(len(edges_true)),
        "edges_learned": int(len(edges_learned)),
    }


def compute_kl_adj_distributions(G_true: nx.DiGraph, G_learned: nx.DiGraph, node_names: List[str], eps: float = 1e-9) -> Dict:
    names = list(node_names)
    A_true = adjacency_from_nx(G_true, names)
    A_learn = adjacency_from_nx(G_learned, names)

    p = A_true.flatten() + eps
    q = A_learn.flatten() + eps
    p /= p.sum()
    q /= q.sum()

    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return {"kl_true_learned": kl_pq, "kl_learned_true": kl_qp}
