"""Anomaly injection: structural cliques + attribute replacement (DOMINANT/CoLA)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances


def inject_anomalies(
    features: np.ndarray,
    adj: sp.spmatrix,
    ratio: float,
    structural_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """
    Inject anomalies into graph. Returns (features, adj, labels) with labels in {0,1}.
    Labels are for evaluation ONLY — never used in partition or training.
    """
    n = features.shape[0]
    n_anom = max(1, int(n * ratio))
    n_struct = int(n_anom * structural_ratio)
    n_attr = n_anom - n_struct

    labels = np.zeros(n, dtype=np.int32)
    normal_idx = np.arange(n)
    anom_nodes = rng.choice(normal_idx, size=n_anom, replace=False)
    labels[anom_nodes] = 1

    features = features.copy()
    adj = sp.lil_matrix(adj)

    # Structural: dense cliques among anomaly nodes
    struct_nodes = anom_nodes[:n_struct]
    for i in range(len(struct_nodes)):
        for j in range(i + 1, len(struct_nodes)):
            u, v = int(struct_nodes[i]), int(struct_nodes[j])
            adj[u, v] = 1
            adj[v, u] = 1

    # Attribute: replace features with maximally distant normal node features
    attr_nodes = anom_nodes[n_struct:]
    normal_pool = np.setdiff1d(normal_idx, anom_nodes)
    if len(attr_nodes) > 0 and len(normal_pool) > 0:
        dist = euclidean_distances(features[normal_pool], features[normal_pool])
        # pick normal with highest avg distance to other normals as donor source pattern
        for node in attr_nodes:
            target = int(rng.choice(normal_pool))
            # replace with feature from node farthest from target in normal pool
            d = euclidean_distances(features[[target]], features[normal_pool]).flatten()
            far_idx = int(normal_pool[np.argmax(d)])
            features[node] = features[far_idx]

    return features, adj.tocsr(), labels
