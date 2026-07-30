"""Graph utilities: normalization, adjacency, feature prep."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def row_l1_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.abs(x).sum(axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def normalize_adjacency(adj: sp.spmatrix) -> sp.csr_matrix:
    adj = sp.csr_matrix(adj)
    adj = adj + sp.eye(adj.shape[0], format="csr")
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float64)
    np.power(deg, -0.5, out=deg_inv_sqrt, where=deg > 0)
    d = sp.diags(deg_inv_sqrt)
    return (d @ adj @ d).tocsr()


def adjacency_to_sparse(adj: sp.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = sp.coo_matrix(adj)
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float64)


def rewire_edges(adj: sp.spmatrix, frac: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Randomly rewire `frac` fraction of undirected edges."""
    if frac <= 0:
        return adj.tocsr()
    adj = sp.triu(sp.csr_matrix(adj), k=1)
    rows, cols = adj.nonzero()
    n_swap = int(len(rows) * frac)
    if n_swap == 0:
        return (adj + adj.T).tocsr()
    n = adj.shape[0]
    rows = rows.copy()
    cols = cols.copy()
    for k in range(n_swap):
        i = rng.integers(0, len(rows))
        u, v = int(rows[i]), int(cols[i])
        w = int(rng.integers(0, n))
        while w == u or w == v:
            w = int(rng.integers(0, n))
        rows[i], cols[i] = min(u, w), max(u, w)
    new_adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    new_adj = new_adj + new_adj.T
    new_adj.data = np.ones_like(new_adj.data)
    return new_adj.tocsr()
