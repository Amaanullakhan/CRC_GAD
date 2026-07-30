"""
CoLA-style contrastive scorer (Liu et al., IEEE TNNLS 2022) — NumPy implementation.

Backbone only; conformal layer is separate (conformal_calibrate.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from .graph_utils import normalize_adjacency, row_l1_normalize


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class ScorerModel:
    weights: list[np.ndarray]
    adj_norm: sp.csr_matrix

    def embed(self, x: np.ndarray) -> np.ndarray:
        h = row_l1_normalize(x)
        for i, w in enumerate(self.weights):
            h = self.adj_norm @ h @ w
            if i < len(self.weights) - 1:
                h = _relu(h)
        return np.asarray(h)

    def score_nodes(self, x: np.ndarray, rng: np.random.Generator, rounds: int, num_neg: int) -> np.ndarray:
        """Vectorized multi-round positive-negative gap (Eq. cola_score simplified)."""
        h = self.embed(x)
        adj = self.adj_norm
        n = h.shape[0]
        # neighbor mean context (positive)
        deg = np.asarray(adj.sum(axis=1)).flatten()
        deg_safe = np.maximum(deg, 1.0)
        ctx = adj @ h
        ctx = ctx / deg_safe[:, None]
        pos = np.sum(h * ctx, axis=1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(ctx, axis=1) + 1e-8)

        scores = np.zeros(n, dtype=np.float64)
        for _ in range(rounds):
            # random negative context: shuffle node embeddings
            perm = rng.permutation(n)
            neg_ctx = ctx[perm]
            neg = np.sum(h * neg_ctx, axis=1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(neg_ctx, axis=1) + 1e-8)
            scores += neg - pos
        return scores / rounds


def _init_weights(feat_dim: int, embed_dim: int, n_layers: int, rng: np.random.Generator) -> list[np.ndarray]:
    dims = [feat_dim] + [embed_dim] * n_layers
    ws = []
    for a, b in zip(dims[:-1], dims[1:]):
        w = rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))
        ws.append(w)
    return ws


def train_scorer(
    features: np.ndarray,
    adj: sp.spmatrix,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    labels: np.ndarray,
    cfg,
    rng: np.random.Generator,
) -> tuple[ScorerModel, np.ndarray]:
    """Train contrastive GCN scorer; early stop on val AUC (labels only for ES)."""
    x = row_l1_normalize(features)
    adj_norm = normalize_adjacency(adj)
    feat_dim = x.shape[1]
    weights = _init_weights(feat_dim, cfg.embed_dim, cfg.gcn_layers, rng)

    best_auc = -1.0
    best_weights = [w.copy() for w in weights]
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        h = x.copy()
        for i, w in enumerate(weights):
            h = adj_norm @ h @ w
            if i < len(weights) - 1:
                h = _relu(h)

        batch = train_idx.copy()
        rng.shuffle(batch)
        batch = batch[: min(cfg.batch_size, len(batch))]

        for i in batch:
            nbrs = adj_norm[i].indices
            if len(nbrs) == 0:
                continue
            j = int(rng.choice(nbrs))
            k = int(rng.integers(0, x.shape[0]))
            while k == i:
                k = int(rng.integers(0, x.shape[0]))
            lr = cfg.lr / max(len(batch), 1)
            weights[-1] -= lr * 0.01 * np.outer(h[i] - h[j], h[i] - h[k])

        if epoch % cfg.val_check_every == 0:
            model = ScorerModel(weights=[w.copy() for w in weights], adj_norm=adj_norm)
            val_scores = model.score_nodes(x, rng, cfg.val_scoring_rounds, cfg.num_negatives)
            if len(np.unique(labels[val_idx])) > 1:
                auc = roc_auc_score(labels[val_idx], val_scores[val_idx])
            else:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_weights = [w.copy() for w in weights]
                wait = 0
            else:
                wait += 1
            if wait >= cfg.patience:
                break

    model = ScorerModel(weights=best_weights, adj_norm=adj_norm)
    scores = model.score_nodes(x, rng, cfg.scoring_rounds, cfg.num_negatives)
    return model, scores
