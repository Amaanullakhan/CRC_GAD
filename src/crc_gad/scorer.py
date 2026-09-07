"""
CoLA-style contrastive scorer (Liu et al., IEEE TNNLS 2022) — NumPy implementation.

Force-improve mode:
- random projection when F is large
- batch InfoNCE training (more stable than pairwise on high-dim)
- optional multi-restart z-scored ensemble
- val∪train polarity alignment so higher score = more anomalous
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from .graph_utils import normalize_adjacency, row_l1_normalize


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def polarity_indices(val_idx: np.ndarray, train_idx: np.ndarray | None = None) -> np.ndarray:
    if train_idx is None or len(train_idx) == 0:
        return np.asarray(val_idx, dtype=np.int64)
    return np.unique(np.concatenate([np.asarray(val_idx), np.asarray(train_idx)])).astype(
        np.int64
    )


def align_score_polarity(
    scores: np.ndarray,
    labels: np.ndarray,
    val_idx: np.ndarray,
    train_idx: np.ndarray | None = None,
) -> np.ndarray:
    idx = polarity_indices(val_idx, train_idx)
    if len(idx) == 0 or len(np.unique(labels[idx])) < 2:
        return scores
    auc = float(roc_auc_score(labels[idx], scores[idx]))
    if auc < 0.5:
        return -scores
    return scores


def _zscore(s: np.ndarray) -> np.ndarray:
    return (s - float(s.mean())) / (float(s.std()) + 1e-8)


@dataclass
class ScorerModel:
    weights: list[np.ndarray]
    adj_norm: sp.csr_matrix
    feat_proj: np.ndarray | None = None

    def embed(self, x: np.ndarray) -> np.ndarray:
        h = row_l1_normalize(x)
        if self.feat_proj is not None:
            h = h @ self.feat_proj
            h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        for i, w in enumerate(self.weights):
            h = self.adj_norm @ h @ w
            if i < len(self.weights) - 1:
                h = _relu(h)
        norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
        return h / norms

    def score_nodes(
        self, x: np.ndarray, rng: np.random.Generator, rounds: int, num_neg: int
    ) -> np.ndarray:
        h = self.embed(x)
        adj = self.adj_norm
        n = h.shape[0]
        deg = np.asarray(adj.sum(axis=1)).flatten()
        deg_safe = np.maximum(deg, 1.0)
        ctx = (adj @ h) / deg_safe[:, None]
        ctx_n = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8)
        pos = np.sum(h * ctx_n, axis=1)
        m = max(1, int(num_neg))
        scores = np.zeros(n, dtype=np.float64)
        for _ in range(max(rounds, 1)):
            neg_acc = np.zeros(n, dtype=np.float64)
            for _m in range(m):
                perm = rng.permutation(n)
                neg_acc += np.sum(h * ctx_n[perm], axis=1)
            scores += (neg_acc / m) - pos
        return scores / max(rounds, 1)


def _init_weights(
    feat_dim: int, embed_dim: int, n_layers: int, rng: np.random.Generator
) -> list[np.ndarray]:
    dims = [feat_dim] + [embed_dim] * n_layers
    ws = []
    for a, b in zip(dims[:-1], dims[1:]):
        w = rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))
        ws.append(w)
    return ws


def _forward(x: np.ndarray, adj_norm: sp.spmatrix, weights: list[np.ndarray]) -> np.ndarray:
    h = x
    for i, w in enumerate(weights):
        h = adj_norm @ h @ w
        if i < len(weights) - 1:
            h = _relu(h)
    norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-8
    return h / norms


def _train_one(
    features: np.ndarray,
    adj: sp.spmatrix,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    labels: np.ndarray,
    cfg,
    rng: np.random.Generator,
) -> tuple[ScorerModel, np.ndarray]:
    x_raw = row_l1_normalize(features)
    # Always project when F > proj threshold (helps Cora bag-of-words too)
    max_feat = int(getattr(cfg, "max_feat_dim", 512))
    proj_dim = int(getattr(cfg, "proj_dim", 512))
    feat_proj = None
    if x_raw.shape[1] > max_feat:
        feat_proj = rng.standard_normal((x_raw.shape[1], proj_dim)) * np.sqrt(
            1.0 / x_raw.shape[1]
        )
        x = x_raw @ feat_proj
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    else:
        x = x_raw

    adj_norm = normalize_adjacency(adj)
    weights = _init_weights(x.shape[1], cfg.embed_dim, cfg.gcn_layers, rng)

    best_auc = -1.0
    best_weights = [w.copy() for w in weights]
    wait = 0
    lr = float(cfg.lr)
    max_epochs = int(cfg.max_epochs)
    if features.shape[0] > 10000 or features.shape[1] > 2000:
        max_epochs = min(max_epochs, 100)

    for epoch in range(1, max_epochs + 1):
        h = _forward(x, adj_norm, weights)
        batch = train_idx.copy()
        rng.shuffle(batch)
        batch = batch[: min(cfg.batch_size, len(batch))]
        if len(batch) < 4:
            continue

        deg = np.asarray(adj_norm.sum(axis=1)).ravel()
        deg_safe = np.maximum(deg, 1.0)
        ctx = (adj_norm @ h) / deg_safe[:, None]
        ctx = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8)

        hb = h[batch]
        cb = ctx[batch]
        logits = hb @ cb.T
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)
        targets = np.eye(len(batch))
        dlogits = (probs - targets) / len(batch)
        dhb = dlogits @ cb

        pre = x.copy()
        for i, w in enumerate(weights[:-1]):
            pre = _relu(adj_norm @ pre @ w)
        support = adj_norm @ pre
        dW = support[batch].T @ dhb / max(len(batch), 1)
        weights[-1] -= lr * dW
        weights[-1] *= 0.9995

        if epoch % cfg.val_check_every == 0 or epoch == max_epochs:
            model = ScorerModel(
                weights=[w.copy() for w in weights], adj_norm=adj_norm, feat_proj=feat_proj
            )
            val_scores = model.score_nodes(
                features, rng, cfg.val_scoring_rounds, cfg.num_negatives
            )
            val_scores = align_score_polarity(val_scores, labels, val_idx, train_idx)
            pidx = polarity_indices(val_idx, train_idx)
            if len(np.unique(labels[pidx])) > 1:
                auc = float(roc_auc_score(labels[pidx], val_scores[pidx]))
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

    model = ScorerModel(weights=best_weights, adj_norm=adj_norm, feat_proj=feat_proj)
    scores = model.score_nodes(features, rng, cfg.scoring_rounds, cfg.num_negatives)
    scores = align_score_polarity(scores, labels, val_idx, train_idx)
    return model, scores


def train_scorer(
    features: np.ndarray,
    adj: sp.spmatrix,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    labels: np.ndarray,
    cfg,
    rng: np.random.Generator,
) -> tuple[ScorerModel, np.ndarray]:
    """Train scorer; ensemble restarts on small graphs to reduce seed variance."""
    n_restarts = int(getattr(cfg, "n_restarts", 1))
    if features.shape[0] <= 5000:
        n_restarts = max(n_restarts, 2)

    models = []
    zs = []
    for r in range(n_restarts):
        # independent streams derived from parent rng
        child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        model, scores = _train_one(
            features, adj, train_idx, val_idx, labels, cfg, child
        )
        models.append(model)
        zs.append(_zscore(scores))

    scores = np.mean(np.stack(zs, axis=0), axis=0)
    scores = align_score_polarity(scores, labels, val_idx, train_idx)
    return models[-1], scores
