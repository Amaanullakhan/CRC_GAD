"""
Score-based GAD backbones for CRC-GAD.

All functions return a length-N anomaly score vector (higher = more anomalous).
CRC-GAD's conformal wrapper is backbone-agnostic and consumes these scores only.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from .graph_utils import normalize_adjacency, row_l1_normalize
from .scorer import ScorerModel, train_scorer


def score_degree(features: np.ndarray, adj: sp.spmatrix, rng: np.random.Generator, **_) -> np.ndarray:
    """Structural baseline: low degree → higher anomaly score (common GAD heuristic)."""
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)
    # Invert so low-degree nodes look anomalous; add tiny noise to break ties
    scores = 1.0 / (deg + 1.0) + 1e-8 * rng.standard_normal(len(deg))
    return scores


def score_feature_norm(features: np.ndarray, adj: sp.spmatrix, rng: np.random.Generator, **_) -> np.ndarray:
    """Attribute baseline: L2 feature norm as nonconformity."""
    x = row_l1_normalize(features)
    scores = np.linalg.norm(x, axis=1)
    return scores + 1e-8 * rng.standard_normal(len(scores))


def score_attr_deviation(features: np.ndarray, adj: sp.spmatrix, rng: np.random.Generator, **_) -> np.ndarray:
    """Attribute deviation from neighbor mean (local inconsistency)."""
    x = row_l1_normalize(features)
    adj_bin = adj.tocsr().copy()
    adj_bin.data = np.ones_like(adj_bin.data)
    deg = np.asarray(adj_bin.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1.0)
    nbr_mean = adj_bin @ x / deg_safe[:, None]
    scores = np.linalg.norm(x - nbr_mean, axis=1)
    return scores + 1e-8 * rng.standard_normal(len(scores))


def score_dominant_style(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    embed_dim: int = 64,
    epochs: int = 60,
    lr: float = 0.05,
    alpha_struct: float = 0.5,
    **_,
) -> np.ndarray:
    """
    Lightweight DOMINANT-inspired reconstruction scorer (Ding et al., 2019).

    NumPy GCN encoder + linear attribute decoder; structure term from
    disagreement with neighbor-mean embeddings. Same *scoring principle* as
    DOMINANT (structure + attribute reconstruction error), not a line-by-line
    reimplementation of the original PyTorch code.
    """
    x = row_l1_normalize(features.astype(np.float64))
    adj_norm = normalize_adjacency(adj)
    n, f = x.shape
    w1 = rng.standard_normal((f, embed_dim)) * np.sqrt(2.0 / (f + embed_dim))
    w2 = rng.standard_normal((embed_dim, embed_dim)) * np.sqrt(1.0 / embed_dim)
    w_dec = rng.standard_normal((embed_dim, f)) * np.sqrt(2.0 / (embed_dim + f))

    for _ in range(epochs):
        h1 = np.maximum(adj_norm @ x @ w1, 0.0)
        h = adj_norm @ h1 @ w2
        x_hat = h @ w_dec
        resid = x_hat - x
        # SGD on decoder then last GCN layer (attribute MSE only for stability)
        w_dec -= lr * (h.T @ resid) / n
        w2 -= lr * (h1.T @ (resid @ w_dec.T)) / n

    h1 = np.maximum(adj_norm @ x @ w1, 0.0)
    h = adj_norm @ h1 @ w2
    x_hat = h @ w_dec
    attr_err = np.linalg.norm(x - x_hat, axis=1)
    deg = np.asarray(adj_norm.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1e-8)
    nbr = (adj_norm @ h) / deg_safe[:, None]
    cos = np.sum(h * nbr, axis=1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(nbr, axis=1) + 1e-8)
    struct_err = 1.0 - cos
    return (alpha_struct * struct_err + (1.0 - alpha_struct) * attr_err).astype(np.float64)


def score_cola(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    cfg=None,
    **_,
) -> np.ndarray:
    """Simplified CoLA-inspired contrastive scorer (pinned implementation)."""
    if cfg is None or train_idx is None or val_idx is None or labels is None:
        raise ValueError("score_cola requires cfg, train_idx, val_idx, labels")
    _, scores = train_scorer(features, adj, train_idx, val_idx, labels, cfg, rng)
    return scores


def score_dominant_pytorch_wrapper(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Real PyTorch DOMINANT (Ding et al. 2019 / official architecture)."""
    from .dominant_torch import score_dominant_pytorch

    return score_dominant_pytorch(features, adj, rng, **kwargs)


BACKBONES: dict[str, Callable] = {
    "cola": score_cola,
    "dominant": score_dominant_pytorch_wrapper,
    "dominant_style": score_dominant_style,
    "degree": score_degree,
    "feature_norm": score_feature_norm,
    "attr_deviation": score_attr_deviation,
}


def get_scores(
    name: str,
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone {name}; choose from {list(BACKBONES)}")
    return BACKBONES[name](features, adj, rng, **kwargs)
