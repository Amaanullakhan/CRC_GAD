"""
Score-based GAD backbones for CRC-GAD.

All functions return a length-N anomaly score vector (higher = more anomalous).
CRC-GAD's conformal wrapper is backbone-agnostic and consumes these scores only.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp

from .graph_utils import normalize_adjacency, row_l1_normalize
from .scorer import train_scorer


def _maybe_align(scores, labels, val_idx, train_idx=None):
    from .scorer import align_score_polarity

    if val_idx is not None and labels is not None:
        return align_score_polarity(scores, labels, val_idx, train_idx)
    return scores


def score_degree(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    val_idx=None,
    labels=None,
    train_idx=None,
    **_,
) -> np.ndarray:
    """Structural baseline: low degree → higher anomaly score (common GAD heuristic)."""
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)
    scores = 1.0 / (deg + 1.0) + 1e-8 * rng.standard_normal(len(deg))
    return _maybe_align(scores, labels, val_idx, train_idx)


def score_feature_norm(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    val_idx=None,
    labels=None,
    train_idx=None,
    **_,
) -> np.ndarray:
    """Attribute baseline: L2 feature norm as nonconformity."""
    x = row_l1_normalize(features)
    scores = np.linalg.norm(x, axis=1) + 1e-8 * rng.standard_normal(x.shape[0])
    return _maybe_align(scores, labels, val_idx, train_idx)


def score_attr_deviation(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    val_idx=None,
    labels=None,
    train_idx=None,
    **_,
) -> np.ndarray:
    """Attribute deviation from neighbor mean (local inconsistency)."""
    x = row_l1_normalize(features)
    adj_bin = adj.tocsr().copy()
    adj_bin.data = np.ones_like(adj_bin.data)
    deg = np.asarray(adj_bin.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1.0)
    nbr_mean = adj_bin @ x / deg_safe[:, None]
    scores = np.linalg.norm(x - nbr_mean, axis=1) + 1e-8 * rng.standard_normal(x.shape[0])
    return _maybe_align(scores, labels, val_idx, train_idx)


def score_dominant_style(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    embed_dim: int = 64,
    epochs: int = 120,
    lr: float = 0.05,
    alpha_struct: float = 0.5,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    **_,
) -> np.ndarray:
    """
    Lightweight DOMINANT-inspired reconstruction scorer (Ding et al., 2019).

    NumPy GCN encoder + linear attribute decoder; structure term from
    disagreement with neighbor-mean embeddings. Polarity-aligned on train∪val.
    """
    x = row_l1_normalize(features.astype(np.float64))
    adj_norm = normalize_adjacency(adj)
    n, f = x.shape
    # Project very high-dim attrs for speed/stability
    if f > 512:
        proj = rng.standard_normal((f, 512)) * np.sqrt(1.0 / f)
        x = x @ proj
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        f = x.shape[1]
    w1 = rng.standard_normal((f, embed_dim)) * np.sqrt(2.0 / (f + embed_dim))
    w2 = rng.standard_normal((embed_dim, embed_dim)) * np.sqrt(1.0 / embed_dim)
    w_dec = rng.standard_normal((embed_dim, f)) * np.sqrt(2.0 / (embed_dim + f))

    for _ in range(epochs):
        h1 = np.maximum(adj_norm @ x @ w1, 0.0)
        h = adj_norm @ h1 @ w2
        x_hat = h @ w_dec
        resid = x_hat - x
        w_dec -= lr * (h.T @ resid) / n
        w2 -= lr * (h1.T @ (resid @ w_dec.T)) / n

    h1 = np.maximum(adj_norm @ x @ w1, 0.0)
    h = adj_norm @ h1 @ w2
    x_hat = h @ w_dec
    attr_err = np.linalg.norm(x - x_hat, axis=1)
    deg = np.asarray(adj_norm.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1e-8)
    nbr = (adj_norm @ h) / deg_safe[:, None]
    cos = np.sum(h * nbr, axis=1) / (
        np.linalg.norm(h, axis=1) * np.linalg.norm(nbr, axis=1) + 1e-8
    )
    struct_err = 1.0 - cos
    scores = (alpha_struct * struct_err + (1.0 - alpha_struct) * attr_err).astype(np.float64)
    return _maybe_align(scores, labels, val_idx, train_idx)


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
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Real PyTorch DOMINANT (Ding et al. 2019 / official architecture)."""
    from .dominant_torch import score_dominant_pytorch

    scores = score_dominant_pytorch(features, adj, rng, **kwargs)
    return _maybe_align(scores, labels, val_idx, train_idx)


def score_conad_pytorch_wrapper(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """CONAD-style Siamese contrastive backbone (Xu et al. 2022 principle)."""
    from .conad_torch import score_conad_pytorch

    scores = score_conad_pytorch(features, adj, rng, **kwargs)
    return _maybe_align(scores, labels, val_idx, train_idx)


BACKBONES: dict[str, Callable] = {
    "cola": score_cola,
    "dominant": score_dominant_pytorch_wrapper,
    "conad": score_conad_pytorch_wrapper,
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
