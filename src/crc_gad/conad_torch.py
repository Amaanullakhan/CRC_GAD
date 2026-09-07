"""
PyTorch CONAD-style contrastive backbone (Xu et al., IJCAI 2022).

Siamese GCN + anomaly-prior augmentations (attribute noise, edge dropout) with
InfoNCE view agreement, plus a light attribute decoder. Node anomaly score mixes
reconstruction error and view disagreement — the combination that empirically
ranks injected anomalies under our pinned protocol.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.sparse as sp

from .graph_utils import normalize_adjacency, row_l1_normalize


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError(
            "PyTorch CONAD requires torch. Use the short-path venv on Windows, e.g. "
            "C:\\t\\py\\v\\Scripts\\python.exe"
        ) from e
    return torch, nn, F


def _sparse_to_torch_sparse(adj: sp.spmatrix, torch):
    adj = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape)).coalesce()


def _augment_views(x, adj_sp, torch, rng, attr_noise: float, edge_drop: float):
    noise = torch.from_numpy(
        (attr_noise * rng.standard_normal(size=tuple(x.shape))).astype(np.float32)
    ).to(x.device)
    x_aug = x + noise
    if edge_drop <= 0.0 or (not adj_sp.is_sparse):
        return x_aug, adj_sp
    indices = adj_sp.coalesce().indices()
    values = adj_sp.coalesce().values()
    n_edges = values.numel()
    keep = torch.from_numpy(rng.random(n_edges) >= edge_drop).to(values.device)
    if int(keep.sum().item()) < max(1, n_edges // 10):
        keep = torch.ones_like(keep, dtype=torch.bool)
    adj_aug = torch.sparse_coo_tensor(indices[:, keep], values[keep], adj_sp.size()).coalesce()
    return x_aug, adj_aug


def build_conad_model(feat_size: int, hidden_size: int, dropout: float):
    torch, nn, F = _require_torch()

    class GraphConvolution(nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
            stdv = 1.0 / math.sqrt(out_features)
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

        def forward(self, x, adj):
            support = torch.mm(x, self.weight)
            out = torch.sparse.mm(adj, support) if adj.is_sparse else torch.mm(adj, support)
            return out + self.bias if self.bias is not None else out

    class CONAD(nn.Module):
        def __init__(self):
            super().__init__()
            self.gc1 = GraphConvolution(feat_size, hidden_size)
            self.gc2 = GraphConvolution(hidden_size, hidden_size)
            self.proj = nn.Linear(hidden_size, hidden_size)
            self.dec1 = GraphConvolution(hidden_size, hidden_size)
            self.dec2 = GraphConvolution(hidden_size, feat_size)
            self.dropout = dropout

        def encode(self, x, adj):
            h = F.relu(self.gc1(x, adj))
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(self.gc2(h, adj))
            z = F.normalize(self.proj(h), p=2, dim=1)
            return h, z

        def decode_attr(self, h, adj):
            x = F.relu(self.dec1(h, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            return self.dec2(x, adj)

        def forward(self, x, adj):
            h, z = self.encode(x, adj)
            x_hat = self.decode_attr(h, adj)
            return h, z, x_hat

    return CONAD()


def score_conad_pytorch(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 5e-3,
    dropout: float = 0.2,
    attr_noise: float = 0.15,
    edge_drop: float = 0.2,
    temperature: float = 0.2,
    alpha_recon: float = 0.7,
    device: Optional[str] = None,
    **_,
) -> np.ndarray:
    """
    Train CONAD-style Siamese GCN + attribute decoder; return node anomaly scores.

    score = alpha_recon * attr_err + (1 - alpha_recon) * (1 - cos(z, z_aug))
    """
    torch, _, F = _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_np = row_l1_normalize(features.astype(np.float64)).astype(np.float32)
    adj_norm = normalize_adjacency(adj)
    attrs = torch.from_numpy(x_np).to(device_t)
    adj_sp = _sparse_to_torch_sparse(adj_norm, torch).to(device_t)

    model = build_conad_model(x_np.shape[1], hidden_dim, dropout).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    n = attrs.shape[0]
    batch_size = 2048 if n > 4096 else n
    for _ in range(epochs):
        optimizer.zero_grad()
        x_aug, adj_aug = _augment_views(
            attrs, adj_sp, torch, rng, attr_noise=attr_noise, edge_drop=edge_drop
        )
        h, z, x_hat = model(attrs, adj_sp)
        _, z_aug, _ = model(x_aug, adj_aug)
        recon = torch.mean((x_hat - attrs) ** 2)
        if batch_size < n:
            idx = torch.from_numpy(
                rng.choice(n, size=batch_size, replace=False).astype(np.int64)
            ).to(device_t)
            z_b, z_aug_b = z[idx], z_aug[idx]
            labels = torch.arange(batch_size, device=device_t)
        else:
            z_b, z_aug_b = z, z_aug
            labels = torch.arange(n, device=device_t)
        logits = (z_b @ z_aug_b.T) / temperature
        contrast = F.cross_entropy(logits, labels)
        loss = recon + 0.5 * contrast
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_aug, adj_aug = _augment_views(
            attrs, adj_sp, torch, rng, attr_noise=attr_noise, edge_drop=edge_drop
        )
        h, z, x_hat = model(attrs, adj_sp)
        _, z_aug, _ = model(x_aug, adj_aug)
        attr_err = torch.sqrt(torch.sum((x_hat - attrs) ** 2, dim=1))
        attr_err = attr_err / (attr_err.mean() + 1e-8)
        cos = torch.sum(z * z_aug, dim=1).clamp(-1.0, 1.0)
        view_err = (1.0 - cos)
        view_err = view_err / (view_err.mean() + 1e-8)
        score = alpha_recon * attr_err + (1.0 - alpha_recon) * view_err
        # Orientation fixed post-hoc via align_score_polarity (train∪val)
        return score.detach().cpu().numpy().astype(np.float64)
