"""
PyTorch DOMINANT (Ding et al., SDM 2019).

Faithful reimplementation of the authors' public PyTorch code
(https://github.com/kaize0409/GCN_AnomalyDetection_pytorch):
shared 2-layer GCN encoder, GCN attribute decoder, GCN structure decoder
with adjacency reconstruction A_hat = Z Z^T, and per-node score
alpha * attr_err + (1-alpha) * struct_err.

Requires: torch (CPU ok). Used as a frozen backbone; CRC-GAD wraps the scores.
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
            "PyTorch DOMINANT requires torch. Install CPU torch in a short-path venv "
            "(Windows long-path issues are common), e.g. "
            "python -m venv C:\\t\\py\\v && C:\\t\\py\\v\\Scripts\\pip install torch"
        ) from e
    return torch, nn, F


def _sparse_to_torch_sparse(adj: sp.spmatrix, torch):
    adj = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([adj.row, adj.col]).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape)).coalesce()


def build_dominant_modules(feat_size: int, hidden_size: int, dropout: float):
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

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gc1 = GraphConvolution(feat_size, hidden_size)
            self.gc2 = GraphConvolution(hidden_size, hidden_size)
            self.dropout = dropout

        def forward(self, x, adj):
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.gc2(x, adj))
            return x

    class AttributeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gc1 = GraphConvolution(hidden_size, hidden_size)
            self.gc2 = GraphConvolution(hidden_size, feat_size)
            self.dropout = dropout

        def forward(self, x, adj):
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.gc2(x, adj))
            return x

    class StructureDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gc1 = GraphConvolution(hidden_size, hidden_size)
            self.dropout = dropout

        def forward(self, x, adj):
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            return x @ x.T

    class Dominant(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_encoder = Encoder()
            self.attr_decoder = AttributeDecoder()
            self.struct_decoder = StructureDecoder()

        def forward(self, x, adj):
            z = self.shared_encoder(x, adj)
            x_hat = self.attr_decoder(z, adj)
            a_hat = self.struct_decoder(z, adj)
            return a_hat, x_hat

    return Dominant()


def dominant_loss(adj_label, a_hat, attrs, x_hat, alpha: float, torch):
    """Per-node DOMINANT reconstruction scores (higher = more anomalous)."""
    attr_err = torch.sqrt(torch.sum((x_hat - attrs) ** 2, dim=1))
    struct_err = torch.sqrt(torch.sum((a_hat - adj_label) ** 2, dim=1))
    score = alpha * attr_err + (1.0 - alpha) * struct_err
    return score, float(struct_err.mean().item()), float(attr_err.mean().item())


def score_dominant_pytorch(
    features: np.ndarray,
    adj: sp.spmatrix,
    rng: np.random.Generator,
    hidden_dim: int = 64,
    epochs: int = 100,
    lr: float = 5e-3,
    dropout: float = 0.3,
    alpha: float = 0.8,
    device: Optional[str] = None,
    **_,
) -> np.ndarray:
    """
    Train PyTorch DOMINANT on the full attributed graph and return node scores.

    Matches the official training loop (full-batch Adam on reconstruction loss).
    Labels are not used. The CRC-GAD conformal layer is applied separately.
    """
    torch, _, _ = _require_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Reproducibility within seed (numpy rng -> torch seed)
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_np = row_l1_normalize(features.astype(np.float64)).astype(np.float32)
    adj_norm = normalize_adjacency(adj)
    # Structure target: binary adjacency with self-loops (as in authors' adj_label)
    adj_bin = adj.tocsr().copy()
    adj_bin.data = np.ones_like(adj_bin.data, dtype=np.float32)
    adj_label = (adj_bin + sp.eye(adj_bin.shape[0], format="csr", dtype=np.float32)).tocsr()

    attrs = torch.from_numpy(x_np).to(device_t)
    adj_sp = _sparse_to_torch_sparse(adj_norm, torch).to(device_t)
    # Dense label adj for structure loss (N×N); Planetoid sizes are fine on CPU
    adj_lab = torch.from_numpy(adj_label.toarray().astype(np.float32)).to(device_t)

    model = build_dominant_modules(x_np.shape[1], hidden_dim, dropout).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        a_hat, x_hat = model(attrs, adj_sp)
        score, _, _ = dominant_loss(adj_lab, a_hat, attrs, x_hat, alpha, torch)
        loss = score.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        a_hat, x_hat = model(attrs, adj_sp)
        score, _, _ = dominant_loss(adj_lab, a_hat, attrs, x_hat, alpha, torch)
        return score.detach().cpu().numpy().astype(np.float64)
