"""Dataset loading for Planetoid and common GAD benchmarks."""
from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

PLANETOID = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
}


def _download_planetoid(name: str) -> Path:
    base = DATA_ROOT / name
    base.mkdir(parents=True, exist_ok=True)
    url_root = f"https://github.com/kimiyoung/planetoid/raw/master/data/ind.{name}."
    for suffix in ["x", "tx", "allx", "graph", "test.index", "y", "ty", "ally"]:
        dest = base / f"ind.{name}.{suffix}"
        if not dest.exists():
            urllib.request.urlretrieve(url_root + suffix, dest)
    return base


def _parse_index_file(path: Path) -> list[int]:
    with open(path, "rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            return [int(line.strip()) for line in f if line.strip()]


def load_planetoid(name: str) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Load Planetoid graph (features + adjacency). Class labels replaced by zeros for GAD."""
    name = name.lower()
    folder = _download_planetoid(name)
    with open(folder / f"ind.{name}.allx", "rb") as f:
        allx = pickle.load(f, encoding="latin1")
    with open(folder / f"ind.{name}.tx", "rb") as f:
        tx = pickle.load(f, encoding="latin1")
    with open(folder / f"ind.{name}.graph", "rb") as f:
        graph = pickle.load(f, encoding="latin1")

    features = sp.vstack((allx, tx)).toarray().astype(np.float64)
    n_feat = features.shape[0]
    max_node = max(max(graph.keys()), max(max(nbrs) for nbrs in graph.values()))
    n = max(n_feat, max_node + 1)
    if n > n_feat:
        pad = np.zeros((n - n_feat, features.shape[1]), dtype=np.float64)
        features = np.vstack([features, pad])
    rows, cols = [], []
    for node, nbrs in graph.items():
        for j in nbrs:
            if node < n and j < n:
                rows.append(node)
                cols.append(j)
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    labels = np.zeros(n, dtype=np.int32)  # replaced by inject_anomaly
    return features, adj, labels


def _load_npz_dataset(path: Path) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    if "adj" in d:
        adj = sp.csr_matrix(
            (d["adj_data"], d["adj_indices"], d["adj_indptr"]),
            shape=tuple(d["adj_shape"]),
        )
    elif "A" in d:
        adj = sp.csr_matrix(d["A"])
    else:
        raise KeyError(f"No adjacency in {path}")
    if "attr" in d or "features" in d:
        key = "attr" if "attr" in d else "features"
        if key == "attr":
            features = sp.csr_matrix(
                (d["attr_data"], d["attr_indices"], d["attr_indptr"]),
                shape=tuple(d["attr_shape"]),
            ).toarray()
        else:
            features = np.asarray(d["features"], dtype=np.float64)
    elif "X" in d:
        features = np.asarray(d["X"], dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
    else:
        features = np.eye(adj.shape[0], dtype=np.float64)
    labels = np.zeros(adj.shape[0], dtype=np.int32)
    return features.astype(np.float64), adj.tocsr(), labels


def _load_mat_dataset_file(path: Path) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    import scipy.io as sio
    smat = sio.loadmat(str(path))
    for k in ("Network", "A", "network", "adj"):
        if k in smat:
            adj = sp.csr_matrix(smat[k])
            break
    else:
        raise KeyError(f"No adjacency key in {path}")
    for k in ("Attributes", "X", "attr", "features"):
        if k in smat:
            raw = smat[k]
            if sp.issparse(raw):
                features = raw.toarray().astype(np.float64)
            else:
                features = np.asarray(raw, dtype=np.float64)
            break
    else:
        features = np.eye(adj.shape[0], dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    labels = np.zeros(adj.shape[0], dtype=np.int32)
    return features, adj, labels


def _load_mat_dataset(name: str) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Load ACM/BlogCatalog/Flickr from data/{name}.npz or .mat; download if missing."""
    npz_path = DATA_ROOT / f"{name}.npz"
    mat_path = DATA_ROOT / f"{name}.mat"
    if not npz_path.exists() and not mat_path.exists():
        from subprocess import run
        import sys
        run([sys.executable, str(DATA_ROOT.parent / "scripts" / "download_datasets.py")], check=False)
    if npz_path.exists():
        return _load_npz_dataset(npz_path)
    if mat_path.exists():
        return _load_mat_dataset_file(mat_path)
    raise FileNotFoundError(
        f"No real data for {name}. Run: python scripts/download_datasets.py"
    )


def load_dataset(name: str) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    name = name.lower()
    if name in PLANETOID:
        return load_planetoid(name)
    if name in ("acm", "blogcatalog", "flickr"):
        return _load_mat_dataset(name)
    raise ValueError(f"Unknown dataset: {name}")
