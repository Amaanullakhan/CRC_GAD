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


def load_planetoid(name: str, return_class_labels: bool = False) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Load Planetoid graph. By default labels are zeros (injection fills them).

    If return_class_labels=True, return Planetoid class ids (organic rare-class proxy).
    """
    name = name.lower()
    folder = _download_planetoid(name)
    with open(folder / f"ind.{name}.allx", "rb") as f:
        allx = pickle.load(f, encoding="latin1")
    with open(folder / f"ind.{name}.tx", "rb") as f:
        tx = pickle.load(f, encoding="latin1")
    with open(folder / f"ind.{name}.graph", "rb") as f:
        graph = pickle.load(f, encoding="latin1")
    with open(folder / f"ind.{name}.ally", "rb") as f:
        ally = pickle.load(f, encoding="latin1")
    with open(folder / f"ind.{name}.ty", "rb") as f:
        ty = pickle.load(f, encoding="latin1")

    features = sp.vstack((allx, tx)).toarray().astype(np.float64)
    y_stack = np.vstack((np.asarray(ally), np.asarray(ty)))
    class_ids = np.argmax(y_stack, axis=1).astype(np.int32)

    n_feat = features.shape[0]
    max_node = max(max(graph.keys()), max(max(nbrs) for nbrs in graph.values()))
    n = max(n_feat, max_node + 1)
    if n > n_feat:
        pad = np.zeros((n - n_feat, features.shape[1]), dtype=np.float64)
        features = np.vstack([features, pad])
        class_ids = np.concatenate([class_ids, np.zeros(n - n_feat, dtype=np.int32)])
    rows, cols = [], []
    for node, nbrs in graph.items():
        for j in nbrs:
            if node < n and j < n:
                rows.append(node)
                cols.append(j)
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    if return_class_labels:
        return features, adj, class_ids
    return features, adj, np.zeros(n, dtype=np.int32)


def load_organic_rare_class(name: str = "cora") -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Organic proxy: mark the rarest Planetoid class as anomalies.

    Not a fraud label set — a non-injected anomaly definition from real class
    structure for calibration stress tests.
    """
    features, adj, class_ids = load_planetoid(name, return_class_labels=True)
    uniq, counts = np.unique(class_ids, return_counts=True)
    rare = int(uniq[np.argmin(counts)])
    labels = (class_ids == rare).astype(np.int32)
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
            features = raw.toarray().astype(np.float64) if sp.issparse(raw) else np.asarray(raw, dtype=np.float64)
            break
    else:
        features = np.eye(adj.shape[0], dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    return features, adj, np.zeros(adj.shape[0], dtype=np.int32)


def _load_mat_dataset(name: str) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
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
    raise FileNotFoundError(f"No real data for {name}. Run: python scripts/download_datasets.py")


def load_yelpchi() -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    """Load YelpChi fraud graph with organic (non-injected) fraud labels.

    Prefers a local ``data/yelpchi.npz`` with keys features/attr, adj_*, label/y.
    If missing, attempts a public CARE-GNN-compatible download into ``data/``.
    Multi-relational edges are collapsed to an undirected binary adjacency.
    """
    npz_path = DATA_ROOT / "yelpchi.npz"
    if not npz_path.exists():
        _try_download_yelpchi(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            "YelpChi not found. Place data/yelpchi.npz with keys "
            "features (or attr_*), adj_* / A, and label (or y)."
        )
    d = np.load(npz_path, allow_pickle=True)
    if "features" in d:
        features = np.asarray(d["features"], dtype=np.float64)
    elif "attr_data" in d:
        features = sp.csr_matrix(
            (d["attr_data"], d["attr_indices"], d["attr_indptr"]),
            shape=tuple(d["attr_shape"]),
        ).toarray().astype(np.float64)
    elif "X" in d:
        features = np.asarray(d["X"], dtype=np.float64)
    else:
        raise KeyError("yelpchi.npz missing features")

    if "adj_data" in d:
        adj = sp.csr_matrix(
            (d["adj_data"], d["adj_indices"], d["adj_indptr"]),
            shape=tuple(d["adj_shape"]),
        )
    elif "A" in d:
        raw_a = d["A"]
        # np.savez may wrap a scipy sparse as a 0-d object array
        if isinstance(raw_a, np.ndarray) and raw_a.dtype == object:
            raw_a = raw_a.item()
        adj = sp.csr_matrix(raw_a)
    else:
        raise KeyError("yelpchi.npz missing adjacency")

    if "label" in d:
        labels = np.asarray(d["label"]).ravel().astype(np.int32)
    elif "y" in d:
        labels = np.asarray(d["y"]).ravel().astype(np.int32)
    else:
        raise KeyError("yelpchi.npz missing labels")

    adj = adj.tocsr()
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    # Drop self-loops for degree heuristics consistency
    adj.setdiag(0)
    adj.eliminate_zeros()
    n = adj.shape[0]
    if features.shape[0] != n:
        raise ValueError(f"YelpChi feature/adj size mismatch: {features.shape[0]} vs {n}")
    if labels.shape[0] != n:
        raise ValueError(f"YelpChi label/adj size mismatch: {labels.shape[0]} vs {n}")
    labels = (labels > 0).astype(np.int32)
    return features, adj, labels


def _try_download_yelpchi(dest_npz: Path) -> None:
    """Best-effort download of a preprocessed YelpChi npz (organic fraud labels)."""
    import zipfile
    import tempfile

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    # Public mirrors used by fraud-graph papers (CARE-GNN / DGL exports)
    urls = [
        "https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip",
        "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip",
    ]
    zip_path = DATA_ROOT / "YelpChi.zip"
    for url in urls:
        try:
            print(f"Downloading YelpChi from {url} ...", flush=True)
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            print(f"YelpChi download failed ({url}): {e}", flush=True)
    if not zip_path.exists():
        return

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with tempfile.TemporaryDirectory() as td:
                zf.extractall(td)
                td_path = Path(td)
                mat_files = list(td_path.rglob("*.mat")) + list(DATA_ROOT.glob("*yelp*.mat"))
                # Some releases ship .mat; convert if scipy can read
                if mat_files:
                    import scipy.io as sio

                    smat = sio.loadmat(str(mat_files[0]))
                    # CARE-GNN YelpChi.mat typically: homo, features, label
                    if "homo" in smat:
                        adj = sp.csr_matrix(smat["homo"])
                    elif "Network" in smat:
                        adj = sp.csr_matrix(smat["Network"])
                    else:
                        raise KeyError(f"No homo/Network in {mat_files[0]}")
                    if "features" in smat:
                        raw = smat["features"]
                        features = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
                    elif "Attributes" in smat:
                        raw = smat["Attributes"]
                        features = raw.toarray() if sp.issparse(raw) else np.asarray(raw)
                    else:
                        raise KeyError("No features in YelpChi mat")
                    if "label" in smat:
                        labels = np.asarray(smat["label"]).ravel()
                    elif "Label" in smat:
                        labels = np.asarray(smat["Label"]).ravel()
                    else:
                        raise KeyError("No label in YelpChi mat")
                    labels = labels.astype(np.int32).ravel()
                    adj = adj.tocsr()
                    np.savez_compressed(
                        dest_npz,
                        features=np.asarray(features, dtype=np.float64),
                        adj_data=adj.data,
                        adj_indices=adj.indices,
                        adj_indptr=adj.indptr,
                        adj_shape=np.asarray(adj.shape, dtype=np.int64),
                        label=labels,
                    )
                    print(f"Wrote {dest_npz}", flush=True)
                    return
                npz_files = list(td_path.rglob("*.npz"))
                if npz_files:
                    Path(npz_files[0]).replace(dest_npz)
                    print(f"Wrote {dest_npz}", flush=True)
    except Exception as e:
        print(f"YelpChi extract/convert failed: {e}", flush=True)


def load_dataset(name: str) -> Tuple[np.ndarray, sp.csr_matrix, np.ndarray]:
    name = name.lower()
    if name.startswith("organic_"):
        return load_organic_rare_class(name.replace("organic_", "", 1))
    if name in ("yelpchi", "yelp_chi", "yelp"):
        return load_yelpchi()
    if name in PLANETOID:
        return load_planetoid(name)
    if name in ("acm", "blogcatalog", "flickr"):
        return _load_mat_dataset(name)
    raise ValueError(f"Unknown dataset: {name}")
