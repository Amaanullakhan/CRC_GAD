#!/usr/bin/env python3
"""Download real GAD benchmark files (DeepRobust .npz format)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://raw.githubusercontent.com/DSE-MSU/DeepRobust/master/deeprobust/graph/data"

DATASETS = {
    "acm": f"{BASE}/acm.npz",
    "blogcatalog": f"{BASE}/blogcatalog.npz",
    "flickr": f"{BASE}/flickr.npz",
}
MAT_FALLBACK = {
    "acm": "https://github.com/zm235575/graph-datasets/raw/main/ACM.mat",
    "flickr": "https://github.com/zm235575/graph-datasets/raw/main/Flickr.mat",
    "blogcatalog": "https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection/raw/main/Datasets/BlogCatalog.mat",
}
MAT_FALLBACK_ALT = {
    "blogcatalog": "http://leitang.net/code/social-dimension/data/blogcatalog.mat",
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in DATASETS.items():
        dest = DATA / f"{name}.npz"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"skip {name} (exists)")
            continue
        print(f"Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  -> {dest} ({dest.stat().st_size} bytes)")
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            mat_url = MAT_FALLBACK.get(name)
            if not mat_url:
                continue
            mat_dest = DATA / f"{name}.mat"
            for mat_url in [MAT_FALLBACK.get(name), MAT_FALLBACK_ALT.get(name)]:
                if not mat_url:
                    continue
                try:
                    urllib.request.urlretrieve(mat_url, mat_dest)
                    print(f"  -> fallback mat {mat_dest}")
                    break
                except Exception as e2:
                    print(f"  mat fallback failed ({mat_url}): {e2}")


if __name__ == "__main__":
    main()
