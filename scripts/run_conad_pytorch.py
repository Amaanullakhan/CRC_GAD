#!/usr/bin/env python3
"""Run PyTorch CONAD-style + CRC-GAD under the pinned injection/partition protocol."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crc_gad.backbones import get_scores
from crc_gad.config import DEFAULT
from crc_gad.conformal_calibrate import compute_metrics
from crc_gad.data import load_dataset
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cora,citeseer")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--out", default=str(ROOT / "results" / "conad_pytorch.csv"))
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = DEFAULT
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "study", "dataset", "seed", "backbone",
        "raw_auc_all", "conformal_auc", "fpr@0.05", "tpr@0.05",
        "flagged@0.05", "runtime_s",
    ]
    rows = []
    for dataset in datasets:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            features, adj, _ = load_dataset(dataset)
            features, adj, labels = inject_anomalies(
                features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
            )
            part = make_partition(
                len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
            )
            t0 = time.perf_counter()
            try:
                scores = get_scores(
                    "conad",
                    features,
                    adj,
                    rng,
                    epochs=args.epochs,
                    hidden_dim=64,
                    lr=5e-3,
                    dropout=0.2,
                    attr_noise=0.1,
                    edge_drop=0.15,
                    temperature=0.2,
                    alpha_recon=0.85,
                    train_idx=part.train_idx,
                    val_idx=part.val_idx,
                    labels=labels,
                )
            except Exception as e:
                print(f"FAIL conad {dataset} seed={seed}: {e}", flush=True)
                continue
            elapsed = time.perf_counter() - t0
            m = compute_metrics(scores, labels, part.cal_idx, part.test_idx, [0.05])
            ba = m["by_alpha"][0.05]
            row = {
                "study": "backbones",
                "dataset": dataset,
                "seed": seed,
                "backbone": "conad",
                "raw_auc_all": m["raw_auc_all"],
                "conformal_auc": m["conformal_auc"],
                "fpr@0.05": ba["fpr"],
                "tpr@0.05": ba["tpr"],
                "flagged@0.05": ba["flagged_rate"],
                "runtime_s": elapsed,
            }
            rows.append(row)
            print(
                f"conad {dataset} seed={seed} "
                f"AUC={m['raw_auc_all']:.3f} FPR={ba['fpr']:.3f} "
                f"TPR={ba['tpr']:.3f} time={elapsed:.1f}s",
                flush=True,
            )

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
