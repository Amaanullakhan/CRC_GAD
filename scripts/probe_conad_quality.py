#!/usr/bin/env python3
"""Probe CONAD quality before adding to the paper."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crc_gad.config import DEFAULT
from crc_gad.conformal_calibrate import compute_metrics
from crc_gad.conad_torch import score_conad_pytorch
from crc_gad.data import load_dataset
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition


def main():
    cfg = DEFAULT
    datasets = sys.argv[1:] or ["cora"]
    seeds = [0, 1, 2]
    print("quality bar: keep CONAD if mean AUC>=0.60 and mean TPR>0.05 on Cora")
    for dataset in datasets:
        aucs, fprs, tprs = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            features, adj, _ = load_dataset(dataset)
            features, adj, labels = inject_anomalies(
                features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
            )
            part = make_partition(
                len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
            )
            scores = score_conad_pytorch(
                features,
                adj,
                rng,
                epochs=120,
                hidden_dim=64,
                attr_noise=0.1,
                edge_drop=0.15,
                temperature=0.2,
                alpha_recon=0.85,
            )
            m = compute_metrics(scores, labels, part.cal_idx, part.test_idx, [0.05])
            ba = m["by_alpha"][0.05]
            aucs.append(m["raw_auc_all"])
            fprs.append(ba["fpr"])
            tprs.append(ba["tpr"])
            print(
                f"{dataset} seed={seed} AUC={m['raw_auc_all']:.3f} "
                f"FPR={ba['fpr']:.3f} TPR={ba['tpr']:.3f}"
            )
        print(
            f"SUMMARY {dataset}: AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f} "
            f"FPR={np.mean(fprs):.3f} TPR={np.mean(tprs):.3f}"
        )
        keep = np.mean(aucs) >= 0.60 and np.mean(tprs) > 0.05
        print(f"KEEP_CONAD_{dataset}={keep}")


if __name__ == "__main__":
    main()
