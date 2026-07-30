#!/usr/bin/env python3
"""Supplementary studies: contamination, dependence (200 splits), heuristics."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crc_gad.config import ALPHAS, CONTAMINATION_LEVELS, DEFAULT, N_PARTITION_SPLITS, REWIRE_LEVELS
from crc_gad.conformal_calibrate import compute_metrics, inject_calibration_contamination
from crc_gad.data import load_dataset
from crc_gad.graph_utils import rewire_edges
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition, random_cal_test_split
from crc_gad.scorer import train_scorer


def run_contamination(seed: int, cfg=DEFAULT) -> list[dict]:
    rng = np.random.default_rng(seed)
    features, adj, _ = load_dataset("cora")
    features, adj, labels = inject_anomalies(
        features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
    )
    part = make_partition(
        len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
    )
    _, scores = train_scorer(
        features, adj, part.train_idx, part.val_idx, labels, cfg, rng
    )
    rows = []
    for frac in CONTAMINATION_LEVELS:
        cal, test = inject_calibration_contamination(
            part.cal_idx, part.test_idx, labels, frac, rng
        )
        m = compute_metrics(scores, labels, cal, test, [0.05])
        ba = m["by_alpha"][0.05]
        rows.append({
            "study": "contamination",
            "seed": seed,
            "contamination_frac": frac,
            "n_cal": len(cal),
            "n_test": len(test),
            "anomalies_in_C": int(labels[cal].sum()),
            "conformal_auc": m["conformal_auc"],
            "fpr@0.05": ba["fpr"],
            "flagged@0.05": ba["flagged_rate"],
        })
    return rows


def run_dependence(seed: int, cfg=DEFAULT) -> list[dict]:
    rng = np.random.default_rng(seed)
    features, adj, _ = load_dataset("cora")
    features, adj, labels = inject_anomalies(
        features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
    )
    part = make_partition(
        len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
    )
    _, scores = train_scorer(
        features, adj, part.train_idx, part.val_idx, labels, cfg, rng
    )
    eval_idx = np.concatenate([part.cal_idx, part.test_idx])
    rows = []
    for rewire_frac in REWIRE_LEVELS:
        adj_r = rewire_edges(adj, rewire_frac, rng)
        fprs = []
        for _ in range(N_PARTITION_SPLITS):
            cal, test = random_cal_test_split(eval_idx, cfg.rho, rng)
            m = compute_metrics(scores, labels, cal, test, [0.05])
            fprs.append(m["by_alpha"][0.05]["fpr"])
        fprs = np.array(fprs)
        rows.append({
            "study": "dependence",
            "seed": seed,
            "rewire_frac": rewire_frac,
            "fpr_mean": float(fprs.mean()),
            "fpr_std": float(fprs.std()),
            "fpr_p5": float(np.percentile(fprs, 5)),
            "fpr_p95": float(np.percentile(fprs, 95)),
        })
    return rows


def run_heuristics(seed: int, cfg=DEFAULT) -> list[dict]:
    rng = np.random.default_rng(seed)
    features, adj, _ = load_dataset("cora")
    features, adj, labels = inject_anomalies(
        features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
    )
    part = make_partition(
        len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
    )
    _, scores = train_scorer(
        features, adj, part.train_idx, part.val_idx, labels, cfg, rng
    )
    test_idx = part.test_idx
    y_test = labels[test_idx]
    s_test = scores[test_idx]
    n_test = len(test_idx)
    alpha = 0.05
    rows = []

    # CRC-GAD conformal
    m = compute_metrics(scores, labels, part.cal_idx, test_idx, [alpha])
    ba = m["by_alpha"][alpha]
    rows.append({
        "method": "CRC-GAD",
        "conformal_auc": m["conformal_auc"],
        "fpr@0.05": ba["fpr"],
        "tpr@0.05": ba["tpr"],
        "flagged@0.05": ba["flagged_rate"],
    })

    # Percentile threshold: flag exactly alpha * |T| nodes
    k = max(1, int(alpha * n_test))
    order = np.argsort(-s_test)
    flagged = np.zeros(n_test, dtype=bool)
    flagged[order[:k]] = True
    rows.append({
        "method": "Percentile",
        "conformal_auc": float("nan"),
        "fpr@0.05": float(flagged[y_test == 0].mean()),
        "tpr@0.05": float(flagged[y_test == 1].mean()),
        "flagged@0.05": float(flagged.mean()),
    })

    # Feature-space split conformal (Bates-style, i.i.d. on features)
    cal_idx = part.cal_idx
    cal_feat = features[cal_idx]
    test_feat = features[test_idx]
    cal_norms = np.linalg.norm(cal_feat, axis=1)
    test_norms = np.linalg.norm(test_feat, axis=1)
    n_cal = len(cal_idx)
    pvals = np.array([
        (1 + np.sum(cal_norms >= test_norms[i])) / (1 + n_cal)
        for i in range(n_test)
    ])
    flagged = pvals <= alpha
    rows.append({
        "method": "Feature-space CP",
        "conformal_auc": float("nan"),
        "fpr@0.05": float(flagged[y_test == 0].mean()),
        "tpr@0.05": float(flagged[y_test == 1].mean()),
        "flagged@0.05": float(flagged.mean()),
    })

    for r in rows:
        r["study"] = "heuristics"
        r["seed"] = seed
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    cfg = DEFAULT
    if args.fast:
        cfg.max_epochs = 40
        cfg.scoring_rounds = 8
        cfg.val_scoring_rounds = 4

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    all_rows = []
    all_rows.extend(run_contamination(args.seed, cfg))
    all_rows.extend(run_dependence(args.seed, cfg))
    all_rows.extend(run_heuristics(args.seed, cfg))

    path = out_dir / "studies.csv"
    all_keys = sorted({k for r in all_rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
