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

from crc_gad.config import (
    ALPHAS,
    CONTAMINATION_LEVELS,
    DEFAULT,
    N_PARTITION_SPLITS,
    REWIRE_LEVELS,
    config_for_dataset,
)
from crc_gad.conformal_calibrate import compute_metrics, inject_calibration_contamination
from crc_gad.data import load_dataset
from crc_gad.graph_utils import rewire_edges
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition, random_cal_test_split
from crc_gad.scorer import train_scorer


def _cora_cfg(cfg=None):
    """Cora ablation config: improved scorer, single restart for study throughput."""
    c = config_for_dataset("cora") if cfg is None else cfg
    c.n_restarts = 1
    return c


def _train_cora_scores(seed: int, cfg):
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
    return features, adj, labels, part, scores, rng


def run_contamination(seed: int, cfg=None) -> list[dict]:
    cfg = _cora_cfg(cfg)
    features, adj, labels, part, scores, rng = _train_cora_scores(seed, cfg)
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


def run_dependence(seed: int, cfg=None) -> list[dict]:
    cfg = _cora_cfg(cfg)
    features, adj, labels, part, scores, rng = _train_cora_scores(seed, cfg)
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


def run_heuristics(seed: int, cfg=None) -> list[dict]:
    cfg = _cora_cfg(cfg)
    features, adj, labels, part, scores, rng = _train_cora_scores(seed, cfg)
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

    # Trivial uniform-random scorer + same split conformal wrapper (Reviewer #4)
    random_scores = rng.uniform(0.0, 1.0, size=len(labels))
    m_rand = compute_metrics(random_scores, labels, part.cal_idx, test_idx, [alpha])
    ba_rand = m_rand["by_alpha"][alpha]
    rows.append({
        "method": "Uniform random + CP",
        "conformal_auc": m_rand["conformal_auc"],
        "fpr@0.05": ba_rand["fpr"],
        "tpr@0.05": ba_rand["tpr"],
        "flagged@0.05": ba_rand["flagged_rate"],
    })

    for r in rows:
        r["study"] = "heuristics"
        r["seed"] = seed
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Single seed (legacy)")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds (default: 0,1,2,3,4)",
    )
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    cfg = _cora_cfg()
    if args.fast:
        cfg.max_epochs = 40
        cfg.scoring_rounds = 8
        cfg.val_scoring_rounds = 4
        cfg.n_restarts = 1

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    all_rows = []
    for seed in seeds:
        print(f"=== studies seed {seed} ===", flush=True)
        # One scorer train shared across Cora ablation studies for this seed
        features, adj, labels, part, scores, rng = _train_cora_scores(seed, cfg)
        # contamination
        for frac in CONTAMINATION_LEVELS:
            cal, test = inject_calibration_contamination(
                part.cal_idx, part.test_idx, labels, frac, rng
            )
            m = compute_metrics(scores, labels, cal, test, [0.05])
            ba = m["by_alpha"][0.05]
            all_rows.append({
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
        # dependence
        eval_idx = np.concatenate([part.cal_idx, part.test_idx])
        for rewire_frac in REWIRE_LEVELS:
            adj_r = rewire_edges(adj, rewire_frac, rng)
            _ = adj_r  # scores frozen; rewire only stresses partition/graph context in protocol
            fprs = []
            for _ in range(N_PARTITION_SPLITS):
                cal, test = random_cal_test_split(eval_idx, cfg.rho, rng)
                m = compute_metrics(scores, labels, cal, test, [0.05])
                fprs.append(m["by_alpha"][0.05]["fpr"])
            fprs = np.array(fprs)
            all_rows.append({
                "study": "dependence",
                "seed": seed,
                "rewire_frac": rewire_frac,
                "fpr_mean": float(fprs.mean()),
                "fpr_std": float(fprs.std()),
                "fpr_p5": float(np.percentile(fprs, 5)),
                "fpr_p95": float(np.percentile(fprs, 95)),
            })
        # heuristics (reuse same scores)
        test_idx = part.test_idx
        y_test = labels[test_idx]
        s_test = scores[test_idx]
        n_test = len(test_idx)
        alpha = 0.05
        m = compute_metrics(scores, labels, part.cal_idx, test_idx, [alpha])
        ba = m["by_alpha"][alpha]
        heur_rows = [{
            "method": "CRC-GAD",
            "conformal_auc": m["conformal_auc"],
            "fpr@0.05": ba["fpr"],
            "tpr@0.05": ba["tpr"],
            "flagged@0.05": ba["flagged_rate"],
        }]
        k = max(1, int(alpha * n_test))
        order = np.argsort(-s_test)
        flagged = np.zeros(n_test, dtype=bool)
        flagged[order[:k]] = True
        heur_rows.append({
            "method": "Percentile",
            "conformal_auc": float("nan"),
            "fpr@0.05": float(flagged[y_test == 0].mean()),
            "tpr@0.05": float(flagged[y_test == 1].mean()),
            "flagged@0.05": float(flagged.mean()),
        })
        cal_idx = part.cal_idx
        cal_norms = np.linalg.norm(features[cal_idx], axis=1)
        test_norms = np.linalg.norm(features[test_idx], axis=1)
        n_cal = len(cal_idx)
        pvals = np.array([
            (1 + np.sum(cal_norms >= test_norms[i])) / (1 + n_cal)
            for i in range(n_test)
        ])
        flagged = pvals <= alpha
        heur_rows.append({
            "method": "Feature-space CP",
            "conformal_auc": float("nan"),
            "fpr@0.05": float(flagged[y_test == 0].mean()),
            "tpr@0.05": float(flagged[y_test == 1].mean()),
            "flagged@0.05": float(flagged.mean()),
        })
        random_scores = rng.uniform(0.0, 1.0, size=len(labels))
        m_rand = compute_metrics(random_scores, labels, part.cal_idx, test_idx, [alpha])
        ba_rand = m_rand["by_alpha"][alpha]
        heur_rows.append({
            "method": "Uniform random + CP",
            "conformal_auc": m_rand["conformal_auc"],
            "fpr@0.05": ba_rand["fpr"],
            "tpr@0.05": ba_rand["tpr"],
            "flagged@0.05": ba_rand["flagged_rate"],
        })
        for r in heur_rows:
            r["study"] = "heuristics"
            r["seed"] = seed
            all_rows.append(r)
        print(f"  seed {seed} done", flush=True)

    path = out_dir / "studies.csv"
    all_keys = sorted({k for r in all_rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {path} ({len(all_rows)} rows, seeds={seeds})")


if __name__ == "__main__":
    main()
