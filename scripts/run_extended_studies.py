#!/usr/bin/env python3
"""Extended studies: multi-backbone, multi-dataset heuristics, degree FPR, runtime, organic."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crc_gad.backbones import BACKBONES, get_scores
from crc_gad.config import CONTAMINATION_LEVELS, DEFAULT
from crc_gad.conformal_calibrate import compute_metrics, inject_calibration_contamination
from crc_gad.data import load_dataset
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition

STUDY_DATASETS = ["cora", "citeseer", "pubmed"]
BACKBONE_NAMES = ["cola", "dominant_style", "degree", "feature_norm", "attr_deviation"]


def _prep_injected(dataset: str, seed: int, cfg):
    rng = np.random.default_rng(seed)
    features, adj, _ = load_dataset(dataset)
    features, adj, labels = inject_anomalies(
        features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
    )
    part = make_partition(
        len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
    )
    return features, adj, labels, part, rng


def run_backbones(seeds, datasets, cfg) -> list[dict]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            features, adj, labels, part, rng = _prep_injected(dataset, seed, cfg)
            for name in BACKBONE_NAMES:
                t0 = time.perf_counter()
                try:
                    scores = get_scores(
                        name,
                        features,
                        adj,
                        rng,
                        train_idx=part.train_idx,
                        val_idx=part.val_idx,
                        labels=labels,
                        cfg=cfg,
                        epochs=40 if cfg.max_epochs <= 40 else 80,
                    )
                except Exception as e:
                    print(f"FAIL backbone {name} {dataset} seed={seed}: {e}", flush=True)
                    continue
                elapsed = time.perf_counter() - t0
                m = compute_metrics(scores, labels, part.cal_idx, part.test_idx, [0.05])
                ba = m["by_alpha"][0.05]
                rows.append({
                    "study": "backbones",
                    "dataset": dataset,
                    "seed": seed,
                    "backbone": name,
                    "raw_auc_all": m["raw_auc_all"],
                    "conformal_auc": m["conformal_auc"],
                    "fpr@0.05": ba["fpr"],
                    "tpr@0.05": ba["tpr"],
                    "flagged@0.05": ba["flagged_rate"],
                    "runtime_s": elapsed,
                })
                print(f"backbone {name} {dataset} seed={seed} FPR={ba['fpr']:.3f} AUC={m['raw_auc_all']:.3f}", flush=True)
    return rows


def run_heuristics_multi(seeds, datasets, cfg) -> list[dict]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            features, adj, labels, part, rng = _prep_injected(dataset, seed, cfg)
            scores = get_scores(
                "cola", features, adj, rng,
                train_idx=part.train_idx, val_idx=part.val_idx, labels=labels, cfg=cfg,
            )
            test_idx = part.test_idx
            y_test = labels[test_idx]
            s_test = scores[test_idx]
            n_test = len(test_idx)
            alpha = 0.05

            # CRC-GAD
            m = compute_metrics(scores, labels, part.cal_idx, test_idx, [alpha])
            ba = m["by_alpha"][alpha]
            rows.append({
                "study": "heuristics", "dataset": dataset, "seed": seed, "method": "CRC-GAD",
                "fpr@0.05": ba["fpr"], "tpr@0.05": ba["tpr"], "flagged@0.05": ba["flagged_rate"],
                "conformal_auc": m["conformal_auc"],
            })

            # Percentile
            k = max(1, int(alpha * n_test))
            flagged = np.zeros(n_test, dtype=bool)
            flagged[np.argsort(-s_test)[:k]] = True
            rows.append({
                "study": "heuristics", "dataset": dataset, "seed": seed, "method": "Percentile",
                "fpr@0.05": float(flagged[y_test == 0].mean()),
                "tpr@0.05": float(flagged[y_test == 1].mean()) if (y_test == 1).any() else float("nan"),
                "flagged@0.05": float(flagged.mean()),
                "conformal_auc": float("nan"),
            })

            # Feature-space CP
            cal_norms = np.linalg.norm(features[part.cal_idx], axis=1)
            test_norms = np.linalg.norm(features[test_idx], axis=1)
            n_cal = len(part.cal_idx)
            pvals = np.array([(1 + np.sum(cal_norms >= test_norms[i])) / (1 + n_cal) for i in range(n_test)])
            flagged = pvals <= alpha
            rows.append({
                "study": "heuristics", "dataset": dataset, "seed": seed, "method": "Feature-space CP",
                "fpr@0.05": float(flagged[y_test == 0].mean()),
                "tpr@0.05": float(flagged[y_test == 1].mean()) if (y_test == 1).any() else float("nan"),
                "flagged@0.05": float(flagged.mean()),
                "conformal_auc": float("nan"),
            })

            # Uniform random + CP
            m_rand = compute_metrics(rng.uniform(0, 1, size=len(labels)), labels, part.cal_idx, test_idx, [alpha])
            ba_r = m_rand["by_alpha"][alpha]
            rows.append({
                "study": "heuristics", "dataset": dataset, "seed": seed, "method": "Uniform random + CP",
                "fpr@0.05": ba_r["fpr"], "tpr@0.05": ba_r["tpr"], "flagged@0.05": ba_r["flagged_rate"],
                "conformal_auc": m_rand["conformal_auc"],
            })
    return rows


def run_degree_fpr(seeds, cfg) -> list[dict]:
    """Stratify normal-node FPR by degree quartile on Cora."""
    rows = []
    for seed in seeds:
        features, adj, labels, part, rng = _prep_injected("cora", seed, cfg)
        scores = get_scores(
            "cola", features, adj, rng,
            train_idx=part.train_idx, val_idx=part.val_idx, labels=labels, cfg=cfg,
        )
        from crc_gad.conformal_calibrate import conformal_pvalues
        pvals = conformal_pvalues(scores, part.cal_idx, part.test_idx)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        test = part.test_idx
        y = labels[test]
        d_test = deg[test]
        normals = y == 0
        qs = np.quantile(d_test[normals], [0.25, 0.5, 0.75]) if normals.any() else [0, 0, 0]
        bins = [
            ("Q1_low", d_test <= qs[0]),
            ("Q2", (d_test > qs[0]) & (d_test <= qs[1])),
            ("Q3", (d_test > qs[1]) & (d_test <= qs[2])),
            ("Q4_high", d_test > qs[2]),
        ]
        flagged = pvals <= 0.05
        for name, mask in bins:
            m = mask & normals
            fpr = float(flagged[m].mean()) if m.any() else float("nan")
            rows.append({
                "study": "degree_fpr", "dataset": "cora", "seed": seed,
                "degree_bin": name, "fpr@0.05": fpr, "n_normals": int(m.sum()),
            })
    return rows


def run_contamination_multi(seeds, datasets, cfg) -> list[dict]:
    levels = list(CONTAMINATION_LEVELS) + [0.25, 0.30]
    rows = []
    for dataset in datasets:
        for seed in seeds:
            features, adj, labels, part, rng = _prep_injected(dataset, seed, cfg)
            scores = get_scores(
                "cola", features, adj, rng,
                train_idx=part.train_idx, val_idx=part.val_idx, labels=labels, cfg=cfg,
            )
            for frac in levels:
                cal, test = inject_calibration_contamination(
                    part.cal_idx, part.test_idx, labels, frac, rng
                )
                m = compute_metrics(scores, labels, cal, test, [0.05])
                ba = m["by_alpha"][0.05]
                rows.append({
                    "study": "contamination", "dataset": dataset, "seed": seed,
                    "contamination_frac": frac, "n_cal": len(cal), "n_test": len(test),
                    "anomalies_in_C": int(labels[cal].sum()),
                    "conformal_auc": m["conformal_auc"],
                    "fpr@0.05": ba["fpr"], "flagged@0.05": ba["flagged_rate"],
                })
    return rows


def run_organic(seeds, cfg, datasets=None) -> list[dict]:
    """Organic (non-injected) labels: rare-class Planetoid proxies + optional YelpChi fraud."""
    if datasets is None:
        datasets = ["organic_cora", "organic_citeseer", "yelpchi"]
    rows = []
    for dataset in datasets:
        try:
            features0, adj0, labels0 = load_dataset(dataset)
        except Exception as e:
            print(f"SKIP organic dataset {dataset}: {e}", flush=True)
            continue
        # Large fraud graphs: heuristics only (full CoLA/DOMINANT-style is too heavy on CPU)
        if dataset in ("yelpchi", "yelp", "yelp_chi") or features0.shape[0] > 15000:
            organic_backbones = ["degree", "feature_norm", "attr_deviation"]
        else:
            organic_backbones = ["cola", "dominant_style", "attr_deviation", "degree"]
        print(
            f"organic dataset={dataset} n={features0.shape[0]} "
            f"anom_rate={labels0.mean():.3f} backbones={organic_backbones}",
            flush=True,
        )
        for seed in seeds:
            rng = np.random.default_rng(seed)
            features, adj, labels = features0, adj0, labels0
            part = make_partition(
                len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain, cfg.rho, rng
            )
            for name in organic_backbones:
                t0 = time.perf_counter()
                try:
                    scores = get_scores(
                        name, features, adj, rng,
                        train_idx=part.train_idx, val_idx=part.val_idx, labels=labels, cfg=cfg,
                        epochs=40 if cfg.max_epochs <= 40 else 80,
                    )
                except Exception as e:
                    print(f"FAIL organic {name} {dataset} seed={seed}: {e}", flush=True)
                    continue
                elapsed = time.perf_counter() - t0
                m = compute_metrics(scores, labels, part.cal_idx, part.test_idx, [0.05])
                ba = m["by_alpha"][0.05]
                rows.append({
                    "study": "organic", "dataset": dataset, "seed": seed,
                    "backbone": name, "anomaly_rate": float(labels.mean()),
                    "raw_auc_all": m["raw_auc_all"], "conformal_auc": m["conformal_auc"],
                    "fpr@0.05": ba["fpr"], "tpr@0.05": ba["tpr"],
                    "flagged@0.05": ba["flagged_rate"], "runtime_s": elapsed,
                })
                print(
                    f"organic {dataset} {name} seed={seed} "
                    f"AUC={m['raw_auc_all']:.3f} FPR={ba['fpr']:.3f}",
                    flush=True,
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--datasets", type=str, default="cora,citeseer,pubmed")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="backbones,heuristics,degree_fpr,contamination,organic",
        help="Comma-separated study groups",
    )
    args = parser.parse_args()
    cfg = DEFAULT
    if args.fast:
        cfg.max_epochs = 40
        cfg.scoring_rounds = 8
        cfg.val_scoring_rounds = 4

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    only = {x.strip() for x in args.only.split(",")}

    all_rows: list[dict] = []
    if "backbones" in only:
        all_rows.extend(run_backbones(seeds, datasets, cfg))
    if "heuristics" in only:
        all_rows.extend(run_heuristics_multi(seeds, datasets, cfg))
    if "degree_fpr" in only:
        all_rows.extend(run_degree_fpr(seeds, cfg))
    if "contamination" in only:
        all_rows.extend(run_contamination_multi(seeds, datasets, cfg))
    if "organic" in only:
        all_rows.extend(run_organic(seeds, cfg))

    # Avoid wiping extended_studies.csv when refreshing only organic rows
    if only == {"organic"}:
        out = ROOT / "results" / "organic_studies.csv"
    else:
        out = ROOT / "results" / "extended_studies.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in all_rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
