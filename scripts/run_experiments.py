#!/usr/bin/env python3
"""Run core CRC-GAD experiments: seeds {0..4} × datasets → results/*.csv"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crc_gad.config import ALPHAS, DATASETS, DEFAULT, SEEDS, config_for_dataset
from crc_gad.conformal_calibrate import compute_metrics
from crc_gad.data import load_dataset
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition
from crc_gad.scorer import train_scorer


def run_single(dataset: str, seed: int, cfg=None) -> dict:
    if cfg is None:
        cfg = config_for_dataset(dataset)
    rng = np.random.default_rng(seed)
    features, adj, _ = load_dataset(dataset)
    features, adj, labels = inject_anomalies(
        features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
    )
    part = make_partition(
        len(labels),
        labels,
        cfg.train_frac,
        cfg.val_frac_of_remain,
        cfg.rho,
        rng,
        use_label_clean_calibration=cfg.use_label_clean_calibration,
    )
    audit = part.audit_log()
    _, scores = train_scorer(
        features, adj, part.train_idx, part.val_idx, labels, cfg, rng
    )
    metrics = compute_metrics(
        scores, labels, part.cal_idx, part.test_idx, ALPHAS
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "audit": audit,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--fast", action="store_true", help="Reduced epochs/rounds")
    args = parser.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    rows = []
    csv_path = out_dir / "main_results.csv"

    # Drop stale rows for datasets we are re-running
    existing: list[dict] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        keep = [r for r in existing if r["dataset"] not in args.datasets]
        if len(keep) != len(existing):
            existing = keep
            if keep:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
                    w.writeheader()
                    w.writerows(keep)
            else:
                csv_path.unlink(missing_ok=True)

    for dataset in args.datasets:
        cfg = config_for_dataset(dataset)
        if args.fast:
            cfg.max_epochs = 40
            cfg.scoring_rounds = 8
            cfg.val_scoring_rounds = 4
        for seed in args.seeds:
            print(f"Running {dataset} seed={seed}...", flush=True)
            try:
                res = run_single(dataset, seed, cfg)
            except Exception as e:
                print(f"FAILED {dataset} seed={seed}: {e}", flush=True)
                continue
            audit = res["audit"]
            m = res["metrics"]
            row = {
                "dataset": dataset,
                "seed": seed,
                "raw_auc_all": m["raw_auc_all"],
                "raw_auc_test": m["raw_auc_test"],
                "conformal_auc": m["conformal_auc"],
                "n_cal": audit["|C|"],
                "n_test": audit["|T|"],
                "anomalies_in_C": audit["anomalies_in_C"],
                "anomalies_in_T": audit["anomalies_in_T"],
                "pi_test": audit["pi_T"],
            }
            for alpha in ALPHAS:
                ba = m["by_alpha"][alpha]
                row[f"fpr@{alpha}"] = ba["fpr"]
                row[f"tpr@{alpha}"] = ba["tpr"]
                row[f"flagged@{alpha}"] = ba["flagged_rate"]
                row[f"theory_max_flagged@{alpha}"] = ba["theory_max_flagged"]
                row[f"identity_gap@{alpha}"] = ba["identity_gap"]
            rows.append(row)
            # Incremental save
            fieldnames = list(row.keys())
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with open(csv_path, "a" if csv_path.exists() else "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow(row)

    if not rows:
        print("No results")
        return

    # Re-read full csv, dedupe dataset+seed (keep last run)
    all_rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    dedup: dict[tuple[str, str], dict] = {}
    for r in all_rows:
        dedup[(r["dataset"], r["seed"])] = r
    all_rows = list(dedup.values())
    all_rows.sort(key=lambda r: (r["dataset"], int(r["seed"])))
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    rows = all_rows

    summary = {}
    for dataset in args.datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        summary[dataset] = {
            "raw_auc_all_mean": float(np.mean([float(r["raw_auc_all"]) for r in ds_rows])),
            "conformal_auc_mean": float(np.mean([float(r["conformal_auc"]) for r in ds_rows])),
            "fpr@0.05_mean": float(np.mean([float(r["fpr@0.05"]) for r in ds_rows])),
            "tpr@0.05_mean": float(np.mean([float(r["tpr@0.05"]) for r in ds_rows])),
            "flagged@0.05_mean": float(np.mean([float(r["flagged@0.05"]) for r in ds_rows])),
            "theory_max@0.05": float(ds_rows[0].get("theory_max_flagged@0.05", 0)),
        }

    with open(out_dir / "main_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path} ({len(rows)} rows)")
    if "cora" in summary:
        s = summary["cora"]
        print(
            f"Cora flagged@0.05={s['flagged@0.05_mean']:.4f} "
            f"theory_max={s['theory_max@0.05']:.4f} "
            f"FPR={s['fpr@0.05_mean']:.4f} TPR={s['tpr@0.05_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
