#!/usr/bin/env python3
"""Audit calibration set C construction — logs |C|, |T|, anomaly counts per seed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crc_gad.config import DEFAULT, SEEDS
from crc_gad.data import load_dataset
from crc_gad.inject_anomaly import inject_anomalies
from crc_gad.partition import make_partition

AUDIT_DOC = ROOT / "docs" / "CALIBRATION_SET_AUDIT.md"


def main():
    cfg = DEFAULT
    lines = [
        "# Calibration Set C — Audit Log",
        "",
        "**Protocol:** Label-free random split of eval nodes. Labels are NEVER used to construct C.",
        "",
        f"- `use_label_clean_calibration = {cfg.use_label_clean_calibration}`",
        f"- `rho = {cfg.rho}` (|C| / |V_eval|)",
        "",
        "| Dataset | Seed | |C| | |T| | Anom in C | Anom in T | pi_C | pi_T |",
        "|---------|------|-----|-----|-----------|-----------|------|------|",
    ]
    audits = []
    for dataset in ["cora", "citeseer", "pubmed", "acm", "blogcatalog", "flickr"]:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            try:
                features, adj, _ = load_dataset(dataset)
            except Exception as e:
                lines.append(f"| {dataset} | {seed} | ERROR: {e} | | | | | |")
                continue
            _, _, labels = inject_anomalies(
                features, adj, cfg.anomaly_ratio, cfg.structural_ratio, rng
            )
            part = make_partition(
                len(labels), labels, cfg.train_frac, cfg.val_frac_of_remain,
                cfg.rho, rng, cfg.use_label_clean_calibration,
            )
            a = part.audit_log()
            a["dataset"] = dataset
            a["seed"] = seed
            audits.append(a)
            lines.append(
                f"| {dataset} | {seed} | {a['|C|']} | {a['|T|']} | "
                f"{a['anomalies_in_C']} | {a['anomalies_in_T']} | "
                f"{a['pi_C']:.4f} | {a['pi_T']:.4f} |"
            )

    lines.extend([
        "",
        "## Code path",
        "",
        "1. `partition.make_partition()` shuffles all node indices without labels/scores",
        "2. Train/val/eval split by configured fractions",
        "3. Eval nodes split: first `rho` fraction → C, remainder → T",
        "4. Anomalies are **not** excluded unless `use_label_clean_calibration=True`",
        "",
        "See `src/crc_gad/partition.py` lines 45–75.",
    ])

    AUDIT_DOC.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DOC.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "results" / "partition_audit.json").write_text(
        json.dumps(audits, indent=2), encoding="utf-8"
    )
    print(f"Wrote {AUDIT_DOC}")


if __name__ == "__main__":
    main()
