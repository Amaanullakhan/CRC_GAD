#!/usr/bin/env python3
"""Publish-readiness summary of pinned CSVs."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    p = ROOT / "results" / name
    if not p.exists():
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def mean_std(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"


def main():
    print("=== MAIN CoLA+CRC (injected) ===")
    by = defaultdict(list)
    for r in load("main_results.csv"):
        by[r["dataset"]].append(r)
    for d in sorted(by):
        sub = by[d]
        print(
            f"{d:12s} rawAUC={mean_std([r['raw_auc_all'] for r in sub])} "
            f"FPR@0.05={mean_std([r['fpr@0.05'] for r in sub])} "
            f"TPR={mean_std([r['tpr@0.05'] for r in sub])}"
        )

    print("=== DOMINANT ===")
    by = defaultdict(list)
    for r in load("dominant_pytorch.csv"):
        by[r["dataset"]].append(r)
    for d in sorted(by):
        sub = by[d]
        print(
            f"{d:12s} AUC={mean_std([r['raw_auc_all'] for r in sub])} "
            f"FPR={mean_std([r['fpr@0.05'] for r in sub])} "
            f"TPR={mean_std([r['tpr@0.05'] for r in sub])}"
        )

    print("=== CONAD ===")
    by = defaultdict(list)
    for r in load("conad_pytorch.csv"):
        by[r["dataset"]].append(r)
    for d in sorted(by):
        sub = by[d]
        print(
            f"{d:12s} AUC={mean_std([r['raw_auc_all'] for r in sub])} "
            f"FPR={mean_std([r['fpr@0.05'] for r in sub])} "
            f"TPR={mean_std([r['tpr@0.05'] for r in sub])}"
        )

    print("=== ORGANIC ===")
    by = defaultdict(list)
    for r in load("organic_studies.csv"):
        by[(r["dataset"], r["backbone"])].append(r)
    for k in sorted(by):
        sub = by[k]
        print(
            f"{k[0]:18s} {k[1]:16s} "
            f"AUC={mean_std([r['raw_auc_all'] for r in sub])} "
            f"FPR={mean_std([r['fpr@0.05'] for r in sub])}"
        )


if __name__ == "__main__":
    main()
