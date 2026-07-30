#!/usr/bin/env python3
"""Generate paper/generated/main_results.tex from results/main_results.csv."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "main_results.csv"
OUT = ROOT / "paper" / "generated" / "main_results.tex"
DATASETS = ["cora", "citeseer", "pubmed", "acm", "blogcatalog", "flickr"]


def ms(rows, key):
    vals = [float(r[key]) for r in rows]
    return np.mean(vals), np.std(vals)


def cell(rows, key):
    m, s = ms(rows, key)
    return f"{m:.3f}$\\pm${s:.3f}"


def main():
    if not RESULTS.exists():
        print("Missing", RESULTS)
        return
    rows = list(csv.DictReader(open(RESULTS, encoding="utf-8")))
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    header = " & ".join(["\\textbf{" + d.title() + "}" for d in DATASETS])
    lines = [
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(DATASETS) + "}",
        "\\toprule",
        f"\\textbf{{Metric}} & {header} \\\\",
        "\\midrule",
    ]

    def row(label, key):
        cells = [cell(by_ds[d], key) if d in by_ds else "---" for d in DATASETS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    row("CoLA backbone AUC (raw)", "raw_auc_all")
    row("CRC-GAD conformal AUC", "conformal_auc")
    row("FPR @ $\\alpha{=}0.05$", "fpr@0.05")
    row("TPR @ $\\alpha{=}0.05$", "tpr@0.05")
    row("Flagged rate @ $0.05$", "flagged@0.05")
    row("$|\\mathcal{C}|$ (mean)", "n_cal")
    row("Anomalies in $\\mathcal{C}$ (mean)", "anomalies_in_C")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
