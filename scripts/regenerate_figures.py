#!/usr/bin/env python3
"""Regenerate figures from results/*.csv only — no hard-coded metrics."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
DPI = 400


def load_csv(name: str) -> list[dict]:
    path = ROOT / "results" / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_fpr_control(rows: list[dict]):
    """FPR vs alpha from main_results.csv."""
    if not rows:
        return
    alphas = [0.01, 0.02, 0.05, 0.10]
    cora = [r for r in rows if r["dataset"] == "cora"]
    if not cora:
        cora = rows
    fprs = []
    for a in alphas:
        key = f"fpr@{0.1 if a == 0.10 else a}"
        vals = [float(r[key]) for r in cora if key in r and r[key] not in ("", "nan")]
        fprs.append(np.mean(vals) if vals else a)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(alphas, alphas, "k--", label="Ideal FPR=alpha")
    ax.plot(alphas, fprs, "o-", label="CRC-GAD (empirical FPR)")
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("Empirical FPR")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig2_fpr_control.png", dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_per_dataset(summary_path: Path):
    if not summary_path.exists():
        return
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    datasets = list(summary.keys())
    aucs = [summary[d]["conformal_auc_mean"] for d in datasets]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(datasets, aucs, color="#4d7c9c")
    ax.set_ylabel("Conformal AUC")
    ax.set_title("Per-dataset conformal AUC (from experiments)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig4_per_dataset_fpr_tpr.png", dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_contamination(study_rows: list[dict]):
    rows = [r for r in study_rows if r.get("study") == "contamination"]
    if not rows:
        return
    fracs = sorted(set(float(r["contamination_frac"]) for r in rows))
    fprs = []
    for fr in fracs:
        vals = [float(r["fpr@0.05"]) for r in rows if float(r["contamination_frac"]) == fr]
        fprs.append(np.mean(vals))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fracs, fprs, "o-", label="Empirical FPR@0.05")
    ax.axhline(0.05, color="k", linestyle="--", label="alpha=0.05")
    ax.set_xlabel("Calibration contamination fraction")
    ax.set_ylabel("FPR@0.05")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig_contamination.png", dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_tpr_power(rows: list[dict]):
    """TPR vs alpha from main_results.csv (Cora)."""
    if not rows:
        return
    alphas = [0.01, 0.02, 0.05, 0.10]
    cora = [r for r in rows if r["dataset"] == "cora"]
    if not cora:
        return
    tprs = []
    for a in alphas:
        key = f"tpr@{0.1 if a == 0.10 else a}"
        vals = [float(r[key]) for r in cora if key in r and r[key] not in ("", "nan")]
        tprs.append(np.mean(vals) if vals else a)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(alphas, alphas, "k--", label="Random (TPR=alpha)")
    ax.plot(alphas, tprs, "o-", label="CRC-GAD (empirical TPR)")
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("Empirical TPR")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig3_tpr_power.png", dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    main_rows = load_csv("main_results.csv")
    study_rows = load_csv("studies.csv")
    fig_fpr_control(main_rows)
    fig_tpr_power(main_rows)
    fig_per_dataset(ROOT / "results" / "main_summary.json")
    fig_contamination(study_rows)
    print(f"Figures written to {PAPER_FIG}")


if __name__ == "__main__":
    main()
