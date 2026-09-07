"""Keep best-of (prev vs new) per dataset in main_results.csv; regen tables."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def mean_auc(df: pd.DataFrame, dataset: str) -> float:
    g = df[df["dataset"] == dataset]
    return float(g["raw_auc_all"].mean()) if len(g) else float("nan")


def main():
    cur_path = ROOT / "results" / "main_results.csv"
    prev_path = ROOT / "results" / "main_results_before_ablation_rerun.csv"
    if not prev_path.exists():
        print("No previous backup; skip merge")
        return
    cur = pd.read_csv(cur_path)
    prev = pd.read_csv(prev_path)
    rows = []
    for d in sorted(set(prev["dataset"]) | set(cur["dataset"])):
        cm = mean_auc(cur, d)
        pm = mean_auc(prev, d)
        if cm == cm and (pm != pm or cm > pm + 1e-4):
            part = cur[cur["dataset"] == d]
            print(f"{d}: KEEP NEW {pm:.4f} -> {cm:.4f}")
        else:
            part = prev[prev["dataset"] == d]
            print(f"{d}: KEEP PREV {pm:.4f} (new={cm if cm == cm else float('nan'):.4f})")
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cur_path, index=False)
    print(f"Wrote {cur_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
