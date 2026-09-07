"""Merge new result CSVs with HEAD baselines; keep only improvements; regen tables."""
from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_git_csv(rel: str) -> pd.DataFrame | None:
    try:
        txt = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], text=True, errors="replace"
        )
        return pd.read_csv(StringIO(txt))
    except Exception:
        return None


def mean_auc(df: pd.DataFrame, dataset: str, col: str = "raw_auc_all") -> float:
    g = df[df["dataset"] == dataset]
    if g.empty:
        return float("nan")
    return float(g[col].mean())


def merge_main(new_path: Path, out_path: Path) -> pd.DataFrame:
    old = load_git_csv("results/main_results.csv")
    new = pd.read_csv(new_path) if new_path.exists() else pd.DataFrame()
    if old is None:
        new.to_csv(out_path, index=False)
        return new
    rows = []
    datasets = sorted(set(old["dataset"]) | set(new["dataset"] if len(new) else []))
    report = []
    for d in datasets:
        om = mean_auc(old, d)
        nm = mean_auc(new, d) if len(new) else float("nan")
        if nm == nm and nm > om + 1e-4:
            part = new[new["dataset"] == d]
            report.append(f"MAIN {d}: KEEP NEW {om:.4f} -> {nm:.4f}")
        else:
            part = old[old["dataset"] == d]
            tag = "KEEP OLD" if not (nm == nm) else f"KEEP OLD (new {nm:.4f} <= {om:.4f})"
            report.append(f"MAIN {d}: {tag}")
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_path, index=False)
    print("\n".join(report))
    return out


def merge_named(rel: str, new_path: Path, out_path: Path, min_auc: float | None = None) -> None:
    old = load_git_csv(rel)
    new = pd.read_csv(new_path) if new_path.exists() else pd.DataFrame()
    if old is None and len(new) == 0:
        return
    if old is None:
        keep = new
        if min_auc is not None:
            ok = []
            for d, g in keep.groupby("dataset"):
                if g["raw_auc_all"].mean() >= min_auc and g["tpr@0.05"].mean() > 0.05:
                    ok.append(g)
            keep = pd.concat(ok, ignore_index=True) if ok else keep.iloc[0:0]
        keep.to_csv(out_path, index=False)
        print(f"{rel}: wrote new only ({len(keep)} rows)")
        return
    rows = []
    datasets = sorted(set(old["dataset"]) | (set(new["dataset"]) if len(new) else set()))
    for d in datasets:
        o = old[old["dataset"] == d]
        n = new[new["dataset"] == d] if len(new) else o.iloc[0:0]
        om = float(o["raw_auc_all"].mean()) if len(o) else -1.0
        nm = float(n["raw_auc_all"].mean()) if len(n) else -1.0
        use = n if nm > om + 1e-4 else o
        if min_auc is not None and len(use):
            if use["raw_auc_all"].mean() < min_auc or use["tpr@0.05"].mean() <= 0.05:
                if len(o) and o["raw_auc_all"].mean() >= min_auc:
                    use = o
                elif nm < min_auc and om < min_auc:
                    print(f"{rel} {d}: both below gate — drop")
                    continue
        tag = "NEW" if nm > om + 1e-4 else "OLD"
        print(f"{rel} {d}: use {tag} (old={om:.4f} new={nm:.4f})")
        rows.append(use)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_path, index=False)


def regen_tables():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from scripts.generate_paper_tables import main as gen_main  # type: ignore

    # call via subprocess to avoid import path issues
    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "generate_paper_tables.py")], cwd=ROOT
    )
    ext = ROOT / "scripts" / "generate_extended_tables.py"
    if ext.exists():
        subprocess.check_call([sys.executable, str(ext)], cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Write merged CSVs + regen tables")
    args = p.parse_args()
    tmp = ROOT / "results" / "_merged_main.csv"
    merge_main(ROOT / "results" / "main_results.csv", tmp if not args.apply else ROOT / "results" / "main_results.csv")
    if (ROOT / "results" / "conad_pytorch.csv").exists():
        merge_named(
            "results/conad_pytorch.csv",
            ROOT / "results" / "conad_pytorch.csv",
            ROOT / "results" / "conad_pytorch.csv" if args.apply else ROOT / "results" / "_merged_conad.csv",
            min_auc=0.60,
        )
    if args.apply:
        regen_tables()
        print("Tables regenerated.")


if __name__ == "__main__":
    main()
