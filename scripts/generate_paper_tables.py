#!/usr/bin/env python3
"""Generate all paper/generated/*.tex from results CSV files."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
MAIN = ROOT / "results" / "main_results.csv"
STUDIES = ROOT / "results" / "studies.csv"
DATASETS = ["cora", "citeseer", "pubmed", "acm", "blogcatalog", "flickr"]


def load_main():
    return list(csv.DictReader(open(MAIN, encoding="utf-8")))


def load_studies():
    if not STUDIES.exists():
        return []
    return list(csv.DictReader(open(STUDIES, encoding="utf-8")))


def ms(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    if not vals:
        return "---", "---"
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.3f}$\\pm${s:.3f}", m


def write(name: str, content: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def gen_main_results(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    header = " & ".join(d.title() for d in DATASETS)
    lines = [
        f"\\begin{{tabular}}{{l{'c' * len(DATASETS)}}}",
        "\\toprule",
        f"\\textbf{{Metric}} & {header} \\\\",
        "\\midrule",
    ]
    for label, key in [
        ("CoLA raw AUC", "raw_auc_all"),
        ("CRC-GAD conf.\\ AUC", "conformal_auc"),
        ("FPR@0.05", "fpr@0.05"),
        ("TPR@0.05", "tpr@0.05"),
        ("Flagged@0.05", "flagged@0.05"),
        ("Theory flagged@0.05", "theory_max_flagged@0.05"),
    ]:
        cells = []
        for d in DATASETS:
            if d in by:
                cells.append(ms(by[d], key)[0])
            else:
                cells.append("---")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    # Max absolute identity gap across seeds / alpha levels (should be ~0)
    gap_cells = []
    gap_keys = [f"identity_gap@{a}" for a in ("0.01", "0.02", "0.05", "0.1")]
    for d in DATASETS:
        if d not in by:
            gap_cells.append("---")
            continue
        gaps = []
        for r in by[d]:
            for k in gap_keys:
                if r.get(k) not in (None, "", "nan"):
                    gaps.append(abs(float(r[k])))
        gap_cells.append(f"{max(gaps):.1e}" if gaps else "---")
    lines.append(
        "Max ID gap & " + " & ".join(gap_cells) + " \\\\"
    )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("main_results.tex", "\n".join(lines) + "\n")


def gen_alpha_sweep(rows):
    cora = [r for r in rows if r["dataset"] == "cora"]
    lines = [
        "\\begin{tabular}{ccc}",
        "\\toprule",
        "$\\alpha$ & Observed FPR & TPR \\\\",
        "\\midrule",
    ]
    for a in ["0.01", "0.02", "0.05", "0.10"]:
        key_a = "0.1" if a == "0.10" else a
        fpr_m, _ = ms(cora, f"fpr@{key_a}")
        tpr_m, _ = ms(cora, f"tpr@{key_a}")
        lines.append(f"{a} & {fpr_m} & {tpr_m} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("alpha_sweep.tex", "\n".join(lines) + "\n")


def gen_ablation_components(rows):
    cora = [r for r in rows if r["dataset"] == "cora"]
    raw_m, _ = ms(cora, "raw_auc_all")
    conf_m, _ = ms(cora, "conformal_auc")
    fpr_m, _ = ms(cora, "fpr@0.05")
    write(
        "ablation_components.tex",
        f"""\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Variant}} & \\textbf{{Raw AUC}} & \\textbf{{Conf. AUC}} & \\textbf{{FPR@0.05}} \\\\
\\midrule
CoLA backbone (val-AUC ES) & {raw_m} & --- & --- \\\\
+ Split conformal (CRC-GAD) & {raw_m} & {conf_m} & {fpr_m} \\\\
\\bottomrule
\\end{{tabular}}
""",
    )


def _mean_cell(vals: list[float]) -> str:
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}$\\pm${np.std(vals):.3f}"


def gen_studies(study_rows):
    cont = [r for r in study_rows if r.get("study") == "contamination"]
    if cont:
        by_pi: dict[float, list] = defaultdict(list)
        for r in cont:
            by_pi[float(r["contamination_frac"])].append(r)
        lines = [
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "$\\pi_{\\mathcal{C}}$ & FPR@0.05 & Flagged rate & Conformal AUC \\\\",
            "\\midrule",
        ]
        for pi in sorted(by_pi):
            rows = by_pi[pi]
            lines.append(
                f"{pi:.2f} & {_mean_cell([float(r['fpr@0.05']) for r in rows])} & "
                f"{_mean_cell([float(r['flagged@0.05']) for r in rows])} & "
                f"{_mean_cell([float(r['conformal_auc']) for r in rows])} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("contamination.tex", "\n".join(lines) + "\n")

    dep = [r for r in study_rows if r.get("study") == "dependence"]
    if dep:
        by_rw: dict[float, list] = defaultdict(list)
        for r in dep:
            by_rw[float(r["rewire_frac"])].append(r)
        lines = [
            "\\begin{tabular}{ccccc}",
            "\\toprule",
            "Rewire frac & FPR mean & FPR std & 5th pct & 95th pct \\\\",
            "\\midrule",
        ]
        for rw in sorted(by_rw):
            rows = by_rw[rw]
            lines.append(
                f"{rw:.1f} & {_mean_cell([float(r['fpr_mean']) for r in rows])} & "
                f"{_mean_cell([float(r['fpr_std']) for r in rows])} & "
                f"{_mean_cell([float(r['fpr_p5']) for r in rows])} & "
                f"{_mean_cell([float(r['fpr_p95']) for r in rows])} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("dependence.tex", "\n".join(lines) + "\n")

    heur = [r for r in study_rows if r.get("study") == "heuristics"]
    if heur:
        order = ["CRC-GAD", "Percentile", "Feature-space CP", "Uniform random + CP"]
        by_m: dict[str, list] = defaultdict(list)
        for r in heur:
            by_m[r["method"]].append(r)
        lines = [
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & FPR@0.05 & TPR@0.05 & Flagged rate & Guarantee? \\\\",
            "\\midrule",
        ]
        for method in order:
            if method not in by_m:
                continue
            rows = by_m[method]
            guarantee = "Marginal" if ("CRC" in method or "Uniform" in method) else "No"
            lines.append(
                f"{method} & {_mean_cell([float(r['fpr@0.05']) for r in rows])} & "
                f"{_mean_cell([float(r['tpr@0.05']) for r in rows])} & "
                f"{_mean_cell([float(r['flagged@0.05']) for r in rows])} & "
                f"{guarantee} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("heuristics.tex", "\n".join(lines) + "\n")


def main():
    if not MAIN.exists():
        print("Run experiments first")
        return
    rows = load_main()
    gen_main_results(rows)
    gen_alpha_sweep(rows)
    gen_ablation_components(rows)
    gen_studies(load_studies())


if __name__ == "__main__":
    main()
