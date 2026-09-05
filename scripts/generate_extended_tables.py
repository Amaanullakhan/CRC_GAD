#!/usr/bin/env python3
"""Generate extended paper tables from results/extended_studies.csv (+ optional studies.csv)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
EXT = ROOT / "results" / "extended_studies.csv"
DOM = ROOT / "results" / "dominant_pytorch.csv"
CONAD = ROOT / "results" / "conad_pytorch.csv"
ORG = ROOT / "results" / "organic_studies.csv"


def ms(vals):
    vals = [float(v) for v in vals if v not in (None, "", "nan")]
    if not vals:
        return "---"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}$\\pm${np.std(vals):.3f}"


def tex_id(s: str) -> str:
    """Escape underscores for LaTeX text mode."""
    return str(s).replace("_", "\\_")


def load():
    rows = []
    for path in (EXT, DOM, CONAD, ORG):
        if path.exists():
            rows.extend(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print(f"Missing result CSVs under {ROOT / 'results'}")
    return rows


def write(name, content):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content, encoding="utf-8")
    print("Wrote", name)


def gen_backbones(rows):
    rows = [r for r in rows if r.get("study") == "backbones"]
    if not rows:
        write("backbones.tex", "\\begin{tabular}{c}\\toprule TBD\\\\\\bottomrule\\end{tabular}\n")
        return
    datasets = sorted({r["dataset"] for r in rows})
    backbones = [
        "cola", "dominant", "conad", "dominant_style",
        "degree", "feature_norm", "attr_deviation",
    ]
    lines = [
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(datasets) + "}",
        "\\toprule",
        "Backbone & Metric & " + " & ".join(d.title() for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for bb in backbones:
        if not any(r["backbone"] == bb for r in rows):
            continue
        for metric, key in [
            ("Raw AUC", "raw_auc_all"),
            ("FPR@0.05", "fpr@0.05"),
            ("TPR@0.05", "tpr@0.05"),
            ("Time (s)", "runtime_s"),
        ]:
            cells = []
            for d in datasets:
                vals = [r[key] for r in rows if r["backbone"] == bb and r["dataset"] == d]
                cells.append(ms(vals))
            label = tex_id(bb) if metric == "Raw AUC" else ""
            lines.append(f"{label} & {metric} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "}"]
    write("backbones.tex", "\n".join(lines) + "\n")


def gen_degree(rows):
    rows = [r for r in rows if r.get("study") == "degree_fpr"]
    order = ["Q1_low", "Q2", "Q3", "Q4_high"]
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Degree bin & FPR@0.05 & \\#normals \\\\", "\\midrule"]
    for b in order:
        sub = [r for r in rows if r.get("degree_bin") == b]
        lines.append(f"{tex_id(b)} & {ms([r['fpr@0.05'] for r in sub])} & {ms([r['n_normals'] for r in sub])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("degree_fpr.tex", "\n".join(lines) + "\n")


def gen_organic(rows):
    rows = [r for r in rows if r.get("study") == "organic"]
    if not rows:
        write("organic.tex", "\\begin{tabular}{c}\\toprule TBD\\\\\\bottomrule\\end{tabular}\n")
        return
    datasets = sorted({r["dataset"] for r in rows})
    backbones = sorted({r["backbone"] for r in rows})
    # Multi-dataset organic table: Backbone x (AUC/FPR per dataset)
    lines = [
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "cc" * len(datasets) + "}",
        "\\toprule",
        "Backbone & " + " & ".join(f"{tex_id(d)} AUC & FPR" for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for bb in backbones:
        cells = []
        for d in datasets:
            sub = [r for r in rows if r["backbone"] == bb and r["dataset"] == d]
            cells.append(ms([r["raw_auc_all"] for r in sub]))
            cells.append(ms([r["fpr@0.05"] for r in sub]))
        lines.append(f"{tex_id(bb)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    write("organic.tex", "\n".join(lines) + "\n")


def gen_contamination_multi(rows):
    rows = [r for r in rows if r.get("study") == "contamination"]
    datasets = sorted({r["dataset"] for r in rows})
    pis = sorted({float(r["contamination_frac"]) for r in rows})
    lines = [
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{c" + "c" * len(datasets) + "}",
        "\\toprule",
        "$\\pi_{\\mathcal{C}}$ & " + " & ".join(f"FPR ({d})" for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for pi in pis:
        cells = []
        for d in datasets:
            vals = [r["fpr@0.05"] for r in rows if r["dataset"] == d and abs(float(r["contamination_frac"]) - pi) < 1e-9]
            cells.append(ms(vals))
        lines.append(f"{pi:.2f} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    write("contamination_multi.tex", "\n".join(lines) + "\n")


def gen_heuristics_multi(rows):
    rows = [r for r in rows if r.get("study") == "heuristics"]
    datasets = sorted({r["dataset"] for r in rows})
    methods = ["CRC-GAD", "Percentile", "Feature-space CP", "Uniform random + CP"]
    lines = [
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "cc" * len(datasets) + "}",
        "\\toprule",
        "Method & & " + " & ".join(f"{d.title()} FPR & TPR" for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = []
        for d in datasets:
            sub = [r for r in rows if r["method"] == method and r["dataset"] == d]
            cells.append(ms([r["fpr@0.05"] for r in sub]))
            cells.append(ms([r["tpr@0.05"] for r in sub]))
        lines.append(f"{method} & & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    write("heuristics_multi.tex", "\n".join(lines) + "\n")


def gen_per_seed_supplement(main_csv: Path):
    if not main_csv.exists():
        return
    rows = list(csv.DictReader(open(main_csv, encoding="utf-8")))
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Seed & Raw AUC & Conf.\\ AUC & FPR@0.05 & TPR@0.05 \\\\",
        "\\midrule",
    ]
    for r in sorted(rows, key=lambda x: (x["dataset"], int(x["seed"]))):
        lines.append(
            f"{r['dataset']} & {r['seed']} & {float(r['raw_auc_all']):.3f} & "
            f"{float(r['conformal_auc']):.3f} & {float(r['fpr@0.05']):.3f} & {float(r['tpr@0.05']):.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("per_seed_main.tex", "\n".join(lines) + "\n")


def main():
    rows = load()
    gen_backbones(rows)
    gen_degree(rows)
    gen_organic(rows)
    gen_contamination_multi(rows)
    gen_heuristics_multi(rows)
    gen_per_seed_supplement(ROOT / "results" / "main_results.csv")


if __name__ == "__main__":
    main()
