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
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * len(DATASETS)}}}",
        "\\toprule",
        f"\\textbf{{Metric}} & {header} \\\\",
        "\\midrule",
    ]
    for label, key in [
        ("CoLA backbone AUC (raw)", "raw_auc_all"),
        ("CRC-GAD conformal AUC", "conformal_auc"),
        ("FPR @ $\\alpha{=}0.05$", "fpr@0.05"),
        ("TPR @ $\\alpha{=}0.05$", "tpr@0.05"),
        ("Flagged rate @ $0.05$", "flagged@0.05"),
        ("Theory max flagged @ $0.05$", "theory_max_flagged@0.05"),
    ]:
        cells = []
        for d in DATASETS:
            if d in by:
                cells.append(ms(by[d], key)[0])
            else:
                cells.append("---")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
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


def gen_studies(study_rows):
    cont = [r for r in study_rows if r.get("study") == "contamination"]
    if cont:
        lines = [
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "$\\pi_{\\mathcal{C}}$ & FPR@0.05 & Flagged rate & Conformal AUC \\\\",
            "\\midrule",
        ]
        for r in cont:
            pi = float(r["contamination_frac"])
            lines.append(
                f"{pi:.2f} & {float(r['fpr@0.05']):.3f} & "
                f"{float(r['flagged@0.05']):.3f} & {float(r['conformal_auc']):.3f} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("contamination.tex", "\n".join(lines) + "\n")

    dep = [r for r in study_rows if r.get("study") == "dependence"]
    if dep:
        lines = [
            "\\begin{tabular}{ccccc}",
            "\\toprule",
            "Rewire frac & FPR mean & FPR std & 5th pct & 95th pct \\\\",
            "\\midrule",
        ]
        for r in sorted(dep, key=lambda x: float(x["rewire_frac"])):
            lines.append(
                f"{float(r['rewire_frac']):.1f} & {float(r['fpr_mean']):.3f} & "
                f"{float(r['fpr_std']):.3f} & {float(r['fpr_p5']):.3f} & "
                f"{float(r['fpr_p95']):.3f} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        write("dependence.tex", "\n".join(lines) + "\n")

    heur = [r for r in study_rows if r.get("study") == "heuristics"]
    if heur:
        lines = [
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & FPR@0.05 & TPR@0.05 & Flagged rate & Guarantee? \\\\",
            "\\midrule",
        ]
        for r in heur:
            fr = float(r["flagged@0.05"])
            lines.append(
                f"{r['method']} & {float(r['fpr@0.05']):.3f} & "
                f"{float(r['tpr@0.05']):.3f} & {fr:.3f} & "
                f"{'Marginal' if 'CRC' in r['method'] else 'No'} \\\\"
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
