#!/usr/bin/env python3
"""Verify paper generated tables and narrative numbers against CSV."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ms(rows, key):
    v = [float(r[key]) for r in rows]
    return float(np.mean(v)), float(np.std(v))


def main():
    main_rows = list(csv.DictReader(open(ROOT / "results" / "main_results.csv", encoding="utf-8")))
    studies = list(csv.DictReader(open(ROOT / "results" / "studies.csv", encoding="utf-8")))
    by = defaultdict(list)
    for r in main_rows:
        by[r["dataset"]].append(r)

    datasets = ["cora", "citeseer", "pubmed", "acm", "blogcatalog", "flickr"]
    issues = []

    print("=== MAIN RESULTS (CSV) ===")
    for d in datasets:
        raw, sraw = ms(by[d], "raw_auc_all")
        conf, sconf = ms(by[d], "conformal_auc")
        fpr, sfpr = ms(by[d], "fpr@0.05")
        tpr, stpr = ms(by[d], "tpr@0.05")
        fl, sfl = ms(by[d], "flagged@0.05")
        th, _ = ms(by[d], "theory_max_flagged@0.05")
        gaps = []
        for r in by[d]:
            for a in ("0.01", "0.02", "0.05", "0.1"):
                gaps.append(abs(float(r[f"identity_gap@{a}"])))
        print(
            f"{d:12} raw={raw:.3f}+/-{sraw:.3f} conf={conf:.3f}+/-{sconf:.3f} "
            f"fpr={fpr:.3f}+/-{sfpr:.3f} tpr={tpr:.3f}+/-{stpr:.3f} "
            f"flagged={fl:.3f} theory={th:.3f} maxgap={max(gaps):.1e}"
        )

    print("\n=== GENERATED main_results.tex vs CSV ===")
    tex = (ROOT / "paper" / "generated" / "main_results.tex").read_text(encoding="utf-8")
    data_lines = [
        ln
        for ln in tex.splitlines()
        if "&" in ln
        and not any(x in ln for x in ("Metric", "toprule", "midrule", "bottomrule", "tabular", "resizebox"))
    ]
    keys = [
        "raw_auc_all",
        "conformal_auc",
        "fpr@0.05",
        "tpr@0.05",
        "flagged@0.05",
        "theory_max_flagged@0.05",
    ]
    for i, key in enumerate(keys):
        cells = re.findall(r"([0-9.]+)\$\\pm\$([0-9.]+)", data_lines[i])
        ok = True
        for j, d in enumerate(datasets):
            m, s = ms(by[d], key)
            tm, ts = float(cells[j][0]), float(cells[j][1])
            if abs(m - tm) > 5e-4 or abs(s - ts) > 5e-4:
                ok = False
                issues.append(f"TEX mismatch {key}/{d}: csv {m:.3f}+/-{s:.3f} vs tex {tm:.3f}+/-{ts:.3f}")
        print(f"{key}: {'MATCH' if ok else 'MISMATCH'}")

    # identity gap row
    gap_cells = re.findall(r"([0-9.]+e-[0-9]+)", data_lines[6])
    for j, d in enumerate(datasets):
        gaps = []
        for r in by[d]:
            for a in ("0.01", "0.02", "0.05", "0.1"):
                gaps.append(abs(float(r[f"identity_gap@{a}"])))
        g = max(gaps)
        tg = float(gap_cells[j])
        if abs(g - tg) / max(g, 1e-30) > 0.05:
            issues.append(f"identity gap mismatch {d}: csv {g:.1e} vs tex {tg:.1e}")
    print(f"identity gaps: {'MATCH' if not any('identity gap' in x for x in issues) else 'MISMATCH'}")

    print("\n=== NARRATIVE CLAIMS in main.tex ===")
    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    acm_fpr = ms(by["acm"], "fpr@0.05")[0]
    blog_fpr = ms(by["blogcatalog"], "fpr@0.05")[0]
    blog_auc = ms(by["blogcatalog"], "raw_auc_all")[0]
    flickr_auc = ms(by["flickr"], "raw_auc_all")[0]
    checks = [
        ("ACM FPR exceeds alpha (0.051)", acm_fpr > 0.05 and abs(acm_fpr - 0.051) < 0.001),
        ("BlogCatalog FPR exceeds alpha (0.055)", blog_fpr > 0.05 and abs(blog_fpr - 0.055) < 0.001),
        ("BlogCatalog below-chance AUC (0.365)", blog_auc < 0.5 and abs(blog_auc - 0.365) < 0.001),
        ("Flickr below-chance AUC (0.342)", flickr_auc < 0.5 and abs(flickr_auc - 0.342) < 0.001),
        ("Cora/Citeseer/PubMed/Flickr FPR <= 0.05", all(ms(by[d], "fpr@0.05")[0] <= 0.05 for d in ("cora", "citeseer", "pubmed", "flickr"))),
        ("Paper mentions ACM 0.051", "0.051" in paper),
        ("Paper mentions BlogCatalog 0.055", "0.055" in paper),
        ("Paper mentions BlogCatalog 0.365", "0.365" in paper),
        ("Paper mentions Flickr 0.342", "0.342" in paper),
    ]
    for name, ok in checks:
        print(f"{'OK' if ok else 'FAIL'}  {name}")
        if not ok:
            issues.append(name)

    print("\n=== HEURISTICS (5 seeds) ===")
    h = defaultdict(list)
    for r in studies:
        if r.get("study") == "heuristics":
            h[r["method"]].append(r)
    heur_tex = (ROOT / "paper" / "generated" / "heuristics.tex").read_text(encoding="utf-8")
    for method in ["CRC-GAD", "Percentile", "Feature-space CP", "Uniform random + CP"]:
        rows = h[method]
        fpr, sf = ms(rows, "fpr@0.05")
        tpr, st = ms(rows, "tpr@0.05")
        fl, sfl = ms(rows, "flagged@0.05")
        print(f"{method:22} fpr={fpr:.3f}+/-{sf:.3f} tpr={tpr:.3f}+/-{st:.3f} fl={fl:.3f}+/-{sfl:.3f}")
        for ln in heur_tex.splitlines():
            if ln.startswith(method):
                nums = re.findall(r"([0-9.]+)\$\\pm\$([0-9.]+)", ln)
                tm = [float(x) for pair in nums for x in pair]
                if abs(tm[0] - fpr) > 5e-4 or abs(tm[2] - tpr) > 5e-4 or abs(tm[4] - fl) > 5e-4:
                    issues.append(f"Heuristics tex mismatch {method}")
                    print("  TEX MISMATCH")
                else:
                    print("  TEX MATCH")

    print("\n=== CONTAMINATION ===")
    c = defaultdict(list)
    for r in studies:
        if r.get("study") == "contamination":
            c[float(r["contamination_frac"])].append(float(r["fpr@0.05"]))
    fps = []
    for pi in sorted(c):
        m = float(np.mean(c[pi]))
        fps.append(m)
        print(f"pi={pi:.2f} fpr={m:.3f}+/-{float(np.std(c[pi])):.3f}")
    if all(fps[i] >= fps[i + 1] - 1e-9 for i in range(len(fps) - 1)):
        print("FPR decreases (or flat) with contamination: YES")
    else:
        issues.append("Contamination FPR not non-increasing")
        print("FPR decreases with contamination: NO")

    print("\n=== DEPENDENCE ===")
    dep = defaultdict(list)
    for r in studies:
        if r.get("study") == "dependence":
            dep[float(r["rewire_frac"])].append(float(r["fpr_mean"]))
    for rw in sorted(dep):
        print(f"rewire={rw:.1f} mean_fpr={float(np.mean(dep[rw])):.3f}")

    print("\n=== SUMMARY ===")
    print(f"main_results rows={len(main_rows)} studies rows={len(studies)}")
    if issues:
        print("ISSUES:")
        for x in issues:
            print(" -", x)
    else:
        print("ALL CHECKS PASSED: tables + narrative numbers match CSV.")


if __name__ == "__main__":
    main()
