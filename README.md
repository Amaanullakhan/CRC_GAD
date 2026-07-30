# CRC-GAD — Reproducible Experiment Code

Conformal Risk-Controlled Graph Anomaly Detection (rebuild after ESWA R1 rejection).

**Repository:** [https://github.com/Amaanullakhan/CRC_GAD](https://github.com/Amaanullakhan/CRC_GAD)  
**Release tag (after push):** `v1.0-rebuild`

## Status

- **Experiments:** `results/main_results.csv` — 30 rows (6 datasets × 5 seeds), real `.mat` for ACM/BlogCatalog/Flickr
- **Paper tables/figures:** `paper/generated/*.tex`, `paper/figures/*.png` (from CSV)
- **GitHub:** Push this folder to replace the old stub repo (see below)

## Push to GitHub

Public repo includes **code, results CSVs, and paper only** — not internal notes, logs, or `data/*.mat`.

```bash
cd /c/Users/DSU-CSE514-38/Desktop/AMAAN_CRC/CRC_GAD
git add .
git commit -m "Keep only reproducibility files on GitHub"
git push origin main
git tag -f v1.0-rebuild
git push origin v1.0-rebuild --force
```

## Quick start

```bash
cd CRC_GAD
pip install -r requirements.txt

# Smoke test (Cora, seed 0)
python scripts/run_experiments.py --datasets cora --seeds 0 --fast

# Core experiments (6 datasets × 5 seeds)
python scripts/run_experiments.py

# Supplementary studies (contamination, 200-split dependence, heuristics)
python scripts/run_studies.py --seed 0

# Figures from CSV only
python scripts/regenerate_figures.py
```

## Protocol (label-free calibration)

1. **Train** (`train_frac=0.6`): fit CoLA-style scorer — no anomaly labels
2. **Val** (10% of remaining): early stopping only
3. **Eval**: random split into **C** (`rho=0.3`) and **T** — **labels never used**
4. Anomalies remain in both C and T (small fraction ~5%)

Audit logs: `anomalies_in_C`, `anomalies_in_T`, `flagged_rate` vs Theorem 2 bound.

## Reproduce a table

| Output | Command |
|--------|---------|
| Table: main FPR/TPR/AUC | `scripts/run_experiments.py` → `results/main_results.csv` |
| Table: contamination | `scripts/run_studies.py` → `results/studies.csv` |
| Figures | `scripts/regenerate_figures.py` → `paper/figures/` |

## Paper source

[`paper/main.tex`](paper/main.tex) — rewritten for marginal FPR claims, CoLA attribution, honest novelty.

## Git tag (after verification)

```bash
git tag v1.0-rebuild
```

Pin this commit in the paper Data Availability section.

## Target venue (Phase 4)

Knowledge-Based Systems or Neural Networks — focused calibration/deployment paper, not broad SOTA GAD claims.
