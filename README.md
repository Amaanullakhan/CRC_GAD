# CRC-GAD — Reproducible Experiment Code

Split-conformal calibration of graph anomaly scores with **marginal** random-test-node false-positive control (not node-conditional FPR).

**Repository:** [https://github.com/Amaanullakhan/CRC_GAD](https://github.com/Amaanullakhan/CRC_GAD)  
**Release tag:** `v1.0-rebuild`

## Quick start

```bash
cd CRC_GAD
pip install -r requirements.txt

# Smoke test (Cora, seed 0)
python scripts/run_experiments.py --datasets cora --seeds 0 --fast

# Core experiments (6 datasets × 5 seeds)
python scripts/run_experiments.py

# Supplementary studies (contamination, dependence, heuristics)
python scripts/run_studies.py --seeds 0,1,2,3,4

# Extended studies (multi-backbone, degree FPR, organic proxy, multi-dataset heuristics/contamination)
python scripts/run_extended_studies.py

# Tables and figures from CSV
python scripts/generate_paper_tables.py
python scripts/generate_extended_tables.py
python scripts/regenerate_figures.py
```

## Protocol (label-free calibration)

1. **Train** (`train_frac=0.6`): fit a frozen scorer (default: simplified CoLA-inspired) — no anomaly labels  
2. **Val** (10% of remaining): early stopping only  
3. **Eval**: random split into **C** (`rho=0.3`) and **T** — labels never used for the split  
4. Anomalies may appear in both C and T (~5% injected rate)

See `docs/CALIBRATION_SET_AUDIT.md`.

## Scorers (`src/crc_gad/backbones.py`)

| Name | Role |
|------|------|
| `cola` | Simplified CoLA-inspired contrastive (main tables) |
| `dominant` | **PyTorch DOMINANT** (Ding et al. 2019 architecture) |
| `dominant_style` | NumPy reconstruction principle (lighter baseline) |
| `degree` / `feature_norm` / `attr_deviation` | Heuristic baselines |

```bash
# PyTorch DOMINANT + CRC-GAD (needs working torch)
python scripts/run_dominant_pytorch.py --datasets cora,citeseer,pubmed --seeds 0,1,2,3,4
```

Organic proxy: rarest Planetoid class as anomalies (`organic_cora` in `data.py`).

## Reproduce paper artifacts

| Output | Command |
|--------|---------|
| Main results CSV | `python scripts/run_experiments.py` |
| Studies CSV | `python scripts/run_studies.py --seeds 0,1,2,3,4` |
| Extended CSV | `python scripts/run_extended_studies.py` |
| LaTeX tables | `python scripts/generate_paper_tables.py` + `generate_extended_tables.py` |
| Figures | `python scripts/regenerate_figures.py` |
| Number check | `python scripts/verify_paper_numbers.py` |

## Paper source (Overleaf)

| Path | Purpose |
|------|---------|
| `paper/main.tex` | Manuscript |
| `paper/supplementary.tex` | Per-seed tables + guarantee notes |
| `paper/ref.bib` | Bibliography |
| `paper/generated/*.tex` | Auto-generated tables |
| `paper/figures/*.png` | Figures |

Compile with a standard LaTeX article/journal template (e.g. `cas-dc` if using Elsevier CAS).
