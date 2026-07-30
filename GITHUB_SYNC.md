# Sync local CRC_GAD → GitHub

**Remote:** https://github.com/Amaanullakhan/CRC_GAD

## What is on GitHub now (old)

The public repo currently has only:

- `README.md`
- `crc_gad/` (old stub, 2 commits)

It does **not** have: `scripts/`, `src/crc_gad/`, `results/main_results.csv`, `paper/main.tex`, or tag `v1.0-rebuild`.

## What you will push (this folder)

| Path | Purpose |
|------|---------|
| `src/crc_gad/` | Scorer + conformal pipeline |
| `scripts/` | Experiments, tables, figures |
| `results/main_results.csv` | 30-row canonical results |
| `results/studies.csv` | Contamination / dependence |
| `paper/main.tex` | Rewritten manuscript |
| `paper/generated/` | Auto tables |
| `paper/figures/` | Auto figures |

Large files in `data/*.mat` are **gitignored** — users run `scripts/download_datasets.py`.

## Steps

1. Install Git: https://git-scm.com/download/win  
2. Create personal access token (if HTTPS push asks): GitHub → Settings → Developer settings → PAT  
3. Run:

```powershell
cd C:\Users\DSU-CSE514-38\Desktop\AMAAN_CRC\CRC_GAD
.\scripts\push_to_github.ps1
git branch -M main
git push -u origin main
git push origin v1.0-rebuild
```

If remote has unrelated history and push is rejected:

```powershell
git push -u origin main --force
git push origin v1.0-rebuild --force
```

4. Verify: https://github.com/Amaanullakhan/CRC_GAD/tree/v1.0-rebuild

## Paper citation

Use in Data Availability:

> Code and scripts: https://github.com/Amaanullakhan/CRC_GAD/tree/v1.0-rebuild
