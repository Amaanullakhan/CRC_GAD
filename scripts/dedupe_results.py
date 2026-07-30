#!/usr/bin/env python3
"""Keep last row per (dataset, seed) in results/main_results.csv."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "results" / "main_results.csv"

rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
dedup: dict[tuple[str, str], dict] = {}
for r in rows:
    dedup[(r["dataset"], r["seed"])] = r
out = list(dedup.values())
out.sort(key=lambda r: (r["dataset"], int(r["seed"])))
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(f"Deduped {len(rows)} -> {len(out)} rows")
