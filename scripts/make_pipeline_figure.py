"""Placeholder pipeline figure — replace with CoLA-attributed diagram."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(__file__).resolve().parents[1] / "paper" / "figures" / "pipeline.png"
out.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.axis("off")
boxes = [
    (0.2, "Input graph"),
    (2.2, "CoLA contrastive\nscorer (Liu et al. 2022)"),
    (5.0, "Anomaly scores"),
    (7.0, "Split conformal\n(label-free C/T)"),
    (9.0, "Decisions"),
]
for x, t in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.5), 1.6, 1.0, boxstyle="round,pad=0.05", fc="#e8eef4", ec="#4d7c9c"))
    ax.text(x + 0.8, 1.0, t, ha="center", va="center", fontsize=9)
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i + 1][0], 1.0), xytext=(boxes[i][0] + 1.6, 1.0),
                arrowprops=dict(arrowstyle="->", color="#333"))
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Wrote {out}")
