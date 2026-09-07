#!/usr/bin/env python3
"""CRC-GAD methodology figure: reference layout, protocol-accurate content."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "pipeline.png"

INK = "#1e3a5f"
BLUE = "#4a90c8"
PALE = "#e8f2fb"
PALE2 = "#f4f8fc"
EDGE = "#7aa3c9"
TEAL = "#1f7a62"
ORANGE = "#e07a3d"
RED = "#d64545"
NAVY = "#243044"


def rounded(ax, x, y, w, h, fc, ec=EDGE, lw=1.3, r=0.12, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={r}",
            linewidth=lw,
            facecolor=fc,
            edgecolor=ec,
            mutation_aspect=0.55,
            zorder=z,
        )
    )


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#5b7c99",
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def panel_title(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=5)


def draw_graph(ax, cx, cy, s=0.18):
    rng = np.random.default_rng(3)
    pts = []
    for k in range(14):
        ang = 2 * np.pi * k / 14 + 0.2
        rad = 0.55 + 0.18 * (k % 3)
        pts.append((cx + rad * np.cos(ang) * 1.15, cy + rad * np.sin(ang) * 0.85))
    pts = np.array(pts)
    anomaly = {2, 7, 11}
    for i, (x, y) in enumerate(pts):
        for j in range(i + 1, len(pts)):
            if np.hypot(*(pts[i] - pts[j])) < 0.72:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#b7c9da", lw=0.7, zorder=3)
    for i, (x, y) in enumerate(pts):
        c = RED if i in anomaly else "#4f8fd0"
        ax.add_patch(Circle((x, y), s * 0.55, facecolor=c, edgecolor="white", lw=0.6, zorder=4))


def draw_matrices(ax, x, y):
    # Adjacency and feature heatmaps
    rng = np.random.default_rng(1)
    A = (rng.random((5, 5)) > 0.55).astype(float)
    np.fill_diagonal(A, 1.0)
    X = rng.random((5, 4))
    ax.imshow(A, extent=(x, x + 0.62, y, y + 0.62), cmap="Blues", vmin=0, vmax=1, zorder=5, interpolation="nearest")
    ax.imshow(X, extent=(x + 0.78, x + 1.28, y, y + 0.62), cmap="YlOrRd", vmin=0, vmax=1, zorder=5, interpolation="nearest")
    for rect, lab in ((x + 0.31, "A"), (x + 1.03, "X")):
        ax.text(rect, y - 0.16, lab, ha="center", va="top", fontsize=7, color=NAVY, zorder=5)


def score_hist(ax, x, y, w, h, color, mark=None):
    rng = np.random.default_rng(7)
    vals = np.concatenate([rng.normal(0.35, 0.08, 80), rng.normal(0.72, 0.06, 12)])
    vals = np.clip(vals, 0.05, 0.95)
    bins = np.linspace(0.05, 0.95, 14)
    hist, edges = np.histogram(vals, bins=bins)
    hist = hist / hist.max()
    bw = (edges[1] - edges[0]) * w * 0.82
    left = x + 0.08 * w
    base = y + 0.08 * h
    span = 0.78 * h
    for i, hv in enumerate(hist):
        ax.add_patch(
            Rectangle(
                (left + i * bw * 1.05, base),
                bw * 0.9,
                hv * span,
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
                zorder=5,
            )
        )
    if mark is not None:
        mx = left + mark * w * 0.9
        ax.plot([mx, mx], [base, base + span], color=RED, lw=1.2, zorder=6)
        ax.text(mx, y + h - 0.02, r"$s_i$", ha="center", va="top", fontsize=7, color=RED, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(15.6, 5.55), dpi=300)
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 5.55)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        7.8,
        5.28,
        "CRC-GAD Framework Overview",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )

    # Panels
    xs = [0.18, 3.35, 6.55, 8.55, 12.15]
    ws = [2.95, 2.95, 1.75, 3.35, 3.22]
    y0, h = 0.28, 4.72
    for x, w in zip(xs, ws):
        rounded(ax, x, y0, w, h, PALE, EDGE, lw=1.2, r=0.14, z=1)

    # ---------- 1 Input ----------
    panel_title(ax, xs[0] + ws[0] / 2, 4.72, "Input graph")
    draw_graph(ax, xs[0] + 1.48, 3.35, 0.16)
    ax.text(xs[0] + 1.48, 2.28, "Attributed network", ha="center", fontsize=8, color=NAVY, zorder=5)
    ax.text(
        xs[0] + 1.48,
        2.05,
        r"$\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$",
        ha="center",
        fontsize=8,
        color=NAVY,
        zorder=5,
    )
    draw_matrices(ax, xs[0] + 0.72, 1.05)
    ax.text(xs[0] + 0.85, 0.72, "Adjacency", ha="center", fontsize=6.5, color="#5a6b7c", zorder=5)
    ax.text(xs[0] + 1.95, 0.72, "Features", ha="center", fontsize=6.5, color="#5a6b7c", zorder=5)
    ax.text(xs[0] + 1.48, 0.48, "Labels unused at calibration", ha="center", fontsize=6.6, color="#7a4a4a", style="italic", zorder=5)

    arrow(ax, xs[0] + ws[0] + 0.02, 2.7, xs[1] - 0.02, 2.7)

    # ---------- 2 Frozen scorer ----------
    panel_title(ax, xs[1] + ws[1] / 2, 4.72, "Frozen scorer")
    ax.text(xs[1] + ws[1] / 2, 4.42, "any backbone, no labels", ha="center", fontsize=7.2, color="#5a6b7c", zorder=5)

    rows = [
        ("CoLA-inspired", "default contrastive"),
        ("DOMINANT", "reconstruction"),
        ("CONAD-style", "Siamese + prior"),
        ("Heuristic", "degree / attribute"),
    ]
    for i, (lab, sub) in enumerate(rows):
        yy = 3.55 - i * 0.72
        rounded(ax, xs[1] + 0.22, yy, ws[1] - 0.44, 0.62, "white", EDGE, lw=1.0, r=0.08, z=3)
        ax.text(xs[1] + ws[1] / 2, yy + 0.40, lab, ha="center", va="center", fontsize=8, fontweight="bold", color=INK, zorder=5)
        ax.text(xs[1] + ws[1] / 2, yy + 0.18, sub, ha="center", va="center", fontsize=6.6, color="#5a6b7c", zorder=5)
    ax.text(xs[1] + ws[1] / 2, 0.55, r"outputs frozen $\{s_i\}$", ha="center", fontsize=7.4, color=TEAL, fontweight="bold", zorder=5)

    arrow(ax, xs[1] + ws[1] + 0.02, 2.7, xs[2] - 0.02, 2.7)

    # ---------- 3 Raw scores ----------
    panel_title(ax, xs[2] + ws[2] / 2, 4.72, "Raw scores")
    score_hist(ax, xs[2] + 0.12, 1.55, ws[2] - 0.24, 2.7, "#5b8fbf")
    ax.text(xs[2] + ws[2] / 2, 1.25, "unscaled ranking", ha="center", fontsize=7, color=NAVY, zorder=5)
    ax.text(xs[2] + ws[2] / 2, 0.95, r"$s_1,\ldots,s_N$", ha="center", fontsize=8, color=NAVY, zorder=5)
    ax.text(xs[2] + ws[2] / 2, 0.62, "no finite-sample\nthreshold", ha="center", fontsize=6.5, color="#7a4a4a", zorder=5)

    arrow(ax, xs[2] + ws[2] + 0.02, 2.7, xs[3] - 0.02, 2.7)

    # ---------- 4 Conformal ----------
    panel_title(ax, xs[3] + ws[3] / 2, 4.72, "Split conformal")
    ax.text(xs[3] + ws[3] / 2, 4.42, r"label-free random $\mathcal{C}/\mathcal{T}$", ha="center", fontsize=7.2, color=TEAL, zorder=5)

    # C / T bar
    bar_x, bar_y, bar_w, bar_h = xs[3] + 0.28, 3.72, ws[3] - 0.56, 0.42
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w * 0.30, bar_h, facecolor="#7dcea0", edgecolor="white", lw=0.6, zorder=5))
    ax.add_patch(Rectangle((bar_x + bar_w * 0.30, bar_y), bar_w * 0.70, bar_h, facecolor="#f0b27a", edgecolor="white", lw=0.6, zorder=5))
    ax.text(bar_x + bar_w * 0.15, bar_y + bar_h / 2, r"$\mathcal{C}$", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=6)
    ax.text(bar_x + bar_w * 0.65, bar_y + bar_h / 2, r"$\mathcal{T}$", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=6)
    ax.text(xs[3] + ws[3] / 2, 3.52, r"$\rho=0.3$ of evaluation nodes; no $y_i$", ha="center", fontsize=6.6, color="#5a6b7c", zorder=5)

    ax.text(
        xs[3] + ws[3] / 2,
        3.05,
        r"$p_i=\dfrac{1+\#\{j\in\mathcal{C}:s_j\geq s_i\}}{1+|\mathcal{C}|}$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK,
        zorder=5,
    )

    score_hist(ax, xs[3] + 0.35, 1.15, ws[3] - 0.7, 1.55, "#7dcea0", mark=0.62)
    ax.text(xs[3] + ws[3] / 2, 0.95, "rank test score against calibration", ha="center", fontsize=6.6, color="#5a6b7c", zorder=5)
    ax.text(xs[3] + ws[3] / 2, 0.62, "validity from partition randomness", ha="center", fontsize=6.6, color=TEAL, zorder=5)

    arrow(ax, xs[3] + ws[3] + 0.02, 2.7, xs[4] - 0.02, 2.7)

    # ---------- 5 Decision ----------
    panel_title(ax, xs[4] + ws[4] / 2, 4.72, r"Decision at $\alpha$")
    ax.text(xs[4] + ws[4] / 2, 4.38, r"flag if $p_i\leq\alpha$", ha="center", fontsize=8, color=INK, zorder=5)

    # p-value axis
    x0, x1, yy = xs[4] + 0.35, xs[4] + ws[4] - 0.35, 3.35
    ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.1), zorder=5)
    ax.plot([x0, x0], [yy - 0.06, yy + 0.06], color=NAVY, lw=1.0, zorder=5)
    ax.text(x0, yy - 0.22, "0", ha="center", fontsize=7, color=NAVY, zorder=5)
    ax.text(x1, yy - 0.22, "1", ha="center", fontsize=7, color=NAVY, zorder=5)
    # Visual α sits slightly right of 0 so the flagged cluster is readable; the box states the 0.05 budget.
    alpha_x = x0 + 0.16 * (x1 - x0)
    ax.plot([alpha_x, alpha_x], [yy - 0.18, yy + 0.28], color=RED, lw=1.3, zorder=6)
    ax.text(alpha_x, yy + 0.36, r"$\alpha$", ha="center", fontsize=8, color=RED, zorder=6)

    rng = np.random.default_rng(4)
    flagged = rng.uniform(0.03, 0.14, 5)
    kept = rng.uniform(0.28, 0.92, 8)
    for p in flagged:
        ax.add_patch(Circle((x0 + p * (x1 - x0), yy), 0.055, facecolor=RED, edgecolor="white", lw=0.4, zorder=6))
    for p in kept:
        ax.add_patch(Circle((x0 + p * (x1 - x0), yy), 0.055, facecolor="#4f8fd0", edgecolor="white", lw=0.4, zorder=6))
    ax.text(xs[4] + 0.7, 2.85, "flagged", ha="center", fontsize=6.5, color=RED, zorder=5)
    ax.text(xs[4] + 2.15, 2.85, "not flagged", ha="center", fontsize=6.5, color="#4f8fd0", zorder=5)

    rounded(ax, xs[4] + 0.28, 1.55, ws[4] - 0.56, 0.95, "white", EDGE, lw=1.0, r=0.08, z=3)
    ax.text(xs[4] + ws[4] / 2, 2.22, r"user budget $\alpha$", ha="center", fontsize=8, fontweight="bold", color=INK, zorder=5)
    ax.text(xs[4] + ws[4] / 2, 1.92, "e.g. 0.05", ha="center", fontsize=7.2, color="#5a6b7c", zorder=5)
    ax.text(xs[4] + ws[4] / 2, 1.70, "TPR is not guaranteed", ha="center", fontsize=6.6, color="#7a4a4a", zorder=5)

    rounded(ax, xs[4] + 0.28, 0.48, ws[4] - 0.56, 0.92, "#e8f6f1", TEAL, lw=1.15, r=0.08, z=3)
    ax.text(xs[4] + ws[4] / 2, 1.12, "Marginal validity", ha="center", fontsize=8, fontweight="bold", color=TEAL, zorder=5)
    ax.text(xs[4] + ws[4] / 2, 0.86, r"random test node: $\Pr(p_i\leq\alpha)\leq\alpha$", ha="center", fontsize=6.4, color=NAVY, zorder=5)
    ax.text(xs[4] + ws[4] / 2, 0.64, "not node-conditional FPR", ha="center", fontsize=6.4, color="#5a6b7c", zorder=5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
