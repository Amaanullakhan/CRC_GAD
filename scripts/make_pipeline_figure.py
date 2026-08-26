#!/usr/bin/env python3
"""Redraw CRC-GAD methodology / pipeline figure for the paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "pipeline.png"


def rounded(ax, x, y, w, h, fc, ec, lw=1.4, r=0.08):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
        mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color="#2c3e50"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main():
    # Wide figure* aspect; ink-friendly academic palette (slate + teal accent)
    fig, ax = plt.subplots(figsize=(12.2, 3.7), dpi=400)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    slate = "#1f2a33"
    teal = "#0d6e6e"
    ink = "#243039"
    soft = "#eef3f5"
    soft2 = "#e6f2f1"
    soft3 = "#eef1f4"
    border = "#3d4f5c"
    border2 = "#0d6e6e"
    border3 = "#4a5560"

    # Phase bands (leave clear headroom for titles)
    ax.add_patch(Rectangle((0.15, 0.4), 4.55, 2.7, facecolor="#f7f9fa", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((4.85, 0.4), 4.55, 2.7, facecolor="#f3f8f7", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((9.55, 0.4), 2.45, 2.7, facecolor="#f5f6f8", edgecolor="none", zorder=0))

    ax.text(2.42, 3.4, "Phase A — Frozen GAD scorer", ha="center", va="center", fontsize=10, fontweight="bold", color=slate)
    ax.text(7.12, 3.4, "Phase B — CRC-GAD calibration", ha="center", va="center", fontsize=10, fontweight="bold", color=teal)
    ax.text(10.78, 3.4, "Decision", ha="center", va="center", fontsize=10, fontweight="bold", color=slate)

    # --- Phase A boxes ---
    # Input
    rounded(ax, 0.35, 1.55, 1.7, 1.05, soft, border)
    ax.text(1.2, 2.25, "Attributed graph", ha="center", va="center", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(1.2, 1.85, r"$\mathcal{G}=(\mathcal{V},\mathcal{E},\mathbf{X})$", ha="center", va="center", fontsize=8, color=ink)

    arrow(ax, 2.1, 2.05, 2.45, 2.05)

    # Backbone options (stacked)
    rounded(ax, 2.5, 0.55, 2.0, 2.55, soft, border)
    ax.text(3.5, 2.85, "Any frozen scorer", ha="center", va="center", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(3.5, 2.55, r"$f_\theta$ (no labels)", ha="center", va="center", fontsize=7.5, color="#55606a")

    for i, (lab, sub) in enumerate(
        [
            ("CoLA-style", "contrastive (default)"),
            ("DOMINANT", "PyTorch reconstr."),
            ("Heuristics", "degree / attributes"),
        ]
    ):
        y = 2.15 - i * 0.55
        rounded(ax, 2.7, y - 0.22, 1.6, 0.44, "#ffffff", border, lw=1.0, r=0.05)
        ax.text(3.5, y + 0.02, lab, ha="center", va="center", fontsize=7.8, fontweight="bold", color=ink)
        ax.text(3.5, y - 0.14, sub, ha="center", va="center", fontsize=6.5, color="#5a6670")

    arrow(ax, 4.55, 2.05, 5.05, 2.05)

    # --- Phase B ---
    # Scores
    rounded(ax, 5.1, 1.55, 1.55, 1.05, soft2, border2)
    ax.text(5.875, 2.25, "Anomaly scores", ha="center", va="center", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(5.875, 1.85, r"$\{s_i\}$  (frozen)", ha="center", va="center", fontsize=8, color=ink)

    arrow(ax, 6.7, 2.05, 7.05, 2.05)

    # Conformal block
    rounded(ax, 7.1, 0.55, 2.1, 2.55, soft2, border2, lw=1.6)
    ax.text(8.15, 2.85, "Split conformal", ha="center", va="center", fontsize=8.5, fontweight="bold", color=teal)
    ax.text(8.15, 2.55, "label-free random split", ha="center", va="center", fontsize=7.2, color="#3a6b6b")

    rounded(ax, 7.3, 1.85, 1.7, 0.5, "#ffffff", border2, lw=1.0, r=0.05)
    ax.text(8.15, 2.2, r"Calibration $\mathcal{C}$", ha="center", va="center", fontsize=7.8, fontweight="bold", color=ink)
    ax.text(8.15, 1.98, r"ratio $\rho$", ha="center", va="center", fontsize=6.8, color="#5a6670")

    rounded(ax, 7.3, 1.2, 1.7, 0.5, "#ffffff", border2, lw=1.0, r=0.05)
    ax.text(8.15, 1.55, r"Test $\mathcal{T}$", ha="center", va="center", fontsize=7.8, fontweight="bold", color=ink)
    ax.text(8.15, 1.33, r"$p_i$ vs.\ $\mathcal{C}$", ha="center", va="center", fontsize=6.8, color="#5a6670")

    rounded(ax, 7.3, 0.65, 1.7, 0.42, "#ffffff", border2, lw=1.0, r=0.05)
    ax.text(8.15, 0.86, r"Marginal validity", ha="center", va="center", fontsize=7.5, fontweight="bold", color=teal)

    arrow(ax, 9.25, 2.05, 9.75, 2.05)

    # --- Decision ---
    rounded(ax, 9.8, 1.35, 2.05, 1.45, soft3, border3, lw=1.6)
    ax.text(10.825, 2.5, "Flag if", ha="center", va="center", fontsize=8.5, fontweight="bold", color=ink)
    ax.text(10.825, 2.1, r"$p_i \leq \alpha$", ha="center", va="center", fontsize=11, fontweight="bold", color=slate)
    ax.text(10.825, 1.65, "user-chosen\nfalse-alarm budget", ha="center", va="center", fontsize=7.2, color="#5a6670")

    # Footer note
    ax.text(
        6.1,
        0.12,
        "CRC-GAD does not retrain the backbone; the same wrapper applies to CoLA-style, DOMINANT, and heuristic scorers.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#55606a",
        style="italic",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
