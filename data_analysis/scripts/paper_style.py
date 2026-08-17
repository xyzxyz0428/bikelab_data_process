#!/usr/bin/env python3
"""Shared plotting style for dataset validation figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "black": "#333333",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "grey": "#8A8A8A",
    "light_grey": "#D9D9D9",
}

RTK_COLORS = {
    0: COLORS["grey"],
    1: COLORS["orange"],
    2: COLORS["green"],
}

RTK_LABELS = {
    0: "No carrier solution",
    1: "RTK float",
    2: "RTK fixed",
}


def apply_paper_style():
    """Apply a compact, colour-blind-safe Matplotlib style."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "lines.linewidth": 1.35,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    })


def panel_label(ax, label):
    """Place a panel label just inside the upper-left corner."""
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )


def save_figure(fig, output_base):
    """Save one figure as 300-dpi PNG and editable SVG."""
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
