"""Results figures. Numbers are the locked paper tables, not a re-score."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent

# Print-safe, close to the teaser palette. Initial is the catalog gray,
# DRAFT a steel that still reads when printed B&W, Ours the filled-id green.
INITIAL = "#C8CAD0"
DRAFT = "#7A92A8"
OURS = "#3E7A62"
EDGE = "#2A2A2A"
INK = "#1A1A1A"
GRID = "#E6E6E8"

DOCS = ("Initial", "DRAFT", "Ours")
COLORS = (INITIAL, DRAFT, OURS)

# Table 1 / Table 3. yerr is (low, high) distance from the mean, not CI bounds.
TMDB_MODELS = ("Ling", "DeepSeek V4")
TMDB_EXEC = {
    "Initial": ((51.0, 3.8, 3.8), (63.1, 3.0, 3.1)),
    "DRAFT": ((15.3, 2.6, 2.7), (23.8, 2.3, 2.3)),
    "Ours": ((57.0, 3.1, 3.2), (62.2, 4.4, 4.4)),
}
TMDB_PATH = {
    "Initial": ((89.9, 2.8, 2.8), (91.9, 1.5, 1.4)),
    "DRAFT": ((85.7, 1.8, 1.7), (90.0, 2.2, 2.3)),
    "Ours": ((92.7, 1.6, 1.5), (88.4, 2.6, 2.6)),
}

# One Ling TMDB seed. Path CP. Stuck counts only where the paper states them.
PLACEMENT = (
    ("schema + banner", 80, "10 stuck"),
    ("hop purpose gate", 82, None),
    ("fill in guideline", 93, "2 stuck"),
)


def _rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE)
    ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(colors=INK, length=3, width=0.6)
    ax.set_ylabel("CP (%)", color=INK)
    ax.set_ylim(0, 100)


def _grouped(ax, models, table, *, error: bool, hatch_last: bool) -> None:
    x = np.arange(len(models), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    for doc, color, dx in zip(DOCS, COLORS, offsets):
        means = []
        lows = []
        highs = []
        for cell in table[doc]:
            if isinstance(cell, tuple):
                mean, lo, hi = cell
            else:
                mean, lo, hi = cell, 0.0, 0.0
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        yerr = np.array([lows, highs]) if error else None
        bars = ax.bar(
            x + dx,
            means,
            width=width,
            color=color,
            edgecolor=EDGE,
            linewidth=0.6,
            yerr=yerr,
            error_kw={"ecolor": EDGE, "elinewidth": 0.7, "capsize": 2.2, "capthick": 0.7},
            zorder=3,
        )
        if hatch_last:
            bars[-1].set_hatch("////")
            bars[-1].set_edgecolor(EDGE)
        for bar, mean, hi in zip(bars, means, highs if error else [0] * len(means)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + (hi if error else 0) + 1.6,
                f"{mean:.0f}" if mean >= 10 else f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(models, color=INK)
    if any("DeepSeek" in label for label in models):
        ax.tick_params(axis="x", labelsize=7.5)


def _legend(fig) -> None:
    handles = [
        Patch(facecolor=c, edgecolor=EDGE, linewidth=0.6, label=name)
        for name, c in zip(DOCS, COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.1,
        handleheight=0.8,
        columnspacing=1.4,
    )


def tmdb_metrics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.55), sharey=True)
    _grouped(axes[0], TMDB_MODELS, TMDB_EXEC, error=True, hatch_last=False)
    _grouped(axes[1], TMDB_MODELS, TMDB_PATH, error=True, hatch_last=False)
    for ax, title in zip(axes, ("(a) Execution-valid CP", "(b) Ordered-path CP")):
        _axis(ax)
        ax.set_title(title, loc="left", pad=4, color=INK, fontweight="bold")
    axes[0].set_ylim(0, 108)
    axes[1].set_ylim(0, 108)
    _legend(fig)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.90))
    fig.savefig(ROOT / "results_tmdb.pdf")
    fig.savefig(ROOT / "results_tmdb.png")
    plt.close(fig)


def placement() -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    labels = [row[0] for row in PLACEMENT]
    values = [row[1] for row in PLACEMENT]
    notes = [row[2] for row in PLACEMENT]
    colors = (INITIAL, INITIAL, OURS)
    bars = ax.bar(
        np.arange(3),
        values,
        width=0.62,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.6,
        zorder=3,
    )
    _axis(ax)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, color=INK)
    ax.set_title("Fill placement, Ling, one seed (path CP)", loc="left", pad=4, color=INK, fontweight="bold")
    ax.set_ylim(0, 108)
    for bar, value, note in zip(bars, values, notes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.8,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
        if note:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value / 2,
                note,
                ha="center",
                va="center",
                fontsize=7,
                color=INK,
            )
    fig.tight_layout()
    fig.savefig(ROOT / "results_placement.pdf")
    fig.savefig(ROOT / "results_placement.png")
    plt.close(fig)


def main() -> None:
    _rc()
    tmdb_metrics()
    placement()


if __name__ == "__main__":
    main()
