"""Paper figures: CP% vs evaluation-set size and final Wilson intervals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tooldoc_nir.restbench_report import load_traces, report_from_traces

ROOT = Path(__file__).resolve().parents[1]
PREPRINT = ROOT / "artifacts" / "preprint"
DOCS = Path(r"C:\Users\banana0081\Desktop\НИР\документы")
LING = PREPRINT / "restbench_tmdb_ling_ci.json"


def _curve(report: dict, name: str) -> tuple[list[int], list[float], list[float], list[float]]:
    points = report["conditions"][name]["learning_curve"]
    ns = [p["n"] for p in points]
    cps = [100.0 * p["cp"] for p in points]
    lows = [100.0 * p["cp_ci95"][0] for p in points]
    highs = [100.0 * p["cp_ci95"][1] for p in points]
    return ns, cps, lows, highs


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    PREPRINT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    for folder in (PREPRINT, DOCS):
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(folder / f"{stem}.png", bbox_inches="tight")


def plot_cp_vs_n(report: dict, title: str, stem: str, caption: str) -> None:
    series = [
        ("DFSDT", "#2563eb", "-"),
        ("DRAFT", "#d97706", "-"),
        ("Ours", "#059669", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xmax = 10
    y_lo, y_hi = 100.0, 0.0
    for name, color, ls in series:
        if name not in report["conditions"]:
            continue
        ns, cps, lows, highs = _curve(report, name)
        xmax = max(xmax, max(ns))
        y_lo = min(y_lo, min(lows))
        y_hi = max(y_hi, max(highs))
        ax.fill_between(ns, lows, highs, color=color, alpha=0.12, linewidth=0)
        ax.plot(ns, cps, color=color, linestyle=ls, linewidth=2.0, marker="o", markersize=4, label=name)
    ax.set_xlabel("Evaluation set size $n$ (queries)")
    ax.set_ylabel("Correct Path (%)")
    ax.set_title(title)
    ax.set_ylim(max(0.0, y_lo - 4), min(105.0, y_hi + 3))
    ax.set_xlim(min(10, xmax), xmax)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.0,
        -0.18,
        caption,
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


def plot_final_ci(report: dict, stem: str) -> None:
    order = [name for name in ("DFSDT", "DRAFT", "Ours") if name in report["conditions"]]
    means = [100.0 * report["conditions"][n]["cp"] for n in order]
    lows = [100.0 * report["conditions"][n]["cp_ci95"][0] for n in order]
    highs = [100.0 * report["conditions"][n]["cp_ci95"][1] for n in order]
    colors = {"DFSDT": "#2563eb", "DRAFT": "#d97706", "Ours": "#059669"}
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(order))
    ax.bar(x, means, color=[colors[n] for n in order], width=0.62)
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="none",
        ecolor="black",
        capsize=4,
        elinewidth=1.2,
    )
    ax.set_xticks(x, order)
    ax.set_ylabel("Correct Path (%)")
    ax.set_title("RestBench-TMDB CP% with Wilson 95% CI")
    ax.set_ylim(70, 102)
    for i, (mean, low, high) in enumerate(zip(means, lows, highs)):
        ax.text(i, high + 0.6, f"{mean:.0f}  [{low:.0f}, {high:.0f}]", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.0,
        -0.18,
        "n = 100 queries, Ling-3.0-Flash, seed 0. Interval is over queries, not over random seeds.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout()
    save(fig, stem)
    plt.close(fig)


def main() -> int:
    _style()
    report = json.loads(LING.read_text(encoding="utf-8"))
    plot_cp_vs_n(
        report,
        "Correct Path vs evaluation-set size",
        "fig_tmdb_ling_cp_vs_n",
        "Wilson 95% interval on the query prefix. RestBench-TMDB, Ling-3.0-Flash, seed 0. ReAct omitted.",
    )
    plot_final_ci(report, "fig_tmdb_ling_cp_ci")

    spotify_path = ROOT / "artifacts" / "results" / "restbench_spotify_ling" / "traces.jsonl"
    if spotify_path.exists():
        spotify = report_from_traces({"spotify": load_traces(spotify_path)})
        PREPRINT.joinpath("restbench_spotify_ling_ci.json").write_text(
            json.dumps(spotify, indent=2) + "\n", encoding="utf-8"
        )
        plot_cp_vs_n(
            spotify,
            "RestBench-Spotify Correct Path vs evaluation-set size",
            "fig_spotify_ling_cp_vs_n",
            "Wilson 95% interval on the query prefix. RestBench-Spotify, Ling-3.0-Flash, seed 0.",
        )
    print("wrote figures to", PREPRINT, "and", DOCS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
