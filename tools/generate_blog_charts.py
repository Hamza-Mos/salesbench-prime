"""Generate publication-quality charts from the eval matrix summary.

Reads tools/results/eval_matrix_summary.json and produces 4 PNG charts
in blog/charts/ for direct embedding in the blog markdown.

Charts:
  1. hero_comparison.png       : Untrained vs trained: conv per lead %
  2. buyer_ablation.png        : 4 buyer variants: reward + conv + MRR side by side
  3. metric_breakdown.png      : Untrained vs trained: 5-metric horizontal comparison
  4. training_economics.png    : Cost & wall clock per curriculum stage

Run: uv run --python .venv/bin/python tools/generate_blog_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Brand-friendly styling
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
rcParams["font.size"] = 11
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.25
rcParams["grid.linestyle"] = "--"
rcParams["axes.axisbelow"] = True
rcParams["figure.dpi"] = 100
rcParams["savefig.dpi"] = 160
rcParams["savefig.bbox"] = "tight"
rcParams["savefig.facecolor"] = "white"

# Palette
GRAY = "#9CA3AF"
INK = "#111827"
ACCENT = "#2563EB"     # blue 600
GOOD = "#10B981"       # emerald 500
WARN = "#F59E0B"       # amber 500
COOL = "#06B6D4"       # cyan 500
PINK = "#EC4899"       # pink 500

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "blog" / "charts"
OUT.mkdir(parents=True, exist_ok=True)


def load_results() -> dict[str, dict]:
    data = json.loads((ROOT / "tools/results/eval_matrix_summary.json").read_text())
    return {row["cell"]: row for row in data}


def chart_hero(results: dict[str, dict]) -> None:
    """Hero chart: per-lead conversion rate, 5 bars (untrained + 4 trained variants)."""
    cells = [
        ("eval-untrained-default", "Untrained\nQwen3.5-2B", GRAY),
        ("eval-trained-default", "Trained\n(default buyer)", ACCENT),
        ("eval-trained-skeptical", "Trained\n(skeptical buyer)", COOL),
        ("eval-trained-impulsive", "Trained\n(impulsive buyer)", GOOD),
        ("eval-trained-analytical", "Trained\n(analytical buyer)", PINK),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    values = []
    colors = []
    for cell_name, label, color in cells:
        r = results[cell_name]
        conv = r["conv_per_ep"] / 50.0 * 100  # to percent
        labels.append(label)
        values.append(conv)
        colors.append(color)

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
    # Annotate each bar
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=INK,
        )

    ax.set_ylabel("Conversion rate per lead", fontsize=12, color=INK)
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    # Title via fig.suptitle for cleaner placement
    fig.suptitle(
        "SalesBench 50-lead eval: conversion per lead",
        fontsize=15, fontweight="bold", color=INK, x=0.08, y=0.96, ha="left",
    )
    fig.text(
        0.08, 0.88,
        "Untrained Qwen3.5-2B vs same model after curriculum training (2 -> 4 -> 8 -> 20 leads).\n"
        "Trained model holds 33-40% per-lead conversion across 4 buyer personalities.",
        fontsize=10.5, color="#4B5563",
    )
    plt.subplots_adjust(top=0.78, bottom=0.13, left=0.08, right=0.97)
    plt.savefig(OUT / "hero_comparison.png")
    plt.close()
    print(f"wrote {OUT / 'hero_comparison.png'}")


def chart_buyer_ablation(results: dict[str, dict]) -> None:
    """Buyer ablation: grouped bars showing reward, conv/lead, MRR across variants."""
    variants = [
        ("default", "Default", ACCENT),
        ("skeptical", "Skeptical", COOL),
        ("impulsive", "Impulsive", GOOD),
        ("analytical", "Analytical", PINK),
    ]
    metrics = [
        ("reward_mean", "Reward", lambda v: v),
        ("conv_per_ep", "Conv/lead", lambda v: v / 50),
        ("mrr_capture", "MRR capture", lambda v: v),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (mkey, mlabel, transform) in zip(axes, metrics):
        values = []
        labels = []
        colors = []
        for vkey, vlabel, color in variants:
            r = results[f"eval-trained-{vkey}"]
            values.append(transform(r[mkey]))
            labels.append(vlabel)
            colors.append(color)
        bars = ax.bar(labels, values, color=colors, width=0.65, edgecolor="white", linewidth=1.2)
        for bar, value in zip(bars, values):
            if mlabel in ("Conv/lead", "MRR capture"):
                label = f"{100*value:.1f}%"
            else:
                label = f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                label, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK,
            )
        ax.set_title(mlabel, fontsize=12, color=INK, pad=10)
        ax.set_ylim(0, max(values) * 1.22)
        if mlabel != "Reward":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(100*v)}%"))
        for tick in ax.get_xticklabels():
            tick.set_fontsize(10)

    fig.suptitle(
        "Trained model across 4 buyer personalities",
        fontsize=14, fontweight="bold", x=0.05, y=0.96, ha="left", color=INK,
    )
    fig.text(
        0.05, 0.89,
        "Spread <8pp on conv/lead, <0.17 on reward: model isn't overfit to one buyer.",
        fontsize=10.5, color="#4B5563",
    )
    plt.subplots_adjust(top=0.78, bottom=0.12, left=0.05, right=0.97, wspace=0.28)
    plt.savefig(OUT / "buyer_ablation.png")
    plt.close()
    print(f"wrote {OUT / 'buyer_ablation.png'}")


def chart_metric_breakdown(results: dict[str, dict]) -> None:
    """Multi-metric horizontal bar chart: untrained vs trained-default."""
    ut = results["eval-untrained-default"]
    tr = results["eval-trained-default"]
    rows = [
        ("Conv per lead", ut["conv_per_ep"] / 50, tr["conv_per_ep"] / 50, lambda v: f"{100*v:.1f}%"),
        ("Leads contacted", ut["leads_contacted"] / 50, tr["leads_contacted"] / 50, lambda v: f"{100*v:.1f}%"),
        ("MRR capture", ut["mrr_capture"], tr["mrr_capture"], lambda v: f"{100*v:.1f}%"),
        ("Budget utilization", ut["budget_util"], tr["budget_util"], lambda v: f"{100*v:.1f}%"),
        ("Offers proposed/ep", ut["offers_proposed"] / 50, tr["offers_proposed"] / 50, lambda v: f"{50*v:.1f}"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = list(range(len(rows)))
    bar_h = 0.36
    for i, (label, ut_v, tr_v, fmt) in enumerate(rows):
        ax.barh(i + bar_h / 2, tr_v, height=bar_h, color=ACCENT, edgecolor="white", linewidth=1.2, label="Trained" if i == 0 else None)
        ax.barh(i - bar_h / 2, ut_v, height=bar_h, color=GRAY, edgecolor="white", linewidth=1.2, label="Untrained" if i == 0 else None)
        # Value labels at bar end
        ax.text(tr_v + max(tr_v, ut_v) * 0.015, i + bar_h / 2, fmt(tr_v), va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(ut_v + max(tr_v, ut_v) * 0.015, i - bar_h / 2, fmt(ut_v), va="center", fontsize=10, color="#6B7280")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    fig.suptitle(
        "Untrained vs Trained on the 50-lead eval",
        fontsize=14, fontweight="bold", color=INK, x=0.05, y=0.97, ha="left",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.16, right=0.97)
    plt.savefig(OUT / "metric_breakdown.png")
    plt.close()
    print(f"wrote {OUT / 'metric_breakdown.png'}")


def chart_training_curriculum() -> None:
    """Curriculum economics chart: lead count + cost per stage."""
    stages = [
        ("v41", 2, "~$15", 24, "near-perfect (99% ceiling)"),
        ("v42", 4, "~$30", 6, "mastered (98% ceiling)"),
        ("v43", 8, "~$15", 1, "mastered in 2 steps"),
        ("v44", 20, "~$80", 4, "78% conv/lead at warm-start"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(stages)))
    leads = [s[1] for s in stages]
    bars = ax.bar(x, leads, color=[ACCENT, COOL, GOOD, WARN], width=0.55, edgecolor="white", linewidth=1.5)
    for bar, (name, lc, cost, hours, status) in zip(bars, stages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{lc} leads", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=INK,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{cost}\n{hours}h",
            ha="center", va="center",
            fontsize=10, color="white", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=12)
    ax.set_ylabel("Leads per episode", fontsize=11)
    ax.set_ylim(0, max(leads) * 1.18)
    fig.suptitle(
        "Curriculum stages: each warm-started from the previous",
        fontsize=14, fontweight="bold", color=INK, x=0.08, y=0.96, ha="left",
    )
    fig.text(
        0.08, 0.88,
        "Total training: ~$140 in compute, ~35 hours wall clock from v41 start to v44 stop.",
        fontsize=10.5, color="#4B5563",
    )
    plt.subplots_adjust(top=0.78, bottom=0.13, left=0.08, right=0.97)
    plt.savefig(OUT / "training_economics.png")
    plt.close()
    print(f"wrote {OUT / 'training_economics.png'}")


def main() -> None:
    results = load_results()
    chart_hero(results)
    chart_buyer_ablation(results)
    chart_metric_breakdown(results)
    chart_training_curriculum()
    print(f"\nAll 4 charts written to {OUT}")


if __name__ == "__main__":
    main()
