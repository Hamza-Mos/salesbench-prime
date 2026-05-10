"""Aggregate eval matrix results into a publication-ready summary.

Input: a TSV/text file mapping cell names to Prime RL run IDs (one per line),
e.g.:
    eval-untrained-default r90m00fyzu2o38wksjs24co4
    eval-trained-default   z2pxmvmgdujq6upcgv63uta4
    ...

Output:
- tools/results/eval_matrix_summary.json    full structured data
- tools/results/eval_matrix_summary.md      markdown table for blog inclusion

Usage:
    python tools/aggregate_eval_results.py tools/results/eval_matrix_runs.txt
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


# Headline metrics to extract per cell.
# Maps display name → metric key in the run.
METRICS = {
    "reward_mean": "reward/all/mean",
    "reward_min": "reward/all/min",
    "reward_max": "reward/all/max",
    "conv_per_ep": "metrics/salesbench/salesbench/metric_conversions",
    "leads_contacted": "metrics/salesbench/salesbench/metric_leads_contacted",
    "mrr_capture": "metrics/salesbench/salesbench/reward_revenue_mrr",
    "budget_util": "metrics/salesbench/salesbench/reward_budget_utilization",
    "offers_proposed": "metrics/salesbench/salesbench/calling_propose_offer_calls",
    "offers_accepted": "metrics/salesbench/salesbench/metric_offers_accepted",
    "offers_rejected": "metrics/salesbench/salesbench/metric_offers_rejected",
    "num_turns": "metrics/salesbench/salesbench/num_turns",
    "invalid_actions": "metrics/salesbench/salesbench/metric_invalid_actions",
    "buyer_calls": "metrics/salesbench/salesbench/metric_buyer_llm_call_count",
    "time_used_min": "metrics/salesbench/salesbench/metric_time_used_minutes",
}


def fetch_run_metrics(run_id: str) -> dict:
    """Pull all-step metrics for a run via prime rl metrics."""
    result = subprocess.run(
        ["prime", "rl", "metrics", run_id],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prime rl metrics {run_id} failed: {result.stderr}")
    payload = json.loads(result.stdout)
    return payload.get("metrics", [])


def summarize_cell(cell_name: str, run_id: str) -> dict:
    """Pull metrics for a cell's run and compute summary stats.

    For an eval-as-1-step-train run, there should be exactly one step
    record (the eval batch). Each metric is a *mean over the batch's 128
    rollouts*, so it's already an aggregate stat.
    """
    print(f"  Fetching metrics for {cell_name} ({run_id})...")
    records = fetch_run_metrics(run_id)
    if not records:
        return {"cell": cell_name, "run_id": run_id, "error": "no metrics records"}

    # Use the LAST record (in case the run did multiple steps somehow)
    final = records[-1]
    summary: dict[str, object] = {
        "cell": cell_name,
        "run_id": run_id,
        "step": int(final.get("step", -1)),
        "n_rollouts": 128,  # prime rl batch_size
    }
    for display_name, key in METRICS.items():
        value = final.get(key)
        summary[display_name] = round(float(value), 4) if value is not None else None
    return summary


def render_markdown_table(summaries: list[dict]) -> str:
    """Render a publication-ready markdown table of the eval matrix."""
    if not summaries:
        return "(no data)"

    cols = [
        ("cell", "Cell"),
        ("reward_mean", "Reward"),
        ("conv_per_ep", "Conv/ep"),
        ("mrr_capture", "MRR cap"),
        ("budget_util", "Budget"),
        ("offers_proposed", "Offers"),
        ("num_turns", "Turns"),
        ("buyer_calls", "Buyer LLM"),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = [header, sep]
    for s in summaries:
        if "error" in s:
            rows.append(f"| {s['cell']} | ERROR: {s['error']} | | | | | | |")
            continue
        row_vals = []
        for key, _ in cols:
            v = s.get(key)
            if isinstance(v, float):
                row_vals.append(f"{v:.3f}")
            elif v is None:
                row_vals.append("?")
            else:
                row_vals.append(str(v))
        rows.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(rows)


def render_blog_summary(summaries: list[dict]) -> str:
    """Render a blog-friendly comparison block."""
    by_cell = {s["cell"]: s for s in summaries if "error" not in s}
    untrained = by_cell.get("eval-untrained-default")
    trained = by_cell.get("eval-trained-default")

    lines = ["## Headline result\n"]
    if untrained and trained:
        ut_r = untrained.get("reward_mean", 0) or 0
        tr_r = trained.get("reward_mean", 0) or 0
        ut_c = untrained.get("conv_per_ep", 0) or 0
        tr_c = trained.get("conv_per_ep", 0) or 0
        ut_m = untrained.get("mrr_capture", 0) or 0
        tr_m = trained.get("mrr_capture", 0) or 0
        improvement = (tr_r / ut_r) if ut_r > 0 else float("inf")
        lines.append(
            f"On the SalesBench 50-lead eval (n=128 episodes), the trained "
            f"Qwen3.5-2B (curriculum: 2→4→8→20 leads) achieves:\n"
        )
        lines.append(f"- **Reward**: {tr_r:.3f} vs untrained {ut_r:.3f} "
                    f"({improvement:.1f}× improvement)")
        lines.append(f"- **Conversions/episode**: {tr_c:.1f}/50 vs {ut_c:.1f}/50 "
                    f"({100*tr_c/50:.0f}% vs {100*ut_c/50:.0f}% per lead)")
        lines.append(f"- **MRR capture**: {tr_m:.1%} vs {ut_m:.1%}")
    else:
        lines.append("(headline cells not found in input)")

    lines.append("\n## Buyer-prompt ablation\n")
    lines.append("Evaluating the trained model against 4 different buyer "
                 "decision-making styles (each 128 episodes, 50 leads):\n")
    ablation_rows = ["| Buyer variant | Reward | Conv/ep | MRR | Δ vs default |",
                     "|---|---|---|---|---|"]
    if trained:
        base_r = trained.get("reward_mean", 0) or 0
        for variant in ["default", "skeptical", "impulsive", "analytical"]:
            cell = by_cell.get(f"eval-trained-{variant}")
            if not cell:
                ablation_rows.append(f"| {variant} | (missing) | | | |")
                continue
            r = cell.get("reward_mean", 0) or 0
            c = cell.get("conv_per_ep", 0) or 0
            m = cell.get("mrr_capture", 0) or 0
            delta = r - base_r
            ablation_rows.append(
                f"| {variant} | {r:.3f} | {c:.1f}/50 | {m:.1%} | "
                f"{delta:+.3f} |"
            )
    lines.extend(ablation_rows)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    runs_file = Path(sys.argv[1])
    if not runs_file.exists():
        print(f"ERROR: {runs_file} not found")
        sys.exit(1)

    pairs: list[tuple[str, str]] = []
    for line in runs_file.read_text().strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            print(f"WARN: skipping malformed line: {line}")
            continue
        pairs.append((parts[0], parts[1]))

    print(f"Aggregating {len(pairs)} cells...\n")
    summaries = [summarize_cell(name, run_id) for name, run_id in pairs]

    out_dir = Path("tools/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "eval_matrix_summary.json"
    json_out.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote JSON summary: {json_out}")

    md_out = out_dir / "eval_matrix_summary.md"
    md_table = render_markdown_table(summaries)
    blog_summary = render_blog_summary(summaries)
    md_out.write_text(f"{blog_summary}\n\n## Full metrics table\n\n{md_table}\n")
    print(f"Wrote markdown summary: {md_out}\n")

    print("=" * 70)
    print(blog_summary)
    print()
    print("=" * 70)
    print(md_table)


if __name__ == "__main__":
    main()
