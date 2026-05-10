# SalesBench eval matrix

Five eval cells for the publication's headline comparison + buyer-prompt
ablation. Each cell is launched as a single `prime rl run` with
`max_steps = checkpoint_step + 1`, which performs exactly **one rollout
pass** (128 episodes) on the configured (model, buyer-variant)
combination. No actual training happens — the trainer is just used as
the cheapest way to run a parallel eval batch.

## Cells

| File | Model | Buyer variant | Purpose |
|---|---|---|---|
| `eval-untrained-default.toml` | Qwen3.5-2B (no ckpt) | default | Headline baseline |
| `eval-trained-default.toml` | Qwen3.5-2B + v44 ckpt | default | Headline trained number |
| `eval-trained-skeptical.toml` | Qwen3.5-2B + v44 ckpt | skeptical | Ablation: hard buyer |
| `eval-trained-impulsive.toml` | Qwen3.5-2B + v44 ckpt | impulsive | Ablation: easy buyer |
| `eval-trained-analytical.toml` | Qwen3.5-2B + v44 ckpt | analytical | Ablation: numerical buyer |

All five use the same env config (50 leads, 50h budget, fixed seed) so
results are directly comparable. The only differences are:
- the model (untrained vs trained checkpoint)
- the `buyer_prompt_variant` env arg

## Why 50 leads not 100

Original target was 100-lead eval, but episode wall clock scales linearly
with leads (~5 min per 50-lead episode → ~10 min per 100-lead). 50 leads
keeps the eval matrix doable in hours instead of days while still being
2.5× the trained scale (v44 = 20 leads), so the generalization claim
holds: "trained on 20, evaluated on 50."

## How to run

After v44 finishes, fill in the v44 checkpoint ID into the four trained
cells (search/replace `<V44_CKPT_ID>`). Then:

```bash
bash tools/run_eval_matrix.sh
```

The script launches all 5 cells (sequentially or in parallel — see
script flag) and writes run IDs to `tools/results/eval_matrix_runs.txt`.

After all cells complete:

```bash
python tools/aggregate_eval_results.py tools/results/eval_matrix_runs.txt
```

This pulls metrics for each cell and writes a publication-ready JSON
summary to `tools/results/eval_matrix_summary.json` plus a Markdown
table for direct inclusion in the blog.

## Validation

Before running the real matrix on v44, validate the harness against the
v42 checkpoint (4-lead trained model). See `tools/validate_eval_harness.sh`
which runs a single small eval cell (8 leads, 16 episodes) to confirm
the whole pipeline works end-to-end.

## Cost & time estimate

- 5 cells × 128 episodes × ~5 min/episode (50 leads) = ~50 min/cell at
  full parallelism on Prime infra
- Sequential: ~4-5 hours total wall clock
- Parallel (if Prime allows concurrent runs on 2B): ~50 min
- Cost: ~$50-150 total (eval is much cheaper than training compute)
