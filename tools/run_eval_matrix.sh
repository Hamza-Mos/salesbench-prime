#!/usr/bin/env bash
# Launch the full eval matrix for the SalesBench publication.
#
# Usage:
#   bash tools/run_eval_matrix.sh <V44_RUN_ID> <V44_CKPT_STEP>
#
# Example:
#   bash tools/run_eval_matrix.sh nig0ix7jiznmshmq9w37wd8j 195
#
# What this does:
#   1. For each cell in configs/eval/, generate a per-cell config with the
#      v44 checkpoint ID + correct max_steps filled in.
#   2. Launch each cell as a `prime rl run` (sequentially by default;
#      pass --parallel to launch all at once).
#   3. Write run IDs to tools/results/eval_matrix_runs.txt for the
#      aggregator to consume.
#
# After all cells complete, run:
#   python tools/aggregate_eval_results.py tools/results/eval_matrix_runs.txt

set -euo pipefail

V44_RUN_ID="${1:-}"
V44_CKPT_STEP="${2:-}"
PARALLEL="${3:-sequential}"  # "parallel" or "sequential"

if [ -z "$V44_RUN_ID" ] || [ -z "$V44_CKPT_STEP" ]; then
    echo "Usage: bash tools/run_eval_matrix.sh <V44_RUN_ID> <V44_CKPT_STEP> [parallel|sequential]"
    echo
    echo "Get the v44 checkpoint id (latest READY checkpoint) via:"
    echo "  prime rl checkpoints <v44-run-id>"
    echo
    echo "V44_RUN_ID is the run ID; V44_CKPT_STEP is the step number of the"
    echo "checkpoint to evaluate (e.g. 295). The script computes max_steps =$V44_CKPT_STEP + 1."
    exit 1
fi

# Step 1: get the checkpoint ID for the chosen step
echo "Looking up checkpoint at step $V44_CKPT_STEP for run $V44_RUN_ID..."
CKPT_TABLE=$(prime rl checkpoints "$V44_RUN_ID" 2>&1)
echo "$CKPT_TABLE"
echo
CKPT_ID=$(echo "$CKPT_TABLE" | awk -v step="$V44_CKPT_STEP" '
    /^│/ && $4 == step { print $2; exit }
')

if [ -z "$CKPT_ID" ]; then
    echo "ERROR: could not find checkpoint at step $V44_CKPT_STEP for run $V44_RUN_ID"
    echo "Verify the step exists in the table above and is READY."
    exit 1
fi
echo "Found checkpoint $CKPT_ID at step $V44_CKPT_STEP"
echo

MAX_STEPS=$((V44_CKPT_STEP + 1))
echo "Will set max_steps = $MAX_STEPS for trained-cell evals"
echo

# Step 2: prepare per-cell configs
WORK_DIR=$(mktemp -d)
echo "Working dir: $WORK_DIR"

declare -a CELLS=(
    "eval-untrained-default"
    "eval-trained-default"
    "eval-trained-skeptical"
    "eval-trained-impulsive"
    "eval-trained-analytical"
)

for cell in "${CELLS[@]}"; do
    src="configs/eval/${cell}.toml"
    dst="$WORK_DIR/${cell}.toml"
    if [[ "$cell" == "eval-untrained-default" ]]; then
        # No checkpoint; use as-is
        cp "$src" "$dst"
    else
        # Replace checkpoint + max_steps placeholders
        sed -e "s|<V44_CKPT_ID>|$CKPT_ID|" \
            -e "s|max_steps = 999|max_steps = $MAX_STEPS|" \
            "$src" > "$dst"
    fi
done
echo "Generated 5 per-cell configs in $WORK_DIR"
echo

# Step 3: launch each cell
mkdir -p tools/results
RESULTS_FILE="tools/results/eval_matrix_runs.txt"
> "$RESULTS_FILE"  # truncate

launch_one() {
    local cell="$1"
    local cfg="$WORK_DIR/${cell}.toml"
    echo "=== Launching $cell ==="
    local output
    output=$(prime rl run "$cfg" 2>&1 | tee /dev/stderr)
    local run_id
    run_id=$(echo "$output" | grep -oE 'training/[a-z0-9]+' | head -1 | cut -d'/' -f2)
    if [ -n "$run_id" ]; then
        echo "$cell $run_id" >> "$RESULTS_FILE"
        echo "→ $cell run_id: $run_id"
    else
        echo "WARN: failed to extract run_id for $cell"
    fi
    echo
}

if [ "$PARALLEL" == "parallel" ]; then
    echo "Launching all 5 cells in PARALLEL..."
    for cell in "${CELLS[@]}"; do
        launch_one "$cell" &
    done
    wait
else
    echo "Launching cells SEQUENTIALLY..."
    for cell in "${CELLS[@]}"; do
        launch_one "$cell"
    done
fi

echo
echo "All cells launched. Run IDs written to $RESULTS_FILE:"
cat "$RESULTS_FILE"
echo
echo "Wait for all runs to complete (status COMPLETED), then run:"
echo "  python tools/aggregate_eval_results.py $RESULTS_FILE"
