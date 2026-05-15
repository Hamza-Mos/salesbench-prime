#!/usr/bin/env bash
# Launch the full SalesBench eval matrix via `prime eval run`.
#
# Five cells: untrained baseline (base Qwen3.5-2B) + trained model against
# 4 buyer personalities (default / skeptical / impulsive / analytical).
# Each cell: 128 episodes, 100 leads, 50 simulated hours, fixed seed.
#
# Usage:
#   bash tools/run_eval_matrix.sh <ADAPTER_ID>
#
# Example:
#   bash tools/run_eval_matrix.sh gjgpwgdb5sh7y2d6epmveh23
#
# Find your adapter ID with `prime deployments list -o json` (it's the
# READY adapter whose rft_run_id matches your trained-curriculum run).
# Deploy it first with `prime deployments create <ADAPTER_ID>`.

set -euo pipefail

ADAPTER_ID="${1:-}"
if [ -z "$ADAPTER_ID" ]; then
    echo "Usage: bash tools/run_eval_matrix.sh <ADAPTER_ID>"
    echo
    echo "Find your adapter:  prime deployments list -o json"
    echo "Deploy:             prime deployments create <ADAPTER_ID>"
    exit 1
fi

mkdir -p tools/results/prime-eval
TRAINED_MODEL="Qwen/Qwen3.5-2B:${ADAPTER_ID}"
BASE_MODEL="Qwen/Qwen3.5-2B"
COMMON_ARGS='"split":"eval","num_examples":128,"num_leads":100,"total_hours":50,"context_rewrite_threshold":0.85,"context_keep_recent":10,"context_max_seq_len":16000'

run_cell() {
    local name="$1" model="$2" buyer="$3" stagger="$4"
    local log="tools/results/prime-eval/cell-${name}.log"
    echo "=== launching $name ==="
    (
        sleep "$stagger"
        prime eval run salesbench \
            -m "$model" \
            -n 128 -r 1 \
            --max-concurrent 16 \
            --save-results \
            --env-args "{${COMMON_ARGS},\"buyer_prompt_variant\":\"${buyer}\"}" \
            > "$log" 2>&1
    ) &
    echo "  pid=$! log=$log"
}

# Stagger launches by 8s to avoid the multiprocessing race on parallel
# `prime eval run` startup that can hit module-loading conflicts.
run_cell "untrained-default"  "$BASE_MODEL"    "default"    0
run_cell "trained-default"    "$TRAINED_MODEL" "default"    8
run_cell "trained-skeptical"  "$TRAINED_MODEL" "skeptical"  16
run_cell "trained-impulsive"  "$TRAINED_MODEL" "impulsive"  24
run_cell "trained-analytical" "$TRAINED_MODEL" "analytical" 32

echo
echo "All 5 cells launched in parallel (staggered start)."
echo "Each cell runs locally as a Python process and calls Prime Inference."
echo
echo "Monitor progress with:"
echo "  tail -f tools/results/prime-eval/cell-*.log"
echo
echo "Wait for all to finish (~1-2 hours wall clock)..."
echo
wait
echo
echo "=================================="
echo "All cells complete. Headline numbers:"
echo "=================================="
printf "%-25s %10s %10s %10s\n" "Cell" "Reward" "Conv/lead" "MRR"
for cell in untrained-default trained-default trained-skeptical trained-impulsive trained-analytical; do
    log="tools/results/prime-eval/cell-${cell}.log"
    rew=$(grep "^reward: avg" "$log" 2>/dev/null | head -1 | grep -oE "avg - [-0-9.]+" | awk '{print $3}')
    conv=$(grep "^reward_conversion_rate: avg" "$log" 2>/dev/null | head -1 | grep -oE "avg - [-0-9.]+" | awk '{print $3}')
    mrr=$(grep "^reward_revenue_mrr: avg" "$log" 2>/dev/null | head -1 | grep -oE "avg - [-0-9.]+" | awk '{print $3}')
    printf "%-25s %10s %10s %10s\n" "$cell" "${rew:-FAIL}" "${conv:-FAIL}" "${mrr:-FAIL}"
done
