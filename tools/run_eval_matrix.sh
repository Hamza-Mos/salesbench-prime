#!/usr/bin/env bash
# Launch the SalesBench buyer-variant eval matrix for any Prime/OpenAI-compatible model.
#
# Usage:
#   bash tools/run_eval_matrix.sh [MODEL]
#
# Examples:
#   bash tools/run_eval_matrix.sh Qwen/Qwen3.5-2B
#   bash tools/run_eval_matrix.sh Qwen/Qwen3.5-2B:gjgpwgdb5sh7y2d6epmveh23
#   bash tools/run_eval_matrix.sh openai/gpt-5-mini

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-2B}"
SAFE_MODEL_NAME="$(echo "$MODEL" | tr '/:' '__')"
OUT_DIR="tools/results/prime-eval/${SAFE_MODEL_NAME}"

mkdir -p "$OUT_DIR"

COMMON_ARGS='"split":"eval","num_examples":128,"num_leads":100,"total_hours":50,"context_rewrite_threshold":0.85,"context_keep_recent":10,"context_max_seq_len":16000'

run_cell() {
    local buyer="$1" stagger="$2"
    local log="$OUT_DIR/cell-${buyer}.log"
    echo "=== launching ${buyer} buyer on ${MODEL} ==="
    (
        sleep "$stagger"
        prime eval run salesbench \
            -m "$MODEL" \
            -n 128 -r 1 \
            --max-concurrent 16 \
            --save-results \
            --env-args "{${COMMON_ARGS},\"buyer_prompt_variant\":\"${buyer}\"}" \
            > "$log" 2>&1
    ) &
    echo "  pid=$! log=$log"
}

# Stagger launches to avoid local multiprocessing startup races.
run_cell "default" 0
run_cell "skeptical" 8
run_cell "impulsive" 16
run_cell "analytical" 24

echo
echo "All 4 buyer-variant cells launched for: $MODEL"
echo "Logs: $OUT_DIR"
echo "Monitor with: tail -f $OUT_DIR/cell-*.log"
echo
wait

echo
echo "=================================="
echo "Eval complete: $MODEL"
echo "=================================="
printf "%-15s %10s %10s %10s\n" "Buyer" "Reward" "Conv/lead" "MRR"
for buyer in default skeptical impulsive analytical; do
    log="$OUT_DIR/cell-${buyer}.log"
    rew=$(grep "^reward: avg" "$log" 2>/dev/null | head -1 | grep -oE "avg - [-0-9.]+" | awk '{print $3}')
    conv=$(grep "^reward_conversion_rate: avg" "$log" 2>/dev/null | head -1 | grep -oE "avg - [-0-9.]+" | awk '{print $3}')
    mrr=$(grep "^reward_revenue_mrr: avg" "$log" 2>/dev/null | head -1 | grep -oE "avg - [-0-9.]+" | awk '{print $3}')
    printf "%-15s %10s %10s %10s\n" "$buyer" "${rew:-FAIL}" "${conv:-FAIL}" "${mrr:-FAIL}"
done
