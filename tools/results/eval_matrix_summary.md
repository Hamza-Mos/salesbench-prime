## Headline result

On the SalesBench 50-lead eval (n=128 episodes), the trained Qwen3.5-2B (curriculum: 2→4→8→20 leads) achieves:

- **Reward**: 0.490 vs untrained -0.035 (inf× improvement)
- **Conversions/episode**: 17.6/50 vs 1.0/50 (35% vs 2% per lead)
- **MRR capture**: 40.8% vs 0.2%

## Buyer-prompt ablation

Evaluating the trained model against 4 different buyer decision-making styles (each 128 episodes, 50 leads):

| Buyer variant | Reward | Conv/ep | MRR | Δ vs default |
|---|---|---|---|---|
| default | 0.490 | 17.6/50 | 40.8% | +0.000 |
| skeptical | 0.377 | 18.5/50 | 31.9% | -0.113 |
| impulsive | 0.549 | 19.7/50 | 45.1% | +0.060 |
| analytical | 0.445 | 16.4/50 | 37.4% | -0.045 |

## Full metrics table

| Cell | Reward | Conv/ep | MRR cap | Budget | Offers | Turns | Buyer LLM |
|---|---|---|---|---|---|---|---|
| eval-untrained-default | -0.035 | 1.023 | 0.002 | 0.002 | 2.648 | 128.133 | 32.148 |
| eval-trained-default | 0.490 | 17.552 | 0.408 | 0.339 | 5.042 | 163.068 | 68.510 |
| eval-trained-skeptical | 0.377 | 18.500 | 0.319 | 0.265 | 6.412 | 299.177 | 168.104 |
| eval-trained-impulsive | 0.549 | 19.740 | 0.451 | 0.381 | 4.495 | 156.302 | 52.135 |
| eval-trained-analytical | 0.445 | 16.406 | 0.374 | 0.316 | 6.880 | 198.068 | 109.516 |
