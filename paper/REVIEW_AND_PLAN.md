# SalesBench WAB Paper Audit Note

Target venue: Workshop on AI Agent Behavior (WAB) at COLM 2026. Behavior-first framing.

## Figures and tables (all grounded in real logs)

- **Figure 1** (harness): clean three-party orchestration diagram (seller <-> orchestrator <-> {runtime, buyer LM}).
- **Figure 2** (model performance, S4.2): native pgfplots, per-lead conversion at 50 vs 100 leads across the 5 eval
  cells. Source: `tools/results/prime-eval/summary.json`.
- **Figure 3** (traces, S4.3): three real rollout phases (broken interface -> clean tools / zero closes -> close).
  Captions tie each phase to the Table 1/2 aggregate numbers.
- **Figure 4** (brittleness, S4.4): fraction of each budget (time / coverage / mistakes) consumed at termination.
  Source: Table 2 means.
- **Table 1**: 50/100-lead evaluation matrix (trained vs untrained, 4 buyer variants), **64 episodes per cell**,
  with **real +/-SEM** (SD/sqrt(64)) on reward and conv/lead.
- **Table 2**: behavioral decomposition at 100 leads (the brittleness evidence).
- **Table 3** (appendix A.7): training hyperparameters (from configs/curriculum).
- **Table 4** (appendix A.9): preliminary external-model reward sweep. Clearly labeled n=1 single-seed,
  OpenRouter harness, not a leaderboard. Included per Hamza's explicit 2026-05-30 decision.

## Grounding rules

- No number appears that is not traceable to code, eval logs, the prime-eval summary, or a paper table.
- **n = 64 episodes per cell** (run logs: "64 total rollouts"; the 128 is the dataset/split size). SEM = SD/sqrt(64).
  Real per-episode std is in tools/results/prime-eval/cell-*.log.
- WAB page limit is a STRICT 9 pages of main text (refs + appendix excluded). Keep main text <= page 9.
- The external-model sweep is preliminary by construction: each cell is one episode and the serving harness matters
  (GPT-5.5: 0.116 OpenRouter vs 0.316 direct). It lives in the appendix with explicit caveats and no ranking claim.
- No em dashes anywhere in the source.

## Compile status

- `latexmk -g -pdf -interaction=nonstopmode -halt-on-error salesbench_wab.tex` succeeds. 15 pages, 0 overfull boxes,
  all references resolve.
- All four figures are native LaTeX (tikz / pgfplots / listings); no `\includegraphics`. The leftover
  `figures/scaling_50_vs_100.png` is unused and can be deleted for a clean release.
