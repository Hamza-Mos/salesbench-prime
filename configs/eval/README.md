# SalesBench Eval Matrix

The canonical publication eval uses `prime eval run` through:

```bash
bash tools/run_eval_matrix.sh
bash tools/run_eval_matrix.sh Qwen/Qwen3.5-2B:<ADAPTER_ID>
```

The script defaults to `Qwen/Qwen3.5-2B`. It also accepts any Prime/OpenAI-compatible model string and evaluates it against four buyer variants:

- default
- skeptical
- impulsive
- analytical

Each cell uses 128 episodes, 100 leads, 50 simulated hours, and context summarization args. The saved apples-to-apples publication matrix is stored in:

```text
tools/results/prime-eval/summary.json
```

## About These TOMLs

The TOML files in this directory are retained as cloud `prime rl run` fallback configs for one-step evals from a checkpoint. They are not the source of the final blog numbers. The final blog numbers come from the `prime eval run` flow above, run once on the base model and once on the deployed trained adapter.

If you do use these fallback configs, replace `<TRAINED_CKPT_ID>` with the trained checkpoint ID and set `max_steps` greater than the checkpoint step.
