# SalesBench

SalesBench is a stateful, tool-use environment for training and evaluating LLM sales agents with [Prime Lab](https://docs.primeintellect.ai/) and the [verifiers](https://docs.primeintellect.ai/verifiers/overview) library.

It simulates an insurance sales pipeline with synthetic leads, a product catalog, quoting, calling, and callback scheduling. The agent must maximize converted monthly recurring premium while staying compliant (do-not-call) and avoiding invalid actions.

## What The Agent Does

- Find and inspect leads via a CRM tool
- Start exactly one active call at a time, propose offers, and end calls
- Quote plans before proposing premiums
- Schedule and manage callbacks
- Optimize for revenue while being efficient and compliant

## Tools

- `crm_search_leads`
- `crm_get_lead`
- `crm_add_note`
- `crm_pipeline_summary`
- `calendar_schedule_callback`
- `calendar_list_callbacks`
- `calling_start_call`
- `calling_propose_offer`
- `calling_end_call`
- `products_list_plans`
- `products_quote_plan`

## Rewards And Metrics

The environment uses a `verifiers.Rubric` with:

- Primary reward: converted monthly recurring revenue (MRR)
- Additional rewards/penalties: conversion rate, efficiency, invalid actions, do-not-call violations
- Logged metrics: revenue MRR, conversions, calls started, time utilization, episode done, etc.

## Setup

Prereqs:

- `uv`
- Prime CLI (`uv tool install -U prime`)
- `prime login`

From the repo root (`/Users/hamza/Desktop/salesbench-prime`):

```bash
prime lab setup
prime env install salesbench
```

## Push To Environments Hub (Private)

Find your team slug:

```bash
prime teams list
```

Then push privately under that team:

```bash
prime env push --path ./environments/salesbench --team <team-slug> -v PRIVATE
```

## Run A Local Evaluation

Run an eval (Prime Inference is used by default; configure endpoints in `configs/endpoints.py`):

```bash
prime eval run salesbench -m openai/gpt-5-nano -n 20 -r 3
```

View results:

```bash
prime eval tui
```

## Environment Arguments

These map to `salesbench.load_environment(...)`:

| Arg                 | Type          | Default   | Description                                              |
| ------------------- | ------------- | --------- | -------------------------------------------------------- |
| `split`             | `str`         | `"train"` | Dataset split to generate (`train`, `eval`, `test`).     |
| `num_examples`      | `int`         | `256`     | Generated dataset size (train).                          |
| `eval_num_examples` | `int`         | `64`      | Generated dataset size (eval).                           |
| `base_seed`         | `int`         | `42`      | Base seed used to create deterministic episodes.         |
| `seed`              | `int \| None` | `None`    | Alias for `base_seed` (some Prime commands pass `seed`). |
| `num_leads`         | `int`         | `100`     | Leads per episode.                                       |
| `work_days`         | `int`         | `10`      | Simulated work days per episode.                         |
| `hours_per_day`     | `int`         | `8`       | Simulated working hours per day.                         |
| `max_turns`         | `int`         | `10000`   | Upper bound on model turns per rollout.                  |
| `max_examples`      | `int`         | `-1`      | Optional cap after dataset generation.                   |
| `context_rewrite_threshold` | `float` | `0.80`  | Fraction of `max_seq_len` at which to trigger context summarization. |
| `context_keep_recent` | `int`       | `10`      | Number of recent messages to keep verbatim after summarization. |

Pass args via `--env-args` / `-a` as JSON:

```bash
prime eval run salesbench -m openai/gpt-5-nano -n 10 -r 1 \
  -a '{"split":"eval","base_seed":123,"num_leads":120,"work_days":5,"hours_per_day":8}'
```

## Context Summarization

In multi-turn episodes, the message history grows with each tool call and buyer response. When the prompt reaches **80%** of `max_seq_len` (set by the training infrastructure), older messages are replaced with a compact, deterministic summary built from runtime state — no LLM call required. The last 10 messages are kept verbatim to preserve the current call context.

This prevents training truncation while minimizing KV cache branching (summarization happens at most once or twice per episode, not every turn).

Example summary injected into the conversation:

```
[CONTEXT SUMMARY — previous turns compressed]
Time: 120/240 min (120 remaining) | Revenue: $250.00/mo | Conversions: 2 | Offers: 5
Calls completed (3):
  - John Smith (warm): accept — 1 offer(s), 25min
  - Jane Doe (cold): reject — 2 offer(s), 18min
  - Bob Wilson (hot): hang_up — 1 offer(s), 8min
Active call: Alice Brown — 1 offer(s) so far
Pipeline: 1 active leads, 4 contacted
```

Tune via env args: `context_rewrite_threshold` (when to trigger) and `context_keep_recent` (how many recent messages to preserve).

## Developer Smoke Test

After installing the environment (`prime env install salesbench`), you can sanity-check dataset generation locally:

If `import salesbench` fails right after install on macOS, run:

```bash
chflags nohidden .venv/lib/python*/site-packages/_salesbench.pth
```

```bash
uv run python -c "import salesbench; env=salesbench.load_environment(num_examples=2, eval_num_examples=1); print(env.get_dataset(1)[0])"
```
