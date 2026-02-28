# SalesBench Prime RL Harness

Standalone, Prime-stack-first harness for training and evaluating sales agents with real model inference.

## What is SalesBench?

SalesBench is a simulated insurance sales environment built on Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework. An LLM agent is dropped into a sales scenario with a pool of leads (prospective buyers) and a time budget, and must maximize monthly recurring revenue (MRR) by searching the CRM, calling leads, quoting insurance plans, and closing deals.

Each episode is fully deterministic given a seed — the same seed always produces identical leads, buyer personalities, and decision outcomes, enabling reproducible evaluation.

## Quick Start

```bash
# install the environment
prime env install salesbench/salesbench

# run an eval (1 episode, 1 attempt, using gpt-4.1-mini)
prime eval run salesbench/salesbench -m openai/gpt-4.1-mini -n 1 -r 1

# more robust eval (10 episodes, 3 attempts each = 30 total runs)
prime eval run salesbench/salesbench -m openai/gpt-4.1-mini -n 10 -r 3
```

### Key flags

| Flag | Name | Default | Meaning |
|------|------|---------|---------|
| `-n` | `num_examples` | 5 | Number of **episodes** to run. Each episode is a full sales scenario with ~100 leads and a time budget. |
| `-r` | `rollouts_per_example` | 3 | Number of **independent attempts** per episode. Same seed/leads, fresh model trajectory each time. |
| `-v` | `verbose` | off | Print detailed logs to the terminal. |
| `-s` | `save_results` | off | Save results to disk locally. |

Total simulations = `n × r`. Results are uploaded to Prime Hub by default (use `--skip-upload` to disable).

## Episode Structure

Each episode gives the agent:

- **~100 leads** — deterministically generated prospects with demographics (age, income, state), behavioral attributes (need, trust, price sensitivity), and a monthly budget.
- **Time budget** — 10 work days × 8 hours = 4,800 simulated minutes. Each tool call costs time (e.g. `propose_offer` = 4 min, `start_call` = 1 min).
- **Insurance product catalog** — Term, Whole, Universal Life, and Disability Income plans with deterministic premium pricing.

The agent interacts via tools: `crm_search_leads`, `crm_get_lead`, `calling_start_call`, `products_quote_plan`, `calling_propose_offer`, `calling_end_call`, `calendar_schedule_callback`, etc.

An episode ends when time runs out, all leads are exhausted (converted/DNC/max calls), or too many invalid actions occur.

## Scoring

The primary reward is **total converted MRR** — the sum of monthly premiums from all successful sales. Additional metrics are tracked but not weighted into the score:

- Conversion rate, call efficiency
- DNC (do-not-call) violations
- Invalid action count, time utilization
