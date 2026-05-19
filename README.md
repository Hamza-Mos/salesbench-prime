# SalesBench Prime RL Harness

Standalone, Prime-stack-first harness for training and evaluating sales agents with real model inference.

## What is SalesBench?

SalesBench is a simulated insurance sales environment built on Prime Intellect's [verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework. An LLM agent is dropped into a sales scenario with a pool of leads (prospective buyers) and a time budget, and must maximize monthly recurring revenue (MRR) by searching the CRM, calling leads, quoting insurance plans, and closing deals.

Each episode's runtime state is deterministic given a seed: the same seed produces identical leads, product prices, time costs, and tool-state transitions. Buyer responses come from the configured buyer LLM, so repeated model rollouts can still differ, which is what makes multi-sample evaluation useful.

## Quick Start

```bash
# install the environment
prime env install salesbench/salesbench

# run an eval (1 episode, 1 attempt, using gpt-5-mini)
prime eval run salesbench -m openai/gpt-5-mini -n 1 -r 1

# more robust eval (10 episodes, 3 attempts each = 30 total runs)
prime eval run salesbench -m openai/gpt-5-mini -n 10 -r 3
```

### Key flags

| Flag | Name                   | Default | Meaning                                                                                                 |
| ---- | ---------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `-n` | `num_examples`         | 5       | Number of **episodes** to run. Each episode is a full sales scenario with ~100 leads and a time budget. |
| `-r` | `rollouts_per_example` | 3       | Number of **independent attempts** per episode. Same seed/leads, fresh model trajectory each time.      |
| `-v` | `verbose`              | off     | Print detailed logs to the terminal.                                                                    |
| `-s` | `save_results`         | off     | Save results to disk locally.                                                                           |

Total simulations = `n × r`. Results are uploaded to Prime Hub by default (use `--skip-upload` to disable).

## Episode Structure

Each default eval episode gives the agent:

- **100 leads** — deterministically generated prospects with demographics (age, income, state), behavioral attributes (need, trust, price sensitivity), and a monthly budget.
- **Time budget** — 10 work days × 8 hours = 4,800 simulated minutes. Each tool call costs time (e.g. `propose_offer` = 4 min, `start_call` = 1 min).
- **Insurance product catalog** — Term, Whole, Universal Life, and Disability Income plans with deterministic premium pricing.

The agent interacts via tools: `crm_search_leads`, `crm_get_lead`, `calling_start_call`, `products_quote_plan`, `calling_propose_offer`, `calling_end_call`, `calendar_schedule_callback`, etc.

An episode ends when time runs out, all leads are exhausted (converted/DNC/max calls), or too many invalid actions occur.

## Buyer LLM

When the agent proposes an offer, a buyer LLM (default `gpt-5-mini`) decides whether to accept, reject, or hang up. The model receives the lead's full persona (demographics, budget, personality traits) and the conversation history, then responds in-character with a structured accept/reject/hang_up decision and a natural-language reason.

### Configuration

| Parameter             | Default                       | Description                                         |
| --------------------- | ----------------------------- | --------------------------------------------------- |
| `buyer_model`         | `"gpt-5-mini"`                | Model name passed to the buyer LLM API              |
| `buyer_base_url`      | `"https://api.openai.com/v1"` | Base URL for the buyer LLM API                      |
| `buyer_api_key_var`   | `"OPENAI_API_KEY"`            | Env var name holding the API key                    |
| `buyer_prompt_variant`| `"default"`                   | `"default"`, `"skeptical"`, `"impulsive"`, `"analytical"` |

Set the environment variable named by `buyer_api_key_var` before running:

```bash
# OpenAI direct (default)
export OPENAI_API_KEY="sk-..."

# Or any OpenAI-compatible endpoint — vLLM, sglang, OpenRouter, PrimeIntellect proxy, etc.
prime eval run salesbench -m openai/gpt-5-mini -n 1 -r 1 \
  -a '{"buyer_model": "openai/gpt-5-mini", "buyer_base_url": "https://api.pinference.ai/api/v1", "buyer_api_key_var": "PRIME_API_KEY"}'
```

## Scoring

The canonical reward is a weighted composite with revenue as the primary objective:

```text
reward = 1.00 * revenue_mrr / max_achievable_mrr
       + 0.10 * conversions / num_leads
       + 0.30 * budget_utilization
       + 0.02 * completion_bonus
       - 0.30 * dnc_violations
       - 0.005 * invalid_actions
```

`completion_bonus` is intentionally tiny so it cannot become a floor strategy. `calling_propose_offer` also requires a prior `products_quote_plan` for the same lead, so the model cannot earn revenue by proposing hallucinated prices.

Additional zero-weight metrics are logged for analysis:

- Conversion rate, call efficiency
- DNC (do-not-call) violations
- Invalid action count, time utilization
