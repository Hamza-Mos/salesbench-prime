# I Trained a 2B Model to Sell Insurance. Here's What It Learned.

*A long-horizon, agent-to-agent RL environment, plus an ablation that suggests the model learned the task instead of just gaming the buyer.*

**TL;DR:** SalesBench is an open RL environment where a model runs an insurance sales pipeline against an LLM buyer, scored by monthly recurring revenue closed. Qwen3.5-2B trained through a 2 -> 4 -> 8 -> 20 lead curriculum (~$140 of compute) converts 35.1% of leads on a held-out 100-lead eval, versus 2.0% for the untrained base. Against three buyer personalities the agent never trained against, conversion stays between 32.8% and 39.5%.

Most agent benchmarks are either long-horizon or agent-to-agent. I wanted both.

Vending Bench is long-horizon and stateful, but the customers are scripted. tau-bench has an LLM user, but the episodes are short. Sotopia has two LLMs in a social setting, but it is single-session and graded by a judge model. SalesBench is my attempt at the missing piece: long-horizon, stateful work against an LLM counterparty, scored by a verifiable business outcome. I think of it primarily as an eval; the training runs are there to show the environment has a real learning signal.

The setup is simple. The seller gets N leads and a fixed number of simulated hours. Each lead has income, budget, household, temperature, latent need, trust level, price sensitivity, and a buyer archetype. A buyer LLM (gpt-5-mini) plays the prospect during the call and returns a structured decision when the seller proposes an offer. The reward comes from runtime state, not an LLM judge score. A sale either closes or it does not.

Conversion per lead, untrained vs trained against 4 buyer personalities

## The Environment

An episode gives the agent a pipeline, a time budget, and tools:

```text
crm_search_leads             1 min
crm_get_lead                 0 min
calling_start_call           1 min
products_quote_plan          1 min
calling_propose_offer        4 min
calling_end_call             1 min
calendar_schedule_callback   1 min
```

The 4-minute offer cost is the dominant constraint. Repeated bad offers to one lead burn budget the agent could have spent on the rest of the pipeline. The workflow the model has to learn is: search the CRM, start one call, discover need, quote a plan, propose the offer, move on.

The reward is mostly monthly recurring revenue:

```text
reward = 1.00 * revenue_mrr / max_achievable_mrr
       + 0.10 * conversions / num_leads
       + 0.30 * budget_utilization
       + 0.02 * completion_bonus
       - 0.30 * dnc_violations
       - 0.005 * invalid_actions
```

MRR dominates. Conversion rate and budget utilization shape behavior. The completion bonus is tiny on purpose: an earlier version paid more for clean termination and the model learned to finish episodes without selling. Stacking the four weighted reward terms during the from-scratch run shows the same hierarchy in the gradient:

Weighted reward components over training: MRR dominates, budget is the second lever, completion bonus is intentionally tiny

## What Training Changed

The base model does not understand the environment. On one sampled rollout, its first action was:

```text
agent -> calling_propose_offer({
           "lead_id": "lead_0019",
           "plan_type": "TERM",
           "coverage_amount": 210000,
           "monthly_premium": 9.04,
           "next_step": "ACCEPT",
           "term_years": 10
         })

tool  -> "got an unexpected keyword argument 'lead_id'"
```

No CRM search, no call started, no quote, invalid argument name, and the buyer outcome stuffed into `next_step` as if it were the agent's to set. Across three sampled episodes, the base model made 43 offer attempts, 0 quote calls, and one accidental conversion.

The trained model looks different:

```text
agent -> crm_search_leads()
tool  -> 10 leads returned, sorted by need and budget

agent -> calling_start_call({"lead_id": "lead_0042"})
tool  -> call started

agent -> "Hi Maria, this is Sam from State Insurance.
          I see you've got two kids, and I want to make sure
          they're covered if anything happens to you."

buyer -> "What's the catch?"

agent -> products_quote_plan({
           "lead_id": "lead_0042",
           "plan_type": "TERM",
           "coverage_amount": 400000,
           "term_years": 20
         })

tool  -> quote: TERM 20yr / $400k / $156/mo

agent -> calling_propose_offer({
           "plan_type": "TERM",
           "coverage_amount": 400000,
           "monthly_premium": 156.00,
           "next_step": "submit_application",
           "term_years": 20
         })

buyer -> ACCEPT. "Premium fits my budget and covers the kids."
tool  -> +$156 MRR
```

That is the whole result in miniature. The trained agent learned the tool schema, the call sequence, the quote-before-propose constraint, and a basic sales policy: match the plan to the lead, keep the premium inside budget, close, move on.

The buyer is not a simple accept/reject oracle. Real rejections include reasons:

> "Premium fits my budget, but you didn't confirm whether a medical exam or any other conditions are required. I need those details and more policy terms before I can commit, so I'll pass for now."

That creates the tradeoff. The agent can spend another 4 minutes trying a revised offer, or move to the next lead. With 100 leads and a fixed budget, that triage is the game.

You can also watch the workflow itself emerge over training. Per-tool call counts per episode tell the story: the model starts by spamming CRM searches and almost never starting a call. By the end, the search-call-quote-propose cycle has balanced into something that looks like a sales pipeline.

Tool calls per episode over training: CRM-search spam collapses, quote and propose climb to a steady cadence

## Curriculum Training

I trained Qwen3.5-2B with GRPO through a four-stage curriculum: 2 leads, then 4, 8, and 20. Each stage warm-started from the previous checkpoint.

Curriculum stages: cost, wall clock, and lead count for each warm-started stage

The first stage did most of the work. Starting from scratch on 2 leads, the model needed about 200 steps to go from broken tool use to near-ceiling performance. Invalid actions dropped from roughly 6 per episode to about 0.5.

From-scratch run: reward and conversion rate climb together as invalid actions per episode crash from ~6 to ~0.5

After that, scaling was cheap. The 4-lead stage started at 95.7% of ceiling, the 8-lead stage at 89.3%, the 20-lead stage at 71.2%. Warm-starting carried the workflow forward; each new stage was mostly about triage, not relearning the tools.

Curriculum learning curve across all 4 stages, cumulative training steps

Total training cost was about $140 on Prime Intellect, over roughly 35 hours of wall clock. I stopped at 20 leads because long episodes were becoming slow, and evaluating on a larger distribution is the cleaner generalization test anyway.

## The 100-Lead Eval

Eval setup: 100 leads, 50 simulated hours, fixed seed, 128 episodes per cell. The model had only trained up to 20 leads.

Untrained vs trained on the 100-lead eval: conversion, contact rate, MRR capture, budget utilization, offers proposed


| Metric          | Untrained base | Trained | Lift  |
| --------------- | -------------- | ------- | ----- |
| Lead conversion | 2.0%           | 35.1%   | 17.5x |
| Leads contacted | 5%             | 37%     | 7.5x  |
| MRR capture     | 0.2%           | 40.8%   | 205x  |
| Reward          | -0.035         | 0.490   | --    |


DNC violations were zero across every eval cell. Per-turn closing rate improved by roughly 13x: the base model converts ~1 lead per 128 turns of dialog; the trained model closes at a rate that is over an order of magnitude higher.

## Buyer-Prompt Ablation

The concern with LLM-vs-LLM training is that the seller might learn the buyer simulator instead of the task. Vending Bench showed a version of this: the trained agent nudged the customer model into purchases unrelated to running the business.

To check for it here, I wrote three additional buyer prompts. The trained seller only ever saw the default buyer:

- **default:** balanced, personality-driven, archetype-aware.
- **skeptical:** default-distrust, rejects borderline offers.
- **impulsive:** gut-decision buyer with hard floors against bad offers.
- **analytical:** pure numerical filter, ignoring pitch quality.

Validated locally on 60 lead-offer pairs (240 buyer calls): 90-point spread in acceptance rate, zero parse errors. Then I ran the trained seller against all four:

Trained model across 4 buyer personalities: reward, conv/lead, and MRR capture side by side


| Buyer      | Conv/lead | Reward | MRR capture |
| ---------- | --------- | ------ | ----------- |
| default    | 35.1%     | 0.490  | 40.8%       |
| skeptical  | 37.0%     | 0.377  | 31.9%       |
| impulsive  | 39.5%     | 0.549  | 45.1%       |
| analytical | 32.8%     | 0.445  | 37.4%       |


Conversion stays between 32.8% and 39.5%. No variant breaks the agent.

The skeptical buyer is the most interesting case: more accepts than default (37.0% vs 35.1%), but lower MRR (31.9% vs 40.8%). The agent closes skeptical buyers by pitching cheaper plans. That is a real sales tradeoff, not a simulator artifact.

The analytical buyer is the cleanest check. It ignores pitch quality and accepts only on affordability, coverage fit, and plan fit. The trained model still converts 32.8%, which says the policy is choosing numbers that work, not exploiting conversational style.

The conversational labor varies by ~3x: a skeptical prospect needs many more buyer-LLM calls per episode than an impulsive one, with offer counts moving the opposite direction because rejected offers force retries.

Buyer LLM calls per episode and offers proposed per episode, by buyer variant

## Takeaway

A 2B-parameter model, trained for ~$140 of compute, learns to run a long-horizon, agent-to-agent sales pipeline well enough to convert 17.5x more leads than its base, and holds 33-40% per-lead conversion against four buyer personalities. The environment has a real learning signal and the buyer-prompt ablation does not surface a catastrophic overfit.

Earlier reward shapes (a larger completion bonus, a redundant quote-coverage term) produced floor-trap collapses where the agent stopped trying to sell. The numbers above are with the reward stripped down to the form shown earlier in the post.

## Run It

```bash
# Install Prime and log in.
pip install prime-cli
prime login

# Set your OpenAI key (used by the buyer LLM).
cp secrets.env.example secrets.env  # then edit OPENAI_API_KEY

# Install the SalesBench environment.
prime env install salesbench/salesbench

# Train the curriculum. After each stage finishes, copy its checkpoint
# id into the next stage's config (the `checkpoint_id` field).
prime rl run configs/curriculum/stage1-2-leads.toml    # 2 leads, from scratch
prime rl run configs/curriculum/stage2-4-leads.toml    # 4 leads, warm-start
prime rl run configs/curriculum/stage3-8-leads.toml    # 8 leads, warm-start
prime rl run configs/curriculum/stage4-20-leads.toml   # 20 leads, warm-start

# Evaluate: untrained baseline + trained against 4 buyer personalities.
bash tools/run_eval_matrix.sh <stage4-run-id> <ckpt-step>
```

Buyer variants live in `environments/salesbench/policy.py` as `_DECISION_GUIDELINES_VARIANTS`. To eval against a different buyer, change `buyer_prompt_variant` in any `configs/eval/*.toml`.

Full code: [github.com/Hamza-Mos/salesbench-prime](https://github.com/Hamza-Mos/salesbench-prime).

Thanks to the Prime Intellect team for the training infrastructure and the free credits!