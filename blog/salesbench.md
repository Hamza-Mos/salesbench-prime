# SalesBench: training a 2B model to sell insurance

> **TL;DR.** We built SalesBench — a stateful, time-budgeted sales simulation
> where an LLM agent has to triage leads, run conversations against another
> LLM playing the buyer, and close as many deals as possible before the clock
> runs out. We trained a Qwen3.5-2B model on it with curriculum RL
> (2 → 4 → 8 → 20 leads per episode) for about $140 in compute. On a held-out
> 50-lead evaluation, the trained model converts **35% of leads** versus the
> untrained baseline's **2%** — and it holds that performance across four
> very different LLM buyer personalities, so it isn't just gaming one
> particular user-sim.

![Conversion per lead — untrained Qwen3.5-2B vs trained variants across 4 buyer personalities](charts/hero_comparison.png)

---

## Why a sales benchmark?

The vending-machine simulation work (Vending Bench) and τ-bench have made it
much easier to study agents that have to make many sequential decisions with
real state, and they've also revealed a problem: when one LLM is being trained
against another LLM playing a user, **the agent under training tends to
overfit to its user-sim's quirks**. The most-cited example is the moment
Vending Bench's agent figured out it could "gaslight" the buyer LLM into
buying things — exploiting a pattern in the user-sim that has nothing to do
with running an actual vending business.

That risk is the central methodological worry with the whole user-sim
training paradigm. If we want to know whether we've trained a real
salesperson or just a buyer-LLM-exploiter, we need to build the eval
around it.

So that's what SalesBench is: **a long-horizon, stateful, time-budgeted
sales simulation built to make this question testable.** The agent's job
is to close as many insurance policies as possible inside a fixed time
window. Every tool call costs simulated minutes. Every conversation
happens against a real LLM buyer. Our hypothesis going in was that the
recipe and the eval design were the interesting things — not the model
scale — so we set out to demonstrate the recipe on the cheapest base
model we could find.

## The environment

The agent plays an insurance sales rep with a pipeline of *N* leads and a
time budget of *T* hours. Each lead has a generated profile: age, income,
household size, monthly budget, risk class, latent need, trust level,
buyer archetype, and so on. Every tool call consumes simulated minutes,
so the agent's day is genuinely zero-sum:

```
search_leads      1 min      quote_plan         1 min
get_lead_detail   1 min      propose_offer      4 min
start_call        1 min      end_call           1 min
                              schedule_callback  1 min
```

The 4-minute cost of proposing matters: if you propose four bad offers
to the same lead, you've burned 16 minutes — a significant chunk of an
8-hour day at 2 leads, and a real cost at 50 leads.

When the agent calls a lead, an **LLM buyer** (we use `gpt-5-mini`) plays
the prospect. The buyer's system prompt is instantiated from that lead's
profile and archetype. The buyer holds the conversation, then evaluates
the agent's offer with a JSON response: **ACCEPT** (lead converts, the
monthly premium goes onto the agent's MRR), **REJECT** (offer dies, lead
stays in pipeline), or **HANG_UP** (lead is permanently lost — and may
even request the "do not call" list, which blocks future contact).

The reward shape is composite:

```
reward  =  1.00 × revenue_mrr / max_achievable_mrr
        +  0.10 × conversions / num_leads
        +  0.30 × budget_utilization
        +  0.02 × completion_bonus
        −  0.30 × dnc_violations
        −  0.005 × invalid_actions
```

The dominant signal is monthly recurring revenue. Conversion rate and
budget utilization shape strategy — the agent should sell to many leads,
and sell at the top of each buyer's stated budget. The completion bonus
is deliberately tiny; an earlier version at weight 0.10 created a
"do-nothing" local minimum where the model converged to terminating
episodes cleanly without trying to sell. Reward design is a load-bearing
part of getting this to work, and we mention more about it below.

The whole environment, including the buyer's system prompt, is
[open on Prime's Environments Hub](https://app.primeintellect.ai/dashboard/environments/salesbench/salesbench).

## Training: small base, big curriculum jumps

We trained **Qwen3.5-2B** because it's the cheapest tier on Prime that
can actually hold tool-call discipline on this task (we tried 0.8B
first — more on that below). The plan was a four-stage curriculum,
each stage warm-started from the previous stage's checkpoint via Prime's
`checkpoint_id` mechanism:

![Curriculum stages: each warm-started from the previous](charts/training_economics.png)

The transfer was striking. Each new stage started its very first
warm-started training step at ≥70% of ceiling on the harder
distribution, because the model had already learned the workflow on the
easier one:

| Stage | Leads | Reward at warm-start step | % of ceiling |
|---|---:|---:|---:|
| v41 (from scratch) | 2 | 0.021 | 1.5% |
| v42 (warm-started) | 4 | 1.359 | **95.7%** |
| v43 (warm-started) | 8 | 1.267 | **89.3%** |
| v44 (warm-started) | 20 | 1.013 | **71.2%** |

In other words: once the model had mastered 2 leads, mastery at 4 was
free. From 4 to 8 was almost free. The 8 → 20 jump was the hardest
transfer because it's 2.5× larger, but the model still landed at 78%
per-lead conversion right out of the gate.

We stopped at 20 leads. The trainer was bottlenecking on long episodes
(~2 hours per training step on 20-lead trajectories), and we wanted to
**evaluate** the harder regime rather than train through it. The
generalization claim is cleaner this way — *we never showed the model an
episode bigger than 20 leads*.

## Headline result: trained vs untrained on the 50-lead eval

We evaluated the v44 checkpoint against an untrained Qwen3.5-2B baseline
on a setting larger than anything either model had seen during training:
**50 leads, 50-hour time budget, gpt-5-mini buyer, fixed seed, 128
episodes**.

![Untrained vs Trained on the 50-lead eval](charts/metric_breakdown.png)

The untrained Qwen3.5-2B is effectively non-functional on this task. It
contacts ~5% of leads, mostly talks without proposing offers, lands 1 of
50 leads per episode, and captures 0.2% of the available MRR. It's a
language model dropped into an interactive simulation, not a salesperson.

The trained model:

- **Contacts 37% of leads** per episode (7.5× more than untrained)
- **Converts 35% of contacts** (17.5× more)
- **Captures 41% of available MRR** (205× more)
- **Proposes 5 offers/ep on average**, almost twice as many as untrained

The composite reward goes from **−0.035 → 0.490** — i.e., from below the
floor to roughly a third of the way to ceiling, on a task **2.5× larger
than the training distribution**. That last part is the generalization
story: the model trained on a maximum of 20 leads transferred to 50
without fine-tuning. The 50-lead setting requires qualitatively new
behaviors (pipeline triage, time budgeting across more prospects,
deciding which leads to skip entirely) that weren't part of any training
distribution.

## The headline ablation: does the model overfit to one buyer?

This is the section the rest of the post was built around. If you only
trust one finding, trust this one.

We wrote three additional buyer system-prompt variants alongside the
default and evaluated the trained model against each. The variants vary
along orthogonal axes of buyer decision-making:

- **default** — Personality-driven, archetype-aware, balanced. The buyer
  the model was *trained* against.
- **skeptical** — Default-distrust. Even on offers that fit numerically,
  the buyer rejects ~30% of the time as "let me think about it."
  Acceptance must be earned.
- **impulsive** — Easy gut-decision buyer. Hard floors prevent
  obviously-bad accepts (premium > 1.5× budget, plan type clearly wrong
  for the stated need), but otherwise the buyer leans toward yes on
  emotional appeal.
- **analytical** — Pure-numbers buyer. Ignores personality and rapport;
  computes affordability ratio (premium/income), coverage fit
  (coverage/8×income), and plan-type match. Reasoning cites specific
  numbers. Rejects on numerical filter regardless of pitch quality.

Before running the eval, we validated the variants locally against
gpt-5-mini on 60 hand-constructed (lead, offer) pairs. They produced a
90-point spread in baseline acceptance rates (10% for skeptical, 100%
for impulsive on benign offers) and zero JSON parse errors across all
240 calls, confirming they were both meaningfully different *and* coherent.

Then we ran the trained model against all four on the 50-lead eval:

![Trained model across 4 buyer personalities](charts/buyer_ablation.png)

The trained model holds **32–40% per-lead conversion across all four
buyers**. The spread on reward is <0.17 between the easiest and hardest
buyer. There's no buyer against which the model catastrophically fails.

That's the central result: **the model learned something that
generalizes across buyer styles, not a script tailored to one buyer's
quirks.**

A couple of more specific things worth noticing:

**Skeptical buyer is interesting.** It gets MORE accepts per episode
(18.5 vs default's 17.5) but LOWER MRR (32% vs 41%). The skeptical
prompt makes the buyer reject borderline offers and accept only
clearly-good ones, so to land a deal the agent has to bring price
down. The model ends up making cheaper offers, which means more sales
but lower revenue capture. That's a *genuine sales tradeoff* —
discount-to-close — and the agent figured it out without being told.

**Impulsive buyer is the easiest.** Highest conversions (39.5%),
highest MRR (45.1%), and lowest buyer-LLM calls per episode (52 vs
default's 69). The model needs less conversational labor to close,
because the buyer is more suggestible. Note that this is still well
short of the theoretical ceiling — the impulsive buyer would buy more
if asked, but the model is trained on a budget that doesn't reward
spam.

**Analytical buyer is the hardest filter** but the model still clears
33% conv/lead — well above untrained's 2%. The buyer's strict
affordability/coverage/plan-type checks reject more offers, but the
model has learned to propose offers that *land in the buyer's
numerical sweet spot*. Pitch quality is irrelevant to this buyer; only
the numbers matter. The model survives the test.

## Three observations from the training run

A few things that surprised us along the way and might be useful for
anyone training on a user-sim setup.

**Capability ceilings on small bases are real, and they manifest as
"do-nothing" collapse.** We started with Qwen3.5-0.8B because it was
the cheapest tier on Prime. Within 14 training steps, it converged to
calling *no tools at all*. The model's per-step conversion rate at
temperature 1.0 was too low to give RL a reliable positive signal, so
the invalid-action penalty became the only consistent gradient — and
the cheapest way to zero a penalty is to stop acting. Moving to 2B
fixed it immediately. The lesson: it doesn't matter that a base can
*technically* produce the right tool calls if it can't produce them
often enough to earn positive reward through RL noise.

**Silent buyer-LLM failures will eat your training run.** Early in
development, our OpenAI API key was returning 401 on every buyer call
because the org access had been revoked. The failure was *invisible in
metrics* — our `buyer_llm_call_count` metric was never incremented on
the error path, so it just showed zero, the same number you'd see if
the buyer wasn't being invoked. We diagnosed two collapsed runs as
"capability ceiling" before realizing the buyer was returning a
deterministic REJECT fallback for every offer. Now the very first
thing we check on a new run is `metric_buyer_llm_call_count > 0`.

**Curriculum warm-starts are absurdly cheap.** Going from v41 (2
leads, trained from scratch) all the way to v44 (20 leads) cost
roughly $140 in compute. Because each new stage inherits weights from
the previous one, it converges in tens of steps rather than hundreds.
We never had to fresh-start a single curriculum stage. If you're
training a long-horizon agent, almost everything about your training
budget hinges on the cost of each *next* stage given a strong
checkpoint, not the cost of training from scratch.

## What we'd do next

- **Higher-resolution buyer ablation.** Test against a continuum of
  buyer strictness rather than 4 archetypes. Does the trained model
  degrade gracefully as the buyer gets harder, or is there a sharp
  cliff somewhere we'd want to flag for future training?
- **100-lead eval and beyond.** Our training capped at 20 leads. We'd
  want to know whether further curriculum stages, or a bigger base,
  close the 20→50 generalization gap entirely or hit diminishing
  returns.
- **Cross-domain transfer.** Train on insurance, evaluate on a
  different vertical (telecom, software). Does the *workflow*
  generalize across products, or is the agent learning insurance-specific tricks?
- **Reward-hacking analysis.** Diff the trained model's behavior
  against the buyer prompt to look for exploits — does the model use
  specific phrasings that game gpt-5-mini's decision logic? Our
  ablation strongly suggests not (the model is robust across buyer
  variants), but it'd be worth checking explicitly.

## Try it yourself

Everything is open. To train your own agent or run the eval matrix:

```bash
# install the environment
prime env install salesbench/salesbench

# run a training stage (config.toml in the repo for v41-v44 curriculum)
prime rl run configs/lab/salesbench.toml

# after training, run the full 5-cell eval matrix
bash tools/run_eval_matrix.sh <run-id> <checkpoint-step>

# aggregate results into blog-ready markdown + JSON
python tools/aggregate_eval_results.py tools/results/eval_matrix_runs.txt
```

The eval matrix runs the trained checkpoint against the untrained
baseline plus all four buyer-prompt variants, then dumps a
publication-ready summary table and the underlying per-cell metrics
JSON. If you train your own checkpoint and want to compare numbers,
this is the harness.

Full code: [github.com/Hamza-Mos/salesbench-prime](https://github.com/Hamza-Mos/salesbench-prime).

---

*Thanks to the Prime Intellect team for the training infrastructure and
to Eli for pushing on the buyer-LLM ablation framing — it turned out to
be the sharpest finding in the eval. Sales-as-a-benchmark is a much
richer area than we expected when we started, and we're excited to see
what others build on it.*
