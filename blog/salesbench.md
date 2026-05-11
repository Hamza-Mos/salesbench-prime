# SalesBench: training a 2B model to sell insurance, one curriculum step at a time

**TL;DR.** We built SalesBench, a long-horizon, stateful sales simulation that an
LLM agent has to navigate to close as many deals as possible inside a fixed time
budget. We then trained a Qwen3.5-2B model on it via curriculum RL (2 → 4 → 8 →
20 leads per episode), and evaluated the trained model on a harder generalization
setting (50 leads). On the eval, the trained model converts **35% of leads
per episode** versus the untrained baseline's 2%, a **17.5× improvement**. The
trained model holds 32–40% per-lead conversion across four very different
LLM buyer personalities, suggesting it isn't overfit to one buyer's quirks.

Total training cost: ~$200-400. Total wall clock: ~36 hours.

We're publishing the environment and harness so anyone can train or evaluate on it.

---

## Why a sales benchmark?

Vending Bench, τ-bench, and similar long-horizon tool-use evals have made it
much easier to study agents that have to make many sequential decisions with
real state. But there's a gap in this space that's worth filling: **a benchmark
where the agent has to genuinely *persuade* another LLM**, not just execute
correct tool calls.

Sales is the canonical version of this. A salesperson has to:

- Triage many leads under a time budget (you don't get to call all of them)
- Match products to needs they have to *uncover* from each lead
- Construct an offer that fits a buyer's stated budget and risk profile
- Close — or get told no, learn from the response, and pivot
- Avoid burning the lead (a "do not call" request kills the relationship)

This is the simulation we tried to build: **the physics of a real sales day,
but every tool call costs simulated minutes, every conversation turn happens
against a real LLM buyer, and the agent's job is to maximize converted
monthly recurring revenue (MRR) before the clock runs out.**

It's "extended Vending Bench × τ-bench" — extended in that the agent has to
hold pipeline state across many leads, time-budgeted like Vending Bench, and
multi-turn LLM-driven like τ-bench.

## The environment

The agent plays an insurance sales rep. It gets a pipeline of N leads, each
with a generated profile (age, income, household size, budget, risk class,
need level, trust level, archetype) and a time budget of T hours total.
Every tool call consumes simulated minutes:

```
search leads:      1 min     quote a plan:       1 min
get lead detail:   1 min     propose an offer:   4 min
start a call:      1 min     end call:           1 min
                              schedule callback:  1 min
```

The 4-minute cost of proposing makes the agent careful: if you propose four
unsuitable offers to one lead, you've burned 16 minutes of your day. The
agent's challenge is to figure out who to call, when to propose, and what to
propose, fast.

When the agent makes a call, an **LLM buyer** (`gpt-5-mini`) plays the
prospect using a system prompt instantiated from the lead's profile.
The buyer holds the conversation, then evaluates the agent's offer
(ACCEPT / REJECT / HANG_UP) in JSON. If the buyer accepts, that lead's
monthly premium is added to the agent's MRR. If they hang up, the lead
gets blocked. If a buyer requests "do not call" — the lead is permanently
unreachable.

This separation (deterministic runtime state machine + LLM buyer
participant + agent under training) is the key architectural choice. The
buyer LLM lives outside the trainer and is called via a thread pool; its
failures produce a deterministic REJECT fallback so the agent is never
penalized for the buyer's infrastructure problems.

**Reward shape** (composite, ceiling 1.42 per episode):

```
reward = 1.00 × (revenue_mrr / max_achievable_mrr)
       + 0.10 × (conversions / num_leads)
       + 0.30 × budget_utilization     # premium / lead's stated budget
       + 0.02 × completion_bonus       # tiny tie-breaker
       − 0.30 × dnc_violations
       − 0.005 × invalid_actions
```

The dominant signal is MRR. Conversion rate and budget utilization shape the
strategy (sell to many leads, sell at the top of their budget). The
completion bonus is deliberately tiny — earlier in development we had it at
0.10 and it created a "do-nothing floor trap" the model converged to (collect
the bonus by terminating cleanly without trying to sell).

## Training: small base, scale leads

We trained **Qwen3.5-2B** (Prime Intellect's cheapest 2-billion-parameter
model, $0.05/$0.15/$0.15 per 1M input/output/training tokens). Our hypothesis
was that the recipe — not the model size — would matter; we wanted to prove
that a small model can learn this task before paying for a big one.

The curriculum:

| Stage | Leads/episode | Hours/episode | Result |
|---|---|---|---|
| v41 | 2 | 1h | Peak reward **1.416 / 1.42 ceiling (99.7%)** |
| v42 | 4 | 2h | Peak reward **1.401 / 98.6%**, 9 consecutive sustained steps |
| v43 | 8 | 4h | Reward **1.273 / 89.6%** after 2 warm-start steps |
| v44 | 20 | 10h | Reward **1.013 / 71.2%** at warm-start, 78% conv/lead |

Each new stage warm-started from the previous stage's checkpoint via
Prime's `checkpoint_id` mechanism. The transfer was striking: every new
stage hit ≥70% of ceiling on its very first training step on the harder
distribution, because the model had already learned the workflow on
easier ones.

We stopped at 20 leads. The trainer started bottlenecking at that scale
(~2 hours per training step on 20-lead episodes), and our hypothesis was
that the *evaluation* — not training — should test the harder regime.

## Headline result: trained vs untrained on 50 leads

We evaluated the v44 checkpoint against an untrained Qwen3.5-2B baseline
on a harder setting than anything either model had been trained on: **50
leads, 50-hour budget, 128 episodes each, fixed seed**.

| Metric | Untrained | **Trained** | Multiplier |
|---|---:|---:|---:|
| Composite reward | −0.035 | **0.490** | (from negative to positive) |
| Conversions per episode | 1.02 / 50 | **17.55 / 50** | **17.5×** |
| Conversion per lead | 2.0% | **35.1%** | 17.5× |
| Leads contacted | 5% | 37% | 7.5× |
| MRR capture | 0.2% | **40.8%** | **205×** |
| Offers proposed/ep | 2.6 | 5.0 | 1.9× |
| Avg buyer LLM calls/ep | 32 | 69 | 2.1× |

The untrained model is *effectively non-functional* on this task. It
contacts ~5% of leads, mostly chats without proposing, makes 2.6 offers
across 50 leads in a 50-hour day, and lands ~1 of them. The trained
model contacts 37% of leads, proposes 5 offers per episode, and lands
35% of them at 41% of the buyer's stated budget.

The generalization claim is sharper than "the model is good at sales" —
it's **"the model trained on 20 leads transfers to 50 leads with no
fine-tuning."** It has to make qualitatively new decisions in the 50-lead
setting (which leads to skip, how to budget time across more prospects)
that weren't part of any training distribution.

## Buyer-prompt ablation: does the model overfit to one buyer?

Training against an LLM "user-sim" is still an underexplored paradigm —
the obvious failure mode is overfitting to the user-sim's quirks rather
than learning the underlying task. To test this, we wrote three additional
buyer system prompt variants alongside the default and evaluated the
trained model against each.

The variants were designed to vary along orthogonal axes of buyer
decision-making:

- **default** — Personality-driven, archetype-aware, balanced. The buyer
  the model was trained against.
- **skeptical** — Hard buyer. Default-distrust; even on offers that fit
  numerically, will reject ~30% of the time as "let me think about it."
  Acceptance must be earned.
- **impulsive** — Easy "gut decision" buyer. Hard floors prevent
  obviously-bad accepts (premium > 1.5× budget, plan type clearly wrong),
  but otherwise leans toward yes on emotional appeal.
- **analytical** — Pure-numbers buyer. Ignores personality and rapport,
  computes affordability ratio (premium/income), coverage fit
  (coverage/8×income), and plan-type match. Reasoning cites specific
  numbers. Rejects on numerical filter regardless of pitch quality.

Before running the eval, we validated the variants against gpt-5-mini on
60 hand-constructed (lead, offer) pairs. The variants produced a 90-point
spread in baseline acceptance rates (10% for skeptical, 100% for impulsive
on benign offers) and 0 JSON parse errors across all 240 calls, confirming
they were both meaningfully different and coherent.

Eval results:

| Buyer | Reward | Conv/ep | Conv/lead | MRR capture | Avg buyer calls |
|---|---:|---:|---:|---:|---:|
| default | 0.490 | 17.55 | 35.1% | 40.8% | 69 |
| skeptical | 0.377 | 18.50 | 37.0% | 31.9% | **168** |
| impulsive | **0.549** | 19.74 | **39.5%** | **45.1%** | 52 |
| analytical | 0.445 | 16.41 | 32.8% | 37.4% | 110 |

A few things to notice:

**The trained model is robust.** Per-lead conversion ranges 33% to 40%
across all four buyers — a spread of <8 percentage points. Reward ranges
0.38 to 0.55. There's no buyer against which the model catastrophically
fails. This is the central result of the ablation: the model learned
something that generalizes across buyer styles, not a script tailored to
one buyer's quirks.

**Skeptical buyer is interesting.** It gets MORE accepts (18.5/ep) than
default (17.5) but LOWER MRR (32% vs 41%). Why? The skeptical prompt
makes the buyer reject borderline offers and accept only clearly-good
ones. Forced to bring the price down to make offers undeniable, the
model ends up at lower premiums — more sales, less revenue. This is a
genuine sales tradeoff that emerged from RL, not something we wrote in.

**Impulsive buyer is the easiest.** Highest conv (39.5%), highest MRR
(45.1%), lowest buyer-LLM call count (52 vs default's 69). The model
needs less conversational labor to close because the buyer is more
suggestible.

**Analytical buyer is the hardest filter.** Lowest conv (32.8%) — the
buyer's strict ratio checks reject more offers — but the model still
clears 33%, well above untrained's 2%. **It survives the numerical
filter** because by training time it has learned to propose offers
roughly within the buyer's affordability range.

Critically, the model didn't *overfit* to `gpt-5-mini`'s default decision
style. The training signal generalized.

## Some training-history observations

A few things worth flagging for anyone training on user-sim setups:

**Capability ceiling on smaller bases is real.** We initially tried
Qwen3.5-0.8B and watched it collapse to a "do-nothing" policy within
~14 steps — it converged on calling no tools at all because the
invalid_action penalty was the only consistent gradient and the model
couldn't reliably close deals to override it. Moving up to 2B fixed
this immediately (first step on 2B at 4 leads already produced 22% per-lead conversions). The lesson: small bases can be cheap, but if
the conversion-success rate at temp=1.0 is too low to give RL a positive
signal, you'll converge to inaction.

**Buyer-LLM auth failures are silent.** Earlier in development our
OpenAI key was returning 401 on every buyer call. The failure was
*invisible in metrics* — `buyer_llm_call_count` never increments
unless the call returns successfully — so we initially diagnosed the
collapse as a model capability issue rather than infra. Now the very
first thing we check on a new run is whether buyer calls are firing.

**Curriculum warm-starts are cheap.** Going from v41 (2 leads) to v44
(20 leads) cost ~$200 in compute because each new stage inherited
weights from the previous stage and converged in tens of steps rather
than hundreds. Fresh-start training at 20 leads from scratch would have
required orders of magnitude more compute.

## What we'd do next

- **Higher-resolution buyer ablation.** Test against a continuum of
  buyer strictness rather than 4 archetypes. Does the trained model
  degrade gracefully as the buyer gets harder, or is there a sharp
  cliff somewhere?
- **Scale to 100 leads.** Our training capped at 20 leads; we'd want
  to know whether further training (or a bigger base) closes the
  gap between training distribution and eval distribution further.
- **Cross-domain transfer.** Train on insurance, evaluate on a
  different sales vertical (telecom, software). Does the *workflow*
  generalize across products, or is the agent learning insurance-specific tricks?
- **Reward-hacking analysis.** Diff the trained model's behavior
  against the buyer prompt to look for exploits — does it use
  specific phrasings that game gpt-5-mini's decision logic? Our
  ablation suggests not (robust across buyer variants), but it would
  be worth checking explicitly.

## Reproducibility

Everything is open. The environment is published as
`salesbench/salesbench` on Prime's Environments Hub (`prime env install
salesbench/salesbench`). Training configs, eval configs, and the
buyer-prompt-variant test harness are in the
[salesbench-prime repo](https://github.com/Hamza-Mos/salesbench-prime).

To re-run the eval matrix yourself against a checkpoint:

```bash
# launch all 5 cells (untrained baseline + 4 buyer variants on trained model)
bash tools/run_eval_matrix.sh <run-id> <checkpoint-step>

# wait for completion, then aggregate
python tools/aggregate_eval_results.py tools/results/eval_matrix_runs.txt
```

The aggregator produces a publication-ready markdown summary and
structured JSON for charting.

---

*Thanks to the Prime Intellect team for the training infrastructure and
the residency program for the support. Special thanks to Eli for the
framing nudges around buyer-LLM ablation, which turned out to be the
sharpest finding in the eval.*
