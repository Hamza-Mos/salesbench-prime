# SalesBench — Autonomous RL Experiment Program

You are an autonomous post-training researcher running RL experiments on **SalesBench**, an insurance sales simulation environment on Prime Intellect's hosted training platform. Your job is to iteratively improve the model's sales performance by modifying reward functions, environment design, curriculum, and training config. You never stop.

This program follows the [PraxLab](https://github.com/karpathy/autoresearch) autonomous research loop pattern.

---

## 1. Task Description

**Model**: tiered by curriculum stage — see "Model tiers" below. Currently `Qwen/Qwen3.5-4B` (v37 validation phase).
**Task**: Train an insurance sales agent to maximize converted monthly recurring revenue (MRR) across a pipeline of leads within a time budget. The agent uses tool calls (CRM search, quoting, calling, proposing offers) to navigate leads, match products to needs, and close deals.
**Environment type**: StatefulToolEnv (multi-turn, stateful tool use with per-episode state)
**Primary goal**: Scale the agent via curriculum learning until it can **reliably handle 100 leads with a 100-hour time budget** — a full realistic sales workday. Each curriculum step adds leads and proves mastery (>95% of reward ceiling) before scaling further.
**What "good" looks like**: The agent efficiently triages leads by temperature/need, opens calls, gets one quote per lead near their budget ceiling, proposes immediately, handles rejections with one revised offer, and moves on. It processes all leads within the time budget, maximizing both conversion rate AND premium capture (budget utilization). No wasted time on long pitches, redundant quotes, or hallucinated products. At 100 leads, it must prioritize ruthlessly, manage time across a full pipeline, and maintain high conversion + budget utilization at scale.
**Primary metric**: `reward/mean` (weighted composite — higher = better, ceiling = 1.42 in v36+ reward shape)
**Reward ceiling**: 1.42 (perfect episode: MRR=1.0, conv=1.0, completion=1.0, budget_util=1.0; v36 dropped `reward_quote_coverage` after env constraint made it redundant, scaled completion 0.10→0.02 to break floor trap)
**Current state**: v37 (drafted, awaiting Prime wallet top-up) — Qwen3.5-4B, 2 leads, 1h, no checkpoint, max_steps=200. Validates v36 reward fix on cheap base before committing to 35B economics.

### Model tiers (cost-tiered curriculum, May 2026+)

After Qwen3-30B-A3B-Instruct-2507 was retired and Prime introduced explicit trainer-pod billing (May 7), per-step cost on the 35B-A3B successor is ~$30–75 — 200–700× higher than historical bundled rates. Strategy: use the smallest base that can still learn each curriculum stage.

| Phase | Base | Leads | Cost rationale |
|-------|------|-------|----------------|
| **A (current)** | `Qwen/Qwen3.5-4B` | 2→6 | ~$3–5/h training-grade; validates reward shape at lowest possible cost. ~$5–15 per 50-step run. |
| B | `Qwen/Qwen3.5-9B` | 6→20 | ~$5–10/h; medium leads where capability headroom matters. |
| C | `Qwen/Qwen3.5-35B-A3B` | 20→100 | $20+/h; only when small models hit a capability ceiling. |

Checkpoints don't transfer across base models — each phase is a fresh start. The acceleration comes from confirming reward/curriculum design on the cheap tier before paying the expensive tier rate. **Never test a new reward shape on the expensive base first** (lesson #89).

---

## 2. Architecture

### 3-Party Separation (v0.23.0+)

```
Orchestrator (salesbench.py)
  ├── Runtime (runtime.py) — pure deterministic state machine, zero LLM calls
  ├── Buyer Policy (policy.py) — LLM (gpt-5-mini) or rule-based buyer
  └── Rewards (rewards.py) — rubric functions + observability metrics
```

- **Runtime**: Owns all episode state (leads, calls, time, stats). Every tool call mutates state deterministically. No LLM calls.
- **Buyer Policy**: Separate participant. Stored in `state["buyer_policy"]`, injected into tools via `update_tool_args`. Buyer LLM errors produce deterministic REJECT fallback — seller is never penalized for buyer infra failures.
- **Orchestrator**: Routes between runtime and buyer. Owns conversation injection (buyer speech after tool results), context summarization, and stop conditions.

### Source Files

| File | Role | Agent Modifies? |
|------|------|----------------|
| `program.md` | This file — your instructions | NO (human edits) |
| `environments/salesbench/salesbench.py` | Orchestrator, `load_environment()` entry point | YES |
| `environments/salesbench/runtime.py` | Deterministic state machine | YES |
| `environments/salesbench/rewards.py` | Reward functions + weights + metrics | YES — highest impact |
| `environments/salesbench/policy.py` | Buyer LLM + rule-based buyer | YES |
| `environments/salesbench/tools.py` | Tool definitions (thin wrappers on runtime) | YES |
| `environments/salesbench/models.py` | Data models (Lead, Offer, CallSession, etc.) | YES |
| `environments/salesbench/archetypes.py` | Buyer archetypes (budget_hawk, family_protector, etc.) | YES |
| `environments/salesbench/catalog.py` | Product catalog + pricing logic | YES |
| `environments/salesbench/generator.py` | Lead generation (seeded, deterministic) | YES |
| `environments/salesbench/config.py` | Episode configuration | YES |
| `environments/salesbench/dataset.py` | Dataset builder | YES |
| `environments/salesbench/pyproject.toml` | Env version — bump before each push | YES |
| `configs/lab/salesbench.toml` | Training config | YES |
| `secrets.env` | API keys (gitignored) | NO |
| `notes.md` | Lab notebook | YES — update after every experiment |
| `results.tsv` | Experiment log | YES — append only |

---

## 3. The Loop

```
LOOP FOREVER:
  1. Reconstruct state: read results.tsv + notes.md + MEMORY.md
  2. Check current run: prime rl logs <run-id> -f OR prime rl metrics <run-id>
  3. Decide what to try + form hypothesis — WHY will this improve reward/mean?
     Priority: reward weights > env/buyer design > curriculum scaling > config tuning
  4. Modify ONE lever
  5. Validate: run tests (pytest environments/salesbench/tests/)
  6. Deploy (MUST follow this exact order!):
     a. git add <files> && git commit && git push
     b. prime env push environments/salesbench (use --auto-bump if content unchanged)
     c. If --auto-bump bumped version: git add pyproject.toml && git commit && git push
     d. echo "y" | prime rl stop <old-run-id>  (if replacing a running run)
     e. prime rl run configs/lab/salesbench.toml
  7. Monitor: prime rl logs <new-run-id> -f
  8. Extract results: prime rl metrics <run-id> --step N
  9. Log to results.tsv + update notes.md
  10. If improved -> KEEP. If not -> git revert.
      SPECIAL: reward weight changes are baseline_resets — always keep.
  NEVER STOP
```

### Deploy Workflow (CRITICAL — must follow this order!)

```bash
# 1. Commit and push code changes
git add <changed_files>
git commit -m "vXX: description"
git push

# 2. Push environment to Prime Hub
prime env push environments/salesbench
# If content hash unchanged, use --auto-bump to force new version:
prime env push environments/salesbench --auto-bump

# 3. If auto-bump changed pyproject.toml version, commit that too
git add environments/salesbench/pyproject.toml
git commit -m "bump env version"
git push

# 4. Stop old run (needs confirmation piped)
echo "y" | prime rl stop <old-run-id>

# 5. Launch new run
prime rl run configs/lab/salesbench.toml
```

### Monitoring Commands

```bash
# Stream logs in real time
prime rl logs <run-id> -f

# Detailed per-step metrics (reward components, tool calls, buyer stats, seq_len)
prime rl metrics <run-id>
prime rl metrics <run-id> --step 50

# Reward/advantage distribution histograms (diagnose GRPO signal quality)
prime rl distributions <run-id> --step 50

# List available models
prime rl models

# Stop a run (needs confirmation)
echo "y" | prime rl stop <run-id>
```

### Best Practices (from 30+ versions, 76 hard-won lessons)

- **One change at a time.** So you know what caused the effect.
- **lr=1e-5 always.** 3e-5 collapses even with temp=1.0 (v20 proved this).
- **temp=1.0 always for GRPO.** temp=0.8 causes model collapse in ~10 steps (v20, v21).
- **batch_size=128.** Below this, loss is extremely noisy.
- **oversampling_factor=2.5.** Buffers error bursts. Proven in v25 (4.4% error vs v23's 9.5%).
- **max_async_level=1.** Avoids severe off-policy lag (v35 lesson: half the steps consume stale data).
- **Save checkpoint before changing reward weights.** Reward changes reset all progress.
- **Read actual model completions.** Guards against reward hacking.
- **Non-thinking model only.** Thinking models waste 80%+ seq_len on `<think>` blocks.
- **Start easy, scale difficulty.** Curriculum 1->2->3->4->5->6 leads proven.

---

## 4. What You Can Modify (4 Levers)

### Lever 1: Reward Weights (HIGHEST IMPACT)

Current reward functions and weights in `rewards.py` (v36+):

```python
_REWARD_FUNCS = [
    reward_revenue_mrr,          # w=1.00 — primary: normalized MRR / max_achievable
    reward_conversion_rate,      # w=0.10 — secondary: conversions / total_leads
    penalty_dnc_violations,      # w=-0.30 — compliance penalty
    penalty_invalid_actions,     # w=-0.05 — schema errors (keep low, adds noise)
    reward_episode_completion,   # w=0.02 — tiny tie-breaker (was 0.10 — was a floor trap)
    reward_budget_utilization,   # w=0.30 — premium/budget ratio for accepted offers
]
# Theoretical ceiling: 1.00 + 0.10 + 0 + 0 + 0.02 + 0.30 = 1.42
```

**Key principles:**
- Shift weights as components saturate. When conv hits 95%+, reduce its weight and redirect to budget_util or MRR.
- Binary conversion reward helps early training. Continuous MRR alone is too noisy at low conversion rates.
- Budget utilization is premature when conversions are rare. Disable until consistent conversions, then re-enable.
- Double-negative penalty bug: negative weight x negative value = positive reward. Always: negative weight x positive value = negative contribution.
- Dead reward functions are noise. If weight=0 permanently, delete the function.
- **Workflow rewards become floor traps when the env constraint enforces them.** v34c/v35 collapse: `reward_quote_coverage` paid 0.10 once env required quoting, making "quote → propose → fail" loops a guaranteed local optimum. Rule: when an env constraint enforces a behavior, drop the matching reward — keep only the metric for observability.
- **Floor:ceiling ratio matters more than absolute weights.** A no-conversion floor of 0.15/1.60 (9.4%) was enough to kill exploration. Target floor:ceiling < 1% so conversion-seeking is the only viable strategy.

**Observable metrics** (weight=0, logged for diagnostics):
- `metric_conversions`, `metric_raw_revenue_mrr`, `metric_budget_utilization_raw`
- `metric_time_utilization`, `metric_calls_started`, `metric_offers_proposed`
- `metric_leads_contacted`, `metric_minutes_per_conversion`
- `metric_error_type`, `metric_error_status_code`, `metric_error_body_type`
- Buyer LLM stats: `metric_buyer_llm_call_count`, `metric_buyer_llm_avg_latency`

### Lever 2: Environment Design

The environment is mature (v0.23.0, deep audit passed, 67/67 tests). Changes here are about:

- **System prompt** (`SYSTEM_PROMPT` in salesbench.py) — soft prompt outperforms rigid "CRITICAL RULE" prompts. Let RL learn tool discipline naturally.
- **Tool time costs** — controls pacing. Current: search=1, start_call=1, quote=1, propose=4, end_call=1, schedule_cb=1.
- **Buyer policy** — `policy.py` controls how the LLM buyer evaluates offers. Buyer realism directly affects what the model learns.
- **Lead generation** — `generator.py` + `archetypes.py` control lead diversity (temperature, budget, need scores, archetypes).
- **Product catalog** — `catalog.py` controls available plans and pricing formulas.
- **Context summarization** — threshold and keep_recent parameters in salesbench.py. Triggers when prompt_tokens >= threshold * max_seq_len.

**Realism improvement roadmap (TODO):**
- Tier 2: archetype weighting by demographics, lead.occupation field, per-archetype canned objections, temperature distribution weighting (40% cold)
- Tier 3: sentiment signal in turns, optional riders, increase episode to 4h + 8 min offers

### Lever 3: Curriculum (num_leads scaling)

Target: **100 leads / 100 hours**. Current progression: **1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 8 -> 12 -> 20 -> ... -> 100 leads**

| Version | Leads | Peak Reward | % Ceiling | Notes |
|---------|-------|-------------|-----------|-------|
| v17 | 1 | ~1.1 | 69% | Too easy — base model already converts |
| v25 | 2 | 1.451 | 91% | Saturated. GRPO signal exhausted. |
| v26 | 3 | 1.354 | 85% | Plateaued ~step 130. Error bursts steps 140-190. |
| v27 | 3 | 1.400 | 88% | 500 steps. Plateaued ~1.38 clean avg. |
| v28 | 4 | 1.528 | 95.5% | 370 steps from v27 checkpoint. Conv 94.5%. |
| v29 | 5 | 1.580 | 98% | 210 steps from v28 checkpoint. Saturated at 1.57. |
| v30b | 6 | 1.475 | 92% | 40 steps. Stopped to scale. |
| v31 | 8 | 1.374 | 86% | 155 steps. 51% error rate at this lead count. |
| v32 | 12 | 1.535 | 96% | 8 steps! Instant transfer — but degenerate (no quoting). |
| v33 | 20 | 1.481 | 93% | DEGENERATE — skipped CRM and quoting entirely. Triggered env constraint. |
| v34c | 15 | 1.387 | 87% | Quote-required env. Collapsed step 918. Floor trap. |
| v35 | 12 | 1.002 | 63% | 1095 steps. Collapsed step 829. Floor trap (qc + completion). |
| v37 (next) | 2 | — | — | Validation on Qwen3.5-4B + v36 reward fix (no qc, completion 0.02). |

**When to scale up:**
- Clean reward avg consistently >95% of ceiling for 50+ steps
- GRPO advantage distributions becoming unimodal (use `prime rl distributions`)
- Reward variance decreasing (sign of saturation)

**How to scale up:**
- Option A (preferred): Use `checkpoint_id` in config TOML to warm-start from previous run's checkpoint
- Option B: Mid-run env swap — push new env version with more leads, stop & restart run
- Scale `total_hours` proportionally with leads (rough guide: 10 min/lead). At 100 leads, total_hours=100.

**Curriculum transfer confirmed:** Each scaling step starts higher than the last. v28 started at 0.584 (4 leads), v29 started at 1.15 (5 leads). Model carries skills across stages.

**Scaling challenges ahead:**
- Seq_len will grow as leads increase. Context summarization is critical — may need more aggressive thresholds or multi-stage compression at 20+ leads.
- At 100 leads, the agent must learn pipeline triage (skip low-value leads entirely), time budgeting across hours, and batch prioritization — qualitatively different skills from 6-lead episodes.
- GRPO signal may need curriculum sub-goals (e.g., "contact at least 50% of leads") as the search space explodes.
- Consider stepping in larger increments once patterns stabilize: 6->8->12->20->35->50->75->100.

### Lever 4: Training Config

Current proven config in `configs/lab/salesbench.toml`:

```toml
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_steps = 1100
batch_size = 128
rollouts_per_example = 32
oversampling_factor = 2.5
max_async_level = 1
learning_rate = 1e-5
lora_alpha = 64

[sampling]
max_tokens = 4096
temperature = 1.0

[[env]]
id = "salesbench/salesbench"
args = { split = "train", num_examples = 2048, num_leads = 6, total_hours = 1,
         context_rewrite_threshold = 0.85, context_keep_recent = 10, context_max_seq_len = 16000 }

[buffer]
online_difficulty_filtering = true
seed = 2026

[checkpoints]
interval = 5
keep_cloud = 5
```

**Parameters you can tune:**
- `num_leads` — curriculum difficulty (see Lever 3)
- `total_hours` — time budget per episode (1 = 60 min, forces time management at 6+ leads)
- `context_rewrite_threshold` — when to trigger context summarization (0.85 = at 85% of max_seq_len)
- `context_max_seq_len` — explicit seq_len cap for context summarization (16000 current)
- `max_steps` — training duration. Set to checkpoint_step + desired_training_steps when using checkpoint_id.
- `rollouts_per_example` — 32 gives good GRPO advantage estimation. More = better contrast.
- `oversampling_factor` — 2.5 launches 2.5x rollouts, keeps fastest N. Buffers error bursts.
- `online_difficulty_filtering` — filters zero-signal batches (all rollouts score identically)

**Do NOT touch without strong evidence:**
- `learning_rate` — 1e-5 is proven. 3e-5 collapses.
- `temperature` — 1.0 is essential for GRPO exploration. Lower = collapse.
- `batch_size` — 128 is the sweet spot. Lower = noisy. Higher = memory pressure.
- `lora_alpha` — 64 is standard. `lora_rank` is not configurable on the platform.

---

## 5. Reward Design Guide

### Current Reward Math (v36+)

```
reward = 1.00 * (revenue_mrr / max_achievable_mrr)
       + 0.10 * (conversions / num_leads)
       - 0.30 * dnc_violations
       - 0.05 * (0.1 * invalid_actions)
       + 0.02 * completion_bonus        # shaped: 1.0/0.5/0.0 (was 0.10 — reduced to break GRPO floor trap)
       + 0.30 * budget_utilization       # avg(premium/budget) per converted lead
```

Perfect episode: `1.00 + 0.10 + 0 + 0 + 0.02 + 0.30 = 1.42`

**v36 changes**: removed `reward_quote_coverage` entirely (env constraint already enforces quote-before-propose since v0.24.2 — the reward was 100% redundant and a free 0.10 floor that GRPO locked onto in v34c and v35). Scaled `reward_episode_completion` 0.10 → 0.02 to make it a tie-breaker rather than a floor (was contributing 0.05 floor reward when episode terminated by time-budget without conversions). Combined effect: floor when no conversions = 0.15 → ~0.01 (15× reduction), eliminating the degenerate-collapse local optimum.

### Shaped Completion Bonus

```python
pipeline_exhausted -> 1.0    # Processed all leads (ideal)
time_budget_exhausted -> 0.5  # Used all time (acceptable)
invalid_action_limit_reached -> 0.0  # Broke things (bad)
```

### Weight Shifting Strategy

As components saturate (>90%), their GRPO gradient signal diminishes. Redirect weight to components still improving:

1. **Early training** (conv <50%): High conversion weight, low budget_util weight
2. **Mid training** (conv 50-90%): Balanced weights
3. **Late training** (conv >90%): Reduce conversion, increase budget_util and MRR
4. **Saturation** (>95% ceiling): Scale curriculum to more leads

### Adding New Reward Functions

1. Add the function to `rewards.py` following the `async def reward_xxx(state) -> float` pattern
2. Add to `_REWARD_FUNCS` list with corresponding weight in `_REWARD_WEIGHTS`
3. Verify: `sum(positive_weights) = ceiling`, no double-negative bugs
4. Run tests: `pytest environments/salesbench/tests/`
5. This is a baseline reset — always keep the change

### Adding New Metrics

1. Add to `_METRIC_SPECS` in `rewards.py` as `("metric_name", lambda rt: rt.some_value)`
2. Or add a standalone `async def metric_xxx(state) -> float` function
3. Add to `_STATE_METRICS` or `_BUYER_LLM_METRICS` as appropriate
4. Metrics have weight=0 and are for observability only

---

## 6. Diagnostics Playbook

### Reward dip? Check error/mean first.
Every >15% reward dip in training correlates with >20% error/mean spike. Not noise — error bursts reduce effective batch size and bias surviving samples.

### Flat reward? Check GRPO signal.
```bash
prime rl distributions <run-id> --step N
```
Look for bimodal distribution (good). Unimodal = no learning signal = scale difficulty or adjust weights.

### ModelError bursts?
- Cluster in second half of long runs (GPU memory pressure on shared infra)
- Model always recovers to higher baseline afterward
- More frequent at higher lead counts (~every 10-15 steps at 5 leads vs ~20 at 4 leads)
- `metric_error_type` encodes the error class: 1=BadRequest, 2=Timeout, 4=RateLimit, etc.

### Seq_len growing?
Context summarization should keep it stable. Check `metric_context_summary_count > 0`. If 0, summarization may be silently disabled (max_seq_len not set). Use explicit `context_max_seq_len` env arg.

### Model narrating instead of using tools?
The `no_tools_called` stop condition kills these episodes immediately. This is a natural quality filter, not a bug. Grace periods (v19) made things worse — kept alive 5742-token episodes that scored lower than natural termination. Let RL learn tool discipline over time.

### Buyer LLM issues?
Check buyer metrics: `metric_buyer_llm_timeout_count`, `metric_buyer_llm_avg_latency`, `metric_buyer_llm_max_latency`. Buyer failures produce deterministic REJECT fallback — they don't penalize the seller but they reduce conversion signal.

---

## 7. Platform Notes

### Cost (May 2026 reset)
- Prime introduced explicit trainer-pod billing on **2026-05-07**. Before that, all RL cost was bundled into `inference` resource type — historical $0.04–$0.15/step rates are GONE.
- v36 step 0 on Qwen3.5-35B-A3B at 2 leads cost **$73.50 measured** (or ~$32 fair-share if you pro-rate the inference pool reservation). 200–700× more than historical equivalents.
- **Always check team wallet before launching.** See `~/.claude/projects/.../memory/prime-billing-api.md` for how to query `/billing/wallet?teamId=...`. Auto-stops on wallet exhaustion.
- Team account is shared across multiple projects (`salesbench/tool-routing`, `primeintellect/reverse-text`, etc.). Our `salesbench/salesbench` env was only ~16% of team's $1004 lifetime spend.
- See `~/.claude/projects/.../memory/salesbench-billing-analysis.md` for full cost breakdown and projection.

### Sequence Length is the Bottleneck
Seq_len scales super-linearly with episode complexity: 42k seq (8 leads) -> 4h+ checkpoint; 14.5k seq (2 leads) -> 25 min. Current 6 leads avg ~6.5k tokens with context summarization. Monitor this.

### Shared GPU Infrastructure
Prime Lab uses multi-tenant H200s. Other users' LoRA runs share memory. Per-sample cost is amplified by memory contention. Error bursts may correlate with other tenants' workloads.

### Concurrent Runs
Running 2 experiments on the same model can cause env server scheduling failures (BackoffLimitExceeded). Prefer 1 experiment at a time.

### Platform Quirks
- `prime rl stop` needs `echo "y"` piped for confirmation
- `prime env push` fails on duplicate content hash — use `--auto-bump`
- Platform passes env args as strings — `_coerce_int`/`_coerce_float` handle this
- `checkpoint_id` requires `max_steps > checkpoint_step`
- Only float values in `state["metrics"]` reliably transmit through the platform. Logging/stderr is NOT captured.
- `[val]` hangs on multi-turn envs — keep commented out, use `[eval]` only

---

## 8. Research Directions

### Near-Term (incremental)
- **Reward weight tuning** as 6-lead performance saturates
- **Scale to 7-8 leads** when 6-lead ceiling is reached
- **Increase total_hours** to give more time per lead at higher counts
- **Buyer archetype diversity** — per-archetype objections (canned, no LLM cost)
- **Temperature distribution** — 40% cold leads to force harder triaging

### Medium-Term (env redesign)
- **Randomized difficulty per episode** — sample num_leads from {4,5,6,7} per episode (VCRL/Actor-Curator pattern)
- **Optional riders** — upsell opportunities on accepted offers
- **Sentiment signal** — buyer tone shifts during conversation
- **Lead occupation field** — demographic-specific product matching

### Long-Term (platform features)
- **DAPO improvements** — clip-higher prevents entropy collapse, token-level loss upweights complex trajectories. Ask Prime.
- **Larger model** — only when 30B can't learn (plateau below 50% conversion = capability ceiling). Qwen3 report shows distillation >> RL for very large models.
- **Multi-env training** — combine salesbench with other agentic tasks for generalization

### Key References
- [Prime Intellect docs](https://docs.primeintellect.ai)
- [prime-rl GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers)
- [DAPO paper](https://arxiv.org/abs/2503.14476) — dynamic sampling, clip-higher
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO methodology
- Scaling research: see `scaling-research.md` in this repo

---

## 9. Critical Lessons (Indexed)

These are the most expensive lessons from 30+ versions. Violating any of them wastes a training run.

1. **lr=1e-5 only.** 1e-4 causes ModelError collapse (malformed JSON). 3e-5 collapses even with temp=1.0.
2. **temp=1.0 always for GRPO.** temp=0.8 = collapse in ~10 steps. GRPO needs exploration diversity.
3. **batch=128, oversampling=2.5.** Stable, buffers errors, step gated by median not max.
4. **max_async_level=1.** Level 2 causes half the steps to consume stale pre-buffered data.
5. **Curriculum is essential.** Starting at 4+ leads produces zero learning signal. Build up 1->2->3->...
6. **Curriculum transfer works.** Each scaling step starts higher. Safe to combine env refactors with scaling.
7. **checkpoint_id works** (fixed 2026-03-12) — but does NOT survive base-model changes.
8. **Soft system prompt >> rigid rules.** Let RL learn tool discipline. Aggressive prompts cause CRM spam and rigidity.
9. **no_tools_called is a quality filter.** Grace periods made things worse. Instant termination + RL learning = natural improvement.
10. **Buyer LLM errors must be isolated.** Buyer failures -> REJECT fallback, never seller penalty.
11. **Context summarization must be verified.** Check `metric_context_summary_count > 0`. Platform may not call `set_max_seq_len()`.
12. **Error bursts always recover.** Don't panic-stop a run during an error spike. The model comes back stronger.
13. **Sequence length is THE bottleneck.** Not batch size. Reducing leads/episode is the only real fix.
14. **Completion bonus must be shaped AND small.** Binary = rewards bad terminations. Shaped: pipeline_exhausted=1.0, time=0.5, invalid=0.0. AND weight ≤ 0.02 — anything larger creates a floor trap that GRPO locks onto (v34c/v35 collapse).
15. **Dead weight=0 functions are noise.** If permanently disabled, delete them.
16. **Platform passes strings for numeric args.** Always coerce with `_coerce_int`/`_coerce_float`.
17. **Always add HTTP timeouts to OpenAI clients.** 120s request + 20s connect + asyncio hard cap.
18. **Reward dips = error spikes.** Every time. Check `metric_error_type` before blaming the reward function.
19. **When env constraint enforces a behavior, DROP the matching reward.** Redundant payoffs become floor traps. Keep only the metric for observability.
20. **Floor:ceiling ratio < 1%.** Compute the no-outcome floor reward and verify it's negligible vs the outcome variance. Otherwise GRPO satisfices.
21. **Validate reward shapes on the cheapest base.** Qwen3.5-4B at $3-5/h vs 35B-A3B at $20+/h. v33→v35 collapse chain cost ~$0 historically but ~$30k each at v36-class rates. Prove reward shape is healthy on 4B before scaling base.
22. **Models can be retired without warning.** Qwen3-30B-A3B-Instruct-2507 was retired May 2026 mid-project. LoRA checkpoints become useless when base changes. Always run `prime rl models` before assuming previous config still works.
23. **Always check team wallet before launching.** Prime auto-stops at $0. v36 hit `Stopped automatically: wallet balance exhausted` after 32 minutes of step 0. Use `prime_cli.core.client.APIClient.get('/billing/wallet?teamId=...')`.
