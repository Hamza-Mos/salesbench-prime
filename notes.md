# SalesBench Lab Notebook

## 2026-05-09: v38 — escalate base to Qwen3.5-4B after v37 do-nothing collapse

**Status**: drafted, about to launch.
**Change vs v37**: ONE LEVER — `model = "Qwen/Qwen3.5-4B"` (was 0.8B). Everything else identical: 2 leads, 1h, no checkpoint, max_steps=200, env v0.24.3, same v36 reward shape (no quote_coverage, completion 0.02).
**Hypothesis**: 4B has enough capability headroom to actually convert occasionally during exploration. Once conversion reward fires (even at low rate), it overrides the −0.005-per-invalid-action penalty and gives GRPO a positive gradient to climb. 0.8B never got there because conversion never fired (capability gap), so the only available gradient was "minimize errors" → degenerate "do nothing."
**Watch for**: by step 30-50, are there any propose_offer attempts (>0.1/ep) and any non-zero conversions? If yes: capability hypothesis confirmed, ride 4B through curriculum. If still degenerate: the issue is reward shape (penalty too dominant), not capability — need to reduce penalty_invalid_actions weight before any further base escalation.

---

## 2026-05-09: v37 — DO-NOTHING COLLAPSE on Qwen3.5-0.8B

**Run**: `r90m00fyzu2o38wksjs24co4` — STOPPED at step 35 (user-requested via loop after diagnosing collapse).
**Reward trajectory**:
- Step 0: −0.0271 (model attempting things: 0.32 start_call/ep, 0.41 propose_offer/ep, 0.12 quoted_proposals/ep)
- Step 7-13: monotonic decline in tool usage; reward climbing toward 0
- Step 14 onward: full collapse — 0 start_call, 0 quote_plan, 0 propose_offer, 0 leads_contacted, num_turns≈1
- Reward stuck at noise floor ~−0.0002 ± 0.0002 for 22 consecutive steps

**Mechanism (clear)**: with conversion reward at 0 (the 0.8B base never closed a deal), the only consistent gradient available to GRPO was the invalid_action penalty (`−0.05 × 0.1 × invalid_count = −0.005/error`). The cheapest way to zero that penalty is to call no tools at all. RL converged on it within ~14 steps.

**Cost**: ~35 steps × ~$0.50–2/step on 0.8B ≈ ~$15–70 to learn this lesson cleanly. Vastly cheaper than learning it on 35B.

**Lesson (new)**: A capability-ceiling base + a non-zero invalid-action penalty + a near-zero outcome-reward ceiling = **degenerate "do nothing" attractor**. The floor:ceiling ratio of the reward shape was correct (~0.7%), but ceiling that the *current model* can actually reach is what matters for GRPO, not the theoretical ceiling. If the model can't reliably trip the outcome reward, the penalty is the only signal and the model learns to minimize it by inaction.

**Implication for tier escalation**: program.md's capability-ceiling rule is right — but the signal isn't "conv rate plateaus low," it's "tool-attempt rate goes to zero in <30 steps." That's faster and more obvious than waiting for plateau.

**Strategy revision**: keep "scale leads on smallest viable base" as the guiding principle, but the smallest viable base for *this* env is at least 4B, not 0.8B. v38 tests that.

---

## 2026-05-08: v37 LAUNCHED — Qwen3.5-0.8B + v36 reward fix

**Run**: `r90m00fyzu2o38wksjs24co4` (launched 2026-05-08, dashboard: https://app.primeintellect.ai/dashboard/training/r90m00fyzu2o38wksjs24co4)
**Wallet**: $2,491.67 at launch (topped up after v36 exhausted it).
**Model**: `Qwen/Qwen3.5-0.8B` — cheapest tier on Prime ($0.02/$0.06/$0.06 per 1M in/out/train). Revised down from 4B post-Eli mtg.
**Config**: 2 leads, 1h, no checkpoint, max_steps=200, env v0.24.3.

**Strategy revision (Eli mtg 2026-05-08)**: drop the tiered base-size plan (4B→9B→35B). Stay on the smallest base for the entire 2→20-lead curriculum. Only escalate base on a real *capability* ceiling (conv flat across more steps), not a reward-signal issue. Goal: cheapest possible validation that the recipe + v36 reward fix work, then scale leads.

**Reward fix unchanged from v36**: removed `reward_quote_coverage` (env enforces quote-before-propose; reward was a redundant 0.10 floor that GRPO locked onto in v34c/v35), scaled `reward_episode_completion` 0.10 → 0.02 (tie-breaker, not floor). Floor when no conversions: 0.15 → ~0.01 (15× reduction). Ceiling: 1.42.

**Hypothesis**:
- (cheap-base capability) 0.8B can hold tool discipline well enough to learn at 2 leads. If yes: cheapest validation tier proven, ride it through curriculum.
- (reward-shape) with the floor gone, GRPO has no degenerate local optimum. Conversion-seeking is the only path to >0.01 reward.

**Watch for**:
- Step-0 wall-clock (35B was 1810s; 0.8B should be much faster, target <300s/step at 2 leads).
- Does it climb past 0.5 in <50 steps?
- Reward variance bimodal-ish (good GRPO signal)?
- Any `metric_error_type` spikes — 0.8B may struggle with structured tool calls.
- If 0.8B can't even hold tool discipline (constant `no_tools_called` terminations or persistent error bursts), bump to 4B fallback per program.md tier table.

**Curriculum plan if v37 succeeds**: 2 → 3 → 4 → 5 → 6 → 8 → 12 → 20 on 0.8B. Scale only after >95% of ceiling for 50+ steps.

**Blog roadmap (Eli ask)**: buyer-prompt ablation study now sits alongside curriculum scaling as a deliberate research artifact for the eventual publication. Co-publishing path TBD by Sebastian.

---

## 2026-05-07: v36 — STOPPED at step 0, wallet exhausted

**Run**: `geh9smdj2wxkzpjn2pro3dtk` — STOPPED 18:19 UTC, ~31 min after launch
**Reason**: `Stopped automatically: wallet balance exhausted (available=$-8.30)`. Hard external block.

**One data point before stop**:
- Step 0: 1809.99s (30.2 min), reward = -0.0005, seq_len 5116 tokens/sample
- 14/128 rollouts filtered (4 gibberish, 10 zero_advantage) — 89% clean, expected on fresh untrained 35B
- No conversions, no real signal yet — typical for step 0 from scratch on a new base

**Action required**: top up Prime wallet, then `prime rl run configs/lab/salesbench.toml` — config and env are already in place (v0.24.3 pushed, salesbench.toml at v36 settings, no checkpoint).

**Cost note**: at 30 min/step for 35B + 2 leads, 500 steps ≈ 250 GPU-hours = ~10 days wall-clock. Monitor wallet proactively before any large-scope curriculum step.

---

## 2026-05-07: v36 — Floor-reduced reward + new base model (LAUNCHED then STOPPED)

**Run**: `geh9smdj2wxkzpjn2pro3dtk`
**Model**: `Qwen/Qwen3.5-35B-A3B` (was Qwen3-30B-A3B-Instruct-2507 — RETIRED by Prime)
**Config**: 2 leads, 1 hour, env v0.24.3, **no checkpoint** (v31 LoRA dead — wrong base), 500 max_steps
**Reward changes** (ceiling 1.60 → 1.42):
- Removed `reward_quote_coverage` entirely (env constraint already enforces quote-before-propose; reward was a redundant floor)
- `reward_episode_completion` weight 0.10 → 0.02 (tie-breaker, not a floor)
- Floor when no conversions: 0.15 → ~0.01 (**15× reduction**)

**Forced events**:
- Old base model retired during the long pause between runs. Available now: Qwen3.5 family (0.8B–397B), Qwen3.6-35B-A3B, Llama 3.2, Nemotron, gpt-oss. Closest successor is Qwen3.5-35B-A3B (same MoE arch, 3B active).
- v31 LoRA checkpoint is dead — different base, weights won't load. Must train from scratch.
- Curriculum reset 12 → 2 leads (proven fresh-GRPO sweet spot, rule #36).

**Hypothesis**: with the floor gone, GRPO has no degenerate local optimum to lock onto. Conversion-seeking becomes the only path to >0.01 reward. New base may need 50–100 steps to match old model's baseline; 2 leads should be quickly mastered if reward shape is healthy.

**Watch for**: (1) does it climb past 0.5 in <50 steps? (2) does reward variance stay healthy (bimodal-ish)? (3) any error_type spikes from the new model?

**Curriculum plan (if v36 succeeds)**: 2 → 3 → 4 → 6 → 8 → 12 leads, scale only after >95% of ceiling for 50+ steps.

---

## 2026-03-19: v35 — Quote requirement + 12 leads from clean v31 (COLLAPSED)

**Run**: `t2odcytmh5975lxg62fzqroq` — COMPLETED 1095 steps (resumed from step 795)
**Config**: 12 leads, 2 hours, v0.24.2 (quote required), v31 checkpoint
**Trajectory**:
- Step 795 (start): reward 0.53, oa=4.5/12
- **Step 816 (peak): reward 1.002 (63% ceiling), oa=8.19/12 (68%), mrr=0.63, qc=0.98**
- Step 829 onward: full collapse — oa=0, mrr=0, qc=1.00, reward stuck at ~0.13–0.15 for 265 more steps
- Step 1094 (final): reward 0.147, 15.4 offers proposed, 14.4 rejected, **0 accepted**

**Mechanism**: GRPO found the qc+completion floor (0.15) and reinforced it. Quote→propose→fail satisfies qc=1.0 and gets half completion bonus, no exploration cost. Real conversions are high-variance, so once the floor strategy paid out reliably the gradient died.

**Key lesson**: workflow rewards (quote_coverage, episode_completion) become floor traps when their guaranteed payout is comparable to the variance of the outcome reward. Either remove them or make them ≪ outcome reward variance.

---

## 2026-03-17: v34c — Quote requirement + 15 leads (STOPPED)

**Run**: `zs5bpb9yt9fi1suaqivm2pfr` — ran ~970 steps total
**Config**: 15 leads, 3 hours, v0.24.2 (quote required before propose), checkpoint from v32 (step 800)
**Changes**:
- Runtime: `propose_offer` requires prior `quote_plan` for the lead
- Reward: added `reward_quote_coverage` (w=0.10), conv 0.15→0.10, budget_util stays 0.30
- Ceiling unchanged at 1.60

**Result**: peak **1.387 (87% ceiling) at step 918**, then collapsed (60 steps of zero conversions, unstable recovery). Stopped Mar 19 to launch v35 from cleaner v31 weights.

**Early progress (first 32 steps)**:
- Reward: 0.43 → 0.94, climbing steadily
- Quote coverage: 0.09 → 1.00 — model fully learned to quote
- Conversions: 4.7/15 → 10.6/15 (71% on best step)
- Invalid actions: ~11/ep, error rate 40–85%

**Key finding**: Reward signal alone (w=0.10, w=0.20) didn't teach quoting in v34/v34b — GRPO can't reinforce behaviors absent from rollouts. Hard env constraint was needed. But **adding the constraint plus keeping the qc reward created the floor trap** that killed v34c and v35.

---

## 2026-03-17: v33 — 20 leads (STOPPED, degenerate)

**Run**: `tg5h5av4oqvnjygtu771dum7`
**Result**: Model skipped CRM search (0.69 vs 6.81) and quoting (0.0 vs 1.3) entirely. Reward hacking — blindly proposing memorized prices. Peak 1.481 with 27% errors. Conv 19.2/20 (96%) but terrible sales process. 100% error steps (zero clean) in 34 steps.

---

## 2026-03-17: v32 — 12 leads (STOPPED, instant mastery)

**Run**: `pfth19p99h6yx3x8kjjorhvm`
**Result**: Peak **1.535** (96% ceiling) in just 8 steps! Conv 11.74/12 (98%). Instant curriculum transfer. Stopped to scale.

---

## 2026-03-16-17: v31 — 8 leads (STOPPED)

**Runs**: `qrholap10uc2aktxjtva4kr7` + `zejyompwevhcpkal7ulhre48`
**Result**: Peak **1.374** (86% ceiling) in 155 steps. Conv 6.95/8 (87%). Clean avg 1.36.

---

## Historical Summary

| Version | Leads | Peak | Ceiling% | Steps | Key Result |
|---------|-------|------|----------|-------|------------|
| v25 | 2 | 1.451 | 91% | 200 | Saturated |
| v28 | 4 | 1.528 | 95.5% | 370 | Mastered |
| v29 | 5 | 1.580 | 98% | 210 | Mastered |
| v30 | 6 | 1.475 | 92% | 40 | Fast |
| v31 | 8 | 1.374 | 86% | 155 | 51% error rate |
| v32 | 12 | 1.535 | 96% | 8 | Instant mastery (no quoting — degenerate) |
| v33 | 20 | 1.481 | 93% | 34 | DEGENERATE — no quoting |
| v34c | 15 | 1.387 | 87% | 970 | Collapsed step 918; quote req + qc reward |
| v35 | 12 | 1.002 | 63% | 1095 | Collapsed step 829; floor trap (qc+completion) |
