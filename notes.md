# SalesBench Lab Notebook

## 2026-05-07: v36 plan — Drop floor reward to break GRPO collapse (DRAFTED)

**Diagnosis from v35**: model locked onto qc(0.10) + completion(0.05) = 0.15 floor reward and gave up exploring conversions. 15.4 offers/ep, 14.4 rejected, 0 accepted — pure quote→propose→fail loop.

**Lever (single rebalance)**:
- Remove `reward_quote_coverage` entirely — env constraint already enforces quote-before-propose, so reward was 100% redundant and a free 0.10 floor
- `reward_episode_completion` weight 0.10 → 0.02 — keeps tie-breaker, removes 0.05 floor from time_budget_exhausted

**New ceiling**: 1.42 (was 1.60). New floor (no conversions): ~0.01 (was 0.15) — **15× reduction**.

**Plan**: bump env to v0.24.3, restart from v31 checkpoint (step 795, clean 8-lead weights), 12 leads, 2 hours. Same proven config (lr=1e-5, temp=1.0, batch=128, oversampling=2.5, max_async=1).

**Hypothesis**: with the floor gone, GRPO has no degenerate local optimum. Conversion-seeking becomes the only path to >0.01 reward.

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
