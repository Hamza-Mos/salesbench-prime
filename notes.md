# SalesBench Lab Notebook

## 2026-03-17: v34c — Quote requirement + 15 leads (RUNNING)

**Run**: `zs5bpb9yt9fi1suaqivm2pfr`
**Config**: 15 leads, 3 hours, v0.24.2 (quote required before propose), checkpoint from v32 (step 800)
**Changes**:
- Runtime: `propose_offer` requires prior `quote_plan` for the lead
- Reward: added `reward_quote_coverage` (w=0.10), conv 0.15→0.10, budget_util stays 0.30
- Ceiling unchanged at 1.60

**Results so far (32 steps)**:
- Reward: 0.43 → 0.94 (best), climbing steadily
- Quote coverage: 0.09 → **1.00** — model fully learned to quote!
- Conversions: 4.7/15 → 10.6/15 (71% on best step)
- Invalid actions: ~11/episode (high but stable — mostly bad params)
- Error rate: 40-85% every step, zero clean steps
- Budget util: still ~99% when model converts

**Key finding**: Reward signal alone (w=0.10 and w=0.20) didn't teach quoting — GRPO can't reinforce behaviors absent from rollouts. Hard env constraint was needed.

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
| v32 | 12 | 1.535 | 96% | 8 | Instant mastery |
| v33 | 20 | 1.481 | 93% | 34 | DEGENERATE — no quoting |
| v34c | 15 | 0.936 | 59% | 32 | Learning with quote req |
