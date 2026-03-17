# SalesBench Lab Notebook

## 2026-03-17: v32 — Scale to 12 leads

**Run**: `pfth19p99h6yx3x8kjjorhvm`
**Config**: 12 leads, 2 hours, checkpoint from v31 (step 795), max_steps=1095 (300 steps)
**Hypothesis**: v31 reached 86% ceiling at 8 leads in 155 steps. Aggressive scaling continues. total_hours=2 gives 10 min/lead. Expect curriculum transfer to start ~0.9+.

**Status**: Just launched. Monitoring.

---

## 2026-03-16-17: v31 — 8 leads (STOPPED at step ~805)

**Runs**: `qrholap10uc2aktxjtva4kr7` (45 steps), `zejyompwevhcpkal7ulhre48` (155 steps)
**Config**: 8 leads, 1 hour, checkpoint from v30 (step 610/650)
**Result**: Peak **1.374** (85.9% ceiling) at step 795. Clean avg 1.36. Conv 6.95/8 (87%). Budget util 99.9%.
**Key**: Curriculum transfer confirmed (started at 1.06). Error rate very high (51%) but model always recovers. Went from 1.06 → 1.37 in 155 steps. Seq_len ~7.5k.

---

## Historical Summary

| Version | Leads | Peak | Ceiling% | Steps | Key Result |
|---------|-------|------|----------|-------|------------|
| v25 | 2 | 1.451 | 91% | 200 | Saturated |
| v26 | 3 | 1.354 | 85% | 200 | Plateaued step 130 |
| v27 | 3 | 1.400 | 88% | 111 | Clean avg ~1.38 |
| v28 | 4 | 1.528 | 95.5% | 370 | Mastered 4 leads |
| v29 | 5 | 1.580 | 98% | 210 | Mastered 5 leads |
| v30 | 6 | 1.475 | 92% | 40 | Fast learning, scaled quickly |
| v31 | 8 | 1.374 | 86% | 155 | 51% error rate, still learned |
| v32 | 12 | - | - | - | RUNNING |
