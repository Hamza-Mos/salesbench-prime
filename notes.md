# SalesBench Lab Notebook

## 2026-03-16: v30b — Relaunch 6 leads

**Run**: `vdbsi8xocjn58kwzchm69ict`
**Config**: 6 leads, 1 hour, checkpoint from v29 (step 575), max_steps=1100
**Hypothesis**: v30a failed on env server timeout (transient cluster issue). Same config should work now. Expecting curriculum transfer: model starts ~1.0+ reward and climbs toward 1.50+ ceiling.

**Status**: Just launched. Monitoring.

---

## Historical Summary

| Version | Leads | Peak | Ceiling% | Key Result |
|---------|-------|------|----------|------------|
| v25 | 2 | 1.451 | 91% | Saturated, GRPO signal exhausted |
| v26 | 3 | 1.354 | 85% | Plateaued step 130, errors 140-190 |
| v27 | 3 | 1.400 | 88% | Clean avg ~1.38 |
| v28 | 4 | 1.528 | 95.5% | Mastered 4 leads |
| v29 | 5 | 1.580 | 98% | Mastered 5 leads |
| v30a | 6 | - | - | FAILED (env server timeout) |
| v30b | 6 | - | - | RUNNING |
