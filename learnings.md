# Experiment Learnings

Structured insights from autoresearch experiments. Each entry captures what was tried, what happened, and the takeaway — so future runs don't repeat dead ends and can build on what worked.

## Format

Each learning follows this structure:

```
### [tag] Short title
- **Experiment**: what was changed
- **Result**: val_bpb and whether it improved/regressed (include delta)
- **Why**: root cause analysis — why did it help or hurt?
- **Takeaway**: actionable rule for future experiments
```

---

## Learnings

### [mar15b] Increasing depth hurts on MPS at fixed time budget
- **Experiment**: depth 4→8 (exp1) and 4→6 (exp2), keeping aspect_ratio=64
- **Result**: val_bpb regressed from 1.412 to 2.065 (depth=8) and 1.714 (depth=6)
- **Why**: With aspect_ratio=64, depth=N gives model_dim=N*64. Deeper models are larger and slower per step, so they complete far fewer optimization steps in 5 minutes on MPS. The baseline (depth=4, dim=256) gets enough steps to converge well; depth=8 (dim=512) gets ~2x fewer steps and doesn't compensate with per-step quality.
- **Takeaway**: On this MPS device, depth=4 is likely near-optimal for throughput. Focus on hyperparameter tuning, architecture tweaks that don't increase per-step cost, or finding a better depth/width tradeoff (e.g. lower aspect_ratio with slightly more depth).
