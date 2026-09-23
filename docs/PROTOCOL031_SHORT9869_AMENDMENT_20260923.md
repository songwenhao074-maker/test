# Protocol-031 short9869 user amendment

Date: 2026-09-23

The user explicitly shortened the active Protocol-031 development run from the original 15469-row horizon to **9869 rows total (9868 scored intervals + 1 guard row)** because the full simulator generation was taking too long.

## What remains unchanged

- Same deterministic seed700 simulator trajectory prefix.
- Same model seed1.
- Same simulator/workload laws and event probability before the new horizon.
- Same C_fixed5 and D_nonblocking_reuse model definitions and online budgets.
- No reroll, extra seed, extra scenario, A/B arm, threshold search, ablation, or tuning.
- Existing generated prefix is reused; no regeneration of already completed intervals.

## What changes

- Data collection stops/freeze occurs at row 9869.
- If a source checkpoint has already progressed past 9869, only the deterministic prefix `0:9869` is frozen for the amended comparison.
- The original six recurrence windows are no longer all present. Before the new horizon, only `U_rec1` and `V_rec1` are available as recurrence128 primary windows.
- The amended development comparison therefore uses the equal-weight mean AP(D-C) over those two available recurrence128 windows. The positive-window reference becomes 2/2, while the existing mean-AP, pooled-normal-FPR, and W-block-delta numerical thresholds remain unchanged.

## Interpretation boundary

This amendment was requested **after generation had already started**. The short9869 result is development-only and must not be presented as the original revision002 six-window preregistered primary result. It is a deterministic prefix analysis used to reduce generation cost and obtain an earlier mechanistic C/D signal.

The original Protocol-031 registration and source hashes are retained as provenance. New short-horizon tooling freezes the prefix into a separate immutable bundle before either C or D begins.
