# Protocol-024 continuation status — 2026-09-14

Source audited: `main @ 92f9559593de41768d7cc4f3ddf40656b95850d5`.

## What was completed

- Re-read `NEXT_EXPERIMENT_LATEST.md`, `docs/GITHUB_EXPERIMENT_HANDOFF.md`, and `docs/ONLINE_D_ANALYSIS_AND_NEXT_SCENARIO_20260914.md`.
- Confirmed Protocol-023 must remain immutable; the new protocol must first fix evaluation/integrity issues and then test A/C/D on a new business-response scenario.
- Reconfirmed the onset-axis bug, the missing penultimate settlement (index 2878, 16 host-steps per arm), and the diagnosis-metric issue.
- Re-read the mature-gradient evidence: the old three resource-order regimes are mostly co-directional, so that scenario does not establish the originally hypothesized fixed-C conflict.
- Built and locally tested corrected Protocol-024 evaluator helpers: same-host temporal onset, complete two-row tail settlement, and positive-only resource macro-F1.
- Built and locally tested a fair dynamic residual-bank prototype that starts from the same 4-expert `64->32->5` residual bank as fixed C, supports shadow candidates, ramp-0 output-continuous activation, retirement, same-id reactivation, and max capacity 8.
- Local regression result: **9/9 tests PASS**.

## Critical architecture finding

Do not use Protocol-020 `OnlineEAGateV3` directly as the Protocol-024 D arm. Its lifecycle logic is useful reference code, but it changes the original model's EAGate, whereas Protocol-023 fixed C is `FrozenResidualFTMoE` with a separate four-expert residual bank. Direct comparison would confound lifecycle with model/router architecture. Port the V3 lifecycle logic around the same residual bank used by C.

## Existing numerical evidence retained

Saved seed-700 late detection AP gains C-A are +0.0049/+0.0020/+0.0203 on first compute/memory/I/O exposure (mean +0.0091) and +0.0571/+0.1297/+0.0972 on recurrence (mean +0.0947). Corrected same-host raw h=1 onset gives essentially no first-exposure gain and a small negative mean recurrence difference, so the current correction must not be described as demonstrated early warning.

## Next execution order

1. Integrate the dynamic residual bank into a strict-prequential D session with the same replay, anchors, loss, optimizer family, update opportunities and prediction-before-label order as C.
2. Add topology-aware save/resume and causal shadow-validation tests.
3. Build a new response-law generator: observable history should distinguish modes with different future outcomes; do not only permute CPU/RAM/Disk order. Start with seed 700 only, long first-learning blocks (about 800+ intervals per mode) and short recurrence (120–240), then expand toward 6–8 modes for the actual lifecycle claim.
4. Check learnability both from allowed raw/history features and directly from the frozen 64-D residual input `z`.
5. Freeze the online budget and residual-router novelty calibration before comparing D and C.
6. Run minimal A/C/D pilot. Primary metric: mean detection AP over the first W=100 intervals after registered switches, plus paired D-C switch differences, worst mode, full AP, corrected onset if warning is claimed, positive-only resource F1, FPR, lifecycle events, p95 latency and peak memory.
7. Only after a material pilot signal, add fixed sparse 8-expert pool, C-budget, D-no-birth, D-no-reactivation, D-no-retirement, then freeze rules and run confirmation seeds 701/702/703.

## Not completed in this execution environment

No new Protocol-024 model training or stream collection was run because the connected GitHub reader could not materialize the private checkpoint/NPZ binaries into the execution runtime. The test/implementation handoff files were produced locally. Historical Protocol-023 files were not modified.
