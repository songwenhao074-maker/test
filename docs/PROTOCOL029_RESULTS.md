# Protocol-029 Results

GitHub Actions run: 35711135230.
Task: passive dormant-memory utility audit on one unchanged Protocol-028 D_memory_protected replay.
Diagnostic valid: False.
C rerun: False. New training arms: 0.

## Source-trajectory consistency

- Probability max absolute error vs Protocol-028: 0.000001192 (required <=1e-6).
- Full-stream AP absolute difference: 0.000000091 (required <=1e-6).
- Discrete lifecycle events exact: True.
- Passive state checks/failures: 5657/0 failures.

## Opportunity audit

- Reuse due opportunities: 154 (Protocol-028 source: 154).
- No-memory opportunities: 94.
- Stream-end censored opportunities: 1.
- Opportunities with at least one candidate satisfying every original reuse gate: 5.
- Such opportunities blocked by busy state: 5.
- Such opportunities blocked by similarity/rank: 5.
- Original selected candidate also passed all gates: 0.

## Main finding

Passive audit found 5 candidate-opportunity records that satisfy all original gates. They are post_hoc_oracle_diagnostic evidence of missed utility opportunities only; they do not establish an online D>C result.

## Nine recurrence first100 windows

| Window | Live AP | 4-generalist AP | Full-window dormant oracle | Notes |
|---|---:|---:|---|---|
| S1_rec1 | 0.694025 | 0.687444 | NA | no dormant full-window coverage |
| S3_rec1 | 0.801780 | 0.799600 | NA | no dormant full-window coverage |
| S2_rec1 | 0.681088 | 0.667778 | NA | no dormant full-window coverage |
| S4_rec1 | 0.681380 | 0.688207 | expert 15 / AP 0.686110 | post-hoc oracle; not online policy |
| S2_rec2 | 0.735750 | 0.732600 | expert 15 / AP 0.735202 | post-hoc oracle; not online policy |
| S6_rec1 | 0.870468 | 0.869262 | expert 15 / AP 0.870566 | post-hoc oracle; not online policy |
| S1_rec2 | 0.825311 | 0.813914 | expert 17 / AP 0.821226 | post-hoc oracle; not online policy |
| S5_rec1 | 0.800463 | 0.771417 | expert 17 / AP 0.789614 | post-hoc oracle; not online policy |
| S3_rec2 | 0.809864 | 0.807582 | expert 9 / AP 0.808848 | post-hoc oracle; not online policy |

## Diagnostic overhead

- Diagnostic replay wall seconds: 1130.026914.
- Diagnostic replay CPU seconds: 1413.787980.
- Passive prediction overhead seconds: 200.077934.
- Passive guard overhead seconds: 11.902724.
- This is diagnostic overhead and is not a new method-cost comparison against historical C.

## Interpretation boundary

All best-candidate choices in this report are post_hoc_oracle_diagnostic. Busy counterfactuals and passive candidates are not deployable policy results and are not formal D>C evidence.

## Evidence

- Artifact: https://github.com/songwenhao074-maker/test/actions/runs/35711135230/artifacts/10688265540
- Artifact SHA-256 digest: 56c5cbe760cea17abeb62b9bbb3d30116ecef338f2fcaa9afda3fe0eaf80bb16
