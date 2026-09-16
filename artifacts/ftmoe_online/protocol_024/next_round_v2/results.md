# Protocol-024 next_round_v2 results

## v2b specialist memory — seed700/model1

Successful GitHub Actions run: `35051303612`. Immutable stream SHA256: `468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`. Confirmation seeds were not used.

Lifecycle: births=2 (specialists 4,5), retirements=3, reactivations=2, purges=0, purged_bytes=0. There were 49 reuse opportunities: 2 accepted, 8 rejected after validation, 2 skipped with no memory, and 37 skipped with no causal representation match. Checkpoint recovery after first reactivation was verified.

Full-stream C: detection AP=0.6695451244, recall=0.6796186624, FPR=0.0379894718, resource macro-F1=0.8387937258, onset AP=0.1611755734.

Full-stream D-v2b: detection AP=0.6671033068, recall=0.6773075256, FPR=0.0379482387, resource macro-F1=0.8381552140, onset AP=0.1578048723.

Nine first100 D-v2b minus C AP deltas: R1_first=+0.0000000; R2_first=-0.0170380; R3_first=+0.0018303; R1_rec1=-0.0003442; R3_rec1=+0.0004084; R2_rec1=+0.0032890; R1_rec2=-0.0018970; R2_rec2=+0.0037276; R3_rec2=+0.0050386.

Six recurrence windows (rec1/rec2 only): mean D-v2b minus C AP=+0.0017037292, 4/6 positive. Aggregated recurrence C AP=0.6982941698 and D-v2b AP=0.7004755054. Aggregated recurrence recall C=0.7062146893 vs D-v2b=0.7039548023; FPR C=0.0378657487 vs D-v2b=0.0380952381, delta=+0.0002294894.

Preregistered development rule: mean recurrence D-C AP >= +0.03, at least 4/6 positive, normal-FPR delta <= +0.01. Result: `development_signal=false` because the mean recurrence AP gain is only +0.0017037. Therefore the registered extra controls (C-u1/compute-matched C-budget, C-fixed8, D-no-memory, D-no-reactivation, and conditional D-no-purge) are not run. No tuning around this result is permitted.
