# Protocol-024 v2c verified development result

Run `35074464132` is the accepted replacement-consistent development replay on immutable stream SHA `468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946`, replay seed 700, model seed 1. Confirmation seeds 701–703 and test seeds 201–205 were not used.

## Engineering qualification

All required engineering gates passed. Replacement preview/deployment equivalence was verified at atol/rtol 2e-6 for no-old-specialist, old-active-specialist, and dormant-reuse paths; legacy preview was audit-only. Opportunity accounting conserved exactly: birth due/accounted 18/18; reuse due/accounted 137/137, with 23 started, 91 busy, 2 no-memory, 21 no causal representation match, 0 capacity skips. Candidate conservation is 18 created = 5 accepted + 12 rejected + 1 pending; the pending candidate is stream-end censored. Lifecycle totals are birth=5, retirement=4, reactivation=0, purge=2.

Strict checkpoint recovery used the real accepted-birth checkpoint at t=764. The uninterrupted prefix matched the main run; next prediction matched; the next complete live update matched in model hash, Adam hash, RNG hash and topology; the prediction after that update matched; and the next non-empty lifecycle event (`specialist_crossfade_step`) matched. `checkpoint_recovery_verified=true`.

## Scientific result

Full-stream C detection AP/Recall/FPR/resource macro-F1/onset AP = 0.6695451 / 0.6796187 / 0.0379895 / 0.8387937 / 0.1611756. D-v2c = 0.6670999 / 0.6810631 / 0.0378383 / 0.8393727 / 0.1591909. D-v2c-no-reuse is byte-for-byte equivalent in reported full metrics because no reuse candidate was accepted/reactivated.

The nine first100 D-v2c−C AP deltas are: R1_first +0.0000000; R2_first -0.0047767; R3_first +0.0004180; R1_rec1 +0.0055102; R3_rec1 -0.0006323; R2_rec1 +0.0034591; R1_rec2 -0.0022645; R2_rec2 -0.0040456; R3_rec2 -0.0025991.

For the six preregistered recurrence windows, only 2/6 are positive and mean D-v2c−C AP = -0.00009536. Aggregate recurrence AP is C 0.6982942 versus D-v2c 0.6977299; Recall is 0.7062147 versus 0.7084746; FPR is 0.03786575 versus 0.03763626, delta -0.00022949. The +0.03 development reference is therefore not met: `development_signal=false`.

The read-only no-match counterfactual used future labels only post hoc and never influenced online selection. It is retained as diagnosis, not as a deployable policy result.

## Decision

The v2c semantic fix is verified, but specialist memory still has no meaningful predictive advantage on this stream. Per `PROTOCOL024_NEXT_DIRECTIVE_20260916.md`, do not tune v2c thresholds and do not launch confirmation seeds. Advance to a separately preregistered Protocol-025 service-entry/exit/recurrence scenario with explicit capacity constraints and stronger common causal prediction features.