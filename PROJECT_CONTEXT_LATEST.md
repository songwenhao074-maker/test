# Project Context Latest

## Current authorized next experiment

Protocol-043 revision1 is registered but not implemented or started. Read [the full directive](docs/PROTOCOL043_BOUNDED_LIFECYCLE_20261003.md) and [registration](artifacts/ftmoe_online/protocol_043/plan.json).

The user clarified that the goal is a logically coherent D exceeding C, not the highest-performing D and not strict expert replacement. The agreed design is demand-driven addition, recent-window sleep, bounded dormant retention, and evidence-gated permanent reclamation when new capacity is required. Active experts are never permanently deleted.

Two arms only: D_no_gc and the predesignated primary D_bounded, differing only in reclamation enablement. Same seen seed3601; no independent generalization claim. Main success no longer requires beating A_hist/D_keep/no-GC; retain C performance safeguards and separately report natural lifecycle, reuse and reclamation coverage. If no natural reclamation occurs, say so; do not force events or add streams.

## Latest completed science (historical)

Protocol-042 revision2 science run37128097274; analysis/publication recovery run37130973490. Publication baseline commit19445006dee657884bca5cd800f477a8428f6493. Frozen execution code da5eaa7e71f44497ede538c884e7ca688e6afbf4 (Python files in repository root). Historical system label invalid_execution remains unchanged.

Published full AP: C .6828006987524914; R_win128 .7014930612986227; A_hist .7061360620708045; A_win128 .7057063219482922. R rejected both replacement candidates. A_win128 accepted E1 at1343, slept E0 at1615 and reused E0 at2351. These are historical audit observations, never hardcoded043 events.
At1615 E0 current-epoch cumulative utility was +.0122337193 versus128-window -.0062407245. Window decisions occurred despite the published analyzer reporting otherwise. Window AP was slightly below A_hist, which does not violate the user's clarified objective.

Identified gaps to recheck from sealed raw: online float32 subtraction versus registered/offline float64; contradictory first-divergence audit; missing active_ids_before in sleep-event selection audit; nonincremental history scanning; snapshot-only resume fixtures; "independent" fixture sharing the scorer implementation. Historical numeric improvements do not repair validity by themselves. Store corrected read-only audit under043/audit042 and preserve original042 reports.

## Scope and budgets

033–042 budgets remain closed. This update authorizes a new plan, not a current scientific launch. After user handoff, the executing model follows043's registered implementation/preflight/freeze/two-arm execution/delivery sequence. Maximum1536 new scientific expert optimizer steps, four post-E0 candidate attempts per arm, exact resume only. Permanent deletion applies solely to eligible dormant online objects in the new043 experiment; never delete historical scientific artifacts. Stop on engineering failure or after delivery; no automatic044.
