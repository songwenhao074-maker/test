# Protocol 020 continuation — 2026-09-08

> Archived execution registration. The 2026-09-08 runs are complete; the active plan is [Protocol 020 revised experiment plan, 2026-09-09](FTMOE_PROTOCOL020_REVISED_EXPERIMENT_PLAN_20260909.md). Begin with R0 validation corrections; do not rerun this sequence as the current plan. The original offline v4 remains protected.

The user authorized continued implementation and experiments with the objective of stable, strong online performance of D relative to A/B/C. Simulator settings and workload distributions may be changed. The original offline v4 model must not be retrained or overwritten.

This continuation keeps all existing offline checkpoints unchanged, including the existing Protocol 020 S6 warm start. Every new run starts from the same saved checkpoint. Online updates operate on in-memory copies and are saved only in new run directories.

## Development experiment order

1. Audit the S7 learning, measurement, and checkpoint-resume code. Compare the existing learning rate with 1e-4 on the existing development drift stream. Recheck stationary stability before selecting an online configuration.
2. Integrate Dynamic Expert v3, with causal matured-label signals, shadow validation, gradual activation, dormant expert reuse, and preserved optimizer state. Run D with the same online learning setup as C.
3. Compare A/B/C/D on the same streams. Preserve failed runs and distinguish failure to trigger from a triggered mechanism without benefit.
4. Diagnose shortcomings and write the next experiment plan if D does not show a stable advantage. Development results are not final held-out confirmation. Any scenario adjusted using model feedback is exploratory and must later be tested on untouched VM sources and seeds.

## Constraints

- No offline checkpoint retraining or replacement.
- One model or simulator process at a time; retain the 3.0 GiB available-memory guard.
- Preserve prior results and the pre-existing modification in `prepare_ftmoe_protocol020_capacity_scan.py`.
- No future labels, phase identifiers, or future capacities in D decisions.
- Do not claim a numerical advantage proves general effectiveness without independent repeated evaluation.

The current user authorization supersedes the older instruction to stop indefinitely after a failed development gate. Failed gates remain recorded and motivate diagnosis; they are not silently relabelled PASS.
