# Current task: execute Protocol-033 complete5969-row rare-recurrence pilot

Read `NEXT_EXPERIMENT_LATEST.md`, `docs/PROTOCOL033_6000_STEP_DIRECTIVE_20260928.md`, `docs/PROTOCOL033_IMPLEMENTATION_HANDOFF.md`, `artifacts/ftmoe_online/protocol_033/scenario_registration.json`, and `artifacts/ftmoe_online/protocol_033/plan.json`.

Protocol-033 implementation is now staged on branch `protocol-033-5969-gpt56-20260928`; no Protocol-033 simulator/model run has started yet. The implementation source remains commit `a215c1161adba4ff29514eb88806762c2f068023` from `protocol-031-rare-recurrence-20260922`. Historical Protocol-031/032 modules, registrations, data and conclusions are unchanged.

Generate exactly5968 scored rows plus1 terminal raw-label support row from one new continuous seed700 simulator trajectory. Use 200-row immutable chunks (29 full +169 final), exact simulator/workload/scheduler/RNG continuation, separate generation/assembly/audit/model processes, and registered RAM telemetry/80% soft limit. Never truncate/splice an old stream or reroll/retune after a failed audit.

Before any generation byte, the workflow must compile the Protocol-033 implementation, run `test_ftmoe_protocol033.py`, run `maintenance/validate_protocol033_plan.py`, and verify frozen Protocol-031/032 source files are unchanged from `a215c116...`. Full input qualification is independent and must recompute ratio/labels from `post_totals/capacities`; model runs are forbidden unless every required gate passes.

After qualification, run exactly one `C_fixed5` process and one `D_guard_budget` process sequentially on the same frozen input. D changes only the registered birth cadence to600+1000k matured intervals on top of Protocol-032 guard-budget policy and adds per-reactivation event logging. Evaluate all six registered128-step recurrence windows and all registered W blocks; do not select best windows or add seeds/threshold searches/A/B runs.

The launch workflow is `.github/workflows/protocol033-5969.yml` and is `workflow_dispatch` only. The first dispatch is a user-authorized execution start; later same-stream generation segments and the terminal assemble/audit/C/D stage self-dispatch through the same workflow. Preserve compact terminal evidence even on blockers and synchronize current pointers after execution.
