# Current task: Protocol-033 complete5969-row rare-recurrence pilot

Read NEXT_EXPERIMENT_LATEST.md, docs/PROTOCOL033_6000_STEP_DIRECTIVE_20260928.md, artifacts/ftmoe_online/protocol_033/scenario_registration.json, and artifacts/ftmoe_online/protocol_033/plan.json.

The user explicitly replaces the9869-row generation plan with an approximately6000-row complete experiment for another model to execute. This authorizes a new033 stream; historical031/032 no-generation/stop restrictions remain historical and must not override this task. Preserve all historical results and source modules.

This handoff contains the plan and validator only;033 generator/replay adaptation is not yet implemented or run. Source implementation: a215c1161adba4ff29514eb88806762c2f068023 on protocol-031-rare-recurrence-20260922. Main does not contain every newer implementation. Apply these033 instructions before continuing that source.

Generate5968 scored rows plus1 terminal raw-label support row. Keep all six128-step recurrence windows. F0=300; U/V/W initial stages=1000 each; five W gaps=380 each. Birth schedule600+1000k matured intervals; preserve all other032 algorithm and guard settings. Do not truncate/splice old streams or resume mismatched checkpoints.

Use200-row chunks (29 full +169 final), sequential generation segments and separate finalization process. Measure effective memory limit and RSS; avoid duplicate full-stream allocations. Preserve simulator/scheduler/workload/RNG state exactly across segments. Run the plan validator and actual input audit; never hard-code eligibility=true.

After complete input qualification, run C_fixed5 and D_guard_budget once each, same job/environment, separate processes. Report every registered window and all measured costs, even on failure. This is development after seeing historical results, not independent confirmation or a hard-delete demonstration.

No extra seeds, threshold grids, A/B or automatic followups. New workflow must be workflow_dispatch only. This planning/upload turn runs no simulation or training; documentation commits use [skip ci]. After later execution, synchronize compact results and current pointers on main and the execution branch.
