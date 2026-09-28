# Current task: Protocol-034 three-hypothesis experiment

Read NEXT_EXPERIMENT_LATEST.md, docs/PROTOCOL034_THREE_HYPOTHESES_20260928.md and artifacts/ftmoe_online/protocol_034/plan.json. The user explicitly authorizes this next experiment for a subsequent executor. Historical033 stop restrictions do not prohibit034. Preserve all031/032/033 source and results.

This handoff is planned_not_implemented_not_run. No034 training or workflow is launched by the planning/upload turn. Use source0463cd365b4cda5a50423b4fccc941fe8d074bbf and frozen033 artifact10968752245/run36417604442. Verify the registered ZIP/stream SHA256. No new simulation, truncation, reroll or alternate data.

Budget: exactly three full training replays, C_ref, D_frozen_ref and D_live, sequential isolated processes in one environment. D_live changes stable active specialist/router trainability only; retain dormant freezing, moments, original loss/features/budgets/gates. Full/Fragment memory observations must not affect D_live training or lifecycle.

The second hypothesis uses paired complete/fragment snapshots with identical creation times and candidate support. Hindsight best-window selection is diagnostic only and must never be described as an online score.

The third hypothesis performs two no-training cache passes, D_route_full and D_route_fragment. Use per-host last32 matured prediction losses; only labels published strictly before current prediction are allowed. Snapshot IDs/support identical across both arms. Keep the registered1pct fallback rule; never tune after viewing results or use phase/service IDs. Routing does not feed back into D_live gradients.

Run the plan validator and required causal/freeze/optimizer/observer fixtures before full replays. Preserve all failed outputs. Result serialization recovery uses saved predictions, not extra training. Stop after registered budget or explicit blocker; no automatic seeds/grids/followups.

Write compact results to docs/PROTOCOL034_RESULTS.md and run-scoped evidence in Git; store raw arrays/checkpoints/cache as indexed artifacts with hashes. Synchronize main and execution-branch current pointers. New workflow is workflow_dispatch only. Documentation commits use [skip ci].
