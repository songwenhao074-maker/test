# Active Experiment Directive

Protocol-035 is **planned_not_implemented_not_run**. The user requested a written plan published for another model to execute. Publishing this plan must not dispatch Actions or start simulations/training.

Authoritative directive: `docs/PROTOCOL035_C_PRESERVING_CORRECTION_20260929.md`.
Machine-readable registration: `artifacts/ftmoe_online/protocol_035/plan.json`.
Validate first: `python maintenance/validate_protocol035_plan.py`.

Execution branch: `codex/protocol-035-c-preserving-correction-20260929`, based on implementation commit `cdf4b1ce3885ef5f5ac67b4265074faa8f7310eb`. Main does not contain all earlier experiment implementation files. After the user hands off this directive, implement and run the registered scope: one existing-cache diagnostic suite, one C reference replay, one correction-branch training replay. No new stream, grid, extra seed, old-D retraining or automatic follow-up. Exact checkpoint resume is permitted; restarting a consumed scientific replay is not.

D_corr is a minimal proposed D component: preserve the entire continuously learning C path and add one input-conditioned correction branch. It does not yet implement dynamic memory. Report full-stream and recurrence results separately and enforce recall/FPR guardrails.

Protocol-034 is complete (run 36436945806). Its exhausted budget remains closed; Protocol-035 has a separate future execution budget. Preserve Protocol-031/032/033/034 source/results. Use `[skip ci]` for plan/report commits. Synchronize all three entry documents and results on main and the actual execution branch at handoff/completion. Do not claim success from a plan validator or workflow exit status alone.
