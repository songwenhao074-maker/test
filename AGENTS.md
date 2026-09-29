# Active Experiment Directive

Protocol-035 implementation has been committed to its execution branch; the formal workflow exists on main. No Protocol035 scientific run existed at this launch-amendment review. The immutable scientific registration retains its original publication status. Publishing this amendment does not launch it.

Authoritative directive: `docs/PROTOCOL035_C_PRESERVING_CORRECTION_20260929.md`.
Machine-readable registration: `artifacts/ftmoe_online/protocol_035/plan.json`.
Validate first: `python maintenance/validate_protocol035_plan.py`.

Execution branch: `codex/protocol-035-c-preserving-correction-20260929`, based on implementation commit `cdf4b1ce3885ef5f5ac67b4265074faa8f7310eb`. Main does not contain all earlier experiment implementation files. After the user hands off this directive, implement and run the registered scope: one existing-cache diagnostic suite, one C reference replay, one correction-branch training replay. No new stream, grid, extra seed, old-D retraining or automatic follow-up. Exact checkpoint resume is permitted; restarting a consumed scientific replay is not.

D_corr is a minimal proposed D component: preserve the entire continuously learning C path and add one input-conditioned correction branch. It does not yet implement dynamic memory. Report full-stream and recurrence results separately and enforce recall/FPR guardrails.

Protocol-034 is complete (run 36436945806). Its exhausted budget remains closed; Protocol-035 has a separate future execution budget. Preserve Protocol-031/032/033/034 source/results. Use `[skip ci]` for plan/report commits. Synchronize all three entry documents and results on main and the actual execution branch at handoff/completion. Do not claim success from a plan validator or workflow exit status alone.

## Launch amendment (binding)

Read `docs/PROTOCOL035_LAUNCH_AMENDMENT_20260929.md` and `artifacts/ftmoe_online/protocol_035/launch_amendment.json`. User authorization covers fixing the missing-first-dispatch blockage. A connector without dispatch is not by itself a blocker: use REST/gh if available, or the explicitly authorized one-shot push helper template under `docs/templates/`. The helper only dispatches the formal workflow, which stays workflow_dispatch-only. The blanket no-push-helper rule is superseded for this one bridge. Its deliberately armed activation commit is the sole skip-ci exception. Do not change plan.json, its validator, scientific budgets or existing experiment code for this amendment. Check for existing formal runs before any dispatch, record the run id, then remove the helper. Do not treat rerun tools as authority to restart consumed scientific sequences.
