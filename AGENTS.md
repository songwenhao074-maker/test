# Current task: Protocol-030 paired passive-diagnostic validation

Read NEXT_EXPERIMENT_LATEST.md, docs/GITHUB_EXPERIMENT_HANDOFF.md and docs/PROTOCOL030_SINGLE_TASK_DIRECTIVE_20260922.md.

Protocol029 finished but diagnostic_valid=false; its Actions final validity gate failed. The user's new instruction authorizes the single Protocol030 task: implement and run one same-job audit_off/audit_on pair under a preregistered single-thread deterministic CPU profile, report memory utility only with explicit validity, publish results and stop.

030 is planned, not implemented/run. Full replay budget: two executions of the same Protocol028 online method, no new scientific comparator and no C rerun. Preserve all historical data, registrations and failed results. Do not relax the 1e-6 tolerance or retroactively relabel 029 valid. Historical028 matching is reported separately from new paired noninterference.

No automatic additional full replay, seeds, parameter search, A/B, threshold changes or scenario redesign. Start from current main, use a dedicated 030 push workflow, and do not trigger training from documentation/result updates. Synchronize README/AGENTS/NEXT/PROJECT_CONTEXT/current handoff when completed. The latest directive supersedes prior waiting instructions; old failure evidence remains immutable.
