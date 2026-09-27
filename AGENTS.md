# Current task: Protocol-032 single guard-budget pilot
Read NEXT_EXPERIMENT_LATEST.md, docs/PROTOCOL032_SINGLE_TASK_DIRECTIVE_20260927.md, docs/PROTOCOL031_INTERIM6800_ANALYSIS_20260927.md, and artifacts/ftmoe_online/protocol_032/plan.json.

One task only: verify the existing6800-row prefix, then run C_fixed5 and D_guard_budget once each on the same data. No simulator generation, continuation to9869, seed sweep, A/B, or automatic followup.
Only change the F0 normal-NLL acceptance bound for D birth/reuse to L_candidate <= L_live + max(0.02*L_live,0.01) + 1e-6. This is an explicitly relaxed development hypothesis after observing031 outcomes, not a bug fix or independent confirmation. Preserve every other predictive gate, training budget and causal rule. Stop after one pair or an explicit blocker.

The source implementation is commit4e7df7e5f7abca2e2470633864b35440858bd4bf on protocol-031-rare-recurrence-20260922; main does not yet contain that implementation. Reusing that branch is permitted: apply this latest directive before execution. Historical031 stop/no-threshold-change restrictions apply to031, not the newly specified032; never overwrite historical modules/registrations/results.
The completed031 interim run36295124291 had no dormant experts or reuse and D was slightly worse and18x slower in measured replay wall time. A snapshot memory registry ID is not evidence of a dormant expert. Read original artifacts and the verified evidence JSON.
Input eligibility must be computed, never hard-coded true. Upload the frozen input bundle and compact evidence/report to GitHub. Keep results even on failure and synchronize main entry points. Documentation-only commits use [skip ci]; do not launch training in this planning turn.
