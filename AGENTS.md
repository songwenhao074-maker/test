# FT-MoE current experiment scope

Read `NEXT_EXPERIMENT_LATEST.md`, then `docs/GITHUB_EXPERIMENT_HANDOFF.md`.
The active protocol is 027: one C_fixed5 versus D_dynamic development pilot on the registered six-service stream. Each directive contains one bounded task. Report its result and stop; do not automatically add A/B, extra comparators, ablations, seeds, or tuning.

Use the current default branch `main`. Historical protocol files and old branch instructions document prior work; their D_eligible/STOP rules and five-arm plans do not govern protocol027. Shared older Python modules remain runtime dependencies and must not be deleted merely because their filenames are old.

Preserve historical evidence and the immutable physical stream. Fix deterministic implementation mismatches; do not relax physical, causal, hash, feature-consistency, or class-coverage checks to obtain a favorable result. Report performance only after actual model runs. The normal-only guard policy is defined by protocol027, not the old protocol025 two-class guard.

The active workflow is `.github/workflows/protocol027-pilot.yml`. Push checks are audit-only. Set workflow_dispatch input `run_models=true` to execute the registered two arms. A passed maintenance audit is not a completed model experiment.
