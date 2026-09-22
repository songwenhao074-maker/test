# Current experiment: Protocol-027 data revision 002

Read NEXT_EXPERIMENT_LATEST.md and docs/GITHUB_EXPERIMENT_HANDOFF.md. Work from main.
The user authorized an explicitly versioned dataset on 2026-09-22, replacing the requirement to reproduce unavailable original file hashes. Original registration and failures remain historical evidence. The active data identity is artifacts/ftmoe_online/protocol_027/data_revision_002.json; do not apply old 025/026/027 STOP or no-revision rules to this authorized revision.

One bounded task remains: one C_fixed5 versus D_dynamic development pilot, seed700/model1, unchanged six-service scenario, methods and metrics. No automatic A/B, other comparators, seeds, ablations or tuning. Stop after results or an actionable blocker.

Prepare revision002 by copying the selected existing artifacts plus the pinned events sidecar. No simulator rerun and no checkpoint unpickling are needed. Audit all physical/causal/feature/coverage gates, freeze the full dataset and archive it successfully before either arm starts. Both arms must verify the same frozen files. Do not change the active data hashes after observing model results or claim equivalence to the missing original complete stream.

The sole workflow is .github/workflows/protocol027-pilot.yml. If the connector lacks workflow_dispatch, change this workflow file (a launch comment suffices) on protocol-027-pilot-gpt56-20260921 and push without [skip ci]; that push runs the registered two arms after audit/archive. Other paths and main pushes do not trigger it. Result commits do not retrigger it. run_models=true prepares, audits, archives and runs the two arms once. run_models=false only prepares/archives. To reuse a complete archived bundle, pass its run ID as frozen_data_run_id; do not simulate again. Use [skip ci] when synchronizing workflow maintenance without intending to launch a run. Do not automatically launch further trials after a result.

Keep imported historical code, data and checkpoints. Report model performance only after actual model runs. D's extra background compute and memory must remain disclosed.
