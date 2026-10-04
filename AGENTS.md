# Active Experiment Directive — Protocol-044 revision 2 registered, not executed

Read [NEXT_EXPERIMENT_LATEST.md](NEXT_EXPERIMENT_LATEST.md), [revision 2 instructions](docs/PROTOCOL044_REVISION002_DURABLE_RECOVERY_20261005.md), and `artifacts/ftmoe_online/protocol_044/revision_002/plan.json` with its SHA256 file.

Current task is the constrained engineering recovery of 044 after user handoff: repair C constructor argument mismatch, unreachable production resume validation, and non-durable workflow checkpoints; pass actual-entrypoint synthetic and cross-runner recovery gates; then execute only the four remaining registered scientific sequences.

044 r1 remains closed incomplete: run 37184889231 reached local next_t=5200, hit the 350-minute timeout, retained zero artifacts, and started zero models/optimizer steps. Its E artifact passed covered tests. Reuse E only with verified source compatibility and new tests for modified entrypoints.

r2 explicitly permits ONE same-seed4401 from-zero reconstruction only if no complete verified r1 state exists. Persist and restore the initialized t=0 state before generating data; thereafter no reinitialization. Each generation job advances at most 200 rows and must upload/download/verify its complete checkpoint before any next segment.

Original root protocol_044/plan.json, plan.sha256, scenario_registration.json and docs/PROTOCOL044_RESULTS.md are immutable r1 scientific/historical files. The active r2 recovery plan is under revision_002; runtime must validate BOTH contracts. Never disable hashes or rewrite the old E gate to pass new code.

Scientific settings, 4 sequences, total optimizer budget 2360 and D>C criteria are unchanged. No model restart on ambiguous optimizer state, no reseeding, threshold search, forced GC, extra arms or automatic045. Historical033–044r1 are read-only. Write r2 results to docs/PROTOCOL044_REVISION002_RESULTS.md.

This publication is instructions only: no workflow creation, dispatch, generation or training. The executing model may implement and run this registered recovery after the user's handoff without asking for additional per-stage permission.
