# Active Experiment Directive

Protocol-036 completed in run36831958978: operating_point_risk. Historical033—036 budgets and results remain closed and immutable.

Next is Protocol-037, the first incremental dynamic-expert experiment. Read NEXT_EXPERIMENT_LATEST.md, docs/PROTOCOL036_INDEPENDENT_REVIEW_20261001.md, docs/PROTOCOL037_SINGLE_DYNAMIC_BIRTH_20261001.md and artifacts/ftmoe_online/protocol_037/plan.json. Run maintenance/validate_protocol037_plan.py before implementation/execution; it only validates registration files.

This publication is instructions only, no automatic launch. When the user hands off execution, run exactly two new sequences on cached036 seed3601: F_extra and D_birth. Keep B=C+D_lin intact. At most one causal expert birth, zero new streams, no trigger sweeps, no sleep/wake/delete trials. Implement and freeze both arms and analysis before inspecting037 real-stream trigger or model results. Stop and report after this step; later lifecycle roadmap needs separate registration and user handoff. Never rerun old budgets to fix publication errors.
