# Active Experiment Directive — Protocol-040

Current registered task: [Protocol-040 minimal two-expert dynamic pool](docs/PROTOCOL040_TWO_EXPERT_POOL_20261003.md).
Machine plan: artifacts/ftmoe_online/protocol_040/plan.json; verify plan.sha256 and the instructions hash before implementation.

Status: instructions registered; implementation and science NOT started. This publication is documentation only. After the human explicitly hands off execution of Protocol-040, implement the frozen plan, run exactly one D_pool2 scientific sequence, analyze, publish results to GitHub/main, and stop. That execution handoff authorizes the registered bounded implementation/run/delivery without another routine permission request.

Goal: preserve D>C while testing two accepted experts, causal selection and repeated sleep/reactivation. D may evolve structurally; the original topology is not mandatory. Keep B=C+D_lin; at most two dynamic expert slots and one deployed expert. Reuse validation must not be blocked by shadow training. Exactly one second-candidate attempt, <=352 live +16 shadow optimizer steps. C/B/D_keep/D_039 are cached controls. Cumulative loss allowance remains against D_keep, not reset against D_039.

Read docs/GITHUB_EXPERIMENT_HANDOFF.md for source commits/artifacts: main does not contain all 039 implementation files. Do not infer algorithm code from an old local checkout.

Historical033–039 budgets and results remain closed and immutable. No new stream/seed, F research, accepted-expert permanent deletion, hyperparameter sweep, extra run, or automatic next protocol is authorized. Preserve negative and unexercised outcomes. Finalize science/analysis/publication statuses separately; publication failures never authorize retraining.
