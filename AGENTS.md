# Current task: Protocol-031 revision002 rare-recurrence scenario pilot

Read NEXT_EXPERIMENT_LATEST.md, docs/GITHUB_EXPERIMENT_HANDOFF.md, docs/PROTOCOL031_SINGLE_TASK_DIRECTIVE_20260922.md and artifacts/ftmoe_online/protocol_031/plan.json.

The user explicitly replaced the unrun old031 goal: construct a realistic deployment scenario where D can benefit from retained experts, then run one C/D development pilot. The current plan_revision is 2 and scenario_id is protocol031_rare_recurrence_v1. It is planned, not implemented or run.

One task: generate/freeze one new15468-step U/V/W simulator stream, perform model-free eligibility checks, run C_fixed5 and D_nonblocking_reuse once each on that same stream, publish all results and stop. Seed700/model1. C/D online updates every16 intervals; replay64; original t+2 label maturity. D birth starts600/every1600; reuse retains independent causal validation with unchanged predictive acceptance gates. Main metric is six recurrence first128 windows, not old nine first100.

Old031 revision001 restrictions (only old027 revision002 data, no new simulation, old nine windows, birth period256) are superseded by this explicit user directive. Do not alter old registrations/data/results. Do not run the old031-nonblocking-reuse workflow as the new experiment. Use protocol031-rare-recurrence and the new scenario_registration.json, data verifier and data_lock. A missing pre-generation hash must be filled and frozen before model runs, never bypassed.

The target is conditional D/C advantage, not guaranteed success, universal optimality, equal total resources or superiority over untested A/B. Disclose D's extra resident memory and shadow computation. Report failure to form useful memory; never prolong training, preload specialists or reroll seeds until favorable. No automatic A/B, extra scenario, seed, tuning or followup.

Resume interrupted generation from matching immutable chunks/RNG snapshots for the same stream only. Formal full-model replay budget is two. Test JSON finalization before long runs; recover reports from saved output without retraining. Documentation/result updates must not launch training. Synchronize all entry points on completion and stop.
