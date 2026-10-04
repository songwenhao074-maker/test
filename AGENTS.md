# Active Experiment Directive — Protocol-044 revision 1 registered

The user requested a new bounded plan after reviewing043: one engineering closeout, then one preregistered full-simulator capacity-pressure experiment. Goal: a coherent D exceeding C, not the best D, no strict one-in/one-out replacement.

Read NEXT_EXPERIMENT_LATEST.md, docs/PROTOCOL044_CLOSEOUT_AND_CAPACITY_20261004.md, artifacts/ftmoe_online/protocol_044/plan.json + plan.sha256, scenario_registration.json, and PROJECT_CONTEXT_LATEST.md.

Status: instructions_registered_not_implemented_not_started. This publication uploads documents only. After the user's execution handoff, implement and run the registered E→S→delivery sequence. E passing authorizes S within that handoff; do not ask again for already scoped steps. Do not launch during plan publication.

E: zero real-data gradients and zero real model forwards. Full 19-case production recovery with fresh-process continuation; real fail-closed entrypoint tests; rolling statistics; bounded online buffers, streamed audit logs and full memory accounting; read-only043 fixed-trace audit. Fixes that change policy/decisions require stopping and a new registration, not an automatic043 retrain. Preserve original043 reports.

S only after E_gate: one new simulator stream seed4401,5952 predictions/5953 raw rows, fixed U/V/W/X first-and-return timeline. Four training sequences only: C_ref → shared D_lin/B → D_no_gc → primary D_bounded. All code/policies/analysis freeze before generation; data hash seals before C training. Dynamic expert thresholds and caps unchanged. Total real optimizer.step ceiling2360, no scans/extra seed/extra stream/F/donor/old043 confirmation.

Only qualified dormant experimental experts may be reclaimed when capacity actually blocks a needed candidate. Never preset events, preload experts in science, lower thresholds or reroll to trigger GC. Untriggered natural GC remains unverified; synthetic tests cannot fill that gap. D_bounded need only beat same-stream C under registered safeguards; no requirement to beat D_no_gc or D_lin.

Historical033–043 budgets/results/checkpoints remain read-only. The new044 registration supersedes the prior blanket no044 instruction only for this explicit handoff scope. Scientific negative results do not stop the next fixed arm; engineering/input/budget failures do. Exact resume only, no restart from zero, no force push. Upload raw before analysis/results; stop after delivery or blockage, no automatic045.
