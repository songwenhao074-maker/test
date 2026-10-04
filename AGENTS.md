# Active Experiment Directive — Protocol-044 revision 1 blocked and closed

Protocol-044 revision 1 Stage E completed successfully. Final E artifact 11296492568, execution SHA 0fb751738718d3d06aeeaa32cfcb593436fd7c87, E_gate=true. All 19 required fresh-process resume cases, crash/fail-closed gates, bounded-memory tests, baseline adapter checks and the read-only 043 fixed-trace audit passed.

Stage S did not reach data lock or model training. The single registered seed4401 generation run 37184889231 advanced to the last observed complete boundary next_t=5200, then the GitHub Actions job hit the configured 350-minute timeout. Because cancellation skipped the failure-only interrupted-state upload, the 5200 simulator/RNG checkpoint was not persisted and the cancelled run has no artifacts.

The registration requires exact resume and forbids restart from zero after start. Therefore Protocol-044 is **stageS_incomplete_exact_resume_unavailable_after_timeout**. C_ref, D_lin, D_no_gc and D_bounded were not started; scientific optimizer steps=0.

Read docs/PROTOCOL044_RESULTS.md and artifacts/ftmoe_online/protocol_044/runs/run_37184889231/blockage_status.json. Historical033–043 remain read-only. Do not rerun seed4401 from zero, choose another seed, change thresholds, or automatically start Protocol-045. A new preregistered instruction and user handoff are required.
