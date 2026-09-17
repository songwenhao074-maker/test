# Protocol-025 continuation handoff

Updated: 2026-09-17. **Read this first in the next session.**

## Canonical state
- Branch: `protocol-025-revision1-gpt56-20260916`.
- Protocol-025 scientific setup is frozen by `registration.json`, `method_registration.json`, and `data_revision_001.json`; **do not tune/change them from results**.
- `data_revision_001` consumed the single allowed pre-model data revision; no further scientific data revision is allowed.
- Development is **seed 700 / model seed 1 only**. Seeds `701-703` and `201-205` remain sealed.
- Pre-revision run `35101949642` is evidence-only and ineligible for model/data lock.
- No valid Protocol-025 `data_lock.json` exists and no Protocol-025 model result has been run.

## Generation status / failure cause
- Best valid revision-1 recovery artifact: run `35114741391`, artifact `protocol025-seed700-generation-revision1-progress`, artifact id `10469412410`.
- Its `resume_manifest.json` has `next_t=5400` of total `5521` (121 intervals remain).
- Recovery run `35152128413` reached attempt 10 and ended `failure`; it produced no new artifact.
- Repeated attempts failed during generation because GitHub **hosted runners received shutdown/SIGTERM** (`exit 143` / `The runner has received a shutdown signal`), not because of a Python traceback. Attempts used different runners/regions.
- Current workflow always restores the old 5400 artifact; when the runner is killed, later `upload-artifact` never executes, so partial progress is lost. **Do not keep blindly rerunning the same workflow.**

## Next action
1. Prefer a persistent environment (Codespace/self-hosted/local runner), or make an **engineering-only** resumability fix: save transient recovery state every ~8-10 intervals and restore the newest artifact matching the frozen registration SHA. This must not change final registered 200-interval chunk semantics, service laws, labels, seeds, features, or scientific parameters.
2. Resume exactly from the valid revision-1 checkpoint, finish all `5521` intervals, then run `audit_ftmoe_protocol025_revision1.py` and the pre-model data gate.
3. Only if the strengthened audit passes, create `data_lock.json` with the successful revision-1 run/artifact, stream SHA, revision SHA, `data_revision_id=data_revision_001`, `pre_revision_generation_run=35101949642`, and `model_results_seen_before_lock=false`.
4. The lock should then run exactly: `C_fixed4`, `C_fixed5`, `C_fixed8_dense`, `C_fixed8_top5`, `D_dynamic` once on seed700/model1. Primary result = equal-weight `D_dynamic - C_fixed5` AP over the 9 recurrence first-100 windows; also report legacy `D_dynamic - C_fixed4`. Run D-no-purge only if actual capacity pressure/block and purge>0.

## Non-negotiable
- Do not alter S1-S6 semantics/parameters/timeline, event probability `0.30`, `raw_next_fault` t+2 target, common 9D causal features, LR/accept/reuse thresholds, comparator definitions, or primary metric.
- Do not unseal confirmation/test seeds yet.
- Preserve negative results; no result-driven tuning.

Key code/workflows: `prepare_ftmoe_protocol025_stream.py`, `maintenance/run_protocol025_stream_collect.py`, `audit_ftmoe_protocol025_revision1.py`, `.github/workflows/protocol025-generate.yml`, `.github/workflows/protocol025-development.yml`.
