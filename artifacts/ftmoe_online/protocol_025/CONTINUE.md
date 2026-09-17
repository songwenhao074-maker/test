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
- Artifact inspection on 2026-09-17 found `resume_state.dill` is about 2.33 GB uncompressed and an interrupted `resume_state.tmp` is about 2.18 GB; the huge serialized state plus the remaining generation work explains why a monolithic hosted-runner resume is fragile.

## Segmented engineering recovery launched 2026-09-17
- Recovery run: `35210301766`, workflow `.github/workflows/protocol025-revision1-segmented-recovery.yml`.
- `prepare-base` completed successfully: it restored the registered run `35114741391`, removed the interrupted `.tmp`, verified `next_t=5400`, verified all 27 immutable chunk hashes, and split the reusable base into immutable chunks plus the necessary 5400 resume state.
- New engineering-only wrapper: `maintenance/run_protocol025_segmented_recovery.py`. It is pinned to canonical generator Git blob `3f3437e45f4d9c58182613e7ad96cf89fcda5d10` and does not modify registration/service/timeline/label/feature/model parameters.
- The wrapper preserves the final registered immutable chunk semantics. Between immutable boundaries it stores rows only in `transient_partial.npz`; each hosted-runner segment advances at most 10 intervals and uploads a new transient resume overlay. Planned boundaries are `5410,5420,...,5520,5521`.
- The canonical collector has a resume-only bookkeeping defect: `stats.saveStats(..., migrations, ...)` references local `migrations` that is assigned only in the non-resume initialization branch. The segmented wrapper defines this bookkeeping argument as the current `executed` migration list. `Stats.saveStats` uses that argument only for bookkeeping counts/metrics; causal `time_series`/`schedule_series` and the collected model-input arrays do not depend on it. This fix is treated as engineering-only resume repair, not a scientific-method change.
- If a segment is killed by hosted-runner shutdown/timeout, rerun failed jobs from the same workflow run so the last successful transient overlay is reused. Do not restart from 5400 unless the segmented artifacts themselves are invalid.

## Next action
1. Continue run `35210301766` through `5521`. On each successful segment, retain the newest transient overlay; on transient runner failure, rerun failed jobs without changing frozen scientific settings.
2. At `5521`, run `audit_ftmoe_protocol025_revision1.py` and the strengthened pre-model data gate. Only a full audit pass is eligible for locking.
3. Only if the strengthened audit passes, create `data_lock.json` with the successful revision-1 run/artifact, stream SHA, revision SHA, `data_revision_id=data_revision_001`, `pre_revision_generation_run=35101949642`, and `model_results_seen_before_lock=false`.
4. The lock should then run exactly: `C_fixed4`, `C_fixed5`, `C_fixed8_dense`, `C_fixed8_top5`, `D_dynamic` once on seed700/model1. Primary result = equal-weight `D_dynamic - C_fixed5` AP over the 9 recurrence first-100 windows; also report legacy `D_dynamic - C_fixed4`. Run D-no-purge only if actual capacity pressure/block and purge>0.

## Non-negotiable
- Do not alter S1-S6 semantics/parameters/timeline, event probability `0.30`, `raw_next_fault` t+2 target, common 9D causal features, LR/accept/reuse thresholds, comparator definitions, or primary metric.
- Do not unseal confirmation/test seeds yet.
- Preserve negative results; no result-driven tuning.

Key code/workflows: `prepare_ftmoe_protocol025_stream.py`, `maintenance/run_protocol025_stream_collect.py`, `maintenance/run_protocol025_segmented_recovery.py`, `audit_ftmoe_protocol025_revision1.py`, `.github/workflows/protocol025-generate.yml`, `.github/workflows/protocol025-revision1-segmented-recovery.yml`, `.github/workflows/protocol025-segment-worker.yml`, `.github/workflows/protocol025-development.yml`.
