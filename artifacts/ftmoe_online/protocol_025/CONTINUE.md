# Protocol-025 continuation handoff

Updated: 2026-09-18. **Read this first in the next session.**

## Canonical state
- Branch: `protocol-025-revision1-gpt56-20260916`.
- Protocol-025 scientific setup is frozen by `registration.json`, `method_registration.json`, and `data_revision_001.json`; **do not tune/change them from results**.
- `data_revision_001` consumed the single allowed pre-model data revision; no further scientific data revision is allowed.
- Development is **seed 700 / model seed 1 only**. Seeds `701-703` and `201-205` remain sealed.
- Pre-revision run `35101949642` is evidence-only and ineligible for model/data lock.
- No valid Protocol-025 `data_lock.json` exists and no Protocol-025 model result has been run.

## Generation status / failure cause
- Best valid revision-1 source checkpoint: run `35114741391`, artifact `protocol025-seed700-generation-revision1-progress`, artifact id `10469412410`; its committed resume manifest is `next_t=5400` with 27 registered immutable chunks ending at 5400.
- Earlier monolithic recovery run `35152128413` failed repeatedly from hosted-runner shutdown/SIGTERM; this motivated segmented recovery.
- Segmented run `35233644677` successfully completed prepare-base and all 12 transient segments through `next_t=5520`. This establishes that the engineering recovery can deterministically advance the frozen state from 5400 through 5520 without changing scientific settings.
- Its final segment failed deterministically before audit with `FileExistsError: immutable chunk already exists: .../chunks/chunk_005400_005521.npz` while attempting the registered final 5400:5521 chunk.
- Root cause: the interrupted source artifact can physically contain an orphan chunk file written after the last committed `resume_manifest.json`. The manifest still registers only 27 chunks ending at 5400, but prepare-base previously verified the registered files without rejecting/deleting extra unreferenced chunk files. The orphan `chunk_005400_005521.npz` was therefore copied into the supposedly immutable base and collided with the correct final write at t=5521.
- This is an engineering checkpoint-hygiene defect, not a scientific/data-gate result. The orphan is explicitly outside the committed checkpoint manifest and must not be treated as valid evidence.
- Fix commit `d6aac325c0cf8cf72f1f6dfb570445732b08f45f`: prepare-base now constructs the exact registered filename set from `resume_manifest.json`, verifies every registered SHA256, deletes only unregistered `chunk_*.npz` files, and asserts the remaining directory equals the registered set before uploading the segmented base. No S1-S6 semantics, registration, seed, response law, label/feature, comparator, or metric is changed.

## Segmented engineering recovery
- Workflow: `.github/workflows/protocol025-revision1-segmented-recovery.yml`.
- Engineering-only wrapper: `maintenance/run_protocol025_segmented_recovery.py`, pinned to canonical generator Git blob `3f3437e45f4d9c58182613e7ad96cf89fcda5d10`.
- The wrapper preserves registered immutable chunk semantics. Between immutable boundaries it stores rows only in `transient_partial.npz`; each segment advances at most 10 intervals.
- The canonical collector also has a resume-only bookkeeping defect: `stats.saveStats(..., migrations, ...)` references local `migrations` assigned only in non-resume initialization. The wrapper defines this bookkeeping argument as current `executed`; causal time_series/schedule_series and collected model-input arrays do not depend on that bookkeeping argument.

## Next action
1. Monitor the new segmented recovery automatically triggered by fix commit `d6aac325c0cf8cf72f1f6dfb570445732b08f45f`. Confirm prepare-base reports removal of the orphan final chunk and exact equality to the 27 manifest-registered chunks.
2. Continue through `5521`; then run `audit_ftmoe_protocol025_revision1.py` and strengthened pre-model data gate. Only a full audit pass is eligible for locking.
3. Only if the strengthened audit passes, create `data_lock.json` with the successful revision-1 run/artifact, stream SHA, revision SHA, `data_revision_id=data_revision_001`, `pre_revision_generation_run=35101949642`, and `model_results_seen_before_lock=false`.
4. The lock should run exactly `C_fixed4`, `C_fixed5`, `C_fixed8_dense`, `C_fixed8_top5`, `D_dynamic` once on seed700/model1. Primary result = equal-weight `D_dynamic - C_fixed5` AP over the 9 recurrence first-100 windows; also report legacy `D_dynamic - C_fixed4`. Run D-no-purge only if actual capacity pressure/block and purge>0.

## Non-negotiable
- Do not alter S1-S6 semantics/parameters/timeline, event probability `0.30`, `raw_next_fault` target, common 9D causal features, LR/accept/reuse thresholds, comparator definitions, or primary metric.
- Do not unseal confirmation/test seeds yet.
- Preserve negative results; no result-driven tuning.

Key code/workflows: `prepare_ftmoe_protocol025_stream.py`, `maintenance/run_protocol025_stream_collect.py`, `maintenance/run_protocol025_segmented_recovery.py`, `audit_ftmoe_protocol025_revision1.py`, `.github/workflows/protocol025-generate.yml`, `.github/workflows/protocol025-revision1-segmented-recovery.yml`, `.github/workflows/protocol025-segment-worker.yml`, `.github/workflows/protocol025-development.yml`.
