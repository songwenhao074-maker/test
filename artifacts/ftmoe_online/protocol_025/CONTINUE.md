# Protocol-025 continuation handoff

Updated: 2026-09-18. **Read this first in the next session.**

## Canonical state
- Branch: `protocol-025-revision1-gpt56-20260916`.
- Protocol-025 scientific setup is frozen by `registration.json`, `method_registration.json`, and `data_revision_001.json`; **do not tune/change them from results**.
- `data_revision_001` consumed the single allowed pre-model data revision; no further scientific data revision is allowed.
- Development is **seed 700 / model seed 1 only**. Seeds `701-703` and `201-205` remain sealed.
- Pre-revision run `35101949642` is evidence-only and ineligible for model/data lock.
- No valid Protocol-025 `data_lock.json` exists and no Protocol-025 model result has been run.

## Generation status / deterministic audit blocker
- Best valid revision-1 source checkpoint: run `35114741391`, artifact `protocol025-seed700-generation-revision1-progress`, artifact id `10469412410`; its committed resume manifest is `next_t=5400` with 27 registered immutable chunks ending at 5400.
- Earlier monolithic recovery run `35152128413` failed repeatedly from hosted-runner shutdown/SIGTERM; this motivated segmented recovery.
- Segmented run `35233644677` advanced the frozen state through `next_t=5520`; its final segment exposed an orphan unregistered `chunk_005400_005521.npz` checkpoint-hygiene defect. Fix commit `d6aac325c0cf8cf72f1f6dfb570445732b08f45f` removed only manifest-unregistered chunk files from the recovery base; no scientific setting changed.
- Recovery run `35254809015` then completed the full frozen generation successfully to `next_t=5521`. Final stream SHA256 is `46b1dbdd885683bd45c146ffc3cbe68dd151bf60a10c1663eda45b68f12d7c42`; final registered chunk `5400:5521` SHA256 is `fc3e9887961e6e99f29d0f386e4da9dd318367010f9028a6233599829eef4c10`. The generator's own `data_audit.json` reported `audit_pass=true`.
- The **strengthened revision-1 audit failed deterministically**, so the stream is NOT eligible for data lock or model runs. All strengthened gates passed except `F0_guard_has_positive_and_negative`.
- Exact evidence from run `35254809015`, final job `105357470988`: strengthened audit reports `F0_guard.normal_rows=960`, `F0_guard.positive_rows=0`, target=`raw_next_fault`, with indices `0,5,...,295`. Thus the registered F0 baseline guard contains no positive next-fault examples under the frozen revision-1 stream.
- Other strengthened gates passed: revision provenance, pre-revision ineligibility, 5521x16 shape, physical-capacity label recomputation, finite common 9D causal features, events for every service, event revision provenance, S4 floor/half-life semantics, S5 periodic release semantics/minimum trace, positive+negative examples in every service phase, and all nine recurrence first100 AP windows being defined.
- This is now a **scientific/data acceptance blocker under the preregistered strengthened gate**, not a transient runner failure. Per the frozen protocol, do not create `data_lock.json`, do not run any comparator, do not change S1-S6/data revision/seed/event probability/labels/features to manufacture F0 positives, and do not unseal seeds 701-703 or 201-205.

## Segmented engineering recovery
- Workflow: `.github/workflows/protocol025-revision1-segmented-recovery.yml`.
- Engineering-only wrapper: `maintenance/run_protocol025_segmented_recovery.py`, pinned to canonical generator Git blob `3f3437e45f4d9c58182613e7ad96cf89fcda5d10`.
- The wrapper preserves registered immutable chunk semantics. Between immutable boundaries it stores rows only in `transient_partial.npz`; each segment advances at most 10 intervals.
- The canonical collector also has a resume-only bookkeeping defect: `stats.saveStats(..., migrations, ...)` references local `migrations` assigned only in non-resume initialization. The wrapper defines this bookkeeping argument as current `executed`; causal time_series/schedule_series and collected model-input arrays do not depend on that bookkeeping argument.

## Next action
1. Treat run `35254809015` as completed generation evidence but **audit-ineligible for model locking** because F0 guard has 0 positive rows.
2. Do not rerun the same frozen seed700 stream expecting a different outcome: generation is deterministic and would reproduce the same F0 guard failure.
3. Do not make another scientific data revision: `data_revision_001` already consumed the single allowed pre-model revision. Any future continuation requires an explicit protocol-level decision outside the frozen Protocol-025 rules (for example, accepting Protocol-025 as a preregistered negative/data-gate outcome and designing a separately preregistered successor protocol). Preserve this result rather than tuning around it.
4. No five-comparator development run is permitted from this stream; therefore no D-no-purge diagnostic is applicable yet.

## Non-negotiable
- Do not alter S1-S6 semantics/parameters/timeline, event probability `0.30`, `raw_next_fault` target, common 9D causal features, LR/accept/reuse thresholds, comparator definitions, or primary metric.
- Do not unseal confirmation/test seeds yet.
- Preserve negative results; no result-driven tuning.

Key code/workflows: `prepare_ftmoe_protocol025_stream.py`, `maintenance/run_protocol025_stream_collect.py`, `maintenance/run_protocol025_segmented_recovery.py`, `audit_ftmoe_protocol025_revision1.py`, `.github/workflows/protocol025-generate.yml`, `.github/workflows/protocol025-revision1-segmented-recovery.yml`, `.github/workflows/protocol025-segment-worker.yml`, `.github/workflows/protocol025-development.yml`.
