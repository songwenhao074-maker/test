# Protocol-044 revision 1 Results — blocked during registered stream generation

Date: 2026-10-04. Protocol: 044 revision 1.

## Executive status

Protocol-044 did **not** complete Stage S. Stage E completed successfully and its final gate is valid, but the single registered seed4401 simulator generation became non-resumable after the formal GitHub Actions job hit the preregistered 350-minute job timeout.

Final status:
- E_gate: **true**
- Stage S model-independent data lock: **not reached**
- C_ref / D_lin / D_no_gc / D_bounded training: **not started**
- scientific optimizer.step calls: **0**
- registered generation key: `protocol044_revision1_seed4401`
- generation was started and advanced on the same seed/state to the last observed complete boundary `next_t=5200`
- no durable artifact was uploaded from the cancelled run, so the 5200 simulator/RNG checkpoint is unavailable
- exact resume is therefore impossible with the retained artifacts
- `restart_from_zero_after_start=false` and `exact_resume_only=true`; regenerating seed4401 from t=0 would violate the registered execution contract
- result: **stageS_incomplete_exact_resume_unavailable_after_timeout**
- no Protocol-045 is authorized.

## Stage E engineering closeout

Final Stage-E workflow run: `37184789603`.

Final Stage-E artifact:
- artifact id: `11296492568`
- name: `protocol044-engineering-final-37184789603`
- digest: `sha256:90b6bab6a149c46da0aa19e551c02f76b5665a27a83e97d7de192e21e08b6591`
- execution SHA: `0fb751738718d3d06aeeaa32cfcb593436fd7c87`

The final gate reports:
- all 19 required fresh-process resume cases passed exactly
- active max = 2, resident max = 3, shadow max = 1
- source lock passed
- engineering core passed
- crash injection passed
- fail-closed production entrypoint tests passed
- baseline adapter checks passed
- E_gate = true

The bounded-memory stress test passed for lengths 8192 / 32768 / 131072. Matched online payload sizes were 15823 / 15824 / 16282 bytes, a spread of 459 bytes versus the registered 65536-byte limit.

The read-only Protocol-043 fixed-trace audit replayed 592 scoring checks for each historical arm. Both D_no_gc and D_bounded had zero scoring mismatches and zero action mismatches; no real-data model forward or gradient was used. Therefore Stage E found no registered semantic decision change requiring an old043 retrain.

## Stage S generation history

First formal Stage-S run: `37178041408`.
It failed before any simulator row was generated because the adapter registration path was unset. The interrupted artifact `11294125846` (digest `sha256:9cf412b0c5e4f77186b3fb428f851d0e263c3cb4c05afe52f978ebe2d30eb103`) proved:
- zero generated rows
- no generation checkpoint/chunks
- zero model runs
- this was a pre-initialization engineering failure.

After the registered adapter repair, formal run `37184889231` continued the same generation key rather than substituting another seed. The immutable 200-row segment sequence advanced through complete checkpoints:
`200, 400, ..., 5000, 5200`.

At `next_t=5200`, the next 200-row segment began. The job was cancelled at 2026-10-04T12:58:47Z. The workflow run started at 2026-10-04T07:08:30Z and completed/cancelled at 12:58:51Z, matching the configured 350-minute job timeout.

The workflow's interrupted-state upload was guarded by `if: failure()`. GitHub marks a timeout cancellation as `cancelled`, so the upload step was skipped. The cancelled run has no artifacts. Consequently, the latest complete local simulator/RNG/checkpoint state at t=5200 was destroyed with the runner.

## Why execution stops here

Protocol-044 preregistered:
- one seed4401 stream only
- no reroll
- exact simulator/RNG resume
- `restart_from_zero_after_start=false`
- `exact_resume_only=true`
- if exact recovery is impossible, mark incomplete and stop.

Although the generator is deterministic, replaying from t=0 to reconstruct t=5200 would still be a restart from zero after this generation key had started, and there is no retained 5200 checkpoint/chunk-hash bundle against which to prove exact identity. It is therefore not an authorized continuation.

No model-independent input audit/data lock was reached, and no C_ref, D_lin, D_no_gc, or D_bounded sequence was started. There are no Protocol-044 performance, lifecycle-on-stream, reuse-on-stream, or GC-on-stream conclusions.

## Interpretation

Protocol-044 successfully closes the registered **engineering** gaps from Protocol-043. It does **not** supply the planned new-stream capacity-pressure scientific result because the single registered stream became non-resumable due to the CI persistence design at the workflow timeout boundary.

A future protocol may preregister a resumable orchestration that persists each simulator chunk/checkpoint outside the runner before beginning the next segment, but Protocol-044 itself is closed as incomplete and is not automatically restarted.
