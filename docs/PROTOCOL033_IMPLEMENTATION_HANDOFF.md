# Protocol-033 Implementation Handoff

Date: 2026-09-28. Status: `implementation_ready_not_run`.

## Execution identity

- Execution branch: `protocol-033-5969-gpt56-20260928`
- Frozen implementation source: `a215c1161adba4ff29514eb88806762c2f068023` on `protocol-031-rare-recurrence-20260922`
- Launch workflow: `.github/workflows/protocol033-5969.yml`
- Workflow trigger: `workflow_dispatch` only
- Registered stream: 5968 scored intervals + 1 terminal raw-label support row = 5969 rows
- Budget: one new continuous seed700 simulator stream; one `C_fixed5` replay; one `D_guard_budget` replay; no other scientific follow-up

## Implemented files

- `prepare_ftmoe_protocol033_stream.py`: Protocol-033 identity, timeline, source hashes, exact resume checkpoint contract, memory telemetry, generation-only sealing without full-stream reassembly.
- `maintenance/run_protocol033_segment.py`: reuses the frozen Protocol-031 simulator loop while replacing only Protocol-033-facing registration/checkpoint/finalize hooks; each process advances at most200 rows.
- `assemble_ftmoe_protocol033.py`: fresh-process immutable-chunk verification, field memmap assembly, compatibility `stream.npz`, causal common features, assembly memory profile.
- `audit_ftmoe_protocol033.py`: independent pre-model audit using target rows `raw[start+1:end+1]`; recomputes overload ratio and labels directly from `post_totals/capacities`; freezes input hashes only on pass.
- `ftmoe_protocol033_guard_budget.py`: Protocol-032 guard-budget decisions with the preregistered600+1000k matured birth clock and event-specific reuse influence logging.
- `run_ftmoe_protocol033.py`: same frozen input and shared-first4 initialization for C/D; separate process per arm; memory telemetry; all registered lifecycle evidence.
- `compare_ftmoe_protocol033.py`: all six recurrence128 windows, first32/first64, all W blocks, full stream, fixed development references, mechanism/cost evidence.
- `test_ftmoe_protocol033.py`: static contract tests for geometry, six windows, birth clock, physical mapping, dispatch-only workflow and one-stream/two-replay budget.

## Safety and reproducibility gates

Before generation starts, the workflow compiles every Protocol-033 module, runs the Protocol-033 tests and plan validator, and verifies by `git diff` that the frozen Protocol-031/032 source files and physical workload implementation are unchanged from `a215c116...`.

Generation uses immutable chunks of at most200 rows. A checkpoint contains the simulator/workload/scheduler/recovery/stats/capacity object graph, active service, applied switches, Python/NumPy/Torch RNG state, next row, registration hash, generator/source hashes and chunk manifest. The same registered stream may resume deterministically; mismatched registration/source hashes are rejected.

The generation process never assembles a second complete stream. When all30 chunks exist (29x200 +169), generation exits. Assembly and audit run in fresh processes. Memory is sampled every20 generated/replayed intervals plus checkpoint boundaries; the registered soft limit is80% of the effective host/cgroup memory limit.

No model starts until the independent input audit passes every required gate. On failure, preserve the stream/evidence and stop: no seed reroll, event-intensity adjustment, missing-window deletion or automatic model replay.

## Automatic continuation after first dispatch

The first `workflow_dispatch` starts stage `generate` with `execution_ref=protocol-033-5969-gpt56-20260928`. Each generation Action run launches up to three separate <=200-row generator processes, uploads the exact resumable state, then dispatches the next generation run if incomplete. Once row5969 is sealed, it dispatches the `assemble_model` stage. That stage assembles, audits, then runs exactly C and D sequentially and writes terminal evidence.

Thus no manual continuation is required after the initial dispatch unless an explicit blocker stops the registered run.

## Current limitation before first dispatch

The ChatGPT local container used during implementation could not resolve `github.com`, so an independent local `git clone`/`py_compile` could not be executed there. This is an environment/network limitation, not experimental evidence. To keep the scientific budget safe, the GitHub workflow performs compilation/tests/plan validation and historical-source isolation checks before the first generated Protocol-033 byte. If that preflight fails, generation does not start and the failure should be treated as an engineering blocker, not as a scientific run.

## Required interpretation

Protocol-033 is a development scenario designed after observing Protocol-031/032. It is not independent confirmation. Report numerical D-vs-C behavior, fixed development-reference status, mechanism evidence and cost separately. Do not infer global superiority, hard-delete benefits, A/B effects, or multi-seed significance from this single stream.
