# Protocol 023 — round-1 handoff (S0–S3)

**For:** the reviewing model / next-round planner
**From:** the executing agent
**Date:** 2026-09-12
**Branch:** `protocol-023` (parent `protocol-022 @ 6c5db86fde8afd10e3997d4ff0bf8016e2e87130`)
**Upstream plan (never rewritten):** `指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md`
**Round scope executed:** S0, S1, S2, S3. **D is not implemented** (plan §37/§38).

---

## 0. The one-paragraph summary

Protocol 022 showed that a single unseen regime cannot justify dynamic experts
(fixed C's PR-AUC kept rising with budget, so no capacity plateau). Protocol 023
replaces the single regime with **three heterogeneous cascading mechanisms** —
compute-first (CPU→RAM→Disk), memory-first (RAM→Disk→CPU), io-first
(Disk→CPU→RAM) — whose marginals are matched while their temporal structure
differs, and asks the six questions of plan §37 *before* any dynamic-expert
implementation. All four streams were collected and independently verified
(exit 0, 61 checks each), and the round's answer is split: **H0 PASS** (the
compute-first arm is byte-identical to the already-audited Protocol 022 regime),
**H2 PASS** (a learner trained on one mechanism does not carry its onset
knowledge to another: gaps +0.286 and +0.073, 2 of 3 regimes), while **H1 is
blocked on a definitional question** (one of nine registered §9 checks fails
under the strict reading of "valid onset/follow-up") and **H3 — the gate that
would most directly argue for or against D — cannot be read at the registered
start state**, because the residual bank is freshly initialised with a zeroed
read-out head, so the registered §14 per-layer and router cosines have no defined
value there. D is not implemented.

---

## 1. What was built (all paths relative to the repo root)

| Piece | File | Status |
|---|---|---|
| Regime family `cascade_v3` | `simulator/workload/BitbrainWorkloadProtocol023.py` | complete, registered |
| Multi-regime audit instrument | `ftmoe_protocol023_core.py` | 66 tests pass |
| Generator tests + verifier | `test_ftmoe_protocol023_regimes.py` (28), `verify_ftmoe_protocol023_generator.py` | green |
| Stream collector + runner | `prepare_ftmoe_protocol023_stream.py`, `run_ftmoe_protocol023_s2.py` | 4 streams collected |
| Independent stream verifier | `verify_ftmoe_protocol023_stream.py` | 61 checks, exit 0 on every stream |
| Data gate / marginal matching | `analyze_ftmoe_protocol023_s2.py` (41 tests) | ran on real streams |
| Specialization probe (§13) | `probe_ftmoe_protocol023_specialization.py` | ready, needs the dev stream |
| Gradient interference (§14) | `probe_ftmoe_protocol023_gradient.py` (71 tests) | ready, see §4.2 |
| Registration | `register_ftmoe_protocol023.py` → `protocol.json`, `gate_status.json`, `source_sha256_initial.json` | written before collection |
| Stage report | `docs/FTMOE_ONLINE_PROTOCOL_023.md` | complete |
| Problem log | `artifacts/ftmoe_online/protocol_023/problem_log.jsonl` (P23-01…P23-06) | complete |

Registered timeline actually collected (replay seed 700, the registered
development seed):

```text
dev stream (2880 scored intervals)
  F0 familiar 300 | A1 compute 420 | B1 memory 420 | C1 io 420
  F1 familiar 240 | A2 recur 360  | C2 recur 360  | B2 recur 360
three single-regime streams (1200 each)
  F0 familiar 150 | <regime> only 1050      for compute/memory/io
```

Run evidence: `artifacts/ftmoe_online/protocol_023/s2_run_report.json` (per
stream: command line, real child exit code, elapsed, stream SHA256, verifier
exit code, per-regime gate verdict).

---

## 2. Measured results (S1 and S2)

### 2.1 S1 — the generator (gate H0)

| Check | Result |
|---|---|
| compute-first is **byte-identical** to Protocol 022's `cascade_v2` (CPU, RAM, disk trajectories and event envelopes) | PASS, 0 differences |
| probability 0 reproduces the frozen familiar transform for all three regimes | PASS |
| each regime's onset fires on its own resource only; every phase is exactly zero at the first age of its window (admission safety) | PASS |
| every registered lag is observable inside the 12-interval history; the two new orders fit their whole chain (ages 9 and 10) | PASS |
| tests: 28 (regimes) + 66 (instrument) + 61 (per-stream verifier, ×4 streams) | green |

### 2.2 S2 — data gate (gate H1)

Measured on the three collected single-regime streams, recomputed from the arrays
by the analyzer (the collector's declared numbers were reproduced exactly, which
validates both paths):

| Regime | prevalence | independent fault events | registered cascade events | deployment rej. | migration rej. | worst event share | valid onset w/ full follow-up (gated) | usable follow-up (P22 def., not gated) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A compute-first | 6.80 % | 410 | 342 | 13.20 % | 36.29 % | 1.57 % | **24** | 185 |
| B memory-first | 3.05 % | 260 | 297 | 8.45 % | 15.43 % | 1.56 % | **36** | 200 |
| C io-first | 3.79 % | 271 | 297 | 9.32 % | 22.17 % | 1.41 % | **31** | 235 |

(`independent fault events` is the registered plan §9 quantity — per-host
contiguous positive-label runs inside the regime's phase window. The registered
cascade-envelope count is listed separately because the two differ: a task that
cascades can produce more than one host-level run and a run can be shared.)

Median observed cohort life is 5–6 intervals against 16–18 interval registered
response horizons, which is what makes the strict follow-up reading unreachable
(§4.1).

- prevalence spread **3.75 pp ≤ 4 pp** → registered marginal-matching check
  **passes, tightly** (this is the plan §9 number that decides whether the three
  scenarios are comparable at all);
- every other registered §9 check passes: prevalence 3–12 %, independent events
  ≥ 80, deployment rejection ≤ 25 %, migration rejection ≤ 40 %, worst-event
  share < 10 %;
- **onset-threshold admissibility is now measured, not assumed** (first real
  confirmation for the two provisional values): familiar per-task maxima
  cpu 1860 / ram 1400 / disk 9000, against thresholds A 2600, B 2810, C 11600 and
  floors 4400 / 4500 / 16000 — all three strictly admissible;
- **one registered check fails**: "valid onset/follow-up ≥ 50 per regime" is
  24 / 36 / 31 when "valid" means the *whole* registered response window is
  observed. Cause is structural, not a defect: median observed cohort life is
  5–6 intervals while the registered response horizons are 16–18 intervals. Under
  Protocol 022's own `MIN_FOLLOWUP_EVENTS` definition (at least one observed
  interval inside the window) the same streams give 185 / 200 / 235.

**H1 status: `FAIL` under the literal reading, `PASS` under the P22-compatible
reading — decision required (§4.1).** The analyzer did not switch definitions
after seeing the number; it reports both and exits non-zero.

### 2.4 S3 — specialization (gate H2) and gradient interference (gate H3)

**§13 cross-regime probe** (`probe_ftmoe_protocol023_specialization.py`,
`specialization/cross_regime_probe.json`): train on one mechanism, evaluate on all
three, target = that mechanism's own onset resource.

| trained on | within-regime AP | mean cross-regime AP | gap | passes |
|---|---:|---:|---:|---|
| compute-first | 0.3383 | 0.0520 | **+0.2863** | yes |
| memory-first | 0.0807 | 0.0081 | **+0.0726** | yes |
| io-first | 0.0136 | 0.1356 | −0.1220 | no |

**H2 = PASS (2/3).** Caveat in the same breath: the io-first row has only **5
disk onsets** in its own test window (prevalence 0.09 %), so its within-regime AP
is not estimable from this stream — the honest reading is "two mechanisms show a
specialization opportunity; the third is positive-starved", not "C is not
learnable".

**§14 gradient probe** (`probe_ftmoe_protocol023_gradient.py`,
`specialization/gradient_interference.json`): 64 events per regime, 20 parameter
groups. At the registered start state all three pairs are strongly
**co-directional** — mean cosines **+0.860** (A|B), **+0.766** (A|C), **+0.929**
(B|C); negative-pair fractions 6.3 % / 3.7 % / 4.9 % — which under the registered
rule would read **STOP-GI (no conflict)**. The gate is nevertheless written
**INADMISSIBLE**, because at that state the bank is untrained: 8 of 20 parameter
groups have exactly zero gradient (zeroed read-out head), so the registered
per-layer and router cosines do not exist. See §4.2.

**H4 sequential forgetting** is read from the same probe's cells: the cross-regime
drop is −0.282 (A→B), −0.291 (A→C), −0.070 (B→A), −0.075 (B→C) — measured at
probe level, not closed (that needs the S4 prequential fixed-C run).

### 2.4 The cross-regime confound is real and is handled

A compute-first cascade also crosses the memory-first RAM threshold and the
io-first disk threshold. On a compute-first-only stream an unmapped
resource+threshold scan reports 32 A / 17 B / 9 C onsets, while the
regime-restricted task cohort reports 32 A and **0** B and C. Every gate number
above uses the restricted cohort, and the confound is reported next to it. Had
this not been handled, the "≥ 80 independent events per regime" requirement would
have been inflated by roughly 50 % (P23-06).

---

## 3. What the round can and cannot claim

**Can claim:** the three mechanisms are implemented, registered, mutually
distinct in temporal order, marginally matched on the plan's registered numbers,
audit-correct at task level, and their onset thresholds are admissible on
measured familiar data; the compute-first arm reproduces the already-audited
Protocol 022 regime byte-for-byte, so any A-vs-B-vs-C difference is attributable
to resource order rather than to a re-tuned envelope; and a probe trained on one
mechanism does **not** carry its onset knowledge to the others while working on
its own (H2, 2 of 3 regimes).

**Cannot claim (yet):** anything about D; that the three mechanisms create
*gradient* conflict (H3 is inadmissible at the untrained start state — and the
only admissible block that exists there points the other way); that the io-first
mechanism is learnable or unlearnable (5 test positives); anything about
confirmation seeds (701–703 untouched, one development seed only); that the
regimes mirror real industrial faults (the phase law is a simulator
construct-validity artefact); that the marginal match is comfortable (3.75 pp
against a 4 pp ceiling is tight, and the instrument-local marginal contrasts on
event count, duration and peak ratio still differ — see §4.3).

---

## 4. Decisions the reviewer must make

### 4.1 The follow-up definition (blocks H1's final status)

Two readings of plan §9's "valid onset/follow-up ≥ 50 / regime":

- **literal / strict**: only onsets whose entire registered response window is
  observed count → 24 / 36 / 31 → **FAIL**;
- **P22-compatible**: an onset counts as usable when at least one interval inside
  the registered window is observed → 185 / 200 / 235 → **PASS**.

The strict reading cannot be satisfied by this simulator: 16–18 interval
response horizons exceed the median task lifetime. The options are (a) accept the
strict reading and record H1 FAIL, which stops the round before S3's method
comparison; (b) amend the registered definition to the P22 one, recorded as an
amendment with the old value kept (this is the precedent already used for the
2.5 GiB RAM guard in P22-17/19); (c) shorten the registered response horizons,
which changes the regimes and invalidates the S1 byte-identity result. My reading
is that (b) is the only option that neither rewrites a result nor changes the
scenario, but the choice is the reviewer's — the analyzer supports either with a
one-line change and does not choose silently.

### 4.2 H3 (gradient interference) cannot be evaluated at the registered start state

Measured on the real model: the frozen checkpoint contains no learner weights,
and Protocol 020 R1 constructs a **fresh** residual bank whose read-out layer is
zeroed (`ftmoe_online_r1.py:80-84`). At that state `correction_logits.abs().max()
== 0.0`, only 10 of the 26 trainable tensors receive a non-zero gradient, and all
16 hidden-layer tensors have exactly zero gradient — so the plan §14 sub-reports
"router gradient cosine" and "per-layer cosine" have **no defined value**, and the
read-out block alone describes an untrained bank.

The instrument (a) reports zero-norm groups as `null`, counts them under
`n_pairs_undefined`, and excludes them from the verdict rather than reading them
as "no conflict"; (b) accepts `--learner-state <checkpoint with learner.*>` so the
registered gate can be evaluated on a **mature** fixed-C learner; and (c) now
refuses to publish a scientific verdict from a degenerate state: the verdict is
written `INADMISSIBLE` with the registered outcome preserved under
`registered_gate_at_this_state`.

**Decision needed:** should H3 be (a) evaluated only on a mature fixed-C learner
— which requires a small extra run to produce one, i.e. a fixed-C online pass on
the dev stream (this is not D, it is the C arm, and it is the cheapest path to a
real H3), or (b) recorded as not-yet-evaluable and carried into the round that
implements D? I recommend (a): without a mature state, §14 stays unevaluated and
the plan's own §38 entry conditions for D cannot be satisfied either way.

### 4.3 Instrument-local contrast tolerances are not met

`regime_contrast_report` (instrument-local tolerances, deliberately not a gate)
fails all three regime pairs on event-count relative difference (0.132 / 0.132 /
0.000), mean-duration relative difference (0.323 / 0.130 / 0.411) and median
peak-ratio relative difference (0.514 / 0.315 / 0.290) — against a 10 % relative
tolerance. The eight registered §9 numbers pass (prevalence spread 3.75 pp
against a 4 pp ceiling); these stricter instrument-local contrasts do not.
Should they become part of the gate (which would require tighter event-duration
and peak-ratio matching, i.e. regenerating the regimes), stay reported as
context, or be re-registered with their own tolerances before the confirmation
round? Note that the peak-ratio difference is partly a *reporting* quantity: the
registered RAM/disk amplitudes are matched by construction (both regimes ramp to
the same 4500–6000 band), while the *measured* per-task ratio differs because the
familiar denominator differs per task.

### 4.4 Seeds and next stage

Seed 700 is a development seed. Confirmation seeds 701–703 are untouched. The
plan's S4 (strict prequential A/B/C) and the D round are not started; §38 lists
the eight conditions for D, of which this round can address at most the first
three.

---

## 5. Reproduce / verify in one command each

```powershell
# generator registration (H0)
D:\Anaconda\envs\dynmoe\python.exe verify_ftmoe_protocol023_generator.py

# instrument + generator test suites
D:\Anaconda\envs\dynmoe\python.exe -m unittest test_ftmoe_protocol023_core test_ftmoe_protocol023_regimes test_ftmoe_protocol023_s2 test_ftmoe_protocol023_gradient

# independent verification of any collected stream (61 checks, non-zero on failure)
D:\Anaconda\envs\dynmoe\python.exe verify_ftmoe_protocol023_stream.py artifacts\ftmoe_online\protocol_023\development_streams\dev_seed700_steps2880

# data gate / marginal matching (rewrites data_audit/*.json + DATA_GATE.md, prints a JSON verdict)
D:\Anaconda\envs\dynmoe\python.exe analyze_ftmoe_protocol023_s2.py

# S3 specialization probe (§13) and gradient probe (§14)
D:\Anaconda\envs\dynmoe\python.exe probe_ftmoe_protocol023_specialization.py
D:\Anaconda\envs\dynmoe\python.exe probe_ftmoe_protocol023_gradient.py
```

Machine-readable gate state: `artifacts/ftmoe_online/protocol_023/gate_status.json`.
Every gate is written `pending` until the analyzer that measured it writes a
verdict — H0 is PASS, H1 is decided by §4.1, H2 by the specialization probe, H3
by §4.2, H4 by the same probe's forgetting table.
