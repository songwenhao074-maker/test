# Protocol 023 — multi-regime recurring drift (dynamic experts vs fixed online fine-tuning)

**Status written:** 2026-09-12 · **Branch:** `protocol-023` (created from `protocol-022 @ 6c5db86fde8afd10e3997d4ff0bf8016e2e87130`)
**Plan (upstream, never rewritten):** [`指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md`](../指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md), SHA256 `097889694e1fdada2ea51016f03b12c6efb69fe612cab55149208fca092e91c3`
**Round scope:** S0–S3 only. **D is not implemented** and must not be started before H2 and H3 are evaluated (plan §37, §38).

> This file reports only what an artifact can prove. Every stage below carries the
> command or file that produced it. Where a stage has not run, the section says so.

---

## 1. Why this protocol exists

Protocol 022 measured the thing that killed the dynamic-expert claim for a single
unseen regime (P22-22):

| updates | late-unseen PR-AUC (fixed C) |
|---:|---:|
| 100 | 0.4256 |
| 200 | 0.4374 |
| 400 | 0.5200 |
| 800 | 0.6568 |
| 1600 | 0.6891 |

The curve was **still climbing** at the largest budget (+0.0323 over the last
doubling) while the pre-onset warning metric did not improve (onset AP 0.0917 for
frozen A versus 0.0898–0.0908 for every residual variant). A fixed expert set
therefore had no demonstrated capacity plateau on that regime, and without a
plateau a dynamic-expert mechanism has no scientific necessity: the residual
error was a training-budget artefact, not a capacity deficit.

Protocol 023 changes the *environment*, not the model: three heterogeneous
cascading mechanisms, short dwell times, switching, and recurrence — the setting
in which a stability–plasticity conflict can actually exist. The plan's round-1
question is deliberately answerable without D:

1. Are the three regimes really different?
2. Does cross-regime generalization degrade?
3. Is there gradient conflict?
4. Does learning a new regime cost the old one?
5. Does recurrence require re-learning?
6. Under a frozen real-time update budget, does fixed C show the conflict?

Only if those hold does D get implemented, in a second round.

---

## 2. S0 — freeze Protocol 022 and register this round

| Item | Value | Evidence |
|---|---|---|
| Branch | `protocol-023` | `git branch` |
| Parent | `protocol-022 @ 6c5db86fde8afd10e3997d4ff0bf8016e2e87130` | `artifacts/ftmoe_online/protocol_023/protocol.json` |
| Bootstrap head | recorded at registration | `protocol.json` → `bootstrap_head` |
| Plan hash | `097889694e…091c3` | `protocol.json` → `plan_sha256` |
| Sources pinned before collection | 18 files, SHA256 each | `source_sha256_initial.json` |
| Frozen online start checkpoint | `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt`, `10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b` | `protocol.json` → `frozen_checkpoint` |
| Round-1 scope | S0–S3 allowed; D, backbone changes and any retraining of v4/P19/P20/P22 forbidden | `protocol.json` → `round1_scope` |
| Gate status | all five gates start `pending` | `gate_status.json` |

Registration is performed by `register_ftmoe_protocol023.py`, which re-reads
`HEAD`, hashes the plan and the sources, and refuses to invent a verdict: the
gate table is written `pending` and is only changed by the analyzer that
measured it.

**History is untouched.** `cascade_v2`, the Protocol-019/020/021/022 artifacts and
every frozen checkpoint keep their bytes; the new work lives in its own tree and
its own generator file.

---

## 3. S1 — the registered regime family `cascade_v3`

Generator: `simulator/workload/BitbrainWorkloadProtocol023.py`
(`Protocol023MultiRegimeBWGD2`).

| Regime | Order (lead → lag → follow) | Onset resource | Onset threshold | Physical story (plan §5–§7) |
|---|---|---|---|---|
| A `compute_first` | CPU →4 RAM →8 Disk | CPU | 2600 | compute-intensive inference burst |
| B `memory_first` | RAM →3 Disk →6 CPU | RAM | 2810 *(provisional)* | working-set growth / leak → paging → paging CPU cost |
| C `io_first` | Disk →3 CPU →6 RAM | Disk | 11600 *(provisional)* | logging / checkpoint accumulation → compaction → cache disturbance |

What the three regimes share, and what they may not share:

- **Shared (matched by construction):** the familiar admission law, the CPU burst
  band 4400–5200, the RAM response band 4500–6000, the disk retention peak 16000
  (cap 24000), the duration distributions of the new orders, the phase shape law,
  and the onset probability.
- **Allowed to differ:** resource order, lag, and which resource the cascade
  starts in. Plan §9 requires the *marginals* to match while the *joint
  structure* changes; §2 of the plan is explicit that a scenario which only
  raises amplitudes would not be evidence.

### 3.1 The inherited regime is exact, not approximate

`compute_first` is Protocol 022's `cascade_v2`: same registered mechanism seed
(22022), same ramp multiplier (2.0), same durations, same shape law. Check A-01
of `verify_ftmoe_protocol023_generator.py` asserts **zero** trajectory
differences on CPU, RAM and disk for every generated task, plus element-wise
equality of the event envelopes. That check is the reason the A-arm of this round
is comparable with Protocol 022 at all; getting there exposed four independent
divergences (problem P23-01).

### 3.2 Admission safety is structural

The registered phase law is exactly **zero at the first age of every window** for
every resource, so no phase can make a task unplaceable at admission. This is the
P21-01 constraint (`Simulator.getPlacementPossible()` evaluates demand at the
admission interval, which produced 93.7 % deployment rejection for a flat burst
floor) generalised from the CPU phase to all three resources and all three
regimes. Check A-03 measures it per regime rather than assuming it.

### 3.3 Observability

Every registered lag must be visible inside the 12-interval model history or the
regime is unseen *and unlearnable*. The two new orders use 4–5 interval phases so
their longest chain (6 + 5 = 11) fits inside it. `compute_first` keeps Protocol
022's longer windows (longest chain age 17) — that regime's onset is detected, not
its whole chain reconstructed — and the exemption is registered explicitly in
`INHERITED_CHAIN_REGIMES` instead of being a silent exception.

### 3.4 Verification of the generator

| Check | What it proves | Result |
|---|---|---|
| `test_ftmoe_protocol023_regimes` (28 tests) | registered physics, per-regime onsets, envelope purity, registration guards | 28/28 PASS |
| `test_ftmoe_protocol023_core` (66 tests) | the multi-regime audit instrument | 66/66 PASS |
| `verify_ftmoe_protocol023_generator.py` A-01 | compute-first is byte-identical to `cascade_v2` | PASS (0 differences) |
| A-02 | probability 0 is the frozen familiar transform for all three regimes | PASS |
| A-03 | each regime's onset fires on its own resource only, and is admission-safe | PASS |
| A-04 | every registered lag is observable; new orders fit the history | PASS |

Command: `D:\Anaconda\envs\dynmoe\python.exe verify_ftmoe_protocol023_generator.py`
(the script prints one JSON object per check and a final `status`).

---

## 4. S2 — data-only streams and the data gate

Collector: `prepare_ftmoe_protocol023_stream.py` · runner: `run_ftmoe_protocol023_s2.py` ·
independent verifier: `verify_ftmoe_protocol023_stream.py` · gate: `analyze_ftmoe_protocol023_s2.py`.

### 4.1 Registered timeline (development, replay seed 700)

```text
F0 familiar      300
A1 compute-first 420        B1 memory-first 420        C1 io-first 420
F1 familiar      240
A2 recurrence    360        C2 recurrence   360        B2 recurrence 360
----------------------
scored          2880 intervals (+1 guard row)
```

The recurrence half deliberately returns the regimes in a different order
(A → C → B) so that "recurrence" is a repetition of a *mechanism*, not of a
schedule position.

Three single-regime streams (`single_<regime>_seed700_steps1200`: 150 familiar +
1050 of one regime) are collected alongside it, because the marginal-matching
gate in plan §9 asks for per-regime cohorts that are not confounded by the other
two mechanisms.

### 4.2 What the collector guarantees

- one process at a time, 3 torch threads, 1 interop thread, BelowNormal priority,
  a 2.5 GiB RAM guard with bounded waits (every wait recorded), a 20 GiB disk
  guard, and `stats.history_limit = 64` / `stats.series_tail = 96` (the P22-18/20
  memory fixes, without which this stream length does not fit on this machine);
- the true child exit code, never a shell wrapper's verdict (the P20 defect that
  Protocol 022 banned);
- `failure.json` with the traceback if anything raises, and refusal to overwrite
  an existing stream directory;
- phase/regime/mechanism/event columns written **audit-only**, with the forbidden
  input tokens registered in the manifest.

### 4.3 Independent verification

`verify_ftmoe_protocol023_stream.py <stream_dir>` recomputes 61 checks from the
saved arrays and JSON — never trusting the manifest's own numbers — and exits
non-zero on any failure. It measures, rather than assumes:

- the row layout (`steps + 1`) and the registered phase geometry;
- that each phase's cascade tasks carry that phase's regime and no other;
- task-timeline integrity, slot reuse and migration continuity;
- per-regime onset counts **with** and **without** the task cohort, so the
  cross-regime confound is reported instead of hidden (P23-06);
- the familiar per-task maximum of each onset resource, next to the registered
  threshold and floor, so an inadmissible threshold is reported rather than
  silently retuned (P23-05);
- finiteness of every model-facing array (a single NaN fails the check — the
  P22-07 failure mode).

### 4.4 Results

Four streams were collected on 2026-09-12 (replay seed 700), all with a real
child exit code of 0 and a verifier exit code of 0
(`artifacts/ftmoe_online/protocol_023/s2_run_report.json`):

| Stream | scored intervals | elapsed | child exit | verifier | per-regime gate |
|---|---:|---:|---:|---:|---|
| `single_compute_first_seed700_steps1200` | 1200 | 439.6 s | 0 | 0 | A true |
| `single_memory_first_seed700_steps1200` | 1200 | 444.4 s | 0 | 0 | B true |
| `single_io_first_seed700_steps1200` | 1200 | 447.8 s | 0 | 0 | C true |
| `dev_seed700_steps2880` | 2880 | 3310.3 s | 0 | 0 | A, B, C true |

Gate numbers (`analyze_ftmoe_protocol023_s2.py`, recomputed from the arrays —
the collector's own audit numbers reproduced exactly on every stream):

| Regime | prevalence | independent fault events | registered cascade events | deployment rej. | migration rej. | worst event share | valid onset w/ full follow-up | usable follow-up (P22 def.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A compute-first | 6.80 % | 410 | 342 | 13.20 % | 36.29 % | 1.57 % | **24** | 185 |
| B memory-first | 3.05 % | 260 | 297 | 8.45 % | 15.43 % | 1.56 % | **36** | 200 |
| C io-first | 3.79 % | 271 | 297 | 9.32 % | 22.17 % | 1.41 % | **31** | 235 |

- prevalence spread **3.75 pp ≤ 4 pp** → the registered marginal-matching check
  passes, tightly;
- every other registered §9 check passes;
- **onset-threshold admissibility is measured, not assumed**: familiar per-task
  maxima cpu 1860 / ram 1400 / disk 9000 against thresholds A 2600 (floor 4400),
  B 2810 (4500), C 11600 (16000) — all three strictly admissible. This is the
  first measured confirmation of the two provisional RAM/disk thresholds;
- **one registered check fails**: "valid onset/follow-up ≥ 50 / regime" is
  24 / 36 / 31 under the strict reading. Cause is structural — median observed
  cohort life is 5–6 intervals against 16–18 interval registered response
  horizons — and the P22-compatible reading gives 185 / 200 / 235.

**H1 = FAIL on the strict reading** (STOP-DATA recorded), with the definitional
question raised for the reviewer rather than resolved silently. See
`data_audit/DATA_GATE.md` (verdict + both counts + warnings) and problem P23-07.

---

## 5. S3 — specialization, gradient conflict, forgetting

### 5.1 Cross-regime specialization probe (plan §13)

Instrument: `probe_ftmoe_protocol023_specialization.py`, built on Protocol 022's
closed-form logistic probe so the numbers stay comparable with P22-S4.

Registered gate: within-regime AP − mean cross-regime AP ≥ **0.05** for at least
**2 of 3** regimes, at the primary horizon h=1 (h=4 reported alongside).

Two additions make the probe fair to the new orders:

1. **Symmetric temporal features.** P22's feature set encodes only the
   compute-first law (RAM/disk response to a CPU rise); a probe carrying only
   that law could not represent B or C, and its failure there would prove
   nothing. The `cross_lag` group adds, for every ordered pair of distinct
   resources and every lag the family registers, `ratio[lead](t−lag) −
   ratio[follow](t)`, so all three orders are representable — and no regime
   identifier is ever a feature (the probe asserts this).
2. **Per-regime onset targets.** A's target is a CPU onset, B's a RAM onset, C's
   a disk onset: a learner that only warns about CPU rises has not specialized
   in B.

#### Results

Probe run on `dev_seed700_steps2880`, 49 features, 18 cells (3 train regimes ×
3 test regimes × horizons 1 and 4), 0.8 s.

| trained on | evaluated on | h | target | AP | ROC-AUC | prevalence | test positives |
|---|---|---|---:|---:|---:|---:|---:|
| compute_first | **compute_first** | 1 | cpu_onset | **0.3383** | 0.9652 | 2.34 % | 127 |
| compute_first | memory_first | 1 | cpu_onset | 0.0565 | 0.8882 | 0.88 % | 50 |
| compute_first | io_first | 1 | cpu_onset | 0.0476 | 0.7998 | 1.32 % | 74 |
| memory_first | **memory_first** | 1 | ram_onset | **0.0807** | 0.9300 | 0.74 % | 42 |
| memory_first | compute_first | 1 | ram_onset | 0.0109 | 0.8354 | 0.25 % | 14 |
| memory_first | io_first | 1 | ram_onset | 0.0054 | 0.7462 | 0.21 % | 12 |
| io_first | **io_first** | 1 | disk_onset | **0.0136** | 0.9599 | 0.09 % | **5** |
| io_first | compute_first | 1 | disk_onset | 0.2407 | 0.9896 | 0.16 % | 9 |
| io_first | memory_first | 1 | disk_onset | 0.0306 | 0.8716 | 0.23 % | 13 |

§13 gate at the primary horizon h=1:

| regime | within-regime AP | mean cross-regime AP | gap | passes |
|---|---:|---:|---:|---|
| compute_first | 0.3383 | 0.0520 | **+0.2863** | yes |
| memory_first | 0.0807 | 0.0081 | **+0.0726** | yes |
| io_first | 0.0136 | 0.1356 | −0.1220 | no |

**H2 = PASS (2/3 regimes).** The compute-first and memory-first learners travel
badly to the other mechanisms while working on their own, which is the
specialization opportunity plan §13 asks for.

**Honest caveats, stated in the same place as the pass:**

- the io-first row is **positive-starved**: its own test window holds only 5 disk
  onsets (prevalence 0.09 %), so its within-regime AP is not estimable from this
  stream and the gate must not be read as "C is not learnable";
- the io-first learner also *transfers better than it fits* (0.2407 on
  compute-first versus 0.0136 on its own regime), which is a symptom of the same
  sparsity rather than a claim about io-first knowledge;
- the probe sees only its registered features; a learner with different inputs
  could transfer differently. What is established is that **this** probe, on
  **this** stream, does not carry one mechanism's onset knowledge to another.

### 5.2 Gradient interference (plan §14)

Instrument: `probe_ftmoe_protocol023_gradient.py` (64 events per regime,
192 events, 20 parameter groups). Registered pass rule: at least one regime pair
with mean residual-gradient cosine ≤ **−0.05**, or ≥ **30 %** of sampled
cross-regime event pairs with cosine < 0.

#### The registered start state cannot answer §14 — measured, not assumed

The frozen checkpoint contains no learner weights, and Protocol 020 R1 builds a
**fresh** residual bank whose read-out layer is zeroed
(`recovery/PreGANSrc/src/ftmoe_online_r1.py:80-84`). Measured at that state:
`correction_logits.abs().max() == 0.0`, only 10 of 26 trainable tensors receive a
non-zero gradient, and all 8 hidden-layer groups have **exactly zero** gradient,
so no cosine exists for them.

The instrument therefore reports those groups as `null`, counts them under
`n_pairs_undefined`, excludes them from the verdict, and writes the gate
**`INADMISSIBLE`** with the registered outcome preserved underneath. It also
accepts `--learner-state <checkpoint with learner.* tensors>` so §14 can be
evaluated on a mature fixed-C learner; no such checkpoint exists in this round.

What the inadmissible reading nevertheless shows (read-out block, the only
non-degenerate part):

| pair | mean cosine | fraction of event pairs with cosine < 0 |
|---|---:|---:|
| compute_first \| memory_first | +0.8599 | 6.30 % |
| compute_first \| io_first | +0.7659 | 3.71 % |
| memory_first \| io_first | +0.9287 | 4.86 % |

All three pairs are strongly co-directional, and the registered rule would have
returned **STOP-GI** (no conflict). Because the bank is untrained, this is
reported as the geometry of an **untrained** read-out head and nothing more:
if the co-directionality survives a mature learner, §14 argues *against*
implementing D — which makes "produce a mature learner state" the decisive next
measurement, not an optional one.

### 5.3 Sequential forgetting (plan §14, second half)

Train on one regime, evaluate all three: if learning B visibly costs A, a real
continual-learning conflict exists; if not, the environment does not yet justify
specialization. This is read from the same probe cells as §5.1
(`train=R/test=R'`), so the forgetting table and the specialization table come
from one instrument and one run. It is a **probe-level bound**, not the fixed-C
online comparison that plan §27 asks for — that belongs to S4.

| trained on | evaluated on | same-regime AP | cross-regime AP | drop | test positives |
|---|---|---:|---:|---:|---:|
| compute_first | memory_first | 0.3383 | 0.0565 | **−0.2818** | 50 |
| compute_first | io_first | 0.3383 | 0.0476 | **−0.2907** | 74 |
| memory_first | compute_first | 0.0807 | 0.0109 | **−0.0698** | 14 |
| memory_first | io_first | 0.0807 | 0.0054 | **−0.0753** | 12 |
| io_first | compute_first | 0.0136 | 0.2407 | +0.2271 | 9 |
| io_first | memory_first | 0.0136 | 0.0306 | +0.0170 | 13 |

Reading: for the two regimes whose within-regime skill is estimable
(compute-first, memory-first) the cross-regime drop is large and negative — a
learner that fits one mechanism does not carry that fit to another, which is the
"probe does not transfer" counterpart of §5.1 and the direction plan §14's
"sequential forgetting" step looks for. The io-first row is the sparse-cohort
row again and is not evidence of transfer.

**H4 = measured (probe-level), not closed.** Closing it requires the S4
prequential fixed-C run (plan §27), which is outside this round's scope.

---

## 6. Gate status

| Gate | Stage | Rule | Status |
|---|---|---|---|
| H0 generator registration correctness | S1 | compute-first byte-identical to `cascade_v2`; familiar identity exact; lags observable | **PASS** |
| H1 data gate + marginal matching | S2 | plan §9 thresholds per regime | **FAIL** (strict follow-up reading: 24/36/31 < 50; every other check passes, prevalence spread 3.75 pp ≤ 4 pp) — definitional decision open |
| H2 specialization opportunity | S3 | plan §13 | **PASS** (2/3 regimes: gaps +0.2863 and +0.0726; io-first positive-starved) |
| H3 gradient interference | S3 | plan §14 | **INADMISSIBLE at the registered start state** (untrained residual bank; registered rule would say STOP-GI on the read-out block) — needs a mature fixed-C learner |
| H4 sequential forgetting | S3 | learning a new regime costs the old one | **MEASURED (probe-level)**, not closed; needs the S4 prequential run |

Machine-readable status: `artifacts/ftmoe_online/protocol_023/gate_status.json`
(written by `summarize_ftmoe_protocol023_s3.py`, which only copies verdicts the
analyzers actually produced).

**Round-1 bottom line:** the environment is registered, collected, verified and
audited; the two gates that this round could decide are decided (H0 PASS,
H2 PASS); H1 is blocked on a definitional question, and H3 — the gate that would
most directly argue for or against dynamic experts — cannot be read at the start
state that this round was scoped with.

---

## 7. What this round can and cannot claim

**Will be claimable (if the gates pass):** that the three registered mechanisms
differ in a way that a single fixed-topology learner cannot absorb without cost —
measured as specialization opportunity, gradient conflict and forgetting — under
a matched-marginal scenario whose compute-first arm is Protocol 022's own
already-audited regime.

**Not claimable from this round, by construction:**

1. **Anything about D.** D is not implemented; no D number exists.
2. **Confirmation-level evidence.** All of S2/S3 runs on replay seed 700, which
   is a development seed. Seeds 701–703 are confirmation seeds and are untouched;
   one development seed cannot close a claim.
3. **That the regimes describe real industrial faults.** They are *controlled
   unseen temporal resource-demand regimes*: the phase law is built to be
   admission-safe under `getPlacementPossible()`, which is a property of this
   simulator, not a claim about how real memory leaks or log accumulation
   progress.
4. **That amplitude matching alone makes the comparison fair.** Marginal matching
   is measured and reported (plan §9), but resource order is exactly the variable
   under test, so any residual marginal difference is a threat to the
   interpretation and is reported next to every result.
5. **That the frozen C model is the best fixed baseline.** Only the registered
   fixed-C configuration is used; C-wide and C-budget comparisons belong to the
   round that implements D (plan §16).

---

## 8. Entry points

| Purpose | File |
|---|---|
| Plan (upstream) | `指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md` |
| Problem log | `artifacts/ftmoe_online/protocol_023/problem_log.jsonl` |
| Registration | `artifacts/ftmoe_online/protocol_023/protocol.json`, `gate_status.json`, `source_sha256_initial.json` |
| Generator | `simulator/workload/BitbrainWorkloadProtocol023.py` |
| Audit instrument | `ftmoe_protocol023_core.py` (+ `test_ftmoe_protocol023_core.py`) |
| Generator tests / verifier | `test_ftmoe_protocol023_regimes.py`, `verify_ftmoe_protocol023_generator.py` |
| Streams | `prepare_ftmoe_protocol023_stream.py`, `run_ftmoe_protocol023_s2.py`, `verify_ftmoe_protocol023_stream.py` |
| Data gate | `analyze_ftmoe_protocol023_s2.py` → `artifacts/ftmoe_online/protocol_023/data_audit/` |
| Specialization / gradient | `probe_ftmoe_protocol023_specialization.py`, `probe_ftmoe_protocol023_gradient.py` → `artifacts/ftmoe_online/protocol_023/specialization/` |
| Streams on disk | `artifacts/ftmoe_online/protocol_023/development_streams/` |
