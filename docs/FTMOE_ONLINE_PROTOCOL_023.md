# Protocol 023 — multi-regime recurring drift (dynamic experts vs fixed online fine-tuning)

**Status:** round 1 (S0–S3) 2026-09-12 · round 2A 2026-09-13 · branch `protocol-023`
**Round-1 commit:** `d7bbc8c` · **round-2A commit:** `edb0e10` · parent `protocol-022 @ 6c5db86f`
**Round scope:** D is **not implemented**. `D_eligible = false` after round 2A.
**Machine-readable:** `artifacts/ftmoe_online/protocol_023/{protocol.json,gate_status.json,source_sha256_initial.json}`, `round2a/gate_status_round2a.json`

> This file reports only what an artifact can prove. Every stage carries the command or
> file that produced it. Where a stage has not run, the section says so.

---

## 1. Why this protocol exists

Protocol 022 measured the thing that killed the dynamic-expert claim for a single unseen
regime (P22-22):

| updates | late-unseen PR-AUC (fixed C) |
|---:|---:|
| 100 | 0.4256 |
| 200 | 0.4374 |
| 400 | 0.5200 |
| 800 | 0.6568 |
| 1600 | 0.6891 |

The curve was **still climbing** at the largest budget (+0.0323 over the last doubling) while
the pre-onset warning metric did not improve (onset AP 0.0917 frozen A vs 0.0898–0.0908 for
every residual variant). A fixed expert set therefore had no demonstrated capacity plateau on
that regime, and without a plateau a dynamic-expert mechanism has no scientific necessity:
the residual error was a training-budget artefact, not a capacity deficit.

Protocol 023 changes the *environment*, not the model: three heterogeneous cascading
mechanisms, short dwell times, switching, and recurrence — the setting in which a
stability–plasticity conflict can actually exist. The round-1 question is deliberately
answerable without D:

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
| Parent | `protocol-022 @ 6c5db86fde8afd10e3997d4ff0bf8016e2e87130` | `protocol.json` |
| Plan hash | `097889694e…091c3` | `protocol.json` → `plan_sha256` |
| Sources pinned before collection | 14 files, SHA256 each | `source_sha256_initial.json` |
| Frozen online start checkpoint | `protocol_020/s6/adapted_v4_seed1/best.pt`, `10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b` | `protocol.json` → `frozen_checkpoint` |
| Round-1 scope | S0–S3 allowed; D, backbone changes and any retraining of v4/P19/P20/P22 forbidden | `protocol.json` → `round1_scope` |
| Gate status | all five gates start `pending` | `gate_status.json` |

Registration is performed by `register_ftmoe_protocol023.py`, which re-reads `HEAD`, hashes
the plan and the sources, and refuses to invent a verdict: the gate table is written
`pending` and is only changed by the analyzer that measured it.

---

## 3. S1 — the registered regime family `cascade_v3`

Generator: `simulator/workload/BitbrainWorkloadProtocol023.py` (`Protocol023MultiRegimeBWGD2`).

| Regime | Order (lead → lag → follow) | Onset resource | Onset threshold | Physical story |
|---|---|---|---|---|
| A `compute_first` | CPU →4 RAM →8 Disk | CPU | 2600 | compute-intensive inference burst |
| B `memory_first` | RAM →3 Disk →6 CPU | RAM | 2810 *(provisional)* | working-set growth / leak → paging → paging CPU cost |
| C `io_first` | Disk →3 CPU →6 RAM | Disk | 11600 *(provisional)* | logging / checkpoint accumulation → compaction → cache disturbance |

What the three regimes share, and what they may not share:

- **Shared (matched by construction):** the familiar admission law, the CPU burst band
  4400–5200, the RAM response band 4500–6000, the disk retention peak 16000 (cap 24000), the
  duration distributions of the new orders, the phase shape law, and the onset probability.
- **Allowed to differ:** resource order, lag, and which resource the cascade starts in. The
  plan requires the *marginals* to match while the *joint structure* changes; a scenario that
  only raised amplitudes would not be evidence.

### 3.1 The inherited regime is exact, not approximate

`compute_first` is Protocol 022's `cascade_v2`: same registered mechanism seed (22022), same
ramp multiplier (2.0), same durations, same shape law. Check A-01 of
`verify_ftmoe_protocol023_generator.py` asserts **zero** trajectory differences on CPU, RAM
and disk for every generated task, plus element-wise equality of the event envelopes. That
check is the reason the A-arm of this round is comparable with Protocol 022 at all.

### 3.2 Admission safety is structural

The registered phase law is exactly **zero at the first age of every window** for every
resource, so no phase can make a task unplaceable at admission. This generalises the P21-01
constraint (`Simulator.getPlacementPossible()` evaluates demand at the admission interval,
which produced 93.7 % deployment rejection for a flat burst floor) from the CPU phase to all
three resources and all three regimes. Check A-03 measures it per regime rather than assuming.

### 3.3 Observability

Every registered lag must be visible inside the 12-interval model history or the regime is
unseen *and unlearnable*. The two new orders use 4–5 interval phases so their longest chain
(6 + 5 = 11) fits inside it. `compute_first` keeps Protocol 022's longer windows (longest
chain age 17) — that regime's onset is detected, not its whole chain reconstructed — and the
exemption is registered explicitly in `INHERITED_CHAIN_REGIMES` instead of being a silent
exception.

### 3.4 Verification of the generator

| Check | What it proves | Result |
|---|---|---|
| `test_ftmoe_protocol023_regimes` (28 tests) | registered physics, per-regime onsets, envelope purity, registration guards | PASS |
| `test_ftmoe_protocol023_core` (66 tests) | the multi-regime audit instrument | PASS |
| `verify_ftmoe_protocol023_generator.py` A-01 | compute-first byte-identical to `cascade_v2` | PASS (0 differences) |
| A-02 | probability 0 is the frozen familiar transform for all three regimes | PASS |
| A-03 | each regime's onset fires on its own resource only, and is admission-safe | PASS |
| A-04 | every registered lag is observable; new orders fit the history | PASS |

```
D:\Anaconda\envs\dynmoe\python.exe verify_ftmoe_protocol023_generator.py
```

---

## 4. S2 — data-only streams and the data gate

Collector `prepare_ftmoe_protocol023_stream.py` · runner `run_ftmoe_protocol023_s2.py` ·
independent verifier `verify_ftmoe_protocol023_stream.py` · gate `analyze_ftmoe_protocol023_s2.py`.

### 4.1 Registered timeline (development, replay seed 700)

```text
F0 familiar      300
A1 compute-first 420        B1 memory-first 420        C1 io-first 420
F1 familiar      240
A2 recurrence    360        C2 recurrence   360        B2 recurrence 360
---------------------
scored          2880 intervals (+1 guard row)
```

The recurrence half deliberately returns the regimes in a different order (A → C → B) so that
"recurrence" is a repetition of a *mechanism*, not of a schedule position.

Three single-regime streams (`single_<regime>_seed700_steps1200`: 150 familiar + 1050 of one
regime) are collected alongside it, because the marginal-matching gate asks for per-regime
cohorts not confounded by the other two mechanisms.

### 4.2 What the collector guarantees

- one process at a time, 3 torch threads, 1 interop thread, BelowNormal priority, a 2.5 GiB
  RAM guard with bounded waits (every wait recorded), a 20 GiB disk guard, and
  `stats.history_limit = 64` / `stats.series_tail = 96` (the P22-18/20 memory fixes, without
  which this stream length does not fit on this machine);
- the true child exit code, never a shell wrapper's verdict (the P20 defect Protocol 022 banned);
- `failure.json` with the traceback if anything raises, and refusal to overwrite an existing
  stream directory;
- phase/regime/mechanism/event columns written **audit-only**, with the forbidden input tokens
  registered in the manifest.

### 4.3 Independent verification

`verify_ftmoe_protocol023_stream.py <stream_dir>` recomputes 61 checks from the saved arrays
and JSON — never trusting the manifest's own numbers — and exits non-zero on any failure. It
measures, rather than assumes: the row layout (`steps + 1`) and phase geometry; that each
phase's cascade tasks carry that phase's regime and no other; task-timeline integrity, slot
reuse and migration continuity; per-regime onset counts **with** and **without** the task
cohort, so the cross-regime confound is reported instead of hidden; the familiar per-task
maximum of each onset resource next to the registered threshold and floor; and finiteness of
every model-facing array (a single NaN fails the check).

### 4.4 Round-1 results (replay seed 700, 2026-09-12)

All four streams had a real child exit code of 0 and a verifier exit code of 0
(`artifacts/ftmoe_online/protocol_023/s2_run_report.json`):

| Stream | scored intervals | elapsed | child exit | verifier |
|---|---:|---:|---:|---:|
| `single_compute_first_seed700_steps1200` | 1200 | 439.6 s | 0 | 0 |
| `single_memory_first_seed700_steps1200` | 1200 | 444.4 s | 0 | 0 |
| `single_io_first_seed700_steps1200` | 1200 | 447.8 s | 0 | 0 |
| `dev_seed700_steps2880` | 2880 | 3310.3 s | 0 | 0 |

Gate numbers (`analyze_ftmoe_protocol023_s2.py`, recomputed from the arrays):

| Regime | prevalence | independent fault events | registered cascade events | deployment rej. | migration rej. | worst event share | valid onset w/ full follow-up | usable follow-up (P22 def.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A compute-first | 6.80 % | 410 | 342 | 13.20 % | 36.29 % | 1.57 % | **24** | 185 |
| B memory-first | 3.05 % | 260 | 297 | 8.45 % | 15.43 % | 1.56 % | **36** | 200 |
| C io-first | 3.79 % | 271 | 297 | 9.32 % | 22.17 % | 1.41 % | **31** | 235 |

- prevalence spread **3.75 pp ≤ 4 pp** → the registered marginal-matching check passes, tightly;
- every other registered check passes;
- **onset-threshold admissibility is measured, not assumed**: familiar per-task maxima
  cpu 1860 / ram 1400 / disk 9000 against thresholds A 2600 (floor 4400), B 2810 (4500),
  C 11600 (16000) — all three strictly admissible;
- **one registered check fails**: "valid onset/follow-up ≥ 50 / regime" is 24 / 36 / 31 under
  the strict reading. Cause is structural — median observed cohort life is 5–6 intervals
  against 16–18 interval registered response horizons — and the P22-compatible reading gives
  185 / 200 / 235.

**H1 = FAIL on the strict reading** (STOP-DATA recorded), with the definitional question raised
for the reviewer rather than resolved silently. See `data_audit/DATA_GATE.md`.

---

## 5. S3 — specialization, gradient conflict, forgetting

### 5.1 Cross-regime specialization probe

Instrument `probe_ftmoe_protocol023_specialization.py`, built on Protocol 022's closed-form
logistic probe so the numbers stay comparable with P22-S4. Registered gate: within-regime
AP − mean cross-regime AP ≥ **0.05** for at least **2 of 3** regimes, at h=1.

Two additions make the probe fair to the new orders: **symmetric temporal features** (P22's
feature set encodes only the compute-first law; the `cross_lag` group adds, for every ordered
pair of distinct resources and every registered lag, `ratio[lead](t−lag) − ratio[follow](t)`,
so all three orders are representable — and no regime identifier is ever a feature), and
**per-regime onset targets** (A's target is a CPU onset, B's a RAM onset, C's a disk onset).

Result on `dev_seed700_steps2880`, 49 features, 18 cells (3 train × 3 test regimes × horizons
1 and 4), 0.8 s:

| trained on | evaluated on | h | target | AP | ROC-AUC | prevalence | test positives |
|---|---|---|---|---:|---:|---:|---:|
| compute_first | **compute_first** | 1 | cpu_onset | **0.3383** | 0.9652 | 2.34 % | 127 |
| compute_first | memory_first | 1 | cpu_onset | 0.0565 | 0.8882 | 0.88 % | 50 |
| compute_first | io_first | 1 | cpu_onset | 0.0476 | 0.7998 | 1.32 % | 74 |
| memory_first | **memory_first** | 1 | ram_onset | **0.0807** | 0.9300 | 0.74 % | 42 |
| memory_first | compute_first | 1 | ram_onset | 0.0109 | 0.8354 | 0.25 % | 14 |
| memory_first | io_first | 1 | ram_onset | 0.0054 | 0.7462 | 0.21 % | 12 |
| io_first | **io_first** | 1 | disk_onset | **0.0136** | 0.9599 | 0.09 % | **5** |
| io_first | compute_first | 1 | disk_onset | 0.2407 | 0.9896 | 0.16 % | 9 |
| io_first | memory_first | 1 | disk_onset | 0.0306 | 0.8716 | 0.23 % | 13 |

| regime | within-regime AP | mean cross-regime AP | gap | passes |
|---|---:|---:|---:|---|
| compute_first | 0.3383 | 0.0520 | **+0.2863** | yes |
| memory_first | 0.0807 | 0.0081 | **+0.0726** | yes |
| io_first | 0.0136 | 0.1356 | −0.1220 | no |

**H2 = PASS (2/3 regimes).** Honest caveats, stated in the same place as the pass: the io-first
row is **positive-starved** (only 5 disk onsets, prevalence 0.09 %), so its within-regime AP is
not estimable from this stream and the gate must not be read as "C is not learnable"; the
io-first learner also *transfers better than it fits*, a symptom of the same sparsity; and the
probe sees only its registered features. What is established is that **this** probe, on
**this** stream, does not carry one mechanism's onset knowledge to another.

### 5.2 Gradient interference

Instrument `probe_ftmoe_protocol023_gradient.py` (64 events per regime, 192 events, 20
parameter groups). Registered pass rule: at least one regime pair with mean residual-gradient
cosine ≤ **−0.05**, or ≥ **30 %** of sampled cross-regime event pairs with cosine < 0.

**The registered start state cannot answer §14 — measured, not assumed.** The frozen
checkpoint contains no learner weights, and Protocol 020 R1 builds a **fresh** residual bank
whose read-out layer is zeroed (`recovery/PreGANSrc/src/ftmoe_online_r1.py:80-84`). Measured
at that state: `correction_logits.abs().max() == 0.0`, only 10 of 26 trainable tensors receive
a non-zero gradient, and all 8 hidden-layer groups have **exactly zero** gradient, so no cosine
exists for them. The instrument reports those groups as `null`, counts them under
`n_pairs_undefined`, excludes them from the verdict, and writes the gate **`INADMISSIBLE`**
with the registered outcome preserved underneath. It accepts
`--learner-state <checkpoint with learner.* tensors>` so §14 can be evaluated on a mature
fixed-C learner. What the inadmissible reading nevertheless shows (read-out block, the only
non-degenerate part):

| pair | mean cosine | fraction of event pairs with cosine < 0 |
|---|---:|---:|
| compute_first \| memory_first | +0.8599 | 6.30 % |
| compute_first \| io_first | +0.7659 | 3.71 % |
| memory_first \| io_first | +0.9287 | 4.86 % |

All three pairs are strongly co-directional, and the registered rule would have returned
**STOP-GI** (no conflict). Because the bank is untrained, this is reported as the geometry of
an **untrained** read-out head and nothing more.

### 5.3 Sequential forgetting

Train on one regime, evaluate all three: if learning B visibly costs A, a real continual-
learning conflict exists. Read from the same probe cells as §5.1, so the forgetting table and
the specialization table come from one instrument and one run. It is a **probe-level bound**,
not the fixed-C online comparison that S4 owns.

| trained on | evaluated on | same-regime AP | cross-regime AP | drop | test positives |
|---|---|---:|---:|---:|---:|
| compute_first | memory_first | 0.3383 | 0.0565 | **−0.2818** | 50 |
| compute_first | io_first | 0.3383 | 0.0476 | **−0.2907** | 74 |
| memory_first | compute_first | 0.0807 | 0.0109 | **−0.0698** | 14 |
| memory_first | io_first | 0.0807 | 0.0054 | **−0.0753** | 12 |
| io_first | compute_first | 0.0136 | 0.2407 | +0.2271 | 9 |
| io_first | memory_first | 0.0136 | 0.0306 | +0.0170 | 13 |

For the two regimes whose within-regime skill is estimable the cross-regime drop is large and
negative — the "probe does not transfer" counterpart of §5.1. The io-first row is the
sparse-cohort row again and is not evidence of transfer. **H4 = measured (probe-level), not
closed.**

### 5.4 Round-1 gate status

| Gate | Stage | Status |
|---|---|---|
| H0 generator registration correctness | S1 | **PASS** |
| H1 data gate + marginal matching | S2 | **FAIL** (strict follow-up reading 24/36/31 < 50; every other check passes) — definitional decision open |
| H2 specialization opportunity | S3 | **PASS** (2/3 regimes) |
| H3 gradient interference | S3 | **INADMISSIBLE at the registered start state** — needs a mature fixed-C learner |
| H4 sequential forgetting | S3 | **MEASURED (probe-level)**, not closed |

---

## 6. Round 2A (S2.5 → S4 → H3-v2/H4-v2/§15)

**Directive:** the round-2A directive. **Verdicts:** `round2a/gate_status_round2a.json`.

Round 2A did four things and no more: it resolved the follow-up definition (H1-v2 registered
and **passing**, H1-original stays **FAIL**), re-tuned B and C on data only so the third
specialist had enough positives to be measurable (H2-v2 **3 of 3**, io-first 12 → 31
positives), ran the first strict-prequential fixed-C comparison under a frozen
deployment-plausible online budget (arms A and C, 720 updates, integrity audited and passing),
and re-read the gradient gate on a **mature** learner — which makes H3 admissible for the first
time and returns a clean **STOP-GI**.

**The round's answer:** the stability–plasticity conflict this protocol exists to test **does
not hold** in this scenario. Fixed C gains only 0.2–2.0 PR-AUC points over the frozen base in
the regimes it is exposed to (registered margin +3.0), never forgets the fixed probes, and
creates no gradient conflict on mature parameters. **`D_eligible = false`.**

### 6.1 Gate state

| condition | result | the number that decides it |
|---|---|---|
| H1-v2 PASS | **true** | primary-window follow-up 185 / 405 / 428 vs ≥ 80; prevalence spread 1.42 pp |
| H2-v2 PASS | **true** | gaps +0.226 / +0.216 / +0.123; positives 185 / 124 / **31** |
| C learns ≥ 2 regimes | **false** | late-100 gain over A: **+0.0049 / +0.0020 / +0.0203** vs +0.03 |
| H3-v2 mature gradient conflict | **false** | mean cosines +0.71…+0.95, negative fractions 0.020…0.094 |
| H4-v2 actual forgetting | **false** | not evaluated (step 1 failed); diagnostic drops 0.0073 / 0.0150, both < 0.03 |
| recurrence relearning | **false** | raw §15 gap holds at B2 (+0.1536) but the *frozen baseline* shows a larger one (+0.2121) |
| online budget frozen | **true** | `amendments/online_budget.json`, unchanged after freeze |
| prequential integrity PASS | **true** | all six checks pass on both arms |

H0 remains PASS and its regression holds: the calibrated `compute_first` arm is
**byte-identical** to the registered one
(`check_ftmoe_protocol023_round2a_calibration.py.ps1`), so the A arm is still Protocol 022's
`cascade_v2`.

### 6.2 What changed in the data (S2.5) and the trade it makes

Measured first, tuned second. The io-first deficit is structural, not a defect: host disk
capacity is 28990.8 on every host, a cascade task's disk phase reaches 16000 with a 24000 cap,
so a *host* disk fault needs two co-located disk phases — and any shorter disk phase destroys
that overlap (candidates at disk duration 2–3 collapsed to 2 disk cells). The only workable
levers are the cascade probability and the registered disk peak.

| regime | parameter | registered | calibrated |
|---|---|---|---|
| compute_first | — | — | **frozen** |
| memory_first | `cascade_task_probability` | 0.25 | **0.50** |
| io_first | `cascade_task_probability` | 0.25 | **0.45** |
| io_first | `disk_retained_peak` | 16000 | **24000** (cap unchanged) |

Result: h=1 positives 24 → **57** (io-first) and 122 → **224** (memory-first) at the registered
1200-interval shape; prevalence A 6.80% / B 5.45% / C 6.88% (spread 1.42 pp, was 3.75 pp).

**The trade, stated plainly:** event counts rose to 600 and 532 against A's 342 — **+75% and
+56% where the target is ≤ 20%** — and the duration and peak-ratio targets are not met for
memory-first (0.346 and 0.638 against 0.25 / 0.30). Every candidate that kept the event count
near A's produced 2–22 h=1 positives instead of 51+. If the reviewer prefers the marginal-match
target over the positive budget, that is a one-line change to `selected_generator.json` plus a
re-collection; the round chose the positives because §5 makes them the precondition for a third
specialist.

**Known side effect, reported not hidden:** the calibrated stream's *post-regime* familiar
phase is visited by far more in-flight regime tasks, so the whole-familiar-phase admissibility
scan now reads cpu 5200 / ram 5424 / disk 20000 against round 1's 1860 / 1400 / 9000. The
mechanism-off phase F0 still measures exactly 1860 / 1400 / 9000, so the registered onset
thresholds stay admissible; the polluted reading is recorded next to the clean one.

### 6.3 The result that matters for D

Fixed C, strict prequential, frozen budget (720 updates over 2880 intervals, update-every-4,
batch 32, one step per opportunity):

| regime | late-100 PR-AUC C | late-100 PR-AUC A | gain |
|---|---:|---:|---:|
| compute_first | 0.4111 | 0.4062 | +0.0049 |
| memory_first | 0.5683 | 0.5663 | +0.0020 |
| io_first | 0.5946 | 0.5743 | +0.0203 |

**0 of 3 reaches +0.03.** §14 step 1 fails, so forgetting is *not* evaluated
(`forgets: null`, not `false`) and the stop condition is **`STOP-NO-LEARN`**.

Two further measurements, both pointing the same way:

* **No forgetting on the fixed probes.** C's three probe slices never fall below their start
  (worst excursion −0.0030) and improve monotonically after F1, while the frozen A arm's slices
  never move at all. C ends uniformly above A (+0.175 / +0.130 / +0.163), i.e. what the
  residual learns is *not* regime-specific.
* **No gradient conflict on mature parameters.** At `C_after_A1/B1/C1` the degenerate-group
  list is empty (round 1's INADMISSIBLE blocker is resolved) and the registered rule reads no
  conflict at every checkpoint.

The sharpest way to put it: under Protocol 022's *offline* budget C kept improving with more
updates; under this round's frozen *online* budget the same arm barely moves. That is the first
measurement in this protocol family that argues *against* needing a dynamic expert, and it is
budget-limited rather than mechanism-limited.

### 6.4 Decisions the reviewer may want to make

1. **Is the calibration trade acceptable?** (+56–75% event count for +5× the h=1 positives.)
   If not, the alternative is a marginal-matched but positive-starved C, which caps the
   specialist claim at 2 of 3.
2. **Is the frozen online budget the right one to test D under?** The whole negative result
   turns on it. A budget sweep (2× / 4× updates) would say whether the plateau is a budget
   artefact — this is *not* C-budget as round 1 defined it (that was an offline diagnostic),
   and the directive forbids C-budget this round, so it is a next-round question.
3. **Is the §15 confound control the right standard?** The round added a baseline comparison
   the directive does not require (the frozen arm shows a *larger* relearning gap on every
   transition) and used it to fail the condition. A reviewer who wants the literal §15 rule
   alone would read that condition as PASS.
4. **io-first clears the reliability bar by exactly one positive** (31 vs 30). Usable, but
   confirmation seeds 701–703 are untouched and would be the honest next check.
5. **Event-count equality vs prevalence equality.** Round 2A keeps prevalence nearly identical
   across regimes and lets event counts diverge. If the reviewer considers event count the more
   important marginal, the regimes need a different response-duration law, which is a new
   family (`cascade_v4`), not a calibration.

### 6.5 Where the round-2A artifacts are

```
artifacts/ftmoe_online/protocol_023/
  development_streams/
    dev_seed700_steps2880                                     round 1, preserved
    single_{compute,memory,io}_first_seed700_steps1200         round 1, preserved
    dev_seed700_steps2880_calibrated                           round 2A (verified, 63 checks)
    single_*_seed700_steps1200_calibrated                      round 2A (verified, 63 checks)
  round2a/
    amendments/{h1_v2_definition,online_budget}.json
    data_calibration/{candidate_table,selected_generator,marginal_match_v2,data_gate_v2}.json
    diagnostics/{disk_onset_deficit,positive_budget,h1_positive_budget_by_window,envelope_landing}.json
    specialization_v2/cross_regime_probe_v2.{json,md}
    probe_slices/{probe_*.npz,probe_registry.json}
    fixed_c_prequential/
      ARTIFACT_INDEX.json                     where each per-arm file is
      arm_{A,C}/{predictions.npz,settlements.jsonl,update_log.jsonl,phase_metrics.json,...}
      probe_matrix.{json,md}                  the checkpoint × probe table
      forgetting_v2.json                      H4-v2
      recurrence_metrics.json                 §15 + the baseline attribution
    mature_gradient/{after_A1,after_B1,after_C1,summary}.json
    checkpoints/C_after_{A1,B1,C1,F1,A2,C2,B2}.pt
    gate_status_round2a.json                  D_eligible = false
    problem_log_round2a.jsonl                 R2A-01 … R2A-09
  (round 1's own artifacts, including H1-original FAIL and gate_status.json, are untouched)
```

Two bugs were found the hard way and are worth knowing if the calibration is re-run: the
generator's `regime_params` overlays `_REGISTERED_COMMON` onto every regime (so writing a
shared value only into `REGIMES_V3` is silently discarded), and the collector's phase switch
resets the probability from the phase table (so a calibrated probability must be written into
the phase table itself). Both produced byte-identical "calibrated" streams before they were
fixed; the calibration guard script exists to make that failure mode impossible to miss again.

---

## 7. What this round can and cannot claim

**Claimable:** that the three registered mechanisms differ in a way that a single fixed-topology
learner does not fully absorb — measured as specialization opportunity (H2/H2-v2) and, after
round 2A, that the fixed-C residual nevertheless captures nearly all of the available gain under
the frozen online budget, so the stability–plasticity conflict is **not** present here.

**Not claimable, by construction:**

1. **Anything about D.** D is not implemented; no D number exists.
2. **Confirmation-level evidence.** All runs use replay seed 700, a development seed. Seeds
   701–703 are confirmation seeds and are untouched; one development seed cannot close a claim.
3. **That the regimes describe real industrial faults.** They are *controlled unseen temporal
   resource-demand regimes*: the phase law is built to be admission-safe under
   `getPlacementPossible()`, a property of this simulator, not a claim about how real memory
   leaks or log accumulation progress.
4. **That amplitude matching alone makes the comparison fair.** Marginal matching is measured
   and reported, but resource order is exactly the variable under test, so any residual
   marginal difference is a threat to the interpretation and is reported next to every result.
5. **That the frozen C model is the best fixed baseline.** Only the registered fixed-C
   configuration is used; C-wide and C-budget comparisons belong to the round that implements D.

---

## 8. Entry points

| Purpose | File |
|---|---|
| Registration | `register_ftmoe_protocol023.py` → `protocol.json`, `gate_status.json`, `source_sha256_initial.json` |
| Generator | `simulator/workload/BitbrainWorkloadProtocol023.py` |
| Audit instrument | `ftmoe_protocol023_core.py` (+ `test_ftmoe_protocol023_core.py`) |
| Generator tests / verifier | `test_ftmoe_protocol023_regimes.py`, `verify_ftmoe_protocol023_generator.py` |
| Streams | `prepare_ftmoe_protocol023_stream.py`, `run_ftmoe_protocol023_s2.py`, `verify_ftmoe_protocol023_stream.py` |
| Data gate | `analyze_ftmoe_protocol023_s2.py` → `protocol_023/data_audit/` |
| Specialization / gradient | `probe_ftmoe_protocol023_specialization.py`, `probe_ftmoe_protocol023_gradient.py` |
| Round 2A calibration | `calibrate_ftmoe_protocol023_s25.py`, `score_ftmoe_protocol023_s25_candidates.py`, `select_ftmoe_protocol023_s25_candidate.py` |
| Round 2A prequential | `run_ftmoe_protocol023_s4.py` → `round2a/fixed_c_prequential/` |
| Round 2A verdicts | `analyze_ftmoe_protocol023_round2a.py`, `analyze_ftmoe_protocol023_round2a_gates.py`, `assemble_ftmoe_protocol023_round2a_status.py` |
| Streams on disk | `artifacts/ftmoe_online/protocol_023/development_streams/` |
