# Protocol 023 — round 2A report (directive `FTMOE_PROTOCOL023_ROUND2A_DIRECTIVE_20260912.md`)

**Branch:** `protocol-023` · **parent commit:** `d7bbc8c75221ed8707bee80ab30fc83adbecc500` (round-1 review HEAD)
**Round scope:** S2.5 data-only calibration → H1-v2 → H2-v2 → S4 (arms A and C only) → H3-v2 / H4-v2 / §15
**D was NOT implemented.** No expert birth / retire / reactivate, no D trigger or ramp tuning, no C-wide, no C-budget.
**Seeds:** 700 only. 701/702/703 untouched.

## 0. The answer this round exists to produce

| condition (directive §18) | result |
|---|---|
| H1-v2 PASS | **true** |
| H2-v2 PASS | **true** |
| C learns ≥ 2 regimes | **false** |
| H3-v2 mature gradient conflict | **false** |
| H4-v2 actual forgetting | **false** |
| recurrence relearning | **false** (the raw §15 gap holds, but the frozen baseline shows a *larger* gap on the same transitions) |
| online budget frozen | **true** |
| prequential integrity PASS | **true** |

**`D_eligible = false`** — four of the eight entry conditions fail, and the two that
carry the scientific claim (does fixed C learn new regimes; does it lose old ones)
are the two that fail hardest. Two things are now measured that round 1 could not
measure at all: the H1-v2 follow-up definitional blocker is resolved, and the H3
gradient gate is admissible for the first time (and reads STOP-GI, not INADMISSIBLE).

---

## 1. S2.5 — data-only marginal calibration

Round 1's io-first arm produced 5 h=1 within-regime test positives (AP 0.0136) and
could not carry a third specialist claim. Before touching anything, the deficit was
measured rather than assumed, and it is structural:

> a single io-first disk phase reaches at most ~0.83 of the registered disk
> capacity (host disk capacity is 28990.8 on every host; a cascade task's disk
> phase peaks at 16000 and its cap is 24000), so a *host* disk fault requires two
> co-located disk phases. Any change that shortens the disk phase destroys the
> overlap and lowers the fault count.

Evidence: `diagnostics/disk_onset_deficit.json`, `diagnostics/positive_budget.json`.
Consequence: the only levers that move the io-first positive budget without touching
the registered cascade law are the **cascade probability** and the **registered disk
peak** (16000 → 24000, the already-registered cap).

**Frozen selection** (`data_calibration/selected_generator.json`):

| regime | parameter | registered | calibrated | why |
|---|---|---|---|---|
| compute_first | — | — | — | **frozen**; must stay byte-identical to Protocol 022's `cascade_v2` |
| memory_first | `cascade_task_probability` | 0.25 | **0.50** | prevalence 3.05% → 5.45% (A is 6.80%), h=1 positives 122 → 224* |
| io_first | `cascade_task_probability` | 0.25 | **0.45** | prevalence 3.79% → 6.88%, h=1 positives 24 → 57* |
| io_first | `disk_retained_peak` | 16000 | **24000** | widens the per-task crossing margin; cap unchanged |

\* measured on the calibration arms at the registered 1200-interval shape.

Nothing was tuned on a model metric: the whole candidate table is data-only
(`data_calibration/candidate_table.json`, 22 candidates). `compute_first` is
byte-identical between the registered and calibrated arms — checked, not assumed
(`check_ftmoe_protocol023_round2a_calibration.py.ps1`).

**Known cost of the calibration (reported, not hidden):** the higher cascade
probability leaves many more regime tasks in flight when the post-regime familiar
phase starts, so the *whole-familiar-phase* admissibility scan now reads
cpu 5200 / ram 5424 / disk 20000 against round 1's 1860 / 1400 / 9000. The phase
that is generated with the mechanism switched off (F0) still measures exactly
1860 / 1400 / 9000, so the registered onset thresholds remain admissible; the
polluted reading is kept next to the clean one in the verifier output.

### Marginal match against the frozen A arm (directive §4 targets)

| regime | event count | prevalence | duration | median peak ratio |
|---|---:|---:|---:|---:|
| compute_first (reference) | 342 | 6.80% | 2.643 | 17.50 |
| memory_first | 600 (+75%) | 5.45% (**−1.35 pp**) | 3.557 (+35%) | 6.33 (+64%) |
| io_first | 532 (+56%) | 6.88% (**+0.07 pp**) | 2.205 (+17%) | 11.98 (+32%) |
| target | ≤ 20% | ≤ 3 pp | ≤ 25% | ≤ 30% |

Prevalence is now matched far better than in round 1 (spread 1.42 pp against the
4 pp ceiling; round 1 was 3.75 pp). The **duration and peak-ratio targets are not
met** for memory_first, and the event count deliberately exceeds the 20% target:
the event count is what produces the h=1 positives the directive requires, and
every candidate that kept the event count near A's produced 2–22 positives instead
of 51+. This is a stated trade, recorded in `problem_log_round2a.jsonl` (R2A-08).

---

## 2. H1 — the original FAIL is preserved, and H1-v2 passes

**H1-original stays `FAIL`** (24 / 36 / 31 against a threshold of 50). It is not
re-evaluated and not rewritten; it is carried verbatim into every round-2A gate
artifact next to the new verdict.

H1-v2 (definition frozen in `amendments/h1_v2_definition.json` before any model run):
an onset counts when the task has at least one valid observation inside its
regime's **first downstream** response window — A: CPU→RAM `[t0+4, t0+14)`,
B: RAM→Disk `[t0+3, t0+13)`, C: Disk→CPU `[t0+3, t0+13)`. These are also each
regime's registered first downstream window, so the directive's numbers and the
registration agree.

| regime | primary-window follow-up (H1-v2) | full-window (round 1) | any-window | events | prevalence | deploy rej. | migrate rej. | worst event share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compute_first | **185** | 24 | 185 | 342 | 6.80% | 13.20% | 36.29% | 1.57% |
| memory_first | **405** | 60 | 405 | 600 | 5.45% | 10.88% | 28.05% | 1.26% |
| io_first | **428** | 54 | 428 | 532 | 6.88% | 12.45% | 25.06% | 0.78% |
| threshold | ≥ 80 | (≥ 50, FAIL preserved) | — | ≥ 80 | 3–12% | ≤ 25% | ≤ 40% | < 10% |

**H1-v2 = PASS**, no failed checks, no STOP-DATA-v2, prevalence spread 1.42 pp.
Read honestly: this is the *P22-compatible* reading that round 1's reviewer was
asked to choose between, now registered under its own name with the strict reading
kept as a permanent FAIL beside it. It passes on the round-1 streams too (185 / 200
/ 235), so it is not an artefact of the calibration.

---

## 3. H2-v2 — specialization, with the reliability condition

`specialization_v2/cross_regime_probe_v2.json`. Two halves, never conflated:
**reliability** (positives ≥ 30) is measured on the calibrated single-regime
streams on a legal exposure/tail split; the **gap** is measured on the development
stream (train on a mechanism's first exposure, test on the recurrence blocks).
Both splits are reported; the registered primary split is 0.50.

| regime | within AP | mean cross AP | gap | gap ok | positives @0.50 | positives @0.30 | passes |
|---|---:|---:|---:|---|---:|---:|---|
| compute_first | 0.2808 | 0.0545 | **+0.2262** | yes | 185 | 249 | **yes** |
| memory_first | 0.2262 | 0.0100 | **+0.2162** | yes | 124 | 202 | **yes** |
| io_first | 0.1366 | 0.0138 | **+0.1228** | yes | **31** | 44 | **yes** |

**H2-v2 = PASS, 3 of 3** (round 1: 2 of 3, with io_first at AP 0.0136 and 12
positives). The third specialist now clears both conditions — by one positive at
the registered split (R2A-09), which is why the 0.30-split reading (44) is reported
beside it and why this row should not be treated as robust without the
confirmation seeds.

---

## 4. S4 — strict prequential fixed C (arms A and C only)

Budget frozen in `amendments/online_budget.json` **before** the run and unchanged:
update every 4 scored intervals, batch 32, one gradient step per opportunity,
64-interval replay buffer, AdamW 1e-4, anchor + distillation as registered.
Runtime benchmark recorded in the same file. Realised: **720 updates** over 2880
intervals (105/105/105/60/90/90/90 per phase, exactly the registered table).

Prequential integrity (directive §11) audited on both arms and passing:
every prediction was written before its own interval's label was read, every
settled label index is strictly older than the read index, the final interval was
settled on the stream's own guard row, no training batch contained an unsettled or
future sample, and the frozen base's hash is unchanged. `predictions.npz`,
`settlements.jsonl`, `update_log.jsonl` are written per arm.

### 4.1 The fixed probe matrix (directive §13) — PR-AUC

Fixed held-out slices (`probe_slices/probe_registry.json`, registered before any
model run, never trained on, never in replay, never used to tune a threshold):

| checkpoint | Probe A | Probe B | Probe C |
|---|---:|---:|---:|
| start (= frozen A baseline) | 0.4138 | 0.4965 | 0.4986 |
| after A1 | 0.4231 | 0.5053 | 0.4983 |
| after B1 | 0.4158 | 0.5029 | 0.4956 |
| after C1 | 0.4292 | 0.4879 | 0.5125 |
| after F1 | 0.4382 | 0.4999 | 0.5279 |
| after A2 | 0.5114 | 0.5422 | 0.5623 |
| after C2 | 0.5824 | 0.5846 | 0.6331 |
| after B2 | **0.5891** | **0.6260** | **0.6620** |

Arm A produces one row and it is the `start` row: the frozen base has no learner,
so every A checkpoint is identical by construction (its S4 run performs 0 updates).

Two readings, both true and worth separating:

* **C never falls below its own start point** on any probe (worst excursion: Probe C
  −0.0030 at `after B1`), and from `after F1` onward every probe improves
  monotonically — there is no measurable forgetting of these slices.
* **C is above A at every checkpoint that matters**, and by a widening margin:
  Probe A 0.4138 → 0.5891 (+0.175), Probe B 0.4965 → 0.6260 (+0.130), Probe C
  0.4986 → 0.6620 (+0.163), against a frozen baseline that never moves. The two
  checkpoints where C sits marginally below A (Probe C at `after A1/B1`, Probe B at
  `after C1`) are within 0.011 and recover immediately.

What C learns is not probe-specific: the slices improve together, in and out of the
regime that is currently active.

### 4.2 H4-v2 — does C learn, and does it forget? (directive §14)

| regime | late-100 PR-AUC C | late-100 PR-AUC A | gain | whole-phase gain | ≥ +0.03? |
|---|---:|---:|---:|---:|---|
| compute_first | 0.4111 | 0.4062 | **+0.0049** | +0.0043 | no |
| memory_first | 0.5683 | 0.5663 | **+0.0020** | +0.0028 | no |
| io_first | 0.5946 | 0.5743 | **+0.0203** | +0.0095 | no |

**Step 1 fails: 0 of 3 first-exposure regimes meet the registered margin.** Per the
directive, forgetting is therefore **not evaluated** (`forgets: null`, not `false`)
and the stopping condition is **`STOP-NO-LEARN`**. The transition drops are still
reported as diagnostics: A1→B1 the compute probe drops 0.0073, B1→C1 the memory
probe drops 0.0150 — both below 0.03, and both smaller than the gains C makes
elsewhere, so they are not forgetting in any usable sense.

The honest summary of the S4 result: **fixed C adapts only marginally under a
deployment-plausible online budget**, gaining 0.2–2.0 PR-AUC points over a frozen
base that already scores 0.41–0.59, and it never beats A by the registered margin
in the regime it is currently in. Protocol 022's "C keeps improving with more
budget" result was measured with 100–1600 *offline* updates on one regime; under
this round's frozen online budget the same arm does not move the metric.

### 4.3 §15 — recurrence relearning, and its confound

| recurrence | gap (C) | gap (frozen A, same transition) | attributable to C? |
|---|---:|---:|---|
| A2_recur ← A1_compute | −0.1003 | −0.0838 | no |
| C2_recur ← C1_io | −0.0843 | −0.0210 | no |
| B2_recur ← B1_memory | **+0.1536** | **+0.2121** | no |

The raw §15 rule (`gap ≥ 0.03`, at least one recurrence) is satisfied by B2 — and
by the **frozen baseline more strongly** (+0.2121). The gap is therefore a property
of the timeline (a regime's first 100 intervals are harder to score), not evidence
that fixed C had to relearn. The gate status reports the raw rule and uses the
attributable reading, which fails (R2A-06). This confound control was not requested
by the directive; it is added because without it the §15 PASS would be unfounded.

---

## 5. H3-v2 — the gradient gate is finally admissible, and it reads STOP-GI

Round 1 was `INADMISSIBLE` because the fresh residual bank has a zeroed read-out
head (8 of 20 parameter groups with exactly zero gradient). Directive §16/§17 asked
for the registered gate to be re-read on a **mature** C. It was, at all three
required checkpoints (`C_after_A1.pt`, `C_after_B1.pt`, `C_after_C1.pt`, 64 mature
independent events per regime, pruning unchanged):

| checkpoint | A\|B mean cos | A\|C mean cos | B\|C mean cos | min negative-pair fraction | degenerate groups | verdict |
|---|---:|---:|---:|---:|---:|---|
| after A1 | +0.9014 | +0.7101 | +0.9000 | 0.0198 | **0** | STOP-GI |
| after B1 | +0.9460 | +0.8256 | +0.9127 | 0.0427 | **0** | STOP-GI |
| after C1 | +0.9454 | +0.7702 | +0.8561 | 0.0681 | **0** | STOP-GI |

Thresholds unchanged from round 1: conflict requires mean cosine ≤ −0.05 **or**
negative-pair fraction ≥ 0.30. Every pair at every mature checkpoint is strongly
**co-directional**; the negative-pair fraction never exceeds 9.4%. The
`INADMISSIBLE` blocker is resolved — this is now a measured **`STOP-GI`**, i.e. the
three mechanisms do not create gradient conflict in this scenario, and the
gradient argument for dynamic experts is not supported.

---

## 6. Reproduce

```powershell
# calibration guard: A byte-identical, B/C genuinely re-calibrated
powershell -File check_ftmoe_protocol023_round2a_calibration.py.ps1

# independent verification of any calibrated stream (63 checks, non-zero on failure)
D:\Anaconda\envs\dynmoe\python.exe verify_ftmoe_protocol023_stream.py `
  artifacts\ftmoe_online\protocol_023\development_streams\dev_seed700_steps2880_calibrated --tag-suffix _calibrated

# H1-v2 data gate + marginal match
D:\Anaconda\envs\dynmoe\python.exe analyze_ftmoe_protocol023_round2a.py

# H2-v2 specialization
D:\Anaconda\envs\dynmoe\python.exe probe_ftmoe_protocol023_specialization_v2.py

# S4 (fixed probes must exist first; then both arms)
D:\Anaconda\envs\dynmoe\python.exe run_ftmoe_protocol023_s4.py --make-probes
D:\Anaconda\envs\dynmoe\python.exe run_ftmoe_protocol023_s4.py --arm A
D:\Anaconda\envs\dynmoe\python.exe run_ftmoe_protocol023_s4.py --arm C

# H3-v2 on the mature checkpoints, then every round-2A verdict
D:\Anaconda\envs\dynmoe\python.exe probe_ftmoe_protocol023_gradient.py --stream <calibrated dev> `
  --learner-state artifacts\ftmoe_online\protocol_023\round2a\checkpoints\C_after_A1.pt `
  --out artifacts\ftmoe_online\protocol_023\round2a\mature_gradient\after_A1.json
D:\Anaconda\envs\dynmoe\python.exe analyze_ftmoe_protocol023_round2a_gates.py
D:\Anaconda\envs\dynmoe\python.exe assemble_ftmoe_protocol023_round2a_status.py
```

Total compute for the whole round: four stream collections (~72 min), two S4 arms
(91 s), three gradient probes (15 s), the analyzers (seconds). Everything else was
network-free CPU work, so no result here is budget-limited.

---

## 7. What the reviewer should look at first

1. **`D_eligible = false`, and specifically why.** Three of the four failures are
   the *scientific* conditions (C learns, C forgets, gradient conflict), not
   measurement problems. Under this frozen online budget, fixed C neither learns
   the new regimes strongly enough to establish a plasticity claim nor forgets the
   old ones enough to establish a stability claim.
2. **The budget is the pivot.** S4 is the first round to hold C to a
   deployment-plausible online budget (720 updates), and the result inverts
   Protocol 022's offline capacity finding. If the next round wants to test D, the
   first question is whether a *learnable* gap can be created at all at this
   budget — that is a data/envelope question, not a mechanism question.
3. **H1-v2 is resolved**; `H1-original = FAIL` is preserved. Nothing else in the
   round changes that.
4. **The calibration trade (R2A-08)** is the one place a reviewer may want to
   overrule this round: event counts are +56–75% over A, which the directive's
   §4 target caps at 20%. It was chosen to reach 51/224 h=1 positives; the
   alternative was 2–22.
