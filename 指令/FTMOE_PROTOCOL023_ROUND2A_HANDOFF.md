# Protocol 023 — round-2A handoff (S2.5 → S4 → H3-v2/H4-v2/§15)

**For:** the reviewing model / next-round planner
**From:** the executing agent
**Date:** 2026-09-13
**Branch:** `protocol-023` · **commit:** `edb0e10` · **parent:** `d7bbc8c` (round-1 review HEAD)
**Directive executed:** `指令/FTMOE_PROTOCOL023_ROUND2A_DIRECTIVE_20260912.md`
**Round report:** `docs/FTMOE_ONLINE_PROTOCOL_023_ROUND2A.md`

---

## 0. One paragraph

Round 2A did the four things the directive asked and no more. It resolved the
follow-up definition (H1-v2 registered and **passing** while H1-original stays
**FAIL**), re-tuned B and C on data only so the third specialist has enough
positives to be measurable (H2-v2 now **3 of 3**, io-first from 12 to 31
positives), ran the first strict-prequential fixed-C comparison under a frozen
deployment-plausible online budget (arms A and C, 720 updates, integrity audited
and passing), and re-read the gradient gate on a **mature** learner — which makes
H3 admissible for the first time and returns a clean **STOP-GI**. The round's
answer is that the stability–plasticity conflict this protocol exists to test
**does not hold** in this scenario: fixed C gains only 0.2–2.0 PR-AUC points over
the frozen base in the regimes it is exposed to (registered margin +3.0), never
forgets the fixed probes, and creates no gradient conflict on mature parameters.
**`D_eligible = false`.**

---

## 1. Gate state (machine-readable: `artifacts/ftmoe_online/protocol_023/round2a/gate_status_round2a.json`)

| condition | result | the number that decides it |
|---|---|---|
| H1-v2 PASS | **true** | primary-window follow-up 185 / 405 / 428 vs ≥ 80; prevalence spread 1.42 pp |
| H2-v2 PASS | **true** | gaps +0.226 / +0.216 / +0.123; positives 185 / 124 / **31** |
| C learns ≥ 2 regimes | **false** | late-100 gain over A: **+0.0049 / +0.0020 / +0.0203** vs +0.03 |
| H3-v2 mature gradient conflict | **false** | mean cosines +0.71…+0.95, negative fractions 0.020…0.094 |
| H4-v2 actual forgetting | **false** | not evaluated (step 1 failed); diagnostic drops 0.0073 / 0.0150, both < 0.03 |
| recurrence relearning | **false** | raw §15 gap holds at B2 (+0.1536) but the *frozen baseline* shows a larger one (+0.2121) |
| online budget frozen | **true** | `amendments/online_budget.json`, benchmark recorded, unchanged after freeze |
| prequential integrity PASS | **true** | all six checks pass on both arms |

H0 remains PASS and its regression holds: the calibrated `compute_first` arm is
**byte-identical** to the registered one (checked, `check_ftmoe_protocol023_round2a_calibration.py.ps1`),
so the A arm is still Protocol 022's `cascade_v2`.

---

## 2. What changed in the data (S2.5), and the trade it makes

Measured first, tuned second. The io-first deficit is structural, not a defect:
host disk capacity is 28990.8 on every host, a cascade task's disk phase reaches
16000 with a 24000 cap, so a *host* disk fault needs two co-located disk phases —
and any shorter disk phase destroys that overlap (candidates at disk duration 2–3
collapsed to 2 disk cells). The only workable levers are the cascade probability
and the registered disk peak.

| regime | parameter | registered | calibrated |
|---|---|---|---|
| compute_first | — | — | **frozen** |
| memory_first | `cascade_task_probability` | 0.25 | **0.50** |
| io_first | `cascade_task_probability` | 0.25 | **0.45** |
| io_first | `disk_retained_peak` | 16000 | **24000** (cap unchanged) |

Result: h=1 positives 24 → **57** (io-first) and 122 → **224** (memory-first) at
the registered 1200-interval shape; prevalence A 6.80% / B 5.45% / C 6.88%
(spread 1.42 pp, was 3.75 pp).

**The trade (directive §4 targets, stated plainly):** event counts rose to 600 and
532 against A's 342 — +75% and +56% where the target is ≤ 20% — and the duration
and peak-ratio targets are not met for memory-first (0.346 and 0.638 against
0.25 / 0.30). Every candidate that kept the event count near A's produced 2–22
h=1 positives instead of 51+. If the reviewer prefers the marginal-match target
over the positive budget, that is a one-line change to
`selected_generator.json` plus a re-collection; the round chose the positives
because directive §5 makes them the precondition for a third specialist.

**Known side effect, reported not hidden:** the calibrated stream's *post-regime*
familiar phase is visited by far more in-flight regime tasks, so the
whole-familiar-phase admissibility scan now reads cpu 5200 / ram 5424 / disk 20000
against round 1's 1860 / 1400 / 9000. The mechanism-off phase F0 still measures
exactly 1860 / 1400 / 9000, so the registered onset thresholds stay admissible; the
polluted reading is recorded next to the clean one.

---

## 3. The result that matters for D

**Fixed C, strict prequential, frozen budget (720 updates over 2880 intervals,
update-every-4, batch 32, one step per opportunity):**

| regime | late-100 PR-AUC C | late-100 PR-AUC A | gain |
|---|---:|---:|---:|
| compute_first | 0.4111 | 0.4062 | +0.0049 |
| memory_first | 0.5683 | 0.5663 | +0.0020 |
| io_first | 0.5946 | 0.5743 | +0.0203 |

**0 of 3 reaches +0.03.** Directive §14 step 1 fails, so forgetting is *not
evaluated* (`forgets: null`, not `false`) and the stop condition is
**`STOP-NO-LEARN`**.

Two further measurements, both pointing the same way:

* **No forgetting on the fixed probes.** C's three probe slices never fall below
  their start (worst excursion −0.0030) and improve monotonically after F1, while
  the frozen A arm's slices never move at all. C ends uniformly above A
  (+0.175 / +0.130 / +0.163), i.e. what the residual learns is *not* regime-specific.
* **No gradient conflict on mature parameters.** At `C_after_A1/B1/C1` the
  degenerate-group list is empty (round 1's INADMISSIBLE blocker is resolved) and
  the registered rule reads no conflict at every checkpoint.

The sharpest way to put it: under Protocol 022's *offline* budget C kept improving
with more updates. Under this round's frozen *online* budget the same arm barely
moves. That is the first measurement in this protocol family that argues *against*
needing a dynamic expert, and it is budget-limited rather than mechanism-limited.

---

## 4. Decisions the reviewer may want to make

1. **Is the calibration trade acceptable?** (+56–75% event count for +5× the h=1
   positives.) If not, the alternative is a marginal-matched but positive-starved
   C, which caps the specialist claim at 2 of 3.
2. **Is the frozen online budget the right one to test D under?** The whole
   negative result turns on it. A budget sweep (e.g. 2× / 4× updates) would say
   whether the plateau is a budget artefact — this is *not* C-budget as round 1
   defined it (that was an offline diagnostic), and the directive forbids
   C-budget this round, so it is a next-round question.
3. **Is the §15 confound control the right standard?** The round added a baseline
   comparison the directive does not require (the frozen arm shows a *larger*
   relearning gap on every transition) and used it to fail the condition. A
   reviewer who wants the literal §15 rule alone would read that condition as PASS.
4. **io-first clears the reliability bar by exactly one positive (31 vs 30).** Usable,
   but confirmation seeds 701–703 are untouched and would be the honest next check.
5. **Event-count equality vs prevalence equality.** Round 2A keeps prevalence
   nearly identical across regimes and lets event counts diverge. If the reviewer
   considers event count the more important marginal, the regimes need a different
   response-duration law, which is a new family (`cascade_v4`), not a calibration.

---

## 5. Where things are

```
artifacts/ftmoe_online/protocol_023/
  development_streams/
    dev_seed700_steps2880                     round 1, preserved
    single_{compute,memory,io}_first_seed700_steps1200        round 1, preserved
    dev_seed700_steps2880_calibrated          round 2A (verified, 63 checks)
    single_*_seed700_steps1200_calibrated     round 2A (verified, 63 checks)
  round2a/
    amendments/{h1_v2_definition,online_budget}.json
    data_calibration/{candidate_table,selected_generator,marginal_match_v2,data_gate_v2}.json
    diagnostics/{disk_onset_deficit,positive_budget,h1_positive_budget_by_window,envelope_landing}.json
    specialization_v2/cross_regime_probe_v2.{json,md}
    probe_slices/{probe_*.npz,probe_registry.json}
    fixed_c_prequential/
      ARTIFACT_INDEX.json                     where each per-arm file is
      arm_{A,C}/{predictions.npz,settlements.jsonl,update_log.jsonl,phase_metrics.json,...}
      probe_matrix.{json,md}                  the §13 checkpoint × probe table
      forgetting_v2.json                      H4-v2 (§14)
      recurrence_metrics.json                 §15 + the baseline attribution
    mature_gradient/{after_A1,after_B1,after_C1,summary}.json
    checkpoints/C_after_{A1,B1,C1,F1,A2,C2,B2}.pt
    gate_status_round2a.json                  D_eligible = false
    problem_log_round2a.jsonl                 R2A-01 … R2A-09
  (round 1's own artifacts, including H1-original FAIL and gate_status.json, are untouched)
```

Two bugs were found the hard way and are worth knowing about if the calibration
is re-run: the generator's `regime_params` overlays `_REGISTERED_COMMON` onto every
regime (so writing a shared value only into `REGIMES_V3` is silently discarded),
and the collector's phase switch resets the probability from the phase table (so a
calibrated probability must be written into the phase table itself). Both produced
byte-identical "calibrated" streams before they were fixed; the calibration guard
script exists to make that failure mode impossible to miss again.
