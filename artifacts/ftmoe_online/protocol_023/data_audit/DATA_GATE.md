# Protocol 023 -- S2 Data Gate (plan §9)

- analyzer: `analyze_ftmoe_protocol023_s2.py` (stage S2)
- generated_at: 2026-09-12T12:54:01+00:00
- streams_root: `F:\PreGANPlus-master\artifacts\ftmoe_online\protocol_023\development_streams`
- plan: `指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md §9 (建议 Data Gate)`
- registered gate: anomaly prevalence 3%-12% / deployment rejection <= 25% / migration rejection <= 40% / independent fault events >= 80 per regime / valid onset-follow-up >= 50 per regime / worst event share < 10% / regime prevalence difference <= 4 percentage points

## Verdict

- data gate (`marginal_match_report` + the analyzer's own re-derivation of the same eight checks): **FAIL**
- onset thresholds admissible: **PASS**
- stream integrity: **PASS**
- **overall: FAIL** (failed_on: data_gate)
- analyzer checks reproduce the instrument's checks: **PASS**
- plan §9 table matches the instrument's registered table: **PASS**
- development stream (multi-regime context): **read** (dev_seed700_steps2880)

## Registered thresholds vs the measurement (per regime)

| quantity | measured | threshold | verdict |
|---|---|---|---|
| A prevalence | 6.80% | 3%-12% | PASS |
| A independent fault events | 410 | >= 80 | PASS |
| A valid onset/follow-up | 24 | >= 50 | FAIL |
| A deployment rejection | 13.20% | <= 25% | PASS |
| A migration rejection | 36.29% | <= 40% | PASS |
| A worst event share | 1.57% | < 10% | PASS |
| B prevalence | 3.05% | 3%-12% | PASS |
| B independent fault events | 260 | >= 80 | PASS |
| B valid onset/follow-up | 36 | >= 50 | FAIL |
| B deployment rejection | 8.45% | <= 25% | PASS |
| B migration rejection | 15.43% | <= 40% | PASS |
| B worst event share | 1.56% | < 10% | PASS |
| C prevalence | 3.79% | 3%-12% | PASS |
| C independent fault events | 271 | >= 80 | PASS |
| C valid onset/follow-up | 31 | >= 50 | FAIL |
| C deployment rejection | 9.32% | <= 25% | PASS |
| C migration rejection | 22.17% | <= 40% | PASS |
| C worst event share | 1.41% | < 10% | PASS |

- regime prevalence difference (spread): 3.75% (threshold <= 4 percentage points, PASS)

## Event structure and matched marginals

| regime | onset res. | tau | onsets | valid follow-up (full window, gated) | usable follow-up (P22 definition, not gated) | mean duration | median duration | mean peak ratio | median peak ratio | cohort tasks |
|---|---|---|---|---|---|---|---|---|---|---|
| A | cpu | 2600.0000 | 342 | 24 | 185 | 2.6433 | 1.0000 | 53.3396 | 17.4980 | 342 |
| B | ram | 2810.0000 | 297 | 36 | 200 | 3.9024 | 2.0000 | 25.8052 | 8.5000 | 342 |
| C | disk | 11600.0000 | 297 | 31 | 235 | 2.2997 | 2.0000 | 57.7981 | 11.9784 | 342 |

- `valid follow-up` is the gated number (the whole registered response window observed, plan §9 >= 50); `usable follow-up` is the P22 U4-v2 definition (at least one observed interval inside a window) and is reported only -- a large gap between the two means most containers die before their registered response window completes.
- cohort observed life (intervals per task, p50 / max vs the follow-up horizon): A 6.0 / 27.0 vs 18; B 5.0 / 29.0 vs 16; C 6.0 / 30.0 vs 16

_Marginal matching per regime pair (instrument-local tolerances -- context, not the registered gate; `joint differs` is REQUIRED to be true):_

| pair | prevalence diff | ok | event count rel diff | ok | duration rel diff | ok | peak ratio rel diff | ok | marginals matched | joint structure differs |
|---|---|---|---|---|---|---|---|---|---|---|
| A|B | 3.75% | PASS | 0.1316 | FAIL | 0.3226 | FAIL | 0.5142 | FAIL | FAIL | PASS |
| A|C | 3.01% | PASS | 0.1316 | FAIL | 0.1300 | FAIL | 0.3154 | FAIL | FAIL | PASS |
| B|C | 0.74% | PASS | 0.0000 | PASS | 0.4107 | FAIL | 0.2904 | FAIL | FAIL | PASS |

## Onset threshold admissibility (measured, never retuned)

| regime | resource | tau | familiar per-task max (all familiar rows) | familiar per-task max (envelope-free tasks) | margin above max | registered floor | margin below floor | admissible | threshold status |
|---|---|---|---|---|---|---|---|---|---|
| A | cpu | 2600.0000 | 1860.0000 | 1860.0000 | 740.0000 | 4400.0000 | 1800.0000 | PASS | registered |
| B | ram | 2810.0000 | 1400.0000 | 1400.0000 | 1410.0000 | 4500.0000 | 1690.0000 | PASS | PROVISIONAL |
| C | disk | 11600.0000 | 9000.0000 | 9000.0000 | 2600.0000 | 16000.0000 | 4400.0000 | PASS | PROVISIONAL |

- the conservative measurement (every familiar-phase row) decides admissibility; the envelope-free measurement separates a genuine familiar maximum from a cascade task whose burst spilled into a later familiar phase.
- `n/a` means NOT MEASURABLE (no familiar row carries that resource); an unmeasurable threshold is reported inadmissible, not assumed safe.

## Streams read (recomputation provenance)

| role | tag | discovery | steps | regimes present | stream sha256 (16) |
|---|---|---|---|---|---|
| A | single_compute_first_seed700_steps1200 | registered_tag | 1200 | A | 2348c6289325d310 |
| B | single_memory_first_seed700_steps1200 | registered_tag | 1200 | B | 322210f124854184 |
| C | single_io_first_seed700_steps1200 | registered_tag | 1200 | C | 1d513d577f53f077 |
| dev | dev_seed700_steps2880 | registered_tag | 2880 | A,B,C | 91561298e77e212f |

- every gate number is recomputed from `stream.npz` and `task_timeline.npz` with `ftmoe_protocol023_core`; the collector's `unseen_data_audit.json` is only ever compared against the recomputation `(agrees)`.
- the development stream above is context: it carries all three regimes and the recurrence, but each regime's gate block is measured on its own single-regime cohort.

## Stop conditions

- STOP-DATA: the plan §9 Data Gate did not pass (A_valid_onset_followup_enough, B_valid_onset_followup_enough, C_valid_onset_followup_enough); the scenes are not matched, so no method comparison is admissible

## Warnings

- WARN-FOLLOWUP-DEFINITION: regime A has 24 onset(s) with a FULL follow-up window and 185 with a usable follow-up (at least one observed interval inside a registered response window; the P22 U4-v2 definition).  The gate applies the >=50 threshold to the full-window count, so the check fails; under the P22-compatible definition it would pass.  The analyzer does not switch definitions after seeing the number.
- WARN-FOLLOWUP-DEFINITION: regime B has 36 onset(s) with a FULL follow-up window and 200 with a usable follow-up (at least one observed interval inside a registered response window; the P22 U4-v2 definition).  The gate applies the >=50 threshold to the full-window count, so the check fails; under the P22-compatible definition it would pass.  The analyzer does not switch definitions after seeing the number.
- WARN-FOLLOWUP-DEFINITION: regime C has 31 onset(s) with a FULL follow-up window and 235 with a usable follow-up (at least one observed interval inside a registered response window; the P22 U4-v2 definition).  The gate applies the >=50 threshold to the full-window count, so the check fails; under the P22-compatible definition it would pass.  The analyzer does not switch definitions after seeing the number.

## Definitions

- **admissibility**: a registered onset threshold is admissible only if it is strictly above the measured familiar per-task maximum of its resource; a failure is reported with its negative margin and never retuned
- **cohort**: onsets are scanned with task_regime_map, so a regime-A envelope whose RAM/Disk responses cross the B/C thresholds is never counted as a B/C event; the unmapped scan is reported as the cross-hit confound count
- **gate**: anomaly prevalence 3%-12% / deployment rejection <= 25% / migration rejection <= 40% / independent fault events >= 80 per regime / valid onset-follow-up >= 50 per regime / worst event share < 10% / regime prevalence difference <= 4 percentage points
- **gate_source**: each regime's gate block is measured on its own single-regime stream (the registered clean per-regime cohort); the development stream is reported as context and never enters the gate
- **independent_fault_events**: per-host contiguous positive-label runs inside the regime's phase window (the registered P22 segmentation rule, reimplemented here)
- **marginal_matching**: core23.marginal_match_report over the eight registered numbers; core23.regime_contrast_report is reported next to it as context -- its tolerances are instrument-local and are not the registered gate
- **mean_duration**: core23.response_durations over the regime's onset cohort: intervals inside the registered response windows whose deviation from the pre-onset baseline is > 0, longest of the two response resources
- **peak_ratio**: core23.peak_ratio_summary over the same cohort; the registered matched marginal uses the MEDIAN peak ratio (robust to one large excursion) and the mean is reported next to it
- **provisional_thresholds**: RAM 2810.0 and disk 11600.0 are PROVISIONAL
- **recomputation**: every gate quantity is recomputed from stream.npz + task_timeline.npz with ftmoe_protocol023_core; the collector's unseen_data_audit.json is only ever compared against the recomputation
- **valid_onset_followup**: onsets whose task observes every interval of [t0, t0 + max registered response window)) -- the plan §9 'valid onset/follow-up' the ≥50 threshold is applied to; the P22-compatible 'usable follow-up' count (at least one observed interval inside a window) is reported next to it and is NOT gated
- **worst_event_share**: positive host-steps attributed to the single worst registered cascade event, over the window's positives (the collector's definition); the label-free worst-run share is reported next to it
