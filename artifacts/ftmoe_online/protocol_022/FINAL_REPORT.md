# Protocol 022 — Round-1 Final Report (S0–S4)

**Date:** 2026-09-10 · **Branch:** `protocol-022` · **Parent:** `f0abb5580d92730eecdc1db4326382c81a5f1e31` (equals the plan's registered parent; no force reset)
**Scope:** first-round submission only, i.e. S0–S4 per plan §25. C (S5–S7) and D (S8–S12) were **not** run.

Structure follows plan §24. Gates: **H0 PASS · H1 (U4-v2) PASS · H2 PASS (with two registered reservations)**. No STOP condition was hit.

---

## 1. Git commit / source hashes

| Item | Value |
|---|---|
| Branch | `protocol-022` (created from `protocol-021`) |
| HEAD at bootstrap | `f0abb5580d92730eecdc1db4326382c81a5f1e31` — **equals** the registered parent SHA |
| Force reset / rebase | **none** |
| Unified online start checkpoint | `artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt`, SHA256 `10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b` (verified, unmodified) |
| Cascade generator | `simulator/workload/BitbrainWorkloadProtocol022.py`, SHA256 `d34ae24ac1f34a1b38fb37ec30da644cad62ebb5358238ba833f29856bded307` |
| Source snapshot | `artifacts/ftmoe_online/protocol_022/source_sha256_initial.json` (278 files) |
| Protocol record | `artifacts/ftmoe_online/protocol_022/protocol.json` |
| Bootstrap state | `artifacts/ftmoe_online/protocol_022/bootstrap_state.json` |
| Registry | `artifacts/ftmoe_online/protocol_022/unseen_registry/cascade_v2.json`, `registry_sha256` `f1d9aebae836d881a710b87a5212f3d32765ab9d589215000b9917a21523c6be` |

Every formal run carries a `run_provenance.json` with command, git SHA, Python/torch versions, seed, start/end, **real child exit code**, elapsed, and source hashes. The three collections returned `[0, 0, 0]` from `subprocess.CompletedProcess.returncode` — no PowerShell-derived exit code anywhere (the P20 defect this plan forbids).

## 2. Historical facts accepted

- **P20 R1**: the fixed small residual correction reduced legacy C's drift degradation but did **not** stably beat frozen A; dev500/dev501 were consumed by both method selection and development evaluation.
- **P21**: `STOP-A` **stands** — the pre-registered U4 gate failed (0.549 < 0.582, 0.443 < 0.521, 0.428 < 0.764), diagnosed as an *instrument-scope* failure: host aggregates cannot separate a per-task cascade from co-residency co-movement. P21's learnability PASS (AP 0.678 vs prevalence 0.078) carried an unresolved persistence/confounding risk (time-shuffled AP 0.242).
- **P21's `STOP-A` was not rewritten.** Protocol 022 is a new registration, not a reinterpretation.

## 3. Data Gate

Unchanged from P21 so the two rounds stay comparable (plan §5.2).

| Candidate | prevalence | cascade positive host-steps | independent cascade events | deployment rej. | migration rej. | worst-event share | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| p=0.15 | 0.0457 | 859 | 304 | 0.0889 | 0.2255 | 0.0159 | **PASS** |
| p=0.25 | 0.0683 | 1308 | 473 | 0.1382 | 0.3519 | 0.0107 | **PASS** |
| p=0.35 | 0.0984 | 1876 | 680 | 0.1786 | **0.4491** | 0.0090 | **FAIL** |

p=0.35 fails **exactly as pre-registered**: migration rejection 44.91% > the 40% ceiling. The plan states in advance that this candidate is a high-load *stress* candidate, that a repeat failure keeps its FAIL status, and that it must not be selected merely for producing more events. It did produce the most events (574 task onsets), and it was still not selected — gate outcome dominates the ordering.

## 4. U1 / U2 / U3 / U4-v2 reported separately

| Level | Status | Evidence |
|---|---|---|
| **U0** new seed / replay | true (not sufficient) | mechanism seed 22022, replay seed 600 |
| **U1** source-disjoint vs P20 S6 | **PASS** | cohort `online` (94 VMs) ∩ P20 S6 train (277) = ∅, ∩ dev (86) = ∅, re-verified from `vm_split.json`; exact 12-step window overlap with P20 S6 **and** P20 dev streams = **0** |
| **U1 limitation** | **UNVERIFIED** | source-unseen relative to the *full* historical chain — P014's upstream VM selection is registered nowhere. Reported, never hidden. |
| **U2** parameter-combo unseen | PASS | the envelope does not exist anywhere in the audited offline chain |
| **U3** generator mechanism unseen | PASS | P21 `exposure_ledger.json`: 1003 files of the 014/019/020 chain searched; `cascade` / `cpu_to_ram` / `delayed` / `mechanism_seed` / `hysteresis` / `feedback` = 0 matches (reused unmodified) |
| **U4-v2** task-level distributional novelty | **PASS** at p=0.15 and p=0.25 | see §5 |

**Primary claim is mechanism-level U3 + task-level U4-v2 only, never absolute source-unseen** (plan §7.2).

## 5. Audit correctness (Gate H0) and the U4-v2 instrument

Gate H0 **PASS**: `test_ftmoe_protocol022_audit.py`, 31 tests, all green — T-AUDIT-01 (same-host mixed isolation, including a test showing the old host-scale instrument invents a RAM response no single task produced), T-AUDIT-02 (migration identity), T-AUDIT-03 (slot reuse, and rejection of a duplicated `creation_id` in one interval), T-AUDIT-04 (future-metadata isolation), T-AUDIT-05 (P20 regression at probability 0), T-AUDIT-06 (determinism), plus registered-physics, registered-onset and offline-threshold guards.

**Instrument.** Unit = `creation_id` + CPU onset event. Onset = the first observed interval at which the task's own CPU reaches τ = 2600, given the previous observed interval was strictly below it; a sustained burst counts once. `RAM_response = max RAM_task[t0+4:t0+14] − baseline`, `Disk_response = max Disk_task[t0+8:t0+18] − baseline`.

Two instrument defects were found and fixed before any gate result was recorded (full detail in the problem log):

- **P22-01** — the plan's `CPU(t) − median(CPU[t−4:t]) ≥ τ` form is *structurally blind* to the registered envelope, because the envelope is admitted at the familiar level and bursts at age 1, so every interior burst interval's median window is already the burst. Measured on the real first candidate: `max delta_cpu = −4400`, zero onsets — while the task-level CPU genuinely reaches 4400–5200. τ was **not** changed; the rule was made baseline-free because the physical claim is identical and the old form is structurally inapplicable.
- **P22-02** — every offline per-task CPU demand is clipped at **1860.0** (per-corpus maxima all 1860.0), so τ = 2600 has **zero** offline task-level onsets and the plan's percentile rule is *undefined* on that scale. Both scales are reported instead of redefining the gate: registered (τ=2600, zero offline onsets = mechanism-level evidence) and calibrated (τ=500, fixed from offline data alone before any candidate was read, giving a usable reference).

**Also fixed (P22-06):** the first M0/M1 controls were no-ops — every cascade task shares one envelope, so swapping whole curves between events produced literally identical distributions (control p90 4500.0 = candidate 4500.0). Controls now apply a per-event circular lag shift to each event's own curve, preserving marginal amplitude, duration distribution, event count, cohort and arrival process while destroying lag alignment.

## 6. Learnability beyond persistence (Gate H2)

Task O (onset) is the gate; Task S (state) is kept for P21 comparability.

| | Task S (state) | **Task O (onset, h=1, gate)** | Task O (onset, h=4) |
|---|---:|---:|---:|
| fit AP | 0.6811 | 0.2936 | 0.2275 |
| dev AP | 0.6189 | 0.2427 | 0.1983 |
| **forward test AP** | 0.6208 | **0.2392** | 0.2025 |
| test ROC-AUC | 0.9327 | **0.9571** | 0.6766 |
| test prevalence | 0.0641 | 0.0221 | 0.0832 |
| forward positives | 368 | **120** | 448 |

Simple baselines on the **onset** target (identical rows and metric): B0 random 0.0221, B1 host prior 0.0414, **B2 persistence 0.0221**, B3 current-ratio-only 0.0201, B4 past-fault-only 0.0221, B5 slopes-only 0.0188. **Persistence scores exactly chance on the onset target** — which is the whole point of separating it: the P21 explanation ("currently faulted → still faulted") is worth nothing here, and the full probe still reaches 0.2392, i.e. 10.8× prevalence and +0.217 over the best simple baseline.

Per-resource onset (h=1): CPU AP 0.2392 (120 positives), RAM AP 0.6003 (14), Disk AP 0.2922 (4); ROC-AUC 0.957 / 0.978 / 0.998. Not driven by a single resource, but the RAM/Disk onset sets are very sparse.

**Gate H2 = PASS** on all five registered checks (`ap_at_least_floor`, `roc_auc_at_least_0.75`, `beats_best_simple_baseline_by_0.05`, `observed_ap_above_all_null_p99`, `positives_at_least_50`).

**Instrument soundness (P22-07).** The first probe run reported onset AP 0.0259 → a clean-looking **STOP-B2 that was false**. A lagged feature was undefined on the first interval of every stream; 48 NaNs poisoned the standardizer, every weight became NaN, and the probe silently degenerated into a constant predictor (in-sample AP 0.0652 at prevalence 0.0667). It was caught only because the plan's own soundness requirement was applied: a probe that cannot detect a signal that is definitely present cannot certify its absence. With the same data and features, fixing it moved in-sample AP 0.0652 → 0.6811, planted-signal AP 0.1209 → 1.0, leaky control 0.0826 → 0.8108, state test AP 0.0826 → 0.6208, and onset test AP 0.0259 → 0.2392.

Permutation controls: 6 families × 100 draws (global, within_host, block8, block12, event, **host**). The `host` family was added specifically to answer the inherited P21 question and permutes which host receives which label stream.

| Family | mean | p95 | p99 | max | observed percentile |
|---|---:|---:|---:|---:|---:|
| global | 0.0433 | 0.1385 | **0.2153** | **0.2562** | 99 |
| within_host | 0.0643 | 0.1420 | 0.1605 | 0.1623 | 100 |
| block8 | 0.0425 | 0.1233 | 0.1683 | 0.1915 | 100 |
| block12 | 0.0374 | 0.1032 | 0.1940 | 0.1950 | 100 |
| event | 0.0398 | 0.1051 | 0.1353 | 0.1381 | 100 |
| **host** | 0.0418 | 0.1076 | 0.1810 | 0.2141 | 100 |

## 7. Fixed C capacity diagnostic

**Not run.** Out of round-1 scope (plan §25).

## 8. Strict A vs C

**Not run.** Out of round-1 scope.

## 9. Whether D was scientifically eligible

**Not evaluated, therefore not eligible.** H3 (C stably beats A on late-unseen) and H4 (C learns but plateaus with irreducible residual error) were never assessed, so the dynamic-expert path remains blocked exactly as the plan requires.

## 10. Additive continuity

**Not implemented.** Out of round-1 scope. The P20 R0-B Top-k ramp discontinuity remains unfixed and must not be reused (plan §12.1).

## 11. C vs D vs C-wide vs C-budget

**Not run.**

## 12. Familiar / anchor retention

**Not measured** — that requires a model run. The data-layer analogue is reported instead: the familiar profile is unchanged from the P20 baseline phase (capacity scales 1.0 / 1.0 / 0.9, adapter `ram_upper=1400`), and T-AUDIT-05 asserts that a non-cascading task's demand transform is unchanged from the frozen familiar generator under an identical seeded arrival stream.

## 13. Runtime / memory / update budget

| Run | Steps | Elapsed | Real exit code |
|---|---:|---:|---:|
| p=0.15 collection | 1200 (+1 guard) | 487.9 s | 0 |
| p=0.25 collection | 1200 (+1 guard) | 496.2 s | 0 |
| p=0.35 collection | 1200 (+1 guard) | 498.8 s | 0 |

Single process throughout, 3 torch threads, 1 interop thread, 3.0 GiB RAM guard, 20 GiB disk guard, BelowNormal priority. No model was loaded in any P22 run: **0 optimizer updates, 0 backward passes, 0 trainable parameters touched**. The probe is a closed-form Newton/IRLS logistic fit on 25 features, not a neural model, and exists only to test predictability.

## 14. Recurrence if executed

**Not executed** (S12 is gated behind S11).

## 15. Every failed gate

| Gate / condition | Outcome | Notes |
|---|---|---|
| H0 audit correctness | **PASS** | 31/31 |
| H1 U4-v2, p=0.15 | **PASS** | |
| H1 U4-v2, p=0.25 | **PASS** | selected |
| H1 U4-v2, p=0.35 | **FAIL** | Pilot Data Gate: migration rejection 44.91% > 40%. Pre-registered; not overridden. |
| H2 learnability | **PASS** | with reservations below |
| **H2 strict bound** | **FAIL** | observed 0.2392 vs best single null draw 0.2562 — passes the registered p99 criterion, not the max-over-all-draws bound |
| **H2 lag-specificity** | **FAIL** | registered lags 0.2392; wrong lags within ±0.014 → learnability may **not** be attributed to the cascade temporal law (plan §8.4) |
| h=4 onset AP floor | **FAIL** | 0.2025 < 0.2080; secondary evidence by design |
| STOP-0 / A2 / B2 / CAP / C2 / D0 / D1 / R | **not hit** | |

**Nothing was silently dropped.** The two H2 reservations and the h=4 shortfall are stated in the same table as the passes.

## 16. What can and cannot be claimed

**Can claim**

1. The registered `cascade_v2` regime is **distributionally novel at task/event scale** relative to every auditable offline corpus (U4-v2 PASS at p=0.15 and p=0.25): candidate RAM response p90 **4500.0** vs offline task-level p97.5 **505.4** (8.9×) and offline max **2043.5**; exceedance probability 1.00 with 95% CI [1.00, 1.00]; registered-lag-4 alignment 0.754 vs 0.37–0.57 for M0/M1/M2 (permutation p = 0.0005, the resolution floor of 2 000 permutations).
2. The registered physical envelope is **unreachable offline by construction**: 0 offline task-level onsets at τ=2600 across 2 074 calibrated-scale onsets in three corpora, because every offline per-task CPU is clipped at 1860.
3. A CPU-fault **onset is predictable from pre-onset information well beyond persistence** (H2 PASS): onset AP 0.2392 at prevalence 0.0221, ROC-AUC 0.9571, 120 forward-test positives, versus persistence / current-ratio / past-fault at chance (0.0221 / 0.0201 / 0.0221).
4. The task-level instrument itself is correct and was demonstrated so before any gate was read: 31 tests including the same-host isolation case that defeated the P21 instrument.

**Cannot claim**

1. **That the cascade temporal law is what was learned.** The lag ablation shows no advantage for the registered lags (P22-11). What is demonstrated is a real, causally available *pre-onset CPU rise*, not the registered CPU→+4 RAM→+8 Disk structure.
2. **A comfortable H2 margin.** The registered p99 criterion passes; the stricter max-over-all-draws bound does not. One development stream and one replay seed is not confirmation (plan §17).
3. **Absolute source-unseen**, relative to the full historical chain (P014 upstream VM selection unregistered) — UNVERIFIED.
4. **Anything about fixed online C or dynamic experts.** S5–S12 were not run; D remains blocked.
5. **That the regime describes real industrial faults.** It is a *controlled unseen temporal resource-demand regime*; the two-stage CPU phase exists because `getPlacementPossible()` evaluates demand at the admission interval (construct-validity limitation, plan §19).

---

## Recommended next round (S5–S7), with the round-1 reservations built in

1. Add **≥2 further replay seeds** (and preferably the registered `mechanism_seed_confirm = 22023`) before any C conclusion is treated as final — required by P22-08.
2. The S5 capacity diagnostic **must include a variant forced to use the registered lag context**, otherwise "C has insufficient capacity" and "C never learned the cascade structure" stay indistinguishable — required by P22-11.
3. Keep the §9.3 gradient-health record and the strict prequential ordering of §10.1.
4. Do not start D until H3 and H4 are evaluated and satisfied, and do not reuse the discontinuous Top-k ramp deployment.
5. Update the entry-point documents (`PROJECT_CONTEXT_LATEST.md`, `README.md`, `docs/README.md`). **Deliberately not done in this round**: those files describe protocol 020 as the current stage and are maintained by the parent/review flow, and silently promoting protocol 022 to "current" from a feature branch would create exactly the document-versus-artifact contradiction the 2026-09-10 correction had to repair.
6. Commit the P22 sources (and the still-untracked P20 R0/R1 sources) so the evidence chain stops depending on untracked working-tree files.
