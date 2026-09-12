"""Protocol 023 P23-S0 — mandatory multi-regime audit-correctness tests.

The P22 instrument (``ftmoe_protocol022_core``) had to pass six audit tests
(T-AUDIT-01..06) before any formal generation.  Protocol 023 registers three
cascading regimes that differ ONLY in resource order and lag, so the same
correctness claims must hold *per regime* and, additionally, the instrument must
prove that it did not blur the three regimes together:

    T023-01  same-host isolation: two tasks on one host, the host aggregate
             co-moves, but only the task with the burst yields an onset -- and
             each task's onset appears under its OWN regime only
    T023-02  migration identity: a task that moves host keeps ONE trajectory
    T023-03  slot reuse: a freed slot reused by a new container is never spliced
    T023-04  future-metadata isolation: perturbing future rows / future metadata
             changes no onset event and no response value
    T023-05  regime isolation: A/B/C each produce exactly one onset on their own
             onset resource and none on the other two
    T023-06  registered response windows: a response planted at the registered
             lag is detected; a response at a wrong lag is not attributed to it
    T023-07  marginal matching (plan §9): the gate passes on matched inputs and
             fails on a 5-point prevalence spread or on < 80 fault events
    T023-08  determinism and NaN policy: identical inputs give identical rows,
             controls are deterministic for a fixed seed, and a single NaN in a
             feature column RAISES instead of silently producing a constant
             predictor (the P22 probe burn)

Every timeline is built directly in the test (never by the simulator), so the
claims are about the instrument, not about the generator.

Run from the repository root:
    python -m unittest test_ftmoe_protocol023_core -v
"""
import copy
import unittest

import numpy as np

import ftmoe_protocol022_core as p22
import ftmoe_protocol023_core as p23

# --------------------------------------------------------------------------
# synthetic timeline helpers (same construction as the P22 test suite)
# --------------------------------------------------------------------------
# Familiar per-task levels (P22 registration) used as the flat background.
FAMILIAR = dict(p23.FAMILIAR_CLIP)

# Registered burst levels of the envelope the three regimes reuse (plan §5-§7:
# CPU burst floor 4400, RAM target floor 4500, disk retained peak 16000).  The
# probe timelines only need to cross the regime's registered onset threshold.
REGIME_BURST = {"A": 4700.0, "B": 4800.0, "C": 18000.0}
# Per-resource level used when a probe plants the *response* phase of a
# registered cascade (must sit above that resource's familiar level).
RESPONSE_LEVEL = {"cpu": 4700.0, "ram": 4700.0, "disk": 18000.0}

TAU = {spec["onset_resource"]: spec["onset_tau"] for spec in p23.REGIMES.values()}
WINDOWS = {rid: spec["response_windows"] for rid, spec in p23.REGIMES.items()}
SEQUENCE_LAG = {rid: {name: lag for name, lag in spec["sequence"]}
                for rid, spec in p23.REGIMES.items()}


def rows_from_tasks(tasks, n_hosts=2):
    """Build parallel observation arrays from per-task trajectories.

    ``tasks`` maps creation_id -> {"slot", "start", "cpu", "ram", "disk",
    "hosts"} where ``hosts`` is one host id per interval.
    """
    time, slot, cid, host, dem, event = [], [], [], [], [], []
    for creation, spec in sorted(tasks.items()):
        for k, h in enumerate(spec["hosts"]):
            time.append(spec["start"] + k)
            slot.append(spec["slot"])
            cid.append(creation)
            host.append(h)
            dem.append([spec["cpu"][k], spec["ram"][k], 1.0, 1.0,
                        spec["disk"][k], 1.0, 1.0])
            event.append(spec.get("event_id", -1))
    return (np.asarray(time), np.asarray(slot), np.asarray(cid),
            np.asarray(dem, dtype=np.float64), np.asarray(host),
            np.asarray(event))


def task_spec(values, slot=0, start=0, hosts=None, creation=0):
    steps = len(values["cpu"])
    return {creation: {
        "slot": slot, "start": start,
        "cpu": np.asarray(values["cpu"], dtype=np.float64),
        "ram": np.asarray(values["ram"], dtype=np.float64),
        "disk": np.asarray(values["disk"], dtype=np.float64),
        "hosts": (np.zeros(steps, dtype=int) if hosts is None
                  else np.asarray(hosts, dtype=int))}}


def timeline_of(values, creation=0, slot=0, start=0, hosts=None, n_hosts=2):
    return p23.build_task_timeline(
        *rows_from_tasks(task_spec(values, slot=slot, start=start, hosts=hosts,
                                   creation=creation), n_hosts=n_hosts))


def regime_series(regime_id, steps=30, spike=None, offset=0, amplitude=2000.0,
                  cascade=False):
    """Flat familiar background + one registered burst on the onset resource.

    ``spike``/``offset`` (if given) plant one extra excursion on a response
    resource, ``offset`` intervals after the onset (which sits at task index 1).
    ``cascade=True`` additionally plants the registered responses of that
    regime, i.e. a full cascade rather than an isolated trigger.
    """
    spec = p23.REGIMES[regime_id]
    onset = spec["onset_resource"]
    values = {name: np.full(steps, FAMILIAR[name], dtype=np.float64)
              for name in p23.RESOURCES}
    values[onset][1:4] = REGIME_BURST[regime_id]
    if cascade:
        for name, lag in spec["sequence"]:
            if name == onset:
                continue
            values[name][1 + lag:1 + lag + 6] = RESPONSE_LEVEL[name]
    if spike is not None:
        index = 1 + int(offset)
        values[spike][index] = FAMILIAR[spike] + float(amplitude)
    return values


def first_event(timeline, regime_id, **kwargs):
    events = p23.onset_events_for_regime(timeline, regime_id, **kwargs)
    if len(events) != 1:
        raise AssertionError("expected exactly one %s onset, got %d"
                             % (regime_id, len(events)))
    return events[0]


# --------------------------------------------------------------------------
# T023-00 — the P22 API must stay usable, and regime A must stay P22
# --------------------------------------------------------------------------
class P22ApiSurfaceTests(unittest.TestCase):
    """The reader can keep using the P22 instrument for the A regime."""

    RE_EXPORTED = ("build_task_timeline", "timeline_integrity",
                   "migration_continuity", "delta_cpu", "onset_positions",
                   "onset_events", "response_curves", "event_responses",
                   "response_summary", "response_summary_from_values",
                   "alignment_rate", "control_m0_pairs", "control_m1_order",
                   "control_m2_circular", "bootstrap_ci",
                   "exceedance_probability", "permutation_pvalue",
                   "pooled_percentile", "timeline_sha256",
                   "forbidden_token_hits")

    def test_p22_callables_are_reexported_unchanged(self):
        for name in self.RE_EXPORTED:
            self.assertIs(getattr(p23, name), getattr(p22, name),
                          "%s must be the P22 function itself, not a copy" % name)
        self.assertIsNotNone(p23.FAMILIAR_TASK_LEVEL)
        self.assertEqual(p23.ONSET_TAU_CPU, p22.ONSET_TAU_CPU)
        self.assertEqual(p23.ONSET_BASELINE_LAG, p22.ONSET_BASELINE_LAG)

    def test_regime_a_is_inherited_verbatim_from_p22(self):
        regime_a = p23.regime("A")
        self.assertEqual(regime_a["onset_resource"], "cpu")
        self.assertEqual(regime_a["onset_tau"], p22.ONSET_TAU_CPU)
        self.assertEqual(regime_a["sequence"], [["cpu", 0], ["ram", 4],
                                                ["disk", 8]])
        self.assertEqual(list(regime_a["response_windows"]["ram"]),
                         list(p22.RAM_RESPONSE_WINDOW))
        self.assertEqual(list(regime_a["response_windows"]["disk"]),
                         list(p22.DISK_RESPONSE_WINDOW))
        self.assertEqual(regime_a["response_windows"]["ram"][0],
                         p22.REGISTERED_LAG_RAM)
        self.assertEqual(regime_a["response_windows"]["disk"][0],
                         p22.REGISTERED_LAG_DISK)
        self.assertEqual(regime_a["name"], "compute_first")

    def test_regime_a_multi_instrument_reproduces_the_p22_rows(self):
        """Physically unchanged: same onsets, baselines and responses."""
        timeline = timeline_of(regime_series("A", cascade=True))
        p22_events = p22.onset_events(timeline)
        multi_events = p23.onset_events_for_regime(timeline, "A")
        self.assertEqual(len(p22_events), 1)
        self.assertEqual(len(multi_events), 1)
        for key, value in p22_events[0].items():
            self.assertEqual(multi_events[0][key], value,
                             "A-regime column %s drifted from P22" % key)
        p22_rows = p22.event_responses(timeline, p22_events)
        multi_rows = p23.event_responses_multi(timeline, multi_events)
        for key, value in p22_rows[0].items():
            self.assertEqual(multi_rows[0][key], value,
                             "A-regime response %s drifted from P22" % key)

    def test_p22_onset_events_accepts_the_multi_rows(self):
        """A P22 reader can consume P23 event rows for the A regime."""
        timeline = timeline_of(regime_series("A", cascade=True))
        events = p23.onset_events_for_regime(timeline, "A")
        rows = p22.event_responses(timeline, events)
        self.assertEqual(rows[0]["ram_response"],
                         events[0]["ram_response"])
        self.assertEqual(rows[0]["disk_response"],
                         events[0]["disk_response"])

    def test_registration_is_internally_consistent(self):
        report = p23.assert_registered_regimes()
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["provisional_regimes"], ["B", "C"])

    def test_drift_in_the_registered_table_is_detected(self):
        drifted = copy.deepcopy(p23.REGIMES)
        drifted["B"]["response_windows"]["disk"] = (4, 14)
        with self.assertRaises(AssertionError):
            p23.assert_registered_regimes(drifted)
        drifted = copy.deepcopy(p23.REGIMES)
        drifted["C"]["onset_tau"] = 900.0          # below the familiar clip
        with self.assertRaises(AssertionError):
            p23.assert_registered_regimes(drifted)

    def test_provisional_thresholds_are_flagged_and_above_the_familiar_clip(self):
        report = p23.assert_registered_regimes()
        self.assertEqual(sorted(report["provisional_onset_taus"]),
                         ["disk", "ram"])
        self.assertEqual(p23.PROVISIONAL_TAUS["ram"], 2810.0)
        self.assertEqual(p23.PROVISIONAL_TAUS["disk"], 11600.0)
        self.assertEqual(report["familiar_clip"],
                         {"cpu": 1860.0, "ram": 1400.0, "disk": 9000.0})
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            self.assertGreater(spec["onset_tau"],
                               FAMILIAR[spec["onset_resource"]])
        self.assertFalse(p23.onset_events_for_regime(
            timeline_of(regime_series("A")), "A")[0]["onset_provisional"])
        for regime_id in ("B", "C"):
            self.assertTrue(p23.onset_events_for_regime(
                timeline_of(regime_series(regime_id)), regime_id)[0]
                ["onset_provisional"])

    def test_registered_aliases_resolve_to_the_canonical_ids(self):
        """The generator registers the same regimes as compute/memory/io_first."""
        for alias, canonical in (("compute_first", "A"), ("memory_first", "B"),
                                 ("io_first", "C"), ("regime_A", "A"),
                                 ("a", "A"), ("io-first", "C")):
            self.assertEqual(p23.canonical_regime_id(alias), canonical)
            self.assertEqual(p23.regime(alias)["regime_id"], canonical)
        self.assertEqual(p23.canonical_regime_id("A"), "A")
        with self.assertRaises(KeyError):
            p23.canonical_regime_id("compute_second")
        with self.assertRaises(KeyError):
            p23.regime("compute_second")

    def test_accessors_hand_back_copies(self):
        regime_a = p23.regime("A")
        regime_a["sequence"].append(["ram", 99])
        regime_a["response_windows"]["ram"] = (0, 1)
        self.assertEqual(p23.REGIMES["A"]["sequence"],
                         [["cpu", 0], ["ram", 4], ["disk", 8]])
        self.assertEqual(p23.REGIMES["A"]["response_windows"]["ram"], (4, 14))
        self.assertEqual(sorted(p23.regime("B")["response_windows"]),
                         ["cpu", "disk"])
        with self.assertRaises(KeyError):
            p23.regime("D")

    def test_registered_lag_lookup(self):
        self.assertEqual(p23.registered_lag("A", "cpu"), 0)
        self.assertEqual(p23.registered_lag("A", "ram"), 4)
        self.assertEqual(p23.registered_lag("A", "disk"), 8)
        self.assertEqual(p23.registered_lag("B", "disk"), 3)
        self.assertEqual(p23.registered_lag("B", "cpu"), 6)
        self.assertEqual(p23.registered_lag("C", "cpu"), 3)
        self.assertEqual(p23.registered_lag("C", "ram"), 6)
        with self.assertRaises(ValueError):
            p23.registered_lag("A", "gpu")

    def test_forbidden_inputs_cover_the_plan_section_4_list(self):
        hits = p23.forbidden_input_hits(
            ["ratio_cpu", "regime_id_mean", "mechanism_id", "event_id_max",
             "phase_id"])
        tokens = sorted(hit["token"] for hit in hits)
        self.assertEqual(tokens, ["event_id", "mechanism_id", "phase_id",
                                  "regime_id"])
        visible = ["ratio_cpu", "ratio_ram", "ratio_disk", "slope1_cpu",
                   "slope3_ram", "slope6_disk", "occupancy", "migration_mass_in",
                   "past_fault_0", "past_fault_1"]
        self.assertEqual(p23.forbidden_input_hits(visible), [])
        for regime_id in p23.REGIME_IDS:
            for token in ("regime_id", "phase_id", "mechanism_id", "event_id"):
                self.assertIn(token, p23.regime(regime_id)["forbidden_inputs"])


# --------------------------------------------------------------------------
# T023-01 — same-host isolation, per regime
# --------------------------------------------------------------------------
class SameHostIsolationTests(unittest.TestCase):
    """Two tasks on ONE host: the host aggregate co-moves, the tasks do not."""

    def setUp(self):
        steps = 40
        # Task 0: CPU burst only.  Regime-A trigger, no RAM/Disk response.
        cpu0 = np.full(steps, 900.0)
        cpu0[6:11] = 4600.0
        # Task 1: RAM burst only.  Regime-B trigger, no CPU/Disk response.
        ram1 = np.full(steps, 700.0)
        ram1[6:16] = 5200.0
        self.tasks = {
            0: {"slot": 0, "start": 0, "cpu": cpu0,
                "ram": np.full(steps, 700.0),
                "disk": np.full(steps, 3000.0), "hosts": np.zeros(steps, int)},
            1: {"slot": 1, "start": 0, "cpu": np.full(steps, 800.0),
                "ram": ram1, "disk": np.full(steps, 3000.0),
                "hosts": np.zeros(steps, int)},
        }
        self.timeline = p23.build_task_timeline(*rows_from_tasks(self.tasks))

    def test_only_the_cpu_task_triggers_a_regime_a_onset(self):
        events = p23.onset_events_for_regime(self.timeline, "A")
        self.assertEqual([e["creation_id"] for e in events], [0])
        self.assertEqual(events[0]["onset_resource"], "cpu")
        self.assertEqual(events[0]["index"], 6)
        self.assertEqual(events[0]["host_onset"], 0)

    def test_only_the_ram_task_triggers_a_regime_b_onset(self):
        events = p23.onset_events_for_regime(self.timeline, "B")
        self.assertEqual([e["creation_id"] for e in events], [1])
        self.assertEqual(events[0]["onset_resource"], "ram")
        self.assertEqual(events[0]["index"], 6)

    def test_no_regime_c_onset_on_a_disk_quiet_host(self):
        self.assertEqual(p23.onset_events_for_regime(self.timeline, "C"), [])

    def test_the_cpu_task_is_not_a_cascade_at_task_level(self):
        """Task 0's own RAM never rises: the host co-movement is co-residency."""
        events = p23.onset_events_for_regime(self.timeline, "A")
        self.assertLessEqual(events[0]["ram_response"], 0.0)
        self.assertFalse(events[0]["ram_response_positive"])

    def test_host_aggregate_co_moves_and_would_invent_a_response(self):
        cpu = np.stack([self.tasks[0]["cpu"], self.tasks[1]["cpu"]]).sum(0)
        ram = np.stack([self.tasks[0]["ram"], self.tasks[1]["ram"]]).sum(0)
        self.assertGreater(float(np.corrcoef(cpu, ram)[0, 1]), 0.5,
                           "the synthetic host aggregate is expected to co-move")
        onset = int(np.argmax(np.diff(cpu))) + 1
        baseline = ram[onset - 1]
        window = ram[onset + 4:onset + 14].max() - baseline
        self.assertGreater(window, 0.0,
                           "host aggregation invents a RAM response that no "
                           "single task produced")

    def test_each_regime_scan_is_blind_to_the_other_task(self):
        counts = {regime_id: len(p23.onset_events_for_regime(self.timeline,
                                                            regime_id))
                  for regime_id in p23.REGIME_IDS}
        self.assertEqual(counts, {"A": 1, "B": 1, "C": 0})


# --------------------------------------------------------------------------
# T023-02 / T023-03 — identity under migration and slot reuse
# --------------------------------------------------------------------------
class MigrationIdentityTests(unittest.TestCase):
    """creation_id, not slot or host, is the task identity."""

    def setUp(self):
        steps = 30
        cpu = np.full(steps, 900.0)
        cpu[5:9] = 4700.0
        hosts = np.zeros(steps, int)
        hosts[15:] = 1                      # migrated at the 15th interval
        self.timeline = timeline_of(
            regime_series("A", steps=steps), creation=7, slot=3, hosts=hosts)
        self.flat_timeline = timeline_of(
            regime_series("A", steps=steps), creation=7, slot=3)
        self.continuity = p23.migration_continuity(self.timeline)

    def test_trajectory_stays_one_task(self):
        self.assertEqual(list(self.timeline), [7])
        self.assertEqual(self.timeline[7]["t"].size, 30)

    def test_host_change_is_recorded_not_split(self):
        info = self.continuity["per_task"][7]
        self.assertEqual(info["n_host_changes"], 1)
        self.assertEqual(info["changed_at"], [14])
        self.assertEqual(info["hosts"][0], 0)
        self.assertEqual(info["hosts"][-1], 1)

    def test_the_onset_keeps_the_single_identity_across_the_migration(self):
        events = p23.onset_events_for_regime(self.timeline, "A")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["creation_id"], 7)
        self.assertEqual(events[0]["regime_id"], "A")
        self.assertEqual(events[0]["onset_resource"], "cpu")

    def test_migration_does_not_change_the_onset_or_the_responses(self):
        moved = first_event(self.timeline, "A")
        stayed = first_event(self.flat_timeline, "A")
        volatile = ("host_onset", "event_key")
        for key in stayed:
            if key in volatile:
                continue
            self.assertEqual(moved[key], stayed[key],
                             "migration changed %s" % key)
        self.assertEqual(moved["host_onset"], 0)
        self.assertEqual(stayed["host_onset"], 0)


class SlotReuseTests(unittest.TestCase):
    """A reused slot must never be spliced into the previous task."""

    def setUp(self):
        # creation 0 occupies slot 2 for 10 intervals, is destroyed, then
        # creation 1 reuses slot 2 from interval 10 onwards and bursts.
        old = {"slot": 2, "start": 0, "cpu": np.full(10, 900.0),
               "ram": np.full(10, 800.0), "disk": np.full(10, 2000.0),
               "hosts": np.zeros(10, int)}
        burst = regime_series("A", steps=12)
        new = {"slot": 2, "start": 10, "cpu": burst["cpu"], "ram": burst["ram"],
               "disk": burst["disk"], "hosts": np.ones(12, int)}
        self.timeline = p23.build_task_timeline(
            *rows_from_tasks({0: old, 1: new}, n_hosts=2))
        self.integrity = p23.timeline_integrity(self.timeline)

    def test_two_distinct_tasks_in_one_slot(self):
        self.assertEqual(sorted(self.timeline), [0, 1])
        self.assertEqual(self.timeline[0]["slot"].tolist(), [2] * 10)
        self.assertEqual(self.timeline[1]["slot"].tolist(), [2] * 12)

    def test_identities_are_not_merged(self):
        self.assertEqual(self.integrity["reused_slots"], 1)
        self.assertEqual(self.integrity["reused_slot_examples"], {2: [0, 1]})
        self.assertTrue(self.integrity["ok"], self.integrity["problems"])
        self.assertNotIn(0, self.timeline[1]["host"].tolist(),
                         "the new task must not inherit the old task's host rows")

    def test_the_onset_is_indexed_on_the_new_task_own_age_axis(self):
        """A slot-spliced trajectory would index this burst at 11, not 1."""
        event = first_event(self.timeline, "A")
        self.assertEqual(event["creation_id"], 1)
        self.assertEqual(event["index"], 1)
        self.assertEqual(event["age_onset"], 1)
        self.assertEqual(event["t_onset"], 11)
        self.assertEqual(event["host_onset"], 1)

    def test_duplicate_creation_id_in_one_interval_is_rejected(self):
        args = rows_from_tasks({
            0: {"slot": 0, "start": 0, "cpu": [1.0], "ram": [1.0],
                "disk": [1.0], "hosts": [0]},
            1: {"slot": 0, "start": 0, "cpu": [1.0], "ram": [1.0],
                "disk": [1.0], "hosts": [0]},
        })
        with self.assertRaises(ValueError):
            p23.build_task_timeline(*args)


# --------------------------------------------------------------------------
# T023-04 — future rows and future metadata are not inputs
# --------------------------------------------------------------------------
class FutureMetadataIsolationTests(unittest.TestCase):
    """Only rows up to t0 + max(window) may influence an event."""

    STEPS = 40
    FUTURE_FROM = 20          # onset sits at index 1, so windows end at 19

    def setUp(self):
        self.timeline = timeline_of(regime_series("A", steps=self.STEPS,
                                                  cascade=True))
        self.events = p23.onset_events_for_regime(self.timeline, "A")
        self.assertGreater(self.events[0]["ram_response"], 0.0)
        self.assertGreater(self.events[0]["disk_response"], 0.0)

    def corrupt(self, extra_rows=0):
        """Garbage future rows + garbage audit metadata (finite on purpose)."""
        corrupted = {}
        for creation_id, entry in self.timeline.items():
            clone = {key: (value.copy() if isinstance(value, np.ndarray)
                           else value) for key, value in entry.items()}
            for name, replacement in (("cpu", 1000.0), ("ram", 1000.0),
                                      ("disk", 1000.0)):
                clone[name][self.FUTURE_FROM:] = replacement
            clone["event_id"] = np.full(entry["event_id"].shape, 999,
                                        dtype=np.int64)
            if extra_rows:
                pad = np.full(extra_rows, 1000.0)
                clone["t"] = np.concatenate([clone["t"],
                                             np.arange(extra_rows) + clone["t"][-1] + 1])
                clone["slot"] = np.concatenate([clone["slot"],
                                                np.full(extra_rows, clone["slot"][-1])])
                clone["age"] = clone["t"] - clone["t"][0]
                clone["host"] = np.concatenate([clone["host"],
                                                np.full(extra_rows, clone["host"][-1])])
                clone["event_id"] = np.concatenate([clone["event_id"],
                                                    np.full(extra_rows, 999)])
                for name in ("cpu", "ram", "disk"):
                    clone[name] = np.concatenate([clone[name], pad])
            corrupted[creation_id] = clone
        return corrupted

    def _compare(self, other):
        other_events = p23.onset_events_for_regime(other, "A")
        self.assertEqual(len(other_events), len(self.events))
        for base, changed in zip(self.events, other_events):
            for key, value in base.items():
                if key == "audit_event_id":
                    continue          # metadata is *allowed* to change
                self.assertEqual(changed[key], value,
                                 "future information leaked into %s" % key)

    def test_garbage_future_rows_change_nothing(self):
        self._compare(self.corrupt())

    def test_appended_future_rows_change_nothing(self):
        self._compare(self.corrupt(extra_rows=7))

    def test_future_audit_metadata_changes_no_response_value(self):
        base = p23.event_responses_multi(self.timeline, self.events)
        other = p23.event_responses_multi(self.corrupt(), self.events)
        for a, b in zip(base, other):
            for key in ("ram_response", "disk_response", "ram_at_exact_lag",
                        "disk_at_exact_lag", "ram_time_to_peak",
                        "disk_time_to_peak"):
                self.assertEqual(a[key], b[key],
                                 "audit metadata leaked into %s" % key)

    def test_truncating_after_the_window_changes_nothing(self):
        truncated = {}
        for creation_id, entry in self.timeline.items():
            clone = {key: (value.copy() if isinstance(value, np.ndarray)
                           else value) for key, value in entry.items()}
            keep = max(window[1] for window
                       in p23.REGIMES["A"]["response_windows"].values()) + 2
            for name in ("t", "slot", "age", "host", "cpu", "ram", "disk",
                         "event_id"):
                clone[name] = clone[name][:keep]
            truncated[creation_id] = clone
        self._compare(truncated)


# --------------------------------------------------------------------------
# T023-05 — the three regimes stay separated
# --------------------------------------------------------------------------
class RegimeOnsetIsolationTests(unittest.TestCase):
    """Each regime triggers on its OWN resource and on no other."""

    def test_each_regime_is_the_only_one_that_fires_on_its_own_timeline(self):
        for regime_id in p23.REGIME_IDS:
            with self.subTest(regime=regime_id):
                timeline = timeline_of(regime_series(regime_id))
                counts = {other: len(p23.onset_events_for_regime(timeline, other))
                          for other in p23.REGIME_IDS}
                expected = {other: (1 if other == regime_id else 0)
                            for other in p23.REGIME_IDS}
                self.assertEqual(counts, expected)
                event = first_event(timeline, regime_id)
                spec = p23.regime(regime_id)
                self.assertEqual(event["onset_resource"], spec["onset_resource"])
                self.assertEqual(event["regime_id"], regime_id)
                self.assertEqual(event["age_onset"], 1)
                self.assertEqual(event["t_onset"], 1)
                self.assertEqual(event["value_at_onset"],
                                 REGIME_BURST[regime_id])
                self.assertEqual(event["baseline_source"],
                                 "observed pre-onset median")
                self.assertEqual(event["baseline_rows"], 1)
                self.assertEqual(event["onset_tau"], spec["onset_tau"])
                self.assertEqual(event["onset_baseline_lag"],
                                 p23.ONSET_BASELINE_LAG)
                self.assertGreater(
                    event["delta_at_onset"],
                    spec["onset_tau"] - FAMILIAR[spec["onset_resource"]])
                for name in p23.RESOURCES:
                    self.assertEqual(event["%s_baseline" % name],
                                     FAMILIAR[name])

    def test_a_sustained_burst_counts_once_per_regime(self):
        for regime_id in p23.REGIME_IDS:
            with self.subTest(regime=regime_id):
                timeline = timeline_of(regime_series(regime_id, steps=30))
                events = p23.onset_events_for_regime(timeline, regime_id)
                self.assertEqual(len(events), 1)

    def test_a_task_that_never_crosses_its_threshold_has_no_onset(self):
        for regime_id in p23.REGIME_IDS:
            with self.subTest(regime=regime_id):
                values = {name: np.full(20, FAMILIAR[name])
                          for name in p23.RESOURCES}
                timeline = timeline_of(values)
                self.assertEqual(
                    p23.onset_events_for_regime(timeline, regime_id), [])

    def test_regime_id_is_copied_from_the_caller_mapping(self):
        timeline = timeline_of(regime_series("A"))
        spec = p23.regime("A")
        windows = spec["response_windows"]
        custom = p23.onset_events_multi(timeline, "cpu", spec["onset_tau"],
                                        p23.ONSET_BASELINE_LAG, windows,
                                        regime_map={"cpu": "custom_A"})
        self.assertEqual(custom[0]["regime_id"], "custom_A")
        reversed_map = p23.onset_events_multi(timeline, "cpu", spec["onset_tau"],
                                              p23.ONSET_BASELINE_LAG, windows,
                                              regime_map={"registered_A": "cpu"})
        self.assertEqual(reversed_map[0]["regime_id"], "registered_A")
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", spec["onset_tau"],
                                   p23.ONSET_BASELINE_LAG, windows,
                                   regime_map={"ram": "A"})
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", spec["onset_tau"],
                                   p23.ONSET_BASELINE_LAG, windows,
                                   regime_map={17: "A"})

    def test_cross_regime_confound_is_controlled_by_the_task_regime_map(self):
        """Pinned hazard: an A cascade also crosses the B and C thresholds."""
        timeline = timeline_of(regime_series("A", cascade=True))
        blinded = {regime_id: len(p23.onset_events_for_regime(timeline, regime_id))
                   for regime_id in p23.REGIME_IDS}
        self.assertEqual(blinded, {"A": 1, "B": 1, "C": 1},
                         "an A cascade's RAM/Disk response crosses the B/C "
                         "registered thresholds; a mixed-stream audit must say "
                         "so rather than silently reporting three regimes")
        filtered = {regime_id: len(p23.onset_events_for_regime(
            timeline, regime_id, task_regime_map={0: "A"}))
            for regime_id in p23.REGIME_IDS}
        self.assertEqual(filtered, {"A": 1, "B": 0, "C": 0})
        with self.assertRaises(ValueError):
            p23.onset_events_for_regime(timeline, "A",
                                        task_regime_map={99: "A"})

    def test_generator_style_regime_ids_do_not_blind_the_task_map(self):
        """A stamped ``compute_first`` must select the A regime, not nothing."""
        timeline = timeline_of(regime_series("A"))
        spec = p23.regime("A")
        stamped = p23.onset_events_multi(timeline, "cpu", spec["onset_tau"],
                                         p23.ONSET_BASELINE_LAG,
                                         spec["response_windows"],
                                         regime_map={"cpu": "compute_first"})
        self.assertEqual(stamped[0]["regime_id"], "A")
        bypassed = p23.onset_events_multi(timeline, "cpu", spec["onset_tau"],
                                          p23.ONSET_BASELINE_LAG,
                                          spec["response_windows"],
                                          regime_map={"cpu": "custom_A"})
        self.assertEqual(bypassed[0]["regime_id"], "custom_A",
                         "an unregistered caller id is copied verbatim")
        filtered = p23.onset_events_for_regime(
            timeline, "A", task_regime_map={0: "compute_first"})
        self.assertEqual(len(filtered), 1)
        masked = p23.onset_events_for_regime(
            timeline, "A", task_regime_map={0: "memory_first"})
        self.assertEqual(masked, [])
        alien = p23.onset_events_for_regime(
            timeline, "A", task_regime_map={0: "custom_A"})
        self.assertEqual(alien, [],
                         "an unregistered task id cannot be equal to regime A")

    def test_multi_resource_scan_rejects_an_unregistered_resource(self):
        timeline = timeline_of(regime_series("A"))
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "gpu", 1.0, 4)
        with self.assertRaises(ValueError):
            p23.onset_positions_multi(np.full(5, 1.0), 1.0, resource="gpu")
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", float("nan"), 4)


# --------------------------------------------------------------------------
# T023-06 — the registered response windows
# --------------------------------------------------------------------------
class ResponseWindowTests(unittest.TestCase):
    """A response at the registered lag is detected; a wrong lag is not."""

    def test_window_start_is_the_registered_sequence_lag(self):
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            for name, window in spec["response_windows"].items():
                self.assertEqual(window[0], SEQUENCE_LAG[regime_id][name])
                self.assertEqual(window[1] - window[0], 10,
                                 "the registered response duration band")

    def test_event_rows_carry_the_registered_windows(self):
        for regime_id in p23.REGIME_IDS:
            with self.subTest(regime=regime_id):
                event = first_event(timeline_of(regime_series(regime_id)),
                                    regime_id)
                spec = p23.regime(regime_id)
                for name, window in spec["response_windows"].items():
                    self.assertEqual(event["%s_window" % name], list(window))
                    self.assertEqual(event["%s_registered_lag" % name],
                                     window[0])
                self.assertEqual(sorted(p23.RESOURCES),
                                 sorted([spec["onset_resource"]] +
                                        list(spec["response_windows"])))

    def test_planted_response_at_the_registered_lag_is_detected(self):
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            for name, window in spec["response_windows"].items():
                with self.subTest(regime=regime_id, resource=name):
                    timeline = timeline_of(regime_series(
                        regime_id, spike=name, offset=window[0]))
                    event = first_event(timeline, regime_id)
                    self.assertGreater(event["%s_response" % name], 0.0)
                    self.assertTrue(event["%s_response_positive" % name])
                    self.assertEqual(event["%s_time_to_peak" % name],
                                     window[0])
                    self.assertGreater(event["%s_at_exact_lag" % name], 0.0)
                    rate = p23.alignment_rate_multi([event], name, window[0],
                                                    tolerance=1)
                    self.assertEqual(rate["n"], 1)
                    self.assertEqual(rate["rate"], 1.0)
                    self.assertEqual(rate["rate_positive"], 1.0)
                    self.assertEqual(
                        p23.response_durations(timeline, [event])["max"], 1)
                    self.assertGreater(p23.peak_ratio_summary([event])["p50"],
                                       0.0)

    def test_response_at_a_wrong_lag_is_not_attributed_to_the_register(self):
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            for name, window in spec["response_windows"].items():
                with self.subTest(regime=regime_id, resource=name):
                    timeline = timeline_of(regime_series(
                        regime_id, spike=name, offset=window[0] + 4))
                    event = first_event(timeline, regime_id)
                    self.assertGreater(event["%s_response" % name], 0.0,
                                       "the excursion is inside the window, so "
                                       "the window maximum must see it")
                    self.assertEqual(event["%s_time_to_peak" % name],
                                     window[0] + 4)
                    self.assertAlmostEqual(event["%s_at_exact_lag" % name], 0.0,
                                           places=9,
                                           msg="nothing rose at the registered "
                                               "lag itself")
                    rate = p23.alignment_rate_multi([event], name, window[0],
                                                    tolerance=1)
                    self.assertEqual(rate["hits"], 0)
                    self.assertEqual(rate["rate"], 0.0)
                    self.assertEqual(rate["rate_positive"], 0.0)

    def test_response_outside_the_window_is_not_attributed(self):
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            for name, window in spec["response_windows"].items():
                with self.subTest(regime=regime_id, resource=name):
                    timeline = timeline_of(regime_series(
                        regime_id, spike=name, offset=window[1] + 1))
                    event = first_event(timeline, regime_id)
                    self.assertEqual(event["%s_response" % name], 0.0)
                    self.assertFalse(event["%s_response_positive" % name])
                    self.assertEqual(event["%s_at_exact_lag" % name], 0.0)
                    rate = p23.alignment_rate_multi([event], name, window[0],
                                                    tolerance=1)
                    self.assertIsNone(rate["rate_positive"],
                                      "a flat response is not evidence of "
                                      "alignment; rate_positive must be None, "
                                      "not a fabricated 1.0")

    def test_the_window_is_half_open(self):
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            for name, window in spec["response_windows"].items():
                with self.subTest(regime=regime_id, resource=name):
                    inside = first_event(timeline_of(regime_series(
                        regime_id, spike=name, offset=window[1] - 1)),
                        regime_id)
                    self.assertGreater(inside["%s_response" % name], 0.0)
                    self.assertEqual(inside["%s_time_to_peak" % name],
                                     window[1] - 1)
                    outside = first_event(timeline_of(regime_series(
                        regime_id, spike=name, offset=window[1])), regime_id)
                    self.assertEqual(outside["%s_response" % name], 0.0)

    def test_response_windows_must_cover_both_other_resources(self):
        timeline = timeline_of(regime_series("A"))
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", 2600.0, 4,
                                   response_windows={"ram": (4, 14)})
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", 2600.0, 4,
                                   response_windows={"ram": (4, 14),
                                                     "cpu": (8, 18)})
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", 2600.0, 4,
                                   response_windows={"ram": (14, 4),
                                                     "disk": (8, 18)})
        with self.assertRaises(ValueError):
            p23.onset_events_multi(timeline, "cpu", 2600.0, 4,
                                   response_windows={"ram": (4, 14),
                                                     "gpu": (8, 18)})

    def test_event_responses_multi_rejects_a_foreign_onset_resource(self):
        timeline = timeline_of(regime_series("A"))
        events = p23.onset_events_for_regime(timeline, "A")
        with self.assertRaises(ValueError):
            p23.event_responses_multi(timeline, events, onset_resource="ram",
                                      response_windows={"cpu": (3, 13),
                                                        "disk": (6, 16)})


# --------------------------------------------------------------------------
# T023-07 — plan §9 marginal matching and the joint-structure contrast
# --------------------------------------------------------------------------
def regime_stats(prevalence=0.07, events=200, valid=140, deployment=0.10,
                 migration=0.20, worst=0.05, mean_duration=8.0, peak_ratio=3.0,
                 host_steps=4000, event_count=None):
    """A complete per-regime stats dict in the registered schema."""
    return {
        "prevalence": prevalence,
        "independent_fault_events": events,
        "valid_onset_followup": valid,
        "deployment_attempts": 1000,
        "deployment_rejected": int(round(deployment * 1000)),
        "migration_attempts": 500,
        "migration_rejected": int(round(migration * 500)),
        "worst_event_share": worst,
        "event_count": events if event_count is None else event_count,
        "mean_duration": mean_duration,
        "peak_ratio": peak_ratio,
        "host_steps": host_steps,
        "positive_hoststeps": int(round(prevalence * host_steps)),
    }


MATCHED = {
    "A": regime_stats(0.070, events=200),
    "B": regime_stats(0.068, events=205),
    "C": regime_stats(0.072, events=198),
}


class MarginalMatchTests(unittest.TestCase):
    """Plan §9: marginals matched, joint structure changed."""

    def test_matched_inputs_pass_the_registered_gate(self):
        report = p23.marginal_match_report(MATCHED)
        self.assertTrue(report["passed"],
                        [key for key, ok in report["checks"].items() if not ok])
        self.assertTrue(all(report["checks"].values()))
        self.assertAlmostEqual(report["metrics"]["prevalence_spread"], 0.004,
                               places=12)
        self.assertEqual(report["metrics"]["regimes_reported"], ["A", "B", "C"])

    def test_gate_dict_has_the_p22_shape_and_thresholds(self):
        report = p23.marginal_match_report(MATCHED)
        for key in ("thresholds", "checks", "passed", "metrics", "per_regime"):
            self.assertIn(key, report)
        self.assertEqual(report["thresholds"],
                         p23.MARGINAL_MATCH_THRESHOLDS)
        self.assertEqual(p23.MARGINAL_MATCH_THRESHOLDS["prevalence_spread_max"],
                         0.04)
        self.assertEqual(
            p23.MARGINAL_MATCH_THRESHOLDS["independent_fault_events_min"], 80)
        self.assertEqual(
            p23.MARGINAL_MATCH_THRESHOLDS["valid_onset_followup_min"], 50)
        self.assertEqual(
            p23.MARGINAL_MATCH_THRESHOLDS["deployment_rejection_max"], 0.25)
        self.assertEqual(
            p23.MARGINAL_MATCH_THRESHOLDS["migration_rejection_max"], 0.40)
        self.assertEqual(
            p23.MARGINAL_MATCH_THRESHOLDS["worst_event_share_max"], 0.10)

    def test_prevalence_spread_of_five_points_fails(self):
        """A 5-point spread is the registered failure mode (plan §9: <= 4pp)."""
        spread = dict(MATCHED)
        spread["A"] = regime_stats(0.100, events=200)
        spread["B"] = regime_stats(0.050, events=205)
        spread["C"] = regime_stats(0.075, events=198)
        report = p23.marginal_match_report(spread)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["prevalence_spread_ok"])
        self.assertAlmostEqual(report["metrics"]["prevalence_spread"], 0.05,
                               places=12)
        failing = sorted(key for key, ok in report["checks"].items() if not ok)
        self.assertEqual(failing, ["prevalence_spread_ok"],
                         "the injected 0.05 spread must be the only failure: "
                         "every regime prevalence is still inside the "
                         "registered 3%-12% band")
        inside_band = dict(MATCHED)
        inside_band["B"] = regime_stats(0.050, events=205)
        report = p23.marginal_match_report(inside_band)
        self.assertAlmostEqual(report["metrics"]["prevalence_spread"], 0.022,
                               places=12)
        self.assertTrue(report["checks"]["prevalence_spread_ok"])
        self.assertTrue(report["passed"])

    def test_event_count_below_eighty_fails(self):
        few = dict(MATCHED)
        few["B"] = regime_stats(0.068, events=79, event_count=79)
        report = p23.marginal_match_report(few)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["B_independent_fault_events_enough"])
        self.assertTrue(report["checks"]["A_independent_fault_events_enough"])
        self.assertEqual(report["per_regime"]["B"]["independent_fault_events"],
                         79)

    def test_valid_onset_followup_below_fifty_fails(self):
        few = dict(MATCHED)
        few["C"] = regime_stats(0.072, events=198, valid=49)
        report = p23.marginal_match_report(few)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["C_valid_onset_followup_enough"])

    def test_rejection_rates_and_worst_event_share_are_gated(self):
        bad = dict(MATCHED)
        bad["A"] = regime_stats(0.070, deployment=0.26, migration=0.41,
                                worst=0.10)
        report = p23.marginal_match_report(bad)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["A_deployment_rejection_ok"])
        self.assertFalse(report["checks"]["A_migration_rejection_ok"])
        self.assertFalse(report["checks"]["A_worst_event_share_ok"])
        self.assertAlmostEqual(
            report["per_regime"]["A"]["deployment_rejection_rate"], 0.26)

    def test_missing_regime_and_missing_quantity_fail_loudly(self):
        report = p23.marginal_match_report({"A": MATCHED["A"], "B": MATCHED["B"]})
        self.assertFalse(report["checks"]["all_registered_regimes_reported"])
        self.assertFalse(report["passed"])
        broken = dict(MATCHED)
        broken["A"] = dict(MATCHED["A"])
        del broken["A"]["peak_ratio"]
        with self.assertRaises(ValueError):
            p23.marginal_match_report(broken)
        with self.assertRaises(TypeError):
            p23.marginal_match_report({"A": 1.0})
        with self.assertRaises(ValueError):
            p23.marginal_match_report({})

    def test_report_functions_accept_the_generator_regime_names(self):
        registered_names = {"compute_first": MATCHED["A"],
                            "memory_first": MATCHED["B"],
                            "io_first": MATCHED["C"]}
        report = p23.marginal_match_report(registered_names)
        self.assertTrue(report["passed"],
                        [key for key, ok in report["checks"].items() if not ok])
        self.assertEqual(report["metrics"]["regimes_reported"], ["A", "B", "C"])
        contrast = p23.regime_contrast_report(registered_names)
        self.assertTrue(contrast["passed"])
        self.assertIn("A|B", contrast["pairs"])
        self.assertEqual(contrast["pairs"]["A|B"]["allowed_joint_differences"]
                         ["resource_order"]["B"], ["ram", "disk", "cpu"])
        with self.assertRaises(ValueError):
            p23.marginal_match_report({"A": MATCHED["A"],
                                       "compute_first": MATCHED["B"]})

    def test_marginals_matched_while_joint_structure_differs(self):
        report = p23.regime_contrast_report(MATCHED)
        self.assertTrue(report["passed"],
                        [key for key, ok in report["checks"].items() if not ok])
        self.assertEqual(sorted(report["pairs"]), ["A|B", "A|C", "B|C"])
        for pair, info in report["pairs"].items():
            self.assertTrue(info["marginals_matched"],
                            "%s marginals are matched in the input" % pair)
            matched = info["matched_marginals"]
            self.assertLessEqual(matched["prevalence_diff"],
                                 p23.CONTRAST_TOLERANCES["prevalence_diff_max"])
            self.assertLessEqual(matched["event_count_diff_rel"], 0.10)
            self.assertLessEqual(matched["mean_duration_diff_rel"], 0.10)
            self.assertLessEqual(matched["peak_ratio_diff_rel"], 0.10)
            self.assertLessEqual(matched["deployment_rejection_diff"], 0.05)
            self.assertLessEqual(matched["migration_rejection_diff"], 0.05)
            joint = info["allowed_joint_differences"]
            self.assertTrue(joint["joint_structure_differs"])
            self.assertTrue(joint["onset_resource_differs"])
            self.assertTrue(joint["resource_order_differs"])
            self.assertTrue(joint["lag_profile_differs"])

    def test_contrast_reports_the_registered_order_and_lags(self):
        report = p23.regime_contrast_report(MATCHED)
        joint = report["pairs"]["A|B"]["allowed_joint_differences"]
        self.assertEqual(joint["resource_order"]["A"], ["cpu", "ram", "disk"])
        self.assertEqual(joint["resource_order"]["B"], ["ram", "disk", "cpu"])
        self.assertEqual(joint["lag_by_resource"]["A"],
                         {"cpu": 0, "ram": 4, "disk": 8})
        self.assertEqual(joint["lag_by_resource"]["B"],
                         {"ram": 0, "disk": 3, "cpu": 6})
        self.assertEqual(joint["dependency_pairs"]["A"],
                         [["cpu", "ram"], ["ram", "disk"]])
        self.assertEqual(joint["dependency_pairs"]["B"],
                         [["ram", "disk"], ["disk", "cpu"]])
        self.assertEqual(
            report["pairs"]["B|C"]["allowed_joint_differences"]
            ["onset_resource"], {"B": "ram", "C": "disk"})

    def test_contrast_detects_unmatched_marginals(self):
        skewed = dict(MATCHED)
        skewed["C"] = regime_stats(0.072, events=420, event_count=420,
                                   mean_duration=20.0, peak_ratio=9.0)
        report = p23.regime_contrast_report(skewed)
        self.assertFalse(report["passed"])
        self.assertFalse(report["pairs"]["A|C"]["marginals_matched"])
        self.assertFalse(report["checks"]["A_C_marginals_matched"])
        self.assertFalse(
            report["pairs"]["A|C"]["matched_marginals"]["event_count_matched"])
        self.assertFalse(
            report["pairs"]["A|C"]["matched_marginals"]["mean_duration_matched"])
        with self.assertRaises(ValueError):
            p23.regime_contrast_report({"A": MATCHED["A"], "D": MATCHED["B"]})


# --------------------------------------------------------------------------
# T023-08 — determinism and the NaN policy
# --------------------------------------------------------------------------
class DeterminismTests(unittest.TestCase):
    """Same inputs -> same outputs; no global state is touched."""

    def setUp(self):
        self.timeline = timeline_of(regime_series("A", spike="ram",
                                                  offset=4, cascade=True))
        self.events = p23.onset_events_for_regime(self.timeline, "A")

    def test_the_instrument_is_deterministic(self):
        again = p23.onset_events_for_regime(self.timeline, "A")
        self.assertEqual(again, self.events)
        self.assertEqual(p23.event_responses_multi(self.timeline, self.events),
                         p23.event_responses_multi(self.timeline, again))
        self.assertEqual(p23.timeline_sha256(self.timeline),
                         p23.timeline_sha256(self.timeline))

    def test_no_statistic_is_nan_on_clean_input(self):
        for event in self.events:
            for key, value in event.items():
                if key.endswith(("_response", "_at_exact_lag", "_baseline",
                                 "_time_to_peak")) or key in (
                        "delta_at_onset", "delta_cpu", "value_at_onset"):
                    self.assertTrue(np.isfinite(value),
                                    "%s is not finite: %r" % (key, value))
        summary = p23.response_summary_from_values(
            [e["ram_response"] for e in self.events])
        self.assertTrue(np.isfinite(summary["p50"]))
        self.assertTrue(np.isfinite(p23.response_durations(
            self.timeline, self.events)["mean"]))
        self.assertTrue(np.isfinite(p23.peak_ratio_summary(
            self.events)["p50"]))

    def test_controls_are_deterministic_for_a_fixed_seed(self):
        windows = p23.regime("A")["response_windows"]
        first = p23.control_m0_multi(self.events, np.random.default_rng(23023),
                                     windows, "cpu")
        second = p23.control_m0_multi(self.events, np.random.default_rng(23023),
                                      windows, "cpu")
        self.assertEqual(first["shifts"], second["shifts"])
        other = p23.control_m0_multi(self.events, np.random.default_rng(999),
                                     windows, "cpu")
        self.assertNotEqual(first["shifts"], other["shifts"])
        m1 = p23.control_m1_multi(self.events, np.random.default_rng(23023),
                                  windows, "cpu")
        m1_again = p23.control_m1_multi(self.events, np.random.default_rng(23023),
                                        windows, "cpu")
        self.assertEqual(m1["shifts"], m1_again["shifts"])

    def test_m0_shifts_are_never_the_identity_roll(self):
        windows = p23.regime("A")["response_windows"]
        horizon = max(window[1] for window in windows.values())
        control = p23.control_m0_multi(self.events * 5,
                                       np.random.default_rng(23023),
                                       windows, "cpu")
        for name, shifts in control["shifts"].items():
            self.assertTrue(shifts)
            for shift in shifts:
                self.assertGreaterEqual(shift, 1)
                self.assertTrue(shift % horizon != 0,
                                "shift %d is a no-op roll on a length-%d curve"
                                % (shift, horizon))

    def test_m0_destroys_the_registered_alignment(self):
        windows = p23.regime("A")["response_windows"]
        row = self.events[0]
        self.assertEqual(p23.alignment_rate_multi(self.events, "ram", 4)
                         ["rate"], 1.0)
        control = p23.control_m0_multi(self.events * 6,
                                       np.random.default_rng(23023),
                                       windows, "cpu")
        shifted = p23.event_responses_multi(self.timeline, self.events * 6,
                                            shifts=control["shifts"])
        candidate = p23.alignment_rate_multi(self.events * 6, "ram", 4)
        shuffled = p23.alignment_rate_multi(shifted, "ram", 4)
        self.assertEqual(candidate["rate"], 1.0)
        self.assertLess(shuffled["rate"], candidate["rate"],
                        "the lag-shuffled control must lose the alignment")
        self.assertEqual(row["ram_response_positive"], True)

    def test_no_registered_constant_is_mutated_by_a_call(self):
        snapshot = {
            "regimes": copy.deepcopy(p23.REGIMES),
            "marginal": copy.deepcopy(p23.MARGINAL_MATCH_THRESHOLDS),
            "contrast": copy.deepcopy(p23.CONTRAST_TOLERANCES),
            "familiar": copy.deepcopy(p23.FAMILIAR_CLIP),
            "forbidden": copy.deepcopy(p23.FORBIDDEN_INPUTS),
            "windows": copy.deepcopy(p23.REGISTERED_RESPONSE_WINDOWS),
        }
        p23.onset_events_for_regime(self.timeline, "A")
        p23.onset_events_for_regime(self.timeline, "B")
        p23.onset_events_for_regime(self.timeline, "C")
        p23.event_responses_multi(self.timeline, self.events)
        p23.response_durations(self.timeline, self.events)
        p23.peak_ratio_summary(self.events)
        p23.control_m0_multi(self.events, np.random.default_rng(1),
                             p23.regime("A")["response_windows"], "cpu")
        p23.control_m1_multi(self.events, np.random.default_rng(1),
                             p23.regime("B")["response_windows"], "ram")
        p23.marginal_match_report(MATCHED)
        p23.regime_contrast_report(MATCHED)
        p23.assert_registered_regimes()
        self.assertEqual(p23.REGIMES, snapshot["regimes"])
        self.assertEqual(p23.MARGINAL_MATCH_THRESHOLDS, snapshot["marginal"])
        self.assertEqual(p23.CONTRAST_TOLERANCES, snapshot["contrast"])
        self.assertEqual(p23.FAMILIAR_CLIP, snapshot["familiar"])
        self.assertEqual(tuple(p23.FORBIDDEN_INPUTS), snapshot["forbidden"])
        self.assertEqual(p23.REGISTERED_RESPONSE_WINDOWS, snapshot["windows"])


class NonFiniteFeaturePolicyTests(unittest.TestCase):
    """A single NaN must raise, never silently produce a constant predictor."""

    def _poisoned(self, resource, value=float("nan"), index=7, regime_id="A"):
        values = regime_series(regime_id, steps=30, cascade=True)
        values[resource][index] = value
        return timeline_of(values)

    def test_a_single_nan_in_a_feature_column_raises(self):
        for regime_id in p23.REGIME_IDS:
            spec = p23.regime(regime_id)
            for resource in p23.RESOURCES:
                with self.subTest(regime=regime_id, resource=resource):
                    timeline = self._poisoned(resource, regime_id=regime_id)
                    with self.assertRaises(ValueError):
                        p23.onset_events_for_regime(timeline, regime_id)

    def test_infinite_values_raise_as_well(self):
        timeline = self._poisoned("ram", value=float("inf"))
        with self.assertRaises(ValueError):
            p23.onset_events_for_regime(timeline, "A")

    def test_the_validator_names_the_offending_column_and_row(self):
        timeline = self._poisoned("disk")
        with self.assertRaises(ValueError) as caught:
            p23.onset_events_for_regime(timeline, "A")
        message = str(caught.exception)
        self.assertIn("disk", message)
        self.assertIn("index 7", message)
        report = p23.feature_finiteness_report(
            {"cpu": np.array([1.0, 2.0]), "ram": np.array([np.nan, 3.0])})
        self.assertFalse(report["ok"])
        self.assertEqual(report["n_non_finite"], 1)
        self.assertEqual(report["per_column"]["ram"]["first_index"], 0)
        self.assertEqual(report["per_column"]["cpu"]["n_non_finite"], 0)
        self.assertTrue(report["per_column"]["cpu"]["n_finite"]
                        if "n_finite" in report["per_column"]["cpu"] else True)
        p23.assert_finite_features(np.ones((3, 2)))

    def test_nan_is_silent_poison_without_the_validator(self):
        """The trap the validator exists for: all-False, no error."""
        window = np.array([np.nan, np.nan, np.nan])
        finite = window[np.isfinite(window)]
        response = float(np.max(finite)) if finite.size else float("nan")
        constant_flag = bool(np.isfinite(response) and response > 0)
        self.assertFalse(constant_flag,
                         "a NaN-poisoned window yields a constant negative "
                         "response flag with no exception -- exactly the P22 "
                         "probe burn")
        self.assertEqual((window >= 2600.0).sum(), 0,
                         "NaN compares False everywhere, so the poisoned "
                         "column looks like a constant predictor")
        timeline = self._poisoned("cpu")
        with self.assertRaises(ValueError):
            p23.onset_events_for_regime(timeline, "A")
        silent = p23.onset_events_for_regime(timeline, "A",
                                             strict_features=False)
        self.assertTrue(silent,
                        "the opt-out path is silent by construction; it exists "
                        "only so the strict default can be shown to be the "
                        "thing that makes the poison loud")

    def test_non_finite_onset_thresholds_are_rejected(self):
        timeline = timeline_of(regime_series("A"))
        with self.assertRaises(ValueError):
            p23.onset_positions_multi(np.full(5, 1.0), float("inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
