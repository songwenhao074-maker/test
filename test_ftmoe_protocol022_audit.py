"""Protocol 022 P22-S0 — mandatory audit-correctness tests (Gate H0).

Plan §4.3 requires all six tests to pass before any formal data generation:

    T-AUDIT-01  same-host mixed isolation: host aggregation shows CPU/RAM
                co-movement, but task A must not be judged a CPU->RAM cascade
    T-AUDIT-02  migration identity: a task that moves host keeps ONE trajectory
    T-AUDIT-03  slot reuse: a freed slot reused by a new container must never be
                spliced into the previous task
    T-AUDIT-04  future-metadata isolation: future cascade_event_id / phase
                metadata must not affect any model-visible feature
    T-AUDIT-05  P20 regression: Protocol022 probability=0 reproduces the frozen
                familiar generator on the demand/label conventions
    T-AUDIT-06  determinism: same replay seed + creation_id + mechanism seed ->
                byte-equivalent envelope / event registry

Run from the repository root:
    python test_ftmoe_protocol022_audit.py
"""
import json
import unittest

import numpy as np

import analyze_ftmoe_protocol022_unseen as az
import ftmoe_protocol022_core as core
from simulator.workload.BitbrainWorkloadProtocol020 import (
    DEFAULT_ADAPTER, Protocol020AdaptedBWGD2)
from simulator.workload.BitbrainWorkloadProtocol021 import CASCADE_V1
from simulator.workload.BitbrainWorkloadProtocol022 import (
    CASCADE_V2, MECHANISM_SEED_DEV, Protocol022CascadeBWGD2)

TARIFF = None


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


def familiar_transform_workload(**kwargs):
    """Protocol020AdaptedBWGD2 limited to the demand transform (no simulator)."""
    workload = Protocol020AdaptedBWGD2(1.0, 1.5, 600, cohort="online", **kwargs)
    workload.creation_id = 0
    return workload


class SameHostIsolationTests(unittest.TestCase):
    """T-AUDIT-01 — the reason protocol 021's U4 gate failed."""

    def setUp(self):
        rng = np.random.default_rng(11)
        steps = 40
        # Task A: CPU burst (>= registered burst floor) starting at k=6.
        cpu_a = np.full(steps, 900.0)
        cpu_a[6:11] = 4600.0
        # Task B: RAM burst only, no CPU activity change at all.
        ram_b = np.full(steps, 700.0)
        ram_b[6:16] = 5200.0
        self.tasks = {
            0: {"slot": 0, "start": 0, "cpu": cpu_a, "ram": np.full(steps, 700.0),
                "disk": np.full(steps, 3000.0), "hosts": np.zeros(steps, int)},
            1: {"slot": 1, "start": 0, "cpu": np.full(steps, 800.0),
                "ram": ram_b, "disk": np.full(steps, 3000.0),
                "hosts": np.zeros(steps, int)},
        }
        self.timeline = core.build_task_timeline(*rows_from_tasks(self.tasks))
        self.events = core.onset_events(self.timeline)
        self.rows = core.event_responses(self.timeline, self.events)

    def test_task_a_has_a_cpu_onset(self):
        by_task = {}
        for event in self.events:
            by_task.setdefault(event["creation_id"], []).append(event)
        self.assertEqual(sorted(by_task), [0],
                         "only the CPU-bursting task has an onset of its own")
        self.assertEqual(len(by_task[0]), 1, "task A must have exactly one onset")

    def test_host_aggregate_co_moves_but_task_a_is_not_a_cascade(self):
        """The host sees CPU and RAM rise together; task A's own RAM is flat."""
        cpu = np.stack([self.tasks[0]["cpu"], self.tasks[1]["cpu"]]).sum(0)
        ram = np.stack([self.tasks[0]["ram"], self.tasks[1]["ram"]]).sum(0)
        host_corr = float(np.corrcoef(cpu, ram)[0, 1])
        self.assertGreater(host_corr, 0.5,
                           "the synthetic host aggregate is expected to co-move")
        row = [r for r in self.rows if r["creation_id"] == 0][0]
        self.assertLessEqual(row["ram_response"], 0.0,
                             "task A's own RAM never rises: not a cascade")

    def test_host_aggregate_reports_a_false_ram_response(self):
        """The old host-scale instrument would have called this a cascade."""
        steps = 40
        features = np.zeros((steps, 1, 7))
        features[:, 0, 0] = (self.tasks[0]["cpu"] + self.tasks[1]["cpu"])
        features[:, 0, 1] = (self.tasks[0]["ram"] + self.tasks[1]["ram"])
        features[:, 0, 4] = (self.tasks[0]["disk"] + self.tasks[1]["disk"])
        host_cpu = features[:, 0, 0]
        jump = np.diff(host_cpu)
        onset = int(np.argmax(jump)) + 1
        baseline = features[onset - 1, 0, 1]
        window = features[onset + 4:onset + 14, 0, 1].max() - baseline
        self.assertGreater(window, 0.0,
                           "host aggregation invents a RAM response that no "
                           "single task produced")


class MigrationIdentityTests(unittest.TestCase):
    """T-AUDIT-02 — creation_id, not slot, is the task identity."""

    def setUp(self):
        steps = 30
        cpu = np.full(steps, 900.0)
        cpu[5:9] = 4700.0
        hosts = np.zeros(steps, int)
        hosts[15:] = 1                      # migrated at the 15th interval
        self.timeline = core.build_task_timeline(*rows_from_tasks({
            7: {"slot": 3, "start": 0, "cpu": cpu, "ram": np.full(steps, 800.0),
                "disk": np.full(steps, 2000.0), "hosts": hosts},
        }, n_hosts=2))
        self.continuity = core.migration_continuity(self.timeline)

    def test_trajectory_stays_one_task(self):
        self.assertEqual(list(self.timeline), [7])
        self.assertEqual(self.timeline[7]["t"].size, 30)

    def test_host_change_is_recorded_not_split(self):
        info = self.continuity["per_task"][7]
        self.assertEqual(info["n_host_changes"], 1)
        self.assertEqual(info["changed_at"], [14])
        self.assertEqual(info["hosts"][0], 0)
        self.assertEqual(info["hosts"][-1], 1)

    def test_onset_event_carries_the_single_identity(self):
        events = core.onset_events(self.timeline)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["creation_id"], 7)


class SlotReuseTests(unittest.TestCase):
    """T-AUDIT-03 — a reused slot must never be spliced."""

    def setUp(self):
        # creation 0 occupies slot 2 for 10 intervals, is destroyed, then
        # creation 1 reuses slot 2 from interval 10 onwards.
        old = {"slot": 2, "start": 0, "cpu": np.full(10, 900.0),
               "ram": np.full(10, 800.0), "disk": np.full(10, 2000.0),
               "hosts": np.zeros(10, int)}
        new = {"slot": 2, "start": 10, "cpu": np.full(12, 950.0),
               "ram": np.full(12, 850.0), "disk": np.full(12, 2100.0),
               "hosts": np.ones(12, int)}
        self.timeline = core.build_task_timeline(
            *rows_from_tasks({0: old, 1: new}, n_hosts=2))
        self.integrity = core.timeline_integrity(self.timeline)

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

    def test_duplicate_creation_id_in_one_interval_is_rejected(self):
        args = rows_from_tasks({
            0: {"slot": 0, "start": 0, "cpu": [1.0], "ram": [1.0],
                "disk": [1.0], "hosts": [0]},
            1: {"slot": 0, "start": 0, "cpu": [1.0], "ram": [1.0],
                "disk": [1.0], "hosts": [0]},
        })
        with self.assertRaises(ValueError):
            core.build_task_timeline(*args)


class FutureMetadataIsolationTests(unittest.TestCase):
    """T-AUDIT-04 — audit metadata can never become a model input."""

    def setUp(self):
        rng = np.random.default_rng(5)
        steps = 60
        cpu = np.full(steps, 900.0)
        cpu[10:14] = 4650.0
        self.timeline = core.build_task_timeline(*rows_from_tasks({
            0: {"slot": 0, "start": 0, "cpu": cpu,
                "ram": np.concatenate([np.full(14, 800.0),
                                       np.full(steps - 14, 4600.0)]),
                "disk": np.concatenate([np.full(18, 2000.0),
                                        np.full(steps - 18, 15000.0)]),
                "hosts": np.zeros(steps, int), "event_id": 0},
        }))
        self.events = core.onset_events(self.timeline)

    def corrupt(self):
        """Replace every future-facing audit field with garbage."""
        corrupted = {}
        for c, entry in self.timeline.items():
            clone = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in entry.items()}
            clone["event_id"] = np.full(entry["event_id"].shape, 999, dtype=np.int64)
            corrupted[c] = clone
        return corrupted

    def test_responses_are_unchanged_when_future_metadata_changes(self):
        base = core.event_responses(self.timeline, self.events)
        other = core.event_responses(self.corrupt(), self.events)
        for a, b in zip(base, other):
            for key in ("ram_response", "disk_response", "ram_at_exact_lag",
                        "disk_at_exact_lag", "ram_time_to_peak",
                        "disk_time_to_peak"):
                self.assertEqual(a[key], b[key],
                                 "audit metadata leaked into %s" % key)

    def test_event_set_is_unchanged_when_future_metadata_changes(self):
        other = core.onset_events(self.corrupt())
        self.assertEqual([e["event_key"] for e in self.events],
                         [e["event_key"] for e in other])

    def test_forbidden_tokens_are_rejected(self):
        hits = core.forbidden_token_hits(["ratio_cpu", "cascade_event_id_mean"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["token"], "cascade_event_id")

    def test_forward_only_columns_have_no_forbidden_token(self):
        visible = ["ratio_cpu", "ratio_ram", "ratio_disk", "slope1_cpu",
                   "slope3_ram", "slope6_disk", "occupancy", "migration_mass_in",
                   "past_fault_0", "past_fault_1"]
        self.assertEqual(core.forbidden_token_hits(visible), [])


class FamiliarRegressionTests(unittest.TestCase):
    """T-AUDIT-05 — probability=0 must reproduce the frozen familiar contract."""

    def _generate(self, factory, calls=6, **kwargs):
        """Generate a multi-interval arrival stream deterministically.

        Both the numpy and the stdlib RNG are seeded: ``BWGD2.
        generateNewContainers`` draws its arrival count and VM index from the
        stdlib generator, so seeding numpy alone is not reproducible.
        """
        import random
        random.seed(4242)
        np.random.seed(4242)
        workload = factory(**kwargs)
        workload.creation_id = 0
        for interval in range(calls):
            workload.generateNewContainers(interval)
        return workload

    def test_zero_probability_matches_the_frozen_familiar_transform(self):
        familiar = self._generate(familiar_transform_workload,
                                  adapter=dict(DEFAULT_ADAPTER, ram_upper=1400))
        candidate = self._generate(
            lambda **kw: Protocol022CascadeBWGD2(
                1.0, 1.5, 600, cohort="online", **kw),
            cascade_probability=0.0,
            adapter=dict(DEFAULT_ADAPTER, ram_upper=1400))
        self.assertGreater(len(candidate.createdContainers), 5,
                           "the regression needs a real arrival stream")
        self.assertEqual(len(familiar.createdContainers),
                         len(candidate.createdContainers))
        self.assertGreater(len(candidate.createdContainers), 0)
        for (cid_a, _, ips_a, ram_a, disk_a), (cid_b, _, ips_b, ram_b, disk_b) \
                in zip(familiar.createdContainers, candidate.createdContainers):
            self.assertEqual(cid_a, cid_b)
            self.assertEqual(list(ips_a.ips_list), list(ips_b.ips_list))
            self.assertEqual(list(ram_a.size_list), list(ram_b.size_list))
            self.assertEqual(type(disk_a).__name__, type(disk_b).__name__)
            self.assertEqual(float(np.max(disk_a.values)), float(np.max(disk_b.values)))
            self.assertTrue(np.allclose(disk_a.transition, disk_b.transition))

    def test_zero_probability_creates_no_cascade_event(self):
        candidate = self._generate(
            lambda **kw: Protocol022CascadeBWGD2(
                1.0, 1.5, 600, cohort="online", **kw),
            cascade_probability=0.0, adapter=dict(DEFAULT_ADAPTER, ram_upper=1400))
        self.assertEqual(candidate.cascade_events, [])

    def test_non_cascade_tasks_are_familiar_in_a_mixed_stream(self):
        workload = self._generate(
            lambda **kw: Protocol022CascadeBWGD2(1.0, 1.5, 600, cohort="online", **kw),
            cascade_probability=0.25, adapter=dict(DEFAULT_ADAPTER, ram_upper=1400))
        flagged = workload.cascade_task_flags()
        cascade_ids = {c for c, flag in flagged.items() if flag}
        self.assertGreater(len(flagged), 5, "arrival stream is too short to test")
        self.assertGreater(len(cascade_ids), 0, "no cascade task was drawn")
        self.assertLess(len(cascade_ids), len(flagged), "every task cascaded")
        for cid, is_cascade in flagged.items():
            max_cpu = max(workload.createdContainers[cid][2].ips_list)
            max_ram = max(workload.createdContainers[cid][3].size_list)
            if not is_cascade:
                self.assertLessEqual(max_cpu, DEFAULT_ADAPTER["cpu_upper"])
                self.assertLessEqual(max_ram, 1400.0)
            else:
                self.assertGreaterEqual(max_cpu, CASCADE_V2["cpu_burst_floor"])


class DeterminismTests(unittest.TestCase):
    """T-AUDIT-06 — reproducible envelopes and event registry."""

    def _run(self, mechanism_seed, probability=0.3, calls=6):
        import random
        random.seed(99)
        np.random.seed(99)
        workload = Protocol022CascadeBWGD2(
            1.0, 1.5, 600, cohort="online",
            adapter=dict(DEFAULT_ADAPTER, ram_upper=1400),
            cascade_probability=probability, mechanism_seed=mechanism_seed)
        workload.creation_id = 0
        for interval in range(calls):
            workload.generateNewContainers(interval)
        registry = [
            {k: v for k, v in event.items() if k not in ("probability_draw",)}
            for event in workload.cascade_events]
        return workload, registry

    def test_same_seed_is_byte_equivalent(self):
        first, registry_a = self._run(MECHANISM_SEED_DEV)
        second, registry_b = self._run(MECHANISM_SEED_DEV)
        self.assertEqual(registry_a, registry_b)
        self.assertGreater(len(registry_a), 0)
        for (_, _, ips_a, ram_a, _), (_, _, ips_b, ram_b, _) in zip(
                first.createdContainers, second.createdContainers):
            self.assertEqual(list(ips_a.ips_list), list(ips_b.ips_list))
            self.assertEqual(list(ram_a.size_list), list(ram_b.size_list))

    def test_different_mechanism_seed_changes_the_mechanism(self):
        _, registry_a = self._run(MECHANISM_SEED_DEV)
        _, registry_b = self._run(22023)
        self.assertNotEqual(registry_a, registry_b)

    def test_envelope_is_a_pure_function_of_seed_and_creation_id(self):
        workload, _ = self._run(MECHANISM_SEED_DEV)
        for cid in list(workload.cascade_task_flags())[:20]:
            first = workload._envelope(cid)
            second = workload._envelope(cid)
            self.assertEqual(first, second)


class RegisteredPhysicsTests(unittest.TestCase):
    """The runtime regime must not drift from the registration."""

    def _workload(self, **cascade):
        return Protocol022CascadeBWGD2(
            1.0, 1.5, 600, cohort="online",
            adapter=dict(DEFAULT_ADAPTER, ram_upper=1400),
            cascade_probability=0.25, cascade=cascade or None)

    def test_registered_physics_holds(self):
        self.assertTrue(self._workload().assert_registered_physics())

    def test_drift_is_detected(self):
        drifted = {k: v for k, v in CASCADE_V2.items()}
        drifted["cpu_burst_floor"] = 5000.0
        with self.assertRaises(AssertionError):
            Protocol022CascadeBWGD2(
                1.0, 1.5, 600, cohort="online",
                adapter=dict(DEFAULT_ADAPTER, ram_upper=1400),
                cascade_probability=0.25, cascade=drifted).assert_registered_physics()

    def test_cascade_v2_differs_from_v1_only_in_seed_and_registration(self):
        physics = ("cpu_duration", "cpu_to_ram_lag", "cpu_to_disk_lag",
                   "ram_duration", "disk_duration", "cpu_burst_mult",
                   "cpu_burst_floor", "cpu_burst_upper", "ram_ramp_peak_mult",
                   "ram_target_floor", "ram_burst_upper", "disk_retained_peak",
                   "disk_retained_cap", "observable_history")
        for key in physics:
            self.assertEqual(CASCADE_V1[key], CASCADE_V2[key],
                             "cascade_v2 must keep the P21 physical envelope: %s" % key)
        self.assertNotEqual(CASCADE_V1["mechanism_seed"],
                            CASCADE_V2["mechanism_seed"])

    def test_onset_threshold_sits_between_familiar_and_burst(self):
        self.assertGreater(CASCADE_V2["onset_tau_cpu"],
                           DEFAULT_ADAPTER["cpu_upper"])
        self.assertLess(CASCADE_V2["onset_tau_cpu"],
                        CASCADE_V2["cpu_burst_floor"])

    def test_onset_definition_is_task_level(self):
        definition = Protocol022CascadeBWGD2.onset_definition()
        self.assertIn("creation_id", definition["unit"])
        self.assertIn("never the host aggregate", definition["source"])


class RegisteredOnsetDetectorTests(unittest.TestCase):
    """The registered envelope must be detectable, and by this rule only.

    P22-01: the envelope is admitted at the familiar CPU level and bursts at
    age 1 (forced by ``getPlacementPossible`` evaluating demand at admission,
    P21-01).  A ``CPU(t) - median(CPU[t-4:t]) >= tau`` detector therefore has no
    pre-onset history: for every interior burst interval the window is already
    the burst.  On the real first candidate it produced max delta_cpu = -4400
    and ZERO onsets while the task genuinely reached 4400-5200 CPU.
    """

    def setUp(self):
        # admitted familiar at age 0, bursting from age 1 (registered shape)
        cpu = np.array([1860.0, 4700.0, 4600.0, 4500.0, 4450.0,
                        1500.0, 1400.0, 1860.0, 1200.0, 1000.0], dtype=float)
        ram = np.array([1400.0, 1400.0, 1400.0, 1400.0, 1400.0,
                        4500.0, 4700.0, 4600.0, 1400.0, 1400.0], dtype=float)
        disk = np.array([3000.0, 3000.0, 3000.0, 3000.0, 3000.0,
                         3000.0, 3000.0, 3000.0, 3000.0, 3000.0], dtype=float)
        disk = np.concatenate([disk[:8], np.full(10, 18000.0)])[:10]
        self.timeline = core.build_task_timeline(*rows_from_tasks({
            0: {"slot": 0, "start": 0, "cpu": cpu, "ram": ram, "disk": disk,
                "hosts": np.zeros(10, int)},
        }))

    def test_registered_envelope_produces_exactly_one_onset(self):
        events = core.onset_events(self.timeline)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["index"], 1)
        self.assertEqual(events[0]["age_onset"], 1)
        self.assertGreaterEqual(events[0]["cpu_onset"],
                                CASCADE_V2["cpu_burst_floor"])

    def test_baseline_falls_back_to_the_familiar_level(self):
        """The task was born inside the burst, so it has no pre-onset rows."""
        events = core.onset_events(self.timeline)
        self.assertEqual(events[0]["baseline_source"],
                         "observed pre-onset median")
        self.assertEqual(events[0]["cpu_baseline"], 1860.0)

    def test_median_baseline_form_is_blind_to_this_envelope(self):
        """Regression guard: the P21-style jump detector finds nothing here."""
        cpu = self.timeline[0]["cpu"]
        deltas = core.delta_cpu(cpu, baseline_lag=core.ONSET_BASELINE_LAG)
        self.assertEqual(int(np.nansum(deltas >= core.ONSET_TAU_CPU)), 0)
        self.assertLess(float(np.nanmax(deltas)), 0.0,
                        "the only large jump after the burst is its release")

    def test_ram_response_uses_the_registered_lag_window(self):
        events = core.onset_events(self.timeline)
        rows = core.event_responses(self.timeline, events)
        # baseline 1400, window is [t0+4, t0+14) -> contains the 4700 peak
        self.assertGreater(rows[0]["ram_response"], 3000.0)
        self.assertIn(rows[0]["ram_time_to_peak"], (5, 6, 7))

    def test_a_task_that_never_reaches_the_threshold_has_no_onset(self):
        familiar_cpu = np.full(20, 1860.0)
        timeline = core.build_task_timeline(*rows_from_tasks({
            0: {"slot": 0, "start": 0, "cpu": familiar_cpu,
                "ram": np.full(20, 1400.0), "disk": np.full(20, 9000.0),
                "hosts": np.zeros(20, int)},
        }))
        self.assertEqual(core.onset_events(timeline), [])

    def test_sustained_burst_counts_once(self):
        cpu = np.concatenate([np.full(3, 1000.0), np.full(15, 4800.0),
                              np.full(5, 1000.0)])
        timeline = core.build_task_timeline(*rows_from_tasks({
            0: {"slot": 0, "start": 0, "cpu": cpu,
                "ram": np.full(cpu.size, 1400.0),
                "disk": np.full(cpu.size, 3000.0),
                "hosts": np.zeros(cpu.size, int)},
        }))
        events = core.onset_events(timeline)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["index"], 3)

    def test_threshold_is_above_every_offline_per_task_cpu(self):
        """Measured offline value (offline_reference_v2.json), not assumed."""
        reference = az.AUDIT / "offline_reference_v2.json"
        if not reference.is_file():
            self.skipTest("offline reference not built yet")
        payload = json.loads(reference.read_text(encoding="utf8"))
        maxima = payload["familiar_maxima"]
        self.assertLess(maxima["pooled"]["cpu"]["max"], core.ONSET_TAU_CPU)
        self.assertTrue(maxima["onset_tau_above_every_offline_task_cpu"])
        self.assertEqual(
            payload["registered_scale"]["task_level"]["ram_response"]["n"], 0,
            "the registered scale must yield zero offline onsets; that is the "
            "mechanism-novelty evidence, not a broken reference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
