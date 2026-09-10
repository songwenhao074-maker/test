"""Protocol 021 — cascade mechanism tests (data layer, no model involved).

Implements the data-layer rows of the protocol's required-test table (§34):

    01 Protocol020 regression   p=0 is byte-identical to Protocol020AdaptedBWGD2
    02 Cascade deterministic    same replay/creation/mechanism seed => same envelope
    03 Lag correctness          CPU ages [0,d), RAM starts at +4, Disk starts at +8
    04 No future leakage        later-window parameters cannot change earlier values
    05 Metadata exclusion       audit metadata never enters the demand arrays
    06 Source split             pilot cohort is disjoint from the P20 S6 train cohort
    07 Event bookkeeping        flags/ids consistent with the recorded events
    (08 event segmentation is exercised on a collected stream by the U4 analyzer)

The familiar demand profile used here is the Protocol-020 baseline phase
adapter (adapter ram_upper=1400 with capacity scales 1.0/1.0/0.9), which is
also what the pilot collector uses.

Run from the repository root:
    python test_ftmoe_protocol021_unseen.py
"""
import json
import random
import unittest
from pathlib import Path

import numpy as np

from simulator.workload.BitbrainWorkloadProtocol021 import _CascadeRetainedDisk

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"
SCENARIO_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json"
PILOT_COHORT = "online"
FAMILIAR_ADAPTER = {"ram_upper": 1400.0}
INTERVALS = 80


def scenario():
    cfg = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
    law = (ROOT / cfg["disk_law_relative"]).resolve()
    return cfg, law


class _StubEnv:
    def __init__(self, interval):
        self.interval = interval


class _StubContainer:
    """Minimal container stand-in for age-indexed model queries."""

    def __init__(self, start_at, interval):
        self.startAt = start_at
        self.env = _StubEnv(interval)


def familiar_adapter(cfg):
    merged = dict(cfg["adapter"])
    merged.update(FAMILIAR_ADAPTER)
    return merged


def build(cascade_probability, replay_seed=7, cascade=None, cohort=PILOT_COHORT,
          regime_id=None):
    from simulator.workload.BitbrainWorkloadProtocol021 import \
        Protocol021CascadeBWGD2
    cfg, law = scenario()
    overrides = dict(cascade or {})
    if regime_id is not None:
        overrides["regime_id"] = regime_id
    return Protocol021CascadeBWGD2(
        cfg["arrival_mean"], cfg.get("arrival_sigma", 1.5), replay_seed,
        cohort=cohort, adapter=familiar_adapter(cfg), disk_law_path=law,
        cascade=overrides, cascade_probability=cascade_probability)


def build_p20(replay_seed=7, cohort=PILOT_COHORT):
    from simulator.workload.BitbrainWorkloadProtocol020 import \
        Protocol020AdaptedBWGD2
    cfg, law = scenario()
    return Protocol020AdaptedBWGD2(cfg["arrival_mean"],
                                   cfg.get("arrival_sigma", 1.5), replay_seed,
                                   cohort=cohort, adapter=familiar_adapter(cfg),
                                   disk_law_path=law)


def collect(workload, intervals=INTERVALS, seed=1234):
    """Deterministically drive generateNewContainers without the simulator."""
    random.seed(seed)
    np.random.seed(seed)
    for t in range(intervals):
        workload.generateNewContainers(t)
    return workload


def cpu_of(workload, index):
    return np.asarray(workload.createdContainers[index][2].ips_list, dtype=float)


def ram_of(workload, index):
    return np.asarray(workload.createdContainers[index][3].size_list, dtype=float)


def disk_model(workload, index):
    return workload.createdContainers[index][4]


class Protocol021DataLayerTests(unittest.TestCase):

    # ---- 01 -----------------------------------------------------------
    def test_01_protocol020_regression_at_zero_probability(self):
        p21 = collect(build(0.0))
        p20 = collect(build_p20())
        self.assertEqual(len(p21.createdContainers), len(p20.createdContainers))
        self.assertEqual(p21.cascade_events, [])
        for i in range(len(p20.createdContainers)):
            cid21, int21, ips21, ram21, disk21 = p21.createdContainers[i]
            cid20, int20, ips20, ram20, disk20 = p20.createdContainers[i]
            self.assertEqual((cid21, int21), (cid20, int20), "identity differs at %d" % i)
            np.testing.assert_array_equal(cpu_of(p21, i), cpu_of(p20, i),
                                          err_msg="CPU differs at %d" % i)
            self.assertEqual(float(ips21.max_ips), float(ips20.max_ips))
            np.testing.assert_array_equal(ram_of(p21, i), ram_of(p20, i),
                                          err_msg="RAM differs at %d" % i)
            self.assertEqual(type(disk21).__name__, type(disk20).__name__)
            np.testing.assert_array_equal(disk21.values, disk20.values)
            self.assertEqual(disk21.states, disk20.states)

    # ---- 02 -----------------------------------------------------------
    def test_02_cascade_envelope_is_deterministic(self):
        a = collect(build(0.35))
        b = collect(build(0.35))
        self.assertEqual(len(a.cascade_events), len(b.cascade_events))
        self.assertGreater(len(a.cascade_events), 0, "no cascade task drawn")
        for ea, eb in zip(a.cascade_events, b.cascade_events):
            self.assertEqual(ea, eb)
        self.assertEqual(a.cascade_task_flags(), b.cascade_task_flags())
        for i in range(len(a.createdContainers)):
            np.testing.assert_array_equal(cpu_of(a, i), cpu_of(b, i))
            np.testing.assert_array_equal(ram_of(a, i), ram_of(b, i))

    def test_02b_different_mechanism_seed_changes_envelope(self):
        a = collect(build(0.35))
        c = build(0.35)
        c.mechanism_seed = 999
        collect(c)
        self.assertNotEqual([e["creation_id"] for e in a.cascade_events],
                            [e["creation_id"] for e in c.cascade_events])

    # ---- 03 -----------------------------------------------------------
    def test_03_lag_correctness(self):
        wl = collect(build(1.0))          # every task cascades
        params = wl.cascade
        factory = wl.adapter
        self.assertGreater(len(wl.cascade_events), 0)
        by_cid = {int(wl.createdContainers[i][0]): i
                  for i in range(len(wl.createdContainers))}
        for event in wl.cascade_events:
            i = by_cid[event["creation_id"]]
            cpu = cpu_of(wl, i)
            burst = cpu[:event["cpu_duration"]]
            self.assertGreaterEqual(event["cpu_duration"], params["cpu_duration"][0])
            self.assertLessEqual(event["cpu_duration"], params["cpu_duration"][1])
            self.assertGreater(burst.max(), factory["cpu_upper"],
                               "CPU burst never exceeded the familiar ceiling")
            # admission interval keeps the familiar demand (placeability), the
            # sustained part of the burst starts at age 1
            self.assertLessEqual(float(burst[0]), factory["cpu_upper"] + 1e-9,
                                 "admission interval is not at the familiar level")
            if burst.size > 1:
                self.assertGreaterEqual(float(burst[1:].min()),
                                        float(params["cpu_burst_floor"]) - 1e-9)
            self.assertLessEqual(float(burst.max()),
                                 float(params["cpu_burst_upper"]) + 1e-9)
            self.assertLessEqual(cpu[event["cpu_duration"]:].max(),
                                 factory["cpu_upper"] + 1e-9,
                                 "CPU elevated outside the registered burst window")

            # ---- RAM: onset at the registered lag, progressive, released ---
            ram = ram_of(wl, i)
            start = params["cpu_to_ram_lag"]
            duration = event["ram_duration"]
            self.assertEqual(event["ram_window"], [start, start + duration - 1])
            self.assertGreaterEqual(duration, params["ram_duration"][0])
            self.assertLessEqual(duration, params["ram_duration"][1])
            window = ram[start:start + duration]
            familiar = float(factory["ram_upper"])
            self.assertLessEqual(float(window[0]), familiar + 1e-9,
                                 "RAM window starts with an instantaneous jump")
            self.assertGreaterEqual(float(window.max()),
                                    float(params["ram_target_floor"]) - 1e-9,
                                    "RAM window never reached its target")
            self.assertLessEqual(float(window.max()),
                                 float(params["ram_burst_upper"]) + 1e-9)
            self.assertEqual(event["ram_peak_age"],
                             int(np.argmax(window)) + start)
            self.assertGreater(event["ram_peak_age"], start)
            # released again after the window
            self.assertLessEqual(float(ram[start + duration]), familiar + 1e-9)

            # ---- Disk: exact window, capped, cleaned back to plain Markov --
            disk = disk_model(wl, i)
            self.assertEqual(type(disk).__name__, "_CascadeRetainedDisk")
            dstart = params["cpu_to_disk_lag"]
            ddur = event["disk_duration"]
            self.assertEqual(event["disk_window"], [dstart, dstart + ddur - 1])
            self.assertEqual(int(disk.start_age), dstart)
            plain = _CascadeRetainedDisk(
                wl.disk_law, wl.replay_seed, event["creation_id"],
                disk.disk_mult, start_age=dstart, duration=ddur, peak=0.0,
                cap=float(params["disk_retained_cap"]))
            disk.allocContainer(_StubContainer(0, 0))
            plain.allocContainer(_StubContainer(0, 0))
            for age in range(0, dstart + ddur + 4):
                disk.container.env.interval = age
                plain.container.env.interval = age
                value, _, _ = disk.disk()
                reference, _, _ = plain.disk()
                self.assertLessEqual(value,
                                     float(params["disk_retained_cap"]) + 1e-9,
                                     "retained term exceeded its registered cap")
                if dstart <= age < dstart + ddur:
                    self.assertGreaterEqual(value, reference - 1e-9)
                else:
                    self.assertAlmostEqual(value, reference, places=9,
                                           msg="disk altered outside its window")
            inside = []
            for age in range(dstart, dstart + ddur):
                disk.container.env.interval = age
                inside.append(disk.disk()[0])
            self.assertGreater(max(inside), 0.0)
            self.assertLess(inside[-1], max(inside), "retained term is not cleaned")

    def test_03b_windows_are_observable_inside_history(self):
        wl = collect(build(0.35))
        params = wl.cascade
        history = params["observable_history"]
        self.assertLess(params["cpu_to_ram_lag"], history)
        self.assertLess(params["cpu_to_disk_lag"], history)
        self.assertEqual(history, 12)

    def test_03c_registered_lags_are_enforced(self):
        from simulator.workload.BitbrainWorkloadProtocol021 import \
            Protocol021CascadeBWGD2
        cfg, law = scenario()
        with self.assertRaises(ValueError):
            Protocol021CascadeBWGD2(cfg["arrival_mean"], 1.5, 7,
                                    cohort=PILOT_COHORT,
                                    adapter=familiar_adapter(cfg),
                                    disk_law_path=law,
                                    cascade={"cpu_to_ram_lag": 12})
        with self.assertRaises(ValueError):
            Protocol021CascadeBWGD2(cfg["arrival_mean"], 1.5, 7,
                                    cohort=PILOT_COHORT,
                                    adapter=familiar_adapter(cfg),
                                    disk_law_path=law,
                                    cascade_probability=1.5)

    # ---- 04 -----------------------------------------------------------
    def test_04_no_future_leakage(self):
        """Changing a late-window parameter must not change earlier values."""
        base = collect(build(1.0))
        changed = collect(build(1.0, cascade={"disk_retained_peak": 3000.0,
                                              "disk_duration": [6, 6]}))
        self.assertEqual(len(base.createdContainers), len(changed.createdContainers))
        for i in range(len(base.createdContainers)):
            np.testing.assert_array_equal(cpu_of(base, i), cpu_of(changed, i),
                                          err_msg="CPU changed at %d" % i)
            np.testing.assert_array_equal(ram_of(base, i), ram_of(changed, i),
                                          err_msg="RAM changed at %d" % i)

    def test_04b_changing_ram_target_does_not_touch_cpu(self):
        a = collect(build(1.0))
        b = collect(build(1.0, cascade={"ram_ramp_peak_mult": 3.0}))
        for i in range(len(a.createdContainers)):
            np.testing.assert_array_equal(cpu_of(a, i), cpu_of(b, i))

    # ---- 05 -----------------------------------------------------------
    def test_05_metadata_never_enters_demand_arrays(self):
        a = collect(build(0.35, regime_id="cascade_v1"))
        b = collect(build(0.35, regime_id="cascade_v1_renamed"))
        for i in range(len(a.createdContainers)):
            np.testing.assert_array_equal(cpu_of(a, i), cpu_of(b, i))
            np.testing.assert_array_equal(ram_of(a, i), ram_of(b, i))
        audit = a.cascade_audit()
        for name in ("regime_id", "cascade_task_flag", "cascade_event_id"):
            self.assertIn(name, a.cascade["forbidden_inputs"])
        json.dumps(audit)

    # ---- 06 -----------------------------------------------------------
    def test_06_source_split_is_disjoint(self):
        split = json.loads(SPLIT_PATH.read_text(encoding="utf8"))
        cohorts = {name: set(int(v) for v in ids)
                   for name, ids in split["cohorts"].items()}
        names = sorted(cohorts)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertEqual(cohorts[left] & cohorts[right], set(),
                                 "cohorts %s and %s overlap" % (left, right))
        self.assertEqual(cohorts[PILOT_COHORT] & cohorts["train"], set())
        self.assertEqual(cohorts[PILOT_COHORT] & cohorts["dev"], set())
        wl = collect(build(0.0, cohort=PILOT_COHORT), intervals=5)
        self.assertEqual(set(wl.possible_indices), cohorts[PILOT_COHORT])

    # ---- 07 -----------------------------------------------------------
    def test_07_event_bookkeeping(self):
        wl = collect(build(0.35))
        flags = wl.cascade_task_flags()
        self.assertEqual(sum(flags.values()), len(wl.cascade_events))
        for event in wl.cascade_events:
            self.assertEqual(wl.cascade_event_id(event["creation_id"]),
                             event["event_id"])
            self.assertIsNone(wl.cascade_event_id(-12345))
        self.assertLessEqual(len(wl.cascade_events), len(wl.createdContainers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
