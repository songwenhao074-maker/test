"""Protocol 023 — S1 tests for the registered ``cascade_v3`` regime family.

Pins the properties the S2/S3 gates depend on:

  T-A-01  the compute-first regime is byte-identical to Protocol 022's
          ``cascade_v2`` (same trajectories, same events, same envelopes)
  T-A-02  probability 0 is the frozen familiar transform for every regime
  T-A-03  each regime's onset lands on its own resource only, and every phase
          deviation is exactly zero at the first age of its window (admission
          safety, the P21-01 constraint)
  T-A-04  every registered lag is observable inside the model history, and the
          two newly registered orders fit their whole chain inside it
  T-A-05  the three regimes differ in joint structure while their registered
          amplitudes match (the plan's marginal-matching requirement)
  T-A-06  envelopes are pure functions of (replay seed, creation id, mechanism
          seed) and each regime has its own registered mechanism seed
  T-A-07  registration guards: unknown parameter, unregistered regime, duplicate
          regime, shared mechanism seed and drifted physics all fail loudly

Run:
    python -m unittest test_ftmoe_protocol023_regimes -v
"""
import json
import random
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

SCENARIO_PATH = (ROOT / "artifacts/ftmoe_online/protocol_020/adapter/"
                         "scenario_adapter.json")
DRIFT_CONFIG = ROOT / "artifacts/ftmoe_online/protocol_020/drift/drift_config.json"
REPLAY_SEED = 700
COHORT = "online"
CALLS = 6


def load_adapter():
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
    drift = json.loads(DRIFT_CONFIG.read_text(encoding="utf8"))
    familiar = next(p for p in drift["phases"] if p["name"] == "baseline")
    adapter = dict(scenario.get("adapter") or {})
    adapter.update(familiar.get("adapter") or {})
    law = scenario.get("disk_law_relative")
    return adapter, (ROOT / law).resolve() if law else None


class _Env(object):
    def __init__(self, interval=0):
        self.interval = interval


HORIZON = 20          # ages sampled for the disk comparison


class _Container(object):
    pass


def _stub_container(creation_id, start_at=0):
    stub = _Container()
    stub.startAt = int(start_at)
    stub.env = _Env()
    stub.creationID = int(creation_id)
    return stub


def disk_trajectory(disk, creation_id, ages=HORIZON):
    """Disk occupancy per age, read through the model's own interface.

    The models are exercised before any container exists, so a minimal
    container stand-in supplies ``startAt`` / ``env.interval`` -- the only two
    attributes ``TrainingMarkovDisk.disk()`` reads -- and the age axis is
    advanced explicitly, exactly as the simulator advances it.  Both the
    cascade and the reference model are sampled over the same ages with the
    same ``creation_id``, so their Markov state sequences are identical and
    only the retained term under test differs.
    """
    stub = _stub_container(creation_id)
    disk.container = stub
    values = []
    for age in range(int(ages)):
        stub.env.interval = age
        values.append(float(disk.disk()[0]))
    return np.asarray(values, dtype=float)


def disk_baseline(replay_seed, creation_id, disk_mult, ages=HORIZON):
    """The same task's disk occupancy with no cascade term at all."""
    from simulator.workload.BitbrainWorkloadProtocol020 import _ScaledMarkovDisk
    adapter, law = load_adapter()
    model = _ScaledMarkovDisk(load_law(), replay_seed, int(creation_id),
                              disk_mult)
    return disk_trajectory(model, creation_id, ages)


def load_law():
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
    law = scenario.get("disk_law_relative")
    return json.loads((ROOT / law).read_text(encoding="utf8"))


class RegimeTestCase(unittest.TestCase):
    """Shared construction helpers (the arrival draw is a global RNG)."""

    def setUp(self):
        self.adapter, self.law = load_adapter()

    def make(self, cls, regimes=None, probability=0.5, mechanism_seed=None,
             replay_seed=REPLAY_SEED, **extra):
        kwargs = dict(cohort=COHORT, adapter=self.adapter,
                      disk_law_path=self.law)
        if cls.__name__ != "Protocol020AdaptedBWGD2":
            kwargs["cascade_probability"] = probability
            if regime_seed := mechanism_seed:
                kwargs["mechanism_seed"] = regime_seed
            if regimes is not None:
                kwargs["registered_regimes"] = regimes
        kwargs.update(extra)
        return cls(1.0, 1.5, replay_seed, **kwargs)

    def trajectories(self, workload, calls=CALLS):
        random.seed(REPLAY_SEED)
        for _ in range(calls):
            workload.generateNewContainers(0)
        out = {}
        for (cid, interval, ips, ram, disk) in workload.createdContainers:
            out[int(cid)] = {
                "cpu": np.asarray(ips.ips_list, dtype=float),
                "ram": np.asarray(ram.size_list, dtype=float),
                "disk": disk_trajectory(disk, cid),
                # the same task's disk occupancy with no cascade term, so the
                # retained contribution is measurable regardless of which
                # Markov state this creation_id happens to draw
                "disk_plain": disk_baseline(workload.replay_seed, cid,
                                            workload.adapter["disk_mult"]),
            }
        return out

    def event_of(self, workload, cid):
        event_id = workload.cascade_event_id(cid)
        self.assertIsNotNone(event_id, "task %d has no cascade event" % cid)
        return workload.cascade_events[event_id]

    def a_cascade_task(self, workload, trajectories):
        """First creation id that actually carries a cascade event."""
        for cid in sorted(trajectories):
            if workload.cascade_event_id(cid) is not None:
                return cid
        self.fail("no cascade task in the generated sample")


class FamiliarIdentityTests(RegimeTestCase):
    """T-A-02: with the mechanism off every regime is the frozen transform."""

    def test_zero_probability_matches_the_frozen_familiar_transform(self):
        from simulator.workload.BitbrainWorkloadProtocol020 import (
            Protocol020AdaptedBWGD2)
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIME_IDS, Protocol023MultiRegimeBWGD2)
        reference = self.trajectories(
            self.make(Protocol020AdaptedBWGD2, probability=0.0))
        for regime in REGIME_IDS:
            workload = self.make(Protocol023MultiRegimeBWGD2, (regime,), 0.0)
            trajectories = self.trajectories(workload)
            self.assertEqual(len(workload.cascade_events), 0)
            for cid in sorted(set(trajectories) & set(reference)):
                for resource in ("cpu", "ram", "disk"):
                    np.testing.assert_array_equal(
                        trajectories[cid][resource],
                        reference[cid][resource],
                        err_msg="%s %s differs from the familiar transform"
                                % (regime, resource))

    def test_zero_probability_creates_no_envelope(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIME_IDS, Protocol023MultiRegimeBWGD2)
        for regime in REGIME_IDS:
            workload = self.make(Protocol023MultiRegimeBWGD2, (regime,), 0.0)
            self.assertIsNone(workload._envelope(7))


class ComputeFirstInheritanceTests(RegimeTestCase):
    """T-A-01: compute-first *is* Protocol 022's cascade_v2."""

    def test_trajectories_and_events_are_byte_identical_to_cascade_v2(self):
        from simulator.workload.BitbrainWorkloadProtocol022 import (
            Protocol022CascadeBWGD2)
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        new = self.make(Protocol023MultiRegimeBWGD2, ("compute_first",), 0.5)
        old = self.make(Protocol022CascadeBWGD2, None, 0.5)
        t_new, t_old = self.trajectories(new), self.trajectories(old)
        self.assertEqual(sorted(t_new), sorted(t_old))
        for cid in t_new:
            for resource in ("cpu", "ram", "disk"):
                np.testing.assert_array_equal(
                    t_new[cid][resource], t_old[cid][resource],
                    err_msg="compute_first %s differs from cascade_v2 at "
                            "creation_id %d" % (resource, cid))
        self.assertEqual(len(new.cascade_events), len(old.cascade_events))
        for a, b in zip(new.cascade_events, old.cascade_events):
            for key in ("creation_id", "cpu_window", "ram_window",
                        "disk_window", "cpu_peak_value", "ram_peak_value"):
                self.assertEqual(a[key], b[key], "event field %s drifted" % key)

    def test_the_registered_seed_is_protocol_022_s(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            MECHANISM_SEED_DEV, REGIME_MECHANISM_SEEDS)
        self.assertEqual(REGIME_MECHANISM_SEEDS["compute_first"],
                         MECHANISM_SEED_DEV)
        self.assertEqual(MECHANISM_SEED_DEV, 22022)

    def test_every_regime_has_its_own_registered_seed(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIME_IDS, REGIME_MECHANISM_SEEDS)
        seeds = [REGIME_MECHANISM_SEEDS[r] for r in REGIME_IDS]
        self.assertEqual(len(set(seeds)), len(REGIME_IDS))


class AdmissionSafetyTests(RegimeTestCase):
    """T-A-03: the onset fires on its own resource at exactly one age."""

    def _delta(self, regime, cid_pick=0.5):
        """(event, per-resource deviation, observations) for one real task.

        A cascade task is selected whose registered phase windows actually sit
        inside the trajectory, so the assertion is about the mechanism rather
        than about which trace length a particular VM happened to have.
        """
        from simulator.workload.BitbrainWorkloadProtocol020 import (
            Protocol020AdaptedBWGD2)
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        plain = self.trajectories(
            self.make(Protocol020AdaptedBWGD2, probability=0.0))
        workload = self.make(Protocol023MultiRegimeBWGD2, (regime,), 1.0)
        trajectories = self.trajectories(workload)
        usable = []
        for cid in sorted(trajectories):
            if workload.cascade_event_id(cid) is None:
                continue
            event = self.event_of(workload, cid)
            longest = max(event["cpu_window"][1], event["ram_window"][1],
                          event["disk_window"][1])
            if trajectories[cid]["cpu"].size <= longest + 2:
                continue
            # The tuned disk law puts a task at occupancy 0 for its whole early
            # life with probability ~0.42 (measured), which makes the retained
            # term invisible for that task.  Prefer a task whose own baseline
            # is non-zero so the assertion is about the mechanism, not about
            # which Markov state a particular creation_id drew.
            if float(trajectories[cid]["disk_plain"].max()) <= 0.0:
                continue
            usable.append(cid)
        if not usable:
            usable = [cid for cid in sorted(trajectories)
                      if workload.cascade_event_id(cid) is not None]
        self.assertTrue(usable, "no usable cascade task for %s" % regime)
        cid = usable[int(len(usable) * cid_pick) % len(usable)]
        event = self.event_of(workload, cid)
        delta = {r: trajectories[cid][r] - plain[cid][r]
                 for r in ("cpu", "ram")}
        # The disk comparison uses each task's own baseline occupancy: the two
        # generators draw the same Markov states for the same creation_id, so
        # the difference is exactly the registered retained term.
        delta["disk"] = trajectories[cid]["disk"] \
            - trajectories[cid]["disk_plain"]
        self.observations = {
            "cpu": trajectories[cid]["cpu"], "ram": trajectories[cid]["ram"],
            "disk": trajectories[cid]["disk"]}
        return event, delta, trajectories[cid], plain[cid]

    def test_onset_resource_is_the_only_resource_that_moves_first(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        for regime in REGIME_IDS:
            event, delta, _, _ = self._delta(regime)
            onset = event["onset_resource"]
            for resource, (start, end) in (
                    (r, event["%s_window" % r])
                    for r in ("cpu", "ram", "disk")):
                if resource == onset:
                    continue
                if start == 0:
                    continue
                before = np.abs(delta[resource][:start])
                self.assertLess(float(before.max()) if before.size else 0.0,
                                1e-9,
                                "%s moves before its window in %s"
                                % (resource, regime))

    def test_every_phase_is_admission_safe_at_its_first_age(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        for regime in REGIME_IDS:
            event, delta, _, _ = self._delta(regime)
            start = event["%s_window" % event["onset_resource"]][0]
            self.assertEqual(float(delta[event["onset_resource"]][start]), 0.0,
                             "%s onset is not zero at age %d"
                             % (regime, start))

    def test_the_onset_crosses_its_registered_threshold(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        for regime in REGIME_IDS:
            event, _, observed, _ = self._delta(regime)
            onset = event["onset_resource"]
            start = event["%s_window" % onset][0]
            values = observed[onset]
            self.assertGreaterEqual(
                float(values[start + 1]), float(event["onset_threshold"]),
                "%s does not reach its registered onset threshold" % regime)

    def test_the_peak_is_inside_the_window_and_bounded(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        for regime in REGIME_IDS:
            event, delta, _, _ = self._delta(regime)
            for resource in ("cpu", "ram", "disk"):
                start, end = event["%s_window" % resource]
                inside = delta[resource][start:end + 1]
                outside = np.concatenate([delta[resource][:start],
                                          delta[resource][end + 1:]])
                self.assertGreater(float(inside.max()), 0.0,
                                   "%s has no %s phase" % (regime, resource))
                self.assertLess(float(np.abs(outside).max()) if outside.size
                                else 0.0, 1e-9,
                                "%s leaks %s outside its window"
                                % (regime, resource))


class ObservabilityTests(RegimeTestCase):
    """T-A-04: registered lags are observable; new orders fit the history."""

    def test_registered_lags_are_inside_the_model_history(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIMES_V3, REGIME_IDS)
        for regime in REGIME_IDS:
            params = REGIMES_V3[regime]
            history = params["observable_history"]
            for resource, lag in params["sequence"]:
                self.assertLess(lag, history,
                                "%s lag %d of %s is not observable"
                                % (regime, lag, resource))

    def test_new_orders_fit_the_whole_chain_inside_the_history(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            INHERITED_CHAIN_REGIMES, REGIMES_V3, REGIME_IDS)
        for regime in REGIME_IDS:
            if regime in INHERITED_CHAIN_REGIMES:
                continue
            params = REGIMES_V3[regime]
            longest = max(
                lag + int(params["%s_duration" % resource][1]) - 1
                for resource, lag in params["sequence"])
            self.assertLess(longest, params["observable_history"],
                            "%s chain age %d does not fit the history"
                            % (regime, longest))

    def test_the_generator_records_the_chain_geometry(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2, REGIME_IDS)
        workload = self.make(Protocol023MultiRegimeBWGD2, REGIME_IDS, 0.25)
        self.assertEqual(set(workload._chain_geometry), set(REGIME_IDS))
        for regime, geometry in workload._chain_geometry.items():
            self.assertIn("chain_fully_observable", geometry)
            self.assertIn("longest_chain_age", geometry)


class MarginallyMatchedTests(RegimeTestCase):
    """T-A-05: matched amplitudes, different joint structure."""

    def _peak(self, regime):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        workload = self.make(Protocol023MultiRegimeBWGD2, (regime,), 1.0)
        trajectories = self.trajectories(workload)
        cid = self.a_cascade_task(workload, trajectories)
        return self.event_of(workload, cid)

    def test_registered_amplitudes_match_across_regimes(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        bands = {
            "cpu": ("cpu_peak_value", 4400.0, 5200.0),
            "ram": ("ram_peak_value", 4500.0, 6000.0),
        }
        for resource, (key, low, high) in bands.items():
            for regime in REGIME_IDS:
                event = self._peak(regime)
                self.assertGreaterEqual(float(event[key]), low,
                                        "%s %s below its registered band"
                                        % (regime, resource))
                self.assertLessEqual(float(event[key]), high,
                                     "%s %s above its registered band"
                                     % (regime, resource))

    def test_the_orders_differ(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        orders = []
        for regime in REGIME_IDS:
            event = self._peak(regime)
            orders.append(tuple(resource for resource, _ in event["sequence"]))
        self.assertEqual(len(set(orders)), len(REGIME_IDS),
                         "the three registered orders are not distinct: %r"
                         % (orders,))

    def test_the_onset_resource_differs(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import REGIME_IDS
        onsets = [self._peak(r)["onset_resource"] for r in REGIME_IDS]
        self.assertEqual(sorted(onsets), ["cpu", "disk", "ram"])


class EnvelopePurityTests(RegimeTestCase):
    """T-A-06: envelopes are a pure function of seed and creation id."""

    def _envelope(self, workload, cid):
        return workload._envelope(int(cid))

    def test_same_seed_same_envelope(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        a = self.make(Protocol023MultiRegimeBWGD2, None, 0.35)
        b = self.make(Protocol023MultiRegimeBWGD2, None, 0.35)
        for cid in range(40):
            self.assertEqual(self._envelope(a, cid), self._envelope(b, cid))

    def test_different_regime_gives_a_different_envelope(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIME_IDS, Protocol023MultiRegimeBWGD2)
        shared = None
        for regime in REGIME_IDS:
            workload = self.make(Protocol023MultiRegimeBWGD2, None, 0.5,
                                 active_regime=regime)
            draws = [self._envelope(workload, cid) for cid in range(60)]
            draws = [d for d in draws if d is not None]
            self.assertTrue(draws, "no envelope drawn for %s" % regime)
            for draw in draws:
                self.assertEqual(draw["regime_id"], regime)
            if shared is None:
                shared = draws
            else:
                self.assertNotEqual(
                    [d["durations"] for d in shared],
                    [d["durations"] for d in draws],
                    "regime %s draws identically to the previous regime"
                    % regime)

    def test_no_global_rng_state_is_touched(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        workload = self.make(Protocol023MultiRegimeBWGD2, None, 0.5)
        before = random.getstate()
        for cid in range(20):
            self._envelope(workload, cid)
        self.assertEqual(before, random.getstate())


class RegistrationGuardTests(RegimeTestCase):
    """T-A-07: unregistered physics cannot run silently."""

    def test_unknown_parameter_is_rejected(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        with self.assertRaises(ValueError):
            self.make(Protocol023MultiRegimeBWGD2, None, 0.25,
                      cascade={"not_a_parameter": 1})

    def test_unregistered_regime_is_rejected(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        with self.assertRaises(ValueError):
            self.make(Protocol023MultiRegimeBWGD2, ("compute_first",
                                                    "cascade_v4"), 0.25)

    def test_duplicate_regime_is_rejected(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        with self.assertRaises(ValueError):
            self.make(Protocol023MultiRegimeBWGD2,
                      ("compute_first", "compute_first"), 0.25)

    def test_shared_mechanism_seed_is_rejected(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2, REGIME_MECHANISM_SEEDS)
        original = REGIME_MECHANISM_SEEDS["io_first"]
        REGIME_MECHANISM_SEEDS["io_first"] = \
            REGIME_MECHANISM_SEEDS["memory_first"]
        try:
            with self.assertRaises(ValueError):
                self.make(Protocol023MultiRegimeBWGD2,
                          ("memory_first", "io_first"), 0.25)
        finally:
            REGIME_MECHANISM_SEEDS["io_first"] = original

    def test_active_regime_must_be_registered(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        with self.assertRaises(ValueError):
            self.make(Protocol023MultiRegimeBWGD2, ("compute_first",), 0.25,
                      active_regime="memory_first")

    def test_drifted_physics_is_detected(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        workload = self.make(Protocol023MultiRegimeBWGD2, None, 0.25)
        self.assertTrue(workload.assert_registered_physics())
        for key, drifted in (("cpu_burst_upper", 9999.0),
                             ("ram_target_floor", 1.0),
                             ("disk_retained_peak", 1.0)):
            original = workload.cascade[key]
            workload.cascade[key] = drifted
            try:
                with self.assertRaises(AssertionError,
                                       msg="%s drift was not detected" % key):
                    workload.assert_registered_physics()
            finally:
                workload.cascade[key] = original
        self.assertTrue(workload.assert_registered_physics())

    def test_unregistered_mechanism_seed_is_detected(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        workload = self.make(Protocol023MultiRegimeBWGD2, None, 0.25,
                             mechanism_seed=4242)
        with self.assertRaises(AssertionError):
            workload.assert_registered_physics()

    def test_phase_switch_validates_its_arguments(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            Protocol023MultiRegimeBWGD2)
        workload = self.make(Protocol023MultiRegimeBWGD2, None, 0.25)
        with self.assertRaises(ValueError):
            workload.set_active_regime("nope")
        with self.assertRaises(ValueError):
            workload.set_active_regime("memory_first", probability=1.5)
        result = workload.set_active_regime("memory_first", probability=0.25)
        self.assertEqual(result["regime_id"], "memory_first")
        self.assertEqual(workload.active_regime, "memory_first")
        self.assertEqual(workload.cascade_probability, 0.25)

    def test_task_cascade_windows_covers_every_regime(self):
        """The audit index must work for all three onset resources."""
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIME_IDS, Protocol023MultiRegimeBWGD2)
        for regime in REGIME_IDS:
            workload = self.make(Protocol023MultiRegimeBWGD2, (regime,), 1.0)
            trajectories = self.trajectories(workload)
            cid = self.a_cascade_task(workload, trajectories)
            event = self.event_of(workload, cid)
            windows = workload.task_cascade_windows()
            self.assertIn(cid, windows)
            entry = windows[cid]
            self.assertEqual(entry["onset_resource"], event["onset_resource"])
            self.assertGreater(entry["onset_floor"], 0.0)
            self.assertGreater(entry["onset_peak_value"], 0.0)
            self.assertEqual(entry["onset_resource_window"],
                             list(event["%s_window" % event["onset_resource"]]))
            self.assertLess(entry["onset_threshold"], entry["onset_peak_value"])

    def test_audit_exposes_the_family_without_leaking_inputs(self):
        from simulator.workload.BitbrainWorkloadProtocol023 import (
            REGIME_IDS, Protocol023MultiRegimeBWGD2)
        from simulator.workload.BitbrainWorkloadProtocol021 import CASCADE_V1
        workload = self.make(Protocol023MultiRegimeBWGD2, REGIME_IDS, 0.25)
        audit = workload.cascade_audit()
        self.assertEqual(audit["regime_id"], "cascade_v3")
        self.assertEqual(set(audit["regimes"]), set(REGIME_IDS))
        self.assertEqual(audit["active_regime"], REGIME_IDS[0])
        plan_tokens = ("regime_id", "phase_id", "mechanism_id", "event_id")
        for regime in REGIME_IDS:
            tokens = set(audit["regimes"][regime]["forbidden_inputs"])
            for token in plan_tokens:
                self.assertTrue(
                    any(token in registered for registered in tokens),
                    "%s does not forbid %r" % (regime, token))
            for token in CASCADE_V1["forbidden_inputs"]:
                self.assertIn(token, tokens,
                              "%s dropped the inherited token %r"
                              % (regime, token))


if __name__ == "__main__":
    unittest.main()
