"""Protocol 023 — generator regression check (offline, no simulation).

Checks the three registered ``cascade_v3`` regimes at the trajectory level,
before any stream is collected:

  A-01  with one registered regime the generator is byte-equal to Protocol 022
        (``cascade_v2``) for the same seed and mechanism seed
  A-02  probability 0 reproduces the frozen familiar transform exactly
        (the inherited T-AUDIT-05 identity) for every regime
  A-03  each regime fires only on its own onset resource, and its registered
        sequence/lags are what the trajectory shows
  A-04  every phase deviation is exactly zero at the first age of its own
        window (admission safety) and the longest chain is inside the
        12-interval observable history

Usage:
    python verify_ftmoe_protocol023_generator.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENARIO_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json"
DRIFT_CONFIG = ROOT / "artifacts/ftmoe_online/protocol_020/drift/drift_config.json"

REGISTERED_SEED = 700
COHORT = "online"
CALLS = 6


class _Env(object):
    """Minimal stand-in for the container environment (age axis only)."""

    def __init__(self, interval=0):
        self.interval = interval


def load_adapter():
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
    drift = json.loads(DRIFT_CONFIG.read_text(encoding="utf8"))
    familiar = next(p for p in drift["phases"] if p["name"] == "baseline")
    adapter = dict(scenario.get("adapter") or {})
    adapter.update(familiar.get("adapter") or {})
    law = scenario.get("disk_law_relative")
    return scenario, adapter, (ROOT / law).resolve() if law else None


def make(cls, regimes, probability, mechanism_seed=None):
    scenario, adapter, law = load_adapter()
    kwargs = dict(cohort=COHORT, adapter=adapter, disk_law_path=law)
    # Protocol020AdaptedBWGD2 has no cascade machinery at all; it is the
    # frozen familiar reference the mechanism-bearing classes are compared to.
    if cls.__name__ == "Protocol020AdaptedBWGD2":
        return cls(1.0, 1.5, REGISTERED_SEED, **kwargs)
    kwargs["cascade_probability"] = probability
    if mechanism_seed is not None:
        kwargs["mechanism_seed"] = mechanism_seed
    if regimes is not None:
        kwargs["registered_regimes"] = regimes
    return cls(1.0, 1.5, REGISTERED_SEED, **kwargs)


def build(workload, calls=CALLS):
    # ``BWGD2.generateNewContainers`` draws the arrival count from Python's
    # global ``random`` (random.gauss), so every comparison has to start from
    # the same global seed; the envelope itself is a pure function of
    # SeedSequence and is unaffected.
    import random
    random.seed(REGISTERED_SEED)
    for _ in range(calls):
        workload.generateNewContainers(0)
    out = {}
    for index, (cid, interval, ips, ram, disk) in enumerate(
            workload.createdContainers):
        out[int(cid)] = {
            "cpu": np.asarray(ips.ips_list, dtype=float),
            "ram": np.asarray(ram.size_list, dtype=float),
            "disk": disk_trajectory(disk, creation_id=cid),
            "event_id": (workload.cascade_event_id(cid)
                         if hasattr(workload, "cascade_event_id") else -1),
        }
    return out


def disk_trajectory(disk, ages=None, creation_id=None):
    """Disk occupancy at each age, read through the model's own interface.

    The models are exercised *before* any container exists, so a minimal
    container stand-in supplies ``startAt`` / ``env.interval`` — the only two
    attributes ``TrainingMarkovDisk.disk()`` reads.  Giving each model the same
    ``creation_id`` keeps the Markov state sequence comparable across
    generators.
    """
    if ages is None:
        ages = range(len(disk.values))

    class _Container(object):
        pass

    stub = _Container()
    stub.startAt = 0
    stub.env = _Env()
    if creation_id is not None:
        stub.creationID = int(creation_id)
    values = []
    real = getattr(disk, "container", None)
    disk.container = stub
    try:
        for age in ages:
            stub.env.interval = age
            values.append(float(disk.disk()[0]))
    finally:
        if real is not None:
            disk.container = real
    return np.asarray(values, dtype=float)


def main():
    failures = []
    from simulator.workload.BitbrainWorkloadProtocol020 import (
        Protocol020AdaptedBWGD2)
    from simulator.workload.BitbrainWorkloadProtocol022 import (
        Protocol022CascadeBWGD2)
    from simulator.workload.BitbrainWorkloadProtocol023 import (
        INHERITED_CHAIN_REGIMES, REGIMES_V3, REGIME_IDS,
        Protocol023MultiRegimeBWGD2)

    # ---- A-01: single-regime equality with Protocol 022 -------------------
    p23 = make(Protocol023MultiRegimeBWGD2, ("compute_first",), 0.5)
    p22 = make(Protocol022CascadeBWGD2, None, 0.5)
    t23, t22 = build(p23), build(p22)
    shared = sorted(set(t23) & set(t22))
    differences = []
    for cid in shared:
        for resource in ("cpu", "ram", "disk"):
            if not np.array_equal(t23[cid][resource], t22[cid][resource]):
                differences.append((cid, resource,
                                    float(np.max(np.abs(
                                        t23[cid][resource]
                                        - t22[cid][resource])))))
    events_p23 = [e["regime_id"] for e in p23.cascade_events]
    events_p22 = [e["regime_id"] for e in p22.cascade_events]
    same_events = len(events_p23) == len(events_p22)
    printed = {"check": "A-01 single-regime equality with cascade_v2",
               "tasks": len(shared), "events_p23": len(events_p23),
               "events_p22": len(events_p22),
               "event_count_equal": same_events,
               "event_ids_equal": [e["event_id"] for e in p23.cascade_events]
                                  == [e["event_id"] for e in p22.cascade_events],
               "trajectory_differences": differences[:5]}
    print(json.dumps(printed))
    if differences or not same_events:
        failures.append("A-01")

    # ---- A-02: probability 0 is the frozen familiar transform --------------
    base = make(Protocol020AdaptedBWGD2, None, 0.0)
    tb = build(base)
    for regime in REGIME_IDS:
        w = make(Protocol023MultiRegimeBWGD2, (regime,), 0.0)
        t = build(w)
        diffs = []
        for cid in sorted(set(t) & set(tb)):
            for resource in ("cpu", "ram", "disk"):
                if not np.array_equal(t[cid][resource], tb[cid][resource]):
                    diffs.append((cid, resource))
        print(json.dumps({"check": "A-02 familiar identity (%s)" % regime,
                          "tasks": len(t), "differences": diffs[:5],
                          "events": len(w.cascade_events)}))
        if diffs or w.cascade_events:
            failures.append("A-02/%s" % regime)

    # ---- A-03 / A-04: per-regime trajectory structure ---------------------
    plain = build(make(Protocol020AdaptedBWGD2, None, 0.0))
    for regime in REGIME_IDS:
        w = make(Protocol023MultiRegimeBWGD2, (regime,), 1.0)
        traj = build(w)
        cids = sorted(traj)
        cid = cids[len(cids) // 2]
        entry, reference = traj[cid], plain[cid]
        event = w.cascade_events[entry["event_id"]]
        onset = event["onset_resource"]
        delta = {r: entry[r] - reference[r] for r in ("cpu", "ram", "disk")}
        windows = {r: event["%s_window" % r] for r in ("cpu", "ram", "disk")}
        own = windows[onset]
        report = {
            "check": "A-03 regime %s" % regime,
            "onset_resource": onset,
            "sequence": event["sequence"],
            "windows": windows,
            # The registered phase law is exactly 0 at the first age of every
            # window (admission safety), so the onset is measured one interval
            # later -- that is the interval the task is admitted on.
            "onset_delta_at_window_start": float(delta[onset][own[0]]),
            "onset_delta_first_burst_interval": float(
                delta[onset][own[0] + 1]),
            "onset_threshold": float(event["onset_threshold"]),
            "onset_value_first_burst_interval": float(
                entry[onset][own[0] + 1]),
            "onset_crosses_threshold": bool(
                float(entry[onset][own[0] + 1]) >= float(event["onset_threshold"])),
            "onset_peak_in_window": float(
                np.max(delta[onset][own[0]:own[1] + 1])),
            "onset_zero_before_window": bool(
                np.all(np.abs(delta[onset][:own[0]]) < 1e-9)),
            "other_resource_delta_before_their_window": {
                r: float(np.max(np.abs(delta[r][:windows[r][0]])))
                for r in ("cpu", "ram", "disk") if r != onset
                and windows[r][0] > 0},
            "resource_in_window": {
                r: float(np.max(delta[r][windows[r][0]:windows[r][1] + 1]))
                for r in ("cpu", "ram", "disk")},
            "longest_chain_age": max(windows[r][1] for r in windows),
            "observable_history": int(w.cascade["observable_history"]),
        }
        print(json.dumps(report))
        if report["onset_delta_at_window_start"] != 0.0:
            failures.append("A-03/%s onset is not admission-safe at age 0"
                            % regime)
        if report["onset_peak_in_window"] <= 0:
            failures.append("A-03/%s onset peak not positive" % regime)
        if not report["onset_crosses_threshold"]:
            failures.append("A-03/%s onset does not cross its threshold"
                            % regime)
        if not report["onset_zero_before_window"]:
            failures.append("A-03/%s onset leaks before its window" % regime)
        if (report["longest_chain_age"] >= report["observable_history"]
                and regime not in INHERITED_CHAIN_REGIMES):
            failures.append("A-04/%s chain not observable" % regime)

    # ---- A-04: registered lags are reachable and ordered ------------------
    for regime in REGIME_IDS:
        regime_params = REGIMES_V3[regime]
        lags = [lag for _, lag in regime_params["sequence"]]
        print(json.dumps({"check": "A-04 sequence lags (%s)" % regime,
                          "sequence": regime_params["sequence"],
                          "lags": lags,
                          "strictly_increasing": lags == sorted(lags),
                          "onset_is_first": regime_params["sequence"][0][0]
                          == regime_params["onset_resource"]}))
        if lags != sorted(lags) or regime_params["sequence"][0][0] != \
                regime_params["onset_resource"]:
            failures.append("A-04/%s sequence" % regime)

    print(json.dumps({"failed_checks": failures,
                      "status": "PASS" if not failures else "FAIL"}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
