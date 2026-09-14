"""Protocol-024 response-law workload for the next-round development pilot.

The three laws operate on the real Bitbrain online cohort and the existing
Protocol-020 adapter/disk process.  They change task demand trajectories, never
labels.  Host fault labels are still produced downstream by the simulator from
executed aggregate demand divided by physical host capacity.

Every law is admission-safe: age 0 is byte-identical to the familiar adapted
Bitbrain demand.  The first non-familiar values appear only after admission.
Audit IDs returned by this module are forbidden model/lifecycle inputs.
"""
from __future__ import annotations

import numpy as np

from .BitbrainWorkloadProtocol020 import (
    Protocol020AdaptedBWGD2, SPLIT_PATH, _ScaledMarkovDisk)


LAW_IDS = ("R1", "R2", "R3")
LAW_MECHANISM_IDS = {"R1": 0, "R2": 1, "R3": 2}
LAW_SEEDS = {"R1": 24011, "R2": 24012, "R3": 24013}
EVENT_PROBABILITY = 0.30
OBSERVABLE_HISTORY = 12

# Shapes are fixed before stream generation.  They are fractions of the
# difference between familiar demand and each law's registered target.  Age 0
# is exactly zero in every shape.
RESPONSE_LAWS = {
    "R1": {
        "name": "short_compute_pulse_fast_release",
        "mechanism_id": 0,
        "seed": 24011,
        "probability": EVENT_PROBABILITY,
        "cpu_shape": [0.0, 0.45, 0.95, 1.0, 0.55, 0.15, 0.0],
        "cpu_floor": 5000.0,
        "cpu_upper": 5400.0,
        "ram_shape": [0.0, 0.08, 0.12, 0.10, 0.05, 0.0, 0.0],
        "ram_floor": 1800.0,
        "ram_upper": 2100.0,
        "disk_shape": [0.0] * 7,
        "disk_retained_peak": 0.0,
        "disk_cap": 36000.0,
        "minimum_trace": 7,
        "observable_cue": "two-step CPU rise, short overload pulse, fast decline",
    },
    "R2": {
        "name": "sustained_inference_working_set_growth",
        "mechanism_id": 1,
        "seed": 24012,
        "probability": EVENT_PROBABILITY,
        "cpu_shape": [0.0, 0.25, 0.55, 0.82, 1.0, 1.0, 1.0, 1.0,
                      0.92, 0.78, 0.55, 0.25, 0.0],
        "cpu_floor": 4600.0,
        "cpu_upper": 5000.0,
        "ram_shape": [0.0, 0.0, 0.10, 0.22, 0.38, 0.58, 0.78, 0.95,
                      1.0, 1.0, 0.82, 0.55, 0.25],
        "ram_floor": 4700.0,
        "ram_upper": 5400.0,
        "disk_shape": [0.0] * 13,
        "disk_retained_peak": 0.0,
        "disk_cap": 36000.0,
        "minimum_trace": 13,
        "observable_cue": "sustained CPU rise followed by lagged RAM working-set growth",
    },
    "R3": {
        "name": "background_writeback_backlog",
        "mechanism_id": 2,
        "seed": 24013,
        "probability": EVENT_PROBABILITY,
        "cpu_shape": [0.0, 0.08, 0.16, 0.24, 0.32, 0.38, 0.34, 0.28,
                      0.20, 0.12, 0.05, 0.0],
        "cpu_floor": 2550.0,
        "cpu_upper": 2900.0,
        "ram_shape": [0.0, 0.05, 0.12, 0.20, 0.30, 0.38, 0.36, 0.30,
                      0.22, 0.14, 0.06, 0.0],
        "ram_floor": 2100.0,
        "ram_upper": 2500.0,
        "disk_shape": [0.0, 0.10, 0.25, 0.45, 0.70, 1.0, 1.0, 0.85,
                       0.60, 0.35, 0.15, 0.0],
        "disk_retained_peak": 32000.0,
        "disk_cap": 36000.0,
        "minimum_trace": 12,
        "observable_cue": "disk backlog rises over several steps then drains with weak CPU/RAM companions",
    },
}

FORBIDDEN_MODEL_INPUTS = (
    "response_law_id", "mode_id", "phase_id", "event_id",
    "known_future_switch_time", "future_label", "future_demand",
    "future_capacity")


def _event_rng(replay_seed, creation_id, law_id):
    return np.random.default_rng(np.random.SeedSequence(
        [int(replay_seed), int(creation_id), int(LAW_SEEDS[law_id])]))


def _blend_to_target(base, shape, floor, upper):
    base = np.asarray(base, dtype=np.float64)
    out = base.copy()
    shape = np.asarray(shape, dtype=np.float64)
    n = min(base.size, shape.size)
    if n == 0:
        return out
    target = np.clip(np.maximum(base[:n], float(floor)), float(floor), float(upper))
    out[:n] = base[:n] + shape[:n] * (target - base[:n])
    out[0] = base[0]  # hard admission-safety invariant
    return out


class _ResponseLawDisk(_ScaledMarkovDisk):
    def __init__(self, law, replay_seed, creation_id, disk_mult,
                 retained_shape, retained_peak, cap):
        super().__init__(law, replay_seed, creation_id, disk_mult)
        self.retained_shape = tuple(float(x) for x in retained_shape)
        self.retained_peak = float(retained_peak)
        self.cap = float(cap)

    def retained(self, age):
        age = int(age)
        if age < 0 or age >= len(self.retained_shape):
            return 0.0
        return self.retained_peak * self.retained_shape[age]

    def disk(self):
        value, read, write = super().disk()
        age = self.container.env.interval - self.container.startAt
        added = self.retained(age)
        if added <= 0.0:
            return value, read, write
        return min(float(value) + added, self.cap), read, write


class Protocol024ResponseLawBWGD2(Protocol020AdaptedBWGD2):
    """Bitbrain workload with causal task-age response laws R1/R2/R3."""

    def __init__(self, mean, sigma, replay_seed, cohort="online",
                 split_path=SPLIT_PATH, adapter=None, disk_law_path=None,
                 active_law=None, event_probability=EVENT_PROBABILITY):
        super().__init__(mean, sigma, replay_seed, cohort=cohort,
                         split_path=split_path, adapter=adapter,
                         disk_law_path=disk_law_path)
        self._active_law = None
        self.event_probability = float(event_probability)
        if not 0.0 <= self.event_probability <= 1.0:
            raise ValueError("event_probability must be in [0,1]")
        self.response_events = []
        self._event_of = {}
        self.short_trace_skips = {law: 0 for law in LAW_IDS}
        self.set_active_law(active_law, probability=event_probability)

    @property
    def active_law(self):
        return self._active_law

    def set_active_law(self, law_id, probability=None):
        if law_id is not None and law_id not in LAW_IDS:
            raise ValueError("unregistered response law %r" % (law_id,))
        if probability is not None:
            probability = float(probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("event probability must be in [0,1]")
            self.event_probability = probability
        self._active_law = law_id
        return {"response_law": law_id, "event_probability": self.event_probability}

    def response_event_id(self, creation_id):
        return self._event_of.get(int(creation_id))

    def response_audit(self):
        return {
            "family": "protocol024_response_law_v1",
            "laws": RESPONSE_LAWS,
            "event_probability": self.event_probability,
            "observable_history": OBSERVABLE_HISTORY,
            "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
            "short_trace_skips": dict(self.short_trace_skips),
            "events": list(self.response_events),
        }

    def _draw_event(self, cid):
        if self._active_law is None or self.event_probability <= 0.0:
            return None
        rng = _event_rng(self.replay_seed, cid, self._active_law)
        draw = float(rng.random())
        if draw >= self.event_probability:
            return None
        return {"law_id": self._active_law, "draw": draw}

    def _apply_plain(self, i, cid, interval, ips, ram, raw, raw_ram):
        a = self.adapter
        cpu = np.where(raw > 0,
                       np.clip(raw * a["cpu_mult"], a["cpu_lower"], a["cpu_upper"]),
                       0.0)
        ips.ips_list = cpu.tolist()
        raw_max = float(ips.max_ips)
        scaled_max = (float(np.clip(raw_max * a["cpu_mult"], a["cpu_lower"],
                                    a["cpu_upper"])) if raw_max > 0 else 0.0)
        ips.max_ips = max(scaled_max, float(cpu.max()) if cpu.size else 0.0)
        ram_scaled = raw_ram * a["ram_mult"]
        if a.get("ram_upper") is not None:
            ram_scaled = np.minimum(ram_scaled, a["ram_upper"])
        ram.size_list = ram_scaled.tolist()
        ram.read_list = [1.0] * len(ram.read_list)
        ram.write_list = [1.0] * len(ram.write_list)
        disk = _ScaledMarkovDisk(self.disk_law, self.replay_seed, cid,
                                 a["disk_mult"])
        self.createdContainers[i] = (cid, interval, ips, ram, disk)
        return cpu, ram_scaled

    def _apply_event(self, i, cid, interval, ips, ram, raw, raw_ram, envelope):
        law_id = envelope["law_id"]
        spec = RESPONSE_LAWS[law_id]
        minimum = int(spec["minimum_trace"])
        if raw.size < minimum or raw_ram.size < minimum:
            self.short_trace_skips[law_id] += 1
            self._apply_plain(i, cid, interval, ips, ram, raw, raw_ram)
            return False

        a = self.adapter
        familiar_cpu = np.where(raw > 0,
                                np.clip(raw * a["cpu_mult"], a["cpu_lower"],
                                        a["cpu_upper"]), 0.0)
        familiar_ram = raw_ram * a["ram_mult"]
        if a.get("ram_upper") is not None:
            familiar_ram = np.minimum(familiar_ram, a["ram_upper"])

        cpu = _blend_to_target(familiar_cpu, spec["cpu_shape"],
                               spec["cpu_floor"], spec["cpu_upper"])
        ram_scaled = _blend_to_target(familiar_ram, spec["ram_shape"],
                                      spec["ram_floor"], spec["ram_upper"])
        ips.ips_list = cpu.tolist()
        raw_max = float(ips.max_ips)
        familiar_max = (float(np.clip(raw_max * a["cpu_mult"], a["cpu_lower"],
                                      a["cpu_upper"])) if raw_max > 0 else 0.0)
        ips.max_ips = max(familiar_max, float(cpu.max()) if cpu.size else 0.0)
        ram.size_list = ram_scaled.tolist()
        ram.read_list = [1.0] * len(ram.read_list)
        ram.write_list = [1.0] * len(ram.write_list)
        disk = _ResponseLawDisk(
            self.disk_law, self.replay_seed, cid, a["disk_mult"],
            spec["disk_shape"], spec["disk_retained_peak"], spec["disk_cap"])
        self.createdContainers[i] = (cid, interval, ips, ram, disk)

        event_id = len(self.response_events)
        event = {
            "event_id": event_id,
            "family": "protocol024_response_law_v1",
            "law_id": law_id,
            "mechanism_id": int(spec["mechanism_id"]),
            "creation_id": int(cid),
            "creation_interval": int(interval),
            "probability_draw": float(envelope["draw"]),
            "cpu_window": [0, len(spec["cpu_shape"]) - 1],
            "ram_window": [0, len(spec["ram_shape"]) - 1],
            "disk_window": [0, len(spec["disk_shape"]) - 1],
            "cpu_peak": float(cpu[:minimum].max()),
            "ram_peak": float(ram_scaled[:minimum].max()),
            "disk_retained_peak": float(spec["disk_retained_peak"]),
            "observable_cue": spec["observable_cue"],
        }
        self.response_events.append(event)
        self._event_of[int(cid)] = event_id
        return True

    def adapt_new_tasks(self, first):
        for i in range(first, len(self.createdContainers)):
            cid, interval, ips, ram, _ = self.createdContainers[i]
            if ips.completedInstructions or ips.totalInstructions:
                raise AssertionError("Task already executed before adaptation")
            raw = np.asarray(ips.ips_list, dtype=np.float64)
            raw_ram = np.asarray(ram.size_list, dtype=np.float64)
            if (not np.isfinite(raw).all() or (raw < 0).any()
                    or not np.isfinite(raw_ram).all() or (raw_ram < 0).any()):
                raise ValueError("invalid Bitbrain demand")
            envelope = self._draw_event(cid)
            if envelope is None:
                self._apply_plain(i, cid, interval, ips, ram, raw, raw_ram)
            else:
                self._apply_event(i, cid, interval, ips, ram, raw, raw_ram,
                                  envelope)
