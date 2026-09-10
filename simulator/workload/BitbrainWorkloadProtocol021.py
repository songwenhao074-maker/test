"""Protocol 021 — audited unseen regime: Temporal Resource Cascade workload.

Registered mechanism (protocol §6, fixed BEFORE any model run):

    CPU burst  --lag 4-->  RAM retention/ramp  --lag 8-->  Disk accumulation

    CPU burst duration  : 3-5 intervals, sustained plateau
    RAM elevated window : starts at CPU start + 4, duration 6-10 intervals,
                          progressively interpolated from the familiar level
                          to a registered target (never an instantaneous jump),
                          then released again
    Disk elevated window: starts at CPU start + 8, duration 6-10 intervals,
                          a capped retained-data term that ramps up and is
                          cleaned back to zero (no unbounded growth)

Why registered floors/targets are required
------------------------------------------
Two measurements drive the design (both re-checkable from committed evidence):

1. artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed500_steps2000:
   the familiar demand adapter saturates every live container at
   cpu_upper=1860 and ram_upper=1400 (p50=p90=p99=max), the synthetic disk law
   tops out at 9000 while host disk capacity is 28990.  A blanket multiplier
   applied after the clip therefore changes nothing.
2. The protocol-020 VM cohorts are stratified by CPU activity
   (vm_split.json bucket_rule: train 0-5, dev 6-7, online 8-9).  Raw CPU
   demand of a typical ``online`` VM is ~160 (median) although its peak can
   reach ~3.7e4; so even a 2x multiplier on a light task stays far below one
   host's CPU capacity (4029.0).

The cascade is therefore defined physically: once admitted, the cascading task
demands *at least* the host capacity in that resource (cpu_burst_floor > 4029
CPU from age 1, ram target >= 4500 > 4295 small-host RAM, disk retained term
sized so co-located load crosses 28990), while the *timing* (lags 4 and 8, both
observable inside the 12-step history) is the novel structure that the U4 audit
must detect.  Simulator.getPlacementPossible() evaluates feasibility with the
demand at the admission interval only, so the CPU phase must start at the
familiar level (measured: a flat 4200 floor yields 93.7% deployment
rejection); RAM and Disk phases start at ages 4 and 8 and are therefore
unaffected by admission.

Determinism
-----------
The envelope is a pure function of SeedSequence([replay_seed, creation_id,
mechanism_seed]); no global RNG state is touched, so a replay seed reproduces
the stream exactly.  Audit metadata (regime id, cascade flags, event ids) must
never reach a model input.
"""
import json
from pathlib import Path

import numpy as np

from .BitbrainWorkloadProtocol020 import (DEFAULT_ADAPTER, SPLIT_PATH,
                                          Protocol020AdaptedBWGD2,
                                          _ScaledMarkovDisk)

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Pre-registered mechanism parameters (regime cascade_v1).  These are physical
# properties of the new runtime regime, NOT tuning knobs: changing any value
# requires registering a new regime id (cascade_v2).  Reference capacities:
# host CPU 4029.0, host RAM 4295.0 (8 hosts) / 8192.0 (8 hosts), host disk
# 28990.8 at the familiar baseline scales.
# --------------------------------------------------------------------------
CASCADE_V1 = {
    "regime_id": "cascade_v1",
    "type": "temporal_resource_cascade",
    "mechanism_seed": 21021,
    "cpu_duration": [3, 5],
    "cpu_to_ram_lag": 4,
    "cpu_to_disk_lag": 8,
    "ram_duration": [6, 10],
    "disk_duration": [6, 10],
    # CPU phase: two-stage burst.  Simulator.getPlacementPossible() evaluates
    # CPU/RAM/Disk feasibility with the demand *at the admission interval*, so
    # a task whose age-0 demand already exceeds a host's capacity can never be
    # placed (measurement: 93.7% deployment rejection with a flat burst floor
    # of 4200).  The regime is therefore defined as: admitted at the familiar
    # demand, bursting from the second interval on.
    "cpu_burst_mult": 2.0,
    "cpu_burst_onset": "familiar",   # age 0 keeps the ordinary 020 transform
    "cpu_burst_floor": 4400.0,       # 1.092 x 4029.0, sustained from age 1
    "cpu_burst_upper": 5200.0,
    # RAM phase: interpolated from the familiar level up to this target.
    "ram_ramp_peak_mult": 2.0,
    "ram_target_floor": 4500.0,     # 1.048 x 4295.0 (small-RAM hosts)
    "ram_burst_upper": 6000.0,
    # Disk phase: capped retained-data term on top of the Markov occupancy.
    "disk_retained_peak": 16000.0,  # co-located load then crosses 28990.8
    "disk_retained_cap": 24000.0,
    "cascade_task_probability": 0.25,
    "observable_history": 12,
    "forbidden_inputs": ["regime_id", "phase_id", "cascade_task_flag",
                         "cascade_event_id", "future_demand",
                         "future_capacity", "unmatured_label"],
}

CASCADE_PROBABILITY_CANDIDATES = (0.15, 0.25, 0.35)

_NUMERIC_KEYS = ("cpu_burst_mult", "cpu_burst_floor", "cpu_burst_upper",
                 "ram_ramp_peak_mult", "ram_target_floor", "ram_burst_upper",
                 "disk_retained_peak", "disk_retained_cap")


def _cascade_rng(replay_seed, creation_id, mechanism_seed):
    """Independent, reproducible generator for one task's envelope."""
    return np.random.default_rng(
        np.random.SeedSequence([int(replay_seed), int(creation_id),
                                int(mechanism_seed)]))


def _trapezoid(fraction, rise=0.25, hold=0.70):
    """Progressive onset/hold/release shape on [0, 1] (0 at both edges)."""
    if fraction <= 0.0 or fraction >= 1.0:
        return 0.0
    if fraction < rise:
        return fraction / rise
    if fraction <= hold:
        return 1.0
    return (1.0 - fraction) / (1.0 - hold)


class _CascadeRetainedDisk(_ScaledMarkovDisk):
    """Markov disk occupancy plus a capped, self-cleaning retained-data term."""

    def __init__(self, law, replay_seed, creation_id, disk_mult,
                 start_age, duration, peak, cap):
        super().__init__(law, replay_seed, creation_id, disk_mult)
        self.start_age = int(start_age)
        self.duration = int(duration)
        self.peak = float(peak)
        self.cap = float(cap)

    def retained(self, age):
        if age < self.start_age or age >= self.start_age + self.duration:
            return 0.0
        fraction = (age - self.start_age) / float(max(self.duration - 1, 1))
        return self.peak * _trapezoid(fraction)

    def disk(self):
        value, read, write = super().disk()
        age = self.container.env.interval - self.container.startAt
        added = self.retained(age)
        if added <= 0.0:
            return value, read, write
        return min(value + added, self.cap), read, write


class Protocol021CascadeBWGD2(Protocol020AdaptedBWGD2):
    """Protocol-020 cohort workload plus the registered cascade mechanism.

    ``cascade_probability=0`` reproduces Protocol020AdaptedBWGD2 exactly for
    the same replay seed and cohort (test 01).
    """

    def __init__(self, mean, sigma, replay_seed, cohort="dev",
                 split_path=SPLIT_PATH, adapter=None, disk_law_path=None,
                 cascade=None, cascade_probability=None, mechanism_seed=None):
        super().__init__(mean, sigma, replay_seed, cohort=cohort,
                         split_path=split_path, adapter=adapter,
                         disk_law_path=disk_law_path)
        params = dict(CASCADE_V1)
        for key, value in (cascade or {}).items():
            if key not in params:
                raise ValueError("Unknown cascade parameter %r" % key)
            params[key] = value
        if cascade_probability is not None:
            params["cascade_task_probability"] = float(cascade_probability)
        probability = float(params["cascade_task_probability"])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("cascade_task_probability must be in [0, 1]: %r"
                             % probability)
        params["cascade_task_probability"] = probability
        for key in _NUMERIC_KEYS:
            value = float(params[key])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("cascade %s must be finite positive: %r"
                                 % (key, params[key]))
            params[key] = value
        for pair in ("cpu_duration", "ram_duration", "disk_duration"):
            low, high = params[pair]
            if not 1 <= int(low) <= int(high):
                raise ValueError("Invalid %s range: %r" % (pair, params[pair]))
        for key in ("cpu_to_ram_lag", "cpu_to_disk_lag"):
            if int(params[key]) < 1:
                raise ValueError("Invalid %s: %r" % (key, params[key]))
        # lags must be observable inside the model history, otherwise the
        # regime would be unseen but unlearnable
        history = int(params["observable_history"])
        for key in ("cpu_to_ram_lag", "cpu_to_disk_lag"):
            if int(params[key]) >= history:
                raise ValueError("%s=%d is not observable inside "
                                 "observable_history=%d"
                                 % (key, int(params[key]), history))
        if float(params["ram_target_floor"]) > float(params["ram_burst_upper"]):
            raise ValueError("ram_target_floor exceeds ram_burst_upper")
        if float(params["cpu_burst_floor"]) > float(params["cpu_burst_upper"]):
            raise ValueError("cpu_burst_floor exceeds cpu_burst_upper")
        self.cascade = params
        self.cascade_probability = probability
        self.mechanism_seed = int(params["mechanism_seed"] if mechanism_seed is None
                                  else mechanism_seed)
        self.cascade_events = []
        self._cascade_event_of = {}

    # -- audit accessors (never used as model input) ------------------------
    def cascade_event_id(self, creation_id):
        return self._cascade_event_of.get(int(creation_id))

    def cascade_task_flags(self):
        """creation_id -> 1/0 for every container created so far."""
        return {int(cid): (1 if int(cid) in self._cascade_event_of else 0)
                for cid in self._cascade_creation_ids()}

    def _cascade_creation_ids(self):
        return [int(self.createdContainers[i][0])
                for i in range(len(self.createdContainers))]

    def cascade_audit(self):
        return {"regime_id": self.cascade["regime_id"],
                "mechanism": {k: v for k, v in self.cascade.items()
                              if k != "forbidden_inputs"},
                "mechanism_seed": self.mechanism_seed,
                "events": list(self.cascade_events)}

    # -- envelope ----------------------------------------------------------
    def _envelope(self, creation_id):
        """Pure function of (replay_seed, creation_id, mechanism_seed)."""
        params = self.cascade
        if self.cascade_probability <= 0.0:
            return None
        rng = _cascade_rng(self.replay_seed, creation_id, self.mechanism_seed)
        draw = float(rng.random())
        if draw >= self.cascade_probability:
            return None
        return {
            "draw": draw,
            "cpu_duration": int(rng.integers(params["cpu_duration"][0],
                                             params["cpu_duration"][1] + 1)),
            "ram_duration": int(rng.integers(params["ram_duration"][0],
                                             params["ram_duration"][1] + 1)),
            "disk_duration": int(rng.integers(params["disk_duration"][0],
                                              params["disk_duration"][1] + 1)),
        }

    # -- transform ---------------------------------------------------------
    def adapt_new_tasks(self, first):
        """Single-pass transform; non-cascade tasks are byte-identical to 020."""
        a = self.adapter
        for i in range(first, len(self.createdContainers)):
            cid, interval, ips, ram, _ = self.createdContainers[i]
            if ips.completedInstructions or ips.totalInstructions:
                raise AssertionError("Task already executed before adaptation")
            raw = np.asarray(ips.ips_list, dtype=float)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Invalid CPU demand")
            raw_ram = np.asarray(ram.size_list, dtype=float)
            if not np.isfinite(raw_ram).all() or (raw_ram < 0).any():
                raise ValueError("Invalid RAM demand")

            envelope = self._envelope(cid)
            if envelope is None:
                self._apply_plain(i, cid, ips, ram, raw, raw_ram)
                continue
            self._apply_cascade(i, cid, interval, ips, ram, raw, raw_ram,
                                envelope)

    def _apply_plain(self, index, cid, ips, ram, raw, raw_ram):
        """Byte-identical to Protocol020AdaptedBWGD2.adapt_new_tasks."""
        a = self.adapter
        ips.ips_list = np.where(
            raw > 0, np.clip(raw * a["cpu_mult"], a["cpu_lower"],
                             a["cpu_upper"]), 0.).tolist()
        raw_max = float(ips.max_ips)
        scaled_max = float(np.clip(raw_max * a["cpu_mult"], a["cpu_lower"],
                                   a["cpu_upper"])) if raw_max > 0 else 0.
        ips.max_ips = max(scaled_max, max(ips.ips_list))
        ram_scaled = raw_ram * a["ram_mult"]
        if a.get("ram_upper") is not None:
            ram_scaled = np.minimum(ram_scaled, a["ram_upper"])
        ram.size_list = ram_scaled.tolist()
        ram.read_list = [1.] * len(ram.read_list)
        ram.write_list = [1.] * len(ram.write_list)
        disk = _ScaledMarkovDisk(self.disk_law, self.replay_seed, cid,
                                 a["disk_mult"])
        self.createdContainers[index] = (cid, self.createdContainers[index][1],
                                         ips, ram, disk)

    def _apply_cascade(self, index, cid, interval, ips, ram, raw, raw_ram,
                       envelope):
        params = self.cascade
        a = self.adapter
        cpu_duration = int(envelope["cpu_duration"])
        ram_start = int(params["cpu_to_ram_lag"])
        ram_duration = int(envelope["ram_duration"])
        disk_start = int(params["cpu_to_disk_lag"])
        disk_duration = int(envelope["disk_duration"])

        for name, start, duration, values in (
                ("cpu", 0, cpu_duration, raw),
                ("ram", ram_start, ram_duration, raw_ram),
                ("disk", disk_start, disk_duration, raw_ram)):
            if values.size <= start + duration:
                raise ValueError(
                    "Trace too short for the registered %s window: len=%d "
                    "needs > %d (creation_id=%s)"
                    % (name, values.size, start + duration, cid))

        # ---- CPU phase: admitted familiar, then a sustained burst ---------
        ceiling = np.full(raw.shape, a["cpu_upper"], dtype=float)
        multiplier = np.full(raw.shape, a["cpu_mult"], dtype=float)
        ceiling[:cpu_duration] = params["cpu_burst_upper"]
        multiplier[:cpu_duration] = params["cpu_burst_mult"] * a["cpu_mult"]
        burst = np.clip(raw[:cpu_duration] * multiplier[:cpu_duration],
                        params["cpu_burst_floor"], params["cpu_burst_upper"])
        # admission interval (age 0) keeps the ordinary transform so that the
        # task remains placeable; the burst starts at age 1
        burst[0] = np.clip(raw[0] * a["cpu_mult"], a["cpu_lower"], a["cpu_upper"]) \
            if raw.size else 0.
        rest = np.where(raw[cpu_duration:] > 0,
                        np.clip(raw[cpu_duration:] * a["cpu_mult"],
                                a["cpu_lower"], a["cpu_upper"]), 0.)
        clipped = np.concatenate([burst, rest])
        ips.ips_list = clipped.tolist()
        raw_max = float(ips.max_ips)
        scaled_max = float(np.clip(raw_max * params["cpu_burst_mult"]
                                   * a["cpu_mult"], params["cpu_burst_floor"],
                                   params["cpu_burst_upper"])) if raw_max > 0 else 0.
        ips.max_ips = max(scaled_max, float(clipped.max()) if clipped.size else 0.)

        # ---- RAM phase: interpolate base -> target across the lagged window
        base = raw_ram * a["ram_mult"]
        if a.get("ram_upper") is not None:
            base = np.minimum(base, a["ram_upper"])
        ram_scaled = base.copy()
        target_all = np.clip(raw_ram * a["ram_mult"] * params["ram_ramp_peak_mult"],
                             params["ram_target_floor"], params["ram_burst_upper"])
        peak_age, peak_value = None, 0.0
        for k in range(ram_duration):
            age = ram_start + k
            fraction = k / float(max(ram_duration - 1, 1))
            target = max(float(target_all[age]), float(base[age]))
            value = float(base[age]) + (target - float(base[age])) \
                * _trapezoid(fraction)
            ram_scaled[age] = value
            if value > peak_value:
                peak_value, peak_age = value, age
        ram.size_list = ram_scaled.tolist()
        ram.read_list = [1.] * len(ram.read_list)
        ram.write_list = [1.] * len(ram.write_list)

        # ---- Disk phase: capped, self-cleaning retained term --------------
        disk = _CascadeRetainedDisk(self.disk_law, self.replay_seed, cid,
                                    a["disk_mult"], start_age=disk_start,
                                    duration=disk_duration,
                                    peak=params["disk_retained_peak"],
                                    cap=params["disk_retained_cap"])
        self.createdContainers[index] = (cid, interval, ips, ram, disk)

        event_id = len(self.cascade_events)
        self.cascade_events.append({
            "event_id": event_id,
            "regime_id": params["regime_id"],
            "creation_id": int(cid),
            "creation_interval": int(interval),
            "probability_draw": float(envelope["draw"]),
            "cpu_window": [0, cpu_duration - 1],
            "cpu_duration": cpu_duration,
            "cpu_onset_value": float(burst[0]) if burst.size else 0.0,
            "cpu_burst_floor": params["cpu_burst_floor"],
            "cpu_peak_value": float(burst.max()) if burst.size else 0.0,
            "ram_window": [ram_start, ram_start + ram_duration - 1],
            "ram_duration": ram_duration,
            "ram_peak_age": peak_age,
            "ram_peak_value": peak_value,
            "ram_window_start_value": float(ram_scaled[ram_start]),
            "disk_window": [disk_start, disk_start + disk_duration - 1],
            "disk_duration": disk_duration,
            "disk_retained_peak": params["disk_retained_peak"],
        })
        self._cascade_event_of[int(cid)] = event_id
