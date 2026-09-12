"""Protocol 022 — registered regime ``cascade_v2`` (task/event-auditable cascade).

Why a second round exists
-------------------------
Protocol 021 registered ``cascade_v1`` and its U4 gate FAILED (P21-09, STOP-A).
The gate was measured on ``[T, H, 7]`` *host aggregates*, where co-located heavy
tasks already make CPU/RAM/Disk co-move, so the instrument could not separate a
per-task temporal cascade from co-residency co-movement.  Protocol 022 therefore
keeps the physical envelope unchanged (comparability with P21 was the stated
reason) and changes three things only:

    1. the audit scale        host aggregate -> task (``creation_id``) / event
    2. a new mechanism seed   21021 -> 22022 (confirm seed 22023, unused here)
    3. matched negative controls in the audit layer (M0/M1/M2, never in the
       generator and never in a model input)

Registered mechanism (identical physics to ``cascade_v1``, plan §5.1):

    CPU burst  --lag 4-->  RAM retention/ramp  --lag 8-->  Disk accumulation

    CPU burst duration      : 3-5 intervals, sustained plateau, admitted at the
                              familiar level and bursting from age 1
    CPU burst floor / upper : 4400 / 5200   (> host CPU capacity 4029)
    RAM lag / duration      : +4, 6-10 intervals, interpolated (never a jump)
    RAM target floor / upper: 4500 / 6000   (> small-host RAM 4295)
    Disk lag / duration     : +8, 6-10 intervals, capped self-cleaning retained
    Disk retained peak / cap: 16000 / 24000
    observable history      : 12

Admission is the reason for the two-stage CPU phase: ``Simulator.
getPlacementPossible()`` evaluates feasibility with the demand *at the admission
interval*, so a flat over-capacity burst at age 0 is rejected before it can
overload anything (P21-01: 93.7% deployment rejection).  This is a property of
the simulator's construct validity and is reported as such (plan §19): the
regime is a *controlled unseen temporal resource-demand regime*, not a claim
that industrial faults must follow this law.

Determinism
-----------
Each task envelope is a pure function of
``SeedSequence([replay_seed, creation_id, mechanism_seed])``; no global RNG
state is touched.  Audit metadata (regime id, cascade flags, event ids, future
windows) is exposed through dedicated accessors and must never become a model
input (``FORBIDDEN_INPUTS``).

Usage
-----
    from simulator.workload.BitbrainWorkloadProtocol022 import (
        Protocol022CascadeBWGD2, CASCADE_V2)
"""
import numpy as np

from .BitbrainWorkloadProtocol021 import (CASCADE_PROBABILITY_CANDIDATES,
                                          CASCADE_V1,
                                          Protocol021CascadeBWGD2)

# --------------------------------------------------------------------------
# Registered regime parameters.  Changing any physical value requires a new
# regime id (cascade_v3) and a fresh registration; the audit instrument may not
# be redefined after seeing candidate results (plan §7.3).
# --------------------------------------------------------------------------
CASCADE_V2 = {
    "regime_id": "cascade_v2",
    "type": "temporal_resource_cascade",
    "mechanism_seed": 22022,
    "cpu_duration": [3, 5],
    "cpu_to_ram_lag": 4,
    "cpu_to_disk_lag": 8,
    "ram_duration": [6, 10],
    "disk_duration": [6, 10],
    # CPU phase: two-stage burst (see module docstring / P21-01)
    "cpu_burst_mult": 2.0,
    "cpu_burst_onset": "familiar",
    "cpu_burst_floor": 4400.0,
    "cpu_burst_upper": 5200.0,
    # RAM phase: interpolated from the familiar level up to this target
    "ram_ramp_peak_mult": 2.0,
    "ram_target_floor": 4500.0,
    "ram_burst_upper": 6000.0,
    # Disk phase: capped retained-data term on top of the Markov occupancy
    "disk_retained_peak": 16000.0,
    "disk_retained_cap": 24000.0,
    "cascade_task_probability": 0.25,
    "observable_history": 12,
    # Registered onset threshold for the task-level audit instrument.  A
    # familiar task is clipped to cpu_upper=1860 everywhere in the offline
    # chain, so 2600 is strictly above every offline per-task CPU value while
    # sitting far below the 4400 burst floor: it can only ever be crossed by a
    # registered cascade burst.  audit_ftmoe_protocol022_unseen.py reports the
    # measured offline maximum next to this constant.
    "onset_tau_cpu": 2600.0,
    "onset_baseline_lag": 4,
    "ram_response_window": [4, 14],
    "disk_response_window": [8, 18],
    "forbidden_inputs": ["regime_id", "phase_id", "cascade_task_flag",
                         "cascade_event_id", "future_demand",
                         "future_capacity", "unmatured_label"],
}

MECHANISM_SEED_DEV = 22022
MECHANISM_SEED_CONFIRM = 22023

# Audit-layer controls registered by plan §6.5.  They exist only as
# transformations of an already-collected task timeline.
MATCHED_CONTROLS = {
    "M0": "marginal-matched lag-shuffled: keep every task's marginal amplitude, "
          "duration distribution and RAM/Disk windows, but re-pair the windows "
          "with a uniformly drawn lag (registered lags destroyed)",
    "M1": "order-shuffled: keep the same resource events but reveal them in a "
          "randomly permuted temporal order",
    "M2": "within-task circular shift: circularly shift each task's RAM/Disk "
          "deviation trajectory, preserving marginals and autocorrelation but "
          "destroying the registered causal alignment",
}


class Protocol022CascadeBWGD2(Protocol021CascadeBWGD2):
    """Protocol-021 cascade workload under the newly registered ``cascade_v2``.

    ``cascade_probability=0`` reproduces ``Protocol020AdaptedBWGD2`` exactly for
    the same replay seed and cohort, which is the T-AUDIT-05 regression.
    """

    def __init__(self, mean, sigma, replay_seed, cohort="dev", split_path=None,
                 adapter=None, disk_law_path=None, cascade=None,
                 cascade_probability=None, mechanism_seed=None):
        params = dict(CASCADE_V2)
        for key, value in (cascade or {}).items():
            if key not in params:
                raise ValueError("Unknown cascade_v2 parameter %r" % key)
            params[key] = value
        if mechanism_seed is not None:
            params["mechanism_seed"] = int(mechanism_seed)
        if split_path is None:
            # defer to the protocol-020 cohort split used by the base class
            from .BitbrainWorkloadProtocol020 import SPLIT_PATH
            split_path = SPLIT_PATH
        # The registered v2 dict carries audit-instrument constants (onset
        # threshold, response windows) that are not envelope parameters.  The
        # base class validates every envelope key it receives, so only those
        # keys are forwarded; the full dict stays on self.cascade for the
        # registry and the drift check.
        inherited_keys = tuple(CASCADE_V1)
        envelope = {k: v for k, v in params.items() if k in inherited_keys}
        super().__init__(mean, sigma, replay_seed, cohort=cohort,
                         split_path=split_path, adapter=adapter,
                         disk_law_path=disk_law_path, cascade=envelope,
                         cascade_probability=cascade_probability,
                         mechanism_seed=params["mechanism_seed"])
        self.cascade = params          # full registered v2 dict, for the audit

    # -- registered audit instrument ----------------------------------------
    @staticmethod
    def onset_definition():
        """The task-level onset rule, quoted verbatim into the audit output."""
        return {
            "unit": "creation_id + CPU onset event",
            "rule": ("the first observed interval at which the task's own CPU "
                     "reaches the registered threshold, given that the previous "
                     "observed interval was strictly below it; a sustained burst "
                     "counts once"),
            "threshold": CASCADE_V2["onset_tau_cpu"],
            "threshold_basis": ("strictly above the familiar per-task CPU clip "
                                "(1860) and below the registered burst floor "
                                "(4400): only a cascade burst can reach it. "
                                "Measured offline per-task CPU maximum is 1860 "
                                "in every auditable corpus, so no familiar task "
                                "can produce an event."),
            "baseline_free": ("True.  The registered envelope is admitted at the "
                              "familiar level and bursts at age 1, so a "
                              "median-baseline jump has no pre-onset history to "
                              "compare against (P22-01: max delta_cpu = -4400, "
                              "zero onsets, while the task really does reach "
                              "4400-5200)."),
            "response_baseline": ("median of the task's own pre-onset rows when "
                                  "they exist, else the registered familiar task "
                                  "level; reported per event"),
            "source": "task's own demand only, never the host aggregate",
        }

    def cascade_event(self, event_id):
        return self.cascade_events[int(event_id)]

    def task_cascade_windows(self):
        """creation_id -> registered windows, for the task-level audit only."""
        out = {}
        for event in self.cascade_events:
            out[int(event["creation_id"])] = {
                "event_id": int(event["event_id"]),
                "cpu_window": list(event["cpu_window"]),
                "ram_window": list(event["ram_window"]),
                "disk_window": list(event["disk_window"]),
                "cpu_burst_floor": float(event["cpu_burst_floor"]),
                "cpu_peak_value": float(event["cpu_peak_value"]),
            }
        return out

    def cascade_audit(self):
        payload = super().cascade_audit()
        payload["onset_definition"] = self.onset_definition()
        payload["matched_controls"] = dict(MATCHED_CONTROLS)
        payload["physics_note"] = (
            "controlled unseen temporal resource-demand regime; the two-stage "
            "CPU phase is required by Simulator.getPlacementPossible() "
            "evaluating demand at the admission interval (P21-01), not a claim "
            "about real industrial fault laws")
        payload["registered_probability_candidates"] = list(
            CASCADE_PROBABILITY_CANDIDATES)
        return payload

    def assert_registered_physics(self):
        """Fail loudly if the runtime regime drifted from the registration."""
        for key, expected in CASCADE_V2.items():
            if key == "cascade_task_probability":
                continue          # intentionally overridden per candidate
            if key not in self.cascade:
                raise AssertionError("cascade_v2 parameter %s is missing" % key)
            observed = self.cascade[key]
            if isinstance(expected, str) or isinstance(expected, bool):
                if observed != expected:
                    raise AssertionError("cascade_v2 %s drifted: %r != %r"
                                         % (key, observed, expected))
            elif isinstance(expected, (list, tuple)):
                if list(observed) != list(expected):
                    raise AssertionError("cascade_v2 %s drifted: %r != %r"
                                         % (key, observed, expected))
            else:
                if abs(float(observed) - float(expected)) > 1e-12:
                    raise AssertionError("cascade_v2 %s drifted: %r != %r"
                                         % (key, observed, expected))
        if self.mechanism_seed not in (MECHANISM_SEED_DEV, MECHANISM_SEED_CONFIRM):
            raise AssertionError("unregistered mechanism seed %r"
                                 % (self.mechanism_seed,))
        return True


def registered_regime(probability=None, mechanism_seed=None):
    """Plain-dict view of the registered regime (for the registry writer)."""
    params = dict(CASCADE_V2)
    if probability is not None:
        params["cascade_task_probability"] = float(probability)
    if mechanism_seed is not None:
        params["mechanism_seed"] = int(mechanism_seed)
    return params
