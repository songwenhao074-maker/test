"""Protocol 023 РІР‚вЂќ registered regime family ``cascade_v3`` (three resource orders).

Why a third generator round exists
----------------------------------
Protocol 022 registered a single unseen regime (``cascade_v2``:
CPU --> RAM --> Disk) and its capacity diagnostic (P22-22) produced the result
that motivates this protocol: with a *fixed* expert set, late-unseen PR-AUC kept
climbing as the update budget grew (0.4256 -> 0.6891 over 100 -> 1600 updates,
still +0.0323 at the tail) while the pre-onset warning metric did not improve.
A single regime therefore cannot establish a capacity plateau, and without a
plateau a dynamic-expert mechanism has no scientific necessity.  The plan
(``Р¶РЉвЂЎРґВ»В¤/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md``) registers three
*heterogeneous* regimes that differ only in resource order and lag, plus short
dwell times and recurrence, as the environment in which that necessity can
actually be tested.

Registered regimes (plan §5, §6, §7)
------------------------------------
    A  compute_first :  CPU burst   --lag 4-->  RAM ramp    --lag 8-->  Disk retained
    B  memory_first  :  RAM ramp    --lag 3-->  Disk retained --lag 6--> CPU burst
    C  io_first      :  Disk retained --lag 3--> CPU burst   --lag 6-->  RAM ramp

A is byte-compatible with Protocol 022's ``cascade_v2`` (same floors, same
durations, same lags, same RNG consumption order).  That is deliberate: the A
arm of Protocol 023 must be the already-audited regime, so any difference
measured between A and B/C is attributable to the resource order rather than to
a re-tuned envelope.

What is matched and what is allowed to differ (plan §8, §9)
----------------------------------------------------------
Matched across the three regimes: overall anomaly prevalence, event count,
event duration, deployment/migration rejection, resource peak ratios and the
number of normal host-steps.  Allowed to differ: the *order* of the resources,
the lags, and which resource the cascade starts in.  ``marginal_match_report``
in ``ftmoe_protocol023_core.py`` evaluates the matched quantities, and
``regime_contrast_report`` separates the two groups so a reader can check that
marginal matching did not silently flatten the joint structure.

Every phase of the cascade has the *same shape law* (a trapezoid deviation that
is exactly zero at age 0), so the physical claim per regime is identical and
only the sequence differs:

    age 0        : the ordinary protocol-020 transform (admission-safe)
    age 1..d-1   : progressive rise to the registered level
    held         : registered plateau
    release      : progressive return to the familiar level

Age 0 matters: ``Simulator.getPlacementPossible()`` evaluates feasibility with
the demand *at the admission interval*, so a container whose age-0 demand
already exceeds host capacity can never be placed (P21-01 measured 93.7%
deployment rejection for a flat burst floor).  Trapezoids make every phase
admission-safe by construction, for all three regimes and for all three
resources, instead of special-casing the CPU phase.

Observability constraint
------------------------
Every registered lag must be visible inside the 12-interval model history or
the regime would be unseen *and unlearnable*.  A keeps Protocol 022's durations
(CPU 3-5, RAM 6-10, Disk 6-10) because that regime is already audited; B and C
use 4-6 interval phases so that their longest chain stays inside the history:

    B  longest chain age = 6 + 5 = 11  <  12
    C  longest chain age = 6 + 5 = 11  <  12

Determinism
-----------
Each task envelope is a pure function of
``SeedSequence([replay_seed, creation_id, mechanism_seed])``; no global RNG
state is touched.  With exactly one registered regime the draw order is
identical to Protocol 022's, which is what makes the A-regime regression
(P23-A-01) an equality rather than an approximation.

Forbidden model inputs
----------------------
``regime_id``, ``phase_id``, ``mechanism_id``, ``event_id``, cascade flags and
future windows are audit metadata.  They are exposed through dedicated
accessors and must never become a model input (``FORBIDDEN_INPUTS``); plan §4
states this explicitly and plan §19 forbids using them in any trigger.

Usage
-----
    from simulator.workload.BitbrainWorkloadProtocol023 import (
        Protocol023MultiRegimeBWGD2, REGIMES_V3, REGIME_IDS)
"""
import numpy as np

from .BitbrainWorkloadProtocol021 import (CASCADE_PROBABILITY_CANDIDATES,
                                          CASCADE_V1,
                                          Protocol021CascadeBWGD2,
                                          _CascadeRetainedDisk, _cascade_rng,
                                          _trapezoid)

# --------------------------------------------------------------------------
# Registered regime family.  Changing any physical value requires a new family
# id (cascade_v4) and a fresh registration; the audit instrument may not be
# redefined after seeing candidate results.
# --------------------------------------------------------------------------
REGIME_IDS = ("compute_first", "memory_first", "io_first")

MECHANISM_SEED_DEV = 22022
MECHANISM_SEED_CONFIRM = 22023
# Registry note: the compute-first regime carries Protocol 022's REGISTERED
# mechanism seed (22022 / confirm 22023), because it *is* cascade_v2.  The two
# newly registered orders get their own seeds, and the generator refuses a
# family whose regimes share a seed (the envelopes would be identical).
REGIME_MECHANISM_SEEDS = {"compute_first": MECHANISM_SEED_DEV,
                          "memory_first": 23023,
                          "io_first": 23033}

# Regimes whose *chain geometry* is inherited verbatim from an already-audited
# protocol.  ``compute_first`` is Protocol 022's ``cascade_v2``: its lags are
# registered and observable inside the 12-interval history, but its longest
# phase window (CPU 3-5 -> RAM +4 for 6-10 -> Disk +8 for 6-10, i.e. up to age
# 17) deliberately extends past the model's lookback, because what is being
# detected is the *onset*, not a reconstruction of the whole chain.  The
# observability rule ("every registered lag < observable_history") therefore
# applies to every regime, while the stricter "longest chain age <
# observable_history" rule is enforced only for the newly registered regimes
# (memory_first, io_first), whose chains were sized to fit inside it.
INHERITED_CHAIN_REGIMES = ("compute_first",)

# Reference capacities at the familiar baseline scales (recorded by P21/P22):
# host CPU 4029.0, small-host RAM 4295.0, host disk 28990.8.
_REGISTERED_COMMON = {
    "type": "temporal_resource_cascade",
    "mechanism_seed": MECHANISM_SEED_DEV,
    "cpu_duration": [3, 5],
    "ram_duration": [6, 10],
    "disk_duration": [6, 10],
    "cpu_burst_mult": 2.0,
    "cpu_burst_onset": "familiar",
    "cpu_burst_floor": 4400.0,        # 1.092 x 4029.0
    "cpu_burst_upper": 5200.0,
    # The RAM ramp peak multiplier is NOT shared across the family: regime A
    # keeps Protocol 022's registered 2.0 (that regime *is* cascade_v2, so its
    # trajectories must stay byte-identical), while the two new orders use 2.5.
    # Both reach the same registered RAM amplitude band, which is what plan §9
    # requires to be matched; only A's inheritance must be exact.
    "ram_ramp_peak_mult": 2.0,
    "ram_target_floor": 4500.0,       # 1.048 x 4295.0 (small-RAM hosts)
    "ram_burst_upper": 6000.0,
    "disk_retained_peak": 16000.0,    # co-located load crosses 28990.8
    "disk_retained_cap": 24000.0,
    "cascade_task_probability": 0.25,
    "observable_history": 12,
    "forbidden_inputs": ["regime_id", "phase_id", "mechanism_id",
                         "cascade_task_flag", "cascade_event_id",
                         "future_demand", "future_capacity",
                         "unmatured_label"],
}

# The RAM and disk onset thresholds are the *provisional* values registered
# before the pilot; ``calibrate_ftmoe_protocol023_thresholds.py`` measures the
# familiar per-task clips from a probability-0 stream and the registered values
# in ``REGISTERED_ONSET_TAU`` are the ones the S2 gate is evaluated with.  A
# threshold is admissible only if it is strictly above every familiar per-task
# value of that resource and strictly below the regime's own floor.
REGISTERED_ONSET_TAU = {"cpu": 2600.0, "ram": 2810.0, "disk": 11600.0}
PROVISIONAL_ONSET_TAU = dict(REGISTERED_ONSET_TAU)

REGIMES_V3 = {
    "compute_first": dict(
        _REGISTERED_COMMON,
        regime_id="compute_first",
        family="cascade_v3",
        onset_resource="cpu",
        mechanism_id=0,
        sequence=[["cpu", 0], ["ram", 4], ["disk", 8]],
        cpu_to_ram_lag=4,
        cpu_to_disk_lag=8,
        response_windows={"ram": [4, 14], "disk": [8, 18]},
        physical_story=("compute-intensive inference burst: the task's own CPU "
                        "demand saturates the host, retention then raises RAM "
                        "and the delayed write-out accumulates disk"),
    ),
    "memory_first": dict(
        _REGISTERED_COMMON,
        regime_id="memory_first",
        family="cascade_v3",
        onset_resource="ram",
        mechanism_id=1,
        # B and C use 4-6 interval phases so the longest chain stays inside the
        # 12-interval observable history.  A keeps P22's durations because that
        # regime is already audited.
        ram_ramp_peak_mult=2.5,
        ram_duration=[4, 5],
        disk_duration=[4, 5],
        sequence=[["ram", 0], ["disk", 3], ["cpu", 6]],
        ram_to_disk_lag=3,
        disk_to_cpu_lag=6,
        response_windows={"disk": [3, 13], "cpu": [6, 16]},
        physical_story=("working-set growth / memory leak: RAM pressure first, "
                        "paging then loads disk, and the paging CPU overhead "
                        "arrives last"),
    ),
    "io_first": dict(
        _REGISTERED_COMMON,
        regime_id="io_first",
        family="cascade_v3",
        onset_resource="disk",
        mechanism_id=2,
        ram_ramp_peak_mult=2.5,
        cpu_duration=[4, 6],
        ram_duration=[4, 6],
        disk_duration=[4, 6],
        sequence=[["disk", 0], ["cpu", 3], ["ram", 6]],
        disk_to_cpu_lag=3,
        cpu_to_ram_lag=6,
        response_windows={"cpu": [3, 13], "ram": [6, 16]},
        physical_story=("logging / checkpoint / data-upload accumulation: disk "
                        "pressure first, cleanup+compaction CPU next, and the "
                        "page-cache disturbance last"),
    ),
}

# Layer-0 negative controls registered by plan §13/§14.  Like P22's M0/M1/M2
# they are transformations of an already-collected timeline and never exist in
# the generator or in a model input.
MATCHED_CONTROLS = {
    "M0": "marginal-matched order-shuffled: keep each task's registered "
          "marginal windows but reveal the three resources in a permuted order "
          "(registered sequence destroyed)",
    "M1": "lag-shuffled: keep amplitude/duration/marginals, re-pair the "
          "windows with uniformly drawn lags (registered lags destroyed)",
    "M2": "within-task circular shift: shift each task's response trajectory, "
          "preserving marginals and autocorrelation but destroying the "
          "registered causal alignment",
}

_NUMERIC_KEYS = ("cpu_burst_mult", "cpu_burst_floor", "cpu_burst_upper",
                 "ram_ramp_peak_mult", "ram_target_floor", "ram_burst_upper",
                 "disk_retained_peak", "disk_retained_cap")

# Keys a regime may inherit from the family dict rather than declaring itself.
# ``*_duration`` and ``ram_ramp_peak_mult`` are deliberately NOT here (the
# generator validates below that each regime declares them explicitly).
_SHARED_FAMILY_KEYS = ("cpu_burst_mult", "cpu_burst_onset", "cpu_burst_floor",
                       "cpu_burst_upper", "ram_target_floor",
                       "ram_burst_upper", "disk_retained_peak",
                       "disk_retained_cap", "observable_history",
                       "forbidden_inputs")


def _phase_shape(duration, rise=None):
    """Deviation shape of one phase over its registered window.

    Protocol 022's phase law, sampled exactly as P22 samples it:

        offset 0            : 0 (the phase starts at the familiar level, which
                              is what makes every phase admission-safe, P21-01)
        offset 1            : the registered level
        interior offsets    : held
        last offset         : released

    ``rise`` selects the onset shape.  ``None`` or a positive value uses
    ``_trapezoid`` -- P22 computes both its CPU burst and its RAM/disk
    deviations from that helper, and for the RAM/disk phases its default
    ``rise=0.25`` reaches the registered level at offset 1 and returns to 0 on
    the last offset, so the same expression reproduces them byte for byte.
    ``rise == 0.0`` gives the step-then-linear-release shape, which is the
    phase law used by the newly registered memory-first and io-first orders.
    """
    duration = int(duration)
    if duration <= 0:
        return np.zeros(0, dtype=float)
    out = np.zeros(duration, dtype=float)
    for k in range(duration):
        fraction = k / float(max(duration - 1, 1))
        if rise is None or rise > 0.0:
            out[k] = _trapezoid(fraction, rise=(0.25 if rise is None else rise))
        else:
            out[k] = _step_release(fraction)
    return out


def _step_release(fraction):
    """Step to 1 at the first interior offset, then linear release to 0."""
    if fraction <= 0.0:
        return 0.0
    if fraction >= 1.0:
        return 0.0
    return 1.0 - fraction


class Protocol023MultiRegimeBWGD2(Protocol021CascadeBWGD2):
    """Protocol-021 cascade machinery generalized to the ``cascade_v3`` family.

    ``cascade_probability=0`` reproduces ``Protocol020AdaptedBWGD2`` exactly for
    the same replay seed and cohort (the inherited T-AUDIT-05 identity).  With
    ``registered_regimes=("compute_first",)`` the generator is byte-equivalent
    to Protocol 022's ``cascade_v2`` for the same mechanism seed.
    """

    def __init__(self, mean, sigma, replay_seed, cohort="dev", split_path=None,
                 adapter=None, disk_law_path=None, cascade=None,
                 cascade_probability=None, mechanism_seed=None,
                 registered_regimes=None, active_regime=None):
        params = dict(REGIMES_V3["compute_first"])
        for key, value in (cascade or {}).items():
            if key not in params:
                raise ValueError("Unknown cascade_v3 parameter %r" % key)
            params[key] = value
        if mechanism_seed is not None:
            params["mechanism_seed"] = int(mechanism_seed)
        if split_path is None:
            from .BitbrainWorkloadProtocol020 import SPLIT_PATH
            split_path = SPLIT_PATH
        regimes = tuple(registered_regimes or REGIME_IDS)
        for regime_id in regimes:
            if regime_id not in REGIMES_V3:
                raise ValueError("Unregistered regime id %r (have %s)"
                                 % (regime_id, sorted(REGIMES_V3)))
        if len(set(regimes)) != len(regimes):
            raise ValueError("Duplicate registered regime: %r" % (regimes,))
        active = active_regime if active_regime is not None else regimes[0]
        if active not in regimes:
            raise ValueError("Active regime %r is not registered %r"
                             % (active, regimes))
        # Protocol-021's constructor validates the shared envelope keys; it is
        # handed the compute-first key set, which is the registered common set
        # plus A's chain geometry.
        inherited_keys = tuple(CASCADE_V1)
        envelope = {k: v for k, v in params.items() if k in inherited_keys}
        super().__init__(mean, sigma, replay_seed, cohort=cohort,
                         split_path=split_path, adapter=adapter,
                         disk_law_path=disk_law_path, cascade=envelope,
                         cascade_probability=cascade_probability,
                         mechanism_seed=params["mechanism_seed"])
        self.cascade = params            # full registered dict, for the audit
        self.registered_regimes = tuple(regimes)
        self._active_regime = active
        self.mechanism_seed_by_regime = {
            r: int(REGIME_MECHANISM_SEEDS[r]) for r in self.registered_regimes}
        self.cascade_events = []
        self._cascade_event_of = {}
        registered_probability = float(
            params["cascade_task_probability"]
            if cascade_probability is None else cascade_probability)
        self.cascade_probability = registered_probability
        self._chain_geometry = {}
        self._validate_family()

    # -- registration validation -------------------------------------------
    def _validate_family(self):
        seeds = {}
        for regime_id in self.registered_regimes:
            regime = REGIMES_V3[regime_id]
            seed = int(self.mechanism_seed_by_regime[regime_id])
            if seed in seeds:
                raise ValueError("cascade_v3 regimes %s and %s share mechanism "
                                 "seed %d; their envelopes would be identical"
                                 % (seeds[seed], regime_id, seed))
            seeds[seed] = regime_id
            for key in ("regime_id", "onset_resource", "sequence",
                        "response_windows"):
                if key not in regime:
                    raise ValueError("cascade_v3 %s is missing %s"
                                     % (regime_id, key))
            if regime["regime_id"] != regime_id:
                raise ValueError("cascade_v3 key/id mismatch: %r != %r"
                                 % (regime_id, regime["regime_id"]))
            sequence = [list(pair) for pair in regime["sequence"]]
            if sorted(pair[0] for pair in sequence) != ["cpu", "disk", "ram"]:
                raise ValueError("cascade_v3 %s sequence must visit cpu, ram "
                                 "and disk exactly once: %r"
                                 % (regime_id, sequence))
            if sequence[0][0] != regime["onset_resource"]:
                raise ValueError("cascade_v3 %s onset_resource %r is not the "
                                 "first sequence element %r"
                                 % (regime_id, regime["onset_resource"],
                                    sequence[0][0]))
            history = int(self.cascade["observable_history"])
            for resource, lag in sequence:
                if lag >= history:
                    raise ValueError("cascade_v3 %s lag %d of %s is not "
                                     "observable inside observable_history=%d"
                                     % (regime_id, lag, resource, history))
            longest = 0
            regime_params = self.regime_params(regime_id)
            for resource, lag in sequence:
                duration = int(regime_params["%s_duration" % resource][1])
                longest = max(longest, lag + duration - 1)
            if longest >= history and regime_id not in INHERITED_CHAIN_REGIMES:
                raise ValueError("cascade_v3 %s longest chain age %d is not "
                                 "observable inside observable_history=%d"
                                 % (regime_id, longest, history))
            self._chain_geometry[regime_id] = {
                "longest_chain_age": int(longest),
                "observable_history": int(history),
                "chain_fully_observable": bool(longest < history),
                "chain_geometry_inherited": bool(
                    regime_id in INHERITED_CHAIN_REGIMES),
            }

    # -- active regime (phase switch, deterministic) ------------------------
    @property
    def active_regime(self):
        return self._active_regime

    def set_active_regime(self, regime_id, probability=None):
        """Phase-level regime switch; affects tasks created from now on.

        Already adapted containers keep the envelope they were created under,
        which is the same rule Protocol 020 uses for adapters.
        """
        if regime_id is not None and regime_id not in self.registered_regimes:
            raise ValueError("Regime %r is not registered %r"
                             % (regime_id, self.registered_regimes))
        self._active_regime = regime_id
        if probability is not None:
            probability = float(probability)
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("cascade_task_probability must be in [0, 1]: "
                                 "%r" % probability)
            self.cascade_probability = probability
            self.cascade["cascade_task_probability"] = probability
        return {"regime_id": regime_id, "cascade_task_probability":
                self.cascade_probability}

    def regime_params(self, regime_id=None):
        """Registered parameters of one regime.

        The registered family shares most numerics but *not* the phase duration
        bands: compute_first keeps Protocol 022's CPU 3-5 / RAM 6-10 /
        Disk 6-10, while memory_first and io_first use 4-5 and 4-6.  Every
        consumer (the transform, the registry writer, the audit) goes through
        this merge, so a per-regime override can never be silently replaced by
        the family default.
        """
        regime_id = self._active_regime if regime_id is None else regime_id
        regime = dict(REGIMES_V3[regime_id])
        # Overlay the live family dict so the shared registered numerics can be
        # audited (and deliberately drifted in a test) in one place: the
        # transform, the registry writer and assert_registered_physics all read
        # this merge, so a drift cannot hide behind a pristine registry copy.
        # Only keys the regime does NOT override itself are overlaid -- the
        # phase duration bands and the ramp multiplier are per regime, and
        # letting the family default win would silently give memory_first
        # compute_first's durations (measured: the memory-first chain then no
        # longer fits the registered 12-interval observable history).
        for key in _SHARED_FAMILY_KEYS:
            if key in self.cascade:
                regime[key] = self.cascade[key]
        regime["cascade_task_probability"] = self.cascade_probability
        # the mechanism seed is per regime, never the shared family value
        regime["mechanism_seed"] = int(
            self.mechanism_seed_by_regime[regime_id])
        return regime

    # -- audit accessors (never used as model input) ------------------------
    def cascade_audit(self):
        payload = super().cascade_audit()
        payload["regime_id"] = "cascade_v3"
        payload["family"] = "cascade_v3"
        payload["registered_regimes"] = list(self.registered_regimes)
        payload["active_regime"] = self._active_regime
        payload["regimes"] = {r: self.regime_params(r)
                              for r in self.registered_regimes}
        payload["event_counts_by_regime"] = {
            r: sum(1 for e in self.cascade_events if e["regime_id"] == r)
            for r in self.registered_regimes}
        payload["matched_controls"] = dict(MATCHED_CONTROLS)
        payload["physics_note"] = (
            "three controlled unseen temporal resource-demand regimes that "
            "differ only in resource order and lag; every phase is a trapezoid "
            "that is exactly zero at age 0, which is required by "
            "Simulator.getPlacementPossible() evaluating demand at the "
            "admission interval (P21-01) and is not a claim about real "
            "industrial fault laws")
        payload["registered_probability_candidates"] = list(
            CASCADE_PROBABILITY_CANDIDATES)
        return payload

    def task_cascade_windows(self):
        """creation_id -> registered windows, for the task-level audit only."""
        out = {}
        for event in self.cascade_events:
            onset = event["onset_resource"]
            # The CPU phase's floor is registered as ``cpu_burst_floor`` (the
            # inherited Protocol-022 name); the RAM and disk phases are
            # registered as ``ram_floor`` / ``disk_floor``.  Looking up
            # ``cpu_floor`` would raise for every compute-first envelope.
            floor_key = "cpu_burst_floor" if onset == "cpu" else "%s_floor" % onset
            out[int(event["creation_id"])] = {
                "event_id": int(event["event_id"]),
                "regime_id": event["regime_id"],
                "onset_resource": onset,
                "sequence": [list(pair) for pair in event["sequence"]],
                "cpu_window": list(event["cpu_window"]),
                "ram_window": list(event["ram_window"]),
                "disk_window": list(event["disk_window"]),
                "onset_resource_window": list(event["%s_window" % onset]),
                "onset_threshold": float(event["onset_threshold"]),
                "onset_floor": float(event[floor_key]),
                "onset_peak_value": float(event["%s_peak_value" % onset]),
            }
        return out

    # -- envelope ----------------------------------------------------------
    def _envelope(self, creation_id):
        """Pure function of (replay_seed, creation_id, mechanism_seed).

        With a single registered regime the draw order is Protocol 022's
        exactly: one probability draw, then cpu/ram/disk durations.
        """
        params = self.cascade
        if self.cascade_probability <= 0.0 or self._active_regime is None:
            return None
        rng = _cascade_rng(self.replay_seed, creation_id, self.mechanism_seed)
        draw = float(rng.random())
        if draw >= self.cascade_probability:
            return None
        regime_id = self._active_regime
        regime = REGIMES_V3[regime_id]
        params = self.regime_params(regime_id)
        durations = {}
        for resource in ("cpu", "ram", "disk"):
            low, high = params["%s_duration" % resource]
            durations[resource] = int(rng.integers(low, high + 1))
        return {"draw": draw, "regime_id": regime_id,
                "durations": durations,
                "sequence": [list(pair) for pair in regime["sequence"]]}

    # -- transform ---------------------------------------------------------
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
        from .BitbrainWorkloadProtocol020 import _ScaledMarkovDisk
        disk = _ScaledMarkovDisk(self.disk_law, self.replay_seed, cid,
                                 a["disk_mult"])
        self.createdContainers[index] = (cid, self.createdContainers[index][1],
                                         ips, ram, disk)

    def _phase_envelope(self, duration, rise):
        """Full-length window shape, exactly zero outside the phase window.

        The shape is Protocol 022's own law for the three resource kinds:

            age 0 of the window : exactly 0 (the phase starts where the task is
                                  still at its familiar level, so every phase is
                                  admission-safe by construction, P21-01)
            interior            : 1 (the phase interpolates from the value at
                                  window start to its registered target)
            last age            : 0 (released; a discrete return to 0 rather
                                  than a second ramp, which is what P22 does)

        ``rise`` additionally gives the CPU burst a progressive onset on its
        first interior interval instead of a step, which P22 applies through
        its doubled ``cpu_burst_mult`` ceiling.
        """
        shape = _phase_shape(duration, rise)
        return shape

    def _phase_window(self, base, kind, params, rise, start, duration,
                      upper, target_values=None):
        """Apply one phase to a demand vector.

        ``kind`` is ``ramp`` (interpolate from the value at window start toward
        a target, clipped up to it) or ``retained`` (add the registered
        retained-data term).  With Protocol 022's registered values this
        reproduces ``Protocol021CascadeBWGD2._apply_cascade``'s RAM and disk
        formulas exactly, which is what makes the compute-first regime
        byte-identical to ``cascade_v2``.
        """
        transformed = np.array(base, dtype=float, copy=True)
        # ``base`` must be sampled from an untouched copy: the loop writes into
        # ``transformed`` as it goes, and reading a later age's baseline from
        # the array being written would let an earlier phase interval feed the
        # next one (measured: the RAM target then propagated backwards through
        # the whole window and the A-regime regression failed by 4493.8).
        reference = np.array(base, dtype=float, copy=True)
        shape = self._phase_envelope(duration, rise)
        peak = 0.0
        for k in range(min(shape.size, transformed.size - int(start))):
            age = int(start) + k
            if age >= transformed.size:
                break
            fraction = shape[k]
            base_value = float(reference[age])
            if kind == "ramp":
                if target_values is None:
                    target = max(float(upper), base_value)
                else:
                    target = max(float(target_values[age]), base_value)
                value = base_value + (target - base_value) * fraction
            else:
                value = base_value + float(upper) * fraction
            transformed[age] = value
            peak = max(peak, value)
        return transformed, float(peak)

    def _apply_regime(self, index, cid, interval, ips, ram, raw, raw_ram,
                      envelope):
        params = self.regime_params(envelope["regime_id"])
        a = self.adapter
        durations = envelope["durations"]
        sequence = envelope["sequence"]
        for resource, lag in sequence:
            start = int(lag)
            duration = int(durations[resource])
            length = raw.size if resource == "cpu" else raw_ram.size
            if length <= start + duration - 1:
                raise ValueError(
                    "Trace too short for the registered %s window of %s: "
                    "len=%d needs > %d (creation_id=%s)"
                    % (resource, envelope["regime_id"], length,
                       start + duration - 1, cid))
        starts = {resource: int(lag) for resource, lag in sequence}

        windows, peaks, floors = {}, {}, {}
        # ---- CPU ---------------------------------------------------------
        clipped = np.where(raw > 0,
                           np.clip(raw * a["cpu_mult"], a["cpu_lower"],
                                   a["cpu_upper"]), 0.)
        cpu_start, cpu_duration = starts["cpu"], int(durations["cpu"])
        # Protocol 022 shapes the CPU phase with a doubled burst ceiling and
        # multiplier; regimes whose CPU phase is a downstream response get the
        # same registered CPU level via the ramp law instead.
        if starts["cpu"] != 0:
            clipped, cpu_peak = self._phase_window(
                clipped, "ramp", params, None, cpu_start, cpu_duration,
                params["cpu_burst_upper"])
        else:
            ceiling = np.full(raw.shape, a["cpu_upper"], dtype=float)
            multiplier = np.full(raw.shape, a["cpu_mult"], dtype=float)
            ceiling[:cpu_duration] = params["cpu_burst_upper"]
            multiplier[:cpu_duration] = params["cpu_burst_mult"] * a["cpu_mult"]
            burst = np.clip(raw[:cpu_duration] * multiplier[:cpu_duration],
                            params["cpu_burst_floor"], params["cpu_burst_upper"])
            if burst.size:
                burst[0] = np.clip(raw[0] * a["cpu_mult"], a["cpu_lower"],
                                   a["cpu_upper"])
            rest = np.where(raw[cpu_duration:] > 0,
                            np.clip(raw[cpu_duration:] * a["cpu_mult"],
                                    a["cpu_lower"], a["cpu_upper"]), 0.)
            clipped = np.concatenate([burst, rest])
            cpu_peak = float(burst.max()) if burst.size else 0.0
        ips.ips_list = clipped.tolist()
        raw_max = float(ips.max_ips)
        scaled_max = float(np.clip(
            raw_max * params["cpu_burst_mult"] * a["cpu_mult"],
            params["cpu_burst_floor"], params["cpu_burst_upper"])) \
            if raw_max > 0 else 0.
        ips.max_ips = max(scaled_max, float(clipped.max()) if clipped.size else 0.)
        windows["cpu"] = [cpu_start, cpu_start + cpu_duration - 1]
        peaks["cpu"] = cpu_peak
        floors["cpu"] = float(params["cpu_burst_floor"])

        # ---- RAM ---------------------------------------------------------
        base = raw_ram * a["ram_mult"]
        if a.get("ram_upper") is not None:
            base = np.minimum(base, a["ram_upper"])
        ram_start, ram_duration = starts["ram"], int(durations["ram"])
        # Protocol 022's registered RAM target is per-age and deliberately uses
        # the *unclipped* raw demand (raw * ram_mult * ramp_peak_mult, clipped
        # into the registered band), while the familiar base it interpolates
        # from IS clipped to the adapter's ram_upper.  Both facts matter: a
        # target built from the clipped base caps every task at ram_upper and
        # changes the regime (measured: 1078.7 divergence on one task).
        ram_targets = np.clip(np.asarray(raw_ram, dtype=float) * a["ram_mult"]
                              * params["ram_ramp_peak_mult"],
                              params["ram_target_floor"],
                              params["ram_burst_upper"])
        ram_scaled, ram_peak = self._phase_window(
            base, "ramp", params, None, ram_start, ram_duration,
            params["ram_burst_upper"], target_values=ram_targets)
        ram.size_list = ram_scaled.tolist()
        ram.read_list = [1.] * len(ram.read_list)
        ram.write_list = [1.] * len(ram.write_list)
        windows["ram"] = [ram_start, ram_start + ram_duration - 1]
        peaks["ram"] = ram_peak
        floors["ram"] = float(params["ram_target_floor"])

        # ---- Disk --------------------------------------------------------
        disk_start, disk_duration = starts["disk"], int(durations["disk"])
        disk_base = _CascadeRetainedDisk(self.disk_law, self.replay_seed, cid,
                                         a["disk_mult"], start_age=disk_start,
                                         duration=disk_duration,
                                         peak=params["disk_retained_peak"],
                                         cap=params["disk_retained_cap"])
        windows["disk"] = [disk_start, disk_start + disk_duration - 1]
        disk_peak = 0.0
        for k in range(disk_duration):
            fraction = k / float(max(disk_duration - 1, 1))
            disk_peak = max(disk_peak, float(params["disk_retained_peak"])
                            * _trapezoid(fraction))
        peaks["disk"] = disk_peak
        floors["disk"] = float(params["disk_retained_peak"])
        self.createdContainers[index] = (cid, interval, ips, ram, disk_base)

        onset_resource = envelope["sequence"][0][0]
        onset_tau = REGISTERED_ONSET_TAU[onset_resource]
        event_id = len(self.cascade_events)
        event = {
            "event_id": event_id,
            "family": "cascade_v3",
            "regime_id": envelope["regime_id"],
            "mechanism_id": REGIMES_V3[envelope["regime_id"]]["mechanism_id"],
            "onset_resource": onset_resource,
            "sequence": sequence,
            "creation_id": int(cid),
            "creation_interval": int(interval),
            "probability_draw": float(envelope["draw"]),
            "cpu_window": windows["cpu"],
            "cpu_duration": int(durations["cpu"]),
            "cpu_onset_value": float(clipped[cpu_start]) if clipped.size else 0.0,
            "cpu_burst_floor": float(params["cpu_burst_floor"]),
            "cpu_peak_value": peaks["cpu"],
            "ram_window": windows["ram"],
            "ram_duration": int(durations["ram"]),
            "ram_window_start_value": float(ram_scaled[ram_start]),
            "ram_peak_value": peaks["ram"],
            "ram_floor": float(params["ram_target_floor"]),
            "disk_window": windows["disk"],
            "disk_duration": int(durations["disk"]),
            "disk_retained_peak": float(params["disk_retained_peak"]),
            "disk_peak_value": peaks["disk"],
            "disk_floor": float(params["disk_retained_peak"]),
            "onset_threshold": float(onset_tau),
            "onset_baseline_lag": 4,
        }
        self.cascade_events.append(event)
        self._cascade_event_of[int(cid)] = event_id

    def _apply_cascade(self, index, cid, interval, ips, ram, raw, raw_ram,
                       envelope):
        """Protocol-021 entry point, redirected to the regime transform.

        ``Protocol021CascadeBWGD2.adapt_new_tasks`` calls this method, so the
        single-pass transform and the "already executed" assertion stay the
        inherited ones.
        """
        return self._apply_regime(index, cid, interval, ips, ram, raw, raw_ram,
                                  envelope)

    # -- registration guards ------------------------------------------------
    def assert_registered_physics(self):
        """Fail loudly if the runtime regime drifted from the registration."""
        if self.mechanism_seed not in (MECHANISM_SEED_DEV,
                                       MECHANISM_SEED_CONFIRM):
            raise AssertionError("Unregistered mechanism seed %r"
                                 % (self.mechanism_seed,))
        for regime_id in self.registered_regimes:
            expected = REGIMES_V3[regime_id]
            observed = self.regime_params(regime_id)
            if int(observed["mechanism_seed"]) != int(
                    REGIME_MECHANISM_SEEDS[regime_id]):
                raise AssertionError(
                    "cascade_v3 %s mechanism_seed drifted: %r != %r"
                    % (regime_id, observed["mechanism_seed"],
                       REGIME_MECHANISM_SEEDS[regime_id]))
            for key, value in expected.items():
                if key.endswith("_duration") or key in (
                        "cascade_task_probability", "mechanism_seed"):
                    continue
                if isinstance(value, str) or isinstance(value, bool):
                    if observed[key] != value:
                        raise AssertionError(
                            "cascade_v3 %s %s drifted: %r != %r"
                            % (regime_id, key, observed[key], value))
                elif isinstance(value, (list, tuple)):
                    if list(observed[key]) != list(value):
                        raise AssertionError(
                            "cascade_v3 %s %s drifted: %r != %r"
                            % (regime_id, key, list(observed[key]),
                               list(value)))
                elif isinstance(value, dict):
                    if dict(observed[key]) != dict(value):
                        raise AssertionError(
                            "cascade_v3 %s %s drifted: %r != %r"
                            % (regime_id, key, observed[key], value))
                else:
                    if abs(float(observed[key]) - float(value)) > 1e-12:
                        raise AssertionError(
                            "cascade_v3 %s %s drifted: %r != %r"
                            % (regime_id, key, observed[key], value))
            for key in ("cpu_duration", "ram_duration", "disk_duration"):
                if list(observed[key]) != list(expected[key]):
                    raise AssertionError("cascade_v3 %s %s drifted: %r != %r"
                                         % (regime_id, key, observed[key],
                                            expected[key]))
        return True


def registered_regime(regime_id="compute_first", probability=None,
                      mechanism_seed=None):
    """Plain-dict view of one registered regime (for the registry writer)."""
    if regime_id not in REGIMES_V3:
        raise ValueError("Unregistered regime id %r" % (regime_id,))
    params = dict(REGIMES_V3[regime_id])
    if probability is not None:
        params["cascade_task_probability"] = float(probability)
    if mechanism_seed is not None:
        params["mechanism_seed"] = int(mechanism_seed)
    return params


def registered_family():
    """The whole registered family, keyed by regime id."""
    return {r: registered_regime(r) for r in REGIME_IDS}
