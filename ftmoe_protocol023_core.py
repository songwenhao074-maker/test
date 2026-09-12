"""Protocol 023 — multi-regime task-level audit instrument (P23-S0).

This module is the Protocol-023 *instrument*.  It is the P22 instrument
(``ftmoe_protocol022_core``) generalized from "one registered resource cascade"
to "N registered cascades that differ only in resource order and lag".  It is
again deliberately free of simulation, model and I/O side effects so the unit
tests can drive it with hand-built synthetic tasks (plan §4.3).

P22 API surface is preserved
----------------------------
Every P22 name is re-exported from here unchanged (``build_task_timeline``,
``timeline_integrity``, ``migration_continuity``, ``onset_positions``,
``onset_events``, ``event_responses``, the M0/M1/M2 controls, the statistics
helpers, ``timeline_sha256``, ...), and the A regime is *physically identical*
to the P22 cascade_v2 registration, so a reader can keep using the P22 functions
for the A regime:

    from ftmoe_protocol023_core import onset_events          # P22, regime A
    from ftmoe_protocol023_core import onset_events_for_regime   # P23, any regime

``onset_events_for_regime(timeline, "A")`` and ``p22.onset_events(timeline)``
return the same onsets, the same baselines and the same response statistics; the
P23 row only carries additional columns (``onset_resource``, ``regime_id``,
``delta_at_onset``, ...).  ``test_ftmoe_protocol023_core`` pins that equivalence,
so the P22 registration cannot silently drift while P23 is built on top of it.

Registered regimes (plan §5-§7, §8)
-----------------------------------
Three cascading regimes that differ ONLY in resource order and lag:

    A "compute_first" : CPU burst --lag 4--> RAM ramp  --lag 8--> Disk retained
    B "memory_first"  : RAM ramp  --lag 3--> Disk retained --lag 6--> CPU burst
    C "io_first"      : Disk accum --lag 3--> CPU burst --lag 6--> RAM ramp

The *onset trigger resource* therefore differs per regime (A: cpu, B: ram,
C: disk) and so do the response resources and their lags.  ``REGIMES`` carries
the registered constants; the audit event row copies ``regime_id`` from a
caller-supplied mapping and never infers it from data.  The P23 generator module
registers the same three regimes under the names ``compute_first`` /
``memory_first`` / ``io_first`` and stamps those names on its audit rows, so
``REGIME_ALIASES`` translates a *registered* alias to the canonical id (any
other caller-supplied id is copied verbatim): an audit wired to the generator
cannot silently match no task because it compared "A" against "compute_first".

PROVISIONAL thresholds (must be finalised against measured familiar clips)
-------------------------------------------------------------------------
The A-regime constants are **inherited verbatim from Protocol 022**::

    onset_resource "cpu", onset_tau 2600.0, sequence [cpu 0, ram 4, disk 8],
    response_windows {ram: (4, 14), disk: (8, 18)}

P22's 2600.0 is not a guess: the familiar per-task CPU clip is 1860.0 in every
auditable corpus (measured, ``offline_reference_v2.json`` -> familiar_maxima)
and the registered CPU burst floor is 4400.0, so 2600 sits strictly between the
familiar maximum and the burst floor and *only* a registered burst can cross it.

The B and C threshold values are **PROVISIONAL** and must be finalised against
measured familiar per-task clips before any formal Protocol-023 collection:

    familiar per-task clips   cpu 1860.0   ram 1400.0   disk top of the law ~9000.0
    B  onset_tau_ram   = 2810.0   provisional
    C  onset_tau_disk  = 11600.0  provisional

Both provisional values were placed the same way P22 placed its CPU threshold
— strictly above the familiar clip and strictly below the regime's registered
burst level — but the "familiar maximum" half of that argument is *inherited
from the registration*, not measured:

    * RAM    : familiar clip 1400.0 (P22 ``FAMILIAR_TASK_LEVEL["ram"]``); the
               registered RAM ramp target is 4500-6000, so 2810 sits between
               them.  Unlike CPU, no offline corpus has yet been measured for a
               per-task RAM maximum under the P23 protocol.
    * Disk   : the P22 disk law tops out near 9000.0 per task; the registered
               retained term is 16000-24000, so 11600 sits between them.  The
               disk law is a *fitted law*, not a clip, so its per-task maximum
               must be measured on the P23 familiar corpus before the C
               threshold is frozen.

Until that measurement exists, B and C onsets are *provisional-detector*
evidence: a comfortable margin is registered on both sides, but the "no familiar
task can produce an event" claim that holds for A is not yet measured for B/C.
``assert_registered_regimes()`` reports the provisional taus explicitly and
``PROVISIONAL_TAUS`` names them, so an artifact reader cannot mistake them for
measured constants.  ``onset_provisional`` on every event row records that the
event came from a provisional detector.  The P23 generator module registers the
same three threshold values (its ``REGISTERED_ONSET_TAU``) and keeps a
probability-0 familiar stream as the calibration source; this instrument leaves
``cpu`` out of ``PROVISIONAL_TAUS`` only because P22 *measured* that basis, not
because the P23 corpus has re-measured it.  Nothing here is final until that
familiar stream is collected and the measured per-task RAM and disk maxima are
compared against 2810.0 and 11600.0 (and, for completeness, 2600.0).

Onset / response definitions (generalized from P22 §6.2-6.4)
------------------------------------------------------------
    onset(R)        : the first observed interval at which the task's *own*
                      ``R`` column reaches ``onset_tau[R]``, given that the
                      previous observed interval was strictly below it; a
                      sustained burst counts once.  Baseline-free, for the same
                      reason as P22-01 (the registered envelope is admitted at
                      the familiar level and bursts at age 1, so a median-baseline
                      jump has no pre-onset history to compare against).
    response(Q)     = max Q_task[t0+lag_Q : t0+lag_Q+band_Q] - baseline_Q,
                      with ``lag_Q``/``band_Q`` read from the regime's registered
                      ``response_windows`` (window start = registered lag; window
                      width = the registered 10-interval search band, which is a
                      superset of the generator's 4-6 interval ramp duration --
                      exactly the construction P22 uses).
    baseline_R      = median of the task's own pre-onset rows when they exist,
                      else the registered familiar task level
                      (``FAMILIAR_TASK_LEVEL``); the source is reported per event.
    delta_at_onset  = R_task[t0] - baseline_R.

A *cross-regime confound the reviewer must keep in view*: the three regimes are
registered on the same three resources, so a regime-A cascade's RAM ramp
(4500-6000) is above regime B's RAM onset threshold (2810) and its Disk
retention (16000-24000) is above regime C's Disk threshold (11600).  Scanning a
mixed stream by resource+tau alone therefore reports "B onsets" and "C onsets"
on tasks that are registered as A.  Pass ``task_regime_map`` (``{creation_id:
regime_id}``) and the onset scan is restricted to the tasks registered to the
scanned regime: that is what "regime_id is copied from the caller-supplied
mapping, never inferred" buys.  ``CrossRegimeConfoundTests`` pins this.

Marginal matching vs joint structure (plan §9, §13)
---------------------------------------------------
Plan §9 requires the three regimes to MATCH on overall anomaly prevalence, event
duration, event count, deployment/migration rejection, resource peak ratio and
normal host-step count, while differing in resource order, lag and
cross-resource dependency.  Two pure report functions implement that:

    marginal_match_report(per_regime_stats)   -- the registered data gate
    regime_contrast_report(regime_stats)      -- per pair: matched marginals side
                                                 by side with the (allowed) joint
                                                 structure differences

Purity / determinism
--------------------
Everything here is a pure function of its inputs: no globals are mutated (the
registered tables are deep-copied on the way out of the accessors), nothing is
written to disk, and no RNG is drawn without an explicit ``rng`` argument.  The
only RNG use is inside functions that take an ``rng`` parameter.

NaN policy (the P22 probe burn)
-------------------------------
P22 lost a full round to a NaN-poisoned probe.  A non-finite value in a feature
column is silent poison: every comparison against NaN is False, so a poisoned
column yields a *constant* predictor (or an all-False response flag) with no
error.  The instrument therefore validates the observed resource columns it
consumes (``assert_finite_features``) and raises ``ValueError`` instead of
quietly dropping an event.  NaN *padding* created by the instrument itself for
intervals beyond a task's observed life is legitimate and stays NaN, exactly as
in P22 (``response_curves``); only NaNs that are present in the data raise.
"""
import copy
import itertools

import numpy as np

import ftmoe_protocol022_core as p22

# --------------------------------------------------------------------------
# P22 API surface, re-exported unchanged (a reader may keep using the P22
# functions for the A regime).
# --------------------------------------------------------------------------
from ftmoe_protocol022_core import (  # noqa: F401
    DISK_RESPONSE_WINDOW,
    FAMILIAR_TASK_LEVEL,
    FORBIDDEN_INPUT_TOKENS,
    IDX_CPU,
    IDX_DISK,
    IDX_RAM,
    ONSET_BASELINE_LAG,
    ONSET_TAU_CALIBRATED,
    ONSET_TAU_CPU,
    RAM_RESPONSE_WINDOW,
    REGISTERED_LAG_DISK,
    REGISTERED_LAG_RAM,
    alignment_rate,
    bootstrap_ci,
    build_task_timeline,
    control_m0_pairs,
    control_m1_order,
    control_m2_circular,
    delta_cpu,
    event_responses,
    exceedance_probability,
    forbidden_token_hits,
    migration_continuity,
    onset_events,
    onset_positions,
    permutation_pvalue,
    pooled_percentile,
    response_curves,
    response_summary,
    response_summary_from_values,
    timeline_integrity,
    timeline_sha256,
)

__all__ = [
    # P22 re-exports
    "DISK_RESPONSE_WINDOW", "FAMILIAR_TASK_LEVEL", "FORBIDDEN_INPUT_TOKENS",
    "IDX_CPU", "IDX_DISK", "IDX_RAM", "ONSET_BASELINE_LAG",
    "ONSET_TAU_CALIBRATED", "ONSET_TAU_CPU", "RAM_RESPONSE_WINDOW",
    "REGISTERED_LAG_DISK", "REGISTERED_LAG_RAM", "alignment_rate",
    "bootstrap_ci", "build_task_timeline", "control_m0_pairs",
    "control_m1_order", "control_m2_circular", "delta_cpu", "event_responses",
    "exceedance_probability", "forbidden_token_hits", "migration_continuity",
    "onset_events", "onset_positions", "permutation_pvalue",
    "pooled_percentile", "response_curves", "response_summary",
    "response_summary_from_values", "timeline_integrity", "timeline_sha256",
    # P23 additions
    "RESOURCES", "REGIME_IDS", "REGIMES", "REGIME_ALIASES", "FORBIDDEN_INPUTS",
    "FAMILIAR_CLIP", "DISK_LAW_TOP", "PROVISIONAL_TAUS",
    "RESPONSE_WINDOW_WIDTH_BAND", "MARGINAL_MATCH_THRESHOLDS",
    "CONTRAST_TOLERANCES", "REGISTERED_RESPONSE_WINDOWS",
    "ONSET_RESOURCE_TO_REGIME", "regime", "canonical_regime_id",
    "registered_lag",
    "onset_positions_multi", "onset_events_multi", "onset_events_for_regime",
    "response_curves_multi", "event_responses_multi", "response_durations",
    "peak_ratio_summary", "alignment_rate_multi", "control_m0_multi",
    "control_m1_multi", "marginal_match_report", "regime_contrast_report",
    "assert_finite_features", "feature_finiteness_report",
    "assert_registered_regimes", "forbidden_input_hits",
]

# --------------------------------------------------------------------------
# registered resources and the three registered regimes (plan §5-§7)
# --------------------------------------------------------------------------
RESOURCES = ("cpu", "ram", "disk")

# Familiar per-task levels (the P22 registration; the disk entry is the top of
# the fitted familiar disk law, ~9000, not a hard clip like CPU/RAM).
FAMILIAR_CLIP = dict(FAMILIAR_TASK_LEVEL)
DISK_LAW_TOP = 9000.0

# Onset thresholds whose "no familiar task can cross it" half of the P22
# argument is NOT yet measured under P23 (see the module docstring).
PROVISIONAL_TAUS = {"ram": 2810.0, "disk": 11600.0}

# Registered response window width, inherited from the P22 envelope (P22's
# response windows are exactly 10 intervals wide, its ram/disk duration bands
# being 6-10).  Note for the reader: the P23 generator module registers 4-6
# interval phases for B/C so that the longest chain stays inside the 12-interval
# observable history, so these audit windows are a SUPERSET of the registered
# ramp duration -- which can only make detection easier.  The plan §9 "event
# duration" marginal is measured from the data (``response_durations``), never
# from the window width.
RESPONSE_WINDOW_WIDTH_BAND = (6, 10)

# Plan §4: these may be used to *label* an audit event, never as a model input.
FORBIDDEN_INPUTS = tuple(FORBIDDEN_INPUT_TOKENS) + ("mechanism_id", "event_id")

REGIMES = {
    "A": {
        "regime_id": "A",
        "name": "compute_first",
        "onset_resource": "cpu",
        "onset_tau": 2600.0,
        "sequence": [["cpu", 0], ["ram", 4], ["disk", 8]],
        "response_windows": {"ram": (4, 14), "disk": (8, 18)},
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    },
    "B": {
        "regime_id": "B",
        "name": "memory_first",
        "onset_resource": "ram",
        "onset_tau": 2810.0,
        "sequence": [["ram", 0], ["disk", 3], ["cpu", 6]],
        "response_windows": {"disk": (3, 13), "cpu": (6, 16)},
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    },
    "C": {
        "regime_id": "C",
        "name": "io_first",
        "onset_resource": "disk",
        "onset_tau": 11600.0,
        "sequence": [["disk", 0], ["cpu", 3], ["ram", 6]],
        "response_windows": {"cpu": (3, 13), "ram": (6, 16)},
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    },
}

REGIME_IDS = tuple(REGIMES)

# Names the same three regimes are registered under elsewhere.  The P23
# generator module registers them as ``compute_first`` / ``memory_first`` /
# ``io_first`` and stamps *that* string on its audit event rows, so an audit
# that compared the stamped value against "A" verbatim would silently match no
# task and report an empty regime.  Registered aliases are therefore translated
# to the canonical id; any other caller-supplied value is copied verbatim (see
# ``_canonical_or_copy``).
REGIME_ALIASES = {
    "A": "A", "a": "A", "regime_A": "A", "regime_a": "A",
    "compute_first": "A", "compute-first": "A",
    "B": "B", "b": "B", "regime_B": "B", "regime_b": "B",
    "memory_first": "B", "memory-first": "B",
    "C": "C", "c": "C", "regime_C": "C", "regime_c": "C",
    "io_first": "C", "io-first": "C",
}

# Convenience views of the registration.  Built once and never mutated.
ONSET_RESOURCE_TO_REGIME = {spec["onset_resource"]: rid
                            for rid, spec in REGIMES.items()}
REGISTERED_RESPONSE_WINDOWS = {
    spec["onset_resource"]: dict(spec["response_windows"])
    for spec in REGIMES.values()}

# --------------------------------------------------------------------------
# registered thresholds (plan §9 "建议 Data Gate")
# --------------------------------------------------------------------------
MARGINAL_MATCH_THRESHOLDS = {
    # plan §9 fixes only the prevalence spread (<= 4 percentage points); the
    # remaining numbers are the registered gate values quoted by plan §9 too.
    "prevalence_min": 0.03,
    "prevalence_max": 0.12,
    "prevalence_spread_max": 0.04,
    "independent_fault_events_min": 80,
    "valid_onset_followup_min": 50,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "worst_event_share_max": 0.10,
}

# Tolerances used by regime_contrast_report to decide whether a *marginal* was
# MATCHED across two regimes.  Only ``prevalence_diff_max`` is plan-registered
# (plan §9, 4 percentage points); plan §9 says the other marginals must be
# "as close as possible" without fixing a number, so the instrument registers
# its own and reports them verbatim in the returned ``thresholds`` dict.  These
# are instrument-local and may be revised before formal collection without
# touching the physical registration -- they never enter a gate that can be
# tuned on candidate model scores.
CONTRAST_TOLERANCES = {
    "prevalence_diff_max": 0.04,          # plan §9, registered
    "event_count_rel_diff_max": 0.10,     # instrument-local
    "mean_duration_rel_diff_max": 0.10,   # instrument-local
    "peak_ratio_rel_diff_max": 0.10,      # instrument-local
    "deployment_rejection_diff_max": 0.05,  # instrument-local
    "migration_rejection_diff_max": 0.05,   # instrument-local
    "normal_hoststeps_rel_diff_max": 0.05,  # instrument-local
}

# Accepted statistic keys of the per-regime stats dict passed to the two report
# functions.  The first alias of every entry is the canonical one used in the
# returned metrics.  Entries marked required are the quantities plan §9
# registers as matched marginals / gate checks; a stats dict that omits one is a
# hard error (a silently missing gate quantity would let a typo pass the gate).
_STAT_ALIASES = {
    "prevalence": ("prevalence",),
    "independent_fault_events": ("independent_fault_events",
                                 "independent_events",
                                 "n_independent_fault_events"),
    "valid_onset_followup": ("valid_onset_followup", "valid_onsets",
                             "valid_onset_follow_up", "n_valid_onset_followup"),
    "deployment_rejection_rate": ("deployment_rejection_rate",),
    "migration_rejection_rate": ("migration_rejection_rate",),
    "worst_event_share": ("worst_event_share",
                          "worst_event_share_of_positives"),
    "event_count": ("event_count", "n_events", "onset_events"),
    "mean_duration": ("mean_duration", "mean_event_duration"),
    "peak_ratio": ("peak_ratio", "peak_ratio_median", "median_peak_ratio"),
    # optional context, reported but not gated
    "host_steps": ("host_steps",),
    "positive_hoststeps": ("positive_hoststeps",),
    "deployment_attempts": ("deployment_attempts",),
    "deployment_rejected": ("deployment_rejected",),
    "migration_attempts": ("migration_attempts",),
    "migration_rejected": ("migration_rejected",),
}

_REQUIRED_STATS = ("prevalence", "independent_fault_events",
                   "valid_onset_followup", "deployment_rejection_rate",
                   "migration_rejection_rate", "worst_event_share",
                   "event_count", "mean_duration", "peak_ratio")


# --------------------------------------------------------------------------
# registration accessors (pure; deep copies so no caller can mutate REGIMES)
# --------------------------------------------------------------------------
def canonical_regime_id(name):
    """Canonical registered id ("A"/"B"/"C") for a registered alias.

    ``canonical_regime_id("compute_first") == "A"``.  Raises ``KeyError`` for a
    name that is not a registered alias, which is what the lookup functions want;
    the instrument's stamping path uses :func:`_canonical_or_copy` instead, so an
    unregistered caller-supplied id is copied verbatim rather than rejected.
    """
    key = str(name)
    if key in REGIME_ALIASES:
        return REGIME_ALIASES[key]
    raise KeyError("unregistered regime %r; registered ids are %s (aliases: %s)"
                   % (name, list(REGIME_IDS), sorted(REGIME_ALIASES)))


def _canonical_or_copy(name):
    key = str(name)
    return REGIME_ALIASES.get(key, key)


def regime(regime_id):
    """Deep copy of the registered constants of one regime (aliases accepted)."""
    canonical = canonical_regime_id(regime_id)
    return copy.deepcopy(REGIMES[canonical])


def registered_lag(regime_id, resource):
    """Registered causal lag of ``resource`` in ``regime_id`` (0 = trigger)."""
    spec = regime(regime_id)
    lags = {name: int(lag) for name, lag in spec["sequence"]}
    if resource not in lags:
        raise ValueError("resource %r is not part of regime %r (sequence %s)"
                         % (resource, regime_id, spec["sequence"]))
    return lags[resource]


def assert_registered_regimes(table=None):
    """Fail loudly if the registered regime table is internally inconsistent.

    Checks the three structural invariants that make the regimes comparable:
    each sequence starts at its own onset resource with lag 0, covers all three
    resources with strictly increasing lags, and the response windows cover
    exactly the two non-onset resources with their start equal to the registered
    lag and their length inside the registered response duration band.

    The threshold check is deliberately one-sided: it can only assert
    ``onset_tau > familiar clip`` (the provisional half of the P22 argument).
    The upper bound (``onset_tau < burst floor``) is a property of the generator
    registration, which this instrument must not import, and is left to the
    registration test of the workload module.
    """
    registered = REGIMES if table is None else table
    checks = {}
    problems = []
    checks["three_regimes_registered"] = len(registered) == 3
    for rid, spec in registered.items():
        sequence = [(str(r), int(l)) for r, l in spec["sequence"]]
        names = [r for r, _ in sequence]
        onset = spec["onset_resource"]
        others = [r for r in RESOURCES if r != onset]
        local = {
            "sequence_starts_at_the_onset_resource":
                bool(sequence) and names[0] == onset and sequence[0][1] == 0,
            "sequence_covers_every_resource": sorted(names) == sorted(RESOURCES),
            "lags_strictly_increasing":
                all(sequence[k][1] < sequence[k + 1][1]
                    for k in range(len(sequence) - 1)),
            "response_windows_cover_the_other_two":
                sorted(spec["response_windows"]) == sorted(others),
            "response_window_start_is_the_registered_lag":
                all(int(spec["response_windows"][r][0]) == lag
                    for r, lag in sequence if r != onset),
            "response_window_width_registered":
                all(RESPONSE_WINDOW_WIDTH_BAND[0]
                    <= int(w[1]) - int(w[0]) <= RESPONSE_WINDOW_WIDTH_BAND[1]
                    for w in spec["response_windows"].values()),
            "onset_tau_above_the_familiar_clip":
                float(spec["onset_tau"]) > float(FAMILIAR_CLIP[onset]),
            "onset_tau_is_finite":
                bool(np.isfinite(float(spec["onset_tau"]))),
        }
        for name, ok in local.items():
            checks["%s_%s" % (rid, name)] = bool(ok)
            if not ok:
                problems.append("%s.%s" % (rid, name))
    if problems:
        raise AssertionError("registered regime table drifted: %s"
                             % ", ".join(problems))
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "familiar_clip": dict(FAMILIAR_CLIP),
        "provisional_onset_taus": dict(PROVISIONAL_TAUS),
        "provisional_regimes": sorted(
            rid for rid, spec in registered.items()
            if spec["onset_resource"] in PROVISIONAL_TAUS),
        "note": ("B/C onset thresholds are provisional until the familiar "
                 "per-task RAM maximum and the familiar per-task disk maximum "
                 "are measured on a P23 familiar corpus"),
    }


def forbidden_input_hits(names, tokens=None):
    """Feature names containing a registered forbidden token are leaks.

    P23 adds ``mechanism_id`` and ``event_id`` to the P22 token list (plan §4),
    so this is a superset of :func:`ftmoe_protocol022_core.forbidden_token_hits`.
    """
    tokens = FORBIDDEN_INPUTS if tokens is None else tuple(tokens)
    hits = []
    for name in names:
        for token in tokens:
            if token in name:
                hits.append({"feature": name, "token": token})
    return hits


# --------------------------------------------------------------------------
# non-finite feature policy (the P22 NaN-poisoned-probe burn)
# --------------------------------------------------------------------------
def _column_views(features, names=None):
    if isinstance(features, dict):
        columns = []
        for name in sorted(features):
            columns.append((str(name),
                            np.asarray(features[name], dtype=np.float64).ravel()))
        return columns
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("features must be a 1-D/2-D array or a dict of "
                         "columns, got ndim=%d" % array.ndim)
    if names is None:
        names = ["column_%d" % j for j in range(array.shape[1])]
    if len(names) != array.shape[1]:
        raise ValueError("names has %d entries but features has %d columns"
                         % (len(names), array.shape[1]))
    return [(str(names[j]), array[:, j]) for j in range(array.shape[1])]


def feature_finiteness_report(features, names=None):
    """Per-column count and first index of non-finite (>NaN/inf) entries.

    Why a report and not just a check: P22 burned a whole round on a
    NaN-poisoned probe.  ``x >= tau`` against a NaN column is False without
    raising (and ``np.max`` of an all-NaN window is NaN), so a poisoned feature
    column silently produces a *constant* predictor -- an all-negative response
    flag or an all-False onset column -- instead of an error.  The report names
    the offending column and row so the failure is locatable.
    """
    columns = _column_views(features, names)
    per_column = {}
    total = 0
    for name, values in columns:
        bad = ~np.isfinite(values)
        count = int(bad.sum())
        total += count
        per_column[name] = {
            "n_entries": int(values.size),
            "n_non_finite": count,
            "first_index": (int(np.argmax(bad)) if count else None),
            "first_value": (float(values[int(np.argmax(bad))]) if count else None),
        }
    return {"names": [name for name, _ in columns],
            "n_rows": (int(columns[0][1].size) if columns else 0),
            "per_column": per_column,
            "n_non_finite": total,
            "ok": total == 0}


def assert_finite_features(features, names=None, context="feature matrix"):
    """Raise ``ValueError`` if any feature entry is NaN or infinite."""
    report = feature_finiteness_report(features, names)
    if report["n_non_finite"]:
        offenders = ["%s (%d non-finite, first at index %s)"
                     % (name, info["n_non_finite"], info["first_index"])
                     for name, info in sorted(report["per_column"].items())
                     if info["n_non_finite"]]
        raise ValueError(
            "%s contains non-finite values: %s.  A NaN/inf feature column is "
            "silent poison (every comparison against NaN is False, so the "
            "column behaves like a constant predictor); fix the data instead of "
            "letting the instrument drop or fake the event"
            % (context, "; ".join(offenders)))
    return report


# --------------------------------------------------------------------------
# resource helpers
# --------------------------------------------------------------------------
def _require_resource(resource):
    if resource not in RESOURCES:
        raise ValueError("unknown resource %r; registered resources are %s"
                         % (resource, list(RESOURCES)))
    return resource


def _resolve_response_windows(resource, response_windows):
    """(start, end) pairs per response resource; ``None`` = registered windows."""
    _require_resource(resource)
    if response_windows is None:
        windows = REGISTERED_RESPONSE_WINDOWS.get(resource)
        if windows is None:
            raise ValueError("no registered response windows for onset resource "
                             "%r; pass response_windows explicitly" % (resource,))
        response_windows = windows
    if resource in response_windows:
        raise ValueError("response_windows must not contain the onset resource "
                         "%r: the onset resource is the trigger, not a response"
                         % (resource,))
    unknown = sorted(set(response_windows) - set(RESOURCES))
    if unknown:
        raise ValueError("unknown response resource(s) %s; registered resources "
                         "are %s" % (unknown, list(RESOURCES)))
    missing = sorted(set(RESOURCES) - {resource} - set(response_windows))
    if missing:
        raise ValueError("registered response windows must cover both non-onset "
                         "resources; missing %s" % (missing,))
    out = {}
    for name, window in response_windows.items():
        pair = tuple(int(x) for x in window)
        if len(pair) != 2 or pair[0] < 0 or pair[1] <= pair[0]:
            raise ValueError("response window for %r must be a (start, end) "
                             "pair with 0 <= start < end, got %r"
                             % (name, window))
        out[name] = pair
    return out


def _resolve_regime_id(resource, regime_map):
    """The regime id stamped on every event row, *copied* from the caller map.

    ``regime_map`` is accepted in either registered orientation,
    ``{onset_resource: regime_id}`` (e.g. ``{"cpu": "A"}``) or
    ``{regime_id: onset_resource}`` (e.g. ``{"A": "cpu"}``); the value is copied
    verbatim and never inferred from the timeline, the resource column or tau.
    ``None`` falls back to the registered table ``ONSET_RESOURCE_TO_REGIME``,
    which is the registration, not an inference from data.
    """
    _require_resource(resource)
    if regime_map is None:
        rid = ONSET_RESOURCE_TO_REGIME.get(resource)
        if rid is None:
            raise ValueError("no registered regime triggers on %r; pass "
                             "regime_map explicitly" % (resource,))
        return rid
    if not isinstance(regime_map, dict):
        raise TypeError("regime_map must be a dict, got %s"
                        % type(regime_map).__name__)
    if resource in regime_map:
        return _canonical_or_copy(regime_map[resource])
    for rid, mapped in regime_map.items():
        if mapped == resource:
            return _canonical_or_copy(rid)
    raise ValueError("regime_map does not map onset resource %r; pass "
                     "{onset_resource: regime_id} (e.g. {'cpu': 'A'}) or "
                     "{regime_id: onset_resource}" % (resource,))


def _mapped_regime(task_regime_map, creation_id):
    for key in (creation_id, str(creation_id)):
        if key in task_regime_map:
            return _canonical_or_copy(task_regime_map[key])
    raise ValueError("task_regime_map has no entry for creation_id %r; every "
                     "task in the timeline must be registered to a regime when "
                     "the map is supplied" % (creation_id,))


# --------------------------------------------------------------------------
# generalized onset / response instrument
# --------------------------------------------------------------------------
def onset_positions_multi(values, tau, baseline_lag=ONSET_BASELINE_LAG,
                          resource=None):
    """Indices of registered onset events on ANY resource column.

    The rule is the registered P22 rule (``onset_positions``) applied to the
    named resource column, and it is *implemented by* that function so the two
    can never drift:

        the first observed interval at which the task's own <resource> reaches
        ``tau``, given that the previous observed interval was strictly below it

    and a sustained burst counts once.  ``baseline_lag`` is accepted for API
    parity with P22 (whose detector is baseline-free by registration) and is
    carried into the event row as provenance.
    """
    if resource is not None:
        _require_resource(resource)
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("onset tau must be finite, got %r" % (tau,))
    return p22.onset_positions(np.asarray(values, dtype=np.float64), tau=tau,
                               baseline_lag=int(baseline_lag))


def _deviation_vector(entry, index, resource, horizon):
    """Task's own values for ``[t0, t0+horizon)``, NaN past its observed life."""
    values = np.full(horizon, np.nan)
    column = entry[resource]
    take = max(min(horizon, column.size - int(index)), 0)
    if take:
        values[:take] = column[int(index):int(index) + take]
    return values


def _response_stats(deviation, window):
    """P22's response statistics for one resource through one window.

    Identical arithmetic to ``ftmoe_protocol022_core.event_responses``: the
    window is the half-open ``[window[0], window[1])`` slice of the deviation
    trajectory, non-finite padding is dropped before the max, and
    ``time_to_peak`` is reported as an offset from the onset (window start plus
    the arg-max inside the finite window).
    """
    values = deviation[window[0]:window[1]]
    values = values[np.isfinite(values)]
    exact = (deviation[window[0]] if window[0] < deviation.size else np.nan)
    response = float(np.max(values)) if values.size else float("nan")
    return {
        "response": response,
        "at_exact_lag": float(exact),
        "time_to_peak": (int(window[0] + int(np.argmax(values)))
                         if values.size else None),
        "response_positive": bool(np.isfinite(response) and response > 0),
    }


def _windows_from_event(event):
    """Response windows stamped on an event row by ``onset_events_multi``."""
    onset = event.get("onset_resource")
    out = {}
    for name in RESOURCES:
        key = "%s_window" % name
        if name != onset and key in event:
            out[name] = tuple(int(x) for x in event[key])
    return out


def response_curves_multi(timeline, event, response_windows=None,
                          onset_resource=None):
    """Per-resource deviation trajectories through the registered windows.

    Same object as P22's ``response_curves`` (a dense vector covering
    ``[t0, t0 + max(window))`` so re-pairing, circular shifting and duration
    counting operate on one representation), generalized to the regime's own
    response resources.  Windows default to the ones stamped on the event row.
    """
    onset = onset_resource or event.get("onset_resource")
    windows = (_resolve_response_windows(onset, response_windows)
               if response_windows is not None
               else _windows_from_event(event))
    if len(windows) != 2:
        raise ValueError("event row %r does not carry the two registered "
                         "response windows; build it with onset_events_multi "
                         "or pass response_windows" % (event.get("event_key"),))
    entry = timeline[event["creation_id"]]
    index = int(event["index"])
    horizon = max(window[1] for window in windows.values())
    baselines, deviations = {}, {}
    for name, _ in sorted(windows.items()):
        baseline = _event_baseline(event, name)
        baselines[name] = float(baseline)
        deviations[name] = (_deviation_vector(entry, index, name, horizon)
                            - float(baseline))
    return {"horizon": int(horizon),
            "windows": {name: tuple(window) for name, window in windows.items()},
            "baselines": baselines,
            "deviations": deviations}


def _event_baseline(event, resource):
    key = "%s_baseline" % resource
    if key not in event:
        raise KeyError("event %r carries no %s; build event rows with "
                       "onset_events_multi (P22 rows only carry the cpu/ram/disk "
                       "baselines of the P22 cascade)"
                       % (event.get("event_key"), key))
    return float(event[key])


def event_responses_multi(timeline, events, response_windows=None,
                          onset_resource=None, shifts=None, overrides=None):
    """Registered response statistics for every onset event, any regime.

    Generalizes ``ftmoe_protocol022_core.event_responses``: instead of the fixed
    ram/disk pair it uses the regime's registered ``response_windows``, and the
    shift/override controls are keyed by resource (``{"disk": 3}`` /
    ``{"disk": {event_key: vector}}``).  For the A regime, with the registered
    windows and no shift, the ram/disk entries are numerically identical to
    P22's ``event_responses`` output.
    """
    if response_windows is None:
        onset_resource = onset_resource or (events[0].get("onset_resource")
                                            if events else None)
        if onset_resource is None:
            raise ValueError("pass response_windows or event rows that carry "
                             "onset_resource")
        windows = _resolve_response_windows(onset_resource, None)
    else:
        if onset_resource is not None:
            windows = _resolve_response_windows(onset_resource, response_windows)
        else:
            if not events:
                raise ValueError("pass onset_resource (or non-empty events) so "
                                 "the onset resource of the windows is known")
            windows = _resolve_response_windows(events[0]["onset_resource"],
                                                response_windows)
    if onset_resource is not None:
        for event in events:
            stamped = event.get("onset_resource")
            if stamped is not None and stamped != onset_resource:
                raise ValueError("event %r was built for onset resource %r but "
                                 "this call registers %r"
                                 % (event.get("event_key"), stamped,
                                    onset_resource))
    horizon = max(window[1] for window in windows.values())
    rows = []
    for position, event in enumerate(events):
        entry = timeline[event["creation_id"]]
        index = int(event["index"])
        row = {
            "event_key": event["event_key"],
            "creation_id": int(event["creation_id"]),
            "t_onset": int(event["t_onset"]),
            "onset_resource": event.get("onset_resource"),
            "regime_id": event.get("regime_id"),
            "audit_event_id": (int(event["audit_event_id"])
                               if "audit_event_id" in event else None),
        }
        for name, window in sorted(windows.items()):
            deviation = (_deviation_vector(entry, index, name, horizon)
                         - _event_baseline(event, name))
            override = (overrides or {}).get(name)
            if override is not None and event["event_key"] in override:
                deviation = np.asarray(override[event["event_key"]],
                                       dtype=np.float64)
            shift = _resolve_shift(shifts, name, position)
            if shift:
                deviation = np.roll(deviation, shift)
            stats = _response_stats(deviation, window)
            row["applied_lag_shift_%s" % name] = int(shift)
            row["%s_window" % name] = [int(window[0]), int(window[1])]
            row["%s_registered_lag" % name] = int(window[0])
            row["%s_response" % name] = stats["response"]
            row["%s_at_exact_lag" % name] = stats["at_exact_lag"]
            row["%s_time_to_peak" % name] = stats["time_to_peak"]
            row["%s_response_positive" % name] = stats["response_positive"]
        rows.append(row)
    return rows


def _resolve_shift(shifts, resource, position):
    if not shifts:
        return 0
    value = shifts.get(resource, 0)
    if np.isscalar(value):
        return int(value)
    return int(np.asarray(value).ravel()[position])


def onset_events_multi(timeline, resource, tau, baseline_lag=ONSET_BASELINE_LAG,
                       response_windows=None, regime_map=None,
                       task_regime_map=None, strict_features=True):
    """One row per registered onset event on ``resource`` (any regime).

    Parameters
    ----------
    timeline : dict
        ``{creation_id: entry}`` as built by ``build_task_timeline``.
    resource : {"cpu", "ram", "disk"}
        The regime's registered onset (trigger) resource.
    tau : float
        The regime's registered onset threshold for that resource.
    baseline_lag : int
        Pre-onset window used for the reported baselines (P22 registers 4; the
        P23 plan registers no per-regime baseline window, so the P22 value is
        used for all three regimes and carried into the row as provenance).
    response_windows : dict or None
        ``{resource: (start, end)}`` with the *registered* lag window of each
        response resource; the window start is relative to the onset and its
        length is the registered response duration band, exactly as P22 does it.
        ``None`` uses the registered windows of the onset resource.
    regime_map : dict or None
        Caller-supplied mapping used to stamp ``regime_id`` on every row; see
        ``_resolve_regime_id``.  ``None`` uses the registered table.
    task_regime_map : dict or None
        Optional ``{creation_id: regime_id}``.  When supplied, only tasks whose
        registered regime equals the stamped ``regime_id`` produce events, which
        is the control for the cross-regime confound described in the module
        docstring (an A cascade's RAM ramp crosses the B threshold).
    strict_features : bool
        Validate the observed resource columns of every scanned task and raise
        on NaN/inf (default).  Never disable this on real data.

    Returns
    -------
    list of dict, ordered by ``(creation_id, t_onset)``, each carrying
    ``onset_resource``, ``regime_id``, ``t_onset``, ``age_onset``,
    ``host_onset``, ``delta_at_onset``, ``baseline_source``, the per-resource
    baselines, and the response statistics (``<resource>_response``,
    ``<resource>_at_exact_lag``, ``<resource>_time_to_peak``,
    ``<resource>_response_positive``) of the other two resources on the
    registered lag windows.
    """
    _require_resource(resource)
    windows = _resolve_response_windows(resource, response_windows)
    regime_id = _resolve_regime_id(resource, regime_map)
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("onset tau for %r must be finite, got %r"
                         % (resource, tau))
    baseline_lag = int(baseline_lag)
    if baseline_lag < 0:
        raise ValueError("baseline_lag must be >= 0, got %d" % baseline_lag)
    if task_regime_map is not None and not isinstance(task_regime_map, dict):
        raise TypeError("task_regime_map must be a dict, got %s"
                        % type(task_regime_map).__name__)

    events = []
    horizon = max(window[1] for window in windows.values())
    for creation_id in sorted(timeline):
        c = int(creation_id)
        entry = timeline[creation_id]
        if task_regime_map is not None:
            if _mapped_regime(task_regime_map, c) != regime_id:
                continue
        if strict_features:
            assert_finite_features(
                {name: entry[name] for name in RESOURCES},
                context="task %d observed resource columns" % c)
        positions = onset_positions_multi(entry[resource], tau,
                                          baseline_lag=baseline_lag,
                                          resource=resource)
        onset_column = entry[resource]
        for i in positions:
            index = int(i)
            t0 = int(entry["t"][index])
            low = max(index - baseline_lag, 0)
            if index > low:
                baselines = {name: float(np.median(entry[name][low:index]))
                             for name in RESOURCES}
                baseline_source = "observed pre-onset median"
                baseline_rows = int(index - low)
            else:
                # admitted at the familiar level and crossing on its very first
                # observed interval: no pre-onset rows of its own exist
                baselines = {name: float(FAMILIAR_TASK_LEVEL[name])
                             for name in RESOURCES}
                baseline_source = "registered familiar level (no pre-onset rows)"
                baseline_rows = 0
            value_at_onset = float(onset_column[index])
            row = {
                "event_key": "%d:%d" % (c, t0),
                "creation_id": c,
                "index": index,
                "t_onset": t0,
                "age_onset": int(entry["age"][index]),
                "host_onset": int(entry["host"][index]),
                "onset_resource": resource,
                "regime_id": regime_id,
                "onset_tau": tau,
                "onset_baseline_lag": baseline_lag,
                "onset_provisional": bool(resource in PROVISIONAL_TAUS),
                "value_at_onset": value_at_onset,
                "delta_at_onset": float(value_at_onset - baselines[resource]),
                "baseline_source": baseline_source,
                "baseline_rows": baseline_rows,
                "audit_event_id": int(entry["event_id"][index]),
            }
            for name in RESOURCES:
                row["%s_baseline" % name] = baselines[name]
            # P22-compatible aliases for the onset resource (cpu -> cpu_onset,
            # delta_cpu), so P22 readers keep working on the A regime.
            row["%s_onset" % resource] = value_at_onset
            row["delta_%s" % resource] = row["delta_at_onset"]
            for name, window in sorted(windows.items()):
                deviation = (_deviation_vector(entry, index, name, horizon)
                             - baselines[name])
                stats = _response_stats(deviation, window)
                row["%s_window" % name] = [int(window[0]), int(window[1])]
                row["%s_registered_lag" % name] = int(window[0])
                row["%s_response" % name] = stats["response"]
                row["%s_at_exact_lag" % name] = stats["at_exact_lag"]
                row["%s_time_to_peak" % name] = stats["time_to_peak"]
                row["%s_response_positive" % name] = stats["response_positive"]
            events.append(row)
    return events


def onset_events_for_regime(timeline, regime_id, baseline_lag=ONSET_BASELINE_LAG,
                            task_regime_map=None, strict_features=True):
    """``onset_events_multi`` using the registered constants of one regime.

    ``regime_id`` is copied from the registered table into a caller-built
    mapping, so the stamped value is still a *copy of a mapping*, never an
    inference from the scanned data.
    """
    spec = regime(regime_id)
    onset_resource = spec["onset_resource"]
    return onset_events_multi(
        timeline, onset_resource, spec["onset_tau"], baseline_lag=baseline_lag,
        response_windows=spec["response_windows"],
        regime_map={onset_resource: spec["regime_id"]},
        task_regime_map=task_regime_map, strict_features=strict_features)


# --------------------------------------------------------------------------
# response-derived marginals (plan §9: duration, peak ratio, alignment)
# --------------------------------------------------------------------------
def response_durations(timeline, events, response_windows=None,
                       onset_resource=None, positive_only=True):
    """Per-event response duration through the registered windows.

    Duration definition (instrument-local, reported so it is auditable): the
    number of intervals inside the two registered response windows at which the
    task's own deviation from its pre-onset baseline is strictly above zero,
    taking the longer of the two resources.  ``positive_only=False`` counts the
    full registered band instead.
    """
    per_event = []
    for event in events:
        curves = response_curves_multi(timeline, event,
                                       response_windows=response_windows,
                                       onset_resource=onset_resource)
        longest = 0
        for name, window in sorted(curves["windows"].items()):
            deviation = curves["deviations"][name][window[0]:window[1]]
            if positive_only:
                keep = deviation[np.isfinite(deviation)] > 0.0
                length = int(keep.sum())
            else:
                length = int(np.isfinite(deviation).sum())
            longest = max(longest, length)
        per_event.append({"event_key": event["event_key"],
                          "creation_id": int(event["creation_id"]),
                          "regime_id": event.get("regime_id"),
                          "duration": longest})
    values = np.asarray([row["duration"] for row in per_event],
                        dtype=np.float64)
    return {
        "per_event": per_event,
        "n": int(values.size),
        "mean": (float(values.mean()) if values.size else None),
        "p50": (float(np.percentile(values, 50)) if values.size else None),
        "max": (int(values.max()) if values.size else None),
        "definition": ("intervals inside the registered response windows whose "
                       "deviation from the pre-onset baseline is > 0, longest "
                       "of the two response resources"
                       if positive_only else
                       "width of the registered response window"),
    }


def peak_ratio_summary(rows, response_windows=None, onset_resource=None):
    """Per-event resource peak ratio (plan §9 "resource peak ratio" marginal).

    Instrument-local definition, reported so it is auditable: the largest
    dimensionless excursion of the two registered response resources,
    ``max_Q(response_Q / baseline_Q)`` with ``response_Q`` the registered window
    maximum and ``baseline_Q`` the task's own pre-onset baseline for ``Q``.
    Both response resources are expressed as multiples of the task's own
    familiar level, which is what makes the number comparable across regimes
    even though the resources differ.
    """
    windows = response_windows
    if windows is None:
        windows = _windows_from_event(rows[0]) if rows else {}
    if not windows:
        raise ValueError("pass response_windows or event rows that carry the "
                         "registered windows")
    if onset_resource is not None:
        windows = _resolve_response_windows(onset_resource, windows)
    per_event = []
    for row in rows:
        ratios = []
        for name in sorted(windows):
            baseline = _event_baseline(row, name)
            response = float(row["%s_response" % name])
            if np.isfinite(response) and baseline > 0.0:
                ratios.append(response / baseline)
        per_event.append({
            "event_key": row["event_key"],
            "creation_id": int(row["creation_id"]),
            "regime_id": row.get("regime_id"),
            "peak_ratio": (float(max(ratios)) if ratios else float("nan")),
        })
    values = np.asarray([row["peak_ratio"] for row in per_event],
                        dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "per_event": per_event,
        "n": int(values.size),
        "mean": (float(values.mean()) if values.size else None),
        "p50": (float(np.percentile(values, 50)) if values.size else None),
        "max": (float(values.max()) if values.size else None),
        "definition": ("max over the registered response resources of "
                       "response_Q / pre-onset baseline_Q"),
    }


def alignment_rate_multi(rows, resource, registered_lag, tolerance=1):
    """Share of events whose ``resource`` response peaks at the registered lag.

    P22-compatible generalization of ``alignment_rate``: a peak counts as
    aligned when ``|time_to_peak - registered_lag| <= tolerance``.

    Caveat carried over from P22 and reported rather than hidden: a deviation
    curve with no response at all (all zeros) has its arg-max at the window
    start by convention, so it *looks* aligned.  The restricted rate over events
    whose response is strictly positive is therefore reported next to the
    P22-compatible one; the gate must quote ``rate_positive`` when a regime's
    response can be flat.
    """
    _require_resource(resource)
    hits = counted = 0
    hits_positive = counted_positive = 0
    for row in rows:
        peak = row.get("%s_time_to_peak" % resource)
        if peak is None:
            continue
        counted += 1
        aligned = abs(int(peak) - int(registered_lag)) <= int(tolerance)
        if aligned:
            hits += 1
        if row.get("%s_response_positive" % resource):
            counted_positive += 1
            if aligned:
                hits_positive += 1
    return {
        "resource": resource,
        "registered_lag": int(registered_lag),
        "tolerance": int(tolerance),
        "n": counted,
        "hits": hits,
        "rate": (hits / counted) if counted else None,
        "n_positive": counted_positive,
        "hits_positive": hits_positive,
        "rate_positive": (hits_positive / counted_positive) if counted_positive
                         else None,
    }


# --------------------------------------------------------------------------
# matched controls, generalized to any pair of response windows
# --------------------------------------------------------------------------
def _m0_shift_choices(horizon):
    """Uniform lag shifts in ``1..horizon`` minus the no-op roll.

    A circular shift by exactly ``horizon`` on a length-``horizon`` deviation
    vector is the identity, so it must be excluded for the control to be
    "never the identity" (P22 drew from ``1..max(horizons)`` for both resources,
    which does not guarantee that).
    """
    choices = [s for s in range(1, int(horizon) + 1) if s % int(horizon) != 0]
    if not choices:
        raise ValueError("horizon %r is too small for a non-identity shift"
                         % (horizon,))
    return choices


def control_m0_multi(events, rng, response_windows, onset_resource=None):
    """Control M0 — marginal-matched, lag-shuffled, for a P23 regime.

    Every event keeps its own deviation trajectory (marginal amplitude, duration
    distribution, event count, task cohort and arrival process untouched) but
    the trajectory is circularly shifted by a uniformly drawn non-identity lag
    per resource, so the registered alignment is replaced by an arbitrary one.
    Deterministic for a given ``rng``.
    """
    if onset_resource is not None:
        windows = _resolve_response_windows(onset_resource, response_windows)
    else:
        windows = {name: tuple(int(x) for x in window)
                   for name, window in response_windows.items()}
    horizon = max(window[1] for window in windows.values())
    shifts = {}
    for name in sorted(windows):
        choices = _m0_shift_choices(horizon)
        shifts[name] = [int(choices[int(rng.integers(0, len(choices)))])
                        for _ in events]
    return {"kind": "marginal-matched lag-shuffled (per-event circular shift)",
            "shifts": shifts,
            "response_windows": {name: tuple(window)
                                 for name, window in windows.items()}}


def control_m1_multi(events, rng, response_windows, onset_resource=None):
    """Control M1 — order-shuffled, for a P23 regime.

    The same shift multiset ``{0..horizon}`` is assigned to the events in a
    random order per resource, so at most one event per resource keeps the
    registered alignment and every other event receives an arbitrary lag while
    the multiset of applied lags is exactly preserved.
    """
    if onset_resource is not None:
        windows = _resolve_response_windows(onset_resource, response_windows)
    else:
        windows = {name: tuple(int(x) for x in window)
                   for name, window in response_windows.items()}
    horizon = max(window[1] for window in windows.values())
    shifts = {}
    for name in sorted(windows):
        assigned = [i % (horizon + 1) for i in range(len(events))]
        shifts[name] = ([int(x) for x in rng.permutation(assigned)]
                        if assigned else [])
    return {"kind": "order-shuffled (permuted lag assignment)",
            "shifts": shifts,
            "response_windows": {name: tuple(window)
                                 for name, window in windows.items()}}


# --------------------------------------------------------------------------
# plan §9 / §13 report functions (pure: stats in, report out)
# --------------------------------------------------------------------------
_MISSING = object()


def _pick(stats, field):
    """First accepted alias present in ``stats``, else ``_MISSING``."""
    for alias in _STAT_ALIASES[field]:
        if alias in stats:
            return stats[alias]
    return _MISSING


def _normalize_regime_stats(regime_id, stats):
    """Canonical per-regime statistic dict, with loud failure on a typo.

    A registered quantity that is *absent* raises (a typo must not silently
    weaken the gate); a quantity that is present but ``None`` is carried through
    and fails its check, which is how a genuinely unmeasurable quantity is
    reported.  Rates and prevalence may be supplied either directly or as the
    counts they are computed from (``deployment_attempts``/``deployment_rejected``,
    ``positive_hoststeps``/``host_steps``), mirroring the P22 gate metric dicts.
    """
    if not isinstance(stats, dict):
        raise TypeError("stats for regime %r must be a dict, got %s"
                        % (regime_id, type(stats).__name__))
    values, present = {}, {}
    for field in _STAT_ALIASES:
        value = _pick(stats, field)
        present[field] = value is not _MISSING
        values[field] = None if value is _MISSING else value

    missing = []
    if not present["prevalence"] and not (present["positive_hoststeps"]
                                          and present["host_steps"]):
        missing.append("prevalence (or positive_hoststeps + host_steps)")
    for kind in ("deployment", "migration"):
        rate_field = "%s_rejection_rate" % kind
        if not present[rate_field] and not (present["%s_attempts" % kind]
                                            and present["%s_rejected" % kind]):
            missing.append("%s_rejection_rate (or %s_attempts + %s_rejected)"
                           % (kind, kind, kind))
    missing.extend(field for field in _REQUIRED_STATS if not present[field]
                   and field not in ("prevalence", "deployment_rejection_rate",
                                     "migration_rejection_rate"))
    if missing:
        raise ValueError(
            "stats for regime %r is missing the registered quantity/quantities "
            "%s; accepted keys are %s"
            % (regime_id, missing,
               {field: list(_STAT_ALIASES[field]) for field in _STAT_ALIASES}))

    out = dict(values)
    out["regime_id"] = regime_id
    out["raw"] = dict(stats)
    if not present["prevalence"]:
        host_steps = float(out["host_steps"])
        out["prevalence"] = (float(out["positive_hoststeps"]) / host_steps
                             if host_steps > 0 else None)
    for kind in ("deployment", "migration"):
        rate_field = "%s_rejection_rate" % kind
        if not present[rate_field]:
            attempts = float(out["%s_attempts" % kind])
            out[rate_field] = (float(out["%s_rejected" % kind]) / attempts
                               if attempts > 0 else None)
    # numeric canonicalization
    for field in ("prevalence", "deployment_rejection_rate",
                  "migration_rejection_rate", "worst_event_share",
                  "mean_duration", "peak_ratio"):
        if out[field] is not None:
            out[field] = float(out[field])
    for field in ("independent_fault_events", "valid_onset_followup",
                  "event_count", "host_steps", "positive_hoststeps",
                  "deployment_attempts", "deployment_rejected",
                  "migration_attempts", "migration_rejected"):
        if out[field] is not None:
            out[field] = int(out[field])
    return out


def _relative_difference(a, b):
    if a is None or b is None:
        return None
    scale = max(abs(float(a)), abs(float(b)))
    if scale == 0.0:
        return 0.0
    return abs(float(a) - float(b)) / scale


def marginal_match_report(per_regime_stats):
    """The plan §9 data gate over the registered marginal quantities.

    ``per_regime_stats`` maps ``regime_id -> stats dict`` with the registered
    quantities:

        prevalence                     (or positive_hoststeps/host_steps)
        independent_fault_events       >= 80 per regime
        valid_onset_followup           >= 50 per regime
        deployment_rejection_rate      <= 0.25 (or attempts+rejected counts)
        migration_rejection_rate       <= 0.40 (or attempts+rejected counts)
        worst_event_share              < 0.10
        event_count / mean_duration / peak_ratio   matched marginals

    and the cross-regime check that the anomaly prevalence spread is <= 0.04
    (plan §9, "regime prevalence difference <= 4 percentage points"); the plan's
    prevalence band 3%-12% is checked per regime as well.

    Pure function; returns a P22-style gate dict
    ``{"thresholds", "checks", "passed", "metrics"}`` plus ``per_regime``.
    """
    if not isinstance(per_regime_stats, dict):
        raise TypeError("per_regime_stats must be a dict {regime_id: stats}, "
                        "got %s" % type(per_regime_stats).__name__)
    if not per_regime_stats:
        raise ValueError("per_regime_stats is empty; the gate needs at least "
                         "one regime")
    thresholds = dict(MARGINAL_MATCH_THRESHOLDS)
    per_regime = {}
    for regime_id in sorted(per_regime_stats, key=str):
        canonical = _canonical_or_copy(regime_id)
        if canonical in per_regime:
            raise ValueError("two stats entries map to the same registered "
                             "regime %r: %r" % (canonical, regime_id))
        per_regime[canonical] = _normalize_regime_stats(
            canonical, per_regime_stats[regime_id])

    checks = {}
    for regime_id, stats in sorted(per_regime.items()):
        prevalence = stats["prevalence"]
        checks["%s_prevalence_in_range" % regime_id] = bool(
            prevalence is not None
            and thresholds["prevalence_min"] <= prevalence
            <= thresholds["prevalence_max"])
        events = stats["independent_fault_events"]
        checks["%s_independent_fault_events_enough" % regime_id] = bool(
            events is not None
            and events >= thresholds["independent_fault_events_min"])
        valid = stats["valid_onset_followup"]
        checks["%s_valid_onset_followup_enough" % regime_id] = bool(
            valid is not None
            and valid >= thresholds["valid_onset_followup_min"])
        deployment = stats["deployment_rejection_rate"]
        checks["%s_deployment_rejection_ok" % regime_id] = bool(
            deployment is not None
            and deployment <= thresholds["deployment_rejection_max"])
        migration = stats["migration_rejection_rate"]
        checks["%s_migration_rejection_ok" % regime_id] = bool(
            migration is not None
            and migration <= thresholds["migration_rejection_max"])
        worst = stats["worst_event_share"]
        checks["%s_worst_event_share_ok" % regime_id] = bool(
            worst is not None and worst < thresholds["worst_event_share_max"])

    prevalences = [stats["prevalence"] for stats in per_regime.values()
                   if stats["prevalence"] is not None]
    spread = (max(prevalences) - min(prevalences)) if len(prevalences) >= 2 \
        else None
    checks["prevalence_spread_ok"] = bool(
        spread is not None and spread <= thresholds["prevalence_spread_max"])
    checks["all_registered_regimes_reported"] = (
        sorted(per_regime) == sorted(REGIME_IDS))

    metrics = {
        "regimes_reported": sorted(per_regime),
        "registered_regimes": sorted(REGIME_IDS),
        "prevalence_spread": spread,
        "per_regime": per_regime,
    }
    return {"thresholds": thresholds, "checks": checks,
            "passed": all(checks.values()), "metrics": metrics,
            "per_regime": per_regime}


def regime_contrast_report(regime_stats):
    """Per-regime pair: matched marginals side by side with the joint structure.

    Plan §9 requires the regimes to MATCH on the marginal quantities
    (prevalence, event count, mean event duration, resource peak ratio,
    deployment/migration rejection, normal host-step count) while differing in
    the joint structure (resource order, lag, cross-resource dependency).  This
    report keeps the two apart so a reader can check exactly that:

        report["pairs"]["A|B"]["matched_marginals"]         -> the §9 quantities
        report["pairs"]["A|B"]["allowed_joint_differences"] -> order + lags
        report["pairs"]["A|B"]["marginals_matched"]         -> gate result

    The joint block is read from the *registration* (``REGIMES``), never from
    the statistics, so a stream cannot make two regimes look structurally
    different by chance.  Registered aliases (``compute_first`` -> ``A``) are
    accepted as keys.  Pure function; P22-style gate dict plus ``pairs``.
    """
    if not isinstance(regime_stats, dict) or not regime_stats:
        raise ValueError("regime_stats must be a non-empty dict "
                         "{regime_id: stats}")
    unknown = sorted(str(k) for k in regime_stats
                     if _canonical_or_copy(k) not in REGIMES)
    if unknown:
        raise ValueError("regime_contrast_report only compares registered "
                         "regimes %s; got %s" % (sorted(REGIME_IDS), unknown))
    thresholds = dict(CONTRAST_TOLERANCES)
    per_regime = {}
    for key, value in sorted(regime_stats.items(), key=lambda kv: str(kv[0])):
        canonical = _canonical_or_copy(key)
        if canonical in per_regime:
            raise ValueError("two stats entries map to the same registered "
                             "regime %r: %r" % (canonical, key))
        per_regime[canonical] = _normalize_regime_stats(canonical, value)

    checks = {"all_registered_regimes_reported":
              sorted(per_regime) == sorted(REGIME_IDS)}
    pairs = {}
    for left, right in itertools.combinations(sorted(per_regime), 2):
        a, b = per_regime[left], per_regime[right]
        spec_a, spec_b = REGIMES[left], REGIMES[right]
        prevalence_diff = abs(a["prevalence"] - b["prevalence"])
        event_count_diff = _relative_difference(a["event_count"],
                                                b["event_count"])
        duration_diff = _relative_difference(a["mean_duration"],
                                             b["mean_duration"])
        peak_diff = _relative_difference(a["peak_ratio"], b["peak_ratio"])
        deployment_diff = abs(a["deployment_rejection_rate"]
                              - b["deployment_rejection_rate"])
        migration_diff = abs(a["migration_rejection_rate"]
                            - b["migration_rejection_rate"])
        normal_a = (a["host_steps"] - a["positive_hoststeps"]
                    if a["host_steps"] is not None
                    and a["positive_hoststeps"] is not None else None)
        normal_b = (b["host_steps"] - b["positive_hoststeps"]
                    if b["host_steps"] is not None
                    and b["positive_hoststeps"] is not None else None)
        normal_diff = _relative_difference(normal_a, normal_b)

        matched = {
            "prevalence_diff": prevalence_diff,
            "prevalence_diff_ok":
                prevalence_diff <= thresholds["prevalence_diff_max"],
            "event_count_diff_rel": event_count_diff,
            "event_count_matched": bool(
                event_count_diff is not None
                and event_count_diff <= thresholds["event_count_rel_diff_max"]),
            "mean_duration_diff_rel": duration_diff,
            "mean_duration_matched": bool(
                duration_diff is not None
                and duration_diff <= thresholds["mean_duration_rel_diff_max"]),
            "peak_ratio_diff_rel": peak_diff,
            "peak_ratio_matched": bool(
                peak_diff is not None
                and peak_diff <= thresholds["peak_ratio_rel_diff_max"]),
            "deployment_rejection_diff": deployment_diff,
            "deployment_rejection_matched":
                deployment_diff <= thresholds["deployment_rejection_diff_max"],
            "migration_rejection_diff": migration_diff,
            "migration_rejection_matched":
                migration_diff <= thresholds["migration_rejection_diff_max"],
            "normal_hoststeps": {"%s" % left: normal_a, "%s" % right: normal_b},
            "normal_hoststeps_diff_rel": normal_diff,
            "normal_hoststeps_matched": bool(
                normal_diff is not None
                and normal_diff <= thresholds["normal_hoststeps_rel_diff_max"]),
        }
        order_a = [name for name, _ in spec_a["sequence"]]
        order_b = [name for name, _ in spec_b["sequence"]]
        lag_a = {name: int(lag) for name, lag in spec_a["sequence"]}
        lag_b = {name: int(lag) for name, lag in spec_b["sequence"]}
        joint = {
            "onset_resource": {left: spec_a["onset_resource"],
                               right: spec_b["onset_resource"]},
            "resource_order": {left: order_a, right: order_b},
            "lag_by_resource": {left: lag_a, right: lag_b},
            "response_windows": {
                left: {name: list(window)
                       for name, window in spec_a["response_windows"].items()},
                right: {name: list(window)
                        for name, window in spec_b["response_windows"].items()}},
            "onset_resource_differs":
                spec_a["onset_resource"] != spec_b["onset_resource"],
            "resource_order_differs": order_a != order_b,
            "lag_profile_differs": lag_a != lag_b,
            "dependency_pairs": {
                left: [[order_a[k], order_a[k + 1]]
                       for k in range(len(order_a) - 1)],
                right: [[order_b[k], order_b[k + 1]]
                        for k in range(len(order_b) - 1)]},
        }
        joint["joint_structure_differs"] = bool(
            joint["onset_resource_differs"] or joint["resource_order_differs"]
            or joint["lag_profile_differs"])
        pairs["%s|%s" % (left, right)] = {
            "regimes": [left, right],
            "matched_marginals": matched,
            "allowed_joint_differences": joint,
            "marginals_matched": all(
                matched[key] for key in
                ("prevalence_diff_ok", "event_count_matched",
                 "mean_duration_matched", "peak_ratio_matched",
                 "deployment_rejection_matched",
                 "migration_rejection_matched",
                 "normal_hoststeps_matched")),
        }
        checks["%s_%s_marginals_matched" % (left, right)] = \
            pairs["%s|%s" % (left, right)]["marginals_matched"]
        checks["%s_%s_joint_structure_differs" % (left, right)] = \
            joint["joint_structure_differs"]

    metrics = {"per_regime": per_regime, "pairs": pairs}
    return {"thresholds": thresholds, "checks": checks,
            "passed": all(checks.values()), "metrics": metrics, "pairs": pairs}
