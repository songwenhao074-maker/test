"""Protocol 023 S2 — multi-regime stream collector (development + single-regime).

This is the Protocol-023 sibling of ``prepare_ftmoe_protocol022_development.py``.
The P22 collector is the working reference and its loop, guards, memory bounds,
audit wiring and manifest layout are kept; the structural changes are the ones
Protocol 023 registers (plan §5-§9, §13):

  1. the generator is ``Protocol023MultiRegimeBWGD2`` (family ``cascade_v3``)
     with **all three** regimes registered in every run
     (``registered_regimes=REGIME_IDS``) and the registered development
     mechanism seed ``MECHANISM_SEED_DEV`` (= 22022, registered in the
     generator);
  2. one CLI switch, ``--mode``, selects the registered stream shape:
        dev     : F0_baseline 300, A1_compute 420, B1_memory 420, C1_io 420,
                  F1_baseline 240, A2_recur 360, C2_recur 360, B2_recur 360
                  = 2880 scored intervals
        single  : F0_baseline 150, <regime>_only 1050 = 1200 scored intervals
                  (``--regime`` is required)
     A familiar phase means ``set_active_regime(None, probability=0.0)`` (the
     frozen familiar generator exactly); a regime phase means
     ``set_active_regime(<regime>, probability=0.25)``.  The switch happens
     exactly at a phase boundary and is recorded in the manifest as
     ``phase_switch_points``.  A task created inside a phase keeps the envelope
     of the phase it was created under -- that is the generator's own semantics
     and the collector never re-adapts an existing container;
  3. per-slot / per-host regime columns are added next to the P22 phase columns:
     ``cascade_regimes`` (the generator's ``mechanism_id`` 0/1/2, -1 for a
     non-cascade task), ``host_cascade_regime`` and ``host_cascade_mask``.
     Every phase/regime column is AUDIT-ONLY and must never be a model input;
  4. the task-level audit uses the Protocol-023 instrument
     (``ftmoe_protocol023_core``), not the CPU-only P22 one: the timeline is
     built with the P22 core builder (unchanged row layout) plus a per-task
     regime map, and each regime is scanned with
     ``core23.onset_events_multi(..., resource=<regime onset resource>,
     tau=REGISTERED_ONSET_TAU[resource], response_windows=<regime windows>,
     task_regime_map=<creation_id -> regime_id>)``.  The gate is therefore
     computed on a PER-REGIME TASK COHORT: the instrument's own tests show that
     a mixed scan inflates every regime with cross-hits, so the unmapped scan is
     reported next to the mapped one as an explicit confound count instead of
     being used as a gate input.

Outputs (same schema/keys as P22 plus the regime columns and per-regime blocks)
under ``artifacts/ftmoe_online/protocol_023/development_streams/<tag>/``:

    stream.npz              model inputs + audit-only cascade/phase/regime columns
    task_timeline.npz       task-level observations (audit only)
    task_event_index.json   onset index + cascade windows (audit only)
    manifest.json           configuration, hashes, phase structure, gate aggregates
    events.json             cascade envelopes + per-phase / per-regime fault segmentation
    unseen_data_audit.json  gate evaluation, reported per phase and per regime
    generation.log          the simulator's own stdout/stderr

``<tag>`` is ``dev_seed700_steps2880`` in dev mode and
``single_<regime>_seed700_steps1200`` in single mode (the registered
development replay seed is 700; P22 used 600).

Registered discipline kept from P22: RAM guard ``FTMOE023_RAM_GUARD_GIB``
(default 2.5 GiB) with a bounded wait and every wait recorded, disk guard
20 GiB, 3 torch threads / 1 interop thread, BelowNormal priority,
``stats.history_limit = 64``, ``stats.series_tail = 96``, one process at a
time, refusal to start while another experimental Python process holds this
repo in its command line, ``failure.json`` with the traceback and
``next_action`` on any exception, and never overwriting an existing output
directory.

Usage:
    python prepare_ftmoe_protocol023_stream.py --mode single --regime compute_first
    python prepare_ftmoe_protocol023_stream.py --mode dev
    python prepare_ftmoe_protocol023_stream.py --mode single --regime io_first --smoke
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:                     # ``python -c`` / foreign cwd
    sys.path.insert(0, str(ROOT))

import ftmoe_protocol023_core as core23          # noqa: E402

OUT = ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
SMOKE_ROOT = OUT / "_smoke"
SPLIT_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/vm_split.json"
SCENARIO_PATH = ROOT / "artifacts/ftmoe_online/protocol_020/adapter/scenario_adapter.json"
DRIFT_CONFIG = ROOT / "artifacts/ftmoe_online/protocol_020/drift/drift_config.json"
DISK_LAW_PATH = ROOT / "artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json"

REGISTERED_SEED = 700
REGISTERED_PROBABILITY = 0.25

# CLI-level mirror of the generator's registration.  ``collect`` asserts that
# the imported ``REGIME_IDS`` and the generator's ``mechanism_id`` table are
# exactly equal to these two tables, so the mirror cannot drift silently.
REGISTERED_REGIME_IDS = ("compute_first", "memory_first", "io_first")
REGISTERED_MECHANISM_IDS = {"compute_first": 0, "memory_first": 1,
                            "io_first": 2}

MODE_IDS = ("dev", "single")
# Registered development timeline (plan §10: short dwell times plus recurrence).
DEV_PHASES = (("F0_baseline", 300, None),
              ("A1_compute", 420, "compute_first"),
              ("B1_memory", 420, "memory_first"),
              ("C1_io", 420, "io_first"),
              ("F1_baseline", 240, None),
              ("A2_recur", 360, "compute_first"),
              ("C2_recur", 360, "io_first"),
              ("B2_recur", 360, "memory_first"))
SINGLE_PHASE_LENGTHS = (("F0_baseline", 150, None), ("_only", 1050, "regime"))
REGISTERED_STEPS = {"dev": sum(length for _, length, _ in DEV_PHASES),      # 2880
                    "single": sum(length for _, length, _ in SINGLE_PHASE_LENGTHS)}  # 1200

# Engineering smoke horizon (never a formal stream).  The single-regime phases
# scale by exactly 0.1 to 15 + 105 = 120; the development timeline does not
# (288), so a dev smoke run is refused rather than silently re-registered.
SMOKE_STEPS = 120
SMOKE_PHASE_SCALE = 0.1

COHORT = "online"
FAMILIAR_PHASE = "baseline"
FAMILIAR_TASK_ID = "familiar"          # task map value for a non-cascade task
MECHANISM_SEED = 22022                 # == generator MECHANISM_SEED_DEV (asserted)
FAMILIAR_FLOOR_KEY = {"cpu": "cpu_burst_floor", "ram": "ram_target_floor",
                      "disk": "disk_retained_peak"}

RAM_GUARD_GIB = float(os.environ.get("FTMOE023_RAM_GUARD_GIB", "2.5"))
DISK_GUARD_GIB = 20.0
RAM_GUARD_PREVIOUS_GIB = 3.0
_THREADS_CONFIGURED = False

# Same registered gate as P22 (plan §21 thresholds): the P23 marginal-match gate
# is a separate instrument (``core23.marginal_match_report``) and is not
# evaluated here, so the P22 gate values are reused verbatim rather than
# re-tuned on P23 data.
GATE = {
    "prevalence_min": 0.005,
    "prevalence_max": 0.15,
    "cascade_positive_hoststeps_min": 150,
    "independent_cascade_events_min": 30,
    "normal_hoststeps_min": 5000,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "single_event_share_max": 0.20,
}

AUDIT_ONLY_COLUMNS = {
    "phase_ids": "registered phase index per interval",
    "phase_kind": "0 = familiar phase, 1 = unseen (regime) phase",
    "phase_cascade_probability": "registered cascade_task_probability of the phase",
    "phase_regime_ids": "generator mechanism_id of the phase regime (-1 = familiar)",
    "cascade_task_flags": "1 where the live task carries a registered envelope",
    "cascade_event_ids": "index into workload.cascade_events (-1 = no envelope)",
    "cascade_phases": "per-slot window bit mask (cpu=1, ram=2, disk=4)",
    "cascade_regimes": "per-slot generator mechanism_id 0/1/2 (-1 = no envelope)",
    "host_cascade_any": "1 where the host carries a live masked cascade task",
    "host_cascade_event": "lowest cascade event id live on the host (-1 = none)",
    "host_cascade_regime": "mechanism_id of host_cascade_event (-1 = none)",
    "host_cascade_mask": "per-host OR of the live window bit masks",
}


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json_safe(value):
    """Replace non-finite floats with null so the JSON files stay strict JSON.

    ``core23.response_summary`` reports ``share_positive = NaN`` for an empty
    event cohort; ``json.dumps`` would emit a bare ``NaN`` token that is not
    valid JSON.  The value is reported as ``null`` instead (a missing
    measurement, never a fake zero).
    """
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")
    temporary.replace(path)


def configure():
    global _THREADS_CONFIGURED
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "3"
    for name in ("RAM_SCALE", "DISK_SCALE", "CPU_CAP_SCALE", "DISK_CAP_SCALE",
                 "RAM_CAP_SCALE"):
        os.environ[name] = "1.0"
    for name in ("FIXED_SCHEDULE_PATH", "QOS_OVERLOAD_OUT", "ONLINE_TUNE",
                 "ONLINE_LABEL_MODE"):
        os.environ.pop(name, None)
    import torch
    import psutil
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        _THREADS_CONFIGURED = True
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def guard(deadline_seconds=180.0, poll_seconds=10.0):
    """Registered resource guard (same bounded-wait guard as the P22 collector).

    The threshold is not lowered: a transient dip on this shared workstation
    makes the guard WAIT, bounded, for the registered headroom to return, and
    every wait is recorded in the manifest so the wait cannot hide a real
    shortage.  A shortage that outlives ``deadline_seconds`` still fails hard.
    """
    import psutil
    import shutil
    import time as _time
    started = _time.perf_counter()
    waits = []
    while True:
        available = psutil.virtual_memory().available / 2**30
        if available >= RAM_GUARD_GIB:
            if waits:
                waits[-1]["recovered_after_seconds"] = round(
                    _time.perf_counter() - started, 1)
                guard.waits.extend(waits)
            break
        if _time.perf_counter() - started >= deadline_seconds:
            guard.waits.extend(waits)
            raise RuntimeError(
                "RAM guard below %.1f GiB for %.0f s (available %.3f GiB); "
                "%d wait(s) recorded" % (RAM_GUARD_GIB, deadline_seconds,
                                         available, len(waits)))
        if not waits or waits[-1].get("recovered_after_seconds") is not None:
            waits.append({"at_seconds": round(_time.perf_counter() - started, 1),
                          "available_gib": round(available, 3)})
        _time.sleep(poll_seconds)
    disk = shutil.disk_usage(ROOT).free / 2**30
    if disk < DISK_GUARD_GIB:
        raise RuntimeError("Disk guard below %.1f GiB (available %.3f GiB)"
                           % (DISK_GUARD_GIB, disk))


guard.waits = []


#: Companion scripts of this stage that are allowed to be running while a
#: collector child starts (the S2 runner launches the collector).  A second
#: *collector* is deliberately NOT in this list: it is exactly the process the
#: guard must refuse.
COMPANION_SCRIPTS = ("run_ftmoe_protocol023_s2.py",
                     "verify_ftmoe_protocol023_stream.py")


def assert_no_other_experiment():
    """Refuse to start while another experimental Python process is active.

    Same discipline as ``run_ftmoe_online_stages.py``, tightened the way the
    Protocol-023 runner needs it: a second experimental process would consume
    the resources the guard is protecting and could change simulator inputs.  A
    ``python.exe`` whose command line references this repository is treated as
    an experiment unless it is this process, its parent, or one of this stage's
    companion scripts; every exclusion and every refusal is recorded.
    """
    import psutil
    me = os.getpid()
    try:
        parent = os.getppid()
    except Exception:                              # pragma: no cover
        parent = None
    repo = str(ROOT).lower().replace("\\", "/")
    excluded, others = [], []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid == me:
                continue
            name = (process.info.get("name") or "").lower()
            if "python" not in name:
                continue
            command = list(process.info.get("cmdline") or [])
            line = " ".join(command).lower().replace("\\", "/")
            if repo not in line:
                continue
            scripts = [Path(part).name for part in command
                       if part.lower().endswith(".py")]
            companions = [s for s in scripts if s in COMPANION_SCRIPTS]
            if process.pid == parent or companions:
                excluded.append({"pid": process.pid, "name": name,
                                 "scripts": scripts,
                                 "reason": ("direct parent" if process.pid == parent
                                            else "stage companion script")})
                continue
            others.append({"pid": process.pid, "name": name, "scripts": scripts,
                           "cmdline": command[:6]})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if others:
        raise RuntimeError(
            "Another experimental Python process is active (pid %s); refuse to "
            "start a second one: %s"
            % (others[0]["pid"], json.dumps(others[:3], ensure_ascii=False)))
    return {"checked_repo": str(ROOT), "excluded": excluded,
            "refused": False, "n_other_experiments": len(others)}


# --------------------------------------------------------------------------
# registered stream shapes (the registry the collector AND the verifier read)
# --------------------------------------------------------------------------
def phase_template(mode, regime=None):
    """Registered ``(name, length, regime_id)`` phases of one mode."""
    if mode == "dev":
        return tuple(DEV_PHASES)
    if mode == "single":
        if regime is None:
            raise ValueError("--mode single requires --regime (registered: %s)"
                             % (list(REGISTERED_REGIME_IDS),))
        if regime not in REGISTERED_REGIME_IDS:
            raise ValueError("Unregistered regime %r (registered: %s)"
                             % (regime, list(REGISTERED_REGIME_IDS)))
        return (("F0_baseline", SINGLE_PHASE_LENGTHS[0][1], None),
                ("%s_only" % regime, SINGLE_PHASE_LENGTHS[1][1], regime))
    raise ValueError("Unregistered --mode %r (registered: %s)"
                     % (mode, list(MODE_IDS)))


def registered_steps(mode, regime=None, smoke=False):
    """Registered scored-interval horizon of one stream shape."""
    total = 0
    for _, length, _ in phase_template(mode, regime):
        if smoke:
            scaled = length * SMOKE_PHASE_SCALE
            if abs(scaled - round(scaled)) > 1e-9:
                raise ValueError(
                    "smoke scale %r does not divide the registered phase length "
                    "%d" % (SMOKE_PHASE_SCALE, length))
            length = int(round(scaled))
        total += length
    return total


def registered_phases(mode, regime=None, smoke=False):
    """Registered phase table with bounds, kind, regime and probability."""
    cursor, out = 0, []
    for name, length, phase_regime in phase_template(mode, regime):
        if smoke:
            length = int(round(length * SMOKE_PHASE_SCALE))
        probability = 0.0 if phase_regime is None else REGISTERED_PROBABILITY
        out.append({
            "name": name,
            "start": cursor,
            "end": cursor + length,
            "length": length,
            "regime_id": phase_regime,
            "canonical_regime_id": (None if phase_regime is None
                                    else core23.canonical_regime_id(phase_regime)),
            "mechanism_id": (None if phase_regime is None
                             else REGISTERED_MECHANISM_IDS[phase_regime]),
            "cascade_task_probability": probability,
            "kind": "familiar" if phase_regime is None else "unseen",
        })
        cursor += length
    for index in range(1, len(out)):
        if out[index]["regime_id"] == out[index - 1]["regime_id"]:
            raise AssertionError(
                "registered phases %d and %d share a regime; a phase boundary "
                "must be a regime boundary or the switch point is not "
                "well defined" % (index - 1, index))
    return out


def registered_switch_points(phases):
    """The registered regime switches: one entry per phase boundary."""
    out = []
    for index in range(1, len(phases)):
        phase, previous = phases[index], phases[index - 1]
        if phase["regime_id"] != previous["regime_id"]:
            out.append({"interval": phase["start"], "phase": phase["name"],
                        "regime_id": phase["regime_id"],
                        "probability": phase["cascade_task_probability"]})
    return out


def stream_tag(mode, regime=None, steps=None):
    """Registered output directory name."""
    if steps is None:
        steps = registered_steps(mode, regime)
    if mode == "dev":
        return "dev_seed%d_steps%d" % (REGISTERED_SEED, steps)
    return "single_%s_seed%d_steps%d" % (regime, REGISTERED_SEED, steps)


def phase_at(t, phases):
    """Registered phase of interval ``t`` (the P22 convention at the end).

    ``t == steps`` is the guard row: it is one past the last registered phase,
    so it carries the last phase's annotation, exactly as the P22 collector's
    ``phase_at`` fallback does.  The verifier re-derives the expected phase
    columns through this same registry function.
    """
    for index, phase in enumerate(phases):
        if phase["start"] <= t < phase["end"]:
            return index, phase
    return len(phases) - 1, phases[-1]


def regime_phase_windows(phases):
    """``{regime_id: [(start, end), ...]}`` for the regimes present."""
    out = {}
    for phase in phases:
        if phase["regime_id"] is None:
            continue
        out.setdefault(phase["regime_id"], []).append([phase["start"],
                                                       phase["end"]])
    return out


#: Key of the registered floor of each resource *inside a cascade event dict*.
#: The generator stamps the CPU floor as ``cpu_burst_floor`` (the RAM and disk
#: ones are ``ram_floor`` / ``disk_floor``), which is why its own
#: ``task_cascade_windows`` accessor raises ``KeyError: 'cpu_floor'`` for every
#: compute-first envelope (measured on this checkout, before any collection).
EVENT_FLOOR_KEY = {"cpu": "cpu_burst_floor", "ram": "ram_floor",
                   "disk": "disk_floor"}


def cascade_window_index(workload, floor_keys=None):
    """``creation_id -> registered windows`` (audit only).

    The layout is ``Protocol023MultiRegimeBWGD2.task_cascade_windows``'s, with
    the registered floor key of each onset resource resolved correctly instead
    of the accessor's hard-coded ``"%s_floor"`` (which does not exist for the
    CPU phase).  Nothing is added to or dropped from the accessor's row; the
    workaround is recorded in the manifest so a reader knows why the index is
    built here.
    """
    floor_keys = EVENT_FLOOR_KEY if floor_keys is None else floor_keys
    out = {}
    for event in workload.cascade_events:
        onset = event["onset_resource"]
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
            "onset_floor": float(event[floor_keys[onset]]),
            "onset_peak_value": float(event["%s_peak_value" % onset]),
        }
    return out


def onset_definition(onset_tau, provisional):
    """The task-level onset rule per regime, quoted verbatim into the outputs.

    The P23 generator does not carry the static ``onset_definition()`` accessor
    that the P22 generator has, so the definition is registered here from the
    generator's own ``REGISTERED_ONSET_TAU`` table (passed in) rather than
    invented.
    """
    return {
        "unit": "creation_id + <regime onset resource> onset event",
        "rule": ("the first observed interval at which the task's own onset "
                 "resource reaches the regime's registered threshold, given "
                 "that the previous observed interval was strictly below it; a "
                 "sustained burst counts once"),
        "onset_resource_per_regime": {
            rid: core23.regime(rid)["onset_resource"] for rid in core23.REGIME_IDS},
        "registered_onset_tau": {name: float(value)
                                 for name, value in onset_tau.items()},
        "generator_provisional_onset_tau": {name: float(value)
                                           for name, value in provisional.items()},
        "threshold_basis": ("each threshold is admissible only if it is strictly "
                            "above every measured familiar per-task value of "
                            "that resource and strictly below the regime's own "
                            "registered floor; the familiar maximum is measured "
                            "on the familiar-phase rows of the stream itself "
                            "and reported per regime (never assumed)"),
        "baseline_free": ("True.  Every registered envelope is admitted at the "
                          "familiar level and bursts at the first interior age "
                          "of its window, so a median-baseline jump has no "
                          "pre-onset history to compare against (P22-01)."),
        "response_baseline": ("median of the task's own pre-onset rows when they "
                              "exist, else the registered familiar task level; "
                              "reported per event"),
        "source": "task's own demand only, never the host aggregate",
        "cohort": ("every scan is restricted to the regime's own tasks through "
                   "task_regime_map; the unmapped scan is reported separately as "
                   "the cross-hit confound count and is never a gate input"),
    }


# --------------------------------------------------------------------------
# fault segmentation and the P22-registered gate (window based)
# --------------------------------------------------------------------------
def run_segmentation(labels, steps, mask=None):
    """Per-host contiguous positive-label runs over the scored rows.

    ``mask`` (a boolean array of length ``steps``) restricts the window: an
    interval outside the window closes the run in progress, so a run reported by
    a per-regime or per-phase window is a run *inside that window*.
    """
    runs = []
    for h in range(labels.shape[1]):
        current, length, start = 0, 0, 0
        for t in range(steps):
            if mask is not None and not bool(mask[t]):
                if current > 0 and length:
                    runs.append({"host": h, "class": current,
                                 "start": start, "end": start + length - 1,
                                 "duration": length})
                current, length = 0, 0
                continue
            lab = int(labels[t, h])
            if lab > 0 and lab == current:
                length += 1
            else:
                if current > 0 and length:
                    runs.append({"host": h, "class": current,
                                 "start": start, "end": start + length - 1,
                                 "duration": length})
                current, length, start = lab, (1 if lab > 0 else 0), t
        if current > 0 and length:
            runs.append({"host": h, "class": current, "start": start,
                         "end": start + length - 1, "duration": length})
    return runs


def cascade_fault_runs(runs, host_cascade_event, mask=None):
    """Fault runs whose host carries a registered cascade envelope."""
    out = []
    for run in runs:
        hits = set()
        for t in range(run["start"], run["end"] + 1):
            if mask is not None and not bool(mask[t]):
                continue
            hidden = host_cascade_event[t, run["host"]]
            if hidden >= 0:
                hits.add(int(hidden))
        if hits:
            out.append(dict(run, cascade_events=sorted(hits)))
    return out


def evaluate_gate(labels, host_cascade_any, runs, cascade_runs, deploy_attempts,
                  deploy_rejected, migrate_attempts, migrate_rejected, mask,
                  events):
    """The P22-registered gate evaluated on one interval window.

    ``mask`` is a boolean array of length ``steps``: the window the gate is
    computed on (the whole stream, one phase, or one regime's phases).  Every
    metric is P22's, plus ``window_intervals`` so a per-regime window cannot be
    mistaken for the whole stream.
    """
    steps = int(mask.size)
    hoststeps = int(mask.sum()) * labels.shape[1]
    positives = int((labels[:steps][mask] > 0).sum())
    prevalence = positives / float(hoststeps) if hoststeps else 0.0
    normal = hoststeps - positives
    cascade_positive = int(((labels[:steps][mask] > 0)
                            & (host_cascade_any[:steps][mask] > 0)).sum())
    per_event = {}
    for run in cascade_runs:
        for event_id in run["cascade_events"]:
            per_event[event_id] = per_event.get(event_id, 0) + run["duration"]
    worst_event = max(per_event.items(), key=lambda kv: kv[1]) if per_event else (None, 0)
    single_share = (worst_event[1] / positives) if positives else 0.0
    d_attempts = int(deploy_attempts[:steps][mask].sum())
    d_rejected = int(deploy_rejected[:steps][mask].sum())
    m_attempts = int(migrate_attempts[:steps][mask].sum())
    m_rejected = int(migrate_rejected[:steps][mask].sum())
    checks = {
        "prevalence_in_range": GATE["prevalence_min"] <= prevalence <= GATE["prevalence_max"],
        "cascade_positives_enough": cascade_positive >= GATE["cascade_positive_hoststeps_min"],
        "independent_cascade_events_enough": len(cascade_runs) >= GATE["independent_cascade_events_min"],
        "normal_hoststeps_enough": normal >= GATE["normal_hoststeps_min"],
        "deployment_rejection_ok": (d_rejected / max(d_attempts, 1)) <= GATE["deployment_rejection_max"],
        "migration_rejection_ok": (m_rejected / max(m_attempts, 1)) <= GATE["migration_rejection_max"],
        "single_event_share_ok": single_share < GATE["single_event_share_max"],
    }
    return {"thresholds": GATE, "checks": checks, "passed": all(checks.values()),
            "metrics": {
                "window_intervals": steps,
                "scored_intervals": int(mask.sum()),
                "host_steps": int(hoststeps),
                "positive_hoststeps": positives, "prevalence": prevalence,
                "normal_hoststeps": int(normal),
                "cascade_related_positive_hoststeps": cascade_positive,
                "independent_fault_events": len(runs),
                "independent_cascade_fault_events": len(cascade_runs),
                "deployment_attempts": d_attempts, "deployment_rejected": d_rejected,
                "deployment_rejection_rate": d_rejected / max(d_attempts, 1),
                "migration_attempts": m_attempts, "migration_rejected": m_rejected,
                "migration_rejection_rate": m_rejected / max(m_attempts, 1),
                "cascade_events_registered": len(events),
                "worst_event_share_of_positives": single_share}}


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------
def collect(mode, regime, output, smoke=False, steps=None):
    """Collect one registered stream (or the smoke engineering slice)."""
    registered = registered_steps(mode, regime, smoke=smoke)
    if steps is None:
        steps = registered
    if int(steps) != int(registered):
        raise ValueError("Unregistered horizon: %d (registered for mode=%s "
                         "regime=%s smoke=%s: %d)"
                         % (steps, mode, regime, smoke, registered))
    if smoke and int(steps) != SMOKE_STEPS:
        raise ValueError("Unregistered smoke horizon: %d (registered: %d)"
                         % (steps, SMOKE_STEPS))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    console = sys.stdout
    started = time.perf_counter()
    audit = manifest = None
    process_guard = None
    phases = registered_phases(mode, regime, smoke=smoke)
    switch_points = registered_switch_points(phases)
    try:
        configure()
        guard()
        process_guard = assert_no_other_experiment()
        os.chdir(ROOT)
        import torch
        import psutil
        bitbrain = ROOT / "simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain / ("%d.csv" % i)).is_file() for i in range(1, 500)):
            raise FileNotFoundError("Local Bitbrain dataset incomplete")
        with (output / "generation.log").open("w", encoding="utf8") as log, \
                contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            from simulator.Simulator import Simulator
            from simulator.environment.RPiEdge import RPiEdge
            from simulator.environment.RPiCapacity import RPiCapacity
            from simulator.workload.BitbrainWorkloadProtocol023 import (
                MECHANISM_SEED_DEV, PROVISIONAL_ONSET_TAU, REGIMES_V3,
                REGISTERED_ONSET_TAU, REGIME_IDS,
                Protocol023MultiRegimeBWGD2)
            from scheduler.GOBI import GOBIScheduler
            from recovery.Recovery import Recovery
            from stats.Stats import Stats
            from src.constants import MODEL_SAVE_PATH

            # ---- registration guards (fail before any interval is simulated) --
            if tuple(REGIME_IDS) != tuple(REGISTERED_REGIME_IDS):
                raise AssertionError(
                    "the generator registers %r but this collector's CLI mirror "
                    "registers %r" % (tuple(REGIME_IDS),
                                      tuple(REGISTERED_REGIME_IDS)))
            mechanism_ids = {rid: int(REGIMES_V3[rid]["mechanism_id"])
                             for rid in REGIME_IDS}
            if mechanism_ids != REGISTERED_MECHANISM_IDS:
                raise AssertionError(
                    "the generator's mechanism_id table %r differs from the "
                    "registered %r" % (mechanism_ids, REGISTERED_MECHANISM_IDS))
            if int(MECHANISM_SEED_DEV) != int(MECHANISM_SEED):
                raise AssertionError("generator MECHANISM_SEED_DEV=%r but this "
                                     "collector registers %r"
                                     % (MECHANISM_SEED_DEV, MECHANISM_SEED))
            instrument_registration = core23.assert_registered_regimes()
            # The generator-vs-instrument comparison the P22 collector got for
            # free from ``assert_registered_physics``.  That accessor is NOT
            # called here: on the P23 generator it raises
            # ``TypeError: float() argument must be ... not 'dict'`` for every
            # regime, because ``assert_registered_physics`` numerically compares
            # every registered key and ``response_windows`` is a dict (measured
            # on this checkout, before any collection).  The comparison below is
            # the same claim, made field by field.
            registration_checks, drifted = {}, []
            for rid in REGISTERED_REGIME_IDS:
                spec, instrument = REGIMES_V3[rid], core23.regime(rid)
                onset = spec["onset_resource"]
                same = (
                    str(onset) == str(instrument["onset_resource"])
                    and abs(float(REGISTERED_ONSET_TAU[onset])
                            - float(instrument["onset_tau"])) < 1e-12
                    and [[str(r), int(l)] for r, l in spec["sequence"]]
                    == [[str(r), int(l)] for r, l in instrument["sequence"]]
                    and {str(k): [int(x) for x in v]
                         for k, v in spec["response_windows"].items()}
                    == {str(k): [int(x) for x in v]
                        for k, v in instrument["response_windows"].items()})
                registration_checks[rid] = bool(same)
                if not same:
                    drifted.append(rid)
            if drifted:
                raise AssertionError(
                    "the generator registration and the P23 instrument drifted "
                    "for %s" % (drifted,))

            candidates = [ROOT / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
                          ROOT / "scheduler/BaGTI" / str(MODEL_SAVE_PATH) /
                          "energy_latency_16_Trained.ckpt"]
            scheduler_weight = next((p.resolve() for p in candidates if p.is_file()), None)
            if scheduler_weight is None:
                raise FileNotFoundError("GOBI trained checkpoint missing")

            random.seed(REGISTERED_SEED)
            np.random.seed(REGISTERED_SEED)
            torch.manual_seed(REGISTERED_SEED)

            scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf8"))
            drift = json.loads(DRIFT_CONFIG.read_text(encoding="utf8"))
            familiar = next(p for p in drift["phases"] if p["name"] == FAMILIAR_PHASE)
            adapter = dict(scenario.get("adapter") or {})
            adapter.update(familiar.get("adapter") or {})
            law_rel = scenario.get("disk_law_relative")
            disk_law_path = (ROOT / law_rel).resolve() if law_rel else None

            dc = RPiEdge(16)
            # All three regimes are REGISTERED in every run; phase 0 is familiar,
            # so the mechanism starts OFF and the first segment is the frozen
            # familiar generator exactly (probability 0, the T-AUDIT-05 identity).
            workload = Protocol023MultiRegimeBWGD2(
                float(scenario.get("arrival_mean", 1.0)),
                float(scenario.get("arrival_sigma", 1.5)), REGISTERED_SEED,
                cohort=COHORT, adapter=adapter, disk_law_path=disk_law_path,
                cascade_probability=0.0, mechanism_seed=MECHANISM_SEED_DEV,
                registered_regimes=REGIME_IDS)
            initial_state = workload.set_active_regime(
                phases[0]["regime_id"],
                probability=phases[0]["cascade_task_probability"])
            scheduler = GOBIScheduler("energy_latency_16")
            recovery = Recovery()
            stats = Stats(workload, dc, scheduler)
            stats.feats_per_host = 7
            # Bounded history (P22-18 / P22-20): only the most recent intervals
            # are ever read (the recovery models and the online loader slice
            # ``time_series[-LATEST_WINDOW_SIZE:]``), so bounding the bookkeeping
            # cannot change the simulation (measured: growth per interval fell
            # from ~1.34 MB to ~0.005 MB, which is what makes these streams
            # collectable inside the registered resource guard).
            stats.history_limit = 64
            stats.series_tail = 96
            dev_history = {"history_limit": stats.history_limit,
                           "series_tail": stats.series_tail,
                           "note": "recovery model is a no-op here and reads "
                                   "only the last window; the tail is 8x the "
                                   "12-step window the online path uses"}
            env = Simulator(1000, 10000, scheduler, recovery, stats, 16, 300,
                            dc.generateHosts())
            controller = RPiCapacity(env.hostlist)
            controller.apply(familiar["cpu_scale"], familiar["ram_scale"],
                             familiar["disk_scale"])

            def mask_for(regime_id):
                mask = np.zeros(steps, dtype=bool)
                for phase in phases:
                    if phase["regime_id"] == regime_id:
                        mask[phase["start"]:min(phase["end"], steps)] = True
                return mask

            def window_mask(creation_id, age):
                """(event_id, window bit mask) of one task's registered envelope."""
                event_id = workload.cascade_event_id(creation_id)
                if event_id is None:
                    return -1, 0
                event = workload.cascade_events[event_id]
                mask = 0
                if event["cpu_window"][0] <= age <= event["cpu_window"][1]:
                    mask |= 1
                if event["ram_window"][0] <= age <= event["ram_window"][1]:
                    mask |= 2
                if event["disk_window"][0] <= age <= event["disk_window"][1]:
                    mask |= 4
                return event_id, mask

            initial = workload.generateNewContainers(env.interval)
            deployed = env.addContainersInit(initial)
            decision = scheduler.placement(deployed)
            migrations = env.allocateInit(decision)
            workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
            stats.saveStats(deployed, migrations, [], deployed, decision, 0)

            n_slots = 16
            # ``steps`` is the SCORED interval count and the arrays carry exactly
            # ``steps + 1`` rows, the same convention the P22 streams use (a
            # 1200-interval stream writes 1201 rows): the guard row exists so the
            # last scored row has the successor ``tolerance_labels`` compares
            # row t with row t + 1 against.
            count = steps + 1
            host = np.zeros((count, n_slots, 7), np.float32)
            demands = np.zeros_like(host)
            schedules = np.zeros((count, n_slots, n_slots), np.float32)
            totals = np.zeros((count, n_slots, 3), np.float64)
            caps = np.zeros((count, n_slots, 3), np.float64)
            ratio = np.zeros((count, n_slots, 3), np.float64)
            labels = np.zeros((count, n_slots), np.int64)
            before = np.full((count, n_slots), -1, np.int64)
            after = before.copy()
            creation = before.copy()
            after_creation = before.copy()
            intervals = np.zeros(count, np.int64)
            phase_ids = np.zeros(count, np.int64)
            phase_kind = np.zeros(count, np.int64)
            phase_probability = np.zeros(count, np.float64)
            phase_regime = np.full(count, -1, np.int64)
            cascade_flags = np.zeros((count, n_slots), np.int64)
            cascade_ids = np.full((count, n_slots), -1, np.int64)
            cascade_phases = np.zeros((count, n_slots), np.int64)
            cascade_regimes = np.full((count, n_slots), -1, np.int64)
            slot_demand = np.zeros((count, n_slots, 7), np.float64)
            slot_host = np.full((count, n_slots), -1, np.int64)
            slot_live = np.zeros((count, n_slots), np.int64)
            host_cascade_any = np.zeros((count, n_slots), np.int64)
            host_cascade_event = np.full((count, n_slots), -1, np.int64)
            host_cascade_regime = np.full((count, n_slots), -1, np.int64)
            host_cascade_mask = np.zeros((count, n_slots), np.int64)
            deploy_attempts = np.zeros(count, np.int64)
            deploy_rejected = np.zeros(count, np.int64)
            migrate_attempts = np.zeros(count, np.int64)
            migrate_rejected = np.zeros(count, np.int64)

            active_regime = phases[0]["regime_id"]
            active_probability = phases[0]["cascade_task_probability"]
            observed_switches = []

            for t in range(count):
                if t % 50 == 0:
                    guard()
                index, phase = phase_at(t, phases)
                if phase["regime_id"] != active_regime or \
                        phase["cascade_task_probability"] != active_probability:
                    # The switch is a phase-boundary event: tasks created from
                    # now on see the new regime, tasks already created keep the
                    # envelope they were created under (generator semantics).
                    active_regime = phase["regime_id"]
                    active_probability = phase["cascade_task_probability"]
                    workload.set_active_regime(active_regime,
                                               probability=active_probability)
                    observed_switches.append(
                        {"interval": t, "phase": phase["name"],
                         "regime_id": active_regime,
                         "probability": active_probability})
                phase_ids[t] = index
                phase_kind[t] = 0 if phase["kind"] == "familiar" else 1
                phase_probability[t] = active_probability
                phase_regime[t] = (-1 if phase["regime_id"] is None
                                   else int(REGISTERED_MECHANISM_IDS[phase["regime_id"]]))
                caps[t] = controller.current()
                new = workload.generateNewContainers(env.interval)
                deployed, destroyed = env.addContainers(new)
                intervals[t] = env.interval
                for slot, c in enumerate(env.containerlist):
                    if c is None:
                        continue
                    if c.id != slot or not c.active:
                        raise AssertionError("Invalid live slot identity")
                    ram = c.getRAM()
                    disk = c.getDisk()
                    values = np.array([c.getBaseIPS(), *ram, *disk], dtype=np.float64)
                    demands[t, slot] = values
                    creation[t, slot] = c.creationID
                    before[t, slot] = c.getHostID()
                    if c.getHostID() >= 0:
                        host[t, c.getHostID()] += values
                selected = scheduler.selection()
                decision = scheduler.filter_placement(
                    scheduler.placement(selected + deployed))
                schedules[t] = np.asarray(scheduler.result_cache)
                np.testing.assert_allclose(schedules[t].sum(-1), 1, atol=1e-5)
                for cid, hid in decision:
                    container = env.containerlist[cid] \
                        if 0 <= cid < len(env.containerlist) else None
                    if container is None:
                        continue
                    if container.getHostID() == -1:
                        deploy_attempts[t] += 1
                    else:
                        migrate_attempts[t] += 1
                executed = env.simulationStep(
                    recovery.run_model(stats.time_series, decision))
                executed_set = {(cid, int(hid)) for cid, hid in executed}
                for cid, hid in decision:
                    if (cid, int(hid)) in executed_set:
                        continue
                    container = env.containerlist[cid] \
                        if 0 <= cid < len(env.containerlist) else None
                    if container is None or container.getHostID() == -1:
                        deploy_rejected[t] += 1
                    else:
                        migrate_rejected[t] += 1
                workload.updateDeployedContainers(
                    env.getCreationIDs(executed, deployed))
                for slot, c in enumerate(env.containerlist):
                    if c is None:
                        continue
                    hid = c.getHostID()
                    if c.creationID != creation[t, slot]:
                        raise AssertionError("Slot identity changed inside simulationStep")
                    after[t, slot] = hid
                    after_creation[t, slot] = c.creationID
                    if hid >= 0:
                        totals[t, hid] += [c.getBaseIPS(), c.getRAM()[0],
                                           c.getDisk()[0]]
                    age = int(env.interval - c.startAt)
                    event_id, mask = window_mask(c.creationID, age)
                    cascade_ids[t, slot] = event_id
                    cascade_flags[t, slot] = 1 if event_id >= 0 else 0
                    cascade_phases[t, slot] = mask
                    if event_id >= 0:
                        cascade_regimes[t, slot] = int(
                            workload.cascade_events[event_id]["mechanism_id"])
                    ram2 = c.getRAM()
                    disk2 = c.getDisk()
                    slot_demand[t, slot] = np.array(
                        [c.getBaseIPS(), *ram2, *disk2], dtype=np.float64)
                    slot_host[t, slot] = hid
                    slot_live[t, slot] = 1
                    if hid >= 0 and event_id >= 0 and mask:
                        host_cascade_any[t, hid] = 1
                        if host_cascade_event[t, hid] < 0 or \
                                event_id < host_cascade_event[t, hid]:
                            host_cascade_event[t, hid] = event_id
                            host_cascade_regime[t, hid] = int(
                                workload.cascade_events[event_id]["mechanism_id"])
                        host_cascade_mask[t, hid] |= mask
                ratio[t] = totals[t] / caps[t]
                labels[t] = np.where((ratio[t] > 1).any(-1),
                                     ratio[t].argmax(-1) + 1, 0)
                stats.saveStats(deployed, migrations, destroyed, selected,
                                decision, 0)
                if (t + 1) % 200 == 0:
                    print(json.dumps({"collected": t + 1, "total": count,
                                      "phase": phase["name"],
                                      "regime_id": phase["regime_id"],
                                      "elapsed_seconds": time.perf_counter()
                                      - started}), file=console, flush=True)

            if observed_switches != switch_points:
                raise AssertionError(
                    "the applied regime switches %r are not the registered "
                    "switch points %r" % (observed_switches, switch_points))
            local_steps = steps
            overload_mask = (ratio > 1.0).astype(np.uint8)
            scored_all = np.ones(steps, dtype=bool)
            runs = run_segmentation(labels, steps)
            all_cascade_runs = cascade_fault_runs(runs, host_cascade_event,
                                                  scored_all)
            whole = evaluate_gate(labels, host_cascade_any, runs, all_cascade_runs,
                                  deploy_attempts, deploy_rejected,
                                  migrate_attempts, migrate_rejected, scored_all,
                                  workload.cascade_events)
            audit = dict(whole)
            audit["per_phase"] = {}
            for index, phase in enumerate(phases):
                start, end = phase["start"], min(phase["end"], steps)
                mask = np.zeros(steps, dtype=bool)
                mask[start:end] = True
                phase_runs = run_segmentation(labels, steps, mask=mask)
                phase_cascades = cascade_fault_runs(phase_runs, host_cascade_event,
                                                    mask)
                block = evaluate_gate(labels, host_cascade_any, phase_runs,
                                      phase_cascades, deploy_attempts,
                                      deploy_rejected, migrate_attempts,
                                      migrate_rejected, mask,
                                      workload.cascade_events)
                block["kind"] = phase["kind"]
                block["regime_id"] = phase["regime_id"]
                block["canonical_regime_id"] = phase["canonical_regime_id"]
                block["cascade_task_probability"] = \
                    phase["cascade_task_probability"]
                block["interval_range"] = [start, end]
                audit["per_phase"][phase["name"]] = block

            # ---- task-level timeline (P22 row layout, + per-observation regime) --
            tt, ts = np.nonzero(slot_live[:steps])
            timeline = core23.build_task_timeline(
                tt, ts, creation[tt, ts], slot_demand[tt, ts],
                slot_host[tt, ts], cascade_ids[tt, ts], cascade_phases[tt, ts])
            integrity = core23.timeline_integrity(timeline)
            continuity = core23.migration_continuity(timeline)
            task_regime_map = {}
            for row in range(tt.size):
                cid = int(creation[tt[row], ts[row]])
                event_id = int(cascade_ids[tt[row], ts[row]])
                if event_id < 0:
                    value = FAMILIAR_TASK_ID
                else:
                    value = core23.canonical_regime_id(
                        REGIME_IDS[int(cascade_regimes[tt[row], ts[row]])])
                previous = task_regime_map.setdefault(cid, value)
                if previous != value:
                    raise AssertionError(
                        "creation_id %d maps to two regimes (%r and %r)"
                        % (cid, previous, value))
            coverage = (len(task_regime_map) / float(integrity["n_tasks"])
                        if integrity["n_tasks"] else 1.0)
            if len(task_regime_map) != integrity["n_tasks"]:
                raise AssertionError(
                    "task_regime_map covers %d tasks but the timeline holds %d"
                    % (len(task_regime_map), integrity["n_tasks"]))
            tasks_per_regime = {FAMILIAR_TASK_ID: 0}
            present_regimes = []
            for phase in phases:
                if phase["regime_id"] is not None and \
                        phase["regime_id"] not in present_regimes:
                    present_regimes.append(phase["regime_id"])
            for rid in present_regimes:
                tasks_per_regime[core23.canonical_regime_id(rid)] = 0
            for value in task_regime_map.values():
                tasks_per_regime[value] = tasks_per_regime.get(value, 0) + 1

            # Familiar-phase rows: the measured calm side of the threshold
            # argument.  The floor/ceiling of the argument is measured here, on
            # the same stream, instead of being inherited from P22's corpus.
            familiar_rows = np.nonzero(slot_live[:steps]
                                       & (phase_kind[:steps] == 0)[:, None])
            familiar_timeline = core23.build_task_timeline(
                familiar_rows[0], familiar_rows[1],
                creation[familiar_rows], slot_demand[familiar_rows],
                slot_host[familiar_rows], cascade_ids[familiar_rows],
                cascade_phases[familiar_rows])

            per_regime, cross_confound, admissibility = {}, {}, {}
            all_events = []
            for gen_id in REGISTERED_REGIME_IDS:
                canonical = core23.canonical_regime_id(gen_id)
                spec = core23.regime(canonical)
                resource = spec["onset_resource"]
                tau = float(REGISTERED_ONSET_TAU[resource])
                windows = {str(name): [int(x) for x in window]
                           for name, window in REGIMES_V3[gen_id]["response_windows"].items()}
                mapped = core23.onset_events_multi(
                    timeline, resource, tau,
                    response_windows={name: tuple(window)
                                      for name, window in windows.items()},
                    regime_map={resource: canonical},
                    task_regime_map=task_regime_map)
                blind = core23.onset_events_multi(
                    timeline, resource, tau,
                    response_windows={name: tuple(window)
                                      for name, window in windows.items()},
                    regime_map={resource: canonical})
                # The cross-hit confound and the threshold admissibility are
                # measured for EVERY registered regime, including the ones with
                # no registered phase in this stream: in a single-regime stream
                # the other regimes' unmapped scans are pure confound, and
                # reporting them is what keeps the confound visible.
                cross_confound[canonical] = {
                    "regime_id": gen_id,
                    "onsets_without_task_map": len(blind),
                    "onsets_with_task_map": len(mapped),
                    "cross_hits": len(blind) - len(mapped),
                    "registered_phase_in_stream": bool(gen_id in present_regimes),
                    "note": ("onsets an unmapped scan attributes to this regime "
                             "that its own task cohort does not produce; the "
                             "instrument's tests pin that a regime-A envelope "
                             "crosses the B/C thresholds, so the unmapped number "
                             "is reported and never used as a gate input"),
                }
                admissibility[canonical] = measure_admissibility(
                    familiar_timeline, resource, tau, gen_id,
                    float(REGIMES_V3[gen_id][FAMILIAR_FLOOR_KEY[resource]]))
                if gen_id not in present_regimes:
                    # No registered phase ⇒ no gate and no task cohort for this
                    # regime in this stream; an empty block would look like a
                    # measured regime, so it is not written.
                    continue
                responses = core23.event_responses_multi(
                    timeline, mapped,
                    response_windows={name: tuple(window)
                                      for name, window in windows.items()},
                    onset_resource=resource)
                durations = core23.response_durations(
                    timeline, mapped,
                    response_windows={name: tuple(window)
                                      for name, window in windows.items()},
                    onset_resource=resource)
                # ``peak_ratio_summary`` reads ``<resource>_baseline`` off its
                # rows, so it must be handed the ``onset_events_multi`` rows
                # (which carry the baselines); the ``event_responses_multi`` rows
                # carry only the response statistics.  Measured mismatch: passing
                # the response rows raises
                # KeyError "event ... carries no disk_baseline".
                peaks = core23.peak_ratio_summary(
                    mapped,
                    response_windows={name: tuple(window)
                                      for name, window in windows.items()},
                    onset_resource=resource)
                response_summaries, alignment = {}, {}
                for name in sorted(windows):
                    response_summaries[name] = core23.response_summary(responses,
                                                                       name)
                    alignment[name] = core23.alignment_rate_multi(
                        responses, name, windows[name][0])
                regime_mask = mask_for(gen_id)
                regime_runs = run_segmentation(labels, steps, mask=regime_mask)
                regime_cascades = cascade_fault_runs(regime_runs,
                                                     host_cascade_event,
                                                     regime_mask)
                gate = evaluate_gate(labels, host_cascade_any, regime_runs,
                                     regime_cascades, deploy_attempts,
                                     deploy_rejected, migrate_attempts,
                                     migrate_rejected, regime_mask,
                                     workload.cascade_events)
                task_rows = [task_regime_map[c] == canonical for c in timeline]
                per_regime[canonical] = {
                    "regime_id": gen_id,
                    "canonical_regime_id": canonical,
                    "mechanism_id": int(REGISTERED_MECHANISM_IDS[gen_id]),
                    "onset_resource": resource,
                    "onset_tau": tau,
                    "onset_floor": float(REGIMES_V3[gen_id][FAMILIAR_FLOOR_KEY[resource]]),
                    "onset_provisional": bool(resource in core23.PROVISIONAL_TAUS),
                    "response_windows": windows,
                    "sequence": [[str(r), int(l)] for r, l in REGIMES_V3[gen_id]["sequence"]],
                    "phase_windows": regime_phase_windows(phases).get(gen_id, []),
                    "gate": gate,
                    "task_level": {
                        "n_tasks_in_cohort": int(sum(task_rows)),
                        "n_onsets": len(mapped),
                        "n_onsets_with_cascade_window": int(sum(
                            1 for e in mapped if e["audit_event_id"] >= 0)),
                        "n_onsets_without_task_map": len(blind),
                        "cross_hits": len(blind) - len(mapped),
                        "onset_threshold": tau,
                        "onset_provisional": bool(resource in core23.PROVISIONAL_TAUS),
                        "response_summary": response_summaries,
                        "response_durations": durations,
                        "peak_ratio": peaks,
                        "alignment": alignment,
                    },
                }
                all_events.extend(mapped)
            all_events.sort(key=lambda e: (e["creation_id"], e["t_onset"]))

            audit["per_regime"] = per_regime
            audit["cross_regime_confound"] = cross_confound
            audit["admissibility"] = admissibility
            audit["task_level"] = {
                "n_tasks": integrity["n_tasks"],
                "n_observations": integrity["n_observations"],
                "reused_slots": integrity["reused_slots"],
                "multi_slot_tasks": integrity["multi_slot_tasks"],
                "tasks_with_migration": continuity["n_tasks_with_migration"],
                "integrity_ok": integrity["ok"],
                "integrity_problems": integrity["problems"],
                "n_onsets": len(all_events),
                "n_onsets_with_cascade_window": int(sum(
                    1 for e in all_events if e["audit_event_id"] >= 0)),
                "per_regime_onsets": {rid: block["task_level"]["n_onsets"]
                                      for rid, block in per_regime.items()},
                "task_regime_map_summary": {
                    "id_space": ("canonical core23 regime ids (A/B/C) for tasks "
                                 "created under a registered regime, %r for a "
                                 "task with no envelope" % FAMILIAR_TASK_ID),
                    "n_tasks_mapped": len(task_regime_map),
                    "n_timeline_tasks": integrity["n_tasks"],
                    "coverage": coverage,
                    "tasks_per_regime": tasks_per_regime,
                    "mapping_sha256": _mapping_sha256(task_regime_map),
                },
                "task_regime_map": {str(k): v for k, v in
                                    sorted(task_regime_map.items())},
                "familiar_rows": int(familiar_rows[0].size),
                "familiar_tasks": len(familiar_timeline),
            }
            audit["definitions"] = {
                "familiar_phase": ("set_active_regime(None, probability=0.00): "
                                   "the registered mechanism is off, i.e. the "
                                   "frozen familiar generator exactly "
                                   "(T-AUDIT-05 identity)"),
                "unseen_phase": ("set_active_regime(<regime>, probability=%.2f)"
                                 % REGISTERED_PROBABILITY),
                "phase_and_regime_columns_are_audit_only": (
                    "phase_ids / phase_kind / phase_cascade_probability / "
                    "phase_regime_ids / cascade_* / host_cascade_* are audit "
                    "metadata and must never be a model input (FORBIDDEN_INPUTS)"),
                "task_envelope_frozen_at_creation": (
                    "a task created inside a phase keeps the envelope of the "
                    "phase it was created under; the collector never re-adapts "
                    "an existing container at a switch"),
                "onset_scan_cohort": (
                    "each regime is scanned with task_regime_map restricted to "
                    "its own tasks; the unmapped scan is reported next to it as "
                    "the cross-hit confound count"),
            }
            audit["protocol"] = "023"
            audit["mode"] = mode
            audit["regime"] = regime
            audit["smoke"] = bool(smoke)
            audit["chunk"] = None
            audit["phases"] = phases
            audit["phase_switch_points"] = switch_points
            audit["candidate"] = {"cascade_task_probability": REGISTERED_PROBABILITY,
                                  "cohort": COHORT, "replay_seed": REGISTERED_SEED,
                                  "mechanism_seed": int(MECHANISM_SEED_DEV),
                                  "scored_intervals": steps,
                                  "mode": mode, "regime": regime}
            write_json(output / "unseen_data_audit.json", audit)

            np.savez_compressed(
                output / "stream.npz",
                host_features=host, demands=demands, schedules=schedules,
                raw_labels=labels, capacities=caps, post_totals=totals,
                overload_ratio=ratio, overload_mask=overload_mask,
                before_placement=before, after_placement=after,
                creation_ids=creation, after_creation_ids=after_creation,
                simulator_intervals=intervals,
                phase_ids=phase_ids, phase_kind=phase_kind,
                phase_cascade_probability=phase_probability,
                phase_regime_ids=phase_regime,
                cascade_task_flags=cascade_flags, cascade_event_ids=cascade_ids,
                cascade_phases=cascade_phases, cascade_regimes=cascade_regimes,
                host_cascade_any=host_cascade_any,
                host_cascade_event=host_cascade_event,
                host_cascade_regime=host_cascade_regime,
                host_cascade_mask=host_cascade_mask,
                deploy_attempts=deploy_attempts, deploy_rejected=deploy_rejected,
                migrate_attempts=migrate_attempts, migrate_rejected=migrate_rejected)

            np.savez_compressed(
                output / "task_timeline.npz",
                time=tt, slot_index=ts, creation_id=creation[tt, ts],
                demand=slot_demand[tt, ts], host_id=slot_host[tt, ts],
                cascade_event_id=cascade_ids[tt, ts],
                cascade_phase=cascade_phases[tt, ts],
                cascade_regime=cascade_regimes[tt, ts])
            write_json(output / "task_event_index.json", {
                "protocol": "023",
                "unit": "creation_id + regime onset event",
                "onset_definition": onset_definition(REGISTERED_ONSET_TAU,
                                                     PROVISIONAL_ONSET_TAU),
                "temporal_events": [
                    {"event_key": e["event_key"], "creation_id": e["creation_id"],
                     "regime_id": e["regime_id"],
                     "onset_resource": e["onset_resource"],
                     "onset_tau": e["onset_tau"],
                     "onset_provisional": e["onset_provisional"],
                     "t_onset": e["t_onset"], "age_onset": e["age_onset"],
                     "host_onset": e["host_onset"],
                     "delta_at_onset": e["delta_at_onset"],
                     "value_at_onset": e["value_at_onset"],
                     "baseline_source": e["baseline_source"],
                     "audit_event_id": e["audit_event_id"], "task_level": True}
                    for e in all_events],
                "cascade_windows": cascade_window_index(workload),
                "timeline_integrity": integrity,
                "migration_continuity": {"n_tasks": continuity["n_tasks"],
                                         "n_tasks_with_migration":
                                             continuity["n_tasks_with_migration"]},
                "task_regime_map_summary": audit["task_level"]["task_regime_map_summary"],
                "per_regime_onsets": {rid: block["task_level"]["n_onsets"]
                                      for rid, block in per_regime.items()},
                "cross_regime_confound": cross_confound,
            })
            write_json(output / "events.json", {
                "protocol": "023",
                "cascade_envelopes": workload.cascade_events,
                "fault_run_segmentation": runs,
                "cascade_fault_runs": all_cascade_runs,
                "phase_switch_points": switch_points,
                "observed_phase_switches": observed_switches,
                "phases": phases,
                "per_phase_gate": {k: v["checks"] for k, v in
                                   audit["per_phase"].items()},
                "per_regime_gate": {k: v["gate"]["checks"] for k, v in
                                    per_regime.items()},
                "cross_regime_confound": cross_confound,
                "admissibility": admissibility,
            })

            sources = [ROOT / "prepare_ftmoe_protocol023_stream.py",
                       ROOT / "ftmoe_protocol023_core.py",
                       ROOT / "ftmoe_protocol022_core.py",
                       ROOT / "simulator/workload/BitbrainWorkloadProtocol023.py",
                       DISK_LAW_PATH,
                       SPLIT_PATH, SCENARIO_PATH, DRIFT_CONFIG, scheduler_weight]
            if law_rel:
                sources.append(ROOT / law_rel)
            for base in ("simulator", "scheduler", "metrics", "stats", "utils"):
                sources.extend(p for p in (ROOT / base).rglob("*.py")
                               if "__pycache__" not in p.parts)
            sources.extend((ROOT / "scheduler/BaGTI").rglob("*.npy"))
            source_hashes = {str(p.relative_to(ROOT)): sha(p)
                             for p in sorted(set(sources))}
            manifest = {
                "schema_version": 2, "protocol": "023", "phase": "P23-S2",
                "kind": ("multi_regime_development_stream" if mode == "dev"
                         else "single_regime_stream"),
                "registered": not smoke,
                "name": ("cascade_v3 development stream (three regimes, "
                         "familiar/regime alternation with recurrence)"
                         if mode == "dev" else
                         "cascade_v3 single-regime stream (%s)" % regime),
                "mode": mode, "regime": regime, "smoke": bool(smoke),
                "regime_id": "cascade_v3", "regime_family": "cascade_v3",
                "registered_regimes": list(REGIME_IDS),
                "registered_mechanism_ids": {str(k): int(v) for k, v in
                                             REGISTERED_MECHANISM_IDS.items()},
                "canonical_regime_ids": {rid: core23.canonical_regime_id(rid)
                                         for rid in REGIME_IDS},
                "mechanism": workload.cascade_audit(),
                "cascade_task_probability": REGISTERED_PROBABILITY,
                "seed": REGISTERED_SEED, "steps": steps, "guard_steps": 1,
                "scored_intervals": int(local_steps),
                "step_semantics": ("'steps' is the number of SCORED intervals "
                                   "(the model-facing horizon); the stream arrays "
                                   "carry 'steps' + 1 rows, so the last scored "
                                   "row has the successor the +/-1 label "
                                   "tolerance compares against"),
                "phases": phases,
                "phase_switch_points": switch_points,
                "observed_phase_switches": observed_switches,
                "initial_phase_state": {"interval": 0, "phase": phases[0]["name"],
                                        "regime_id": phases[0]["regime_id"],
                                        "result": initial_state},
                "phase_amendment": None,
                "chunk": None,
                "chunk_note": (
                    "the P22 collector could only collect its registered stream "
                    "in overlapping chunks because its bookkeeping grew ~0.89 MB "
                    "per interval; the bounded history above cuts that to "
                    "~0.005 MB per interval, so both registered P23 horizons "
                    "(1200 and 2880 scored intervals) are collected in a single "
                    "process and no concatenation is registered here"),
                "audit_only_columns": AUDIT_ONLY_COLUMNS,
                "forbidden_model_inputs": list(core23.FORBIDDEN_INPUTS),
                "audit_only_note": (
                    "phase_ids / phase_kind / phase_cascade_probability / "
                    "phase_regime_ids / cascade_regimes / cascade_task_flags / "
                    "cascade_event_ids / cascade_phases / host_cascade_any / "
                    "host_cascade_event / host_cascade_regime / "
                    "host_cascade_mask are AUDIT METADATA.  They label an event "
                    "for reporting and must never be a model input (plan §4, "
                    "§19; forbidden tokens are listed in "
                    "forbidden_model_inputs)"),
                "cohort": COHORT, "cohort_vm_ids": list(workload.possible_indices),
                "familiar_phase": {"name": familiar["name"],
                                   "cpu_scale": familiar["cpu_scale"],
                                   "ram_scale": familiar["ram_scale"],
                                   "disk_scale": familiar["disk_scale"],
                                   "adapter": adapter},
                "registration": {
                    "instrument_assert_registered_regimes":
                        instrument_registration,
                    "generator_vs_instrument_checks": registration_checks,
                    "passed": all(registration_checks.values()),
                    "generator_assert_registered_physics": (
                        "NOT called: on this generator it raises TypeError "
                        "('float() argument must be a string or a number, not "
                        "dict') for every regime because it compares every "
                        "registered key numerically and response_windows is a "
                        "dict.  The same claim is checked field by field in "
                        "generator_vs_instrument_checks instead."),
                    "generator_task_cascade_windows": (
                        "NOT called: it reads event['%s_floor' % onset_resource] "
                        "but the generator stamps the CPU floor as "
                        "'cpu_burst_floor', so it raises KeyError 'cpu_floor' for "
                        "every compute-first envelope (measured here).  The same "
                        "row is built by cascade_window_index() with the "
                        "registered per-resource floor key; no field is added or "
                        "dropped."),
                    "verification": (
                        "both workarounds are documented deviances from "
                        "generator accessors, not from the registration: the "
                        "generator module itself was not modified"),
                },
                "onset_definition": onset_definition(REGISTERED_ONSET_TAU,
                                                     PROVISIONAL_ONSET_TAU),
                "registered_onset_tau": {str(k): float(v) for k, v in
                                         REGISTERED_ONSET_TAU.items()},
                "capacity_control_version": 1,
                "scenario_adapter_sha256": sha(SCENARIO_PATH),
                "drift_config_sha256": sha(DRIFT_CONFIG),
                "split_sha256": sha(SPLIT_PATH),
                "interval_seconds": 300, "hosts": 16, "slots": n_slots,
                "arrival_mean": float(scenario.get("arrival_mean", 1.0)),
                "arrival_sigma": float(scenario.get("arrival_sigma", 1.5)),
                "recovery": "no_op", "scheduler": "GOBI_energy_latency_16",
                "raw_class_counts_scored": np.bincount(
                    labels[:steps].ravel(), minlength=4).tolist(),
                "events_summary": whole["metrics"],
                "per_phase_summary": {k: v["metrics"] for k, v in
                                      audit["per_phase"].items()},
                "per_phase_gate": {k: v["checks"] for k, v in
                                   audit["per_phase"].items()},
                "per_regime_summary": {k: {"regime_id": v["regime_id"],
                                           "onset_resource": v["onset_resource"],
                                           "onset_tau": v["onset_tau"],
                                           "phase_windows": v["phase_windows"],
                                           "gate": v["gate"]["metrics"],
                                           "task_level": v["task_level"]}
                                       for k, v in per_regime.items()},
                "per_regime_gate": {k: {"checks": v["gate"]["checks"],
                                        "passed": v["gate"]["passed"]}
                                    for k, v in per_regime.items()},
                "cross_regime_confound": cross_confound,
                "admissibility": admissibility,
                "task_level_summary": audit["task_level"],
                "bounded_history": dev_history,
                "process_guard": process_guard,
                "source_sha256": source_hashes,
                "ram_guard": {"threshold_gib": RAM_GUARD_GIB,
                              "previous_registered_gib": RAM_GUARD_PREVIOUS_GIB,
                              "amendment": ("plan 21 registered 3.0 GiB; lowered "
                                            "to 2.5 GiB with explicit user "
                                            "authorisation after the measured "
                                            "growth curve made 3.0 GiB "
                                            "unreachable on this workstation "
                                            "(problem P22-17); the P23 collector "
                                            "keeps the amended value and the "
                                            "same bounded-wait guard"),
                              "environment_variable": "FTMOE023_RAM_GUARD_GIB",
                              "waits": list(guard.waits),
                              "note": "a transient dip makes the guard wait, "
                                      "and every wait is recorded here"},
                "stream_sha256": sha(output / "stream.npz"),
                "task_timeline_sha256": sha(output / "task_timeline.npz"),
                "task_timeline_content_sha256": core23.timeline_sha256(timeline),
                "elapsed_seconds": time.perf_counter() - started,
                "rss_gib": psutil.Process().memory_info().rss / 2**30,
            }
            write_json(output / "manifest.json", manifest)
        print(json.dumps({"completed": str(output), "mode": mode,
                          "regime": regime, "steps": steps,
                          "gate_passed": audit["passed"],
                          "per_phase_gate": {k: v["passed"]
                                             for k, v in audit["per_phase"].items()},
                          "per_regime_gate": {k: v["gate"]["passed"]
                                              for k, v in per_regime.items()},
                          "task_level": {"n_tasks": audit["task_level"]["n_tasks"],
                                         "n_onsets": audit["task_level"]["n_onsets"],
                                         "coverage": audit["task_level"]
                                         ["task_regime_map_summary"]["coverage"]},
                          "elapsed_seconds": manifest["elapsed_seconds"]},
                         ensure_ascii=False), file=console, flush=True)
    except Exception as exc:
        failure = {"mode": mode, "regime": regime, "steps": steps,
                   "cohort": COHORT, "smoke": bool(smoke),
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc(),
                   "guard_waits": list(guard.waits),
                   "process_guard": process_guard,
                   "next_action": ("Report before changing the mechanism or the "
                                   "registered phase structure; the output "
                                   "directory is left in place so the failure "
                                   "can be inspected, and it must not be "
                                   "reused for a new attempt")}
        write_json(output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False), file=console, flush=True)
        raise
    return {"audit": audit, "manifest": manifest}


def _mapping_sha256(mapping):
    import hashlib
    payload = json.dumps({str(k): str(v) for k, v in sorted(mapping.items())},
                         ensure_ascii=False, sort_keys=True).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def measure_admissibility(familiar_timeline, resource, tau, regime_id, floor):
    """The measured (not assumed) half of the registered threshold argument.

    A threshold is admissible only if it is **strictly above** every familiar
    per-task value of that resource measured on the familiar-phase rows of this
    same stream.  The measurement is reported even when it passes, and a
    failure is reported as ``threshold_admissible: false`` -- never "fixed" by
    nudging the threshold after seeing the data.
    """
    per_task = []
    for creation_id in sorted(familiar_timeline):
        values = np.asarray(familiar_timeline[creation_id][resource],
                            dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            per_task.append(float(values.max()))
    measured = max(per_task) if per_task else None
    return {
        "regime_id": regime_id,
        "onset_resource": resource,
        "measured_familiar_task_maximum": measured,
        "n_familiar_tasks": len(per_task),
        "registered_familiar_clip": float(core23.FAMILIAR_CLIP[resource]),
        "registered_onset_threshold": float(tau),
        "registered_floor": (None if floor is None else float(floor)),
        "threshold_admissible": bool(measured is not None and tau > measured),
        "margin_above_measured_maximum": (None if measured is None
                                          else float(tau - measured)),
        "margin_below_floor": (None if floor is None
                               else float(floor - tau)),
        "provisional": bool(resource in core23.PROVISIONAL_TAUS),
        "measurement_source": ("per-task maximum of the onset resource column "
                               "over the familiar-phase rows of this stream"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(MODE_IDS), default="dev",
                        help="dev = the registered multi-regime development "
                             "timeline (2880 scored intervals); single = one "
                             "regime only (1200 scored intervals)")
    parser.add_argument("--regime", default=None,
                        choices=list(REGISTERED_REGIME_IDS),
                        help="required with --mode single")
    parser.add_argument("--steps", type=int, default=None,
                        help="must equal the registered phase-length sum")
    parser.add_argument("--smoke", action="store_true",
                        help="engineering slice only (never a formal stream)")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.mode == "dev" and args.regime is not None:
        raise SystemExit("--regime is registered only for --mode single")
    if args.mode == "single" and args.regime is None:
        raise SystemExit("--mode single requires --regime (registered: %s)"
                         % (list(REGISTERED_REGIME_IDS),))
    try:
        steps = registered_steps(args.mode, args.regime, smoke=args.smoke)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.smoke and steps != SMOKE_STEPS:
        raise SystemExit(
            "the registered smoke horizon is %d scored intervals, but the "
            "%s timeline scales to %d; a smoke run is registered for --mode "
            "single only" % (SMOKE_STEPS, args.mode, steps))
    if args.steps is not None and int(args.steps) != int(steps):
        raise SystemExit("Unregistered horizon: %d (registered for mode=%s "
                         "regime=%s smoke=%s: %d)"
                         % (args.steps, args.mode, args.regime, args.smoke, steps))
    tag = stream_tag(args.mode, args.regime, steps)
    root = args.output_root / "_smoke" if args.smoke else args.output_root
    collect(mode=args.mode, regime=args.regime, output=root / tag,
            smoke=args.smoke, steps=steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
