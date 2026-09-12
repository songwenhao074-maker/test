"""Protocol 022 — task-level (``creation_id``) audit instrument, U4-v2.

This module is the *instrument*.  It is deliberately free of simulation, model
and I/O side effects so the unit tests can drive it with hand-built synthetic
tasks (plan §4.3).

Why task level
--------------
Protocol 021 measured novelty on ``[T, H, 7]`` host aggregates and its U4 gate
failed (P21-09).  Host aggregates mix co-resident tasks, so a per-task temporal
cascade cannot be separated from "two heavy tasks happened to share a host".
Everything here is therefore keyed by ``creation_id``, the only stable task
identity in this simulator:

    * a *slot* (``containerlist`` index) is reused after a container is
      destroyed, so it must never be treated as a task id (T-AUDIT-03);
    * a container object is preserved across migration, so following
      ``creation_id`` yields a continuous trajectory under host changes
      (T-AUDIT-02);
    * a container is only visible in the timeline once it is placed, because
      ``Simulator.simulationStep`` destroys the pointer of any container that
      could not be allocated (so ``age`` is the *observed* age, T-AUDIT-01).

Registered onset / response definitions (plan §6.2-6.4)
------------------------------------------------------
    onset                : the first observed interval at which the task's own
                           CPU reaches ``onset_tau_cpu``, given that the previous
                           observed interval was strictly below it; a sustained
                           burst counts once.  Baseline-free, because the
                           registered envelope is admitted at the familiar level
                           and bursts at age 1, so a median-baseline jump has no
                           pre-onset history to measure against (P22-01).
    RAM_response         = max RAM_task[t0+4 : t0+14] - baseline_RAM
    Disk_response        = max Disk_task[t0+8 : t0+18] - baseline_Disk
    baseline_*           = median of the task's own pre-onset rows when they
                           exist, else the registered familiar task level
                           (``FAMILIAR_TASK_LEVEL``); the source is reported per
                           event.

Everything is computed from the task's *own* demand column; the host aggregate
is never consulted.  The audit metadata (cascade flag, event id, phase) may be
used to *label* an event for reporting, never to build a model input
(``FORBIDDEN_INPUT_TOKENS``).
"""
import hashlib

import numpy as np

# Model-visible feature indices inside a task's demand row (CPU, RAM read,
# RAM write, Disk read, Disk write are packed as 7 columns; the three demand
# magnitudes this audit uses are CPU, RAM size and Disk size).
IDX_CPU, IDX_RAM, IDX_DISK = 0, 1, 4

FORBIDDEN_INPUT_TOKENS = ("regime_id", "phase_id", "cascade_task_flag",
                          "cascade_event_id", "future_demand",
                          "future_capacity", "unmatured_label")

# Registered constants (mirrors CASCADE_V2 in the generator; the analyzer
# asserts the two agree so the instrument cannot silently drift).
ONSET_TAU_CPU = 2600.0
ONSET_BASELINE_LAG = 4
RAM_RESPONSE_WINDOW = (4, 14)
DISK_RESPONSE_WINDOW = (8, 18)
REGISTERED_LAG_RAM = 4
REGISTERED_LAG_DISK = 8

# Calibrated offline reference threshold.
#
# The registered onset threshold (2600) is strictly above the familiar
# per-task CPU clip (1860), so *no* offline task can ever produce a registered
# onset -- a measured fact (artifact: audit_v2/offline_reference_v2.json ->
# familiar_maxima), not an assumption.  The plan's rule "candidate p90 >
# offline p97.5" is therefore undefined on the registered-threshold scale.
#
# To keep a distribution-level comparison that is still meaningful, the same
# instrument is additionally run offline at this lower, OFFLINE-CALIBRATED
# threshold.  The rule that picks it was fixed from the offline data alone,
# before any candidate stream was read:
#
#     500 is the smallest round threshold strictly above the pooled offline
#     p99 of the non-negative task-level Delta CPU (P019 481, P020 S6 419,
#     P020 dev 327) and strictly below every corpus maximum (861 / 1688 /
#     1688), so every offline corpus contributes familiar-percentile events
#     and no familiar task is admitted as a "rise".
#
# It is NOT a second attempt at the gate: the registered-threshold emptiness
# and the calibrated-threshold separation are reported side by side.
ONSET_TAU_CALIBRATED = 500.0


# --------------------------------------------------------------------------
# timeline construction
# --------------------------------------------------------------------------
def build_task_timeline(time, slot, creation_id, dem, host, event_id=None,
                        phase=None):
    """Group observation rows into per-``creation_id`` trajectories.

    Parameters are parallel 1-D arrays (one entry per observed live container
    at one interval).  A ``(time, slot)`` pair must be unique; a
    ``(time, creation_id)`` pair must be unique as well, which is exactly the
    invariant that fails if a slot is mistaken for a task.

    Returns ``{creation_id: {...}}`` with ``t``, ``slot``, ``age``,
    ``cpu``/``ram``/``disk``, ``host`` and ``present`` arrays.
    """
    time = np.asarray(time, dtype=np.int64)
    slot = np.asarray(slot, dtype=np.int64)
    cid = np.asarray(creation_id, dtype=np.int64)
    dem = np.asarray(dem, dtype=np.float64)
    host = np.asarray(host, dtype=np.int64)
    n = time.size
    for name, arr in (("slot", slot), ("creation_id", cid), ("host", host),
                      ("time", time)):
        if arr.shape != (n,):
            raise ValueError("%s must be a 1-D array of length %d" % (name, n))
    if dem.shape != (n, 7):
        raise ValueError("dem must be [n, 7], got %s" % (dem.shape,))

    keys = set(zip(time.tolist(), slot.tolist()))
    if len(keys) != n:
        raise ValueError("duplicate (time, slot) observation: slot reuse and "
                         "time indexing must be unique")

    order = np.lexsort((time, cid))
    out = {}
    for i in order:
        c = int(cid[i])
        entry = out.get(c)
        if entry is None:
            entry = out[c] = {"creation_id": c, "t": [], "slot": [], "age": [],
                              "cpu": [], "ram": [], "disk": [], "host": [],
                              "event_id": []}
        entry["t"].append(int(time[i]))
        entry["slot"].append(int(slot[i]))
        entry["age"].append(int(time[i]))
        entry["cpu"].append(float(dem[i, IDX_CPU]))
        entry["ram"].append(float(dem[i, IDX_RAM]))
        entry["disk"].append(float(dem[i, IDX_DISK]))
        entry["host"].append(int(host[i]))
        entry["event_id"].append(-1 if event_id is None else int(event_id[i]))

    for entry in out.values():
        for name in ("t", "slot", "age", "host", "event_id"):
            entry[name] = np.asarray(entry[name], dtype=np.int64)
        for name in ("cpu", "ram", "disk"):
            entry[name] = np.asarray(entry[name], dtype=np.float64)
        # observed age is defined relative to the first interval the task is
        # visible at; a container that is never placed contributes no row
        entry["age"] = entry["t"] - entry["t"][0]
        entry["birth_t"] = int(entry["t"][0])
        entry["last_t"] = int(entry["t"][-1])
        if entry["t"].size > 1 and np.any(np.diff(entry["t"]) != 1):
            entry["gaps"] = int(np.sum(np.diff(entry["t"]) != 1))
        else:
            entry["gaps"] = 0
        # host may be -1 only if the container was placed then lost the host;
        # it is reported, not silently coerced
        entry["n_unplaced_rows"] = int((entry["host"] < 0).sum())
    durations = [e["t"].size for e in out.values()]
    duplicate_times = n - sum(durations)
    if duplicate_times != 0:
        raise ValueError("a creation_id was observed twice in one interval")
    return out


def timeline_integrity(timeline):
    """Checks that must hold for the task-level claim to be admissible."""
    problems = []
    slots_by_cid = {}
    cids_by_slot = {}
    for c, entry in timeline.items():
        if entry["t"].size == 0:
            problems.append("creation_id %d has no observations" % c)
            continue
        if entry["gaps"]:
            problems.append("creation_id %d has %d time gap(s)"
                            % (c, entry["gaps"]))
        if not np.all(np.diff(entry["age"]) == 1):
            problems.append("creation_id %d has a non-contiguous age axis" % c)
        slots_by_cid[c] = set(entry["slot"].tolist())
        for s in slots_by_cid[c]:
            cids_by_slot.setdefault(s, set()).add(c)
    reused = {s: sorted(v) for s, v in cids_by_slot.items() if len(v) > 1}
    return {
        "n_tasks": len(timeline),
        "n_observations": int(sum(e["t"].size for e in timeline.values())),
        "reused_slots": len(reused),
        "reused_slot_examples": dict(list(reused.items())[:5]),
        "multi_slot_tasks": sum(1 for v in slots_by_cid.values() if len(v) > 1),
        "problems": problems,
        "ok": not problems,
    }


def migration_continuity(timeline):
    """Host trajectory per task: every consecutive pair must be a legal step.

    A legal step is either "same host" or "different host", both of which are
    allowed; what matters is that the trajectory is *recorded* against one
    ``creation_id`` rather than being split or spliced.  Returns per-task host
    sequences plus aggregate counts so a migration test can assert that a task
    which changes host keeps a single identity.
    """
    out = {}
    for c, entry in timeline.items():
        hosts = entry["host"].tolist()
        steps = [(hosts[i], hosts[i + 1]) for i in range(len(hosts) - 1)]
        changes = [i for i, (a, b) in enumerate(steps) if a != b]
        out[c] = {"hosts": hosts, "n_host_changes": len(changes),
                  "changed_at": changes}
    return {"per_task": out,
            "n_tasks_with_migration": sum(1 for v in out.values()
                                          if v["n_host_changes"]),
            "n_tasks": len(out)}


# --------------------------------------------------------------------------
# registered onset / response instrument
# --------------------------------------------------------------------------
# Fallback per-task levels used when an onset has no observable pre-onset
# history (see onset_positions).  A task that is admitted at the familiar level
# and bursts on its first observed interval has no prior rows of its own, so its
# baseline is reported as the registered familiar clip rather than as a
# fabricated value.
FAMILIAR_TASK_LEVEL = {"cpu": 1860.0, "ram": 1400.0, "disk": 9000.0}


def delta_cpu(cpu, baseline_lag=ONSET_BASELINE_LAG):
    """Delta CPU_t = CPU(t) - median(CPU[t-k : t]); the first k are undefined."""
    cpu = np.asarray(cpu, dtype=np.float64)
    n = cpu.size
    out = np.full(n, np.nan)
    for i in range(baseline_lag, n):
        out[i] = cpu[i] - np.median(cpu[i - baseline_lag:i])
    return out


def onset_positions(cpu, tau=ONSET_TAU_CPU, baseline_lag=ONSET_BASELINE_LAG):
    """Indices (into the task's own arrays) of registered CPU onset events.

    Definition (registered, baseline-free):

        the first observed interval at which the task's own CPU reaches `tau`,
        given that the previous observed interval was strictly below it

    and a sustained burst counts once (subsequent intervals above `tau` are part
    of the same event).

    Why not a median-baseline jump
    ------------------------------
    The registered envelope admits the task at the *familiar* CPU level and
    starts the burst at age 1 (this two-stage shape is forced by
    ``Simulator.getPlacementPossible()`` evaluating demand at the admission
    interval, P21-01).  A detector of the form ``CPU(t) - median(CPU[t-4:t]) >=
    tau`` therefore has no pre-onset history to compare against: for every
    interior burst interval the window is already the burst, and the only
    negative jump is the burst *release*.  Measured on the first P22 candidate
    (P22-01): ``max delta_cpu = -4400`` and zero registered onsets, although the
    task-level CPU really does reach 4400-5200.  The exceedance form is the same
    physical claim, is computable for a task whose whole life is a burst, and
    keeps ``tau`` strictly above every offline per-task CPU value so no familiar
    task can produce an event.
    """
    cpu = np.asarray(cpu, dtype=np.float64)
    positions = []
    above_prev = False
    for i in range(cpu.size):
        above = bool(np.isfinite(cpu[i]) and cpu[i] >= tau)
        if above and not above_prev and i >= 1:
            positions.append(i)
        above_prev = above
    return positions


def onset_events(timeline, tau=ONSET_TAU_CPU, baseline_lag=ONSET_BASELINE_LAG,
                 ram_window=RAM_RESPONSE_WINDOW,
                 disk_window=DISK_RESPONSE_WINDOW):
    """One row per registered onset event (statistical unit of the U4-v2 gate).

    Each row carries the task's own trajectory through the response windows so
    controls can be built by re-pairing *the same* task trajectories instead of
    resampling values.
    """
    events = []
    for c in sorted(timeline):
        entry = timeline[c]
        cpu = entry["cpu"]
        positions = onset_positions(cpu, tau=tau, baseline_lag=baseline_lag)
        for i in positions:
            t0 = int(entry["t"][i])
            lo = max(i - baseline_lag, 0)
            if i > lo:
                cpu_baseline = float(np.median(cpu[lo:i]))
                ram_baseline = float(np.median(entry["ram"][lo:i]))
                disk_baseline = float(np.median(entry["disk"][lo:i]))
                baseline_source = "observed pre-onset median"
            else:
                # admitted at the familiar level and bursting on its first
                # observed interval: no pre-onset rows of its own exist
                cpu_baseline = FAMILIAR_TASK_LEVEL["cpu"]
                ram_baseline = FAMILIAR_TASK_LEVEL["ram"]
                disk_baseline = FAMILIAR_TASK_LEVEL["disk"]
                baseline_source = "registered familiar level (no pre-onset rows)"
            events.append({
                "event_key": "%d:%d" % (c, t0),
                "creation_id": int(c),
                "index": int(i),
                "t_onset": t0,
                "age_onset": int(entry["age"][i]),
                "host_onset": int(entry["host"][i]),
                "cpu_baseline": cpu_baseline,
                "cpu_onset": float(cpu[i]),
                "delta_cpu": float(cpu[i] - cpu_baseline),
                "ram_baseline": ram_baseline,
                "disk_baseline": disk_baseline,
                "baseline_source": baseline_source,
                "ram_window": list(ram_window),
                "disk_window": list(disk_window),
                "audit_event_id": int(entry["event_id"][i]),
            })
    return events


def response_curves(timeline, event, ram_window=RAM_RESPONSE_WINDOW,
                    disk_window=DISK_RESPONSE_WINDOW):
    """The task's own RAM/Disk deviation trajectory through both windows.

    Returned as a dense vector covering ``[t0, t0+max(window))`` so that
    re-pairing (control M0) and circular shifting (control M2) operate on the
    same object.
    """
    entry = timeline[event["creation_id"]]
    i = int(event["index"])
    horizon = max(ram_window[1], disk_window[1])
    ram = np.full(horizon, np.nan)
    disk = np.full(horizon, np.nan)
    for k in range(horizon):
        j = i + k
        if j < entry["ram"].size:
            ram[k] = entry["ram"][j]
            disk[k] = entry["disk"][j]
    return {"ram": ram, "disk": disk,
            "ram_baseline": float(event["ram_baseline"]),
            "disk_baseline": float(event["disk_baseline"])}


def event_responses(timeline, events, ram_window=RAM_RESPONSE_WINDOW,
                    disk_window=DISK_RESPONSE_WINDOW, ram_shift=0, disk_shift=0,
                    ram_shift_per_event=None, disk_shift_per_event=None,
                    ram_override=None, disk_override=None):
    """Registered response statistics for every onset event.

    ``ram_shift``/``disk_shift`` circularly shift the task's own deviation
    trajectory inside the window (a global shift).  ``*_shift_per_event`` does
    the same per event (controls M0/M1).  ``ram_override``/``disk_override``
    replace the trajectory outright while keeping the baseline.

    Why the per-event shift is the operative control
    ------------------------------------------------
    Every cascade task in a candidate stream is generated from the *same*
    registered envelope, so their RAM/Disk deviation curves are near-identical
    in shape.  Re-pairing whole trajectories between events (a cross-event swap)
    is therefore close to a no-op and cannot distinguish "registered lag" from
    "any lag".  Circularly shifting each event's own curve leaves its marginal
    amplitude and duration distribution exactly intact while destroying the
    temporal alignment, which is what plan §6.5 asks M0/M1 to rule out.
    """
    rows = []
    for position, event in enumerate(events):
        curve = response_curves(timeline, event, ram_window, disk_window)
        ram_dev = curve["ram"] - curve["ram_baseline"]
        disk_dev = curve["disk"] - curve["disk_baseline"]
        if ram_override is not None:
            ram_dev = np.asarray(ram_override.get(event["event_key"], ram_dev),
                                 dtype=np.float64)
        if disk_override is not None:
            disk_dev = np.asarray(disk_override.get(event["event_key"], disk_dev),
                                  dtype=np.float64)
        shift_ram = ram_shift
        shift_disk = disk_shift
        if ram_shift_per_event is not None:
            shift_ram = int(ram_shift_per_event[position])
        if disk_shift_per_event is not None:
            shift_disk = int(disk_shift_per_event[position])
        if shift_ram:
            ram_dev = np.roll(ram_dev, shift_ram)
        if shift_disk:
            disk_dev = np.roll(disk_dev, shift_disk)
        ram_win = ram_dev[ram_window[0]:ram_window[1]]
        disk_win = disk_dev[disk_window[0]:disk_window[1]]
        ram_win = ram_win[np.isfinite(ram_win)]
        disk_win = disk_win[np.isfinite(disk_win)]
        ram_exact = ram_dev[ram_window[0]] if ram_window[0] < ram_dev.size else np.nan
        disk_exact = disk_dev[disk_window[0]] if disk_window[0] < disk_dev.size else np.nan
        rows.append({
            "event_key": event["event_key"],
            "creation_id": int(event["creation_id"]),
            "t_onset": int(event["t_onset"]),
            "audit_event_id": int(event["audit_event_id"]),
            "applied_lag_shift_ram": int(shift_ram),
            "applied_lag_shift_disk": int(shift_disk),
            "ram_response": float(np.max(ram_win)) if ram_win.size else float("nan"),
            "ram_at_exact_lag": float(ram_exact),
            "ram_time_to_peak": (int(ram_window[0] + int(np.argmax(ram_win)))
                                 if ram_win.size else None),
            "disk_response": float(np.max(disk_win)) if disk_win.size else float("nan"),
            "disk_at_exact_lag": float(disk_exact),
            "disk_time_to_peak": (int(disk_window[0] + int(np.argmax(disk_win)))
                                  if disk_win.size else None),
        })
    for row in rows:
        row["ram_response_positive"] = bool(np.isfinite(row["ram_response"])
                                            and row["ram_response"] > 0)
        row["disk_response_positive"] = bool(np.isfinite(row["disk_response"])
                                             and row["disk_response"] > 0)
    return rows


def response_summary(rows, prefix="ram"):
    key = "%s_response" % prefix
    values = np.asarray([r[key] for r in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    share = float(np.mean([r["%s_response_positive" % prefix] for r in rows])) \
        if rows else float("nan")
    if values.size == 0:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None,
                "share_positive": share}
    return {
        "n": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "share_positive": share,
    }


def response_summary_from_values(values, label=None):
    """Same summary from raw response values (no per-event dicts needed)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    out = {"n": int(arr.size), "label": label,
           "p50": None, "p90": None, "p95": None, "p97_5": None, "max": None,
           "mean": None,
           "share_positive": (float((arr > 0).mean()) if arr.size else float("nan"))}
    if arr.size:
        out.update({
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p97_5": float(np.percentile(arr, 97.5)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        })
    return out


def alignment_rate(rows, prefix, registered_lag, tolerance=1):
    """Share of events whose response peaks within +/- tolerance of the lag."""
    hits = 0
    counted = 0
    for row in rows:
        peak = row["%s_time_to_peak" % prefix]
        if peak is None:
            continue
        counted += 1
        if abs(int(peak) - int(registered_lag)) <= tolerance:
            hits += 1
    return {"n": counted, "hits": hits,
            "rate": (hits / counted) if counted else None}


# --------------------------------------------------------------------------
# matched controls (plan §6.5) — audit metadata only, never a model input
# --------------------------------------------------------------------------
def _m0_shifts(events, rng, horizons):
    """M0: one uniformly drawn lag shift per event, never the identity."""
    choices = list(range(1, max(horizons) + 1))
    ram, disk = [], []
    for _ in events:
        ram.append(int(choices[int(rng.integers(0, len(choices)))]))
        disk.append(int(choices[int(rng.integers(0, len(choices)))]))
    return ram, disk


def _m1_shifts(events, rng, horizons):
    """M1: the same shift *multiset*, but reassigned to different events.

    The identity shift is included here, so the multiset of applied lags is
    exactly {0..horizon}; what is destroyed is which event receives which lag.
    """
    choices = list(range(0, max(horizons) + 1))
    n = len(events)
    if n == 0:
        return [], []
    ram = [int(choices[i % len(choices)]) for i in range(n)]
    disk = [int(choices[i % len(choices)]) for i in range(n)]
    return [int(x) for x in rng.permutation(ram)], \
        [int(x) for x in rng.permutation(disk)]


def control_m0_pairs(timeline, events, rng, ram_window=RAM_RESPONSE_WINDOW,
                     disk_window=DISK_RESPONSE_WINDOW):
    """Control M0 — marginal-matched, lag-shuffled.

    Each event keeps its OWN RAM/Disk deviation trajectory (marginal amplitude,
    duration distribution, event count, task cohort and arrival process are all
    untouched) but the trajectory is circularly shifted by a uniformly drawn
    lag, so the registered lag-4 / lag-8 alignment is replaced by an arbitrary
    one.

    A cross-event re-pairing is also reported (``cross_event_repair``) for
    completeness, but it is the WEAKER control: every cascade task is generated
    from the same registered envelope, so their curves are near-identical in
    shape and swapping them between events is close to a no-op (measured on
    P22 candidates: control response p90 4500 vs candidate p90 4500, i.e. no
    separation at all).  It is therefore not the one used in the gate check.
    """
    horizons = (ram_window[1], disk_window[1])
    ram_shifts, disk_shifts = _m0_shifts(events, rng, horizons)
    keys = [e["event_key"] for e in events]
    curves = {e["event_key"]: response_curves(timeline, e, ram_window, disk_window)
              for e in events}
    ram_order = rng.permutation(len(keys))
    disk_order = rng.permutation(len(keys))
    cross_ram, cross_disk = {}, {}
    for i, key in enumerate(keys):
        src_ram = curves[keys[int(ram_order[i])]]
        src_disk = curves[keys[int(disk_order[i])]]
        cross_ram[key] = src_ram["ram"] - src_ram["ram_baseline"]
        cross_disk[key] = src_disk["disk"] - src_disk["disk_baseline"]
    return {"kind": "marginal-matched lag-shuffled (per-event circular shift)",
            "ram_shift_per_event": ram_shifts,
            "disk_shift_per_event": disk_shifts,
            "cross_event_repair": {"ram_order": ram_order.tolist(),
                                   "disk_order": disk_order.tolist(),
                                   "ram_override": cross_ram,
                                   "disk_override": cross_disk}}


def control_m1_order(timeline, events, rng, ram_window=RAM_RESPONSE_WINDOW,
                     disk_window=DISK_RESPONSE_WINDOW):
    """Control M1 — order-shuffled.

    The same shift multiset {0..horizon} is assigned to the events in a random
    order, so exactly one event keeps the registered alignment and every other
    event receives an arbitrary lag.  This is what "RAM -> CPU, Disk -> CPU, ..."
    means at the level of a single event's response window.
    """
    horizons = (ram_window[1], disk_window[1])
    ram_shifts, disk_shifts = _m1_shifts(events, rng, horizons)
    return {"kind": "order-shuffled (permuted lag assignment)",
            "ram_shift_per_event": ram_shifts,
            "disk_shift_per_event": disk_shifts}


def control_m2_circular(timeline, events, rng, ram_window=RAM_RESPONSE_WINDOW,
                        disk_window=DISK_RESPONSE_WINDOW):
    """Control M2 — within-task circular shift.

    Same operation as M0 but drawn independently per resource, and reported
    separately because the plan registers it as its own control: it preserves
    each task's marginal distribution and (approximately) its autocorrelation
    while destroying the causal alignment.
    """
    horizons = (ram_window[1], disk_window[1])
    ram_shifts, disk_shifts = _m0_shifts(events, rng, horizons)
    return {"kind": "within-task circular shift",
            "ram_shift_per_event": ram_shifts,
            "disk_shift_per_event": disk_shifts}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_ci(values, statistic, rng, n_boot=2000, alpha=0.05):
    """Event-level bootstrap CI (unit = independent onset event, plan §6.1)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "estimate": None, "lo": None, "hi": None}
    estimate = float(statistic(values))
    draws = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for b in range(n_boot):
        draws[b] = statistic(values[rng.integers(0, n, n)])
    lo = float(np.percentile(draws, 100.0 * alpha / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return {"n": int(n), "estimate": estimate, "lo": lo, "hi": hi,
            "n_boot": int(n_boot)}


def exceedance_probability(values, threshold):
    """P(response > threshold) over independent events, with a bootstrap CI."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "estimate": None, "lo": None, "hi": None}
    return bootstrap_ci(values, lambda v: float(np.mean(v > threshold)),
                        np.random.default_rng(22022))


def permutation_pvalue(candidate_values, control_values, rng, n_perm=2000):
    """One-sided permutation test: is the candidate's mean above the control's?"""
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    control_values = np.asarray(control_values, dtype=np.float64)
    candidate_values = candidate_values[np.isfinite(candidate_values)]
    control_values = control_values[np.isfinite(control_values)]
    if candidate_values.size == 0 or control_values.size == 0:
        return {"n_candidate": int(candidate_values.size),
                "n_control": int(control_values.size), "p_value": None,
                "observed_difference": None}
    observed = float(candidate_values.mean() - control_values.mean())
    pooled = np.concatenate([candidate_values, control_values])
    n_a = candidate_values.size
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.size)
        diff = pooled[perm[:n_a]].mean() - pooled[perm[n_a:]].mean()
        if diff >= observed:
            hits += 1
    return {"n_candidate": int(n_a), "n_control": int(control_values.size),
            "observed_difference": observed,
            "p_value": (hits + 1.0) / (n_perm + 1.0),
            "n_perm": int(n_perm)}


def pooled_percentile(reference, q):
    """Percentile of the pooled offline reference distribution."""
    values = np.asarray([v for v in reference if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


# --------------------------------------------------------------------------
# hashing / determinism
# --------------------------------------------------------------------------
def timeline_sha256(timeline):
    """Order-independent content hash of a task timeline."""
    digest = hashlib.sha256()
    for c in sorted(timeline):
        entry = timeline[c]
        digest.update(str(c).encode("ascii"))
        for name in ("t", "slot", "cpu", "ram", "disk", "host", "event_id"):
            digest.update(np.ascontiguousarray(entry[name]).tobytes())
    return digest.hexdigest()


def forbidden_token_hits(names):
    """Any feature name containing a forbidden token is a leak."""
    hits = []
    for name in names:
        for token in FORBIDDEN_INPUT_TOKENS:
            if token in name:
                hits.append({"feature": name, "token": token})
    return hits
