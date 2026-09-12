"""Protocol 022 P22-S2/S3 — task/event-level U4-v2 unseen audit.

Replaces the P21 host-aggregate U4 instrument that failed as an instrument
(P21-09): protocol 021 measured ``Corr(CPU_t, RAM_{t+4})`` on ``[T, H, 7]`` host
aggregates, where co-resident heavy tasks already make the resources co-move, so
a per-task temporal cascade could not be separated from co-residency.

This audit works at the scale the mechanism is defined at:

    statistical unit  = creation_id + CPU onset event  (one sustained event
                        counts once, plan §6.1)
    onset             = task's own Delta CPU >= registered threshold
    RAM_response      = max RAM_task[t0+4 : t0+14] - median RAM_task[t0-4 : t0]
    Disk_response     = max Disk_task[t0+8 : t0+18] - median Disk_task[t0-4 : t0]
    reference         = the SAME instrument over every auditable offline corpus,
                        built and locked BEFORE any candidate is read
    controls          = M0 lag-shuffled, M1 order-shuffled, M2 within-task
                        circular shift (plan §6.5)

Offline corpora that do carry task identity (verified, not assumed):
    P014  container_demand_series.npy                  [replays, T, H*7]
    P019  adaptation_data/raw/*/stream.npz             demands + creation_ids
    P019  stationary_streams, drift_streams, dev_streams
    P020  adaptation_data/v1/episodes/*.npz            demands + creation_ids
    P020  drift_streams/*/stream.npz                   demands + creation_ids

Corpora that do NOT (only host aggregates) are reported as excluded with the
reason, never silently substituted.

U4-v2 pass rule (plan §7.2), RAM is primary, Disk is reported as secondary:

    A. candidate RAM_response p90 > offline RAM_response p97.5
    B. bootstrap 95% CI lower bound of P(candidate RAM_response > offline p97.5)
       >= 0.20
    C. registered-lag alignment significantly > M0/M1/M2 (permutation p < 0.01)
    D. n_valid_followup_events >= 50
    plus  exact 12-step scoring-window overlap with P20 S6 == 0
    plus  VM overlap with P20 S6 train/dev == empty
    plus  Data Gate == PASS

Usage:
    python analyze_ftmoe_protocol022_unseen.py --build-reference
    python analyze_ftmoe_protocol022_unseen.py
    python analyze_ftmoe_protocol022_unseen.py --candidates p025_seed600_steps1200
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import ftmoe_protocol022_core as core

ROOT = Path(__file__).resolve().parent
P22 = ROOT / "artifacts/ftmoe_online/protocol_022"
STREAMS = P22 / "pilot_streams"
AUDIT = P22 / "audit_v2"
P14 = ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical"
P19 = ROOT / "artifacts/ftmoe_online/protocol_019"
P20 = ROOT / "artifacts/ftmoe_online/protocol_020"
SPLIT_PATH = P20 / "vm_split.json"

FEATURE_CPU, FEATURE_RAM, FEATURE_DISK = 0, 1, 4
WINDOW = 12
RAM_WINDOW = core.RAM_RESPONSE_WINDOW
DISK_WINDOW = core.DISK_RESPONSE_WINDOW
REGISTERED_LAG_RAM = core.REGISTERED_LAG_RAM
REGISTERED_LAG_DISK = core.REGISTERED_LAG_DISK
ONSET_TAU = core.ONSET_TAU_CPU
MIN_FOLLOWUP_EVENTS = 50
EXCEEDANCE_LOWER_BOUND = 0.20
PERMUTATION_ALPHA = 0.01
N_PERMUTATIONS = 2000
N_BOOTSTRAP = 2000

# Familiar per-container ceilings.  Per-sample offline maxima are measured and
# reported instead of assumed (the P20 drift stream switches ram_upper per
# phase, so a single constant would be wrong).
FAMILIAR_CEILING = {"cpu": 1860.0, "ram": 1400.0, "disk": 9000.0}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf8")
    temporary.replace(path)


# --------------------------------------------------------------------------
# offline corpora -> task timelines
# --------------------------------------------------------------------------
def timeline_from_slot_arrays(demands, creation_ids):
    """Build a task timeline from per-slot demand and creation-id arrays.

    ``(demands[t, slot], creation_ids[t, slot])`` is exactly the layout every
    019/020 collector wrote, so the offline reference uses the same instrument
    as the candidate instead of a proxy.
    """
    demands = np.asarray(demands, dtype=np.float64)
    creation_ids = np.asarray(creation_ids, dtype=np.int64)
    steps, slots = creation_ids.shape
    if demands.shape[:2] != creation_ids.shape:
        raise ValueError("demands/creation_ids shape mismatch: %s vs %s"
                         % (demands.shape, creation_ids.shape))
    t_idx, s_idx = np.nonzero(creation_ids >= 0)
    live = creation_ids[t_idx, s_idx] >= 0
    t_idx, s_idx = t_idx[live], s_idx[live]
    return core.build_task_timeline(t_idx, s_idx, creation_ids[t_idx, s_idx],
                                   demands[t_idx, s_idx],
                                   np.full(t_idx.shape, -1, dtype=np.int64))


def offline_task_corpora(max_streams=60):
    """{name: {"timelines": [...], "source": str}} for task-indexed corpora."""
    corpora = {}

    p14 = P14 / "container_demand_series.npy"
    if p14.is_file():
        arr = np.load(p14)
        timelines = []
        for i in range(arr.shape[0]):
            flat = np.asarray(arr[i], dtype=np.float64)
            if flat.ndim == 2 and flat.shape[1] % 7 == 0:
                flat = flat.reshape(flat.shape[0], flat.shape[1] // 7, 7)
            if flat.ndim != 3 or flat.shape[2] != 7:
                continue
            # P014 stores per-container demand on the host axis (no creation
            # ids); each host column is treated as one aggregate "task" and the
            # corpus is therefore reported as HOST-AGGREGATE ONLY.
            timelines.append(("host_aggregate", flat))
        if timelines:
            corpora["p014_physical_container_demand"] = {
                "kind": "host_aggregate_only",
                "arrays": [a for _, a in timelines],
                "source": str(p14.relative_to(ROOT)).replace("\\", "/"),
                "exclusion_reason": (
                    "no creation_id column exists in the P014 artifact: the "
                    "instrument can only be applied to its host axis, so this "
                    "corpus contributes to the host-aggregate reference, not to "
                    "the task-level one"),
            }

    p19 = []
    dirs = [P19 / "adaptation_data/raw"]
    for sub in ("stationary_streams", "drift_streams", "dev_streams"):
        dirs.append(P19 / sub)
    for base in dirs:
        for seed_dir in sorted(base.glob("seed*_steps*"))[:max_streams]:
            f = seed_dir / "stream.npz"
            if not f.is_file():
                continue
            with np.load(f, allow_pickle=True) as z:
                if "demands" not in z.files or "creation_ids" not in z.files:
                    continue
                p19.append((str(f.relative_to(ROOT)).replace("\\", "/"),
                            timeline_from_slot_arrays(z["demands"],
                                                      z["creation_ids"])))
    if p19:
        corpora["p019_adaptation_and_development"] = {
            "kind": "task_level",
            "timelines": [t for _, t in p19],
            "streams": [s for s, _ in p19],
            "source": "artifacts/ftmoe_online/protocol_019/**/stream.npz",
        }

    p20_eps = []
    for f in sorted((P20 / "adaptation_data/v1/episodes").glob("*.npz"))[:max_streams]:
        with np.load(f, allow_pickle=True) as z:
            keys = z.files
            ids = z["creation_ids"] if "creation_ids" in keys else None
            dem = z["demands"] if "demands" in keys else None
            if dem is None or ids is None:
                continue
            p20_eps.append((str(f.relative_to(ROOT)).replace("\\", "/"),
                            timeline_from_slot_arrays(dem, ids)))
    if p20_eps:
        corpora["p020_s6_adaptation"] = {
            "kind": "task_level",
            "timelines": [t for _, t in p20_eps],
            "streams": [s for s, _ in p20_eps],
            "source": "artifacts/ftmoe_online/protocol_020/adaptation_data/v1/episodes/*.npz",
        }

    p20_dev = []
    for f in sorted((P20 / "drift_streams").glob("*/stream.npz"))[:max_streams]:
        with np.load(f, allow_pickle=True) as z:
            if "demands" not in z.files or "creation_ids" not in z.files:
                continue
            p20_dev.append((str(f.relative_to(ROOT)).replace("\\", "/"),
                            timeline_from_slot_arrays(z["demands"],
                                                      z["creation_ids"])))
    if p20_dev:
        corpora["p020_development_streams"] = {
            "kind": "task_level",
            "timelines": [t for _, t in p20_dev],
            "streams": [s for s, _ in p20_dev],
            "source": "artifacts/ftmoe_online/protocol_020/drift_streams/*/stream.npz",
            "note": ("dev500/dev501 were consumed by P20 R1 method selection and "
                     "development evaluation (P21-05); they may be used as "
                     "REFERENCE distribution only, never as a clean candidate"),
        }
    return corpora


def host_arrays_of(timeline):
    """Host-aggregate analogue of a timeline (for the P014-only corpus)."""
    return timeline


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------
def summarize(values, label):
    """Percentile summary of a pooled response distribution."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "label": label}
    return {
        "n": int(values.size), "label": label,
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p97_5": float(np.percentile(values, 97.5)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def reference_from(corpora, rng_seed=22022, bootstrap=None):
    """Task-level offline reference at BOTH thresholds.

    ``registered`` uses the registered onset threshold (2600).  Every offline
    per-task CPU demand is clipped at the familiar ceiling (1860), so this
    threshold yields zero offline onsets -- measured and reported, never
    substituted.

    ``calibrated`` re-runs the identical instrument at
    ``core.ONSET_TAU_CALIBRATED`` (500, chosen from offline data only before
    any candidate was read), which does produce familiar-percentile events and
    therefore a usable p90/p97.5 reference.
    """
    scales = {"registered": ONSET_TAU, "calibrated": core.ONSET_TAU_CALIBRATED}
    out = {}
    for scale, tau in scales.items():
        ram_pool, disk_pool = [], []
        ram_stream_p90, disk_stream_p90 = [], []
        per_corpus = {}
        zero_onset_timelines = 0
        for name, corpus in corpora.items():
            if corpus["kind"] != "task_level":
                per_corpus[name] = {
                    "kind": corpus["kind"],
                    "streams": len(corpus.get("arrays", [])),
                    "exclusion_reason": corpus.get("exclusion_reason"),
                    "task_level_onsets": None,
                }
                continue
            onsets = 0
            corpus_ram, corpus_disk = [], []
            for timeline in corpus["timelines"]:
                events = core.onset_events(timeline, tau=tau)
                onsets += len(events)
                if not events:
                    zero_onset_timelines += 1
                    continue
                rows = core.event_responses(timeline, events)
                corpus_ram.extend([r["ram_response"] for r in rows])
                corpus_disk.extend([r["disk_response"] for r in rows])
                finite = [v for v in (r["ram_response"] for r in rows)
                          if np.isfinite(v)]
                if finite:
                    ram_stream_p90.append(float(np.percentile(finite, 90)))
                finite = [v for v in (r["disk_response"] for r in rows)
                          if np.isfinite(v)]
                if finite:
                    disk_stream_p90.append(float(np.percentile(finite, 90)))
            ram_pool.extend(corpus_ram)
            disk_pool.extend(corpus_disk)
            per_corpus[name] = {
                "kind": "task_level",
                "streams": len(corpus["timelines"]),
                "task_level_onsets": onsets,
                "ram_response": core.response_summary_from_values(
                    corpus_ram, "corpus RAM response"),
                "disk_response": core.response_summary_from_values(
                    corpus_disk, "corpus Disk response"),
            }
        out[scale] = {
            "tau_cpu": tau,
            "corpora": per_corpus,
            "task_level": {
                "ram_response": summarize(ram_pool, "offline task-level RAM response"),
                "disk_response": summarize(disk_pool, "offline task-level Disk response"),
                "ram_stream_p90": summarize(ram_stream_p90, "per-stream p90 of RAM response"),
                "disk_stream_p90": summarize(disk_stream_p90, "per-stream p90 of Disk response"),
                "streams_without_any_onset": int(zero_onset_timelines),
            },
        }
    reference = {
        "protocol": "022",
        "instrument": "task-level U4-v2 (creation_id + CPU onset event)",
        "unit": "creation_id + CPU onset event",
        "onset_definition": {
            "registered_tau_cpu": ONSET_TAU,
            "calibrated_tau_cpu": core.ONSET_TAU_CALIBRATED,
            "baseline_lag": core.ONSET_BASELINE_LAG,
            "ram_window": list(RAM_WINDOW),
            "disk_window": list(DISK_WINDOW),
        },
        "scale_selection_rule": (
            "the calibrated scale is a reference instrument, not a second "
            "attempt at the gate: its threshold (500) is the smallest round value "
            "strictly above the pooled offline p99 of non-negative task-level "
            "Delta CPU (481/419/327) and strictly below every offline corpus "
            "maximum (861/1688/1688), fixed from offline data only before any "
            "candidate stream was read"),
        "registered_scale": out["registered"],
        "calibrated_scale": out["calibrated"],
        # flat aliases kept for the candidate report and for the selected.json
        "task_level": out["calibrated"]["task_level"],
        "corpora": out["calibrated"]["corpora"],
    }
    if bootstrap:
        reference["bootstrap"] = bootstrap
    return reference


def measure_familiar_maxima(corpora, max_streams=40):
    """PER-TASK demand maxima of the offline corpora, measured not assumed.

    Only live-container demand rows count.  ``host_features`` is deliberately
    NOT used: it is a host aggregate (sum over co-resident containers), so its
    magnitudes say nothing about what a single task demanded.  A mixture of the
    two would make the "familiar ceiling" meaningless.
    """
    maxima = {"cpu": [], "ram": [], "disk": []}
    per_corpus = {}
    for name, corpus in corpora.items():
        local = {k: 0.0 for k in maxima}
        counted = 0
        timelines = corpus.get("timelines") or []
        for timeline in timelines[:max_streams]:
            if not isinstance(timeline, dict):
                continue
            for entry in timeline.values():
                local["cpu"] = max(local["cpu"], float(entry["cpu"].max()))
                local["ram"] = max(local["ram"], float(entry["ram"].max()))
                local["disk"] = max(local["disk"], float(entry["disk"].max()))
                counted += 1
        if corpus["kind"] != "task_level":
            per_corpus[name] = {
                "kind": corpus["kind"],
                "max": None,
                "exclusion_reason": corpus.get("exclusion_reason"),
            }
            continue
        for key in maxima:
            maxima[key].append(local[key])
        per_corpus[name] = {"kind": "task_level", "n_tasks_measured": counted,
                            "max": local}
    return {
        "per_corpus_max": per_corpus,
        "pooled": {k: {"max": float(np.max(v)) if v else None,
                       "min": float(np.min(v)) if v else None}
                   for k, v in maxima.items()},
        "registered_onset_tau_cpu": ONSET_TAU,
        "fallback_familiar_task_level": dict(core.FAMILIAR_TASK_LEVEL),
        "onset_tau_above_every_offline_task_cpu": bool(
            maxima["cpu"] and max(maxima["cpu"]) < ONSET_TAU),
        "note": ("the registered onset threshold must sit strictly above every "
                 "offline per-task CPU demand, otherwise an 'onset' could be a "
                 "familiar event rather than the registered mechanism"),
    }


# --------------------------------------------------------------------------
# candidate
# --------------------------------------------------------------------------
def load_candidate(stream_dir):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    audit_path = stream_dir / "unseen_data_audit.json"
    data_gate = json.loads(audit_path.read_text(encoding="utf8")) \
        if audit_path.is_file() else None
    timeline_path = stream_dir / "task_timeline.npz"
    if not timeline_path.is_file():
        raise FileNotFoundError("candidate has no task_timeline.npz: %s" % stream_dir)
    with np.load(timeline_path, allow_pickle=True) as z:
        timeline = core.build_task_timeline(
            np.asarray(z["time"]), np.asarray(z["slot_index"]),
            np.asarray(z["creation_id"]), np.asarray(z["demand"], dtype=np.float64),
            np.asarray(z["host_id"]), np.asarray(z["cascade_event_id"]),
            np.asarray(z["cascade_phase"]))
    return manifest, timeline, data_gate


def exact_window_overlap(stream_dir):
    """Exact 12-step scoring-window hashes shared with P20 S6 / P20 dev."""
    with np.load(stream_dir / "stream.npz", allow_pickle=True) as z:
        demands = np.asarray(z["demands"], dtype=np.float64)

    def hashes_of(array):
        out = set()
        for t in range(array.shape[0] - WINDOW + 1):
            block = np.ascontiguousarray(array[t:t + WINDOW])
            out.add(hashlib.sha256(block.tobytes()).hexdigest())
        return out

    candidate = hashes_of(demands)
    s6, p20_dev = set(), set()
    for f in sorted((P20 / "adaptation_data/v1/episodes").glob("*.npz")):
        with np.load(f, allow_pickle=True) as ep:
            if "demands" in ep.files:
                s6 |= hashes_of(np.asarray(ep["demands"], dtype=np.float64))
    for f in sorted((P20 / "drift_streams").glob("*/stream.npz")):
        with np.load(f, allow_pickle=True) as st:
            if "demands" in st.files:
                p20_dev |= hashes_of(np.asarray(st["demands"], dtype=np.float64))
    return {
        "scoring_window_exact_overlap_p20_s6": len(candidate & s6),
        "scoring_window_exact_overlap_p20_dev_streams": len(candidate & p20_dev),
    }


def p20_s6_cohorts():
    train, dev = set(), set()
    for split, target in (("train", train), ("dev", dev)):
        for man in sorted((P20 / "adaptation_data/raw" / split).glob(
                "*/seed*_steps*/manifest.json")):
            payload = json.loads(man.read_text(encoding="utf8"))
            ids = payload.get("cohort_vm_ids")
            if ids:
                target.update(int(i) for i in ids)
    return train, dev


def analyse_candidate(stream_dir, reference, rng_seed=22022):
    manifest, timeline, data_gate = load_candidate(stream_dir)
    events = core.onset_events(timeline)                     # registered tau
    events_cal = core.onset_events(timeline, tau=core.ONSET_TAU_CALIBRATED)
    rows = core.event_responses(timeline, events)
    rows_cal = core.event_responses(timeline, events_cal)
    integrity = core.timeline_integrity(timeline)
    continuity = core.migration_continuity(timeline)

    offline_cal = reference["calibrated_scale"]["task_level"]
    offline_registered = reference["registered_scale"]["task_level"]
    ram_values = np.asarray([r["ram_response"] for r in rows], dtype=np.float64)
    ram_values_cal = np.asarray([r["ram_response"] for r in rows_cal],
                                dtype=np.float64)
    off_p97_5 = offline_cal["ram_response"].get("p97_5")
    off_p90 = offline_cal["ram_response"].get("p90")
    off_max = offline_cal["ram_response"].get("max")
    # NOTE: only like-for-like comparisons are admitted.  A per-task *response*
    # (baseline -> window peak) may never be compared against an offline absolute
    # demand level; the offline reference therefore carries its own response
    # distribution at the calibrated threshold, computed by the same function.
    familiar_task_level = reference.get("familiar_maxima", {}).get(
        "fallback_familiar_task_level", {})

    rng = np.random.default_rng(rng_seed)
    exceedance = core.exceedance_probability(
        ram_values, off_p97_5 if off_p97_5 is not None else np.inf)
    exceedance_cal = core.exceedance_probability(
        ram_values_cal, off_p97_5 if off_p97_5 is not None else np.inf)

    # ---- controls -----------------------------------------------------
    control_rng = np.random.default_rng(rng_seed + 1)
    m0 = core.control_m0_pairs(timeline, events, control_rng)
    m1 = core.control_m1_order(timeline, events, control_rng)
    m2 = core.control_m2_circular(timeline, events, control_rng)
    control_rows = {
        name: core.event_responses(
            timeline, events,
            ram_shift_per_event=spec["ram_shift_per_event"],
            disk_shift_per_event=spec["disk_shift_per_event"])
        for name, spec in (("M0", m0), ("M1", m1), ("M2", m2))}
    # the weaker cross-event re-pairing is reported but is not the gate control
    cross_repair = core.event_responses(
        timeline, events, ram_override=m0["cross_event_repair"]["ram_override"],
        disk_override=m0["cross_event_repair"]["disk_override"])
    candidate_alignment = core.alignment_rate(rows, "ram", REGISTERED_LAG_RAM)
    control_alignment = {name: core.alignment_rate(crows, "ram", REGISTERED_LAG_RAM)
                         for name, crows in control_rows.items()}
    perm_rng = np.random.default_rng(rng_seed + 2)

    def aligned_flags(crows, tolerance=1):
        out = []
        for row in crows:
            peak = row["ram_time_to_peak"]
            if peak is None:
                continue
            out.append(1.0 if abs(int(peak) - REGISTERED_LAG_RAM) <= tolerance
                       else 0.0)
        return np.asarray(out, dtype=np.float64)

    candidate_flags = aligned_flags(rows)
    alignment_tests = {}
    for name, crows in control_rows.items():
        alignment_tests[name] = core.permutation_pvalue(
            candidate_flags, aligned_flags(crows), perm_rng, n_perm=N_PERMUTATIONS)
    control_perm_rng = np.random.default_rng(rng_seed + 3)
    response_tests = {
        name: core.permutation_pvalue(
            ram_values,
            np.asarray([r["ram_response"] for r in crows], dtype=np.float64),
            control_perm_rng, n_perm=N_PERMUTATIONS)
        for name, crows in control_rows.items()}

    rng_boot = np.random.default_rng(rng_seed + 4)
    ram_bootstrap = core.bootstrap_ci(
        ram_values, lambda v: float(np.percentile(v, 90)), rng_boot,
        n_boot=N_BOOTSTRAP)
    ram_bootstrap_cal = core.bootstrap_ci(
        ram_values_cal, lambda v: float(np.percentile(v, 90)),
        np.random.default_rng(rng_seed + 5), n_boot=N_BOOTSTRAP)

    # ---- overlaps -----------------------------------------------------
    overlap = exact_window_overlap(stream_dir)
    train_ids, dev_ids = p20_s6_cohorts()
    cohort = set(int(i) for i in manifest.get("cohort_vm_ids") or [])
    overlap["cohort_intersection_p20_s6_train"] = sorted(cohort & train_ids)
    overlap["cohort_intersection_p20_s6_dev"] = sorted(cohort & dev_ids)
    overlap["cohort_size"] = len(cohort)
    overlap["cohort_disjoint_from_p20_s6"] = not (cohort & (train_ids | dev_ids))

    followup = int(np.isfinite(ram_values).sum())
    registered_reference_onsets = sum(
        v.get("task_level_onsets") or 0
        for v in reference["registered_scale"]["corpora"].values())
    checks = {
        "A_candidate_p90_above_offline_p97_5": bool(
            ram_bootstrap["estimate"] is not None and off_p97_5 is not None
            and ram_bootstrap["estimate"] > off_p97_5),
        "A2_candidate_response_above_every_offline_task": bool(
            off_max is not None and ram_values.size
            and float(np.nanmedian(ram_values)) > off_max),
        "A3_candidate_beats_every_lag_control_response": bool(
            all(response_tests[name]["p_value"] is not None
                and response_tests[name]["p_value"] < PERMUTATION_ALPHA
                for name in response_tests)),
        "B_exceedance_ci_lower_ge_0.20": bool(
            exceedance["lo"] is not None and exceedance["lo"] >= EXCEEDANCE_LOWER_BOUND),
        "C_alignment_beats_all_controls": bool(
            all(alignment_tests[name]["p_value"] is not None
                and alignment_tests[name]["p_value"] < PERMUTATION_ALPHA
                for name in alignment_tests)),
        "D_followup_events_ge_50": bool(followup >= MIN_FOLLOWUP_EVENTS),
        "window_overlap_zero": bool(
            overlap["scoring_window_exact_overlap_p20_s6"] == 0
            and overlap["scoring_window_exact_overlap_p20_dev_streams"] == 0),
        "p20_s6_source_overlap_empty": bool(overlap["cohort_disjoint_from_p20_s6"]),
        "data_gate_passed": bool(data_gate and data_gate.get("passed")),
    }
    u4_passed = all(checks.values())
    result = {
        "candidate": manifest["name"],
        "stream_dir": str(stream_dir.relative_to(ROOT)).replace("\\", "/"),
        "regime_id": manifest.get("regime_id"),
        "cascade_task_probability": manifest["cascade_task_probability"],
        "mechanism_seed": manifest.get("mechanism", {}).get("mechanism_seed"),
        "stream_sha256": manifest.get("stream_sha256"),
        "task_timeline_sha256": manifest.get("task_timeline_sha256"),
        "steps": manifest["steps"],
        "instrument": "task-level U4-v2 (creation_id + CPU onset event)",
        "onset_definition": manifest.get("onset_definition"),
        "data_gate": data_gate,
        "task_level_integrity": integrity,
        "migration_continuity": {
            "n_tasks": continuity["n_tasks"],
            "n_tasks_with_migration": continuity["n_tasks_with_migration"]},
        "n_task_onsets": len(events),
        "n_task_onsets_calibrated": len(events_cal),
        "n_valid_followup_events": followup,
        "n_onsets_with_registered_cascade_window": int(sum(
            1 for e in events if e["audit_event_id"] >= 0)),
        "ram_response": core.response_summary(rows, "ram"),
        "disk_response": core.response_summary(rows, "disk"),
        "ram_response_calibrated": core.response_summary(rows_cal, "ram"),
        "disk_response_calibrated": core.response_summary(rows_cal, "disk"),
        "ram_at_exact_lag": {
            "p50": float(np.nanpercentile(
                [r["ram_at_exact_lag"] for r in rows], 50)) if rows else None,
            "p90": float(np.nanpercentile(
                [r["ram_at_exact_lag"] for r in rows], 90)) if rows else None,
        },
        "ram_response_bootstrap_p90": ram_bootstrap,
        "ram_response_bootstrap_p90_calibrated": ram_bootstrap_cal,
        "exceedance_probability": exceedance,
        "exceedance_probability_calibrated": exceedance_cal,
        "registered_lag_alignment": candidate_alignment,
        "control_alignment": control_alignment,
        "control_response_tests": response_tests,
        "alignment_tests": alignment_tests,
        "controls": {
            "definition": reference.get("control_definitions"),
            "gate_control": ("M0/M1/M2 are per-event circular lag shifts: each "
                             "event keeps its own deviation curve (marginal "
                             "amplitude and duration untouched) while the "
                             "registered lag alignment is destroyed"),
            "M0_kind": m0["kind"],
            "M1_kind": m1["kind"],
            "M2_kind": m2["kind"],
            "M0_ram_response": core.response_summary(control_rows["M0"], "ram"),
            "M1_ram_response": core.response_summary(control_rows["M1"], "ram"),
            "M2_ram_response": core.response_summary(control_rows["M2"], "ram"),
            "M0_disk_response": core.response_summary(control_rows["M0"], "disk"),
            "M1_disk_response": core.response_summary(control_rows["M1"], "disk"),
            "M2_disk_response": core.response_summary(control_rows["M2"], "disk"),
            "M0_cross_event_repair_ram_response":
                core.response_summary(cross_repair, "ram"),
            "M0_cross_event_repair_note": (
                "weak control, reported for completeness only: all cascade tasks "
                "share the registered envelope, so swapping whole curves between "
                "events is nearly a no-op and shows no separation"),
        },
        "disk_secondary": {
            "candidate_p90": core.response_summary(rows, "disk").get("p90"),
            "offline_p97_5": offline_cal["disk_response"].get("p97_5"),
            "role": "secondary evidence only; the Disk branch never gates U4-v2 "
                    "(plan §7.2)",
        },
        "offline_reference_used": {
            "scale": "calibrated (tau=%.0f), with the registered scale reported "
                     "separately" % core.ONSET_TAU_CALIBRATED,
            "ram_response_p90": off_p90,
            "ram_response_p97_5": off_p97_5,
            "ram_response_max": off_max,
            "ram_response_n": offline_cal["ram_response"].get("n"),
            "disk_response_p97_5": offline_cal["disk_response"].get("p97_5"),
            "registered_scale_onsets_offline": registered_reference_onsets,
            "registered_scale_ram_response": offline_registered["ram_response"],
            "comparison_discipline": ("response statistics are only ever compared "
                                      "against offline response statistics "
                                      "computed by the same function; offline "
                                      "absolute demand levels are reported "
                                      "separately under familiar_maxima"),
            "offline_familiar_task_level": familiar_task_level,
        },
        "overlap": overlap,
        "u4_v2_gate": {"checks": checks, "passed": u4_passed},
    }
    result["claim_level"] = {
        "U0_new_seed_or_replay": True,
        "U1_source_disjoint_p20_s6": checks["p20_s6_source_overlap_empty"],
        "U2_parameter_combo_unseen": True,
        "U3_generator_mechanism_unseen": True,
        "U4_distributional_novelty_task_level": u4_passed,
        "U4_limitation": ("source-unseen relative to the FULL historical chain "
                          "(P014 upstream VM selection is not registered "
                          "anywhere) remains UNVERIFIED; the primary claim is "
                          "mechanism-level U3 + task-level distributional U4-v2, "
                          "never absolute source-unseen"),
    }
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    ref_path = AUDIT / "offline_reference_v2.json"
    if args.skip_reference and ref_path.is_file():
        reference = json.loads(ref_path.read_text(encoding="utf8"))
    else:
        corpora = offline_task_corpora()
        reference = reference_from(corpora)
        reference["familiar_maxima"] = measure_familiar_maxima(corpora)
        from simulator.workload.BitbrainWorkloadProtocol022 import MATCHED_CONTROLS
        reference["control_definitions"] = dict(MATCHED_CONTROLS)
        reference["corpus_inventory"] = {
            name: {"kind": c["kind"], "path": c.get("source"),
                   "n_streams": len(c.get("timelines") or c.get("arrays") or []),
                   "exclusion_reason": c.get("exclusion_reason")}
            for name, c in corpora.items()}
        reference["p20_s6_cohort"] = {
            "train": len(p20_s6_cohorts()[0]), "dev": len(p20_s6_cohorts()[1])}
        write_json(ref_path, reference)
        print(json.dumps({
            "offline_reference": str(ref_path.relative_to(ROOT)).replace("\\", "/"),
            "task_level_onsets": {k: v.get("task_level_onsets")
                                  for k, v in reference["corpora"].items()},
            "offline_ram_response": reference["task_level"]["ram_response"],
            "familiar_maxima": reference["familiar_maxima"]["pooled"],
        }, ensure_ascii=False), flush=True)
    if args.build_reference:
        return

    names = args.candidates or [d.name for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                                if (d / "task_timeline.npz").is_file()]
    if not names:
        raise SystemExit("no candidate streams in %s" % STREAMS)
    results = []
    for name in names:
        stream_dir = STREAMS / name
        if not (stream_dir / "task_timeline.npz").is_file():
            print("skip (incomplete):", name)
            continue
        result = analyse_candidate(stream_dir, reference)
        tag = "p%03d" % round(result["cascade_task_probability"] * 100)
        write_json(AUDIT / ("candidate_%s.json" % tag), result)
        results.append(result)
        print(json.dumps({
            "candidate": name,
            "data_gate": result["data_gate"]["passed"] if result["data_gate"] else None,
            "u4_v2": result["u4_v2_gate"]["passed"],
            "n_onsets": result["n_task_onsets"],
            "n_onsets_calibrated": result["n_task_onsets_calibrated"],
            "ram_p90": result["ram_response"]["p90"],
            "ram_p90_calibrated": result["ram_response_calibrated"]["p90"],
            "offline_p97_5_calibrated": result["offline_reference_used"]["ram_response_p97_5"],
            "offline_calibrated_ram_response": result["offline_reference_used"]["ram_response_p97_5"],
            "exceedance_calibrated": result["exceedance_probability_calibrated"],
        }, ensure_ascii=False), flush=True)

    if results:
        def score(r):
            return (1 if r["data_gate"] and r["data_gate"].get("passed") else 0,
                    1 if r["u4_v2_gate"]["passed"] else 0,
                    r["n_valid_followup_events"])
        ranked = sorted(results, key=score, reverse=True)
        best = ranked[0]
        gates_ok = bool(best["data_gate"] and best["data_gate"].get("passed")
                        and best["u4_v2_gate"]["passed"])
        stop = {}
        if not any(r["u4_v2_gate"]["passed"] for r in results):
            stop["STOP-A2"] = ("U4-v2 task/event-level novelty not established for "
                              "any candidate: no unseen model experiment may be "
                              "run under the unseen name (plan §7.3, §22)")
        if not any(r["data_gate"] and r["data_gate"].get("passed") for r in results):
            stop["STOP-data-gate"] = "no candidate passed the Pilot Data Gate"
        payload = {
            "protocol": "022", "step": "P22-S2/S3",
            "selection_rule": "data physics and event structure only; model scores were never consulted",
            "instrument": "task-level U4-v2",
            "offline_reference": str(ref_path.relative_to(ROOT)).replace("\\", "/"),
            "offline_reference_sha256": sha(ref_path),
            "gate_outcome": {
                "data_gate_passed": [r["candidate"] for r in results
                                     if r["data_gate"] and r["data_gate"].get("passed")],
                "data_gate_failed": [r["candidate"] for r in results
                                     if not (r["data_gate"] and r["data_gate"].get("passed"))],
                "u4_v2_passed": [r["candidate"] for r in results
                                 if r["u4_v2_gate"]["passed"]],
                "u4_v2_failed": [r["candidate"] for r in results
                                 if not r["u4_v2_gate"]["passed"]],
            },
            "candidates": [{
                "candidate": r["candidate"],
                "stream_dir_name": Path(r["stream_dir"]).name,
                "cascade_task_probability": r["cascade_task_probability"],
                "data_gate_passed": bool(r["data_gate"] and r["data_gate"].get("passed")),
                "u4_v2_passed": r["u4_v2_gate"]["passed"],
                "u4_v2_checks": r["u4_v2_gate"]["checks"],
                "n_task_onsets": r["n_task_onsets"],
                "n_valid_followup_events": r["n_valid_followup_events"],
                "ram_response": r["ram_response"],
                "disk_response": r["disk_response"],
                "exceedance_probability": r["exceedance_probability"],
                "registered_lag_alignment": r["registered_lag_alignment"],
                "deployment_rejection_rate":
                    (r["data_gate"] or {}).get("metrics", {}).get("deployment_rejection_rate"),
                "migration_rejection_rate":
                    (r["data_gate"] or {}).get("metrics", {}).get("migration_rejection_rate"),
                "prevalence": (r["data_gate"] or {}).get("metrics", {}).get("prevalence"),
            } for r in results],
            "selected": best["candidate"] if gates_ok else None,
            "selected_stream_dir": best["stream_dir"] if gates_ok else None,
            "selected_reason": ("both the Pilot Data Gate and the U4-v2 gate passed"
                                if gates_ok else
                                "no candidate passed both gates; STOP-A2 applies"),
            "stop_conditions_hit": [k for k in sorted(stop)],
            "stop_details": stop,
            "next_step_if_pass": "P22-S4 learnability-v2 (onset task, persistence-controlled)",
            "next_step_if_fail": "STOP-A2: report the instrument and distributional result; do not run A/C/D under the unseen name",
        }
        write_json(AUDIT / "selected.json", payload)
        print(json.dumps({"selected": payload["selected"],
                          "stop_conditions_hit": payload["stop_conditions_hit"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
