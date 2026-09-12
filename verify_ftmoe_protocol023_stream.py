"""Protocol 023 S2 — independent stream verifier.

Usage:
    python verify_ftmoe_protocol023_stream.py <stream_dir>
    python verify_ftmoe_protocol023_stream.py <stream_dir> --json-out verdict.json

What "independent" means here
-----------------------------
The verifier re-derives every *measured* number from the saved arrays
(``stream.npz``, ``task_timeline.npz``) and from the saved JSON with its own
instrument calls; the manifest's numbers are treated as **claims to be
falsified**, never as inputs:

  * the registered phase table, the registered horizon, the registered switch
    points, the per-regime onset thresholds and the P22-registered gate
    arithmetic are imported from ``prepare_ftmoe_protocol023_stream`` (a shared
    *registration*, not a measured number), while
  * the row counts, phase columns, cascade/regime stamping, timeline integrity,
    slot reuse, migration continuity, onset counts (mapped and unmapped),
    cross-hit confound counts, gate metrics and the familiar per-task maxima are
    all recomputed here and compared against the stream's own files.

Checks printed per stream
-------------------------
    row layout        every array carries ``manifest['steps'] + 1`` rows and
                      ``manifest['steps']`` is the registered phase-length sum
    phase structure   the phase id / kind / probability / regime columns are the
                      registered timeline, switch points included
    regime isolation  every cascade task carries exactly the regime of the phase
                      it was created in, and no cascade task is born familiar
    task integrity    core23.timeline_integrity / migration_continuity on the
                      rebuilt timeline, slot reuse and reuse counts
    onset counts      per-regime scans recomputed WITH the task regime map and
                      compared to the audit file; the unmapped scan is reported
                      next to it as the cross-hit confound count
    admissibility     the familiar per-task maximum of each regime's onset
                      resource is MEASURED on this stream's familiar-phase rows
                      and reported next to the registered threshold and floor; a
                      threshold that is not strictly above the measured maximum
                      is reported as ``threshold_admissible: false`` and is NOT
                      nudged here
    determinism       dtype/shape stability and zero NaN/inf in the model-facing
                      arrays (``host_features``, ``demands``, ``schedules``,
                      ``raw_labels``, ``capacities``)

Exit codes: 0 = every check passed, 1 = at least one check failed, 2 = the
verifier could not complete (missing/corrupt input).  ``threshold_admissible``
and the cross-hit confound count are REPORTED booleans/metrics, not exit-code
gates: the S2 decision about a provisional threshold belongs to the reviewer,
and the verifier's job is to make the number impossible to miss.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ftmoe_protocol023_core as core23                      # noqa: E402
from prepare_ftmoe_protocol023_stream import (               # noqa: E402
    FAMILIAR_TASK_ID, REGISTERED_MECHANISM_IDS, REGISTERED_REGIME_IDS,
    cascade_fault_runs, evaluate_gate, phase_at, registered_phases,
    registered_steps, registered_switch_points, run_segmentation, stream_tag)

PROTOCOL = "023"
REQUIRED_STREAM_KEYS = (
    "host_features", "demands", "schedules", "raw_labels", "capacities",
    "post_totals", "overload_ratio", "overload_mask", "before_placement",
    "after_placement", "creation_ids", "after_creation_ids",
    "simulator_intervals", "phase_ids", "phase_kind",
    "phase_cascade_probability", "phase_regime_ids", "cascade_task_flags",
    "cascade_event_ids", "cascade_phases", "cascade_regimes",
    "host_cascade_any", "host_cascade_event", "host_cascade_regime",
    "host_cascade_mask", "deploy_attempts", "deploy_rejected",
    "migrate_attempts", "migrate_rejected")
REQUIRED_TIMELINE_KEYS = ("time", "slot_index", "creation_id", "demand",
                          "host_id", "cascade_event_id", "cascade_phase",
                          "cascade_regime")
#: dtype pinning (the model-facing arrays plus the audit columns).
EXPECTED_DTYPES = {"host_features": "float32", "demands": "float32",
                   "schedules": "float32", "raw_labels": "int64",
                   "capacities": "float64", "post_totals": "float64",
                   "overload_ratio": "float64", "overload_mask": "uint8",
                   "before_placement": "int64", "after_placement": "int64",
                   "creation_ids": "int64", "after_creation_ids": "int64",
                   "simulator_intervals": "int64", "phase_ids": "int64",
                   "phase_kind": "int64", "phase_cascade_probability": "float64",
                   "phase_regime_ids": "int64", "cascade_task_flags": "int64",
                   "cascade_event_ids": "int64", "cascade_phases": "int64",
                   "cascade_regimes": "int64", "host_cascade_any": "int64",
                   "host_cascade_event": "int64", "host_cascade_regime": "int64",
                   "host_cascade_mask": "int64", "deploy_attempts": "int64",
                   "deploy_rejected": "int64", "migrate_attempts": "int64",
                   "migrate_rejected": "int64"}
MODEL_FACING = ("host_features", "demands", "schedules", "raw_labels",
                "capacities")
GATE_METRIC_KEYS = ("window_intervals", "scored_intervals", "host_steps",
                    "positive_hoststeps", "prevalence", "normal_hoststeps",
                    "cascade_related_positive_hoststeps",
                    "independent_fault_events", "independent_cascade_fault_events",
                    "deployment_attempts", "deployment_rejected",
                    "migration_attempts", "migration_rejected",
                    "worst_event_share_of_positives")
AUDIT_TOP_KEYS = ("protocol", "mode", "regime", "smoke", "thresholds", "checks",
                  "passed", "metrics", "per_phase", "per_regime", "task_level",
                  "cross_regime_confound", "admissibility", "definitions",
                  "candidate")
MANIFEST_TOP_KEYS = ("schema_version", "protocol", "phase", "kind", "registered",
                     "name", "mode", "regime", "smoke", "regime_id",
                     "regime_family", "registered_regimes", "mechanism", "seed",
                     "steps", "scored_intervals", "phases",
                     "phase_switch_points", "audit_only_columns",
                     "forbidden_model_inputs", "audit_only_note", "cohort",
                     "onset_definition", "registered_onset_tau",
                     "stream_sha256", "task_timeline_sha256",
                     "task_timeline_content_sha256", "task_level_summary",
                     "per_regime_gate", "cross_regime_confound",
                     "admissibility", "ram_guard", "process_guard",
                     "source_sha256")


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def npz_keys(path):
    with np.load(path) as data:
        return sorted(data.files)


def finiteness(array):
    array = np.asarray(array)
    if array.dtype.kind in ("i", "u", "b"):
        return {"n_non_finite": 0, "first_index": None}
    bad = ~np.isfinite(array)
    count = int(bad.sum())
    return {"n_non_finite": count,
            "first_index": (int(np.argmax(bad.ravel())) if count else None)}


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


#: The audit measures the familiar per-task extremes from its own in-process
#: arrays while this verifier re-reads them from the float32 ``task_timeline``
#: file, so the two can differ in the last float32 digit.  Round 1 never hit it
#: (its familiar maxima were round numbers); round 2A's calibrated stream did
#: (5424.630666666667 vs 5424.630859375, a relative difference of 3.6e-8), which
#: is a storage-precision artefact and not a disagreement about the data.
FAMILIAR_MAX_TOLERANCE = 1e-3


def compare_metrics(declared, recomputed, keys=GATE_METRIC_KEYS):
    """Per-key disagreement list between a declared and a recomputed dict."""
    out = {}
    for key in keys:
        left = (declared or {}).get(key)
        right = (recomputed or {}).get(key)
        if isinstance(left, float) or isinstance(right, float):
            out[key] = {"declared": left, "recomputed": right,
                        "match": close(left, right)}
        else:
            out[key] = {"declared": left, "recomputed": right,
                        "match": left == right}
    out["all_match"] = all(v["match"] for v in out.values()
                           if isinstance(v, dict))
    return out


def regime_masks(phases, steps):
    out = {}
    for phase in phases:
        if phase["regime_id"] is None:
            continue
        mask = out.setdefault(phase["regime_id"], np.zeros(steps, dtype=bool))
        mask[phase["start"]:min(phase["end"], steps)] = True
    return out


def task_groups(timeline_rows):
    """``{creation_id: {"first_t", "regimes", "cascade"}}`` from timeline rows."""
    groups = {}
    creation = timeline_rows["creation_id"]
    times = timeline_rows["time"]
    events = timeline_rows["cascade_event_id"]
    regimes = timeline_rows["cascade_regime"]
    for row in range(creation.size):
        cid = int(creation[row])
        entry = groups.get(cid)
        if entry is None:
            entry = groups[cid] = {"first_t": int(times[row]),
                                   "regimes": set(), "n_rows": 0,
                                   "n_cascade_rows": 0}
        entry["first_t"] = min(entry["first_t"], int(times[row]))
        entry["n_rows"] += 1
        if int(events[row]) >= 0:
            entry["n_cascade_rows"] += 1
            entry["regimes"].add(int(regimes[row]))
    for entry in groups.values():
        entry["cascade"] = entry["n_cascade_rows"] > 0
        entry["regimes"] = sorted(entry["regimes"])
    return groups


def apply_declared_calibration(manifest):
    """Read a stream's own declared calibration and check it against the law.

    A calibrated stream (round 2A, directive §4-§6) is generated with amended
    B/C parameters, and the collector records the amendment in the manifest.
    Verifying such a stream against the *registered* tables would either fail on
    the amendment or, worse, pass by re-checking the wrong numbers.

    This function does NOT patch the module-level registry: mutating it would
    also move the phase table this verifier re-derives, which would make the
    comparison circular.  Instead it returns the probability the stream's own
    manifest declares and the amendment report, and the phase checks compare
    against that declared value.

    The amendment is still validated against the registered law:

    * ``compute_first`` may never appear (it must stay byte-identical to
      Protocol 022's ``cascade_v2``);
    * the registered cascade order, onset resource and response windows may
      never be touched.
    """
    amendment = (manifest.get("round2a_amendment")
                 or manifest.get("phase_amendment"))
    declared = manifest.get("cascade_task_probability")
    if not amendment:
        return (None if declared is None else float(declared)), {
            "calibrated": False,
            "declared_cascade_task_probability": declared,
            "source": "registered tables (no amendment declared)"}
    diff = amendment.get("parameter_diff") or {}
    if "compute_first" in diff:
        raise ValueError("the stream's amendment touches compute_first, which "
                         "is frozen; refusing to verify it")
    forbidden = ("sequence", "onset_resource", "response_windows", "family",
                 "regime_id", "mechanism_id")
    for regime_id, overrides in sorted(diff.items()):
        for key in sorted(overrides):
            if key in forbidden:
                raise ValueError(
                    "the stream's amendment changes %s.%s, which the "
                    "directive forbids re-tuning" % (regime_id, key))
    if declared is None:
        raise ValueError("a calibrated stream must declare "
                         "cascade_task_probability at the manifest top level")
    from simulator.workload import BitbrainWorkloadProtocol023 as generator
    for regime_id in sorted(diff):
        if regime_id not in generator.REGIMES_V3:
            raise ValueError("the stream's amendment names an unregistered "
                             "regime %r" % regime_id)
        for key in sorted(diff[regime_id]):
            if key not in generator.REGIMES_V3[regime_id]:
                raise ValueError("the stream's amendment changes an unknown "
                                 "key %s.%s" % (regime_id, key))
    return float(declared), {
        "calibrated": True,
        "source": "the stream's own round2a amendment",
        "declared_cascade_task_probability": float(declared),
        "parameter_diff": diff,
        "regimes_touched": sorted(diff),
        "generator_source_sha256": (amendment.get("generator_version") or {})
        .get("source_sha256"),
        "source_parent_hash": amendment.get("source_parent_hash"),
        "regimes_frozen": amendment.get("regimes_frozen"),
        "note": ("the phase table this verifier re-derives is the REGISTERED "
                 "timeline; only the per-phase probability is compared against "
                 "the value the stream declares, which is what the collector "
                 "actually applied"),
    }


def pristine_familiar_admissibility(stream, manifest, timeline_path):
    """Per-task extremes of the mechanism-off familiar window (phase F0).

    Round 1 measured the admissibility argument on the whole familiar phase,
    which for a calibrated stream is no longer envelope-free.  This measurement
    uses only the interval range of the first phase, whose cascade probability is
    zero by construction (the stream is generated with the mechanism switched
    off there), so the number answers the question the admissibility argument
    actually asks: what can a genuinely familiar task demand?
    """
    phases = manifest.get("phases") or []
    off = [phase for phase in phases
           if phase.get("regime_id") is None
           and float(phase.get("cascade_task_probability") or 0.0) == 0.0]
    if not off:
        return {"measured": False, "reason": "no mechanism-off phase declared"}
    first = off[0]
    start, end = int(first["start"]), int(first["end"])
    with np.load(timeline_path) as data:
        saved = {name: data[name] for name in data.files}
    inside = (np.asarray(saved["time"]) >= start) & (np.asarray(saved["time"]) < end)
    if not inside.any():
        return {"measured": False,
                "reason": "phase %s has no rows" % first.get("name")}
    demand = np.asarray(saved["demand"], dtype=np.float64)[inside]
    task = np.asarray(saved["creation_id"], dtype=np.int64)[inside]
    envelope = np.asarray(saved["cascade_event_id"], dtype=np.int64)[inside]
    per_task = {}
    for index, cid in enumerate(task):
        key = int(cid)
        record = per_task.setdefault(key, {"cpu": 0.0, "ram": 0.0, "disk": 0.0,
                                           "enveloped": False})
        for column, name in ((0, "cpu"), (1, "ram"), (4, "disk")):
            record[name] = max(record[name], float(demand[index, column]))
        record["enveloped"] = record["enveloped"] or bool(envelope[index] >= 0)
    by_resource = {}
    for column, name in ((0, "cpu"), (1, "ram"), (4, "disk")):
        values = [record[name] for record in per_task.values()]
        clean = [record[name] for record in per_task.values()
                 if not record["enveloped"]]
        by_resource[name] = {
            "max_all_f0_tasks": max(values) if values else None,
            "max_envelope_free_f0_tasks": max(clean) if clean else None,
            "n_tasks": len(values),
            "n_envelope_free": len(clean),
        }
    return {"measured": True, "phase": first.get("name"),
            "interval_range": [start, end],
            "per_resource": by_resource,
            "registered_familiar_clip": {
                name: float(core23.FAMILIAR_CLIP[name])
                for name in ("cpu", "ram", "disk")},
            "registered_onset_threshold": {
                name: float(spec23["onset_tau"]) for name, spec23 in
                ((rid, core23.regime(rid)) for rid in ("A", "B", "C"))},
            "note": ("F0 is generated with the mechanism off, so its "
                     "envelope-free per-task maxima are the round-1 comparison "
                     "for the calibration's effect on the familiar window")}


def verify(directory, tag_suffix=""):
    """Verify one collected stream.

    ``tag_suffix`` is the registered suffix of a calibrated stream (round 2A
    writes ``_calibrated``).  It is an explicit argument rather than a relaxed
    comparison: the expected directory name is
    ``stream_tag(...) + tag_suffix``, so a stream whose directory is wrong by
    anything other than that exact registered suffix still fails the check.
    """
    directory = Path(directory).resolve()
    checks, reported, failures = {}, {}, []

    def check(name, ok, detail=None):
        checks[name] = bool(ok)
        if detail is not None:
            reported[name] = detail
        if not ok:
            failures.append(name)
        return bool(ok)

    verdict = {
        "protocol": PROTOCOL, "stage": "S2", "stream_dir": str(directory),
        "tag": directory.name,
        "registry_source": ("prepare_ftmoe_protocol023_stream (registered phase "
                            "table, gate arithmetic, switch points) + "
                            "ftmoe_protocol023_core (instrument)"),
        "checks": checks, "reported": reported, "failed_checks": failures,
    }

    stream_path = directory / "stream.npz"
    timeline_path = directory / "task_timeline.npz"
    missing = [str(p.name) for p in (stream_path, timeline_path,
                                     directory / "manifest.json",
                                     directory / "unseen_data_audit.json",
                                     directory / "task_event_index.json",
                                     directory / "events.json",
                                     directory / "generation.log")
               if not p.exists()]
    if not check("all_seven_outputs_present", not missing, {"missing": missing}):
        verdict.update({"passed": False, "exit_code": 2, "error":
                        "missing output file(s): %s" % missing})
        return verdict
    if (directory / "failure.json").exists():
        check("no_failure_json", False,
              {"failure": load_json(directory / "failure.json")})
        verdict.update({"passed": False, "exit_code": 1})
        return verdict
    check("no_failure_json", True)

    manifest = load_json(directory / "manifest.json")
    audit = load_json(directory / "unseen_data_audit.json")
    event_index = load_json(directory / "task_event_index.json")
    events_json = load_json(directory / "events.json")
    # A calibrated stream declares its own amended B/C parameters; read them
    # before any registered-number check derby the registry.
    declared_probability, calibration = apply_declared_calibration(manifest)
    verdict["calibration"] = calibration
    check("declared_calibration_is_registered_law_compatible",
          True, calibration)
    with np.load(stream_path) as data:
        stream = {name: data[name] for name in data.files}
    with np.load(timeline_path) as data:
        saved_timeline = {name: data[name] for name in data.files}

    # ---------------------------------------------------------------- registry
    mode = manifest.get("mode")
    regime = manifest.get("regime")
    smoke = bool(manifest.get("smoke"))
    verdict.update({"mode": mode, "regime": regime, "smoke": smoke})
    check("registered_mode_declared", mode in ("dev", "single"),
          {"mode": mode})
    check("single_mode_declares_one_registered_regime",
          (regime is None) if mode == "dev"
          else (regime in REGISTERED_REGIME_IDS), {"regime": regime})
    phases = registered_phases(mode, regime, smoke=smoke)
    if declared_probability is not None:
        # The phase table the checks compare against carries the probability the
        # stream's own amendment declares, per phase.  Nothing else about the
        # registered timeline moves.
        phases = [dict(phase) for phase in phases]
        for phase in phases:
            override = ((calibration.get("parameter_diff") or {})
                        .get(phase.get("regime_id")) or {})
            if "cascade_task_probability" in override:
                phase["cascade_task_probability"] = float(
                    override["cascade_task_probability"])
            elif phase.get("regime_id") is not None:
                phase["cascade_task_probability"] = float(declared_probability)
    steps = registered_steps(mode, regime, smoke=smoke)
    verdict["steps"] = steps
    verdict["registered_phases"] = phases
    check("manifest_steps_is_the_registered_phase_sum",
          int(manifest.get("steps", -1)) == steps,
          {"manifest_steps": manifest.get("steps"), "registered_steps": steps})
    check("manifest_scored_intervals_is_registered",
          int(manifest.get("scored_intervals", -1)) == steps)
    if not smoke:
        expected_tag = stream_tag(mode, regime, steps) + str(tag_suffix or "")
        check("output_directory_is_the_registered_tag",
              directory.name == expected_tag,
              {"directory": directory.name, "registered_tag": expected_tag,
               "tag_suffix": str(tag_suffix or "")})
    else:
        check("smoke_horizon_is_the_registered_engineering_slice",
              steps == 120 and directory.name == stream_tag(mode, regime, steps),
              {"steps": steps, "directory": directory.name})
    missing_keys = [key for key in MANIFEST_TOP_KEYS if key not in manifest]
    check("manifest_top_level_keys_present", not missing_keys,
          {"missing": missing_keys})
    missing_keys = [key for key in AUDIT_TOP_KEYS if key not in audit]
    check("audit_top_level_keys_present", not missing_keys,
          {"missing": missing_keys})
    check("manifest_declares_audit_only_columns",
          bool(manifest.get("audit_only_columns"))
          and bool(manifest.get("audit_only_note"))
          and "phase_ids" in (manifest.get("audit_only_columns") or {}))
    forbidden = list(manifest.get("forbidden_model_inputs") or [])
    check("forbidden_input_tokens_registered",
          all(token in forbidden for token in ("regime_id", "phase_id",
                                               "mechanism_id", "event_id")),
          {"forbidden_model_inputs": forbidden})

    # ------------------------------------------------------------- row layout
    count = int(manifest.get("steps", -1)) + 1
    layout = {}
    for name in sorted(stream):
        layout[name] = {"rows": int(stream[name].shape[0]),
                        "expected_rows": count,
                        "ok": int(stream[name].shape[0]) == count}
    check("every_stream_array_carries_steps_plus_one_rows",
          all(info["ok"] for info in layout.values()),
          {"expected_rows": count,
           "offenders": {k: v for k, v in layout.items() if not v["ok"]}})
    missing_keys = [key for key in REQUIRED_STREAM_KEYS if key not in stream]
    extra_keys = [key for key in stream if key not in REQUIRED_STREAM_KEYS]
    check("stream_key_set_matches_schema", not missing_keys,
          {"missing": missing_keys, "extra": extra_keys})
    timeline_lengths = {name: int(saved_timeline[name].shape[0])
                        for name in sorted(saved_timeline)}
    check("task_timeline_columns_are_parallel",
          len(set(timeline_lengths.values())) == 1,
          {"rows": timeline_lengths})
    missing_keys = [key for key in REQUIRED_TIMELINE_KEYS
                    if key not in saved_timeline]
    check("task_timeline_key_set_matches_schema", not missing_keys,
          {"missing": missing_keys,
           "extra": [k for k in saved_timeline
                     if k not in REQUIRED_TIMELINE_KEYS]})
    dtype_report = {}
    for name in REQUIRED_STREAM_KEYS:
        if name not in stream:
            continue
        actual = str(stream[name].dtype)
        dtype_report[name] = {"dtype": actual,
                              "expected": EXPECTED_DTYPES.get(name),
                              "ok": actual == EXPECTED_DTYPES.get(name)}
    check("stream_dtypes_are_stable",
          all(info["ok"] for info in dtype_report.values()),
          {"offenders": {k: v for k, v in dtype_report.items() if not v["ok"]}})
    shape_report = {
        "host_features": {"shape": list(stream["host_features"].shape),
                          "expected": [count, 16, 7]},
        "demands": {"shape": list(stream["demands"].shape),
                    "expected": [count, 16, 7]},
        "schedules": {"shape": list(stream["schedules"].shape),
                      "expected": [count, 16, 16]},
        "raw_labels": {"shape": list(stream["raw_labels"].shape),
                       "expected": [count, 16]},
        "capacities": {"shape": list(stream["capacities"].shape),
                       "expected": [count, 16, 3]},
    }
    for info in shape_report.values():
        info["ok"] = info["shape"] == info["expected"]
    check("model_facing_array_shapes_are_stable",
          all(info["ok"] for info in shape_report.values()),
          {"shapes": shape_report})

    # --------------------------------------------------------- phase structure
    expected_phase_ids = np.zeros(count, dtype=np.int64)
    expected_kind = np.zeros(count, dtype=np.int64)
    expected_probability = np.zeros(count, dtype=np.float64)
    expected_regime = np.full(count, -1, dtype=np.int64)
    for t in range(count):
        index, phase = phase_at(t, phases)
        expected_phase_ids[t] = index
        expected_kind[t] = 0 if phase["kind"] == "familiar" else 1
        expected_probability[t] = phase["cascade_task_probability"]
        expected_regime[t] = (-1 if phase["regime_id"] is None
                              else int(REGISTERED_MECHANISM_IDS[phase["regime_id"]]))
    check("phase_id_column_matches_registered_timeline",
          np.array_equal(stream["phase_ids"], expected_phase_ids),
          {"n_mismatch": int((stream["phase_ids"] != expected_phase_ids).sum())})
    check("phase_kind_column_matches_registered_timeline",
          np.array_equal(stream["phase_kind"], expected_kind),
          {"n_mismatch": int((stream["phase_kind"] != expected_kind).sum())})
    check("phase_probability_column_matches_registry",
          np.allclose(stream["phase_cascade_probability"],
                      expected_probability, atol=0.0, rtol=0.0),
          {"n_mismatch": int(np.sum(stream["phase_cascade_probability"]
                                    != expected_probability))})
    check("phase_regime_column_matches_registry",
          np.array_equal(stream["phase_regime_ids"], expected_regime),
          {"n_mismatch": int((stream["phase_regime_ids"] != expected_regime).sum())})
    check("familiar_phases_are_probability_zero",
          bool(np.all(stream["phase_cascade_probability"][
              stream["phase_kind"] == 0] == 0.0)),
          {"n_familiar_intervals": int((stream["phase_kind"] == 0).sum())})
    check("regime_phases_carry_the_registered_probability",
          bool(np.all(stream["phase_cascade_probability"][
              stream["phase_kind"] == 1]
              == np.asarray(expected_probability)[
                  stream["phase_kind"] == 1])),
          {"n_regime_intervals": int((stream["phase_kind"] == 1).sum()),
           "expected_probability": sorted(set(
               float(value) for value, kind in
               zip(expected_probability, stream["phase_kind"]) if kind == 1)),
           "declared_cascade_task_probability": declared_probability})
    check("manifest_phase_table_matches_registry",
          manifest.get("phases") == phases,
          {"manifest_phases": manifest.get("phases")})
    registered_switches = registered_switch_points(phases)
    applied_switches = []
    for t in range(1, count):
        if (stream["phase_ids"][t] != stream["phase_ids"][t - 1]
                or stream["phase_cascade_probability"][t]
                != stream["phase_cascade_probability"][t - 1]
                or stream["phase_regime_ids"][t] != stream["phase_regime_ids"][t - 1]):
            index, phase = phase_at(t, phases)
            applied_switches.append(
                {"interval": int(t), "phase": phase["name"],
                 "regime_id": phase["regime_id"],
                 "probability": float(stream["phase_cascade_probability"][t])})
    check("switch_points_are_where_the_registry_says",
          applied_switches == registered_switches,
          {"applied": applied_switches, "registered": registered_switches})
    check("manifest_switch_points_match_registry",
          manifest.get("phase_switch_points") == registered_switches)
    check("manifest_observed_switches_match_registry",
          manifest.get("observed_phase_switches") == registered_switches)

    # ------------------------------------------------- cascade column coherence
    cascade = stream["cascade_event_ids"]
    cascade_regimes = stream["cascade_regimes"]
    flags = stream["cascade_task_flags"]
    row_cascade = cascade >= 0
    check("cascade_task_flag_matches_the_event_column",
          bool(np.array_equal(flags == 1, row_cascade)),
          {"n_cascade_rows": int(row_cascade.sum())})
    check("cascade_regime_column_is_stamped_exactly_where_an_envelope_is",
          bool(np.array_equal(cascade_regimes >= 0, row_cascade))
          and bool(np.all(np.isin(cascade_regimes[row_cascade], [0, 1, 2]))),
          {"mechanism_ids": sorted(set(cascade_regimes[row_cascade].tolist()))})
    host_event = stream["host_cascade_event"]
    host_regime = stream["host_cascade_regime"]
    host_any = stream["host_cascade_any"]
    host_mask = stream["host_cascade_mask"]
    check("host_cascade_regime_and_mask_track_host_cascade_event",
          bool(np.array_equal(host_event >= 0, host_regime >= 0))
          and bool(np.array_equal(host_event >= 0, host_any == 1))
          and bool(np.all(host_mask[host_event < 0] == 0))
          and bool(np.all(host_mask[host_event >= 0] > 0)),
          {"n_host_cascade_cells": int((host_event >= 0).sum())})

    # --------------------------------------------------------- task integrity
    timeline = core23.build_task_timeline(
        saved_timeline["time"], saved_timeline["slot_index"],
        saved_timeline["creation_id"], saved_timeline["demand"],
        saved_timeline["host_id"], saved_timeline["cascade_event_id"],
        saved_timeline["cascade_phase"])
    integrity = core23.timeline_integrity(timeline)
    continuity = core23.migration_continuity(timeline)
    # A live audit cell is one whose container SURVIVED the interval
    # (``after_creation_ids >= 0``).  ``creation_ids != -1`` is NOT equivalent:
    # a task created and then dropped inside the same interval (unplaceable --
    # ``Simulator.simulationStep`` removes it from ``containerlist``) appears in
    # the pre-step snapshot and not in the post-step one, so it contributes
    # demand but no observation row.  Both the collector's timeline and this
    # re-derivation therefore use the post-step identity.
    live = stream["after_creation_ids"][:steps] >= 0
    created = stream["creation_ids"][:steps] >= 0
    admission_dropped = created & ~live
    live_t, live_s = np.nonzero(live)
    row_set_ok = (int(live_t.size) == int(saved_timeline["time"].size)
                  and np.array_equal(saved_timeline["time"], live_t)
                  and np.array_equal(saved_timeline["slot_index"], live_s))
    check("task_timeline_rows_are_the_surviving_live_cells", row_set_ok,
          {"surviving_live_cells": int(live_t.size),
           "timeline_rows": int(saved_timeline["time"].size),
           "created_cells": int(created.sum()),
           "admission_dropped_cells": int(admission_dropped.sum()),
           "admission_dropped_tasks": int(len(set(
               stream["creation_ids"][:steps][admission_dropped].tolist())))})
    column_ok = row_set_ok and all([
        np.array_equal(saved_timeline["creation_id"],
                       stream["creation_ids"][live_t, live_s]),
        np.array_equal(saved_timeline["creation_id"],
                       stream["after_creation_ids"][live_t, live_s]),
        np.array_equal(saved_timeline["host_id"],
                       stream["after_placement"][live_t, live_s]),
        np.array_equal(saved_timeline["cascade_event_id"],
                       stream["cascade_event_ids"][live_t, live_s]),
        np.array_equal(saved_timeline["cascade_phase"],
                       stream["cascade_phases"][live_t, live_s]),
        np.array_equal(saved_timeline["cascade_regime"],
                       stream["cascade_regimes"][live_t, live_s])])
    check("task_timeline_columns_are_the_stream_columns", column_ok)
    reported["admission_dropped_note"] = (
        "%d cell(s) hold a task that was created and dropped inside the same "
        "interval (unplaceable): they carry demand but no audit observation, "
        "exactly as in the P22 collector" % int(admission_dropped.sum()))
    reported["demand_column_note"] = (
        "task_timeline.demand is the audit's own per-observation demand read "
        "after simulationStep; stream.demands is read before it, so the two "
        "7-column matrices are NOT the same object by construction (the "
        "container's CPU/RAM/disk are read at two different points of the "
        "interval).  The audit's column is finite and positive, which is what "
        "the instrument requires.")
    demand_finite = finiteness(saved_timeline["demand"])
    check("task_timeline_demand_is_finite",
          demand_finite["n_non_finite"] == 0, demand_finite)
    check("timeline_integrity_ok", integrity["ok"],
          {"problems": integrity["problems"][:10],
           "n_problems": len(integrity["problems"])})
    declared_task = (audit.get("task_level") or {})
    check("timeline_integrity_matches_audit",
          int(declared_task.get("n_tasks", -1)) == int(integrity["n_tasks"])
          and int(declared_task.get("n_observations", -1))
          == int(integrity["n_observations"])
          and int(declared_task.get("reused_slots", -1))
          == int(integrity["reused_slots"]),
          {"audit": {k: declared_task.get(k) for k in
                     ("n_tasks", "n_observations", "reused_slots")},
           "recomputed": {k: integrity[k] for k in
                          ("n_tasks", "n_observations", "reused_slots")}})
    check("migration_continuity_matches_audit",
          int(declared_task.get("tasks_with_migration", -1))
          == int(continuity["n_tasks_with_migration"]),
          {"audit": declared_task.get("tasks_with_migration"),
           "recomputed": continuity["n_tasks_with_migration"]})
    check("timeline_content_sha256_matches_recomputed_timeline",
          manifest.get("task_timeline_content_sha256")
          == core23.timeline_sha256(timeline))
    check("task_timeline_file_sha256_matches_manifest",
          manifest.get("task_timeline_sha256") == sha(timeline_path))
    check("stream_file_sha256_matches_manifest",
          manifest.get("stream_sha256") == sha(stream_path))
    reported["task_integrity"] = {
        "n_tasks": int(integrity["n_tasks"]),
        "n_observations": int(integrity["n_observations"]),
        "reused_slots": int(integrity["reused_slots"]),
        "reused_slot_examples": integrity["reused_slot_examples"],
        "multi_slot_tasks": int(integrity["multi_slot_tasks"]),
        "tasks_with_migration": int(continuity["n_tasks_with_migration"]),
        "n_tasks": int(continuity["n_tasks"]),
    }

    # ------------------------------------------------------- task regime map
    groups = task_groups(saved_timeline)
    mechanism_to_gen = {int(value): name for name, value
                        in REGISTERED_MECHANISM_IDS.items()}
    mechanism_to_canonical = {int(value): core23.canonical_regime_id(name)
                              for name, value in REGISTERED_MECHANISM_IDS.items()}
    task_regime_map, ambiguous = {}, []
    for cid, entry in groups.items():
        if not entry["cascade"]:
            task_regime_map[cid] = FAMILIAR_TASK_ID
            continue
        if len(entry["regimes"]) != 1:
            ambiguous.append({"creation_id": cid, "mechanism_ids": entry["regimes"]})
        task_regime_map[cid] = mechanism_to_canonical.get(
            entry["regimes"][0], "unregistered:%d" % entry["regimes"][0])
    check("every_task_carries_one_mechanism_id", not ambiguous,
          {"ambiguous": ambiguous[:5]})
    declared_map = {int(k): v for k, v in
                    (declared_task.get("task_regime_map") or {}).items()}
    check("recomputed_task_regime_map_matches_audit",
          declared_map == task_regime_map,
          {"n_declared": len(declared_map), "n_recomputed": len(task_regime_map),
           "n_differing": len([c for c in set(declared_map) | set(task_regime_map)
                               if declared_map.get(c) != task_regime_map.get(c)])})
    tasks_per_regime = {FAMILIAR_TASK_ID: 0}
    for value in task_regime_map.values():
        tasks_per_regime[value] = tasks_per_regime.get(value, 0) + 1
    declared_summary = declared_task.get("task_regime_map_summary") or {}
    check("task_regime_map_covers_every_timeline_task",
          len(task_regime_map) == int(integrity["n_tasks"])
          and close(declared_summary.get("coverage"), 1.0),
          {"n_mapped": len(task_regime_map), "n_tasks": int(integrity["n_tasks"]),
           "declared_coverage": declared_summary.get("coverage"),
           "tasks_per_regime": tasks_per_regime,
           "declared_tasks_per_regime": declared_summary.get("tasks_per_regime")})
    check("audit_reports_the_task_map_coverage_and_cohort_sizes",
          "coverage" in declared_summary
          and "tasks_per_regime" in declared_summary)
    reported["tasks_per_regime"] = tasks_per_regime

    # --------------------------------------------------------- regime isolation
    familiar_phase_intervals = set(int(t) for t in
                                   np.nonzero(stream["phase_kind"][:steps] == 0)[0])
    born_familiar, mismatched, per_phase_counts = [], [], {}
    for cid, entry in groups.items():
        index, phase = phase_at(entry["first_t"], phases)
        key = phase["name"]
        if entry["cascade"]:
            per_phase_counts[key] = per_phase_counts.get(key, 0) + 1
            if phase["regime_id"] is None:
                born_familiar.append({"creation_id": cid,
                                      "first_t": entry["first_t"],
                                      "phase": phase["name"]})
            elif entry["regimes"] != [int(REGISTERED_MECHANISM_IDS[phase["regime_id"]])]:
                mismatched.append({"creation_id": cid, "first_t": entry["first_t"],
                                   "phase": phase["name"],
                                   "phase_regime": phase["regime_id"],
                                   "task_mechanism_ids": entry["regimes"]})
    check("no_cascade_task_is_born_in_a_familiar_phase", not born_familiar,
          {"n_born_familiar": len(born_familiar), "examples": born_familiar[:5],
           "familiar_intervals": len(familiar_phase_intervals)})
    check("cascade_tasks_carry_the_regime_of_their_birth_phase", not mismatched,
          {"n_mismatched": len(mismatched), "examples": mismatched[:5]})
    check("every_regime_phase_produced_cascade_tasks",
          smoke or all(per_phase_counts.get(phase["name"], 0) > 0
                       for phase in phases if phase["regime_id"] is not None),
          {"cascade_tasks_per_phase": per_phase_counts})
    if mode == "single":
        single_canonical = core23.canonical_regime_id(regime)
        foreign = sorted(set(value for value in task_regime_map.values()
                             if value != FAMILIAR_TASK_ID
                             and value != single_canonical))
        check("single_regime_stream_holds_only_that_regime",
              not foreign,
              {"registered_regime": regime, "canonical": single_canonical,
               "foreign_regime_ids": foreign})
    else:
        present = sorted(value for value in set(task_regime_map.values())
                         if value != FAMILIAR_TASK_ID)
        check("dev_stream_holds_all_three_registered_regimes",
              present == sorted(core23.REGIME_IDS), {"present": present})
    check("declared_tasks_per_regime_matches_recomputed",
          dict(declared_summary.get("tasks_per_regime") or {}) == tasks_per_regime,
          {"declared": declared_summary.get("tasks_per_regime"),
           "recomputed": tasks_per_regime})

    # ------------------------------------------------------------ onset counts
    # The regimes with a registered phase in THIS stream come from the registry,
    # never from the audit; the cross-hit confound and the admissibility are
    # measured for every registered regime, because in a single-regime stream the
    # other regimes' unmapped scans are pure confound.
    registry_present = []
    for phase in phases:
        if phase["regime_id"] is not None and phase["regime_id"] not in registry_present:
            registry_present.append(phase["regime_id"])
    expected_regimes = sorted(core23.canonical_regime_id(name)
                              for name in registry_present)
    audit_regimes = sorted((audit.get("per_regime") or {}).keys())
    check("audit_reports_exactly_the_regimes_with_a_registered_phase",
          audit_regimes == expected_regimes,
          {"audit": audit_regimes, "expected": expected_regimes})
    declared_events = {}
    for event in event_index.get("temporal_events") or []:
        declared_events.setdefault(event["regime_id"], set()).add(
            (int(event["creation_id"]), int(event["t_onset"])))
    per_regime_onsets, cross_confound, admissibility = {}, {}, {}
    onset_ok, blind_ok, cross_ok, keys_ok = True, True, True, True
    declared_registration_ok = True
    declared_onsets = {}
    all_canonical = sorted(core23.REGIME_IDS)
    for canonical in all_canonical:
        present = canonical in expected_regimes
        block = (audit.get("per_regime") or {}).get(canonical) or {}
        gen_id = (block.get("regime_id")
                  or [name for name, value in REGISTERED_MECHANISM_IDS.items()
                      if core23.canonical_regime_id(name) == canonical][0])
        spec = core23.regime(gen_id)
        resource = spec["onset_resource"]
        tau = float(spec["onset_tau"])
        windows = {name: tuple(int(x) for x in window)
                   for name, window in spec["response_windows"].items()}
        mapped = core23.onset_events_multi(
            timeline, resource, tau, response_windows=windows,
            regime_map={resource: canonical}, task_regime_map=task_regime_map)
        blind = core23.onset_events_multi(
            timeline, resource, tau, response_windows=windows,
            regime_map={resource: canonical})
        keys = set((int(e["creation_id"]), int(e["t_onset"])) for e in mapped)
        declared = (block.get("task_level") or {})
        declared_onsets[canonical] = declared.get("n_onsets") if present else None
        cross_confound[canonical] = {
            "regime_id": gen_id, "onset_resource": resource, "onset_tau": tau,
            "registered_phase_in_stream": bool(present),
            "onsets_without_task_map": len(blind),
            "onsets_with_task_map": len(mapped),
            "cross_hits": len(blind) - len(mapped),
            "note": ("onsets an unmapped resource+tau scan attributes to this "
                     "regime that its own task cohort does not produce; the "
                     "instrument's CrossRegimeConfoundTests pin that a "
                     "regime-A envelope crosses the B/C thresholds, so this "
                     "number is reported and is never a gate input"),
        }
        declared_cross = (audit.get("cross_regime_confound") or {}).get(canonical) or {}
        cross_ok = cross_ok and int(declared_cross.get("cross_hits", -1)) == \
            len(blind) - len(mapped)
        if not present:
            continue
        onset_ok = onset_ok and int(declared.get("n_onsets", -1)) == len(mapped)
        blind_ok = blind_ok and int(
            declared.get("n_onsets_without_task_map", -1)) == len(blind)
        keys_ok = keys_ok and declared_events.get(canonical, set()) == keys
        declared_registration_ok = declared_registration_ok and \
            close(block.get("onset_tau"), tau) and \
            block.get("onset_resource") == resource
        per_regime_onsets[canonical] = {
            "regime_id": gen_id, "onset_resource": resource, "onset_tau": tau,
            "n_onsets_with_task_map": len(mapped),
            "n_onsets_without_task_map": len(blind),
            "cross_hits": len(blind) - len(mapped),
            "n_onsets_with_cascade_window": int(sum(
                1 for e in mapped if e["audit_event_id"] >= 0)),
            "declared_n_onsets": declared.get("n_onsets"),
            "declared_n_onsets_with_cascade_window":
                declared.get("n_onsets_with_cascade_window"),
            "n_tasks_in_cohort": int(sum(
                1 for value in task_regime_map.values() if value == canonical)),
            "declared_n_tasks_in_cohort": declared.get("n_tasks_in_cohort"),
            "onset_provisional": bool(resource in core23.PROVISIONAL_TAUS),
        }
        # per-regime gate recomputed from the arrays
        regime_mask = regime_masks(phases, steps).get(gen_id)
        if regime_mask is None:
            regime_mask = np.zeros(steps, dtype=bool)
        regime_runs = run_segmentation(stream["raw_labels"], steps,
                                       mask=regime_mask)
        regime_cascades = cascade_fault_runs(regime_runs,
                                             stream["host_cascade_event"],
                                             regime_mask)
        gate = evaluate_gate(stream["raw_labels"], stream["host_cascade_any"],
                             regime_runs, regime_cascades,
                             stream["deploy_attempts"], stream["deploy_rejected"],
                             stream["migrate_attempts"],
                             stream["migrate_rejected"], regime_mask,
                             events_json.get("cascade_envelopes") or [])
        declared_gate = (block.get("gate") or {})
        comparison = compare_metrics(declared_gate.get("metrics"),
                                     gate["metrics"])
        per_regime_onsets[canonical]["gate_passed"] = gate["passed"]
        per_regime_onsets[canonical]["gate_checks"] = gate["checks"]
        per_regime_onsets[canonical]["gate_metrics_match_audit"] = \
            comparison["all_match"]
        per_regime_onsets[canonical]["gate_metrics"] = gate["metrics"]
        per_regime_onsets[canonical]["declared_gate_metrics"] = \
            declared_gate.get("metrics")
        per_regime_onsets[canonical]["gate_metric_comparison"] = comparison
        if not comparison["all_match"]:
            onset_ok = False

    # familiar-phase timeline, built once from the stream arrays: the MEASURED
    # calm side of every threshold argument, on this stream's familiar rows.
    familiar_cells = np.nonzero(live & (stream["phase_kind"][:steps] == 0)[:, None])
    familiar_timeline = core23.build_task_timeline(
        familiar_cells[0], familiar_cells[1],
        stream["creation_ids"][familiar_cells],
        stream["demands"][familiar_cells],
        stream["after_placement"][familiar_cells],
        stream["cascade_event_ids"][familiar_cells],
        stream["cascade_phases"][familiar_cells])
    floors = {}
    for cid, window in (event_index.get("cascade_windows") or {}).items():
        gen_id = window.get("regime_id")
        if gen_id is None:
            continue
        floors.setdefault(gen_id, []).append(float(window["onset_floor"]))
    declared_admissibility = audit.get("admissibility") or {}
    admissible_ok = True
    for canonical in all_canonical:
        block = (audit.get("per_regime") or {}).get(canonical) or {}
        gen_id = (block.get("regime_id")
                  or [name for name, value in REGISTERED_MECHANISM_IDS.items()
                      if core23.canonical_regime_id(name) == canonical][0])
        spec = core23.regime(gen_id)
        resource = spec["onset_resource"]
        tau = float(spec["onset_tau"])
        per_task = []
        for cid in sorted(familiar_timeline):
            values = np.asarray(familiar_timeline[cid][resource], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                per_task.append(float(values.max()))
        measured = max(per_task) if per_task else None
        declared = declared_admissibility.get(canonical) or {}
        floor = min(floors[gen_id]) if floors.get(gen_id) else \
            block.get("onset_floor") or declared.get("registered_floor")
        entry = {
            "regime_id": gen_id,
            "onset_resource": resource,
            "registered_phase_in_stream": bool(canonical in expected_regimes),
            "measured_familiar_task_maximum": measured,
            "n_familiar_tasks_measured": len(per_task),
            "registered_familiar_clip": float(core23.FAMILIAR_CLIP[resource]),
            "registered_onset_threshold": tau,
            "registered_floor": (None if floor is None else float(floor)),
            "threshold_admissible": bool(measured is not None and tau > measured),
            "margin_above_measured_maximum": (None if measured is None
                                              else float(tau - measured)),
            "margin_below_floor": (None if floor is None
                                   else float(floor - tau)),
            "provisional": bool(resource in core23.PROVISIONAL_TAUS),
            "declared_measured_familiar_task_maximum":
                declared.get("measured_familiar_task_maximum"),
            "declared_threshold_admissible": declared.get("threshold_admissible"),
            "measurement_source": ("per-task maximum of the onset resource over "
                                   "the familiar-phase rows of THIS stream"),
            "familiar_cells": int(familiar_cells[0].size),
        }
        entry["measurement_matches_audit"] = close(
            entry["measured_familiar_task_maximum"],
            entry["declared_measured_familiar_task_maximum"],
            tol=FAMILIAR_MAX_TOLERANCE)
        admissibility[canonical] = entry
        admissible_ok = admissible_ok and entry["measurement_matches_audit"]

    check("per_regime_onset_counts_match_audit", onset_ok,
          {"declared": declared_onsets,
           "recomputed": {k: v["n_onsets_with_task_map"]
                          for k, v in per_regime_onsets.items()}})
    check("unmapped_onset_counts_match_audit", blind_ok)
    check("cross_hit_confound_counts_match_audit", cross_ok,
          {"cross_hits": {k: v["cross_hits"] for k, v in cross_confound.items()}})
    check("onset_event_keys_match_task_event_index", keys_ok)
    check("audit_onset_registration_matches_instrument",
          declared_registration_ok)
    # A calibrated stream's NULL phases are no longer pristine: with a higher
    # cascade probability, many more regime tasks are still in flight when the
    # familiar phase starts, and a familiar-phase row then carries an envelope
    # (round 2A measured a familiar per-task CPU maximum of 5200 against 1860 in
    # round 1).  That is a real, reported consequence of the calibration: the
    # stream-level admissibility number becomes incomparable with round 1, so on
    # a calibrated stream the gating measurement is repeated on the F0 window
    # alone -- the phase generated with the mechanism switched off -- and the
    # polluted-window reading is recorded next to it.
    pristine = pristine_familiar_admissibility(stream, manifest, timeline_path)
    verdict["pristine_familiar_admissibility"] = pristine
    check("pristine_familiar_admissibility_measured",
          bool(pristine.get("measured")), pristine)
    if calibration.get("calibrated"):
        polluted = admissibility
        admissibility = {}
        for canonical, entry in polluted.items():
            resource = entry["onset_resource"]
            clean = pristine["per_resource"][resource]
            measured = clean["max_envelope_free_f0_tasks"]
            entry = dict(entry)
            entry["measured_familiar_task_maximum"] = measured
            entry["threshold_admissible"] = bool(measured is not None
                                                 and entry[
                                                     "registered_onset_threshold"]
                                                 > measured)
            entry["margin_above_measured_maximum"] = (
                None if measured is None
                else float(entry["registered_onset_threshold"] - measured))
            entry["measurement_source"] = (
                "per-task maximum of the onset resource over the "
                "MECHANISM-OFF F0 window of THIS stream (the calibrated "
                "stream's later familiar phase is visited by in-flight regime "
                "tasks and is reported as polluted_familiar_admissibility)")
            entry["polluted_window_reading"] = {
                "measured_familiar_task_maximum":
                    polluted[canonical]["measured_familiar_task_maximum"],
                "threshold_admissible":
                    polluted[canonical]["threshold_admissible"],
                "n_familiar_tasks_measured":
                    polluted[canonical]["n_familiar_tasks_measured"],
            }
            entry["measurement_matches_audit"] = (
                polluted[canonical]["measurement_matches_audit"])
            admissibility[canonical] = entry
        verdict["admissibility_measurement_note"] = (
            "the audit's own measurement scans the whole familiar phase, which "
            "on a calibrated stream is no longer envelope-free; the gating "
            "number here is re-measured on F0 and the polluted reading is kept "
            "next to it")
    check("admissibility_measurement_matches_audit",
          bool(admissible_ok),
          {"admissibility": admissibility})
    # per-phase gate metrics cross-check (the phase windows are the registry's)
    per_phase_ok, phase_report = True, {}
    for phase in phases:
        declared_phase = (audit.get("per_phase") or {}).get(phase["name"])
        if declared_phase is None:
            per_phase_ok = False
            phase_report[phase["name"]] = {"declared": None}
            continue
        mask = np.zeros(steps, dtype=bool)
        mask[phase["start"]:min(phase["end"], steps)] = True
        runs = run_segmentation(stream["raw_labels"], steps, mask=mask)
        cascades = cascade_fault_runs(runs, stream["host_cascade_event"], mask)
        gate = evaluate_gate(stream["raw_labels"], stream["host_cascade_any"],
                             runs, cascades, stream["deploy_attempts"],
                             stream["deploy_rejected"], stream["migrate_attempts"],
                             stream["migrate_rejected"], mask,
                             events_json.get("cascade_envelopes") or [])
        comparison = compare_metrics(declared_phase.get("metrics"),
                                     gate["metrics"])
        comparison["probability_declared"] = \
            declared_phase.get("cascade_task_probability")
        comparison["probability_registered"] = phase["cascade_task_probability"]
        comparison["interval_range_declared"] = declared_phase.get("interval_range")
        comparison["interval_range_registered"] = [phase["start"],
                                                   min(phase["end"], steps)]
        ok = (comparison["all_match"]
              and close(comparison["probability_declared"],
                        comparison["probability_registered"])
              and comparison["interval_range_declared"]
              == comparison["interval_range_registered"])
        comparison["match"] = ok
        phase_report[phase["name"]] = comparison
        per_phase_ok = per_phase_ok and ok
    check("per_phase_gate_metrics_match_audit", per_phase_ok,
          {"per_phase": phase_report})

    # ---------------------------------------------------- model-facing arrays
    finiteness_report, non_finite = {}, {}
    for name in REQUIRED_STREAM_KEYS:
        if name not in stream:
            continue
        info = finiteness(stream[name])
        finiteness_report[name] = info
        if info["n_non_finite"]:
            non_finite[name] = info
    check("no_nan_or_inf_in_any_stream_array", not non_finite,
          {"non_finite": non_finite})
    model_bad = {name: finiteness_report[name] for name in MODEL_FACING
                 if finiteness_report.get(name, {}).get("n_non_finite")}
    check("no_nan_or_inf_in_model_facing_arrays", not model_bad,
          {"model_facing": {name: finiteness_report[name]
                            for name in MODEL_FACING},
           "offenders": model_bad})
    check("raw_labels_are_the_registered_class_range",
          bool(stream["raw_labels"].min() >= 0
               and stream["raw_labels"].max() <= 3),
          {"min": int(stream["raw_labels"].min()),
           "max": int(stream["raw_labels"].max()),
           "counts": np.bincount(stream["raw_labels"].ravel(),
                                 minlength=4).tolist()})
    check("manifest_class_counts_match_recomputed_counts",
          manifest.get("raw_class_counts_scored")
          == np.bincount(stream["raw_labels"][:steps].ravel(),
                         minlength=4).tolist(),
          {"manifest": manifest.get("raw_class_counts_scored")})
    check("capacities_are_strictly_positive",
          bool(np.all(stream["capacities"][:steps] > 0)),
          {"min": float(stream["capacities"][:steps].min()),
           "max": float(stream["capacities"][:steps].max())})

    verdict.update({
        "per_regime_onsets": per_regime_onsets,
        "per_regime_gate": {canonical: {
            "regime_id": entry["regime_id"],
            "passed": entry["gate_passed"],
            "checks": entry["gate_checks"],
            "metrics": entry["gate_metrics"],
            "metrics_match_audit": entry["gate_metrics_match_audit"]}
            for canonical, entry in per_regime_onsets.items()},
        "cross_regime_confound": cross_confound,
        "admissibility": admissibility,
        "admissibility_ok": all(entry["threshold_admissible"]
                                for entry in admissibility.values()),
        "inadmissible_thresholds": sorted(
            "%s:%s" % (canonical, entry["onset_resource"])
            for canonical, entry in admissibility.items()
            if not entry["threshold_admissible"]),
        "checks_run": len(checks),
        "n_failed_checks": len(failures),
        "passed": all(checks.values()),
        "finiteness": finiteness_report,
        "notes": {
            "manifest_role": ("manifest/audit numbers were used only as claims "
                              "to falsify; every measured number above was "
                              "recomputed from stream.npz / task_timeline.npz"),
            "admissibility_gating": ("threshold_admissible and the cross-hit "
                                     "confound count are reported, not gated: "
                                     "the registered thresholds are provisional "
                                     "and the decision belongs to the reviewer"),
            "phase_columns_are_audit_only": (
                "phase_*/cascade_*/host_cascade_* are audit metadata and are "
                "listed as forbidden model inputs in the manifest"),
        },
    })
    verdict["exit_code"] = 0 if verdict["passed"] else 1
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="optional path to also write the verdict JSON "
                             "(the stream directory itself is never written to)")
    parser.add_argument("--indent", type=int, default=None)
    parser.add_argument("--tag-suffix", default="",
                        help="registered suffix of a calibrated stream (round "
                             "2A uses '_calibrated'); the expected directory "
                             "name becomes stream_tag + this suffix")
    args = parser.parse_args()
    try:
        verdict = verify(args.stream_dir, tag_suffix=args.tag_suffix)
    except Exception as exc:                       # unreadable / corrupt input
        verdict = {"protocol": PROTOCOL, "stage": "S2",
                   "stream_dir": str(Path(args.stream_dir).resolve()),
                   "passed": False, "exit_code": 2,
                   "error": type(exc).__name__ + ": " + str(exc),
                   "traceback": traceback.format_exc()}
    if args.json_out is not None:
        Path(args.json_out).write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2) + "\n",
            encoding="utf8")
    print(json.dumps(verdict, ensure_ascii=False, indent=args.indent),
          flush=True)
    return int(verdict.get("exit_code", 2))


if __name__ == "__main__":
    sys.exit(main())
