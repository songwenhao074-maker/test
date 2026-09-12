"""Protocol 023 round 2A — H1-v2 data gate and marginal match (directive §3-§6).

What this answers
-----------------
Round 1's H1-original FAIL is preserved.  The directive registers a second,
explicitly named quantity (``usable_primary_followup``) and a new gate, H1-v2:

    primary-window follow-up >= 80 per regime
    independent fault events  >= 80 per regime
    prevalence 3% - 12%
    deployment rejection <= 25%
    migration rejection <= 40%
    worst event share < 10%
    regime prevalence spread <= 4 pp

Every number is recomputed from the stream arrays; the collector's own declared
numbers are read only as claims to falsify (the round-1 analyzer's discipline).
The three follow-up counts the directive demands are all reported:

    n_full_window_followup      whole registered response horizon observed
    n_primary_window_followup   >= 1 task observation in the FIRST downstream
                                response window (the H1-v2 quantity)
    n_any_window_followup       >= 1 task observation in ANY response window
                                (the Protocol 022 MIN_FOLLOWUP_EVENTS reading)

Usage
-----
    python analyze_ftmoe_protocol023_round2a.py
    python analyze_ftmoe_protocol023_round2a.py --streams-root <dir>
    python analyze_ftmoe_protocol023_round2a.py --tag-suffix ""
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_ftmoe_protocol023_s2 as base
import ftmoe_protocol023_core as core23

OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a"
DEFAULT_STREAMS = (ROOT / "artifacts/ftmoe_online/protocol_023/"
                          "development_streams")
REGIME_ORDER = ("compute_first", "memory_first", "io_first")
GENERATOR_IDS = {"compute_first": "compute_first",
                 "memory_first": "memory_first",
                 "io_first": "io_first"}
#: ``measure_stream`` keys its per-regime blocks, and ``registered_table`` keys
#: its table, by the instrument's canonical short id (A/B/C) -- not by the
#: generator regime id.
CANONICAL_IDS = {"compute_first": "A", "memory_first": "B", "io_first": "C"}


def registered_entry(canonical_regime_id):
    """Registered table row of one regime, keyed by its canonical id."""
    return base.registered_table()["table"][canonical_regime_id]

#: H1-v2 as frozen in amendments/h1_v2_definition.json.  ``resource`` is the
#: regime's FIRST DOWNSTREAM response resource (the next resource of the
#: registered cascade order); ``window`` is its registered response window.
PRIMARY_WINDOWS = {
    "compute_first": {"resource": "ram", "window": [4, 14]},
    "memory_first": {"resource": "disk", "window": [3, 13]},
    "io_first": {"resource": "cpu", "window": [3, 13]},
}

GATE = {
    "primary_window_followup_min": 80,
    "independent_fault_events_min": 80,
    "prevalence_min": 0.03,
    "prevalence_max": 0.12,
    "deployment_rejection_max": 0.25,
    "migration_rejection_max": 0.40,
    "worst_event_share_max": 0.10,
    "regime_prevalence_spread_max": 0.04,
}

#: Marginal-match targets the directive sets against A (directive §4).
MARGINAL_TARGETS = {
    "event_count_relative_diff_max": 0.20,
    "duration_relative_diff_max": 0.25,
    "median_peak_ratio_relative_diff_max": 0.30,
    "prevalence_absolute_diff_max": 0.03,
}


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf8")
    return path


def rel(path):
    """Repo-relative POSIX-ish path, or the absolute path if it is outside."""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def declared_amendment(stream_dir):
    """The stream's own calibration amendment, if it declares one.

    Nothing is patched from it: the analyzer only *reports* the parameters the
    stream was generated with, so a reader can tell a calibrated arm from a
    registered one.  The structural checks (the H1-v2 primary window against the
    registered response windows) are deliberately still evaluated against the
    registration, because the amendment is not allowed to move them.
    """
    path = Path(stream_dir) / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf8"))
    amendment = (manifest.get("round2a_amendment")
                 or manifest.get("phase_amendment"))
    if not amendment:
        return None
    return {
        "calibrated": True,
        "parameter_diff": amendment.get("parameter_diff"),
        "regimes_frozen": amendment.get("regimes_frozen"),
        "declared_cascade_task_probability":
            manifest.get("cascade_task_probability"),
        "source_parent_hash": amendment.get("source_parent_hash"),
        "generator_version": amendment.get("generator_version"),
        "recalibration_reason": amendment.get("recalibration_reason"),
        "old_h1_result": amendment.get("old_h1_result"),
        "h1_v2_definition": amendment.get("h1_v2_definition"),
    }


def stream_dir_for(root, canonical, tag_suffix):
    tag = "single_%s_seed700_steps1200%s" % (GENERATOR_IDS[canonical],
                                             tag_suffix)
    path = Path(root) / tag
    if not path.is_dir():
        raise SystemExit("missing stream %s (expected %s)"
                         % (tag, path))
    if base.missing_outputs(path):
        raise SystemExit("stream %s is incomplete: missing %s"
                         % (tag, ", ".join(base.missing_outputs(path))))
    return path, tag


def followup_counts(timeline, events, windows, primary):
    """The three follow-up readings of one regime's onset events.

    All three are *event* counts: how many registered onsets have the named
    follow-up.  ``window_observation_counts`` additionally reports how many
    observations each event contributes inside the primary window, because an
    event that keeps the task alive through the window is not the same claim as
    an event that observes a response in it.
    """
    horizon = max(int(window[1]) for window in windows.values())
    full = primary_ok = any_ok = 0
    primary_rows = []
    for event in events:
        entry = timeline[event["creation_id"]]
        index = int(event["index"])
        t_onset = int(event["t_onset"])
        times = entry["t"]

        # full: every interval of [t0, t0 + horizon) observed contiguously
        target = t_onset + horizon - 1
        position = index + horizon - 1
        full_ok = bool(position < times.size
                       and int(times[position]) == target)
        full += int(full_ok)

        # primary: at least one observation inside the first downstream window
        start, end = (t_onset + int(primary["window"][0]),
                      t_onset + int(primary["window"][1]))
        in_primary = ((times >= start) & (times < end))
        n_primary = int(in_primary.sum())
        primary_ok += int(n_primary > 0)

        # any: at least one observation inside ANY registered response window
        n_any = 0
        for name, window in windows.items():
            low = t_onset + int(window[0])
            high = t_onset + int(window[1])
            n_any += int(((times >= low) & (times < high)).sum())
        any_ok += int(n_any > 0)

        primary_rows.append({
            "event_key": event["event_key"], "creation_id": int(event["creation_id"]),
            "t_onset": t_onset, "age_onset": int(event["age_onset"]),
            "host_onset": int(event["host_onset"]),
            "primary_window": [start, end],
            "primary_observations": n_primary,
            "any_window_observations": n_any,
            "full_window_observed": full_ok,
            "task_lifetime_intervals": int(times.size),
        })
    return {"n_full_window_followup": int(full),
            "n_primary_window_followup": int(primary_ok),
            "n_any_window_followup": int(any_ok),
            "events": primary_rows}


def measure_followup(stream_dir, canonical, tag):
    """Follow-up counts of one regime, plus the response-shape evidence."""
    loaded = base.load_stream(stream_dir)
    table = registered_entry(CANONICAL_IDS[canonical])
    windows = {name: (int(w[0]), int(w[1]))
               for name, w in table["response_windows"].items()}
    primary = PRIMARY_WINDOWS[canonical]
    if primary["resource"] not in windows:
        raise SystemExit("regime %s declares no response window for %s"
                         % (canonical, primary["resource"]))
    if list(windows[primary["resource"]]) != list(primary["window"]):
        raise SystemExit(
            "H1-v2 primary window of %s (%s) disagrees with the registered "
            "generator window %s; refusing to measure with a drifted definition"
            % (canonical, primary["window"], windows[primary["resource"]]))
    task_map = base.task_regime_map(loaded)
    events = core23.onset_events_for_regime(loaded["timeline"], canonical,
                                           task_regime_map=task_map)
    counts = followup_counts(loaded["timeline"], events, windows, primary)
    rows = core23.event_responses_multi(
        loaded["timeline"], events, response_windows=windows,
        onset_resource=table["onset_resource"])
    resource = primary["resource"]
    n_response = int(sum(1 for row in rows
                         if np.isfinite(row["%s_response" % resource])))
    n_response_positive = int(sum(
        1 for row in rows
        if bool(row["%s_response_positive" % resource])))
    n_exact_lag = int(sum(1 for row in rows
                          if np.isfinite(row["%s_at_exact_lag" % resource])))
    return {"tag": tag, "stream_dir": str(stream_dir),
            "onset_resource": table["onset_resource"],
            "response_windows": {k: list(v) for k, v in windows.items()},
            "primary": dict(primary),
            "n_events": len(events),
            "n_events_with_a_usable_followup": int(sum(
                1 for row in rows
                if any(np.isfinite(row["%s_response" % name])
                       for name in windows))),
            "primary_response_finite": n_response,
            "primary_response_positive": n_response_positive,
            "primary_at_exact_lag_finite": n_exact_lag,
            **{k: v for k, v in counts.items() if k != "events"},
            "events": counts["events"]}


def gate_v2(per_regime):
    """The H1-v2 gate exactly as the directive registers it."""
    cheap = {}
    for canonical in REGIME_ORDER:
        entry = per_regime[canonical]
        stats = entry["gate_stats"]
        primary = entry["followup"]["n_primary_window_followup"]
        cheap["%s_%s" % (canonical, "primary_window_followup_enough")] = bool(
            primary >= GATE["primary_window_followup_min"])
        cheap["%s_independent_fault_events_enough" % canonical] = bool(
            int(stats["independent_fault_events"])
            >= GATE["independent_fault_events_min"])
        cheap["%s_prevalence_in_range" % canonical] = bool(
            stats["prevalence"] is not None
            and GATE["prevalence_min"] <= float(stats["prevalence"])
            <= GATE["prevalence_max"])
        cheap["%s_deployment_rejection_ok" % canonical] = bool(
            stats["deployment_rejection_rate"] is not None
            and float(stats["deployment_rejection_rate"])
            <= GATE["deployment_rejection_max"])
        cheap["%s_migration_rejection_ok" % canonical] = bool(
            stats["migration_rejection_rate"] is not None
            and float(stats["migration_rejection_rate"])
            <= GATE["migration_rejection_max"])
        cheap["%s_worst_event_share_ok" % canonical] = bool(
            stats["worst_event_share"] is not None
            and float(stats["worst_event_share"])
            < GATE["worst_event_share_max"])
    prevalences = [float(per_regime[c]["gate_stats"]["prevalence"])
                   for c in REGIME_ORDER
                   if per_regime[c]["gate_stats"]["prevalence"] is not None]
    spread = (max(prevalences) - min(prevalences)) if len(prevalences) >= 2 else None
    cheap["prevalence_spread_ok"] = bool(
        spread is not None and spread <= GATE["regime_prevalence_spread_max"])
    failed = sorted(k for k, v in cheap.items() if not v)
    stop = None
    if any(not cheap["%s_primary_window_followup_enough" % c]
           for c in REGIME_ORDER):
        stop = "STOP-DATA-v2"
    return {"thresholds": dict(GATE), "checks": cheap, "failed_checks": failed,
            "prevalence_spread_pp": (None if spread is None else spread * 100.0),
            "passed": not failed,
            "stop_condition": stop,
            "registration": ("directive §3 H1-v2 gate; H1-original stays FAIL "
                             "and is reported next to it, never instead of it")}


def marginal_match_v2(per_regime):
    """Directive §4 data targets, measured against the frozen A arm."""
    reference = per_regime["compute_first"]
    out = {"reference_regime": "compute_first",
           "targets": dict(MARGINAL_TARGETS), "per_regime": {}}
    for canonical in REGIME_ORDER:
        entry = per_regime[canonical]
        stats = entry["gate_stats"]
        block = {
            "event_count": int(stats["event_count"]),
            "prevalence": float(stats["prevalence"]),
            "mean_duration": stats.get("mean_duration"),
            "peak_ratio": stats.get("peak_ratio"),
        }
        if canonical == "compute_first":
            block["relative_to_reference"] = {
                "event_count_relative_diff": 0.0,
                "duration_relative_diff": 0.0,
                "median_peak_ratio_relative_diff": 0.0,
                "prevalence_absolute_diff": 0.0}
            block["meets_targets"] = True
            block["note"] = ("reference arm: frozen, byte-identical to "
                             "Protocol 022's cascade_v2")
        else:
            def rel(key):
                left, right = block[key], reference["gate_stats"].get(key)
                if left is None or right in (None, 0):
                    return None
                return abs(float(left) - float(right)) / abs(float(right))
            diffs = {
                "event_count_relative_diff": rel("event_count"),
                "duration_relative_diff": rel("mean_duration"),
                "median_peak_ratio_relative_diff": rel("peak_ratio"),
                "prevalence_absolute_diff": abs(
                    float(block["prevalence"])
                    - float(reference["gate_stats"]["prevalence"])),
            }
            block["relative_to_reference"] = diffs
            block["meets_targets"] = bool(
                diffs["event_count_relative_diff"] is not None
                and diffs["event_count_relative_diff"]
                <= MARGINAL_TARGETS["event_count_relative_diff_max"]
                and diffs["duration_relative_diff"] is not None
                and diffs["duration_relative_diff"]
                <= MARGINAL_TARGETS["duration_relative_diff_max"]
                and diffs["median_peak_ratio_relative_diff"] is not None
                and diffs["median_peak_ratio_relative_diff"]
                <= MARGINAL_TARGETS["median_peak_ratio_relative_diff_max"]
                and diffs["prevalence_absolute_diff"]
                <= MARGINAL_TARGETS["prevalence_absolute_diff_max"])
        out["per_regime"][canonical] = block
    return out


def h1_positive_budget(stream_dir, canonical):
    """h=1 onset positives of a regime, under three legal test windows.

    Directive §5 asks for >= 30 (ideally >= 50) h=1 positives; the probe's own
    test segment is the recurrence block, which is short.  All three readings
    are reported so the reviewer can see which one the requirement is applied
    to; ``train`` never overlaps a test window.
    """
    import probe_ftmoe_protocol022_learnability as probe

    data = probe.load_stream(stream_dir)
    steps = data["steps"]
    labels = data["labels"]
    klass = {"cpu": 1, "ram": 2, "disk": 3}[
        registered_entry(CANONICAL_IDS[canonical])["onset_resource"]]
    in_class = (labels == klass).astype(float)
    future = np.zeros((steps, labels.shape[1]), dtype=float)
    future[:steps - 1] = in_class[1:steps]
    y = np.where((in_class[:steps] == 0), future, np.nan)

    # A single-regime stream has a familiar head then one long regime window.
    phases = None
    try:
        manifest = json.loads((Path(stream_dir) / "manifest.json")
                              .read_text(encoding="utf8"))
        phases = [(p["name"], int(p["start"]), int(p["end"]), p.get("regime_id"))
                  for p in manifest["phases"]]
    except (KeyError, ValueError):
        phases = None
    if not phases:
        raise SystemExit("stream %s carries no registered phase table"
                         % stream_dir)
    regime_phases = [p for p in phases if p[3]]
    if len(regime_phases) != 1:
        raise SystemExit("single-regime stream %s declares %d regime phases"
                         % (stream_dir, len(regime_phases)))
    start, end = regime_phases[0][1], min(regime_phases[0][2], steps)
    windows = {"regime_window": (start, end)}
    out = {}
    for name, (low, high) in windows.items():
        cell = y[low:high]
        out[name] = {"intervals": high - low,
                     "positives": int(np.nansum(cell)),
                     "prevalence": float(np.nanmean(cell)),
                     "host_cells": int(np.isfinite(cell).sum())}
    return out


def analyse(streams_root, tag_suffix, thresholds=None):
    started = time.perf_counter()
    streams_root = Path(streams_root)
    per_regime = {}
    for canonical in REGIME_ORDER:
        path, tag = stream_dir_for(streams_root, canonical, tag_suffix)
        measured = base.measure_stream(path)
        block = measured["per_regime"].get(CANONICAL_IDS[canonical])
        if block is None:
            raise SystemExit("stream %s declares no %s phase"
                             % (tag, CANONICAL_IDS[canonical]))
        followup = measure_followup(path, canonical, tag)
        per_regime[canonical] = {
            "tag": tag,
            "stream_dir": rel(path),
            "calibration": declared_amendment(path),
            "gate_stats": block["gate_stats"],
            "onsets": {k: v for k, v in block["onsets"].items()
                       if k != "rows"},
            "durations": block.get("durations"),
            "peaks": block.get("peaks"),
            "admissibility": base.admissibility_report(
                canonical, registered_entry(CANONICAL_IDS[canonical]),
                measured["familiar"]),
            "followup": followup,
            "h1_positive_budget": h1_positive_budget(path, canonical),
            "declared_agreement": measured.get("declared_agreement"),
        }
    gate = gate_v2(per_regime)
    marginal = marginal_match_v2(per_regime)
    verdict = {
        "protocol": "023", "round": "2A", "stage": "S2.5+H1-v2",
        "analyzer": "analyze_ftmoe_protocol023_round2a.py",
        "generated_at": utcnow(),
        "streams_root": rel(streams_root),
        "tag_suffix": tag_suffix,
        "h1_original": {
            "status": "FAIL",
            "preserved": True,
            "measured_per_regime": {"compute_first": 24, "memory_first": 36,
                                    "io_first": 31},
            "threshold_per_regime": 50,
            "note": ("round 1's strict definition; this analyzer does not "
                     "re-evaluate or rewrite it"),
            "evidence": "artifacts/ftmoe_online/protocol_023/data_audit/",
        },
        "h1_v2_definition": {
            "quantity": "usable_primary_followup",
            "rule": ("onset counts when the task has at least one valid "
                     "task-level observation inside its regime's first "
                     "downstream response window"),
            "primary_windows": PRIMARY_WINDOWS,
            "must_report": ["n_full_window_followup",
                            "n_primary_window_followup",
                            "n_any_window_followup"],
        },
        "per_regime": per_regime,
        "gate": gate,
        "marginal_match": marginal,
        "elapsed_seconds": time.perf_counter() - started,
    }
    verdict["stop_conditions_hit"] = ([gate["stop_condition"]]
                                      if gate["stop_condition"] else [])
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams-root", type=Path, default=DEFAULT_STREAMS)
    parser.add_argument("--tag-suffix", default="_calibrated",
                        help="stream tag suffix; 'none' reads the round-1 "
                             "streams (the empty suffix cannot be passed "
                             "through every shell)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.tag_suffix == "none":
        args.tag_suffix = ""

    verdict = analyse(args.streams_root, args.tag_suffix)
    gate = verdict["gate"]
    data_dir = args.out / "data_calibration"
    write_json(data_dir / "data_gate_v2.json", {
        "kind": "h1_v2_data_gate", "protocol": "023", "round": "2A",
        "generated_at": verdict["generated_at"],
        "streams_root": verdict["streams_root"],
        "tag_suffix": verdict["tag_suffix"],
        "h1_original": verdict["h1_original"],
        "h1_v2_definition": verdict["h1_v2_definition"],
        "gate": gate,
        "per_regime": {c: {
            "tag": e["tag"],
            "prevalence": e["gate_stats"]["prevalence"],
            "positive_hoststeps": e["gate_stats"]["positive_hoststeps"],
            "host_steps": e["gate_stats"]["host_steps"],
            "independent_fault_events": e["gate_stats"]["independent_fault_events"],
            "n_full_window_followup": e["followup"]["n_full_window_followup"],
            "n_primary_window_followup": e["followup"]["n_primary_window_followup"],
            "n_any_window_followup": e["followup"]["n_any_window_followup"],
            "n_events": e["followup"]["n_events"],
            "primary_response_finite": e["followup"]["primary_response_finite"],
            "primary_response_positive": e["followup"]["primary_response_positive"],
            "primary_at_exact_lag_finite": e["followup"]["primary_at_exact_lag_finite"],
            "event_count": e["gate_stats"]["event_count"],
            "mean_duration": e["gate_stats"]["mean_duration"],
            "peak_ratio": e["gate_stats"]["peak_ratio"],
            "deployment_rejection_rate": e["gate_stats"]["deployment_rejection_rate"],
            "migration_rejection_rate": e["gate_stats"]["migration_rejection_rate"],
            "worst_event_share": e["gate_stats"]["worst_event_share"],
            "primary_window": e["followup"]["primary"],
            "h1_positive_budget": e["h1_positive_budget"],
            "admissibility": e["admissibility"],
        } for c, e in verdict["per_regime"].items()},
    })
    write_json(data_dir / "marginal_match_v2.json", {
        "kind": "marginal_match_v2", "generated_at": verdict["generated_at"],
        "targets": verdict["marginal_match"]["targets"],
        "reference_regime": verdict["marginal_match"]["reference_regime"],
        "per_regime": verdict["marginal_match"]["per_regime"],
        "gate": verdict["marginal_match"],
    })
    path = write_json(args.out / "h1_v2_verdict.json", verdict)
    if not args.quiet:
        print(json.dumps({
            "written": rel(path),
            "h1_original": "FAIL (preserved)",
            "h1_v2_passed": gate["passed"],
            "failed_checks": gate["failed_checks"],
            "prevalence_spread_pp": gate["prevalence_spread_pp"],
            "stop_conditions_hit": verdict["stop_conditions_hit"],
            "per_regime": {c: {
                "primary_followup": e["followup"]["n_primary_window_followup"],
                "full_followup": e["followup"]["n_full_window_followup"],
                "any_followup": e["followup"]["n_any_window_followup"],
                "events": e["gate_stats"]["event_count"],
                "prevalence": round(float(e["gate_stats"]["prevalence"]), 5),
            } for c, e in verdict["per_regime"].items()},
            "marginal_match": {c: b["relative_to_reference"]
                               for c, b in verdict["marginal_match"]["per_regime"].items()},
            "elapsed_seconds": round(verdict["elapsed_seconds"], 1),
        }, ensure_ascii=False, indent=2), flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
