"""Protocol 023 S2 — Data Gate (plan §9) and marginal-matching analyzer.

This is the S2 *gate analyzer*: it evaluates the plan §9 Data Gate on the four
registered S2 streams and reports the marginal-matching picture.  It is the
Protocol-023 sibling of ``analyze_ftmoe_protocol022_unseen.py`` (which produced
the P22 U4-v2 artifact) and it keeps that house discipline:

    the collector's own audit numbers are CLAIMS TO BE FALSIFIED, never inputs.
    Every gate quantity is recomputed here from ``stream.npz`` and
    ``task_timeline.npz``, and the declared numbers are compared against the
    recomputation afterwards (``declared_vs_recomputed``, reported and warned).

Inputs (produced by ``prepare_ftmoe_protocol023_stream.py``):

    <streams-root>/dev_seed700_steps2880                       (all three regimes)
    <streams-root>/single_<regime>_seed700_steps1200           (one per regime)
        for <regime> in compute_first / memory_first / io_first

A missing single-regime stream is a hard error (no gate verdict is written on
absent data).  A missing development stream is NOT: the plan §9 gate is measured
per regime on the three single-regime cohorts, so its absence is recorded as
``dev_stream_missing`` and raised as a loud WARN-NO-DEV-STREAM warning -- but
the verdict still says what it could not see (the plan §13 recurrence/switch
context), instead of withholding a computable gate.

Gate source
-----------
A regime's registered gate block is measured on its **own single-regime
stream**: that is the registered purpose of those streams ("clean per-regime
cohorts for the marginal matching gate", protocol.json), and it gives all three
regimes the same 1050-interval / 16-host window so the three prevalence numbers
are comparable.  The development stream is the multi-regime context: its
per-regime blocks are reported next to the gate blocks (recurrence and phase
switching included) but they never enter the gate.

What is recomputed here (never read from the audit)
--------------------------------------------------
    task timeline          core23.build_task_timeline over the saved per-task rows
    task cohort            {creation_id -> regime} rebuilt from the saved
                           per-observation mechanism ids (cascade_regime, or the
                           stream's cascade_regimes[time, slot] as a fallback),
                           canonicalized through the registered table
    onsets                 core23.onset_events_for_regime(..., task_regime_map=)
                           -> the regime's OWN task cohort only, so a
                           regime-A cascade whose RAM/Disk responses cross the
                           B/C thresholds is never counted as a B/C event
                           (the unmapped scan is reported next to it as the
                           cross-hit confound count)
    prevalence             positive host-steps / host-steps inside the regime's
                           registered phase window, from stream.raw_labels
    fault events           per-host contiguous positive-label runs inside the
                           same window (the registered P22 segmentation rule)
    rejections             deploy/migrate attempts and rejections summed over
                           the window, from the stream's own counters
    worst event share      positive host-steps attributed to the single worst
                           registered cascade event, over the window positives;
                           the label-free worst-run share is reported next to it
    duration / peak ratio  core23.response_durations / core23.peak_ratio_summary
    familiar maxima        per-task maxima over the familiar-phase rows of the
                           SAME stream, measured twice: over every familiar-phase
                           row (conservative) and over envelope-free tasks only
                           (a cascade task created in a preceding regime phase
                           can still carry its burst into a later familiar phase,
                           which would otherwise be read as "a familiar task did
                           this")

Threshold admissibility (plan §6-§7 registration; NOT tunable here)
-------------------------------------------------------------------
A registered onset threshold is admissible only if it is **strictly above** the
measured familiar per-task maximum of its resource.  It is reported next to the
regime's own registered floor, and a threshold that fails is reported as
``threshold_admissible: false`` with the negative margin -- it is NEVER nudged
after seeing the data.  The RAM (2810.0) and disk (11600.0) thresholds are
PROVISIONAL (the generator registers all three as provisional; the instrument
marks RAM/disk as the ones whose "no familiar task can cross it" half is not yet
measured), and every artifact flags them as such.

The follow-up definition (the one place plan §9 needed a decision)
------------------------------------------------------------------
plan §9 registers ``valid onset/follow-up >= 50 / regime`` without saying what
"valid" measures.  Two readings exist: (a) the task observes the WHOLE registered
response window ``[t0, t0 + max(window))``, so the full response shape is
measurable; (b) the P22 U4-v2 reading (``MIN_FOLLOWUP_EVENTS``), where the event
merely has at least one OBSERVED interval inside a window, so a response value
exists at all.  This analyzer GATES (a) -- it is the reading the S2 brief states
("valid onsets with a full follow-up window") and the conservative one -- and
reports (b) next to it as ``n_events_with_a_usable_followup``.  On short-lived
containers the two differ by an order of magnitude, so whenever the choice flips
the check a ``WARN-FOLLOWUP-DEFINITION`` names both numbers.  The definition is
never switched after seeing the number; a reviewer who wants (b) gated changes
one line, deliberately.

Verdict
-------
``gate_passed``  the plan §9 registered gate (``core23.marginal_match_report``
                 over the eight registered numbers), with an independent
                 re-derivation of the same eight checks from the plan's literal
                 numbers (``GATE_THRESHOLDS``); a drift between the two, or
                 between the analyzer's table and the instrument's, is a failed
                 check rather than a silent difference.
``passed``       ``gate_passed`` AND every measured threshold admissible AND the
                 stream-integrity checks clean.  ``failed_on`` names which of the
                 three fired.  The process exits 0 iff ``passed``.

Outputs (written BEFORE the exit code is decided)
-------------------------------------------------
    artifacts/ftmoe_online/protocol_023/data_audit/regime_A.json
    artifacts/ftmoe_online/protocol_023/data_audit/regime_B.json
    artifacts/ftmoe_online/protocol_023/data_audit/regime_C.json
    artifacts/ftmoe_online/protocol_023/data_audit/marginal_match.json
    artifacts/ftmoe_online/protocol_023/data_audit/DATA_GATE.md

Usage:
    python analyze_ftmoe_protocol023_s2.py
    python analyze_ftmoe_protocol023_s2.py --only memory_first
    python analyze_ftmoe_protocol023_s2.py --streams-root <root> --audit-out <dir>

Exit codes: 0 = PASS, 1 = the gate/admissibility/integrity failed (the artifacts
are on disk and say which), 2 = the analysis could not complete (a missing or
malformed stream): ``data_audit/analysis_error.json`` records it and no verdict
file is written -- a missing stream never becomes a zero-filled PASS.
"""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:                     # ``python -c`` / foreign cwd
    sys.path.insert(0, str(ROOT))

import ftmoe_protocol023_core as core23          # noqa: E402

PROTOCOL = "023"
STAGE = "S2"
ANALYZER = "analyze_ftmoe_protocol023_s2.py"
P23 = ROOT / "artifacts/ftmoe_online/protocol_023"
DEFAULT_STREAMS_ROOT = P23 / "development_streams"
DEFAULT_AUDIT_OUT = P23 / "data_audit"

#: Registered stream layout, mirrored from ``prepare_ftmoe_protocol023_stream``
#: (``stream_tag``).  The registered tag is preferred; a single
#: ``<role>_seed*_steps*`` directory is accepted as a fallback so a stream can be
#: re-analysed under a different root, and the choice is recorded in the
#: artifact.  Nothing else about the collector is imported: this analyzer must
#: stay runnable (and torch-free) on its own.
REGISTERED_SINGLE_TAG = "single_%s_seed700_steps1200"
REGISTERED_DEV_TAG = "dev_seed700_steps2880"
SINGLE_GLOB = "single_%s_seed*_steps*"
DEV_GLOB = "dev_seed*_steps*"

#: The registered regime ids of the generator/family ``cascade_v3`` and their
#: mechanism ids (plan §5-§7; ``REGIMES_V3``).  Canonical A/B/C ids come from the
#: instrument, never from here.
REGISTERED_REGIME_IDS = ("compute_first", "memory_first", "io_first")
REGISTERED_MECHANISM_IDS = {"compute_first": 0, "memory_first": 1, "io_first": 2}
GENERATOR_MODULE = "simulator.workload.BitbrainWorkloadProtocol023"
#: ``REGIMES_V3[regime][key]`` -- the registered burst floor of each resource.
RESOURCE_FLOOR_KEY = {"cpu": "cpu_burst_floor", "ram": "ram_target_floor",
                      "disk": "disk_retained_peak"}
#: Sentinel for a task with no registered envelope (the collector's value).
FAMILIAR_TASK_ID = "familiar"

#: plan §9 "建议 Data Gate", transcribed from the plan text -- NOT read from the
#: instrument.  ``threshold_drift_report`` compares this table against
#: ``core23.MARGINAL_MATCH_THRESHOLDS`` at every run, so a drifted threshold
#: (in either place) fails the verdict instead of quietly changing the gate.
GATE_THRESHOLDS = {
    "prevalence_min": 0.03,                 # anomaly prevalence 3%-12%
    "prevalence_max": 0.12,
    "deployment_rejection_max": 0.25,       # deployment rejection <= 25%
    "migration_rejection_max": 0.40,        # migration rejection <= 40%
    "independent_fault_events_min": 80,     # >= 80 per regime
    "valid_onset_followup_min": 50,         # >= 50 per regime
    "worst_event_share_max": 0.10,          # worst event share < 10%
    "prevalence_spread_max": 0.04,          # <= 4 percentage points
}
PLAN_SECTION = ("指令/FTMOE_PROTOCOL023_DYNAMIC_EXPERT_PLAN_20260912.md §9 "
                "(建议 Data Gate)")
PLAN_SECTION_9_TEXT = (
    "anomaly prevalence 3%-12% / deployment rejection <= 25% / migration "
    "rejection <= 40% / independent fault events >= 80 per regime / valid "
    "onset-follow-up >= 50 per regime / worst event share < 10% / regime "
    "prevalence difference <= 4 percentage points")
#: Checks that need more than one regime; a ``--only`` run reports them as not
#: evaluated instead of silently failing (or silently passing) them.
CROSS_REGIME_CHECKS = ("prevalence_spread_ok", "all_registered_regimes_reported")

REQUIRED_STREAM_KEYS = (
    "raw_labels", "phase_ids", "phase_kind", "phase_cascade_probability",
    "phase_regime_ids", "creation_ids", "after_creation_ids",
    "cascade_event_ids", "cascade_regimes", "cascade_phases",
    "host_cascade_any", "host_cascade_event", "deploy_attempts",
    "deploy_rejected", "migrate_attempts", "migrate_rejected")
REQUIRED_TIMELINE_KEYS = ("time", "slot_index", "creation_id", "demand",
                          "host_id", "cascade_event_id", "cascade_phase")
RESOURCES = tuple(core23.RESOURCES)


class AnalyzerError(Exception):
    """The analysis could not complete (exit 2); never a silent PASS."""


class StreamMissingError(AnalyzerError):
    """A registered stream directory (or a required file in it) is absent."""


class StreamFormatError(AnalyzerError):
    """A stream exists but cannot be read as a registered P23 stream."""


# --------------------------------------------------------------------------
# small utilities (P22 house style: atomic JSON writes, strict-JSON safe floats)
# --------------------------------------------------------------------------
def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def sha256(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_safe(value):
    """Replace non-finite floats with null so the JSON files stay strict JSON.

    A NaN here means "not measurable" (e.g. the mean duration of an empty event
    cohort); it is reported as null, never as a fake zero.
    """
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf8")
    temporary.replace(path)
    return path


def configure_threads():
    """Registered resource discipline: 3 threads, single process, numpy only."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "3")


# --------------------------------------------------------------------------
# registration read-outs (the instrument + the generator; both are registrations,
# not measurements, and a disagreement between them fails the verdict)
# --------------------------------------------------------------------------
_GENERATOR_CACHE = {}


def generator_registration():
    """``REGISTERED_ONSET_TAU`` / floors / mechanism ids of the generator.

    Imported lazily (and cached) so ``import analyze_ftmoe_protocol023_s2``
    never depends on the simulator package being importable.  The P23 workload
    module is torch-free on import (verified on this checkout); if it cannot be
    imported the analyzer still runs, reports ``available: false`` and falls back
    to the instrument's registered table -- the fallback is named in the artifact
    and is never silent.
    """
    if _GENERATOR_CACHE:
        return _GENERATOR_CACHE
    out = {"available": False, "module": GENERATOR_MODULE, "error": None,
           "onset_tau": {}, "provisional_onset_tau": {}, "floors": {},
           "regime_ids": (), "mechanism_ids": {}, "source": None}
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = importlib.import_module(GENERATOR_MODULE)
        regime_ids = tuple(str(name) for name in module.REGIME_IDS)
        out.update(
            available=True,
            source=GENERATOR_MODULE,
            regime_ids=regime_ids,
            mechanism_ids={rid: int(module.REGIMES_V3[rid]["mechanism_id"])
                           for rid in regime_ids},
            onset_tau={str(k): float(v)
                       for k, v in module.REGISTERED_ONSET_TAU.items()},
            provisional_onset_tau={str(k): float(v) for k, v in
                                   module.PROVISIONAL_ONSET_TAU.items()},
            floors={rid: {res: float(module.REGIMES_V3[rid][key])
                          for res, key in RESOURCE_FLOOR_KEY.items()}
                    for rid in regime_ids})
    except Exception as exc:                      # pragma: no cover - env specific
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    _GENERATOR_CACHE.update(out)
    return _GENERATOR_CACHE


def registered_table():
    """``{canonical: {...}}`` -- the registered thresholds, windows and floors.

    ``onset_tau`` is the generator's ``REGISTERED_ONSET_TAU[resource]`` when that
    module is importable (the plan names that table), else the instrument's
    registered value; ``tau_source`` records which one was used, and
    ``tau_table_matches_instrument`` records whether they agree.
    """
    generator = generator_registration()
    table, drift = {}, []
    for gen_id in REGISTERED_REGIME_IDS:
        canonical = core23.canonical_regime_id(gen_id)
        spec = core23.regime(canonical)
        resource = spec["onset_resource"]
        tau_instrument = float(spec["onset_tau"])
        tau_generator = (generator["onset_tau"].get(resource)
                         if generator["available"] else None)
        if tau_generator is not None and \
                abs(float(tau_generator) - tau_instrument) > 1e-9:
            drift.append({"regime_id": gen_id, "resource": resource,
                          "instrument": tau_instrument,
                          "generator": float(tau_generator)})
        floor = None
        if generator["available"]:
            floor = generator["floors"].get(gen_id, {}).get(resource)
        table[canonical] = {
            "canonical_regime_id": canonical,
            "regime_id": gen_id,
            "mechanism_id": int(REGISTERED_MECHANISM_IDS[gen_id]),
            "onset_resource": resource,
            "onset_tau": (tau_instrument if tau_generator is None
                          else float(tau_generator)),
            "onset_tau_instrument": tau_instrument,
            "onset_tau_generator": tau_generator,
            "tau_source": ("generator REGISTERED_ONSET_TAU" if tau_generator
                           is not None else
                           "instrument REGIMES (generator module not importable)"),
            "response_windows": {name: [int(x) for x in window] for name, window
                                 in spec["response_windows"].items()},
            "sequence": [[str(name), int(lag)]
                         for name, lag in spec["sequence"]],
            "registered_floor": floor,
            "floor_key": RESOURCE_FLOOR_KEY[resource],
            "provisional": bool(resource in core23.PROVISIONAL_TAUS),
            "generator_provisional": bool(
                generator["available"]
                and resource in generator["provisional_onset_tau"]),
            "registered_familiar_clip": float(core23.FAMILIAR_CLIP[resource]),
        }
    return {"table": table, "tau_drift": drift,
            "tau_table_matches_instrument": not drift,
            "generator": {k: v for k, v in generator.items() if k != "floors"}}


def threshold_drift_report(thresholds=None):
    """The analyzer's plan §9 table against the instrument's registered table.

    Two independent transcriptions of the same eight registered numbers: the
    analyzer's literal table and ``core23.MARGINAL_MATCH_THRESHOLDS``.  A
    difference in either direction is a hard failed check (a gate that drifted
    must not be evaluated as if it had not).
    """
    table = dict(GATE_THRESHOLDS if thresholds is None else thresholds)
    instrument = dict(core23.MARGINAL_MATCH_THRESHOLDS)
    keys = sorted(set(table) | set(instrument))
    drift = [{"key": key, "analyzer": table.get(key),
              "instrument": instrument.get(key)}
             for key in keys if table.get(key) != instrument.get(key)]
    return {"analyzer": table, "instrument": instrument, "drift": drift,
            "passed": not drift, "source": PLAN_SECTION,
            "plan_text": PLAN_SECTION_9_TEXT}


def resolve_only(value):
    """Canonical regime id for ``--only`` (registered aliases accepted)."""
    try:
        canonical = core23.canonical_regime_id(value)
    except KeyError as exc:
        raise AnalyzerError(
            "--only %r is not a registered regime; registered ids are %s "
            "(aliases: A/B/C, compute_first/memory_first/io_first) -- %s"
            % (value, list(core23.REGIME_IDS), exc))
    return canonical


# --------------------------------------------------------------------------
# stream discovery
# --------------------------------------------------------------------------
STREAM_OUTPUTS = ("stream.npz", "task_timeline.npz", "manifest.json")


def missing_outputs(directory):
    """The registered output files a stream directory does not (yet) carry."""
    directory = Path(directory)
    return [name for name in STREAM_OUTPUTS
            if not (directory / name).is_file()]


def _pick_directory(root, registered_name, pattern):
    """The registered tag if it exists, else the single matching directory."""
    registered = Path(root) / registered_name
    if registered.is_dir():
        return registered, "registered_tag", []
    alternatives = sorted(str(path.name) for path in
                          Path(root).glob(pattern) if path.is_dir())
    if alternatives:
        return (Path(root) / alternatives[0], "glob", alternatives)
    return None, None, []


def generator_id(canonical):
    """The registered generator name of a canonical A/B/C id."""
    for name in REGISTERED_REGIME_IDS:
        if core23.canonical_regime_id(name) == canonical:
            return name
    raise AnalyzerError(
        "no registered generator regime maps to canonical id %r (registered: "
        "%s)" % (canonical, list(REGISTERED_REGIME_IDS)))


def resolve_streams(streams_root, only=None):
    """Locate the registered stream directories, or fail with the exact path.

    The three single-regime streams carry the gate cohorts, so a missing one is
    a hard error: no gate verdict may be reported on absent data.  The
    development stream is the multi-regime CONTEXT (recurrence and phase
    switching, plan §13); it is required for the registered S2 record but not for
    the plan §9 arithmetic, which is per regime, so its absence is reported as a
    ``dev_stream_missing`` block plus a loud warning instead of blocking a gate
    that is fully computable from the three single-regime cohorts.
    ``only`` relaxes the run to one single-regime stream.
    """
    root = Path(streams_root)
    if not root.is_dir():
        raise StreamMissingError(
            "streams root %s does not exist; the S2 collector writes the four "
            "registered streams there (%s and %s)"
            % (root, REGISTERED_DEV_TAG, REGISTERED_SINGLE_TAG % "<regime>"))
    wanted = ([core23.canonical_regime_id(only)] if only
              else [core23.canonical_regime_id(name)
                    for name in REGISTERED_REGIME_IDS])
    resolved = {"root": root, "single": {}, "dev": None, "sources": {},
                "alternatives": {}, "missing": [], "only": only,
                "dev_missing": None}
    for canonical in wanted:
        gen_id = generator_id(canonical)
        path, source, alternatives = _pick_directory(
            root, REGISTERED_SINGLE_TAG % gen_id, SINGLE_GLOB % gen_id)
        if path is None:
            resolved["missing"].append({
                "role": "single_%s" % gen_id, "canonical_regime_id": canonical,
                "expected_path": str(root / (REGISTERED_SINGLE_TAG % gen_id)),
                "glob": str(root / (SINGLE_GLOB % gen_id))})
            continue
        resolved["single"][canonical] = path
        resolved["sources"][canonical] = source
        if alternatives:
            resolved["alternatives"][canonical] = alternatives
    dev, dev_source, dev_alternatives = _pick_directory(
        root, REGISTERED_DEV_TAG, DEV_GLOB)
    if dev is not None and missing_outputs(dev):
        # The collector creates the output directory before it simulates and
        # writes the arrays at the END of the run, so a registered tag holding
        # only generation.log is a run still in progress -- not a stream.
        resolved["dev_missing"] = {
            "role": "dev", "expected_path": str(dev), "glob": str(root / DEV_GLOB),
            "reason": ("incomplete: missing %s (the collector writes the arrays "
                       "at the end of the run, so this is most likely a "
                       "collection still in progress)" % missing_outputs(dev)),
            "consequence": ("the plan §9 gate is per regime and is unaffected "
                            "(its cohorts come from the single-regime streams), "
                            "but the multi-regime recurrence/switch context of "
                            "plan §13 is missing from this verdict")}
        dev = None
    if dev is not None:
        resolved["dev"] = dev
        resolved["sources"]["dev"] = dev_source
        if dev_alternatives:
            resolved["alternatives"]["dev"] = dev_alternatives
    elif only is None and resolved["dev_missing"] is None:
        resolved["dev_missing"] = {
            "role": "dev", "expected_path": str(root / REGISTERED_DEV_TAG),
            "glob": str(root / DEV_GLOB), "reason": "not found",
            "consequence": ("the plan §9 gate is per regime and is unaffected "
                            "(its cohorts come from the single-regime streams), "
                            "but the multi-regime recurrence/switch context of "
                            "plan §13 is missing from this verdict")}
    if resolved["missing"]:
        details = ["%s: expected %s" % (item["role"], item["expected_path"])
                   for item in resolved["missing"]]
        raise StreamMissingError(
            "missing registered single-regime stream directory/ies under %s -- "
            "%s.  The analyzer refuses to report a gate on absent data (no "
            "zero-filled PASS); collect the stream or re-run with --only "
            "<regime>." % (root, "; ".join(details)))
    return resolved


# --------------------------------------------------------------------------
# stream loading and recomputation
# --------------------------------------------------------------------------
def load_stream(stream_dir):
    """Read one stream directory into arrays + timeline + manifest."""
    stream_dir = Path(stream_dir)
    if not stream_dir.is_dir():
        raise StreamMissingError("stream directory %s does not exist"
                                 % stream_dir)
    paths = {name: stream_dir / name for name in STREAM_OUTPUTS}
    absent = sorted(name for name, path in paths.items() if not path.is_file())
    if absent:
        raise StreamMissingError(
            "stream directory %s is incomplete: missing %s (expected "
            "stream.npz, task_timeline.npz, manifest.json)"
            % (stream_dir, absent))
    with np.load(paths["stream.npz"], allow_pickle=False) as data:
        stream = {name: data[name] for name in data.files}
    with np.load(paths["task_timeline.npz"], allow_pickle=False) as data:
        saved = {name: data[name] for name in data.files}
    try:
        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf8"))
    except ValueError as exc:
        raise StreamFormatError("manifest.json of %s is not valid JSON: %s"
                                % (stream_dir, exc))
    missing = [key for key in REQUIRED_STREAM_KEYS if key not in stream]
    if missing:
        raise StreamFormatError(
            "stream.npz of %s is missing the key(s) %s; this analyzer recomputes "
            "every gate quantity from the arrays and cannot run on a partial "
            "stream" % (stream_dir, missing))
    missing = [key for key in REQUIRED_TIMELINE_KEYS if key not in saved]
    if missing:
        raise StreamFormatError(
            "task_timeline.npz of %s is missing the key(s) %s"
            % (stream_dir, missing))
    if "steps" not in manifest:
        raise StreamFormatError(
            "manifest.json of %s carries no 'steps' (the scored-interval "
            "horizon); refusing to guess it from the array shapes" % stream_dir)
    steps = int(manifest["steps"])
    labels = np.asarray(stream["raw_labels"])
    rows = int(labels.shape[0])
    if labels.ndim != 2:
        raise StreamFormatError("raw_labels must be [rows, hosts], got %s"
                                % (labels.shape,))
    if steps <= 0 or steps > rows:
        raise StreamFormatError(
            "manifest steps=%d is not inside the %d saved row(s) of %s"
            % (steps, rows, stream_dir))
    if any(int(stream[key].shape[0]) != rows for key in REQUIRED_STREAM_KEYS
           if np.asarray(stream[key]).ndim >= 1):
        offenders = {key: list(np.asarray(stream[key]).shape)
                     for key in REQUIRED_STREAM_KEYS
                     if int(np.asarray(stream[key]).shape[0]) != rows}
        raise StreamFormatError(
            "stream arrays of %s are not parallel on the row axis: %s"
            % (stream_dir, offenders))
    time = np.asarray(saved["time"], dtype=np.int64)
    slot = np.asarray(saved["slot_index"], dtype=np.int64)
    creation = np.asarray(saved["creation_id"], dtype=np.int64)
    demand = np.asarray(saved["demand"], dtype=np.float64)
    host = np.asarray(saved["host_id"], dtype=np.int64)
    event_id = np.asarray(saved["cascade_event_id"], dtype=np.int64)
    phase = np.asarray(saved["cascade_phase"], dtype=np.int64)
    try:
        timeline = core23.build_task_timeline(time, slot, creation, demand,
                                              host, event_id, phase)
    except (ValueError, TypeError) as exc:
        raise StreamFormatError(
            "the task timeline of %s is inadmissible: %s" % (stream_dir, exc))
    if "cascade_regime" in saved:
        regime_values = np.asarray(saved["cascade_regime"], dtype=np.int64)
        regime_source = "task_timeline.cascade_regime"
    else:
        regime_values = np.asarray(
            stream["cascade_regimes"][np.clip(time, 0, rows - 1),
                                      np.clip(slot, 0, np.asarray(
                                          stream["cascade_regimes"]).shape[1]
                                              - 1)], dtype=np.int64)
        regime_source = "stream.cascade_regimes[time, slot]"
        if int(regime_values.size) != int(creation.size):
            raise StreamFormatError(
                "cannot resolve a per-observation regime for %s: "
                "task_timeline.npz lacks 'cascade_regime' and the stream lookup "
                "shapes do not line up" % stream_dir)
    return {"stream_dir": stream_dir, "manifest": manifest, "stream": stream,
            "saved": saved, "timeline": timeline, "steps": steps, "rows": rows,
            "time": time, "slot": slot, "creation": creation,
            "regime_values": regime_values, "regime_source": regime_source,
            "n_hosts": int(labels.shape[1]),
            "n_slots": int(np.asarray(stream["creation_ids"]).shape[1]),
            "stream_sha256": sha256(paths["stream.npz"]),
            "task_timeline_sha256": sha256(paths["task_timeline.npz"])}


def task_regime_map(loaded):
    """``{creation_id: canonical regime id or FAMILIAR_TASK_ID}``.

    Built from the saved per-observation mechanism ids, canonicalized through the
    registered table.  A task stamped with two different regimes is a hard error:
    the envelope is frozen at creation, so that would make the cohort ambiguous
    and the whole gate uninterpretable.
    """
    mechanism_to_canonical = {
        int(value): core23.canonical_regime_id(name)
        for name, value in REGISTERED_MECHANISM_IDS.items()}
    creation = loaded["creation"]
    values = loaded["regime_values"]
    if int(values.size) != int(creation.size):
        raise StreamFormatError(
            "the regime column of %s has %d entry/entries for %d timeline row(s)"
            % (loaded["stream_dir"], int(values.size), int(creation.size)))
    per_task, offenders = {}, []
    for index in range(int(creation.size)):
        cid = int(creation[index])
        mechanism = int(values[index])
        canonical = None
        if mechanism >= 0:
            canonical = mechanism_to_canonical.get(mechanism)
            if canonical is None:
                offenders.append({"creation_id": cid, "mechanism_id": mechanism,
                                  "row": index})
                continue
        previous = per_task.setdefault(cid, canonical)
        if previous != canonical:
            raise StreamFormatError(
                "creation_id %d of %s is stamped with two regimes (%r and %r); "
                "a task's envelope is frozen at creation, so its cohort is "
                "ambiguous" % (cid, loaded["stream_dir"], previous, canonical))
    if offenders:
        raise StreamFormatError(
            "%s stamps unregistered mechanism id(s) %s"
            % (loaded["stream_dir"], offenders[:5]))
    missing = [cid for cid in loaded["timeline"] if cid not in per_task]
    if missing:
        raise StreamFormatError(
            "the recomputed regime map of %s covers %d of %d timeline task(s) "
            "(e.g. %s)" % (loaded["stream_dir"], len(per_task),
                           len(loaded["timeline"]), missing[:5]))
    return {cid: (FAMILIAR_TASK_ID if value is None else value)
            for cid, value in per_task.items()}


def ranges_of(mask):
    """``[[start, end), ...]`` contiguous True runs of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    out, start = [], None
    for index in range(mask.size):
        if mask[index] and start is None:
            start = index
        elif not mask[index] and start is not None:
            out.append([int(start), int(index)])
            start = None
    if start is not None:
        out.append([int(start), int(mask.size)])
    return out


def positive_runs(labels, steps, mask=None):
    """Per-host contiguous positive-label runs inside a window.

    The registered P22 segmentation rule (``run_segmentation`` in the P22/P23
    collectors): a run is a maximal sequence of consecutive intervals whose label
    is the same positive class; a different positive class, a negative interval,
    or an interval outside the window ends it.  Reimplemented here on purpose --
    the gate must not depend on the collector's arithmetic -- and cross-checked
    against the stream's declared counts.
    """
    labels = np.asarray(labels)
    runs = []
    for h in range(int(labels.shape[1])):
        current, length, start = 0, 0, 0
        for t in range(int(steps)):
            if mask is not None and not bool(mask[t]):
                if current > 0 and length:
                    runs.append({"host": h, "class": current, "start": start,
                                 "end": start + length - 1, "duration": length})
                current, length = 0, 0
                continue
            label = int(labels[t, h])
            if label > 0 and label == current:
                length += 1
            else:
                if current > 0 and length:
                    runs.append({"host": h, "class": current, "start": start,
                                 "end": start + length - 1, "duration": length})
                current, length, start = label, (1 if label > 0 else 0), t
        if current > 0 and length:
            runs.append({"host": h, "class": current, "start": start,
                         "end": start + length - 1, "duration": length})
    return runs


def attribute_runs_to_events(runs, host_cascade_event, mask):
    """Positive host-steps attributed to the registered cascade events.

    Mirrors the collector's definition: a run contributes its duration to every
    cascade event that is live and inside one of its registered windows on that
    host at some interval of the run.
    """
    host_cascade_event = np.asarray(host_cascade_event)
    per_event = {}
    for run in runs:
        hits = set()
        for t in range(run["start"], run["end"] + 1):
            if mask is not None and not bool(mask[t]):
                continue
            if t >= host_cascade_event.shape[0] or \
                    run["host"] >= host_cascade_event.shape[1]:
                continue
            event = int(host_cascade_event[t, run["host"]])
            if event >= 0:
                hits.add(event)
        for event in hits:
            per_event[event] = per_event.get(event, 0) + run["duration"]
    return per_event


def full_followup_flags(timeline, events, windows):
    """Which onsets observe their whole registered response horizon.

    An onset is valid with a full follow-up window when the task's own
    observations cover ``[t0, t0 + max(window))`` continuously -- the same
    horizon the instrument's response statistics read.
    """
    horizon = max(int(window[1]) for window in windows.values())
    flags = []
    for event in events:
        entry = timeline[event["creation_id"]]
        index = int(event["index"])
        target = int(event["t_onset"]) + horizon - 1
        position = index + horizon - 1
        ok = bool(position < entry["t"].size
                  and int(entry["t"][position]) == target)
        flags.append(ok)
    return flags, horizon


def measure_familiar(loaded):
    """Per-task maxima of the familiar-phase rows of THIS stream.

    Two measurements, reported side by side because they answer two different
    questions:

    ``all_familiar_rows``      every task that has a row inside a familiar phase,
                               which is the conservative reading of "what the
                               familiar phase contains" (a cascade task created
                               in a preceding regime phase can still carry its
                               burst into a later familiar phase);
    ``envelope_free_tasks``    only tasks whose familiar-phase rows all carry no
                               registered envelope -- i.e. what a genuinely
                               familiar task can demand.

    The conservative one decides admissibility (a threshold must clear the worst
    thing the familiar window contains); the other is the corroborating
    measurement and is what tells a reader that a failure is spill-over rather
    than a familiar surprise.
    """
    stream = loaded["stream"]
    steps = loaded["steps"]
    timeline = loaded["timeline"]
    familiar_intervals = np.asarray(stream["phase_kind"][:steps]) == 0
    per_task_all = {name: [] for name in RESOURCES}
    per_task_clean = {name: [] for name in RESOURCES}
    n_tasks_with_rows, n_tasks_envelope_free, n_tasks_with_envelope = 0, 0, 0
    for cid in sorted(timeline):
        entry = timeline[cid]
        times = entry["t"]
        inside = familiar_intervals[times] if times.size else np.zeros(0, bool)
        if not bool(inside.any()):
            continue
        n_tasks_with_rows += 1
        envelope_free = bool(np.all(entry["event_id"][inside] < 0))
        if envelope_free:
            n_tasks_envelope_free += 1
        else:
            n_tasks_with_envelope += 1
        for name in RESOURCES:
            values = entry[name][inside]
            values = values[np.isfinite(values)]
            if values.size:
                per_task_all[name].append(float(values.max()))
                if envelope_free:
                    per_task_clean[name].append(float(values.max()))

    def summarize(values):
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {"n": 0, "max": None, "median": None}
        return {"n": int(array.size), "max": float(array.max()),
                "median": float(np.median(array))}

    summary_all = {name: summarize(per_task_all[name]) for name in RESOURCES}
    summary_clean = {name: summarize(per_task_clean[name]) for name in RESOURCES}
    return {
        "familiar_intervals": int(familiar_intervals.sum()),
        "familiar_ranges": ranges_of(familiar_intervals),
        "n_tasks_with_familiar_rows": n_tasks_with_rows,
        "n_tasks_envelope_free": n_tasks_envelope_free,
        "n_tasks_carrying_an_envelope": n_tasks_with_envelope,
        "per_task_maxima_all_familiar_rows": summary_all,
        "per_task_maxima_envelope_free_tasks": summary_clean,
        "measured_familiar_task_maximum": {
            name: summary_all[name]["max"] for name in RESOURCES},
        "envelope_free_familiar_task_maximum": {
            name: summary_clean[name]["max"] for name in RESOURCES},
        "registered_familiar_clip": dict(core23.FAMILIAR_CLIP),
        "measurement_source": ("per-task maxima over the familiar-phase rows "
                               "(phase_kind == 0) of this stream"),
    }


def admissibility_report(canonical, table_entry, familiar):
    """Is the regime's registered onset threshold above the familiar maximum?"""
    resource = table_entry["onset_resource"]
    tau = float(table_entry["onset_tau"])
    measured = familiar["measured_familiar_task_maximum"].get(resource)
    clean = familiar["envelope_free_familiar_task_maximum"].get(resource)
    floor = table_entry["registered_floor"]
    admissible = None if measured is None else bool(tau > float(measured))
    return {
        "regime_id": table_entry["regime_id"],
        "canonical_regime_id": canonical,
        "onset_resource": resource,
        "registered_onset_threshold": tau,
        "tau_source": table_entry["tau_source"],
        "registered_familiar_clip": table_entry["registered_familiar_clip"],
        "registered_floor": floor,
        "floor_key": table_entry["floor_key"],
        "measured_familiar_task_maximum": measured,
        "n_familiar_tasks_measured": familiar["per_task_maxima_all_familiar_rows"]
                                        [resource]["n"],
        "threshold_admissible": admissible,
        "margin_above_measured_maximum": (None if measured is None
                                          else float(tau - float(measured))),
        "margin_below_floor": (None if floor is None
                               else float(float(floor) - tau)),
        "envelope_free_familiar_task_maximum": clean,
        "admissible_against_envelope_free_tasks": (
            None if clean is None else bool(tau > float(clean))),
        "provisional": bool(table_entry["provisional"]),
        "generator_provisional": bool(table_entry["generator_provisional"]),
        "provisional_note": (
            "PROVISIONAL threshold: the 'no familiar task can cross it' half of "
            "the registered argument is not measured for RAM/disk under P23 "
            "(plan §6-§7); the measured maximum above is this stream's own "
            "evidence and the threshold is reported, never retuned, when it "
            "fails"),
        "rule": ("admissible iff the registered threshold is STRICTLY above the "
                 "measured familiar per-task maximum of its resource; an "
                 "unmeasurable maximum is reported inadmissible (it cannot be "
                 "shown to be above the familiar level)"),
        "measurement_source": familiar["measurement_source"],
    }


def measure_regime(loaded, task_map, canonical, table_entry, mask):
    """Every recomputed quantity the gate and the marginal matching need."""
    spec = core23.regime(canonical)
    resource = table_entry["onset_resource"]
    tau = float(table_entry["onset_tau"])
    windows = {name: (int(window[0]), int(window[1]))
               for name, window in table_entry["response_windows"].items()}
    if spec["onset_resource"] != resource or \
            {name: (int(w[0]), int(w[1]))
             for name, w in spec["response_windows"].items()} != windows:
        raise AnalyzerError(
            "the generator registration and the instrument disagree for regime "
            "%s: instrument %r/%r vs generator registration %r/%r"
            % (canonical, spec["onset_resource"], spec["response_windows"],
               resource, windows))
    timeline = loaded["timeline"]
    stream = loaded["stream"]
    labels = np.asarray(stream["raw_labels"])
    steps = loaded["steps"]
    n_hosts = loaded["n_hosts"]

    events = core23.onset_events_for_regime(timeline, canonical,
                                            task_regime_map=task_map)
    blind = core23.onset_events_multi(
        timeline, resource, tau, response_windows=windows,
        regime_map={resource: canonical})
    followup, horizon = full_followup_flags(timeline, events, windows)
    valid_events = [event for event, ok in zip(events, followup) if ok]
    response_rows = core23.event_responses_multi(
        timeline, events, response_windows=windows, onset_resource=resource)
    #: The P22 U4-v2 definition of "valid follow-up" (``MIN_FOLLOWUP_EVENTS``):
    #: the event has at least one OBSERVED interval inside a registered response
    #: window, so a response value exists.  Reported next to the gate's own
    #: definition (the whole window observed) because the two can differ by an
    #: order of magnitude on short-lived containers, and the reviewer must see
    #: which one the number 50 was applied to.
    usable_followup = int(sum(
        1 for row in response_rows
        if any(np.isfinite(row["%s_response" % name]) for name in windows)))
    durations = core23.response_durations(
        timeline, events, response_windows=windows, onset_resource=resource)
    durations_valid = core23.response_durations(
        timeline, valid_events, response_windows=windows,
        onset_resource=resource)
    peaks = core23.peak_ratio_summary(events, response_windows=windows,
                                      onset_resource=resource)
    alignment = {name: core23.alignment_rate_multi(response_rows, name,
                                                   windows[name][0])
                 for name in sorted(windows)}

    positives = int(np.count_nonzero(labels[:steps][mask] > 0))
    host_steps = int(mask.sum()) * n_hosts
    runs = positive_runs(labels, steps, mask)
    attribution = attribute_runs_to_events(runs, stream["host_cascade_event"],
                                           mask)
    worst_cascade = max(attribution.values()) if attribution else 0
    worst_run = max((run["duration"] for run in runs), default=0)
    deployment_attempts = int(np.asarray(
        stream["deploy_attempts"])[:steps][mask].sum())
    deployment_rejected = int(np.asarray(
        stream["deploy_rejected"])[:steps][mask].sum())
    migration_attempts = int(np.asarray(
        stream["migrate_attempts"])[:steps][mask].sum())
    migration_rejected = int(np.asarray(
        stream["migrate_rejected"])[:steps][mask].sum())

    gate_stats = {
        "prevalence": (float(positives) / float(host_steps)
                       if host_steps else None),
        "positive_hoststeps": positives,
        "host_steps": host_steps,
        "independent_fault_events": len(runs),
        "valid_onset_followup": len(valid_events),
        "deployment_attempts": deployment_attempts,
        "deployment_rejected": deployment_rejected,
        "deployment_rejection_rate": (float(deployment_rejected)
                                      / float(deployment_attempts)
                                      if deployment_attempts else None),
        "migration_attempts": migration_attempts,
        "migration_rejected": migration_rejected,
        "migration_rejection_rate": (float(migration_rejected)
                                     / float(migration_attempts)
                                     if migration_attempts else None),
        "worst_event_share": (float(worst_cascade) / float(positives)
                              if positives else 0.0),
        "event_count": len(events),
        "mean_duration": durations["mean"],
        "peak_ratio": peaks["p50"],
    }
    cohort = [cid for cid, value in task_map.items() if value == canonical]
    return {
        "canonical_regime_id": canonical,
        "regime_id": table_entry["regime_id"],
        "mechanism_id": table_entry["mechanism_id"],
        "onset_resource": resource,
        "onset_tau": tau,
        "onset_provisional": bool(table_entry["provisional"]),
        "response_windows": {name: list(window)
                             for name, window in windows.items()},
        "sequence": table_entry["sequence"],
        "phase_window": {
            "intervals": int(mask.sum()),
            "ranges": ranges_of(mask),
            "host_steps": host_steps,
        },
        "cohort": {
            "n_tasks": len(cohort),
            "creation_ids_sample": sorted(cohort)[:10],
            "rule": ("tasks whose saved per-observation mechanism id is the "
                     "regime's own; a familiar task and another regime's task "
                     "are never in this cohort"),
        },
        "gate_stats": gate_stats,
        "onsets": {
            "n_events": len(events),
            "n_valid_followup": len(valid_events),
            "n_invalid_followup": len(events) - len(valid_events),
            "n_events_with_a_usable_followup": usable_followup,
            "followup_horizon": horizon,
            "followup_rule": ("the task observes every interval of "
                              "[t0, t0 + max(registered window))"),
            "usable_followup_rule": ("at least one interval inside a registered "
                                     "response window is observed, so a response "
                                     "value exists (the P22 U4-v2 "
                                     "MIN_FOLLOWUP_EVENTS definition); reported "
                                     "for comparison, NOT the gated number"),
            "gated_definition": ("n_valid_followup (full window) -- plan §9 "
                                 "'valid onset/follow-up'; the P22-compatible "
                                 "usable count is reported next to it and the "
                                 "definition is never switched after seeing the "
                                 "number"),
            "event_keys": [event["event_key"] for event in events][:50],
            "cohort_observed_life": _summarize(
                [float(timeline[cid]["t"].size) for cid in cohort]),
            "cohort_observed_life_note": (
                "observed intervals per cohort task; a median well below the "
                "followup_horizon is why the full-window count can be far below "
                "the usable count"),
            "n_events_with_a_registered_envelope": int(sum(
                1 for event in events if event["audit_event_id"] >= 0)),
            "n_events_without_a_registered_envelope": int(sum(
                1 for event in events if event["audit_event_id"] < 0)),
            "value_at_onset": _summarize([event["value_at_onset"]
                                          for event in events]),
            "delta_at_onset": _summarize([event["delta_at_onset"]
                                          for event in events]),
            "baseline_sources": _counts(event["baseline_source"]
                                        for event in events),
        },
        "cross_regime_confound": {
            "onsets_with_task_map": len(events),
            "onsets_without_task_map": len(blind),
            "cross_hits": len(blind) - len(events),
            "note": ("onsets an unmapped resource+tau scan attributes to this "
                     "regime that its own task cohort does not produce; the "
                     "instrument's tests pin that a regime-A envelope crosses "
                     "the B/C thresholds, so this number is reported and is "
                     "NEVER a gate input"),
        },
        "durations": {
            "n": durations["n"], "mean": durations["mean"],
            "p50": durations["p50"], "max": durations["max"],
            "mean_valid_followup_only": durations_valid["mean"],
            "n_valid_followup_only": durations_valid["n"],
            "definition": durations["definition"],
        },
        "peak_ratio": {
            "n": peaks["n"], "mean": peaks["mean"], "p50": peaks["p50"],
            "max": peaks["max"], "definition": peaks["definition"],
        },
        "alignment": alignment,
        "fault_runs": {
            "n_runs": len(runs),
            "worst_run_share_of_positives": (
                float(worst_run) / float(positives) if positives else None),
            "worst_cascade_event_share_of_positives": (
                float(worst_cascade) / float(positives) if positives else 0.0),
            "n_distinct_cascade_events_attributed": len(attribution),
            "definition": ("per-host contiguous positive-label runs inside the "
                           "regime's phase window; a run is attributed to every "
                           "registered cascade event live and in-window on its "
                           "host"),
        },
        "rejection": {
            "deployment_attempts": deployment_attempts,
            "deployment_rejected": deployment_rejected,
            "migration_attempts": migration_attempts,
            "migration_rejected": migration_rejected,
            "note": ("a rate with zero attempts is reported as null (not "
                     "measurable) rather than as a fake 0.0, and the count is "
                     "reported next to it"),
        },
    }


def _summarize(values):
    array = np.asarray([value for value in values
                        if value is not None and np.isfinite(value)],
                       dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "p50": None, "max": None}
    return {"n": int(array.size), "mean": float(array.mean()),
            "p50": float(np.median(array)), "max": float(array.max())}


def _counts(values):
    out = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def measure_declared(loaded):
    """The stream's own declared numbers, as claims to be falsified.

    Read only AFTER the recomputation and never used as an input; the comparison
    is reported in the artifact and raises a warning when it disagrees.
    """
    path = Path(loaded["stream_dir"]) / "unseen_data_audit.json"
    if not path.is_file():
        return {"present": False, "path": str(path), "note":
                "the collector's audit file is absent; nothing to compare "
                "(it was never an input)"}
    try:
        declared = json.loads(path.read_text(encoding="utf8"))
    except ValueError as exc:
        return {"present": True, "path": str(path), "error":
                "unparseable: %s" % exc}
    return {"present": True, "path": str(path), "declared": declared}


def measure_stream(stream_dir):
    """Recompute everything one stream has to say about the gate."""
    loaded = load_stream(stream_dir)
    task_map = task_regime_map(loaded)
    stream = loaded["stream"]
    steps = loaded["steps"]
    table = registered_table()["table"]
    phase_regime = np.asarray(stream["phase_regime_ids"])[:steps]
    masks, present = {}, []
    for gen_id in REGISTERED_REGIME_IDS:
        canonical = core23.canonical_regime_id(gen_id)
        mask = phase_regime == int(REGISTERED_MECHANISM_IDS[gen_id])
        masks[canonical] = mask
        if bool(mask.any()):
            present.append(canonical)
    familiar = measure_familiar(loaded)
    per_regime, confound = {}, {}
    for canonical in present:
        block = measure_regime(loaded, task_map, canonical, table[canonical],
                               masks[canonical])
        block["familiar"] = {
            "measured_familiar_task_maximum": familiar
            ["measured_familiar_task_maximum"],
            "envelope_free_familiar_task_maximum": familiar
            ["envelope_free_familiar_task_maximum"],
        }
        per_regime[canonical] = block
    for gen_id in REGISTERED_REGIME_IDS:
        canonical = core23.canonical_regime_id(gen_id)
        entry = table[canonical]
        windows = {name: tuple(window) for name, window
                   in entry["response_windows"].items()}
        blind = core23.onset_events_multi(
            loaded["timeline"], entry["onset_resource"], float(entry["onset_tau"]),
            response_windows=windows, regime_map={entry["onset_resource"]:
                                                  canonical})
        mapped = per_regime.get(canonical, {}).get("onsets", {}).get("n_events")
        confound[canonical] = {
            "regime_id": gen_id,
            "onset_resource": entry["onset_resource"],
            "onset_tau": float(entry["onset_tau"]),
            "registered_phase_in_stream": bool(canonical in present),
            "onsets_without_task_map": len(blind),
            "onsets_with_task_map": mapped,
            "cross_hits": (None if mapped is None else len(blind) - mapped),
        }
    live = np.asarray(stream["after_creation_ids"])[:steps] >= 0
    live_t, live_s = np.nonzero(live)
    saved_t = np.asarray(loaded["saved"]["time"])
    saved_s = np.asarray(loaded["saved"]["slot_index"])
    rows_match = bool(live_t.size == saved_t.size
                      and np.array_equal(live_t, saved_t)
                      and np.array_equal(live_s, saved_s))
    integrity = core23.timeline_integrity(loaded["timeline"])
    manifest = loaded["manifest"]
    declared_phases = manifest.get("phases")
    phase_report = {}
    for canonical in present:
        phase_report[canonical] = {
            "intervals": int(masks[canonical].sum()),
            "ranges": ranges_of(masks[canonical]),
        }
    return {
        "tag": Path(stream_dir).name,
        "stream_dir": str(Path(stream_dir)),
        "mode": manifest.get("mode"),
        "declared_regime": manifest.get("regime"),
        "steps": steps,
        "rows": loaded["rows"],
        "guard_rows": int(loaded["rows"] - steps),
        "n_hosts": loaded["n_hosts"],
        "n_slots": loaded["n_slots"],
        "stream_sha256": loaded["stream_sha256"],
        "task_timeline_sha256": loaded["task_timeline_sha256"],
        "manifest_steps": int(manifest.get("steps")),
        "phase_structure": {
            "familiar_intervals": familiar["familiar_intervals"],
            "familiar_ranges": familiar["familiar_ranges"],
            "regime_windows": phase_report,
            "regimes_present": sorted(present),
            "declared_phases": declared_phases,
            "declared_phases_present": bool(declared_phases),
        },
        "timeline": {
            "n_tasks": int(integrity["n_tasks"]),
            "n_observations": int(integrity["n_observations"]),
            "reused_slots": int(integrity["reused_slots"]),
            "multi_slot_tasks": int(integrity["multi_slot_tasks"]),
            "problems": integrity["problems"][:10],
            "ok": bool(integrity["ok"]),
            "live_cells": int(live_t.size),
            "rows_match_the_stream_live_cells": rows_match,
        },
        "task_regime_map": {
            "source": loaded["regime_source"],
            "n_tasks": len(task_map),
            "tasks_per_regime": _counts(task_map.values()),
        },
        "familiar": familiar,
        "per_regime": per_regime,
        "cross_regime_confound": confound,
        "declared": measure_declared(loaded),
    }


# --------------------------------------------------------------------------
# the registered gate (plan §9) -- analyzer-side re-derivation
# --------------------------------------------------------------------------
def plan_section_9_checks(per_regime_stats, thresholds=None):
    """The eight registered plan §9 checks, re-derived from the literal table.

    Deliberately the same check names and the same comparison operators as
    ``core23.marginal_match_report``: the two must agree key by key, and a
    disagreement (a drifted threshold, a different None policy) is itself a
    failed check.  Returns ``(checks, prevalence_spread)``.
    """
    table = dict(GATE_THRESHOLDS if thresholds is None else thresholds)
    checks = {}
    for regime_id in sorted(per_regime_stats):
        stats = per_regime_stats[regime_id]
        prevalence = stats.get("prevalence")
        checks["%s_prevalence_in_range" % regime_id] = bool(
            prevalence is not None
            and table["prevalence_min"] <= prevalence
            <= table["prevalence_max"])
        events = stats.get("independent_fault_events")
        checks["%s_independent_fault_events_enough" % regime_id] = bool(
            events is not None
            and int(events) >= int(table["independent_fault_events_min"]))
        valid = stats.get("valid_onset_followup")
        checks["%s_valid_onset_followup_enough" % regime_id] = bool(
            valid is not None
            and int(valid) >= int(table["valid_onset_followup_min"]))
        deployment = stats.get("deployment_rejection_rate")
        checks["%s_deployment_rejection_ok" % regime_id] = bool(
            deployment is not None
            and float(deployment) <= table["deployment_rejection_max"])
        migration = stats.get("migration_rejection_rate")
        checks["%s_migration_rejection_ok" % regime_id] = bool(
            migration is not None
            and float(migration) <= table["migration_rejection_max"])
        worst = stats.get("worst_event_share")
        checks["%s_worst_event_share_ok" % regime_id] = bool(
            worst is not None
            and float(worst) < table["worst_event_share_max"])
    prevalences = [stats["prevalence"] for stats in per_regime_stats.values()
                   if stats.get("prevalence") is not None]
    spread = ((max(prevalences) - min(prevalences))
              if len(prevalences) >= 2 else None)
    checks["prevalence_spread_ok"] = bool(
        spread is not None and spread <= table["prevalence_spread_max"])
    checks["all_registered_regimes_reported"] = bool(
        sorted(per_regime_stats) == sorted(core23.REGIME_IDS))
    return checks, spread


def analyze_declared(streams, dev):
    """Declared collector numbers vs the recomputation (never an input)."""
    per_stream, disagreements = {}, []
    for canonical in sorted(streams):
        block = streams[canonical]["declared"]
        entry = {"stream": streams[canonical]["tag"],
                 "audit_file_present": bool(block.get("present"))}
        declared = block.get("declared") or {}
        block_declared = (declared.get("per_regime") or {}).get(canonical)
        if block_declared is None:
            entry["comparison"] = None
            entry["note"] = ("the collector's audit file carries no per_regime "
                             "block for this regime; nothing to compare")
        else:
            metrics = {}
            recomputed = streams[canonical]["per_regime"][canonical]["gate_stats"]
            declared_metrics = (block_declared.get("gate") or {}).get("metrics") \
                or {}
            for key in ("prevalence", "positive_hoststeps", "host_steps",
                        "independent_fault_events", "deployment_attempts",
                        "deployment_rejected", "deployment_rejection_rate",
                        "migration_attempts", "migration_rejected",
                        "migration_rejection_rate",
                        "worst_event_share_of_positives"):
                left = declared_metrics.get(key)
                right = recomputed.get(key)
                if key == "worst_event_share_of_positives":
                    right = recomputed.get("worst_event_share")
                match = bool(
                    (left is None and right is None)
                    or (left is not None and right is not None
                        and abs(float(left) - float(right)) <= 1e-9))
                metrics[key] = {"declared": left, "recomputed": right,
                                "match": match}
                if not match:
                    disagreements.append({
                        "stream": streams[canonical]["tag"],
                        "regime_id": canonical, "metric": key,
                        "declared": left, "recomputed": right})
            declared_onsets = ((block_declared.get("task_level") or {})
                               .get("n_onsets"))
            recomputed_onsets = streams[canonical]["per_regime"][canonical][
                "onsets"]["n_events"]
            match = bool(declared_onsets is not None
                         and int(declared_onsets) == int(recomputed_onsets))
            metrics["n_onsets"] = {"declared": declared_onsets,
                                   "recomputed": recomputed_onsets,
                                   "match": match}
            if not match:
                disagreements.append({
                    "stream": streams[canonical]["tag"], "regime_id": canonical,
                    "metric": "n_onsets", "declared": declared_onsets,
                    "recomputed": recomputed_onsets})
            entry["comparison"] = metrics
            entry["all_match"] = all(info["match"]
                                     for info in metrics.values())
        per_stream[canonical] = entry
    if dev is not None:
        per_stream["dev"] = {"stream": dev["tag"],
                             "audit_file_present": bool(
                                 dev["declared"].get("present")),
                             "comparison": None,
                             "note": ("the development stream is context, not a "
                                      "gate source; its declared numbers are "
                                      "not compared metric by metric here")}
    return {"per_stream": per_stream, "disagreements": disagreements,
            "agree": not disagreements,
            "note": ("the collector's audit file is read AFTER the "
                     "recomputation and is only ever compared against it; a "
                     "disagreement is a warning about the collector (or a stale "
                     "file), never a gate input")}


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------
def integrity_report(streams, dev):
    """Structural checks that make the recomputation meaningful."""
    checks, details = {}, {}
    for name, block in sorted(streams.items()):
        checks["%s_timeline_integrity_ok" % name] = bool(
            block["timeline"]["ok"])
        checks["%s_timeline_matches_the_stream_live_cells" % name] = bool(
            block["timeline"]["rows_match_the_stream_live_cells"])
        checks["%s_guard_rows_registered" % name] = bool(
            block["guard_rows"] == 1)
        checks["%s_regime_window_present" % name] = bool(
            block["per_regime"].get(name) is not None)
        details[name] = {
            "n_tasks": block["timeline"]["n_tasks"],
            "n_observations": block["timeline"]["n_observations"],
            "problems": block["timeline"]["problems"],
            "rows_match_the_stream_live_cells":
                block["timeline"]["rows_match_the_stream_live_cells"],
        }
    if dev is not None:
        checks["dev_timeline_integrity_ok"] = bool(dev["timeline"]["ok"])
        checks["dev_timeline_matches_the_stream_live_cells"] = bool(
            dev["timeline"]["rows_match_the_stream_live_cells"])
        for canonical in sorted(dev["per_regime"]):
            checks["dev_%s_regime_window_present" % canonical] = True
        details["dev"] = {"n_tasks": dev["timeline"]["n_tasks"],
                          "problems": dev["timeline"]["problems"]}
    for name, block in sorted(streams.items()):
        for canonical in sorted(block["per_regime"]):
            checks["%s_%s_cohort_nonempty" % (name, canonical)] = bool(
                block["per_regime"][canonical]["cohort"]["n_tasks"] > 0)
    return {"checks": checks, "passed": all(checks.values()),
            "details": details}


def analyse(streams_root, only=None, thresholds=None):
    """Full analysis (no writes): every number the gate and artifacts need."""
    canonical_only = resolve_only(only) if only is not None else None
    resolved = resolve_streams(streams_root, only=canonical_only)
    partial = canonical_only is not None
    generator = generator_registration()
    registration = registered_table()
    table = registration["table"]
    drift = threshold_drift_report(thresholds)

    streams = {}
    for canonical in sorted(resolved["single"]):
        streams[canonical] = measure_stream(resolved["single"][canonical])
    dev = measure_stream(resolved["dev"]) if resolved["dev"] is not None else None

    gate_stats, per_regime, admissibility = {}, {}, {}
    for canonical in sorted(streams):
        block = streams[canonical]["per_regime"].get(canonical)
        if block is None:
            raise AnalyzerError(
                "the stream registered for regime %s (%s) declares no %s phase "
                "(regimes present: %s); refusing to fabricate a gate block from "
                "another regime's window"
                % (canonical, streams[canonical]["stream_dir"], canonical,
                   streams[canonical]["phase_structure"]["regimes_present"]))
        gate_stats[canonical] = block["gate_stats"]
        admissibility[canonical] = admissibility_report(
            canonical, table[canonical], streams[canonical]["familiar"])
        per_regime[canonical] = dict(
            block,
            gate_source_stream=streams[canonical]["tag"],
            admissibility=admissibility[canonical],
            dev_context=(dev["per_regime"].get(canonical) if dev else None),
        )

    dev_admissibility = {}
    if dev is not None:
        for canonical in sorted(dev["per_regime"]):
            dev_admissibility[canonical] = admissibility_report(
                canonical, table[canonical], dev["familiar"])

    marginal = core23.marginal_match_report(gate_stats)
    contrast = core23.regime_contrast_report(gate_stats)
    checks, spread = plan_section_9_checks(gate_stats, drift["analyzer"])
    agreement = {"keys": {}, "mismatched": []}
    for key in sorted(set(checks) | set(marginal["checks"])):
        left = checks.get(key)
        right = marginal["checks"].get(key)
        agreement["keys"][key] = bool(left == right)
        if left != right:
            agreement["mismatched"].append(
                {"check": key, "analyzer": left, "instrument": right})
    agreement["passed"] = not agreement["mismatched"]

    integrity = integrity_report(streams, dev)
    declared = analyze_declared(streams, dev)

    not_evaluated = ([key for key in CROSS_REGIME_CHECKS if key in checks]
                     if partial else [])
    evaluated = {key: value for key, value in checks.items()
                 if key not in not_evaluated}
    gate_passed = bool(all(evaluated.values())
                       and agreement["passed"] and drift["passed"])
    followup_summary = {
        canonical: {
            "n_events": per_regime[canonical]["onsets"]["n_events"],
            "n_valid_followup_full_window_gated":
                per_regime[canonical]["onsets"]["n_valid_followup"],
            "n_usable_followup_p22_definition":
                per_regime[canonical]["onsets"]["n_events_with_a_usable_followup"],
            "followup_horizon":
                per_regime[canonical]["onsets"]["followup_horizon"],
            "cohort_observed_life":
                per_regime[canonical]["onsets"]["cohort_observed_life"],
        } for canonical in sorted(per_regime)}
    inadmissible = sorted(
        canonical for canonical, entry in admissibility.items()
        if entry["threshold_admissible"] is not True)
    passed = bool(gate_passed and not inadmissible and integrity["passed"])
    failed_on = []
    if not gate_passed:
        failed_on.append("data_gate")
    if inadmissible:
        failed_on.append("threshold_admissibility")
    if not integrity["passed"]:
        failed_on.append("stream_integrity")

    stop_conditions = []
    if not gate_passed:
        stop_conditions.append(
            "STOP-DATA: the plan §9 Data Gate did not pass (%s); the scenes are "
            "not matched, so no method comparison is admissible"
            % ", ".join(sorted(key for key, value in evaluated.items()
                               if not value)))
    if inadmissible:
        stop_conditions.append(
            "STOP-THRESHOLD: the registered onset threshold of %s is not "
            "strictly above the measured familiar per-task maximum; the event "
            "counts for those regimes may be familiar tasks crossing a "
            "too-low threshold, so the gate numbers above are not usable"
            % ", ".join(inadmissible))
    if not integrity["passed"]:
        stop_conditions.append(
            "STOP-INTEGRITY: %s" % ", ".join(sorted(
                key for key, value in integrity["checks"].items() if not value)))
    warnings = []
    if resolved.get("dev_missing"):
        warnings.append(
            "WARN-NO-DEV-STREAM: the multi-regime development stream is not "
            "readable under %s (%s: %s).  The per-regime gate blocks and the "
            "marginal-match gate are unaffected -- they are measured on the "
            "three single-regime cohorts -- but the recurrence/switch context "
            "of plan §13 is missing from this verdict, so collect %s before "
            "using it as the registered S2 record."
            % (resolved["root"], REGISTERED_DEV_TAG,
               resolved["dev_missing"]["reason"],
               resolved["dev_missing"]["expected_path"]))
    for canonical in sorted(per_regime):
        onsets = per_regime[canonical]["onsets"]
        minimum = int(GATE_THRESHOLDS["valid_onset_followup_min"])
        if (onsets["n_valid_followup"] < minimum
                <= onsets["n_events_with_a_usable_followup"]):
            warnings.append(
                "WARN-FOLLOWUP-DEFINITION: regime %s has %d onset(s) with a "
                "FULL follow-up window and %d with a usable follow-up (at least "
                "one observed interval inside a registered response window; the "
                "P22 U4-v2 definition).  The gate applies the >=%d threshold to "
                "the full-window count, so the check fails; under the "
                "P22-compatible definition it would pass.  The analyzer does "
                "not switch definitions after seeing the number."
                % (canonical, onsets["n_valid_followup"],
                   onsets["n_events_with_a_usable_followup"], minimum))
    if not declared["agree"]:
        warnings.append(
            "WARN-DECLARED-AUDIT: the collector's own audit numbers disagree "
            "with the recomputation for %d metric(s) (%s); the recomputed "
            "numbers above are the gate inputs"
            % (len(declared["disagreements"]),
               ", ".join(sorted({item["stream"] for item
                                 in declared["disagreements"]}))))
    if not drift["passed"]:
        warnings.append(
            "WARN-THRESHOLD-DRIFT: the analyzer's plan §9 table and the "
            "instrument's registered table differ (%s)" % drift["drift"])
    if not registration["tau_table_matches_instrument"]:
        warnings.append(
            "WARN-TAU-DRIFT: the generator's REGISTERED_ONSET_TAU and the "
            "instrument's registered taus differ (%s)"
            % registration["tau_drift"])
    if not generator["available"]:
        warnings.append(
            "WARN-GENERATOR-UNIMPORTABLE: %s (%s); the registered taus fall back "
            "to the instrument's table"
            % (generator["module"], generator["error"]))

    return {
        "protocol": PROTOCOL, "stage": STAGE, "analyzer": ANALYZER,
        "generated_at": utcnow(),
        "streams_root": str(resolved["root"]),
        "partial": partial,
        "only": canonical_only,
        "requested_only": only,
        "plan_section": PLAN_SECTION,
        "plan_section_9_text": PLAN_SECTION_9_TEXT,
        "thresholds": drift["analyzer"],
        "threshold_drift": drift,
        "registration": {
            "table": table,
            "tau_drift": registration["tau_drift"],
            "tau_table_matches_instrument":
                registration["tau_table_matches_instrument"],
            "generator": registration["generator"],
            "provisional_taus": dict(core23.PROVISIONAL_TAUS),
            "provisional_note": (
                "the RAM (%.1f) and disk (%.1f) onset thresholds are "
                "PROVISIONAL (plan §6-§7; the generator marks all three as "
                "provisional): the 'no familiar task can cross it' half of the "
                "registered argument is measured here on this stream's familiar "
                "rows and reported, never assumed"
                % (core23.PROVISIONAL_TAUS.get("ram", float("nan")),
                   core23.PROVISIONAL_TAUS.get("disk", float("nan")))),
        },
        "streams": {canonical: {
            "tag": streams[canonical]["tag"],
            "stream_dir": streams[canonical]["stream_dir"],
            "discovery": resolved["sources"].get(canonical),
            "steps": streams[canonical]["steps"],
            "mode": streams[canonical]["mode"],
            "declared_regime": streams[canonical]["declared_regime"],
            "stream_sha256": streams[canonical]["stream_sha256"],
            "task_timeline_sha256":
                streams[canonical]["task_timeline_sha256"],
            "regimes_present":
                streams[canonical]["phase_structure"]["regimes_present"],
            "gate_role": "gate source (per-regime cohort)",
        } for canonical in sorted(streams)},
        "dev_stream": (None if dev is None else {
            "tag": dev["tag"], "stream_dir": dev["stream_dir"],
            "discovery": resolved["sources"].get("dev"),
            "steps": dev["steps"], "mode": dev["mode"],
            "stream_sha256": dev["stream_sha256"],
            "regimes_present": dev["phase_structure"]["regimes_present"],
            "gate_role": "multi-regime context (never a gate input)",
        }),
        "dev_stream_missing": resolved.get("dev_missing"),
        "evaluated_regimes": sorted(streams),
        "per_regime": per_regime,
        "admissibility": admissibility,
        "dev_admissibility": dev_admissibility,
        "familiar_measurements": {
            canonical: streams[canonical]["familiar"]
            for canonical in sorted(streams)},
        "dev_familiar_measurement": (None if dev is None else dev["familiar"]),
        "measurements": {canonical: streams[canonical]
                         for canonical in sorted(streams)},
        "dev_measurement": dev,
        "gate_stats": gate_stats,
        "followup": followup_summary,
        "gate_checks": marginal["checks"],
        "analyzer_checks": checks,
        "analyzer_matches_the_instrument": agreement,
        "marginal_match": marginal,
        "regime_contrast": contrast,
        "prevalence_spread": spread,
        "gate_passed": gate_passed,
        "checks_not_evaluated": not_evaluated,
        "thresholds_admissible": not inadmissible,
        "inadmissible_thresholds": inadmissible,
        "integrity": integrity,
        "declared_vs_recomputed": declared,
        "stop_conditions": stop_conditions,
        "warnings": warnings,
        "passed": passed,
        "failed_on": failed_on,
        "definitions": {
            "gate_source": (
                "each regime's gate block is measured on its own single-regime "
                "stream (the registered clean per-regime cohort); the "
                "development stream is reported as context and never enters the "
                "gate"),
            "gate": PLAN_SECTION_9_TEXT,
            "marginal_matching": (
                "core23.marginal_match_report over the eight registered numbers; "
                "core23.regime_contrast_report is reported next to it as "
                "context -- its tolerances are instrument-local and are not the "
                "registered gate"),
            "recomputation": (
                "every gate quantity is recomputed from stream.npz + "
                "task_timeline.npz with ftmoe_protocol023_core; the collector's "
                "unseen_data_audit.json is only ever compared against the "
                "recomputation"),
            "cohort": (
                "onsets are scanned with task_regime_map, so a regime-A "
                "envelope whose RAM/Disk responses cross the B/C thresholds is "
                "never counted as a B/C event; the unmapped scan is reported as "
                "the cross-hit confound count"),
            "independent_fault_events": (
                "per-host contiguous positive-label runs inside the regime's "
                "phase window (the registered P22 segmentation rule, "
                "reimplemented here)"),
            "valid_onset_followup": (
                "onsets whose task observes every interval of [t0, t0 + max "
                "registered response window)) -- the plan §9 'valid "
                "onset/follow-up' the ≥50 threshold is applied to; the "
                "P22-compatible 'usable follow-up' count (at least one observed "
                "interval inside a window) is reported next to it and is NOT "
                "gated"),
            "worst_event_share": (
                "positive host-steps attributed to the single worst registered "
                "cascade event, over the window's positives (the collector's "
                "definition); the label-free worst-run share is reported next to "
                "it"),
            "mean_duration": (
                "core23.response_durations over the regime's onset cohort: "
                "intervals inside the registered response windows whose "
                "deviation from the pre-onset baseline is > 0, longest of the "
                "two response resources"),
            "peak_ratio": (
                "core23.peak_ratio_summary over the same cohort; the registered "
                "matched marginal uses the MEDIAN peak ratio (robust to one "
                "large excursion) and the mean is reported next to it"),
            "admissibility": (
                "a registered onset threshold is admissible only if it is "
                "strictly above the measured familiar per-task maximum of its "
                "resource; a failure is reported with its negative margin and "
                "never retuned"),
            "provisional_thresholds": (
                "RAM %.1f and disk %.1f are PROVISIONAL"
                % (core23.PROVISIONAL_TAUS.get("ram", float("nan")),
                   core23.PROVISIONAL_TAUS.get("disk", float("nan")))),
        },
    }


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
def compact_verdict(verdict):
    """The one-line verdict a runner reads: gate booleans + overall PASS/FAIL."""
    return {
        "protocol": verdict["protocol"], "stage": verdict["stage"],
        "analyzer": verdict["analyzer"], "generated_at": verdict["generated_at"],
        "partial": verdict["partial"], "only": verdict["only"],
        "gate_passed": verdict["gate_passed"],
        "thresholds_admissible": verdict["thresholds_admissible"],
        "inadmissible_thresholds": verdict["inadmissible_thresholds"],
        "integrity_ok": verdict["integrity"]["passed"],
        "checks": verdict["gate_checks"],
        "checks_not_evaluated": verdict["checks_not_evaluated"],
        "per_regime": {
            canonical: {
                "prevalence": entry["gate_stats"]["prevalence"],
                "independent_fault_events":
                    entry["gate_stats"]["independent_fault_events"],
                "valid_onset_followup":
                    entry["gate_stats"]["valid_onset_followup"],
                "n_events_with_a_usable_followup":
                    entry["onsets"]["n_events_with_a_usable_followup"],
                "worst_event_share": entry["gate_stats"]["worst_event_share"],
                "onset_tau": entry["onset_tau"],
                "onset_provisional": entry["onset_provisional"],
                "threshold_admissible":
                    entry["admissibility"]["threshold_admissible"],
            } for canonical, entry in sorted(verdict["per_regime"].items())},
        "passed": verdict["passed"],
        "failed_on": verdict["failed_on"],
        "stop_conditions": verdict["stop_conditions"],
        "artifacts": verdict.get("artifacts") or {},
    }


def regime_document(verdict, canonical):
    """One self-contained per-regime artifact (``regime_<A|B|C>.json``)."""
    block = verdict["per_regime"][canonical]
    dev_block = block.get("dev_context")
    return {
        "protocol": PROTOCOL, "stage": STAGE, "analyzer": ANALYZER,
        "generated_at": verdict["generated_at"],
        "artifact": "regime_%s.json" % canonical,
        "canonical_regime_id": canonical,
        "regime_id": block["regime_id"],
        "partial_run": verdict["partial"],
        "plan_section": PLAN_SECTION,
        "plan_section_9_text": PLAN_SECTION_9_TEXT,
        "thresholds": verdict["thresholds"],
        "threshold_drift": verdict["threshold_drift"],
        "gate_source_stream": {
            "tag": block["gate_source_stream"],
            "stream_dir": verdict["streams"][canonical]["stream_dir"],
            "discovery": verdict["streams"][canonical]["discovery"],
            "stream_sha256": verdict["streams"][canonical]["stream_sha256"],
            "task_timeline_sha256":
                verdict["streams"][canonical]["task_timeline_sha256"],
            "steps": verdict["streams"][canonical]["steps"],
        },
        "registration": {
            "onset_resource": block["onset_resource"],
            "onset_tau": block["onset_tau"],
            "onset_provisional": block["onset_provisional"],
            "tau_source": block["admissibility"]["tau_source"],
            "registered_floor": block["admissibility"]["registered_floor"],
            "registered_familiar_clip":
                block["admissibility"]["registered_familiar_clip"],
            "response_windows": block["response_windows"],
            "sequence": block["sequence"],
            "provisional_note": verdict["registration"]["provisional_note"],
        },
        "measurement": {key: block[key] for key in
                        ("phase_window", "cohort", "gate_stats", "onsets",
                         "cross_regime_confound", "durations", "peak_ratio",
                         "alignment", "fault_runs", "rejection", "familiar")},
        "admissibility": block["admissibility"],
        "familiar_measurement": verdict["familiar_measurements"][canonical],
        "checks": {key: value for key, value in
                   verdict["analyzer_checks"].items()
                   if key.startswith("%s_" % canonical)},
        "instrument_checks": {key: value for key, value in
                              verdict["gate_checks"].items()
                              if key.startswith("%s_" % canonical)},
        "dev_stream_context": dev_block,
        "dev_admissibility": verdict["dev_admissibility"].get(canonical),
        "declared_vs_recomputed":
            verdict["declared_vs_recomputed"]["per_stream"].get(canonical),
        "gate_passed": verdict["gate_passed"],
        "passed": verdict["passed"],
        "failed_on": verdict["failed_on"],
        "stop_conditions": verdict["stop_conditions"],
        "warnings": verdict["warnings"],
        "definitions": verdict["definitions"],
    }


def marginal_document(verdict):
    """``marginal_match.json`` -- both instrument reports, stored in full."""
    return {
        "protocol": PROTOCOL, "stage": STAGE, "analyzer": ANALYZER,
        "generated_at": verdict["generated_at"],
        "artifact": "marginal_match.json",
        "partial_run": verdict["partial"],
        "only": verdict["only"],
        "plan_section": PLAN_SECTION,
        "plan_section_9_text": PLAN_SECTION_9_TEXT,
        "thresholds": verdict["thresholds"],
        "threshold_drift": verdict["threshold_drift"],
        "gate_stats": verdict["gate_stats"],
        "gate_source_streams": {
            canonical: verdict["streams"][canonical]["stream_dir"]
            for canonical in sorted(verdict["streams"])},
        "marginal_match_report": verdict["marginal_match"],
        "regime_contrast_report": verdict["regime_contrast"],
        "analyzer_checks": verdict["analyzer_checks"],
        "analyzer_matches_the_instrument":
            verdict["analyzer_matches_the_instrument"],
        "prevalence_spread": verdict["prevalence_spread"],
        "checks_not_evaluated": verdict["checks_not_evaluated"],
        "threshold_admissibility": verdict["admissibility"],
        "inadmissible_thresholds": verdict["inadmissible_thresholds"],
        "integrity": verdict["integrity"]["checks"],
        "declared_vs_recomputed": verdict["declared_vs_recomputed"],
        "gate_passed": verdict["gate_passed"],
        "passed": verdict["passed"],
        "failed_on": verdict["failed_on"],
        "stop_conditions": verdict["stop_conditions"],
        "warnings": verdict["warnings"],
        "definitions": verdict["definitions"],
    }


def _fmt(value, digits=4, percent=False):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        if percent:
            return ("%.2f%%" % (100.0 * value))
        return ("%.*f" % (digits, value))
    return str(value)


def _mark(ok):
    if ok is None:
        return "NOT EVALUATED"
    return "PASS" if ok else "FAIL"


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def data_gate_markdown(verdict):
    """``DATA_GATE.md`` -- the human-readable summary table."""
    regimes = sorted(verdict["per_regime"])
    lines = []
    lines.append("# Protocol 023 -- S2 Data Gate (plan §9)")
    lines.append("")
    lines.append("- analyzer: `%s` (stage %s)" % (verdict["analyzer"],
                                                  verdict["stage"]))
    lines.append("- generated_at: %s" % verdict["generated_at"])
    lines.append("- streams_root: `%s`" % verdict["streams_root"])
    lines.append("- plan: `%s`" % verdict["plan_section"])
    lines.append("- registered gate: %s" % verdict["plan_section_9_text"])
    if verdict["partial"]:
        lines.append("- **PARTIAL RUN** (`--only %s`): the cross-regime checks "
                     "are not evaluated; only one regime was analysed"
                     % verdict["only"])
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- data gate (`marginal_match_report` + the analyzer's own "
                 "re-derivation of the same eight checks): **%s**"
                 % _mark(verdict["gate_passed"]))
    lines.append("- onset thresholds admissible: **%s**%s"
                 % (_mark(verdict["thresholds_admissible"]),
                    ("" if verdict["thresholds_admissible"] else
                     " (inadmissible: %s)" % ", ".join(
                         verdict["inadmissible_thresholds"]))))
    lines.append("- stream integrity: **%s**"
                 % _mark(verdict["integrity"]["passed"]))
    lines.append("- **overall: %s**%s"
                 % ("PASS" if verdict["passed"] else "FAIL",
                    ("" if not verdict["failed_on"] else
                     " (failed_on: %s)" % ", ".join(verdict["failed_on"]))))
    lines.append("- analyzer checks reproduce the instrument's checks: **%s**"
                 % _mark(verdict["analyzer_matches_the_instrument"]["passed"]))
    lines.append("- plan §9 table matches the instrument's registered table: "
                 "**%s**" % _mark(verdict["threshold_drift"]["passed"]))
    lines.append("- development stream (multi-regime context): %s"
                 % ("**read** (%s)" % verdict["dev_stream"]["tag"]
                    if verdict["dev_stream"] is not None else
                    "**NOT FOUND** -- the gate above is unaffected (it is "
                    "measured per regime on the single-regime cohorts) but the "
                    "plan §13 recurrence/switch context is missing"))
    lines.append("")
    lines.append("## Registered thresholds vs the measurement (per regime)")
    lines.append("")
    rows = []
    for canonical in regimes:
        entry = verdict["per_regime"][canonical]
        stats = entry["gate_stats"]
        checks = verdict["analyzer_checks"]
        rows.append([
            "%s prevalence" % canonical,
            _fmt(stats["prevalence"], percent=True),
            "3%-12%",
            _mark(checks.get("%s_prevalence_in_range" % canonical)),
        ])
        rows.append([
            "%s independent fault events" % canonical,
            _fmt(stats["independent_fault_events"]), ">= 80",
            _mark(checks.get("%s_independent_fault_events_enough" % canonical))])
        rows.append([
            "%s valid onset/follow-up" % canonical,
            _fmt(stats["valid_onset_followup"]), ">= 50",
            _mark(checks.get("%s_valid_onset_followup_enough" % canonical))])
        rows.append([
            "%s deployment rejection" % canonical,
            _fmt(stats["deployment_rejection_rate"], percent=True), "<= 25%",
            _mark(checks.get("%s_deployment_rejection_ok" % canonical))])
        rows.append([
            "%s migration rejection" % canonical,
            _fmt(stats["migration_rejection_rate"], percent=True), "<= 40%",
            _mark(checks.get("%s_migration_rejection_ok" % canonical))])
        rows.append([
            "%s worst event share" % canonical,
            _fmt(stats["worst_event_share"], percent=True), "< 10%",
            _mark(checks.get("%s_worst_event_share_ok" % canonical))])
    lines.append(_md_table(["quantity", "measured", "threshold", "verdict"],
                           rows))
    lines.append("")
    lines.append("- regime prevalence difference (spread): %s"
                 % ("not evaluated (partial run)" if "prevalence_spread_ok"
                    in verdict["checks_not_evaluated"]
                    else "%s (threshold <= 4 percentage points, %s)"
                    % (_fmt(verdict["prevalence_spread"], percent=True),
                       _mark(verdict["gate_checks"].get(
                           "prevalence_spread_ok")))))
    lines.append("")
    lines.append("## Event structure and matched marginals")
    lines.append("")
    rows = []
    for canonical in regimes:
        entry = verdict["per_regime"][canonical]
        stats = entry["gate_stats"]
        rows.append([
            canonical,
            entry["onset_resource"],
            _fmt(entry["onset_tau"]),
            _fmt(entry["onsets"]["n_events"]),
            _fmt(entry["onsets"]["n_valid_followup"]),
            _fmt(entry["onsets"]["n_events_with_a_usable_followup"]),
            _fmt(entry["durations"]["mean"]),
            _fmt(entry["durations"]["p50"]),
            _fmt(entry["peak_ratio"]["mean"]),
            _fmt(entry["peak_ratio"]["p50"]),
            _fmt(entry["cohort"]["n_tasks"]),
        ])
    lines.append(_md_table(
        ["regime", "onset res.", "tau", "onsets", "valid follow-up (full window, "
         "gated)", "usable follow-up (P22 definition, not gated)",
         "mean duration", "median duration", "mean peak ratio",
         "median peak ratio", "cohort tasks"], rows))
    lines.append("")
    lines.append("- `valid follow-up` is the gated number (the whole registered "
                 "response window observed, plan §9 >= 50); `usable follow-up` "
                 "is the P22 U4-v2 definition (at least one observed interval "
                 "inside a window) and is reported only -- a large gap between "
                 "the two means most containers die before their registered "
                 "response window completes.")
    lines.append("- cohort observed life (intervals per task, p50 / max vs the "
                 "follow-up horizon): " + "; ".join(
                     "%s %s / %s vs %s"
                     % (canonical,
                        _fmt(verdict["followup"][canonical]
                             ["cohort_observed_life"]["p50"], 1),
                        _fmt(verdict["followup"][canonical]
                             ["cohort_observed_life"]["max"], 1),
                        _fmt(verdict["followup"][canonical]
                             ["followup_horizon"]))
                     for canonical in regimes))
    lines.append("")
    contrast_pairs = verdict["regime_contrast"].get("pairs") or {}
    if contrast_pairs:
        rows = []
        for key in sorted(contrast_pairs):
            pair = contrast_pairs[key]
            matched = pair["matched_marginals"]
            rows.append([
                key,
                _fmt(matched["prevalence_diff"], percent=True),
                _mark(matched["prevalence_diff_ok"]),
                _fmt(matched["event_count_diff_rel"]),
                _mark(matched["event_count_matched"]),
                _fmt(matched["mean_duration_diff_rel"]),
                _mark(matched["mean_duration_matched"]),
                _fmt(matched["peak_ratio_diff_rel"]),
                _mark(matched["peak_ratio_matched"]),
                _mark(pair["marginals_matched"]),
                _mark(pair["allowed_joint_differences"]
                      ["joint_structure_differs"]),
            ])
        lines.append("_Marginal matching per regime pair "
                     "(instrument-local tolerances -- context, not the "
                     "registered gate; `joint differs` is REQUIRED to be true):_")
        lines.append("")
        lines.append(_md_table(
            ["pair", "prevalence diff", "ok", "event count rel diff", "ok",
             "duration rel diff", "ok", "peak ratio rel diff", "ok",
             "marginals matched", "joint structure differs"], rows))
    else:
        lines.append("_No regime pair to contrast in a partial run._")
    lines.append("")
    lines.append("## Onset threshold admissibility (measured, never retuned)")
    lines.append("")
    rows = []
    for canonical in regimes:
        entry = verdict["per_regime"][canonical]
        adm = entry["admissibility"]
        rows.append([
            canonical,
            adm["onset_resource"],
            _fmt(adm["registered_onset_threshold"]),
            _fmt(adm["measured_familiar_task_maximum"]),
            _fmt(adm["envelope_free_familiar_task_maximum"]),
            _fmt(adm["margin_above_measured_maximum"]),
            _fmt(adm["registered_floor"]),
            _fmt(adm["margin_below_floor"]),
            _mark(adm["threshold_admissible"]),
            "PROVISIONAL" if adm["provisional"] else "registered",
        ])
    lines.append(_md_table(
        ["regime", "resource", "tau", "familiar per-task max (all familiar "
         "rows)", "familiar per-task max (envelope-free tasks)", "margin above "
         "max", "registered floor", "margin below floor", "admissible",
         "threshold status"], rows))
    lines.append("")
    lines.append("- the conservative measurement (every familiar-phase row) "
                 "decides admissibility; the envelope-free measurement "
                 "separates a genuine familiar maximum from a cascade task "
                 "whose burst spilled into a later familiar phase.")
    lines.append("- `n/a` means NOT MEASURABLE (no familiar row carries that "
                 "resource); an unmeasurable threshold is reported "
                 "inadmissible, not assumed safe.")
    lines.append("")
    lines.append("## Streams read (recomputation provenance)")
    lines.append("")
    rows = []
    for canonical in regimes:
        info = verdict["streams"][canonical]
        rows.append([canonical, info["tag"], info["discovery"],
                     _fmt(info["steps"]), ",".join(info["regimes_present"]),
                     info["stream_sha256"][:16]])
    dev = verdict["dev_stream"]
    if dev is not None:
        rows.append(["dev", dev["tag"], dev["discovery"], _fmt(dev["steps"]),
                     ",".join(dev["regimes_present"]),
                     dev["stream_sha256"][:16]])
    else:
        rows.append(["dev", "NOT FOUND", "-", "-", "-", "-"])
    lines.append(_md_table(["role", "tag", "discovery", "steps",
                            "regimes present", "stream sha256 (16)"], rows))
    lines.append("")
    lines.append("- every gate number is recomputed from `stream.npz` and "
                 "`task_timeline.npz` with `ftmoe_protocol023_core`; the "
                 "collector's `unseen_data_audit.json` is only ever compared "
                 "against the recomputation `%s`."
                 % ("(agrees)" if verdict["declared_vs_recomputed"]["agree"]
                    else "(DISAGREES -- see warnings)"))
    if dev is not None:
        lines.append("- the development stream above is context: it carries all "
                     "three regimes and the recurrence, but each regime's gate "
                     "block is measured on its own single-regime cohort.")
    else:
        lines.append("- no development stream was read (partial run): the "
                     "per-regime gate blocks above are still each measured on "
                     "their own single-regime cohort.")
    lines.append("")
    if verdict["stop_conditions"]:
        lines.append("## Stop conditions")
        lines.append("")
        for item in verdict["stop_conditions"]:
            lines.append("- %s" % item)
        lines.append("")
    if verdict["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for item in verdict["warnings"]:
            lines.append("- %s" % item)
        lines.append("")
    lines.append("## Definitions")
    lines.append("")
    for key in sorted(verdict["definitions"]):
        lines.append("- **%s**: %s" % (key, verdict["definitions"][key]))
    lines.append("")
    return "\n".join(lines)


def write_artifacts(verdict, audit_out):
    """Write every artifact; called BEFORE the exit code is decided."""
    audit_out = Path(audit_out)
    audit_out.mkdir(parents=True, exist_ok=True)
    written = {}
    for canonical in verdict["evaluated_regimes"]:
        path = write_json(audit_out / ("regime_%s.json" % canonical),
                          regime_document(verdict, canonical))
        written["regime_%s" % canonical] = str(path)
    written["marginal_match"] = str(
        write_json(audit_out / "marginal_match.json",
                   marginal_document(verdict)))
    markdown = audit_out / "DATA_GATE.md"
    markdown.write_text(data_gate_markdown(verdict), encoding="utf8")
    written["data_gate_markdown"] = str(markdown)
    return written


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams-root", type=Path, default=DEFAULT_STREAMS_ROOT,
                        help="directory holding the four registered S2 streams "
                             "(default: %s)" % DEFAULT_STREAMS_ROOT)
    parser.add_argument("--only", default=None,
                        help="analyse one regime only (A/B/C or the registered "
                             "generator name); the other streams are not read")
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT,
                        help="artifact directory (default: %s)"
                             % DEFAULT_AUDIT_OUT)
    args = parser.parse_args(argv)
    configure_threads()
    try:
        verdict = analyse(args.streams_root, only=args.only)
    except AnalyzerError as exc:
        payload = {"protocol": PROTOCOL, "stage": STAGE, "analyzer": ANALYZER,
                   "generated_at": utcnow(), "streams_root":
                       str(args.streams_root), "only": args.only,
                   "error": "%s: %s" % (type(exc).__name__, exc),
                   "analysis_complete": False, "gate_passed": False,
                   "passed": False, "exit_code": 2,
                   "next_action": ("collect the missing stream (or fix the "
                                   "malformed one) and re-run; no gate verdict "
                                   "was written")}
        try:
            write_json(Path(args.audit_out) / "analysis_error.json", payload)
        except OSError as write_exc:              # pragma: no cover
            payload["artifact_write_error"] = str(write_exc)
        # ASCII-escaped: the one-line verdict must survive any console encoding
        # a runner decodes its child's stdout with (the artifact files stay
        # UTF-8 and are never routed through stdout).
        print(json.dumps(json_safe(payload)), flush=True)
        return 2
    verdict["streams_root"] = str(Path(args.streams_root))
    try:
        verdict["artifacts"] = write_artifacts(verdict, args.audit_out)
    except OSError as exc:                        # pragma: no cover
        print(json.dumps(json_safe({
            "protocol": PROTOCOL, "stage": STAGE, "analyzer": ANALYZER,
            "error": "could not write the artifacts: %s" % exc,
            "gate_passed": verdict["gate_passed"], "passed": False,
            "exit_code": 2})), flush=True)
        return 2
    # ASCII-escaped: the one-line verdict must survive any console encoding a
    # runner decodes its child's stdout with.
    print(json.dumps(json_safe(compact_verdict(verdict))), flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
