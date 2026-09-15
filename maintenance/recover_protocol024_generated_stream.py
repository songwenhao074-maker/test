"""Finalize an already-generated Protocol-024 seed700 stream without rerunning simulation.

Recovery is intentionally array-only: stream.npz is copied byte-for-byte from
GitHub Actions run 34918496058.  Audit/manifest fields are recomputed from those
arrays.  Live-only metadata that was lost when finalization crashed is marked
unrecoverable instead of being fabricated.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# When this file is executed as ``python maintenance/...py`` Python puts only
# maintenance/ on sys.path.  Add the repository root before importing project
# modules; the first resume attempt intentionally exposed this missing boundary.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prepare_ftmoe_protocol024_stream import (  # noqa: E402
    SCORDED_STEPS, GUARD_ROWS, REGISTERED_SEED, phase_table, _summary_block,
    sha, SCENARIO_PATH, DRIFT_CONFIG,
)
from simulator.workload.BitbrainWorkloadProtocol024 import (  # noqa: E402
    RESPONSE_LAWS, LAW_IDS, LAW_MECHANISM_IDS, EVENT_PROBABILITY,
    FORBIDDEN_MODEL_INPUTS,
)

SOURCE_RUN_ID = 34918496058
SOURCE_ARTIFACT_ID = 10381312155
EXPECTED_SHA = "468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946"
AUDIT_KEYS = [
    "audit_phase_ids", "audit_response_law_ids", "audit_event_ids",
    "audit_event_law_ids", "audit_event_active", "audit_host_event_any",
    "audit_deploy_attempts", "audit_deploy_rejected",
    "audit_migrate_attempts", "audit_migrate_rejected",
]
MODEL_KEYS = ["host_features", "demands", "schedules", "capacities",
              "creation_ids", "before_placement"]


def _events(event_ids, event_laws):
    found = {}
    for t, (ids_row, laws_row) in enumerate(zip(event_ids[:SCORDED_STEPS],
                                                event_laws[:SCORDED_STEPS])):
        for eid, mechanism in zip(ids_row, laws_row):
            eid, mechanism = int(eid), int(mechanism)
            if eid < 0:
                continue
            item = found.setdefault(eid, {
                "event_id": eid, "mechanism_id": mechanism,
                "first_observed_stream_row": t, "last_observed_stream_row": t})
            if item["mechanism_id"] != mechanism:
                raise AssertionError("event mechanism changed inside stream")
            item["last_observed_stream_row"] = t
    ids = sorted(found)
    if ids and ids != list(range(ids[-1] + 1)):
        raise AssertionError("response event IDs are not contiguous from zero")
    return [found[i] for i in ids]


def recover(source: Path, output: Path):
    source, output = Path(source), Path(output)
    src = source if source.is_file() else source / "stream.npz"
    if not src.is_file():
        raise FileNotFoundError(src)
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    output.mkdir(parents=True)
    dst = output / "stream.npz"
    shutil.copy2(src, dst)
    digest = sha(dst)
    if digest != EXPECTED_SHA:
        raise AssertionError("unexpected immutable stream SHA %s" % digest)

    with np.load(dst) as d:
        required = set(MODEL_KEYS + AUDIT_KEYS + [
            "raw_labels", "overload_ratio", "demands", "schedules",
            "post_totals", "overload_mask", "after_placement", "intervals"])
        missing = sorted(required.difference(d.files))
        if missing:
            raise KeyError("stream missing keys %r" % missing)
        labels = np.asarray(d["raw_labels"], np.int64)
        ratio = np.asarray(d["overload_ratio"], np.float64)
        host_event = np.asarray(d["audit_host_event_any"], np.uint8)
        da = np.asarray(d["audit_deploy_attempts"], np.int64)
        dr = np.asarray(d["audit_deploy_rejected"], np.int64)
        ma = np.asarray(d["audit_migrate_attempts"], np.int64)
        mr = np.asarray(d["audit_migrate_rejected"], np.int64)
        event_ids = np.asarray(d["audit_event_ids"], np.int64)
        event_laws = np.asarray(d["audit_event_law_ids"], np.int64)
        if labels.shape != (SCORDED_STEPS + GUARD_ROWS, 16):
            raise AssertionError("unexpected labels shape %r" % (labels.shape,))
        expected = np.where((ratio > 1.0).any(-1), ratio.argmax(-1) + 1, 0)
        if not np.array_equal(expected, labels):
            raise AssertionError("physical raw-label recomputation mismatch")
        for key in ("host_features", "demands", "schedules", "capacities",
                    "overload_ratio"):
            if not np.isfinite(d[key]).all():
                raise AssertionError("non-finite %s" % key)
        recovered_events = _events(event_ids, event_laws)

    phases = phase_table()
    per_phase = {
        p["name"]: dict(_summary_block(labels, ratio, host_event, da, dr, ma, mr,
                                        p["start"], p["end"]),
                        response_law=p["response_law"])
        for p in phases
    }
    inverse_law = {v: k for k, v in LAW_MECHANISM_IDS.items()}
    event_counts = {law: 0 for law in LAW_IDS}
    for event in recovered_events:
        event_counts[inverse_law[event["mechanism_id"]]] += 1
    per_law = {}
    for law in LAW_IDS:
        mask = np.zeros(SCORDED_STEPS, dtype=bool)
        for phase in phases:
            if phase["response_law"] == law:
                mask[phase["start"]:phase["end"]] = True
        y = labels[:SCORDED_STEPS][mask]
        per_law[law] = {
            "scored_intervals": int(mask.sum()),
            "positive_host_steps": int((y > 0).sum()),
            "positive_rate": float((y > 0).mean()),
            "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
            "task_events_recovered_from_audit_arrays": int(event_counts[law]),
        }

    audit = {
        "protocol": "024", "kind": "response_law_v1_stream_audit_recovered",
        "labels_source": "post-simulator aggregate demand / physical host capacity",
        "direct_label_assignment_from_response_law": False,
        "whole": _summary_block(labels, ratio, host_event, da, dr, ma, mr,
                                0, SCORDED_STEPS),
        "per_phase": per_phase, "per_response_law": per_law,
        "task_events_total_recovered_from_audit_arrays": len(recovered_events),
        "short_trace_skips": None, "short_trace_skips_recoverable": False,
        "applied_switches": [
            {"interval": int(p["start"]), "phase": p["name"],
             "response_law": p["response_law"],
             "event_probability": p["event_probability"]}
            for p in phases if p["start"] > 0],
        "recovery": {"source_generation_run_id": SOURCE_RUN_ID,
                     "source_artifact_id": SOURCE_ARTIFACT_ID,
                     "scientific_arrays_modified": False},
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (output / "recovered_events.json").write_text(json.dumps({
        "schema": "recovered_event_index_only",
        "full_live_workload_event_payload_recoverable": False,
        "events": recovered_events}, indent=2, allow_nan=False) + "\n")

    manifest = {
        "protocol": "024", "round": "next_round_v1",
        "family": "protocol024_response_law_v1", "seed": REGISTERED_SEED,
        "steps": SCORDED_STEPS, "guard_rows": GUARD_ROWS,
        "stream_file": "stream.npz", "stream_sha256": digest,
        "timeline": phases, "response_laws": RESPONSE_LAWS,
        "event_probability": EVENT_PROBABILITY,
        "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
        "audit_only_npz_keys": AUDIT_KEYS, "model_input_keys": MODEL_KEYS,
        "label_key": "raw_labels",
        "label_rule": "argmax(CPU,RAM,Disk aggregate/capacity) when any ratio>1 else 0",
        "admission_rule": "all response-law shapes are exactly zero at task age 0",
        "scenario_adapter_sha256": sha(SCENARIO_PATH),
        "drift_config_sha256": sha(DRIFT_CONFIG),
        "generation_elapsed_seconds": None,
        "audit_file": "audit.json", "events_file": "recovered_events.json",
        "recovered_from_failed_finalization": True,
        "source_generation_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "scientific_arrays_modified_during_recovery": False,
        "unrecoverable_live_only_metadata": [
            "full_response_events_payload", "short_trace_skips",
            "exact_generation_elapsed_seconds"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for name in ("generation.log", "failure.json"):
        candidate = src.parent / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)
    recovery = {
        "source_generation_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID, "stream_sha256": digest,
        "rows": int(labels.shape[0]), "scored_intervals": SCORDED_STEPS,
        "guard_rows": GUARD_ROWS, "label_recompute_equal": True,
        "all_required_numeric_arrays_finite": True,
        "task_events_recovered": len(recovered_events),
        "scientific_arrays_modified": False,
    }
    (output / "recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    print(json.dumps(recovery, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.source, args.output)


if __name__ == "__main__":
    main()
