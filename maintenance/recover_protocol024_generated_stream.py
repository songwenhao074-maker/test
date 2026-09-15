"""Recover Protocol-024 finalization metadata from an already generated stream.

This is a recovery-only path for generation run 34918496058.  The expensive
simulator loop completed and wrote stream.npz, then finalization failed because
prepare_ftmoe_protocol024_stream.py referenced the non-existent name
SCORED_STEPS instead of the registered SCORDED_STEPS constant.

No scientific array is regenerated or modified here.  All audit summaries are
recomputed only from stream.npz.  Information that existed only in the live
workload object (for example short_trace_skips and the full event payload) is
explicitly marked unrecoverable rather than invented.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from prepare_ftmoe_protocol024_stream import (
    ROOT, SCORDED_STEPS, GUARD_ROWS, REGISTERED_SEED, phase_table,
    _summary_block, sha, SCENARIO_PATH, DRIFT_CONFIG,
)
from simulator.workload.BitbrainWorkloadProtocol024 import (
    RESPONSE_LAWS, LAW_IDS, LAW_MECHANISM_IDS, EVENT_PROBABILITY,
    FORBIDDEN_MODEL_INPUTS,
)

SOURCE_RUN_ID = 34918496058
SOURCE_ARTIFACT_ID = 10381312155
EXPECTED_STREAM_SHA256 = "468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946"


def _scheduler_checkpoint():
    from src.constants import MODEL_SAVE_PATH
    candidates = [
        ROOT / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
        ROOT / "scheduler/BaGTI" / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
    ]
    result = next((p.resolve() for p in candidates if p.is_file()), None)
    if result is None:
        raise FileNotFoundError("GOBI trained checkpoint missing during recovery")
    return result


def _recovered_events(event_ids, event_laws):
    found = {}
    for t in range(SCORDED_STEPS):
        for slot in range(event_ids.shape[1]):
            eid = int(event_ids[t, slot])
            if eid < 0:
                continue
            law_mechanism_id = int(event_laws[t, slot])
            if eid not in found:
                found[eid] = {
                    "event_id": eid,
                    "mechanism_id": law_mechanism_id,
                    "first_observed_stream_row": t,
                    "last_observed_stream_row": t,
                }
            else:
                if found[eid]["mechanism_id"] != law_mechanism_id:
                    raise AssertionError("event mechanism changed inside recovered stream")
                found[eid]["last_observed_stream_row"] = t
    ids = sorted(found)
    if ids and ids != list(range(ids[-1] + 1)):
        raise AssertionError("recorded response-event IDs are not contiguous from zero")
    return [found[i] for i in ids]


def recover(source, output):
    source = Path(source)
    output = Path(output)
    stream_source = source if source.is_file() else source / "stream.npz"
    if not stream_source.is_file():
        raise FileNotFoundError(stream_source)
    if output.exists():
        raise FileExistsError("refusing to overwrite recovery output %s" % output)
    output.mkdir(parents=True)
    stream_path = output / "stream.npz"
    shutil.copy2(stream_source, stream_path)
    digest = sha(stream_path)
    if digest != EXPECTED_STREAM_SHA256:
        raise AssertionError("unexpected recovered stream SHA256: %s" % digest)

    with np.load(stream_path) as d:
        required = (
            "host_features", "demands", "schedules", "raw_labels", "capacities",
            "post_totals", "overload_ratio", "overload_mask", "before_placement",
            "after_placement", "creation_ids", "intervals", "audit_phase_ids",
            "audit_response_law_ids", "audit_event_ids", "audit_event_law_ids",
            "audit_event_active", "audit_host_event_any", "audit_deploy_attempts",
            "audit_deploy_rejected", "audit_migrate_attempts", "audit_migrate_rejected",
        )
        missing = [k for k in required if k not in d.files]
        if missing:
            raise KeyError("recovered stream missing keys: %r" % missing)
        labels = np.asarray(d["raw_labels"], dtype=np.int64)
        ratio = np.asarray(d["overload_ratio"], dtype=np.float64)
        host_event_any = np.asarray(d["audit_host_event_any"], dtype=np.uint8)
        deploy_attempts = np.asarray(d["audit_deploy_attempts"], dtype=np.int64)
        deploy_rejected = np.asarray(d["audit_deploy_rejected"], dtype=np.int64)
        migrate_attempts = np.asarray(d["audit_migrate_attempts"], dtype=np.int64)
        migrate_rejected = np.asarray(d["audit_migrate_rejected"], dtype=np.int64)
        event_ids = np.asarray(d["audit_event_ids"], dtype=np.int64)
        event_laws = np.asarray(d["audit_event_law_ids"], dtype=np.int64)
        if labels.shape != (SCORDED_STEPS + GUARD_ROWS, 16):
            raise AssertionError("unexpected raw_labels shape %r" % (labels.shape,))
        expected = np.where((ratio > 1.0).any(-1), ratio.argmax(-1) + 1, 0)
        if not np.array_equal(expected, labels):
            raise AssertionError("raw labels cannot be recomputed from overload_ratio")
        for key in ("host_features", "demands", "schedules", "capacities", "overload_ratio"):
            if not np.isfinite(d[key]).all():
                raise AssertionError("non-finite values in %s" % key)
        recovered_events = _recovered_events(event_ids, event_laws)

    phases = phase_table()
    by_phase = {}
    for phase in phases:
        by_phase[phase["name"]] = dict(
            _summary_block(labels, ratio, host_event_any,
                           deploy_attempts, deploy_rejected,
                           migrate_attempts, migrate_rejected,
                           phase["start"], phase["end"]),
            response_law=phase["response_law"],
        )

    law_name_for_mechanism = {v: k for k, v in LAW_MECHANISM_IDS.items()}
    events_per_law = {law: 0 for law in LAW_IDS}
    for event in recovered_events:
        law = law_name_for_mechanism[event["mechanism_id"]]
        events_per_law[law] += 1

    by_law = {}
    for law in LAW_IDS:
        blocks = [p for p in phases if p["response_law"] == law]
        mask = np.zeros(SCORDED_STEPS, dtype=bool)
        for phase in blocks:
            mask[phase["start"]:phase["end"]] = True
        y = labels[:SCORDED_STEPS][mask]
        by_law[law] = {
            "scored_intervals": int(mask.sum()),
            "positive_host_steps": int((y > 0).sum()),
            "positive_rate": float((y > 0).mean()) if y.size else None,
            "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
            "task_events_recovered_from_audit_arrays": int(events_per_law[law]),
        }

    applied_switches = [
        {"interval": int(p["start"]), "phase": p["name"],
         "response_law": p["response_law"], "event_probability": p["event_probability"]}
        for p in phases if p["start"] > 0
    ]
    audit = {
        "protocol": "024",
        "kind": "response_law_v1_stream_audit_recovered",
        "labels_source": "post-simulator aggregate demand / physical host capacity",
        "direct_label_assignment_from_response_law": False,
        "whole": _summary_block(labels, ratio, host_event_any,
                                deploy_attempts, deploy_rejected,
                                migrate_attempts, migrate_rejected,
                                0, SCORDED_STEPS),
        "per_phase": by_phase,
        "per_response_law": by_law,
        "task_events_total_recovered_from_audit_arrays": len(recovered_events),
        "short_trace_skips": None,
        "short_trace_skips_recoverable": False,
        "applied_switches": applied_switches,
        "recovery": {
            "source_generation_run_id": SOURCE_RUN_ID,
            "source_artifact_id": SOURCE_ARTIFACT_ID,
            "reason": "simulator stream completed; finalization failed on SCORED_STEPS/SCORDED_STEPS typo",
            "scientific_arrays_modified": False,
        },
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (output / "recovered_events.json").write_text(
        json.dumps({
            "schema": "recovered_event_index_only",
            "full_live_workload_event_payload_recoverable": False,
            "events": recovered_events,
        }, indent=2, allow_nan=False) + "\n", encoding="utf8")

    scheduler_weight = _scheduler_checkpoint()
    manifest = {
        "protocol": "024",
        "round": "next_round_v1",
        "family": "protocol024_response_law_v1",
        "seed": REGISTERED_SEED,
        "steps": SCORDED_STEPS,
        "guard_rows": GUARD_ROWS,
        "stream_file": "stream.npz",
        "stream_sha256": digest,
        "timeline": phases,
        "response_laws": RESPONSE_LAWS,
        "event_probability": EVENT_PROBABILITY,
        "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
        "audit_only_npz_keys": [
            "audit_phase_ids", "audit_response_law_ids", "audit_event_ids",
            "audit_event_law_ids", "audit_event_active", "audit_host_event_any",
            "audit_deploy_attempts", "audit_deploy_rejected",
            "audit_migrate_attempts", "audit_migrate_rejected",
        ],
        "model_input_keys": ["host_features", "demands", "schedules",
                             "capacities", "creation_ids", "before_placement"],
        "label_key": "raw_labels",
        "label_rule": "argmax(CPU,RAM,Disk aggregate/capacity) when any ratio>1 else 0",
        "admission_rule": "all response-law shapes are exactly zero at task age 0",
        "scheduler_checkpoint": str(scheduler_weight),
        "scheduler_checkpoint_sha256": sha(scheduler_weight),
        "scenario_adapter_sha256": sha(SCENARIO_PATH),
        "drift_config_sha256": sha(DRIFT_CONFIG),
        "generation_elapsed_seconds": None,
        "audit_file": "audit.json",
        "events_file": "recovered_events.json",
        "recovered_from_failed_finalization": True,
        "source_generation_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "scientific_arrays_modified_during_recovery": False,
        "unrecoverable_live_only_metadata": ["full_response_events_payload", "short_trace_skips", "exact_generation_elapsed_seconds"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")

    for name in ("generation.log", "failure.json"):
        src = stream_source.parent / name
        if src.is_file():
            shutil.copy2(src, output / name)
    recovery = {
        "source_generation_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "stream_sha256": digest,
        "rows": int(labels.shape[0]),
        "scored_intervals": SCORDED_STEPS,
        "guard_rows": GUARD_ROWS,
        "label_recompute_equal": True,
        "all_required_numeric_arrays_finite": True,
        "task_events_recovered": len(recovered_events),
        "scientific_arrays_modified": False,
    }
    (output / "recovery.json").write_text(json.dumps(recovery, indent=2) + "\n", encoding="utf8")
    print(json.dumps(recovery, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.source, args.output)


if __name__ == "__main__":
    main()
