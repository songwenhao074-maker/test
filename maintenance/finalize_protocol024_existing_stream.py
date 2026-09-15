"""Finalize an already-generated Protocol-024 response-law stream.

This recovery path is intentionally limited to metadata/audit finalization.  It
never regenerates or mutates ``stream.npz``.  It exists because generation run
34918496058 completed the 4981-row simulator stream and then failed in the
post-save audit block on a misspelled constant name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "artifacts/ftmoe_online/protocol_024/next_round_v1/response_law_config.json"
SCORDED_STEPS = 4980
GUARD_ROWS = 1
LAW_IDS = ("R1", "R2", "R3")
LAW_MECHANISM_IDS = {"R1": 0, "R2": 1, "R3": 2}
AUDIT_ONLY_KEYS = [
    "audit_phase_ids", "audit_response_law_ids", "audit_event_ids",
    "audit_event_law_ids", "audit_event_active", "audit_host_event_any",
    "audit_deploy_attempts", "audit_deploy_rejected",
    "audit_migrate_attempts", "audit_migrate_rejected",
]
MODEL_INPUT_KEYS = [
    "host_features", "demands", "schedules", "capacities",
    "creation_ids", "before_placement",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_table(cfg):
    rows, cursor = [], 0
    for name, length, law in cfg["timeline"]:
        rows.append({
            "name": name,
            "start": cursor,
            "end": cursor + int(length),
            "length": int(length),
            "response_law": law,
            "event_probability": 0.0 if law is None else float(cfg["event_probability"]),
            "kind": "familiar" if law is None else "response_law",
        })
        cursor += int(length)
    if cursor != SCORDED_STEPS:
        raise AssertionError("registered timeline does not sum to 4980")
    return rows


def fault_runs(labels, start, end):
    out = []
    for host in range(labels.shape[1]):
        klass, run_start = 0, None
        for t in range(start, end):
            value = int(labels[t, host])
            if value == klass and value > 0:
                continue
            if klass > 0:
                out.append((host, klass, run_start, t - 1, t - run_start))
            klass = value
            run_start = t if value > 0 else None
        if klass > 0:
            out.append((host, klass, run_start, end - 1, end - run_start))
    return out


def summary_block(arrays, start, end):
    labels = arrays["raw_labels"]
    ratio = arrays["overload_ratio"]
    event = arrays["audit_host_event_any"]
    y = labels[start:end]
    r = ratio[start:end]
    e = event[start:end]
    runs = fault_runs(labels, start, end)
    lengths = [x[-1] for x in runs]
    da = int(arrays["audit_deploy_attempts"][start:end].sum())
    dr = int(arrays["audit_deploy_rejected"][start:end].sum())
    ma = int(arrays["audit_migrate_attempts"][start:end].sum())
    mr = int(arrays["audit_migrate_rejected"][start:end].sum())
    host_steps = int(y.size)
    positive = int((y > 0).sum())
    return {
        "intervals": [int(start), int(end)],
        "host_steps": host_steps,
        "positive_host_steps": positive,
        "positive_rate": positive / float(host_steps) if host_steps else None,
        "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
        "fault_runs": len(runs),
        "run_length_mean": float(np.mean(lengths)) if lengths else None,
        "run_length_p95": float(np.percentile(lengths, 95)) if lengths else None,
        "run_length_max": int(max(lengths)) if lengths else None,
        "event_related_positive_host_steps": int(((y > 0) & (e > 0)).sum()),
        "peak_overload_ratio": float(r.max()) if r.size else None,
        "deployment_attempts": da,
        "deployment_rejected": dr,
        "deployment_rejection_rate": dr / float(max(da, 1)),
        "migration_attempts": ma,
        "migration_rejected": mr,
        "migration_rejection_rate": mr / float(max(ma, 1)),
    }


def expected_phase_vectors(phases):
    count = SCORDED_STEPS + GUARD_ROWS
    phase_ids = np.zeros(count, dtype=np.int64)
    law_ids = np.full(count, -1, dtype=np.int64)
    for t in range(count):
        phase_index = len(phases) - 1
        phase = phases[-1]
        for i, candidate in enumerate(phases):
            if candidate["start"] <= t < candidate["end"]:
                phase_index, phase = i, candidate
                break
        phase_ids[t] = phase_index
        law = phase["response_law"]
        law_ids[t] = -1 if law is None else LAW_MECHANISM_IDS[law]
    return phase_ids, law_ids


def applied_switches(phases):
    current_law, current_probability = None, 0.0
    out = []
    for phase in phases:
        law = phase["response_law"]
        probability = phase["event_probability"]
        if law != current_law or probability != current_probability:
            if phase["start"] < SCORDED_STEPS:
                out.append({
                    "interval": int(phase["start"]),
                    "phase": phase["name"],
                    "response_law": law,
                    "event_probability": probability,
                })
            current_law, current_probability = law, probability
    return out


def recover_event_index(arrays):
    event_ids = arrays["audit_event_ids"]
    event_laws = arrays["audit_event_law_ids"]
    creation = arrays["creation_ids"]
    intervals = arrays["intervals"]
    events = []
    ids = np.unique(event_ids[event_ids >= 0])
    for raw_id in ids.tolist():
        mask = event_ids == raw_id
        law_values = np.unique(event_laws[mask])
        law_values = law_values[law_values >= 0]
        if law_values.size != 1:
            raise AssertionError("event has inconsistent mechanism id")
        mechanism = int(law_values[0])
        law = LAW_IDS[mechanism]
        positions = np.argwhere(mask)
        first_t, first_slot = map(int, positions[0])
        creation_values = creation[mask]
        creation_values = creation_values[creation_values >= 0]
        creation_id = int(creation_values[0]) if creation_values.size else None
        events.append({
            "event_id": int(raw_id),
            "law_id": law,
            "mechanism_id": mechanism,
            "creation_id": creation_id,
            "first_observed_stream_row": first_t,
            "first_observed_interval": int(intervals[first_t]),
            "recovered_from_audit_arrays": True,
        })
    return events


def finalize(data_dir, source_run_id, source_artifact, expected_sha=None):
    started = time.perf_counter()
    data_dir = Path(data_dir)
    stream_path = data_dir / "stream.npz"
    if not stream_path.is_file():
        raise FileNotFoundError(stream_path)
    digest = sha256(stream_path)
    if expected_sha and digest != expected_sha:
        raise AssertionError("immutable source stream SHA mismatch")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf8"))
    phases = phase_table(cfg)
    with np.load(stream_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    required = set(MODEL_INPUT_KEYS + AUDIT_ONLY_KEYS + [
        "raw_labels", "overload_ratio", "post_totals", "intervals"])
    missing = sorted(required.difference(arrays))
    if missing:
        raise AssertionError("source stream missing keys: %s" % missing)
    if arrays["raw_labels"].shape != (SCORDED_STEPS + GUARD_ROWS, 16):
        raise AssertionError("unexpected raw label shape")
    if arrays["host_features"].shape != (SCORDED_STEPS + GUARD_ROWS, 16, 7):
        raise AssertionError("unexpected host feature shape")
    for key in MODEL_INPUT_KEYS + ["overload_ratio", "post_totals"]:
        value = arrays[key]
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError("non-finite values in %s" % key)

    ratio = arrays["overload_ratio"]
    expected_labels = np.where((ratio > 1.0).any(-1), ratio.argmax(-1) + 1, 0)
    if not np.array_equal(expected_labels, arrays["raw_labels"]):
        raise AssertionError("raw labels do not recompute from overload ratio")
    exp_phase, exp_law = expected_phase_vectors(phases)
    if not np.array_equal(exp_phase, arrays["audit_phase_ids"]):
        raise AssertionError("audit phase ids do not match registered timeline")
    if not np.array_equal(exp_law, arrays["audit_response_law_ids"]):
        raise AssertionError("audit response-law ids do not match registered timeline")
    if set(AUDIT_ONLY_KEYS) & set(MODEL_INPUT_KEYS):
        raise AssertionError("audit/model-input key overlap")

    events = recover_event_index(arrays)
    by_phase = {}
    for phase in phases:
        by_phase[phase["name"]] = dict(
            summary_block(arrays, phase["start"], phase["end"]),
            response_law=phase["response_law"])
    by_law = {}
    for law in LAW_IDS:
        mask = np.zeros(SCORDED_STEPS, dtype=bool)
        for phase in phases:
            if phase["response_law"] == law:
                mask[phase["start"]:phase["end"]] = True
        y = arrays["raw_labels"][:SCORDED_STEPS][mask]
        mechanism = LAW_MECHANISM_IDS[law]
        event_count = len({e["event_id"] for e in events if e["mechanism_id"] == mechanism})
        by_law[law] = {
            "scored_intervals": int(mask.sum()),
            "positive_host_steps": int((y > 0).sum()),
            "positive_rate": float((y > 0).mean()) if y.size else None,
            "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
            "task_events_recovered": int(event_count),
        }

    audit = {
        "protocol": "024",
        "kind": "response_law_v1_stream_audit_recovered",
        "labels_source": "post-simulator aggregate demand / physical host capacity",
        "direct_label_assignment_from_response_law": False,
        "whole": summary_block(arrays, 0, SCORDED_STEPS),
        "per_phase": by_phase,
        "per_response_law": by_law,
        "task_events_total_recovered": len(events),
        "short_trace_skips": None,
        "short_trace_skips_note": "not persisted before the original post-save NameError; unavailable from immutable stream bytes",
        "applied_switches": applied_switches(phases),
        "source_generation_run_id": int(source_run_id),
        "stream_sha256": digest,
        "stream_bytes_mutated": False,
    }
    (data_dir / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (data_dir / "events_recovered.json").write_text(json.dumps(events, indent=2, allow_nan=False) + "\n", encoding="utf8")

    manifest = {
        "protocol": "024",
        "round": "next_round_v1",
        "family": "protocol024_response_law_v1",
        "seed": int(cfg["development_seed"]),
        "steps": SCORDED_STEPS,
        "guard_rows": GUARD_ROWS,
        "stream_file": "stream.npz",
        "stream_sha256": digest,
        "timeline": phases,
        "response_laws": {law: cfg[law] for law in LAW_IDS},
        "event_probability": float(cfg["event_probability"]),
        "forbidden_model_inputs": list(cfg["forbidden_model_or_lifecycle_inputs"]),
        "audit_only_npz_keys": AUDIT_ONLY_KEYS,
        "model_input_keys": MODEL_INPUT_KEYS,
        "label_key": "raw_labels",
        "label_rule": "argmax(CPU,RAM,Disk aggregate/capacity) when any ratio>1 else 0",
        "admission_rule": "all response-law shapes are exactly zero at task age 0",
        "generation_elapsed_seconds": None,
        "audit_file": "audit.json",
        "events_file": "events_recovered.json",
        "recovery_provenance": {
            "source_generation_run_id": int(source_run_id),
            "source_artifact_name": source_artifact,
            "source_stream_sha256": digest,
            "source_failure": "NameError: name 'SCORED_STEPS' is not defined after stream.npz save",
            "finalization_only": True,
            "simulator_rerun": False,
            "stream_bytes_mutated": False,
            "response_law_config_changed": False,
            "confirmation_seeds_used": False,
        },
        "recovery_finalization_elapsed_seconds": time.perf_counter() - started,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    result = {
        "source_run_id": int(source_run_id),
        "stream_sha256": digest,
        "rows": int(arrays["raw_labels"].shape[0]),
        "label_recompute_equal": True,
        "phase_vector_equal": True,
        "law_vector_equal": True,
        "events_recovered": len(events),
        "stream_bytes_mutated": False,
    }
    (data_dir / "recovery_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--expected-sha", default=None)
    args = parser.parse_args()
    finalize(args.data_dir, args.source_run_id, args.source_artifact, args.expected_sha)


if __name__ == "__main__":
    main()
