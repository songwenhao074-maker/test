"""Protocol-032 retrospective eligibility audit and frozen-prefix assembler.

This audit is intentionally retrospective: Protocol-031 model results were
already visible.  It may only decide pass/stop for the already-existing 6800
rows; it never tunes physical parameters, windows, seeds or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import numpy as np

import prepare_ftmoe_protocol025_stream as p25
import prepare_ftmoe_protocol031_stream as p31
import run_ftmoe_protocol031_pilot as p31runner
from ftmoe_protocol024_eval import raw_next_target
from ftmoe_protocol024_session import NEXT_TARGET_MODE, Protocol024Session
from simulator.workload.BitbrainWorkloadProtocol025 import (
    FORBIDDEN_MODEL_INPUTS,
    SERVICE_MECHANISM_IDS,
)

EXPECTED_SOURCE_ARTIFACT_DIGEST = "sha256:c8c2e467a1615c3a8be0e7ee6628a55098491fd97c10d489cf28449a6daeeafc"
EXPECTED_STREAM_SHA = "1e8b6bde1fa3f906586030547777428b28fca23c43e06868077dc0e1e46e4d1a"
SCORED = 6799
TOTAL_ROWS = 6800


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gate(ok, evidence=None, reason=None):
    return {
        "status": "pass" if bool(ok) else "blocked",
        "evidence": evidence,
        "reason": None if bool(ok) else str(reason or "condition_failed"),
    }


def unknown(reason, evidence=None):
    return {"status": "unknown", "evidence": evidence, "reason": str(reason)}


def load_chunks(source_root, resume_manifest):
    chunks = list(resume_manifest.get("chunks") or [])
    chunk_rows = []
    cursor = 0
    keys = None
    parts = None
    hashes_ok = True
    for entry in chunks:
        start, end = int(entry["start"]), int(entry["end"])
        path = source_root / "chunks" / entry["file"]
        digest = sha(path) if path.is_file() else None
        row = {
            "file": entry["file"], "start": start, "end": end,
            "expected_sha256": entry["sha256"], "actual_sha256": digest,
            "hash_ok": digest == entry["sha256"],
            "contiguous": start == cursor,
            "rows": end - start,
        }
        chunk_rows.append(row)
        hashes_ok = hashes_ok and row["hash_ok"] and row["contiguous"] and row["rows"] == 200
        if not path.is_file():
            continue
        with np.load(path) as z:
            if keys is None:
                keys = list(z.files)
                parts = {k: [] for k in keys}
            if set(z.files) != set(keys):
                hashes_ok = False
            for key in keys:
                parts[key].append(z[key].copy())
        cursor = end
    arrays = None if parts is None else {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    geometry_ok = (
        len(chunks) == 34 and cursor == TOTAL_ROWS and
        chunks and int(chunks[0]["start"]) == 0 and int(chunks[-1]["end"]) == TOTAL_ROWS
    )
    if arrays is not None:
        geometry_ok = geometry_ok and all(int(v.shape[0]) == TOTAL_ROWS for v in arrays.values())
    return arrays, chunk_rows, hashes_ok and geometry_ok


def truncate_phases(reg):
    phases = []
    for original_index, p in enumerate(p31.phase_table(reg)):
        if int(p["start"]) >= SCORED:
            break
        q = dict(p)
        q["source_phase_index"] = int(original_index)
        q["end"] = min(int(q["end"]), SCORED)
        q["length"] = int(q["end"] - q["start"])
        if q["name"] == "W_gap2" and q["end"] < int(p["end"]):
            q["name"] = "W_gap2_partial"
        phases.append(q)
    return phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--source-registration", required=True)
    ap.add_argument("--source-artifact-digest", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    src = Path(args.source_root)
    out = Path(args.output_root)
    data = out / "data"
    evidence = out / "evidence"
    shutil.rmtree(out, ignore_errors=True)
    data.mkdir(parents=True)
    evidence.mkdir(parents=True)

    plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
    reg = json.loads(Path(args.source_registration).read_text(encoding="utf8"))
    rm = json.loads((src / "resume_manifest.json").read_text(encoding="utf8"))
    receipt = json.loads((src / "pre_generation_receipt.json").read_text(encoding="utf8"))

    gates = {}
    gates["source_artifact_digest"] = gate(
        args.source_artifact_digest == EXPECTED_SOURCE_ARTIFACT_DIGEST ==
        "sha256:" + str(plan["source_generation_zip_sha256"]),
        {"observed": args.source_artifact_digest, "expected": EXPECTED_SOURCE_ARTIFACT_DIGEST},
        "source generation artifact digest mismatch",
    )
    resume_ok = (
        rm.get("protocol") == "031" and int(rm.get("plan_revision", -1)) == 3 and
        rm.get("scenario_id") == "protocol031_rare_recurrence_v2" and
        rm.get("data_revision") == "protocol031_data_revision_002" and
        int(rm.get("next_t", -1)) == TOTAL_ROWS and
        receipt.get("replay_seed") == 700 and receipt.get("model_seed") == 1 and
        rm.get("registration_sha256") == receipt.get("registration_sha256")
    )
    gates["registered_source_identity"] = gate(
        resume_ok,
        {
            "resume_next_t": rm.get("next_t"),
            "scenario_id": rm.get("scenario_id"),
            "data_revision": rm.get("data_revision"),
            "receipt_implementation_commit": receipt.get("implementation_commit"),
            "receipt_workflow_sha": receipt.get("workflow_sha"),
            "registration_sha256": receipt.get("registration_sha256"),
        },
        "source resume/receipt identity mismatch",
    )

    current_reg_sha = sha(args.source_registration)
    registered_files = {}
    receipt_files_ok = True
    for rel, expected in sorted((receipt.get("file_sha256") or {}).items()):
        p = Path(rel)
        actual = sha(p) if p.is_file() else None
        ok = actual == expected
        registered_files[rel] = {"expected": expected, "actual": actual, "match": ok}
        receipt_files_ok = receipt_files_ok and ok
    registration_origin_ok = (
        current_reg_sha == receipt.get("registration_sha256") and receipt_files_ok
    )
    gates["registration_origin"] = gate(
        registration_origin_ok,
        {"current_registration_sha256": current_reg_sha, "receipt_files": registered_files},
        "registered scientific source no longer matches pre-generation receipt",
    )

    arrays, chunk_rows, chunks_ok = load_chunks(src, rm)
    gates["chunk_coverage_and_hashes"] = gate(
        chunks_ok,
        {"chunk_count": len(chunk_rows), "chunks": chunk_rows},
        "immutable chunk coverage/hash check failed",
    )
    if arrays is None:
        audit = {
            "protocol": "032", "plan_revision": 1,
            "kind": "retrospective_input_eligibility_audit_after_031_results_seen",
            "model_results_seen_before_audit": True,
            "run_id": str(args.run_id), "gates": gates, "audit_pass": False,
        }
        (evidence / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        raise SystemExit("Protocol032 source chunks unavailable")

    finite_arrays = {
        k: bool(np.isfinite(v).all()) for k, v in arrays.items()
        if np.issubdtype(v.dtype, np.floating)
    }
    caps = np.asarray(arrays["capacities"], dtype=np.float64)
    capacity_ok = bool(np.isfinite(caps).all() and (caps > 0).all())
    gates["capacity_finite_positive"] = gate(
        capacity_ok,
        {"min_capacity": float(np.min(caps)), "max_capacity": float(np.max(caps)),
         "all_floating_arrays_finite": finite_arrays},
        "capacity nonfinite/nonpositive or source float array nonfinite",
    )

    post = np.asarray(arrays["post_totals"], dtype=np.float64)
    recomputed_ratio = post / caps
    ratio = np.asarray(arrays["overload_ratio"], dtype=np.float64)
    ratio_equal = bool(np.array_equal(recomputed_ratio, ratio))
    physical = np.where((recomputed_ratio > 1.0).any(-1), recomputed_ratio.argmax(-1) + 1, 0)
    label_equal = bool(np.array_equal(physical.astype(np.int64), arrays["raw_labels"]))
    gates["physical_recompute"] = gate(
        ratio_equal and label_equal,
        {"max_abs_ratio_error": float(np.max(np.abs(recomputed_ratio - ratio))),
         "raw_labels_exact": label_equal},
        "post_totals/capacities did not reproduce ratio/raw labels",
    )

    schedules = np.asarray(arrays["schedules"], dtype=np.float64)
    placements = np.concatenate([
        np.asarray(arrays["before_placement"]).reshape(-1),
        np.asarray(arrays["after_placement"]).reshape(-1),
    ])
    schedule_ok = bool(
        np.isfinite(schedules).all() and (schedules >= 0).all() and (schedules <= 1).all() and
        np.allclose(schedules.sum(-1), 1.0, atol=1e-6)
    )
    placement_ok = bool(((placements >= -1) & (placements < 16)).all())
    interval_delta = np.diff(np.asarray(arrays["intervals"], dtype=np.int64))
    interval_ok = bool(interval_delta.size == TOTAL_ROWS - 1 and (interval_delta == 1).all())
    gates["scheduler_and_placement"] = gate(
        schedule_ok and placement_ok and interval_ok,
        {"schedule_sum_max_abs_error": float(np.max(np.abs(schedules.sum(-1) - 1.0))),
         "placement_min": int(placements.min()), "placement_max": int(placements.max()),
         "intervals_strictly_consecutive": interval_ok},
        "scheduler probability, placement index or interval sequence invalid",
    )

    phases = truncate_phases(reg)
    phase_checks = {}
    event_annotations_ok = True
    service_event_seen = {"U": False, "V": False, "W": False}
    for p in phases:
        s, e = int(p["start"]), int(p["end"])
        idx = int(p["source_phase_index"])
        phase_id_ok = bool((np.asarray(arrays["audit_phase_ids"])[s:e] == idx).all())
        expected_service = -1 if p["service"] is None else int(SERVICE_MECHANISM_IDS[p["service"]])
        service_id_ok = bool((np.asarray(arrays["audit_service_ids"])[s:e] == expected_service).all())
        if p["service"] is None:
            expected_event_count = 0
            event_ok = True
        else:
            event_service = np.asarray(arrays["audit_event_service_ids"])[s:e]
            event_active = np.asarray(arrays["audit_event_active"])[s:e] > 0
            expected_event_count = int(((event_service == expected_service) & event_active).sum())
            event_ok = expected_event_count > 0
            service_event_seen[str(p["logical_service"])] = (
                service_event_seen[str(p["logical_service"])] or event_ok
            )
        phase_checks[p["name"]] = {
            "intervals": [s, e], "source_phase_index": idx,
            "expected_service_mechanism_id": expected_service,
            "phase_id_exact": phase_id_ok, "service_id_exact": service_id_ok,
            "expected_service_active_event_hoststeps": expected_event_count,
            "expected_service_event_seen": event_ok,
        }
        event_annotations_ok = event_annotations_ok and phase_id_ok and service_id_ok and event_ok
    event_annotations_ok = event_annotations_ok and all(service_event_seen.values())
    gates["UVW_event_annotations"] = gate(
        event_annotations_ok,
        {"phase_checks": phase_checks, "logical_services_seen": service_event_seen},
        "phase/service audit IDs or U/V/W active event annotations inconsistent",
    )

    common = p25._common_features(arrays["host_features"], arrays["capacities"])
    common_source = inspect.getsource(p25._common_features)
    forbidden_hits = [name for name in FORBIDDEN_MODEL_INPUTS if name in common_source]
    common_ok = bool(np.isfinite(common[:SCORED]).all() and not forbidden_hits)
    gates["common_feature_causality"] = gate(
        common_ok,
        {"function": "prepare_ftmoe_protocol025_stream._common_features",
         "inputs": ["host_features", "capacities"],
         "forbidden_tokens_found": forbidden_hits,
         "output_shape": list(common.shape),
         "finite": bool(np.isfinite(common[:SCORED]).all())},
        "common feature function depends on forbidden/future identifiers or nonfinite values",
    )

    target_source = inspect.getsource(raw_next_target)
    mature_source = inspect.getsource(Protocol024Session._mature_label)
    step_source = inspect.getsource(Protocol024Session.step)
    target_ok = bool(
        NEXT_TARGET_MODE == "raw_next_fault" and
        "index + 1" in target_source and
        "observed_until < index + 2" in target_source and
        "return raw_next_target" in mature_source and
        "settled_now = t - 2" in step_source and
        "label of %d was read before prediction" in step_source
    )
    gates["raw_next_tplus1_tplus2_maturity"] = gate(
        target_ok,
        {"target_mode": NEXT_TARGET_MODE,
         "raw_next_target_index_plus_1": "index + 1" in target_source,
         "publication_gate_index_plus_2": "observed_until < index + 2" in target_source,
         "session_settlement_t_minus_2": "settled_now = t - 2" in step_source,
         "prediction_before_label_guard": "label of %d was read before prediction" in step_source},
        "raw[t+1]/t+2 maturity semantics could not be verified from execution source",
    )

    build_guard_source = inspect.getsource(p31runner.build_guard)
    f0 = next(p for p in phases if p["name"] == "F0")
    guard_indices = np.arange(int(f0["start"]), int(f0["end"]) - 1, dtype=np.int64)
    guard_indices = guard_indices[(guard_indices - int(f0["start"])) % 5 == 0]
    guard_targets = np.asarray(arrays["raw_labels"])[guard_indices + 1].reshape(-1)
    guard_normal = int((guard_targets == 0).sum())
    guard_source_ok = bool(
        guard_indices.size > 0 and guard_normal > 0 and
        "raw[np.asarray(batch)+1]" in build_guard_source and
        "F0" in build_guard_source
    )
    gates["F0_normal_guard_source"] = gate(
        guard_source_ok,
        {"prediction_indices_count": int(guard_indices.size),
         "prediction_index_min": int(guard_indices.min()),
         "prediction_index_max": int(guard_indices.max()),
         "target": "same-host raw[t+1] inside F0",
         "normal_hoststeps": guard_normal,
         "positive_hoststeps": int((guard_targets > 0).sum())},
        "F0 known-normal guard provenance/availability not verified",
    )

    phase_coverage = {}
    coverage_ok = True
    recurrence_ok = True
    for p in phases:
        if p["logical_service"] is None:
            continue
        s, e = int(p["start"]), int(p["end"])
        y = np.asarray(arrays["raw_labels"])[s:e]
        pos, neg = int((y > 0).sum()), int((y == 0).sum())
        ok = pos >= 32 and neg >= 32
        phase_coverage[p["name"]] = {
            "intervals": [s, e], "positive_hoststeps": pos,
            "negative_hoststeps": neg, "min32_each": ok,
        }
        coverage_ok = coverage_ok and ok
        if p["name"] in ("U_rec1", "V_rec1"):
            recurrence_ok = recurrence_ok and ok
    recurrence_present = set(k for k in phase_coverage if k in ("U_rec1", "V_rec1")) == {"U_rec1", "V_rec1"}
    gates["nonF0_and_recurrence_class_coverage"] = gate(
        coverage_ok and recurrence_ok and recurrence_present,
        phase_coverage,
        "one or more non-F0 phases / recurrence windows lacks >=32 positive and >=32 negative host-steps",
    )

    # Full frozen serialization must reproduce the already-observed interim stream byte-for-byte.
    np.savez_compressed(
        data / "stream.npz", **arrays,
        overload_mask=(np.asarray(arrays["overload_ratio"]) > 1.0).astype(np.uint8),
    )
    stream_sha = sha(data / "stream.npz")
    gates["frozen_stream_identity"] = gate(
        stream_sha == EXPECTED_STREAM_SHA == str(plan["input_stream_sha256"]),
        {"actual_stream_sha256": stream_sha, "expected_stream_sha256": EXPECTED_STREAM_SHA},
        "assembled frozen stream differs from registered 031 interim prefix",
    )

    required_gate_names = [
        "source_artifact_digest", "registered_source_identity", "registration_origin",
        "chunk_coverage_and_hashes", "capacity_finite_positive", "physical_recompute",
        "scheduler_and_placement", "UVW_event_annotations", "common_feature_causality",
        "raw_next_tplus1_tplus2_maturity", "F0_normal_guard_source",
        "nonF0_and_recurrence_class_coverage", "frozen_stream_identity",
    ]
    audit_pass = all(gates[name]["status"] == "pass" for name in required_gate_names)

    manifest = {
        "protocol": "032", "plan_revision": 1,
        "kind": "existing_031_interim6800_prefix_frozen_for_032",
        "source_protocol": "031", "source_generation_run_id": int(plan["source_generation_run_id"]),
        "source_checkpoint_next_t": TOTAL_ROWS,
        "scenario_id": reg["scenario_id"], "data_revision": reg["data_revision"],
        "steps": SCORED, "guard_rows": 1, "total_rows": TOTAL_ROWS,
        "stream_file": "stream.npz", "stream_sha256": stream_sha,
        "timeline": phases, "replay_seed": 700, "model_seed": 1,
        "retrospective_eligibility_audit": True,
        "model_results_seen_before_audit": True,
        "no_new_simulation": True,
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    frozen = dict(manifest)
    frozen["frozen"] = True
    (data / "frozen_data_manifest.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(data / "common_observable_features.npz", features=common)

    registration = {
        "protocol": "032", "plan_revision": 1, "status": "registered_for_single_development_pair",
        "source_code_commit": plan["source_code_commit"],
        "implementation_branch": plan["implementation_branch"],
        "source_generation_run_id": int(plan["source_generation_run_id"]),
        "source_generation_artifact_id": int(plan["source_generation_artifact_id"]),
        "source_generation_zip_sha256": plan["source_generation_zip_sha256"],
        "input_stream_sha256": EXPECTED_STREAM_SHA,
        "scored_intervals": SCORED, "guard_rows": 1, "total_rows": TOTAL_ROWS,
        "replay_seed": 700, "model_seed": 1,
        "arms": ["C_fixed5", "D_guard_budget"],
        "method_change": plan["method_change"],
        "preserved_gates": plan["preserved_gates"],
        "recurrence_windows": plan["recurrence_windows"],
        "full_model_replay_budget": 2,
        "full_generation_budget": 0,
        "statistical_confirmation": False,
        "input_audit_required": True,
    }
    (evidence / "registration.json").write_text(json.dumps(registration, indent=2, allow_nan=False) + "\n")

    audit = {
        "protocol": "032", "plan_revision": 1,
        "kind": "retrospective_input_eligibility_audit_after_031_results_seen",
        "model_results_seen_before_audit": True,
        "not_claimed_as_pre_031_audit": True,
        "run_id": str(args.run_id),
        "source_generation_run_id": int(plan["source_generation_run_id"]),
        "source_artifact_id": int(plan["source_generation_artifact_id"]),
        "scored_intervals": SCORED, "guard_rows": 1,
        "gates": gates, "required_gate_names": required_gate_names,
        "audit_pass": bool(audit_pass),
        "reroll_or_parameter_change_allowed_after_failure": False,
    }
    (evidence / "input_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    lock = {
        "protocol": "032", "plan_revision": 1, "locked": bool(audit_pass),
        "input_audit_passed": bool(audit_pass),
        "stream_sha256": stream_sha,
        "source_generation_run_id": int(plan["source_generation_run_id"]),
        "source_artifact_id": int(plan["source_generation_artifact_id"]),
        "source_checkpoint_next_t": TOTAL_ROWS,
        "scored_intervals": SCORED, "guard_rows": 1,
        "retrospective_audit": True,
        "no_new_simulation": True,
    }
    (evidence / "input_lock.json").write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    shutil.copy2(src / "resume_manifest.json", evidence / "source_resume_manifest.json")
    shutil.copy2(src / "pre_generation_receipt.json", evidence / "source_pre_generation_receipt.json")

    print(json.dumps({
        "protocol": "032", "audit_pass": audit_pass,
        "stream_sha256": stream_sha,
        "blocked_gates": [k for k in required_gate_names if gates[k]["status"] != "pass"],
    }, indent=2))
    if not audit_pass:
        raise SystemExit("Protocol032 input eligibility audit blocked; no model run allowed")


if __name__ == "__main__":
    main()
