"""Protocol-032 single C_fixed5 / D_guard_budget development runner.

This runner consumes only a pre-frozen 6800-row input bundle and never invokes
simulator generation.  It reuses Protocol-031 mechanics while keeping all
Protocol-032 outputs and registrations isolated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
from ftmoe_protocol025_session import Protocol025FixedSession
from ftmoe_protocol032_guard_budget import Protocol032DynamicSession

COMPARATORS = ("C_fixed5", "D_guard_budget")


def write_json(path, value):
    p31.write_json(path, value)


def validate_input(stream, registration_path, input_lock_path):
    root = Path(stream)
    reg = json.loads(Path(registration_path).read_text(encoding="utf8"))
    lock = json.loads(Path(input_lock_path).read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    frozen = json.loads((root / "frozen_data_manifest.json").read_text(encoding="utf8"))
    if reg.get("protocol") != "032" or int(reg.get("plan_revision", -1)) != 1:
        raise AssertionError("Protocol032 registration mismatch")
    if lock.get("protocol") != "032" or lock.get("locked") is not True:
        raise AssertionError("Protocol032 input lock missing")
    if lock.get("input_audit_passed") is not True:
        raise RuntimeError("Protocol032 retrospective input audit did not pass")
    digest = p31.sha(root / "stream.npz")
    expected = "1e8b6bde1fa3f906586030547777428b28fca23c43e06868077dc0e1e46e4d1a"
    if not (digest == expected == manifest.get("stream_sha256") ==
            frozen.get("stream_sha256") == lock.get("stream_sha256") ==
            reg.get("input_stream_sha256")):
        raise AssertionError("Protocol032 frozen input identity mismatch")
    if int(manifest.get("steps", -1)) != 6799 or int(manifest.get("guard_rows", -1)) != 1:
        raise AssertionError("Protocol032 input geometry mismatch")
    return digest, manifest, lock, reg


def topology_snapshot(session, cursor, label, pdefs):
    bank = session.model.learner
    ctrl = session.lifecycle_controller
    return {
        "label": str(label),
        "cursor": int(cursor),
        "phase": p31.phase_name(max(0, int(cursor) - 1), pdefs),
        "active_specialist_id": None if ctrl.active_specialist_id is None else str(ctrl.active_specialist_id),
        "real_dormant_expert_ids": sorted([str(k) for k in bank.dormant_experts.keys()], key=int),
        "memory_registry_ids": sorted([str(k) for k in ctrl.specialist_memory.keys()], key=int),
        "shadow_id": None if getattr(bank, "shadow_id", None) is None else str(bank.shadow_id),
        "pending_reuse": deepcopy(getattr(ctrl, "pending_reuse", None)),
        "accepted_birth_ids": [str(x) for x in ctrl.accepted_ids],
        "topology": bank.topology_manifest(),
    }


def run_arm(name, stream, arm_dir, registration, input_lock, run_id):
    p31.deterministic_runtime()
    if name not in COMPARATORS:
        raise ValueError(name)
    out = Path(arm_dir)
    out.mkdir(parents=True, exist_ok=False)
    stream_sha, manifest, lock, reg = validate_input(stream, registration, input_lock)
    if reg.get("arms") != list(COMPARATORS):
        raise AssertionError("Protocol032 arm registration mismatch")

    bundle = s4.build_replay(Path(stream))
    if bundle["steps"] != 6799 or bundle["manifest"].get("stream_sha256") != stream_sha:
        raise AssertionError("Protocol032 replay identity mismatch")
    pdefs = p31.phases(manifest)
    pmap = {p["name"]: p for p in pdefs}
    snapshot_points = {}
    for key, label, edge in [
        ("U_first", "after_U_first", "end"),
        ("V_first", "after_V_first", "end"),
        ("U_rec1", "before_U_rec1", "start"),
        ("V_rec1", "before_V_rec1", "start"),
    ]:
        if key in pmap:
            snapshot_points[int(pmap[key][edge])] = label

    guard = p31.build_guard(bundle)
    guard["meta"]["protocol"] = "032"
    guard["meta"]["role"] = "F0_known_normal_regression_guard_same_rows_as_031"
    write_json(out / "guard_manifest.json", guard["meta"])

    runtime_registration = {
        "protocol": "032",
        "plan_revision": 1,
        "comparator": name,
        "source_scenario_id": manifest.get("scenario_id"),
        "source_data_revision": manifest.get("data_revision"),
        "stream_sha256": stream_sha,
        "replay_seed": 700,
        "model_seed": 1,
        "development_only": True,
        "statistical_confirmation": False,
        "input_lock_sha256": p31.sha(input_lock),
    }
    common = dict(
        seed=1,
        replay_bundle=bundle,
        budget=p31.budget(),
        out_dir=out,
        run_id=f"protocol032_{run_id}_{name}",
        stream_dir=Path(stream),
        phase_defs=pdefs,
        stream_sha=stream_sha,
        registration=runtime_registration,
        learning_rate=1e-4,
    )
    session = (
        Protocol032DynamicSession(guard_anchor=guard, **common)
        if name == "D_guard_budget"
        else Protocol025FixedSession("C_fixed5", **common)
    )
    init_prefix = p31.shared_prefix_hash(session.model, 4)
    write_json(out / "initialization.json", {
        "shared_first4_expert_and_router_rows_sha256": init_prefix,
        "model_seed": 1,
        "registered_shared_prefix_initialization": True,
    })

    snapshots = []
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    peak_p = p31.param_bytes(session)
    peak_o = p31.opt_bytes(session)
    peak_live = 5 if name == "C_fixed5" else 4
    peak_resident = peak_live
    for i in range(session.steps):
        if name == "D_guard_budget" and i in snapshot_points:
            snapshots.append(topology_snapshot(session, i, snapshot_points[i], pdefs))
        session.step()
        if i % 64 == 0 or i + 1 == session.steps:
            peak_p = max(peak_p, p31.param_bytes(session))
            peak_o = max(peak_o, p31.opt_bytes(session))
        if name == "D_guard_budget":
            bank = session.model.learner
            peak_live = max(peak_live, len(bank.ids))
            peak_resident = max(peak_resident, bank.resident_count())
            if bank.resident_count() > 8:
                raise AssertionError("Protocol032 resident capacity exceeded")

    if name == "D_guard_budget":
        session.lifecycle_controller.mark_stream_end()
    session.finish()
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    session.save()

    prob = session.predictions["probability"]
    cls = session.predictions["class_probability"]
    y = session.predictions["labels"]
    if (y < 0).any() or (session.predictions["raw_labels"] < 0).any():
        raise AssertionError("Protocol032 unsettled model outputs")
    cost = {
        "wall_seconds": float(wall),
        "cpu_seconds": float(cpu),
        "max_rss_bytes": p31.rss_bytes(),
        "p95_inference_seconds": float(np.percentile(session.predictions["prediction_seconds"], 95)),
        "online_update_opportunities_completed": int(session.updates),
        "online_sample_draws": p31.samples(session),
        "final_resident_parameter_bytes": p31.param_bytes(session),
        "peak_resident_parameter_bytes": int(peak_p),
        "final_optimizer_state_bytes": p31.opt_bytes(session),
        "peak_optimizer_state_bytes": int(peak_o),
        "peak_live_expert_count": int(peak_live),
        "peak_resident_expert_count": int(peak_resident),
        "runtime_profile": p31.deterministic_runtime(),
    }
    summary = {
        "protocol": "032",
        "plan_revision": 1,
        "comparator": name,
        "run_id": str(run_id),
        "completed": True,
        "development_only": True,
        "statistical_confirmation": False,
        "stream_sha256": stream_sha,
        "initialization_shared_prefix_sha256": init_prefix,
        "manifest": session.comparator_manifest(),
        "full": p31.metrics(prob, cls, y),
        "phases": [],
        "cost": cost,
    }
    for p in pdefs:
        a, b = int(p["start"]), int(p["end"])
        yy = y[a:b]
        summary["phases"].append({
            "phase": p["name"],
            "service": p["regime"],
            "intervals": [a, b],
            "positive_rows": int((yy > 0).sum()),
            "negative_rows": int((yy == 0).sum()),
            **p31.metrics(prob[a:b], cls[a:b], yy),
        })

    if name == "D_guard_budget":
        ctrl = session.lifecycle_controller
        events = list(ctrl.events)
        records = list(ctrl.candidate_records.values())
        ec = Counter(e["kind"] for e in events)
        byphase = defaultdict(Counter)
        for e in events:
            byphase[p31.phase_name(e.get("cursor", 0), pdefs)][e["kind"]] += 1
        accepted = sum(r.get("accepted") is True and not r.get("cancelled_by_reuse") for r in records)
        cancelled = sum(bool(r.get("cancelled_by_reuse")) for r in records)
        rejected = sum(r.get("accepted") is False and not r.get("cancelled_by_reuse") for r in records)
        pending = sum(r.get("accepted") is None for r in records)
        due = ctrl.due_conservation()
        if not due["birth"]["conserved"]:
            raise AssertionError("Protocol032 birth due accounting not conserved")
        if not due["reuse_nonblocking"]["due_conserved"] or not due["reuse_nonblocking"]["started_conserved"]:
            raise AssertionError("Protocol032 reuse accounting not conserved")
        if ctrl.purges != 0:
            raise AssertionError("Protocol032 unexpectedly purged accepted memory")

        candidate_sources = []
        for rec in records:
            idx = list(rec.get("training_indices") or [])
            decision = deepcopy(rec.get("decision"))
            guard_decision = None if not isinstance(decision, dict) else deepcopy(decision.get("guard"))
            candidate_sources.append({
                "candidate_id": rec.get("candidate_id"),
                "created_cursor": rec.get("created_cursor"),
                "accepted": rec.get("accepted"),
                "cancelled_by_reuse": bool(rec.get("cancelled_by_reuse")),
                "training_indices_count": len(idx),
                "training_source_phases": sorted({p31.phase_name(i, pdefs) for i in idx}),
                "accepted_cursor": rec.get("accepted_cursor"),
                "first_influence_cursor": rec.get("first_influence_cursor"),
                "decision": decision,
                "F0_guard": guard_decision,
            })
        extra = dict(ctrl.extra_compute)
        cost.update({
            "shadow_train_steps": int(extra.get("shadow_train_steps", 0)),
            "shadow_train_examples": int(extra.get("shadow_train_examples", 0)),
            "reuse_validation_forwards": int(extra.get("p031_reuse_validation_forwards", 0)),
            "reuse_guard_forwards": int(extra.get("p031_reuse_guard_forwards", 0)),
            "shadow_train_seconds": float(extra.get("shadow_train_seconds", 0.0)),
            "shadow_validation_seconds": float(extra.get("shadow_validation_seconds", 0.0)),
            "birth_guard_seconds": float(extra.get("guard_seconds", 0.0)),
            "reuse_validation_seconds": float(extra.get("p031_reuse_validation_seconds", 0.0)),
            "reuse_guard_seconds": float(extra.get("p031_reuse_guard_seconds", 0.0)),
        })
        bank = session.model.learner
        lifecycle = {
            "birth_candidates_created": len(records),
            "birth_candidates_accepted": accepted,
            "birth_candidates_rejected": rejected,
            "birth_candidates_cancelled": cancelled,
            "birth_candidates_pending": pending,
            "birth_conservation": len(records) == accepted + rejected + cancelled + pending,
            "retirements": int(ctrl.retirements),
            "reactivations": int(ctrl.reactivations),
            "purges": int(ctrl.purges),
            "capacity_preserve_skips": int(ctrl.capacity_preserve_skips),
            "accepted_birth_ids": [str(x) for x in ctrl.accepted_ids],
            "memory_registry_ids": sorted([str(k) for k in ctrl.specialist_memory.keys()], key=int),
            "real_dormant_expert_ids": sorted([str(k) for k in bank.dormant_experts.keys()], key=int),
            "active_specialist_id": ctrl.active_specialist_id,
            "reuse_first_influence_cursor": ctrl.first_reuse_influence_cursor,
            "reuse_records": ctrl.reuse_records,
            "reuse_outcomes": ctrl.reuse_outcomes,
            "due_accounting": due,
            "event_counts": dict(ec),
            "event_counts_by_phase": {k: dict(v) for k, v in byphase.items()},
            "candidate_sources": candidate_sources,
            "topology_snapshots": snapshots,
            "final_topology": bank.topology_manifest(),
            "extra_compute": extra,
        }
        summary["lifecycle"] = lifecycle
        p31.write_jsonl(out / "lifecycle.jsonl", events)
        write_json(out / "candidate_records.json", records)
        write_json(out / "reuse_records.json", ctrl.reuse_records)
        write_json(out / "opportunity_accounting.json", due)
        write_json(out / "topology_snapshots.json", snapshots)
        write_json(out / "lifecycle_summary.json", lifecycle)

    write_json(out / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--input-lock", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", choices=COMPARATORS, required=True)
    a = ap.parse_args()
    try:
        result = run_arm(a.arm, a.stream, a.arm_dir, a.registration, a.input_lock, a.run_id)
        print(json.dumps(p31.json_ready({
            "completed": True,
            "comparator": a.arm,
            "full_detection": result["full"]["detection"],
        }), indent=2, allow_nan=False))
    except Exception:
        root = Path(a.arm_dir)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "runner_exception.json", {
            "protocol": "032",
            "comparator": a.arm,
            "traceback": traceback.format_exc(),
            "no_automatic_extra_replay": True,
        })
        raise


if __name__ == "__main__":
    main()
