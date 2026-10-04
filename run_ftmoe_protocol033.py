"""Protocol-033 single-arm replay runner.

C_fixed5 and D_guard_budget are launched as separate sequential processes by the
workflow, consuming the same frozen 5969-row input bundle.  No simulator code is
called here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
from ftmoe_protocol025_session import Protocol025FixedSession
from ftmoe_protocol033_guard_budget import Protocol033DynamicSession, P033_CONFIG

COMPARATORS = ("C_fixed5", "D_guard_budget")
RECURRENCE = ("U_rec1", "V_rec1", "U_rec2", "V_rec2", "U_rec3", "V_rec3")


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    p31.write_json(path, value)


def _cgroup_limit_bytes():
    vals = []
    for p in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            t = p.read_text().strip()
            if t and t != "max":
                v = int(t)
                if 0 < v < (1 << 60): vals.append(v)
        except Exception:
            pass
    return min(vals) if vals else None


def _rss_tree():
    p = psutil.Process(); total = p.memory_info().rss
    for c in p.children(recursive=True):
        try: total += c.memory_info().rss
        except Exception: pass
    return int(total)


def validate_input(stream, registration_path, input_lock_path):
    root = Path(stream)
    reg = J(registration_path)
    lock = J(input_lock_path)
    manifest = J(root / "manifest.json")
    frozen = J(root / "frozen_data_manifest.json")
    if reg.get("protocol") != "033" or int(reg.get("plan_revision", -1)) != 1:
        raise AssertionError("Protocol033 frozen registration mismatch")
    if reg.get("scenario_id") != "protocol033_rare_recurrence_5969_v1":
        raise AssertionError("Protocol033 scenario mismatch")
    if lock.get("protocol") != "033" or lock.get("locked") is not True or lock.get("input_audit_passed") is not True:
        raise RuntimeError("Protocol033 input lock is not eligible")
    digest = p31.sha(root / "stream.npz")
    expected = reg["generation"].get("expected_stream_sha256")
    if not expected or not (digest == expected == manifest.get("stream_sha256") == frozen.get("stream_sha256") == lock.get("stream_sha256")):
        raise AssertionError("Protocol033 frozen stream identity mismatch")
    if int(manifest.get("steps", -1)) != 5968 or int(manifest.get("guard_rows", -1)) != 1 or int(manifest.get("total_rows", -1)) != 5969:
        raise AssertionError("Protocol033 input geometry mismatch")
    if reg["evaluation"]["primary_windows"] != list(RECURRENCE):
        raise AssertionError("Protocol033 recurrence registration mismatch")
    return digest, manifest, lock, reg


def topology_snapshot(session, cursor, label, pdefs):
    bank = session.model.learner
    ctrl = session.lifecycle_controller
    return {
        "label": str(label), "cursor": int(cursor),
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

    bundle = s4.build_replay(Path(stream))
    if bundle["steps"] != 5968 or bundle["manifest"].get("stream_sha256") != stream_sha:
        raise AssertionError("Protocol033 replay identity mismatch")
    pdefs = p31.phases(manifest)
    pmap = {p["name"]: p for p in pdefs}
    snapshot_points = {
        int(pmap["U_first"]["end"]): "after_U_first",
        int(pmap["V_first"]["end"]): "after_V_first",
        int(pmap["W_long"]["end"]): "after_W_long",
    }
    for phase in RECURRENCE:
        snapshot_points[int(pmap[phase]["start"])] = "before_" + phase

    guard = p31.build_guard(bundle)
    guard["meta"].update({
        "protocol": "033", "plan_revision": 1,
        "role": "F0_known_normal_regression_guard_protocol033",
        "normal_nll_rule": "candidate <= live + max(0.02*live,0.01) + 1e-6",
        "candidate_training_on_guard_rows": False,
    })
    write_json(out / "guard_manifest.json", guard["meta"])

    runtime_registration = {
        "protocol": "033", "plan_revision": 1, "comparator": name,
        "scenario_id": reg["scenario_id"], "data_revision": reg["data_revision"],
        "stream_sha256": stream_sha, "replay_seed": 700, "model_seed": 1,
        "development_only": True, "statistical_confirmation": False,
        "input_lock_sha256": p31.sha(input_lock),
        "birth_start_matured": 600, "birth_every_matured": 1000,
    }
    common = dict(
        seed=1, replay_bundle=bundle, budget=p31.budget(), out_dir=out,
        run_id=f"protocol033_{run_id}_{name}", stream_dir=Path(stream),
        phase_defs=pdefs, stream_sha=stream_sha, registration=runtime_registration,
        learning_rate=1e-4,
    )
    session = (Protocol033DynamicSession(guard_anchor=guard, v2c_config=P033_CONFIG, **common)
               if name == "D_guard_budget" else Protocol025FixedSession("C_fixed5", **common))
    init_prefix = p31.shared_prefix_hash(session.model, 4)
    write_json(out / "initialization.json", {
        "shared_first4_expert_and_router_rows_sha256": init_prefix,
        "model_seed": 1, "registered_shared_prefix_initialization": True,
    })

    vm = psutil.virtual_memory(); cg = _cgroup_limit_bytes()
    effective = min(int(vm.total), int(cg)) if cg else int(vm.total)
    soft = int(effective * 0.8)
    mem_samples = []
    def mem_sample(label, cursor):
        available = int(psutil.virtual_memory().available); r = _rss_tree()
        mem_samples.append({"label": label, "cursor": int(cursor), "process_tree_rss_bytes": r, "system_available_bytes": available})
        if r >= soft:
            raise MemoryError("Protocol033 replay crossed registered 80% effective-memory soft limit")

    snapshots = []
    wall0 = time.perf_counter(); cpu0 = time.process_time()
    peak_p = p31.param_bytes(session); peak_o = p31.opt_bytes(session)
    peak_live = 5 if name == "C_fixed5" else 4
    peak_resident = peak_live
    mem_sample("replay_start", 0)
    for i in range(session.steps):
        if name == "D_guard_budget" and i in snapshot_points:
            snapshots.append(topology_snapshot(session, i, snapshot_points[i], pdefs))
        session.step()
        if i % 20 == 0 or i + 1 == session.steps:
            mem_sample("interval", i)
        if i % 64 == 0 or i + 1 == session.steps:
            peak_p = max(peak_p, p31.param_bytes(session)); peak_o = max(peak_o, p31.opt_bytes(session))
        if name == "D_guard_budget":
            bank = session.model.learner
            peak_live = max(peak_live, len(bank.ids)); peak_resident = max(peak_resident, bank.resident_count())
            if bank.resident_count() > 8:
                raise AssertionError("Protocol033 resident expert capacity exceeded")

    if name == "D_guard_budget":
        session.lifecycle_controller.mark_stream_end()
    session.finish()
    wall = time.perf_counter() - wall0; cpu = time.process_time() - cpu0
    session.save()
    mem_sample("replay_complete", session.steps)

    prob = session.predictions["probability"]
    cls = session.predictions["class_probability"]
    y = session.predictions["labels"]
    if (y < 0).any() or (session.predictions["raw_labels"] < 0).any():
        raise AssertionError("Protocol033 unsettled model outputs")
    cost = {
        "wall_seconds": float(wall), "cpu_seconds": float(cpu),
        "max_rss_bytes": max(x["process_tree_rss_bytes"] for x in mem_samples),
        "effective_memory_limit_bytes": effective, "memory_soft_limit_bytes": soft,
        "minimum_system_available_bytes": min(x["system_available_bytes"] for x in mem_samples),
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
    write_json(out / "memory_profile.json", {
        "protocol": "033", "comparator": name, "samples_every_intervals": 20,
        "effective_memory_limit_bytes": effective, "soft_limit_bytes": soft,
        "samples": mem_samples, "peak_process_tree_rss_bytes": cost["max_rss_bytes"],
    })

    summary = {
        "protocol": "033", "plan_revision": 1, "comparator": name,
        "run_id": str(run_id), "completed": True, "development_only": True,
        "statistical_confirmation": False, "stream_sha256": stream_sha,
        "initialization_shared_prefix_sha256": init_prefix,
        "manifest": session.comparator_manifest(),
        "full": p31.metrics(prob, cls, y), "phases": [], "cost": cost,
    }
    for p in pdefs:
        a, b = int(p["start"]), int(p["end"]); yy = y[a:b]
        summary["phases"].append({
            "phase": p["name"], "service": p["regime"], "intervals": [a, b],
            "positive_host_steps": int((yy > 0).sum()), "negative_host_steps": int((yy == 0).sum()),
            **p31.metrics(prob[a:b], cls[a:b], yy),
        })

    if name == "D_guard_budget":
        ctrl = session.lifecycle_controller; bank = session.model.learner
        events = list(ctrl.events); records = list(ctrl.candidate_records.values())
        ec = Counter(e["kind"] for e in events); byphase = defaultdict(Counter)
        for e in events:
            byphase[p31.phase_name(e.get("cursor", 0), pdefs)][e["kind"]] += 1
        due = ctrl.due_conservation()
        if not due["birth"]["conserved"] or not due["reuse_nonblocking"]["due_conserved"] or not due["reuse_nonblocking"]["started_conserved"]:
            raise AssertionError("Protocol033 lifecycle due accounting not conserved")
        if ctrl.purges != 0:
            raise AssertionError("Protocol033 unexpectedly purged accepted memory")
        candidate_sources = []
        for rec in records:
            idx = list(rec.get("training_indices") or []); decision = deepcopy(rec.get("decision"))
            candidate_sources.append({
                "candidate_id": rec.get("candidate_id"), "created_cursor": rec.get("created_cursor"),
                "accepted": rec.get("accepted"), "cancelled_by_reuse": bool(rec.get("cancelled_by_reuse")),
                "training_indices_count": len(idx),
                "training_source_phases": sorted({p31.phase_name(i, pdefs) for i in idx}),
                "accepted_cursor": rec.get("accepted_cursor"), "decision": decision,
            })
        lifecycle = {
            "birth_candidates_created": len(records),
            "birth_candidates_accepted": sum(r.get("accepted") is True and not r.get("cancelled_by_reuse") for r in records),
            "retirements": int(ctrl.retirements), "reactivations": int(ctrl.reactivations), "purges": int(ctrl.purges),
            "capacity_preserve_skips": int(ctrl.capacity_preserve_skips),
            "accepted_birth_ids": [str(x) for x in ctrl.accepted_ids],
            "memory_registry_ids": sorted([str(k) for k in ctrl.specialist_memory.keys()], key=int),
            "real_dormant_expert_ids": sorted([str(k) for k in bank.dormant_experts.keys()], key=int),
            "active_specialist_id": ctrl.active_specialist_id,
            "reuse_records": ctrl.reuse_records, "reuse_outcomes": ctrl.reuse_outcomes,
            "protocol033_reuse_event_records": deepcopy(ctrl.p033_reuse_event_records),
            "due_accounting": due, "event_counts": dict(ec),
            "event_counts_by_phase": {k: dict(v) for k, v in byphase.items()},
            "candidate_sources": candidate_sources, "topology_snapshots": snapshots,
            "final_topology": bank.topology_manifest(), "extra_compute": dict(ctrl.extra_compute),
        }
        summary["lifecycle"] = lifecycle
        p31.write_jsonl(out / "lifecycle.jsonl", events)
        write_json(out / "candidate_records.json", records)
        write_json(out / "reuse_records.json", ctrl.reuse_records)
        write_json(out / "reuse_event_records.json", ctrl.p033_reuse_event_records)
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
        print(json.dumps(p31.json_ready({"completed": True, "comparator": a.arm, "full_detection": result["full"]["detection"]}), indent=2))
    except Exception:
        root = Path(a.arm_dir); root.mkdir(parents=True, exist_ok=True)
        write_json(root / "runner_exception.json", {
            "protocol": "033", "comparator": a.arm, "traceback": traceback.format_exc(),
            "no_automatic_extra_replay": True,
        })
        raise


if __name__ == "__main__":
    main()
