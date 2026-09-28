"""Independent pre-model eligibility audit for Protocol-033.

The audit is run after generation and assembly processes have exited.  It
recomputes overload ratios and raw labels from post_totals/capacities, checks
prediction-index target coverage raw[t+1], freezes hashes, and stops the
experiment on any required gate failure.  It never rerolls or tunes the stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psutil

import prepare_ftmoe_protocol033_stream as P

RECURRENCE = ("U_rec1", "V_rec1", "U_rec2", "V_rec2", "U_rec3", "V_rec3")


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def W(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def rss():
    p = psutil.Process()
    total = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            total += c.memory_info().rss
        except Exception:
            pass
    return int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-root", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--evidence-root", required=True, type=Path)
    args = ap.parse_args()
    genroot, data, evidence = args.generation_root, args.data_root, args.evidence_root
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    samples = []

    def sample(label, row=None):
        vm = psutil.virtual_memory()
        samples.append({"label": label, "row": row, "process_tree_rss_bytes": rss(), "system_available_bytes": int(vm.available)})

    sample("audit_start")
    reg = P.registration()
    reg_sha = P.json_sha(P.REGISTRATION_PATH)
    manifest = J(data / "manifest.json")
    resume = J(genroot / "resume_manifest.json")
    generation = J(genroot / "generation_manifest.json")
    chunks = list(resume.get("chunks") or [])

    gates = {}
    details = {}
    gates["registration_identity"] = bool(
        manifest.get("protocol") == "033" and manifest.get("scenario_id") == P.SCENARIO_ID
        and resume.get("protocol") == "033" and generation.get("protocol") == "033"
        and resume.get("registration_sha256") == reg_sha
        and resume.get("source_sha256") == P.source_identity())

    try:
        covered = int(P._verify_chunks(genroot, chunks))
        chunk_geometry = (covered == 5969 and len(chunks) == 30 and
                          int(chunks[-1]["end"]) - int(chunks[-1]["start"]) == 169)
    except Exception as exc:
        covered = -1
        chunk_geometry = False
        details["chunk_error"] = str(exc)
    gates["all_rows_and_chunks_complete"] = bool(chunk_geometry)

    # Memory-map assembled fields.  Each validation block is at most 200 rows.
    field = data / "fields"
    post = np.load(field / "post_totals.npy", mmap_mode="r")
    caps = np.load(field / "capacities.npy", mmap_mode="r")
    saved_ratio = np.load(field / "overload_ratio.npy", mmap_mode="r")
    raw = np.load(field / "raw_labels.npy", mmap_mode="r")
    schedules = np.load(field / "schedules.npy", mmap_mode="r")
    before = np.load(field / "before_placement.npy", mmap_mode="r")
    after = np.load(field / "after_placement.npy", mmap_mode="r")
    host = np.load(field / "host_features.npy", mmap_mode="r")
    creation = np.load(field / "creation_ids.npy", mmap_mode="r")
    audit_phase = np.load(field / "audit_phase_ids.npy", mmap_mode="r")
    audit_service = np.load(field / "audit_service_ids.npy", mmap_mode="r")

    shapes_ok = all(int(x.shape[0]) == 5969 for x in
                    (post, caps, saved_ratio, raw, schedules, before, after, host, creation, audit_phase, audit_service))
    gates["exact_5969_row_geometry"] = bool(shapes_ok)

    ratio_exact = True
    label_exact = True
    finite_features = True
    capacity_ok = True
    scheduler_ok = True
    placement_ok = True
    for a in range(0, 5969, 200):
        b = min(5969, a + 200)
        c = np.asarray(caps[a:b], dtype=np.float64)
        p = np.asarray(post[a:b], dtype=np.float64)
        sr = np.asarray(saved_ratio[a:b], dtype=np.float64)
        y = np.asarray(raw[a:b], dtype=np.int64)
        calc = p / c
        calc_y = np.where((calc > 1.0).any(-1), calc.argmax(-1) + 1, 0)
        ratio_exact = ratio_exact and bool(np.array_equal(calc, sr))
        label_exact = label_exact and bool(np.array_equal(calc_y, y))
        finite_features = finite_features and bool(np.isfinite(np.asarray(host[a:b])).all())
        capacity_ok = capacity_ok and bool(np.isfinite(c).all() and (c > 0).all())
        s = np.asarray(schedules[a:b], dtype=np.float64)
        scheduler_ok = scheduler_ok and bool(np.isfinite(s).all() and (s >= -1e-7).all() and np.allclose(s.sum(-1), 1.0, atol=1e-5))
        bp = np.asarray(before[a:b]); ap_ = np.asarray(after[a:b])
        placement_ok = placement_ok and bool(((bp >= -1) & (bp < 16)).all() and ((ap_ >= -1) & (ap_ < 16)).all())
        sample("audit_block", a)
    gates["overload_ratio_recomputed_exact_from_post_totals_and_capacity"] = bool(ratio_exact)
    gates["label_recompute_exact"] = bool(label_exact)
    gates["finite_causal_features"] = bool(finite_features)
    gates["capacity_audit"] = bool(capacity_ok)
    gates["scheduler_audit"] = bool(scheduler_ok)
    gates["placement_audit"] = bool(placement_ok)

    common = np.load(data / "common_observable_features.npy", mmap_mode="r")
    gates["common_features_finite"] = bool(common.shape == (5969, 16, 9) and np.isfinite(common).all())

    phases = P.phase_table(reg)
    per_phase = {}
    recurrence = {}
    class_coverage = True
    phase_id_ok = True
    service_id_ok = True
    for phase_index, p in enumerate(phases):
        a, b = int(p["start"]), int(p["end"])
        # Prediction indices [a,b) must evaluate targets raw[a+1:b+1].
        y = np.asarray(raw[a + 1:b + 1], dtype=np.int64)
        row = {
            "prediction_intervals": [a, b], "target_raw_rows": [a + 1, b + 1],
            "host_steps": int(y.size), "positive_host_steps": int((y > 0).sum()),
            "negative_host_steps": int((y == 0).sum()),
            "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
            "logical_service": p["logical_service"], "source_service": p["service"],
        }
        per_phase[p["name"]] = row
        if p["logical_service"] is not None:
            ok = row["positive_host_steps"] >= 32 and row["negative_host_steps"] >= 32
            class_coverage = class_coverage and ok
        if p["name"] in RECURRENCE:
            recurrence[p["name"]] = {**row,
                "positive_min_32": row["positive_host_steps"] >= 32,
                "negative_min_32": row["negative_host_steps"] >= 32,
                "ap_defined": row["positive_host_steps"] > 0 and row["negative_host_steps"] > 0}
        # audit_phase/service are creation-time annotations for row t, not target t+1.
        phase_id_ok = phase_id_ok and bool((np.asarray(audit_phase[a:b]) == phase_index).all())
        expected_sid = -1 if p["service"] is None else int(P.SERVICE_MECHANISM_IDS[p["service"]])
        service_id_ok = service_id_ok and bool((np.asarray(audit_service[a:b]) == expected_sid).all())
    gates["all_nonbaseline_phases_min32_positive_and_negative_target_hoststeps"] = bool(class_coverage)
    gates["six_recurrence_windows_present"] = bool(set(recurrence) == set(RECURRENCE))
    gates["six_recurrence_windows_min32_positive_and_negative"] = bool(
        set(recurrence) == set(RECURRENCE) and all(x["positive_min_32"] and x["negative_min_32"] and x["ap_defined"] for x in recurrence.values()))
    gates["phase_annotation_matches_registered_prediction_timeline"] = bool(phase_id_ok)
    gates["service_annotation_matches_registered_creation_timeline"] = bool(service_id_ok)
    gates["last_prediction_has_raw_t_plus_1_support"] = bool(phases[-1]["end"] == 5968 and raw.shape[0] == 5969)

    guard_idx = np.arange(0, 299, dtype=np.int64)
    guard_idx = guard_idx[guard_idx % 5 == 0]
    guard_y = np.asarray(raw[guard_idx + 1], dtype=np.int64).reshape(-1)
    normal_guard = int((guard_y == 0).sum())
    gates["normal_guard_available"] = normal_guard > 0

    events = J(data / "events_final.json")
    counts = {logical: int(sum(1 for e in events if e.get("service_id") == source)) for logical, source in P.SOURCE_SERVICE.items()}
    gates["events_U_V_W_present"] = all(counts[x] > 0 for x in ("U", "V", "W"))

    switches = J(data / "applied_switches.json")
    expected_switches = [(int(p["start"]), p["name"], p["service"], p["logical_service"]) for p in phases[1:]]
    actual_switches = [(int(x["interval"]), x["phase"], x.get("service"), x.get("logical_service")) for x in switches]
    gates["actual_phase_switches_match_registration"] = actual_switches == expected_switches

    source_param_match = {}
    for logical, source in P.SOURCE_SERVICE.items():
        implemented = P.SERVICE_LAWS[source]["registered_parameters"]
        registered = reg["services"][logical]["parameters"]
        source_param_match[logical] = all(
            str(k) in registered and float(registered[k]) == float(v)
            for k, v in implemented.items() if isinstance(v, (int, float)))
    gates["registered_physical_parameters_match_implementation"] = all(source_param_match.values())

    audit_pass = bool(all(gates.values()))
    audit = {
        "protocol": "033", "plan_revision": 1, "kind": "pre_model_input_eligibility_audit",
        "model_results_seen": False, "audit_pass": audit_pass,
        "reroll_after_failure_allowed": False, "no_intensity_retuning": True,
        "target_semantics": "prediction indices [start,end); labels raw_labels[start+1:end+1] on same host; maturity t+2",
        "gates": gates, "per_phase_target_coverage": per_phase,
        "recurrence128_target_coverage": recurrence,
        "normal_guard": {"prediction_indices": guard_idx.tolist(), "normal_host_steps": normal_guard,
                         "positive_host_steps": int((guard_y > 0).sum()), "positive_required": False},
        "service_event_counts": counts,
        "source_parameter_match": source_param_match,
        "expected_switches": expected_switches,
        "actual_switches": actual_switches,
        "details": details,
    }
    W(evidence / "input_audit.json", audit)

    stream_sha = P.sha(data / "stream.npz")
    common_sha = P.sha(data / "common_observable_features.npy")
    events_sha = P.sha(data / "events_final.json")
    switches_sha = P.sha(data / "applied_switches.json")
    frozen_reg = json.loads(json.dumps(reg))
    frozen_reg["status"] = "frozen_pre_model" if audit_pass else "blocked_pre_model_audit"
    frozen_reg["generation"]["expected_stream_sha256"] = stream_sha
    frozen_reg["generation"]["actual_chunk_sha256"] = [x["sha256"] for x in chunks]
    W(evidence / "registration_frozen.json", frozen_reg)
    frozen_reg_sha = P.sha(evidence / "registration_frozen.json")

    lock = {
        "protocol": "033", "plan_revision": 1,
        "locked": bool(audit_pass), "input_audit_passed": bool(audit_pass),
        "model_free_eligibility_passed": bool(audit_pass),
        "stream_sha256": stream_sha, "common_features_sha256": common_sha,
        "events_sha256": events_sha, "applied_switches_sha256": switches_sha,
        "registration_original_sha256": reg_sha, "registration_frozen_sha256": frozen_reg_sha,
        "checkpoint_sha256": resume.get("state_sha256"),
        "checkpoint_next_t": resume.get("next_t"),
        "source_sha256": resume.get("source_sha256"),
        "chunk_sha256": [x["sha256"] for x in chunks],
        "chunk_count": len(chunks), "total_rows": 5969, "scored_rows": 5968,
        "failed_audit_allows_model_runs": False, "reroll_after_failure": False,
    }
    W(evidence / "input_lock.json", lock)

    frozen_manifest = {
        "protocol": "033", "stream_sha256": stream_sha,
        "registration_frozen_sha256": frozen_reg_sha,
        "input_lock_sha256": P.sha(evidence / "input_lock.json"),
        "input_audit_sha256": P.sha(evidence / "input_audit.json"),
        "common_features_sha256": common_sha, "events_sha256": events_sha,
        "chunk_sha256": lock["chunk_sha256"], "eligibility_passed": audit_pass,
    }
    W(data / "frozen_data_manifest.json", frozen_manifest)
    manifest["audit_pass"] = audit_pass
    manifest["stream_sha256"] = stream_sha
    manifest["frozen_data_manifest"] = "frozen_data_manifest.json"
    W(data / "manifest.json", manifest)

    sample("audit_complete")
    vm = psutil.virtual_memory()
    cg = P._cgroup_limit_bytes()
    effective = min(int(vm.total), int(cg)) if cg else int(vm.total)
    mem = {
        "protocol": "033", "stage": "audit", "effective_memory_limit_bytes": effective,
        "soft_limit_bytes": int(effective * 0.8), "samples": samples,
        "peak_process_tree_rss_bytes": max(x["process_tree_rss_bytes"] for x in samples),
        "minimum_system_available_bytes": min(x["system_available_bytes"] for x in samples),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    W(evidence / "memory_profile_audit.json", mem)
    print(json.dumps({"protocol": "033", "audit_pass": audit_pass, "gates": gates, "stream_sha256": stream_sha}, indent=2))
    if not audit_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
