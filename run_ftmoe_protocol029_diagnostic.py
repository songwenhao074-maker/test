"""Protocol-029: one unchanged Protocol-028 D replay with passive memory utility audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol027_pilot as p27
import run_ftmoe_protocol028_pilot as p28
from ftmoe_protocol024_eval import binary_detection_metrics
from ftmoe_protocol027_data import EXPECTED_STREAM_SHA, REVISION_ID, verify_frozen
from ftmoe_protocol029_memory_utility import Protocol029DiagnosticSession

RECURRENCE = tuple(p27.RECURRENCE)
SOURCE_028_RUN = 35705211072
SOURCE_028_ARTIFACT = 10684617851
SOURCE_028_DIGEST = "sha256:e3be1c97d0f629c9ed40f61cfd4561b6b04e37b68357a2c92a23ac8f14e2e27e"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def load_npz(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def softmax_positive(logits):
    value = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), -1)[..., 1]
    return value.detach().cpu().numpy().astype(np.float32)


def _discrete_event(event):
    keys = (
        "kind", "cursor", "matured_count", "phase", "candidate_id", "expert_id",
        "parent_id", "old_id", "new_id", "replacement_id", "accepted",
        "reject_reasons", "reason", "transition_kind", "prediction_index",
        "step", "preserved_expert_ids", "shadow_created", "purge_performed",
    )
    return {key: event.get(key) for key in keys if key in event}


def save_passive_predictions(ctrl, out_path):
    candidates = ctrl.passive_prediction_rows
    generalists = ctrl.generalist_prediction_rows
    if candidates:
        c_idx = np.asarray([r["prediction_index"] for r in candidates], dtype=np.int32)
        c_id = np.asarray([int(r["expert_id"]) for r in candidates], dtype=np.int32)
        c_det = np.stack([r["detection_logits"] for r in candidates]).astype(np.float32)
        c_cls = np.stack([r["class_logits"] for r in candidates]).astype(np.float32)
        c_route = np.asarray([r["route_mean"] for r in candidates], dtype=np.float32)
    else:
        c_idx = np.empty((0,), dtype=np.int32)
        c_id = np.empty((0,), dtype=np.int32)
        c_det = np.empty((0, 16, 2), dtype=np.float32)
        c_cls = np.empty((0, 16, 3), dtype=np.float32)
        c_route = np.empty((0,), dtype=np.float32)
    g_idx = np.asarray([r["prediction_index"] for r in generalists], dtype=np.int32)
    g_det = np.stack([r["detection_logits"] for r in generalists]).astype(np.float32)
    g_cls = np.stack([r["class_logits"] for r in generalists]).astype(np.float32)
    g_route = np.asarray([r["route_mean"] for r in generalists], dtype=np.float32)
    np.savez_compressed(
        out_path,
        candidate_prediction_index=c_idx,
        candidate_expert_id=c_id,
        candidate_detection_logits=c_det,
        candidate_class_logits=c_cls,
        candidate_route_mean=c_route,
        generalists_prediction_index=g_idx,
        generalists_detection_logits=g_det,
        generalists_class_logits=g_cls,
        generalists_route_mean=g_route,
        recorded_before_label=np.asarray([True], dtype=np.bool_),
    )


def compact_opportunity_audit(opportunities):
    rows = []
    for o in opportunities:
        row = {k: v for k, v in o.items() if k not in ("candidate_validation", "generalists_validation")}
        row["candidate_validation"] = {}
        for key, rec in o["candidate_validation"].items():
            row["candidate_validation"][key] = {
                "validation_pair_count": len(rec.get("pairs", [])),
                "censored": bool(rec.get("censored")),
                "censor_reason": rec.get("censor_reason"),
                "censor_prediction_index": rec.get("censor_prediction_index"),
                "decision": rec.get("decision"),
            }
        row["generalists_validation"] = {
            "validation_pair_count": len(o["generalists_validation"].get("pairs", [])),
            "censored": bool(o["generalists_validation"].get("censored")),
        }
        rows.append(row)
    return rows


def summarize_opportunities(opportunities):
    total = len(opportunities)
    no_memory = sum(not o["frozen_dormant_ids"] for o in opportunities)
    stream_censored = sum(o.get("completion_reason") == "stream_end_censored" for o in opportunities)
    with_candidate = [o for o in opportunities if o["frozen_dormant_ids"]]
    complete = sum(
        any((r.get("decision") or {}).get("available") for r in o["candidate_validation"].values())
        for o in with_candidate
    )
    passing = [o for o in opportunities if o.get("has_passing_candidate")]
    busy = [o for o in passing if o.get("passing_blocked_by_busy")]
    similarity = [o for o in passing if o.get("passing_blocked_by_similarity")]
    selected_pass = [o for o in passing if o.get("selected_candidate_passed")]
    intersections = Counter()
    for o in passing:
        flags = []
        if o.get("passing_blocked_by_busy"):
            flags.append("busy")
        if o.get("passing_blocked_by_similarity"):
            flags.append("similarity")
        if o.get("selected_candidate_passed"):
            flags.append("selected_passing")
        if not flags:
            flags.append("other_or_selected_nonpassing")
        intersections["+".join(flags)] += 1
    per_expert = defaultdict(lambda: {
        "opportunities_present": 0, "completed_validations": 0,
        "all_gates_pass": 0, "censored": 0,
        "similarity_ok": 0, "highest_similarity": 0,
    })
    for o in opportunities:
        sim = o.get("similarity", {}).get("candidates", {})
        for key, row in o["candidate_validation"].items():
            x = per_expert[key]
            x["opportunities_present"] += 1
            if row.get("censored"):
                x["censored"] += 1
            if row.get("decision"):
                x["completed_validations"] += 1
                x["all_gates_pass"] += int(bool(row["decision"].get("all_gates_pass")))
            x["similarity_ok"] += int(bool(sim.get(key, {}).get("similarity_ok")))
            x["highest_similarity"] += int(bool(sim.get(key, {}).get("is_highest_similarity")))
    return {
        "reuse_due_opportunities": total,
        "no_memory_opportunities": no_memory,
        "opportunities_with_memory": len(with_candidate),
        "opportunities_with_at_least_one_completed_candidate_validation": complete,
        "stream_end_censored_opportunities": stream_censored,
        "opportunities_with_any_all_gates_candidate": len(passing),
        "passing_candidate_opportunities_blocked_by_busy": len(busy),
        "passing_candidate_opportunities_blocked_by_similarity_or_rank": len(similarity),
        "original_selected_candidate_passed": len(selected_pass),
        "passing_condition_intersections": dict(intersections),
        "per_expert": dict(per_expert),
        "post_hoc_oracle_diagnostic": True,
    }


def window_memory_utility(session, ctrl, stream, source_comparison):
    labels = session.predictions["labels"]
    phase_map = {p["name"]: p for p in p27.phases(load_json(Path(stream) / "manifest.json"))}
    candidate_by_expert = defaultdict(dict)
    for row in ctrl.passive_prediction_rows:
        candidate_by_expert[str(row["expert_id"])][int(row["prediction_index"])] = row
    general_by_index = {int(r["prediction_index"]): r for r in ctrl.generalist_prediction_rows}
    source_c = {r["phase"]: r["C_fixed5"] for r in source_comparison["recurrence_first100"]}
    all_experts = sorted(set(ctrl.accepted_ids) | set(ctrl.specialist_memory.keys()), key=int)
    rows = []
    for phase_name in RECURRENCE:
        phase = phase_map[phase_name]
        start, end = int(phase["start"]), min(int(phase["end"]), int(phase["start"]) + 100)
        idx = np.arange(start, end, dtype=np.int64)
        yy = labels[idx]
        live = binary_detection_metrics(session.predictions["probability"][idx], yy, .5)
        g_prob = np.stack([softmax_positive(general_by_index[int(i)]["detection_logits"]) for i in idx])
        general = binary_detection_metrics(g_prob, yy, .5)
        expert_rows = []
        for key in all_experts:
            covered = [int(i) for i in idx if int(i) in candidate_by_expert.get(key, {})]
            if not covered:
                expert_rows.append({
                    "expert_id": key, "coverage_intervals": 0, "coverage_rows": 0,
                    "coverage_fraction": 0.0, "candidate": None,
                    "live_same_rows": None, "generalists_same_rows": None,
                    "candidate_minus_live_ap": None,
                    "candidate_minus_generalists_ap": None,
                    "C_fixed5_full_window": None,
                    "direct_C_comparison_allowed": False,
                })
                continue
            cidx = np.asarray(covered, dtype=np.int64)
            cprob = np.stack([
                softmax_positive(candidate_by_expert[key][int(i)]["detection_logits"])
                for i in covered
            ])
            gsub = np.stack([softmax_positive(general_by_index[int(i)]["detection_logits"]) for i in covered])
            cy = labels[cidx]
            cm = binary_detection_metrics(cprob, cy, .5)
            lm = binary_detection_metrics(session.predictions["probability"][cidx], cy, .5)
            gm = binary_detection_metrics(gsub, cy, .5)
            full = len(covered) == (end - start)
            expert_rows.append({
                "expert_id": key,
                "coverage_intervals": len(covered),
                "coverage_rows": int(len(covered) * labels.shape[1]),
                "coverage_fraction": float(len(covered) / max(end - start, 1)),
                "covered_interval_range": [min(covered), max(covered) + 1],
                "candidate": cm,
                "live_same_rows": lm,
                "generalists_same_rows": gm,
                "candidate_minus_live_ap": None if cm["ap"] is None or lm["ap"] is None else float(cm["ap"] - lm["ap"]),
                "candidate_minus_generalists_ap": None if cm["ap"] is None or gm["ap"] is None else float(cm["ap"] - gm["ap"]),
                "C_fixed5_full_window": source_c[phase_name] if full else None,
                "direct_C_comparison_allowed": bool(full),
                "partial_coverage_not_comparable_to_C_full100": not full,
            })
        full_candidates = [r for r in expert_rows if r["direct_C_comparison_allowed"] and r["candidate"] and r["candidate"]["ap"] is not None]
        oracle = None
        if full_candidates:
            best = max(full_candidates, key=lambda r: r["candidate"]["ap"])
            oracle = {
                "expert_id": best["expert_id"],
                "candidate_ap": best["candidate"]["ap"],
                "post_hoc_oracle_diagnostic": True,
                "not_online_policy": True,
            }
        rows.append({
            "phase": phase_name,
            "service": phase.get("regime"),
            "intervals": [start, end],
            "live": live,
            "generalists_only": general,
            "experts": expert_rows,
            "full_window_post_hoc_best_expert": oracle,
        })
    return rows


def compare_consistency(session, ctrl, source_root):
    source_root = Path(source_root)
    source_pred = load_npz(source_root / "D_memory_protected" / "predictions.npz")
    current_prob = session.predictions["probability"]
    current_cls = session.predictions["class_probability"]
    prob_err = float(np.max(np.abs(current_prob.astype(np.float64) - source_pred["probability"].astype(np.float64))))
    cls_err = float(np.max(np.abs(current_cls.astype(np.float64) - source_pred["class_probability"].astype(np.float64))))
    labels_equal = bool(np.array_equal(session.predictions["labels"], source_pred["labels"]))
    raw_equal = bool(np.array_equal(session.predictions["raw_labels"], source_pred["raw_labels"]))
    source_summary = load_json(source_root / "D_memory_protected" / "summary.json")
    current_ap = binary_detection_metrics(current_prob, session.predictions["labels"], .5)["ap"]
    source_ap = source_summary["full"]["detection"]["ap"]
    ap_diff = None if current_ap is None or source_ap is None else float(abs(current_ap - source_ap))
    source_events = [_discrete_event(x) for x in load_jsonl(source_root / "D_memory_protected" / "lifecycle.jsonl")]
    current_events = [_discrete_event(x) for x in ctrl.events]
    lifecycle_equal = source_events == current_events
    source_life = load_json(source_root / "D_memory_protected" / "lifecycle_summary.json")
    final_counts_equal = all([
        int(source_life.get("births", -1)) == int(Counter(e["kind"] for e in ctrl.events).get("candidate_accepted", 0)),
        int(source_life.get("retirements", -1)) == int(ctrl.retirements),
        int(source_life.get("reactivations", -1)) == int(ctrl.reactivations),
        int(source_life.get("purges", -1)) == int(ctrl.purges),
        int(source_life.get("capacity_preserve_skips", -1)) == int(ctrl.capacity_preserve_skips),
        list(source_life.get("accepted_birth_ids", [])) == list(ctrl.accepted_ids),
        list(source_life.get("resident_memory_ids", [])) == sorted(ctrl.specialist_memory.keys(), key=int),
    ])
    valid = bool(
        labels_equal and raw_equal and prob_err <= 1e-6 and cls_err <= 1e-6
        and ap_diff is not None and ap_diff <= 1e-6
        and lifecycle_equal and final_counts_equal
        and ctrl.passive_state_check_failures == 0
    )
    return {
        "diagnostic_valid": valid,
        "required_probability_max_abs_error": 1e-6,
        "probability_max_abs_error": prob_err,
        "class_probability_max_abs_error": cls_err,
        "labels_exact": labels_equal,
        "raw_labels_exact": raw_equal,
        "source_full_ap": source_ap,
        "diagnostic_full_ap": current_ap,
        "full_ap_absolute_difference": ap_diff,
        "required_full_ap_absolute_difference_max": 1e-6,
        "discrete_lifecycle_events_exact": lifecycle_equal,
        "source_discrete_event_count": len(source_events),
        "diagnostic_discrete_event_count": len(current_events),
        "final_lifecycle_counts_exact": final_counts_equal,
        "passive_state_checks": int(ctrl.passive_state_checks),
        "passive_state_check_failures": int(ctrl.passive_state_check_failures),
    }


def run(args):
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=False)
    stream = Path(args.stream)
    verify_frozen(stream)
    reg = load_json(args.registration)
    if reg.get("protocol") != "029" or reg.get("replay_seed") != 700 or reg.get("model_seed") != 1:
        raise AssertionError("Protocol029 registration mismatch")
    if reg.get("rerun_C") is not False or reg.get("new_training_arms") != 0:
        raise AssertionError("Protocol029 may not rerun C or add a training arm")
    source_root = Path(args.source_028_root)
    for required in (
        source_root / "D_memory_protected" / "predictions.npz",
        source_root / "D_memory_protected" / "lifecycle.jsonl",
        source_root / "D_memory_protected" / "summary.json",
        source_root / "D_memory_protected" / "lifecycle_summary.json",
        source_root / "comparison.json",
    ):
        if not required.is_file():
            raise FileNotFoundError("missing Protocol028 source evidence: %s" % required)

    bundle = s4.build_replay(stream)
    bundle["stream_dir"] = str(stream)
    stream_sha = p27.sha(stream / "stream.npz")
    if stream_sha != EXPECTED_STREAM_SHA or bundle["manifest"].get("stream_sha256") != EXPECTED_STREAM_SHA:
        raise AssertionError("Protocol029 frozen stream mismatch")
    phases = p27.phases(bundle["manifest"])
    guard = p28.build_guard(bundle)
    write_json(out / "guard_manifest.json", guard["meta"])
    runtime_registration = {
        "protocol": "029",
        "diagnostic_only": True,
        "online_method": "Protocol028 D_memory_protected unchanged",
        "source_registration": reg,
        "stream_sha256": stream_sha,
        "replay_seed": 700,
        "model_seed": 1,
        "post_hoc_oracle_is_online_method": False,
    }
    arm_out = out / "D_memory_protected_with_passive_audit"
    arm_out.mkdir(parents=True, exist_ok=False)
    session = Protocol029DiagnosticSession(
        guard_anchor=guard,
        seed=1,
        replay_bundle=bundle,
        budget=p27.budget(),
        out_dir=arm_out,
        run_id=f"protocol029_{args.run_id}_D_memory_protected_with_passive_audit",
        stream_dir=stream,
        phase_defs=phases,
        stream_sha=stream_sha,
        registration=runtime_registration,
        learning_rate=1e-4,
    )
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    for _ in range(session.steps):
        session.step()
    session.lifecycle_controller.mark_stream_end()
    session.lifecycle_controller.finalize_passive_audits(session.steps)
    session.finish()
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    session.save()
    ctrl = session.lifecycle_controller
    write_jsonl(out / "D_memory_protected_with_passive_audit" / "lifecycle.jsonl", ctrl.events)
    save_passive_predictions(ctrl, out / "D_memory_protected_with_passive_audit" / "passive_predictions.npz")
    write_json(out / "opportunity_audit_full.json", ctrl.opportunity_audit)
    write_json(out / "opportunity_audit.json", compact_opportunity_audit(ctrl.opportunity_audit))

    source_comparison = load_json(source_root / "comparison.json")
    windows = window_memory_utility(session, ctrl, stream, source_comparison)
    write_json(out / "window_memory_utility.json", windows)
    consistency = compare_consistency(session, ctrl, source_root)
    write_json(out / "consistency.json", consistency)
    opp_summary = summarize_opportunities(ctrl.opportunity_audit)
    due_source = load_json(source_root / "D_memory_protected" / "opportunity_accounting.json")["reuse"]["due"]
    opp_summary["source_protocol028_reuse_due"] = int(due_source)
    opp_summary["due_count_matches_source"] = int(due_source) == int(opp_summary["reuse_due_opportunities"])

    passing_rows = []
    for opportunity in ctrl.opportunity_audit:
        for key, row in opportunity["candidate_validation"].items():
            decision = row.get("decision")
            if decision and decision.get("all_gates_pass"):
                passing_rows.append({
                    "opportunity_id": opportunity["opportunity_id"],
                    "due_cursor": opportunity["due_cursor"],
                    "expert_id": key,
                    "controller": opportunity["controller"],
                    "similarity": opportunity["similarity"].get("candidates", {}).get(key),
                    "decision": decision,
                    "post_hoc_oracle_diagnostic": True,
                })
    diagnostic_summary = {
        "protocol": "029",
        "run_id": str(args.run_id),
        "task": "passive dormant-memory utility audit on unchanged Protocol028 D trajectory",
        "diagnostic_valid": bool(consistency["diagnostic_valid"] and opp_summary["due_count_matches_source"]),
        "online_method_changed": False,
        "new_training_arms": 0,
        "C_rerun": False,
        "data_revision": REVISION_ID,
        "stream_sha256": EXPECTED_STREAM_SHA,
        "source_protocol028_run": SOURCE_028_RUN,
        "source_protocol028_artifact": SOURCE_028_ARTIFACT,
        "consistency": consistency,
        "opportunities": opp_summary,
        "all_gates_passing_candidates": passing_rows,
        "all_gates_passing_candidate_count": len(passing_rows),
        "nine_recurrence_windows": windows,
        "post_hoc_oracle_diagnostic": True,
        "interpretation_limit": "Passive/oracle diagnostics do not constitute an online D>C result.",
        "cost": {
            "diagnostic_replay_wall_seconds": float(wall),
            "diagnostic_replay_cpu_seconds": float(cpu),
            "passive_prediction_overhead_seconds": float(ctrl.passive_overhead_seconds),
            "passive_guard_overhead_seconds": float(ctrl.passive_guard_seconds),
            "passive_prediction_forwards": int(ctrl.passive_prediction_forwards),
            "passive_guard_evaluations": int(ctrl.passive_guard_forwards),
            "timing_note": "Protocol029 total includes passive diagnostic overhead and is not a new method-cost comparison with historical C.",
        },
    }
    write_json(out / "diagnostic_summary.json", diagnostic_summary)
    write_json(out / "cost_profile.json", diagnostic_summary["cost"])
    status = {
        "protocol": "029",
        "run_id": str(args.run_id),
        "completed": True,
        "diagnostic_valid": diagnostic_summary["diagnostic_valid"],
        "source_028_consistency_verified": consistency["diagnostic_valid"],
        "reuse_due_count_matches_source": opp_summary["due_count_matches_source"],
        "new_training_arms": 0,
        "C_rerun": False,
        "automatic_followups_started": [],
        "blocker": None if diagnostic_summary["diagnostic_valid"] else "passive_diagnostic_failed_source_trajectory_consistency",
    }
    write_json(out / "status.json", status)
    print(json.dumps({"status": status, "opportunities": opp_summary, "passing": len(passing_rows)}, indent=2, allow_nan=False))
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--method-registration", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--source-028-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        root = Path(args.out_root)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "status.json", {
            "protocol": "029", "run_id": str(args.run_id), "completed": False,
            "diagnostic_valid": False, "new_training_arms": 0, "C_rerun": False,
            "automatic_followups_started": [], "blocker": "runner_exception",
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
