"""Protocol-033 fixed six-window C_fixed5 vs D_guard_budget comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from ftmoe_protocol024_eval import binary_detection_metrics

ARMS = ("C_fixed5", "D_guard_budget")
RECURRENCE = ("U_rec1", "V_rec1", "U_rec2", "V_rec2", "U_rec3", "V_rec3")
W_BLOCKS = ("W_long", "W_gap1", "W_gap2", "W_gap3", "W_gap4", "W_gap5")


def J(path): return json.loads(Path(path).read_text(encoding="utf8"))
def N(path):
    with np.load(path, allow_pickle=True) as z: return {k: z[k].copy() for k in z.files}
def W(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")

def metric(prob, y, a, b): return binary_detection_metrics(prob[int(a):int(b)], y[int(a):int(b)], 0.5)
def ap_delta(c, d): return None if c.get("ap") is None or d.get("ap") is None else float(d["ap"] - c["ap"])
def fpr_delta(c, d): return None if c.get("fpr") is None or d.get("fpr") is None else float(d["fpr"] - c["fpr"])

def in_window(cursor, pmap):
    if cursor is None: return None
    i = int(cursor)
    for name in RECURRENCE:
        p = pmap[name]
        if int(p["start"]) <= i < int(p["end"]): return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    root, data = args.run_root, args.data_root
    summaries = {name: J(root / name / "summary.json") for name in ARMS}
    pred = {name: N(root / name / "predictions.npz") for name in ARMS}
    c, d = pred["C_fixed5"], pred["D_guard_budget"]
    if not np.array_equal(c["labels"], d["labels"]): raise AssertionError("Protocol033 C/D labels differ")
    if not np.array_equal(c["raw_labels"], d["raw_labels"]): raise AssertionError("Protocol033 C/D raw labels differ")
    if summaries["C_fixed5"]["stream_sha256"] != summaries["D_guard_budget"]["stream_sha256"]: raise AssertionError("Protocol033 C/D stream differs")
    if summaries["C_fixed5"]["initialization_shared_prefix_sha256"] != summaries["D_guard_budget"]["initialization_shared_prefix_sha256"]: raise AssertionError("Protocol033 shared first4 initialization differs")

    y = c["labels"]
    if int(y.shape[0]) != 5968: raise AssertionError("Protocol033 prediction count mismatch")
    manifest = J(data / "manifest.json")
    pmap = {p["name"]: p for p in manifest["timeline"]}

    rows, deltas, pooled = [], [], []
    for name in RECURRENCE:
        p = pmap[name]; a, b = int(p["start"]), int(p["end"])
        if b - a != 128: raise AssertionError("Protocol033 recurrence width mismatch")
        pooled.extend(range(a, b))
        cm, dm = metric(c["probability"], y, a, b), metric(d["probability"], y, a, b)
        delta = ap_delta(cm, dm)
        if delta is not None: deltas.append(delta)
        prefixes = {}
        for width in (32, 64):
            cc, dd = metric(c["probability"], y, a, a + width), metric(d["probability"], y, a, a + width)
            prefixes[f"first{width}"] = {"C_fixed5": cc, "D_guard_budget": dd,
                                         "D_minus_C_ap": ap_delta(cc, dd), "D_minus_C_fpr": fpr_delta(cc, dd)}
        rows.append({
            "phase": name, "prediction_intervals": [a, b], "target_raw_rows": [a + 1, b + 1],
            "positive_hoststeps": int((y[a:b] > 0).sum()), "negative_hoststeps": int((y[a:b] == 0).sum()),
            "C_fixed5": cm, "D_guard_budget": dm, "D_minus_C_ap": delta,
            "D_minus_C_fpr": fpr_delta(cm, dm), "prefixes": prefixes,
        })

    idx = np.asarray(pooled, dtype=np.int64)
    pooled_c = binary_detection_metrics(c["probability"][idx], y[idx], 0.5)
    pooled_d = binary_detection_metrics(d["probability"][idx], y[idx], 0.5)
    pooled_fpr_delta = fpr_delta(pooled_c, pooled_d)
    mean_delta = float(np.mean(deltas)) if len(deltas) == 6 else None
    positive_windows = int(sum(x > 0 for x in deltas)) if len(deltas) == 6 else 0

    w_rows, w_deltas = [], []
    for name in W_BLOCKS:
        p = pmap[name]; a, b = int(p["start"]), int(p["end"])
        cm, dm = metric(c["probability"], y, a, b), metric(d["probability"], y, a, b)
        delta = ap_delta(cm, dm)
        if delta is not None: w_deltas.append(delta)
        w_rows.append({"phase": name, "intervals": [a, b], "C_fixed5": cm, "D_guard_budget": dm, "D_minus_C_ap": delta})
    w_mean = float(np.mean(w_deltas)) if len(w_deltas) == len(W_BLOCKS) else None

    criteria = {
        "all_six_recurrence_windows_valid": len(deltas) == 6,
        "equal_weight_mean_ap_delta_ge_0p03": bool(mean_delta is not None and mean_delta >= 0.03),
        "at_least_4_of_6_ap_deltas_positive": positive_windows >= 4,
        "pooled_recurrence_normal_fpr_delta_le_0p01": bool(pooled_fpr_delta is not None and pooled_fpr_delta <= 0.01),
        "W_blocks_equal_weight_mean_ap_delta_ge_minus_0p02": bool(w_mean is not None and w_mean >= -0.02),
    }
    criteria["all_development_reference_pass"] = all(criteria.values())

    lifecycle = summaries["D_guard_budget"].get("lifecycle") or {}
    event_records = list(lifecycle.get("protocol033_reuse_event_records") or [])
    reuse_event_evidence = []
    accepted_in_recurrence = False
    for rec in event_records:
        accepted = rec.get("accepted_cursor")
        influence = rec.get("first_influence_prediction_index")
        if influence is not None and accepted is not None and int(influence) < int(accepted):
            raise AssertionError("Protocol033 per-event reuse influence predates acceptance")
        window = in_window(influence, pmap)
        accepted_in_recurrence = accepted_in_recurrence or bool(window)
        reuse_event_evidence.append({**rec, "first_influence_recurrence_window": window})

    snapshots = list(lifecycle.get("topology_snapshots") or [])
    before_recurrence = {s["label"]: {
        "active_specialist_id": s.get("active_specialist_id"),
        "real_dormant_expert_ids": s.get("real_dormant_expert_ids", []),
        "memory_registry_ids": s.get("memory_registry_ids", []),
        "shadow_id": s.get("shadow_id"),
    } for s in snapshots if s.get("label", "").startswith("before_")}
    mechanism = {
        "per_reactivation_event_logging": reuse_event_evidence,
        "accepted_reuse_with_first_influence_in_recurrence": accepted_in_recurrence,
        "topology_before_each_registered_recurrence": before_recurrence,
        "retirements": lifecycle.get("retirements"), "reactivations": lifecycle.get("reactivations"),
        "reuse_outcomes": lifecycle.get("reuse_outcomes"),
        "candidate_sources": lifecycle.get("candidate_sources"),
        "mechanism_has_event_specific_evidence": bool(event_records),
        "reuse_mechanism_established_for_registered_recurrence": bool(event_records and accepted_in_recurrence),
    }

    full_c = summaries["C_fixed5"]["full"]["detection"]
    full_d = summaries["D_guard_budget"]["full"]["detection"]
    c_wall, d_wall = float(summaries["C_fixed5"]["cost"]["wall_seconds"]), float(summaries["D_guard_budget"]["cost"]["wall_seconds"])
    cost = {
        "C_fixed5": summaries["C_fixed5"]["cost"], "D_guard_budget": summaries["D_guard_budget"]["cost"],
        "D_over_C_wall_time_ratio": None if c_wall <= 0 else d_wall / c_wall,
        "equal_total_cost_claim_allowed": False,
    }
    comparison = {
        "protocol": "033", "plan_revision": 1, "run_id": str(args.run_id),
        "kind": "single_5969_row_six_recurrence_development_pair",
        "development_only": True, "statistical_confirmation": False,
        "timeline_selected_after_observing_031_032": True,
        "same_input_and_labels_verified": True, "shared_first4_initialization_verified": True,
        "stream_sha256": summaries["C_fixed5"]["stream_sha256"],
        "recurrence128": rows,
        "primary": {
            "metric": "equal-weight mean AP(D_guard_budget)-AP(C_fixed5) across all six recurrence128 windows",
            "equal_weight_mean_D_minus_C_ap": mean_delta,
            "positive_window_count": positive_windows,
            "pooled_recurrence_C_fixed5": pooled_c, "pooled_recurrence_D_guard_budget": pooled_d,
            "pooled_recurrence_normal_fpr_delta_D_minus_C": pooled_fpr_delta,
            "development_reference": criteria,
        },
        "W_blocks": w_rows, "W_blocks_equal_weight_mean_D_minus_C_ap": w_mean,
        "full_stream": {"C_fixed5": full_c, "D_guard_budget": full_d,
                        "D_minus_C_ap": ap_delta(full_c, full_d), "D_minus_C_fpr": fpr_delta(full_c, full_d)},
        "mechanism": mechanism, "cost_profile": cost,
        "interpretation_limits": {
            "independent_confirmation": False, "global_D_optimality_claim_allowed": False,
            "A_B_inference_allowed": False, "hard_delete_benefit_claim_allowed": False,
            "additional_seed_or_threshold_search_allowed": False,
        },
    }
    status = {
        "protocol": "033", "plan_revision": 1, "run_id": str(args.run_id), "completed": True,
        "full_model_replays_started": 2, "full_model_replay_budget": 2,
        "completed_comparators": list(ARMS), "automatic_followups_started": [],
        "development_reference_all_pass": criteria["all_development_reference_pass"],
        "reuse_mechanism_established_for_registered_recurrence": mechanism["reuse_mechanism_established_for_registered_recurrence"],
        "blocker": None, "stop_after_this_pair": True,
    }
    W(root / "comparison.json", comparison); W(root / "mechanism_evidence.json", mechanism)
    W(root / "cost_profile.json", cost); W(root / "status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
