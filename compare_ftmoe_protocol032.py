"""Protocol-032 fixed two-window comparison and mechanism evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ftmoe_protocol024_eval import binary_detection_metrics

ARMS = ("C_fixed5", "D_guard_budget")
RECURRENCE = ("U_rec1", "V_rec1")
COMPLETE_W = ("W_long", "W_gap1")


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def N(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def metric(prob, y, a, b):
    return binary_detection_metrics(prob[int(a):int(b)], y[int(a):int(b)], 0.5)


def ap_delta(c, d):
    return None if c.get("ap") is None or d.get("ap") is None else float(d["ap"] - c["ap"])


def fpr_delta(c, d):
    return None if c.get("fpr") is None or d.get("fpr") is None else float(d["fpr"] - c["fpr"])


def cursor_in_recurrence(cursor, pmap):
    if cursor is None:
        return False
    i = int(cursor)
    return any(int(pmap[name]["start"]) <= i < int(pmap[name]["end"]) for name in RECURRENCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--baseline-consistency", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    root = Path(args.run_root)
    bundle = Path(args.bundle_root)
    summaries = {name: J(root / name / "summary.json") for name in ARMS}
    pred = {name: N(root / name / "predictions.npz") for name in ARMS}
    c, d = pred["C_fixed5"], pred["D_guard_budget"]
    if not np.array_equal(c["labels"], d["labels"]):
        raise AssertionError("Protocol032 C/D labels differ")
    if summaries["C_fixed5"]["stream_sha256"] != summaries["D_guard_budget"]["stream_sha256"]:
        raise AssertionError("Protocol032 C/D stream hash differs")
    if summaries["C_fixed5"]["initialization_shared_prefix_sha256"] != summaries["D_guard_budget"]["initialization_shared_prefix_sha256"]:
        raise AssertionError("Protocol032 C/D shared first4 initialization differs")

    baseline = J(args.baseline_consistency)
    if baseline.get("passed") is not True:
        raise AssertionError("Protocol032 baseline C consistency did not pass")

    y = c["labels"]
    manifest = J(bundle / "data" / "manifest.json")
    pmap = {p["name"]: p for p in manifest["timeline"]}

    recurrence = []
    recurrence_deltas = []
    pooled = []
    for name in RECURRENCE:
        p = pmap[name]
        a, b = int(p["start"]), int(p["end"])
        pooled.extend(range(a, b))
        cm = metric(c["probability"], y, a, b)
        dm = metric(d["probability"], y, a, b)
        delta = ap_delta(cm, dm)
        if delta is not None:
            recurrence_deltas.append(delta)
        prefixes = {}
        for width in (32, 64):
            cc = metric(c["probability"], y, a, min(b, a + width))
            dd = metric(d["probability"], y, a, min(b, a + width))
            prefixes[f"first{width}"] = {
                "C_fixed5": cc,
                "D_guard_budget": dd,
                "D_minus_C_ap": ap_delta(cc, dd),
                "D_minus_C_fpr": fpr_delta(cc, dd),
            }
        recurrence.append({
            "phase": name,
            "intervals": [a, b],
            "positive_hoststeps": int((y[a:b] > 0).sum()),
            "negative_hoststeps": int((y[a:b] == 0).sum()),
            "C_fixed5": cm,
            "D_guard_budget": dm,
            "D_minus_C_ap": delta,
            "D_minus_C_fpr": fpr_delta(cm, dm),
            "prefixes": prefixes,
        })

    idx = np.asarray(pooled, dtype=np.int64)
    pooled_c = binary_detection_metrics(c["probability"][idx], y[idx], 0.5)
    pooled_d = binary_detection_metrics(d["probability"][idx], y[idx], 0.5)
    pooled_fpr_delta = fpr_delta(pooled_c, pooled_d)
    mean_delta = float(np.mean(recurrence_deltas)) if len(recurrence_deltas) == 2 else None
    both_positive = bool(len(recurrence_deltas) == 2 and all(x > 0 for x in recurrence_deltas))

    w_rows = []
    w_deltas = []
    for name in COMPLETE_W:
        p = pmap[name]
        a, b = int(p["start"]), int(p["end"])
        cm = metric(c["probability"], y, a, b)
        dm = metric(d["probability"], y, a, b)
        delta = ap_delta(cm, dm)
        if delta is not None:
            w_deltas.append(delta)
        w_rows.append({
            "phase": name, "intervals": [a, b],
            "C_fixed5": cm, "D_guard_budget": dm,
            "D_minus_C_ap": delta,
        })
    w_mean = float(np.mean(w_deltas)) if len(w_deltas) == len(COMPLETE_W) else None

    partial = None
    if "W_gap2_partial" in pmap:
        p = pmap["W_gap2_partial"]
        a, b = int(p["start"]), int(p["end"])
        cm = metric(c["probability"], y, a, b)
        dm = metric(d["probability"], y, a, b)
        partial = {
            "phase": "W_gap2_partial", "intervals": [a, b],
            "C_fixed5": cm, "D_guard_budget": dm,
            "D_minus_C_ap": ap_delta(cm, dm),
        }

    full_c = summaries["C_fixed5"]["full"]["detection"]
    full_d = summaries["D_guard_budget"]["full"]["detection"]
    criteria = {
        "mean_recurrence_ap_delta_gt_0": bool(mean_delta is not None and mean_delta > 0),
        "both_recurrence_windows_ap_delta_gt_0": both_positive,
        "pooled_recurrence_normal_fpr_delta_le_0p01": bool(
            pooled_fpr_delta is not None and pooled_fpr_delta <= 0.01),
        "complete_W_long_W_gap1_mean_ap_delta_ge_minus_0p02": bool(
            w_mean is not None and w_mean >= -0.02),
    }
    criteria["all_development_reference_pass"] = all(criteria.values())

    lifecycle = summaries["D_guard_budget"].get("lifecycle") or {}
    candidate_records = J(root / "D_guard_budget" / "candidate_records.json")
    reuse_records = J(root / "D_guard_budget" / "reuse_records.json")
    snapshots = J(root / "D_guard_budget" / "topology_snapshots.json")

    candidate_evidence = []
    for rec in candidate_records:
        decision = rec.get("decision") or {}
        guard = decision.get("guard") or {}
        candidate_evidence.append({
            "candidate_id": rec.get("candidate_id"),
            "created_cursor": rec.get("created_cursor"),
            "accepted": rec.get("accepted"),
            "accepted_cursor": rec.get("accepted_cursor"),
            "first_influence_cursor": rec.get("first_influence_cursor"),
            "relative_improvement": decision.get("relative_improvement"),
            "local_loss_ok": None if not decision else bool(
                decision.get("relative_improvement", -1e99) >= decision.get("required_relative_improvement", 0.01)),
            "normal_probability_ok": decision.get("normal_safety_ok"),
            "validation_fpr_ok": decision.get("validation_fpr_ok"),
            "F0_guard_fpr_ok": decision.get("guard_fpr_ok"),
            "F0_live_normal_nll": guard.get("live_loss"),
            "F0_candidate_normal_nll": guard.get("candidate_loss"),
            "F0_raw_relative_nll_increase": guard.get("relative_loss_increase_raw"),
            "F0_old_nll_limit": guard.get("old_candidate_normal_nll_limit"),
            "F0_new_nll_limit": guard.get("candidate_normal_nll_limit"),
            "F0_old_nll_ok": guard.get("old_normal_nll_ok_direct"),
            "F0_new_nll_ok": guard.get("normal_nll_ok_direct"),
            "F0_birth_equivalent_check_ok": guard.get("inherited_birth_relative_check_ok"),
            "reject_reasons": decision.get("reject_reasons"),
        })

    reuse_evidence = []
    accepted_reuse_in_recurrence = False
    for rec in reuse_records:
        decision = rec.get("decision") or {}
        guard = decision.get("guard") or {}
        first_influence = rec.get("first_influence_cursor")
        if first_influence is None and rec.get("outcome") == "accepted":
            first_influence = lifecycle.get("reuse_first_influence_cursor")
        in_rec = cursor_in_recurrence(first_influence, pmap)
        accepted_reuse_in_recurrence = accepted_reuse_in_recurrence or bool(
            rec.get("outcome") == "accepted" and in_rec)
        reuse_evidence.append({
            "expert_id": rec.get("expert_id"),
            "started_cursor": rec.get("started_cursor"),
            "outcome": rec.get("outcome"),
            "accepted_cursor": rec.get("accepted_cursor"),
            "first_influence_cursor": first_influence,
            "first_influence_in_registered_recurrence_window": in_rec,
            "selection_similarity": rec.get("selection_similarity"),
            "similarity_threshold_used_as_gate": rec.get("similarity_threshold_used_as_gate"),
            "relative_improvement": decision.get("relative_improvement"),
            "normal_probability_ok": decision.get("normal_safety_ok"),
            "validation_fpr_ok": decision.get("validation_fpr_ok"),
            "F0_guard_fpr_ok": decision.get("guard_fpr_ok"),
            "F0_live_normal_nll": guard.get("live_loss"),
            "F0_candidate_normal_nll": guard.get("candidate_loss"),
            "F0_raw_relative_nll_increase": guard.get("relative_loss_increase_raw"),
            "F0_old_nll_ok": guard.get("old_normal_nll_ok_direct"),
            "F0_new_nll_ok": guard.get("normal_nll_ok_direct"),
            "accepted": decision.get("accepted"),
            "reject_reasons": decision.get("reject_reasons"),
        })

    real_dormant_before_recurrence = {
        s["label"]: s.get("real_dormant_expert_ids", [])
        for s in snapshots if s["label"] in ("before_U_rec1", "before_V_rec1")
    }
    memory_registry_before_recurrence = {
        s["label"]: s.get("memory_registry_ids", [])
        for s in snapshots if s["label"] in ("before_U_rec1", "before_V_rec1")
    }
    any_real_dormant_before_recurrence = any(bool(v) for v in real_dormant_before_recurrence.values())
    mechanism = {
        "real_dormant_ids_before_recurrence": real_dormant_before_recurrence,
        "memory_registry_ids_before_recurrence": memory_registry_before_recurrence,
        "any_real_dormant_before_recurrence": any_real_dormant_before_recurrence,
        "retirements": lifecycle.get("retirements"),
        "reactivations": lifecycle.get("reactivations"),
        "reuse_outcomes": lifecycle.get("reuse_outcomes"),
        "reuse_first_influence_cursor": lifecycle.get("reuse_first_influence_cursor"),
        "accepted_reuse_with_first_influence_in_recurrence": accepted_reuse_in_recurrence,
        "reuse_mechanism_established_for_registered_recurrence": bool(
            any_real_dormant_before_recurrence and accepted_reuse_in_recurrence),
        "candidate_evidence": candidate_evidence,
        "reuse_evidence": reuse_evidence,
        "topology_snapshots": snapshots,
    }

    c_wall = float(summaries["C_fixed5"]["cost"]["wall_seconds"])
    d_wall = float(summaries["D_guard_budget"]["cost"]["wall_seconds"])
    cost = {
        "C_fixed5": summaries["C_fixed5"]["cost"],
        "D_guard_budget": summaries["D_guard_budget"]["cost"],
        "D_over_C_wall_time_ratio": None if c_wall <= 0 else d_wall / c_wall,
        "equal_total_cost_claim_allowed": False,
    }

    comparison = {
        "protocol": "032", "plan_revision": 1, "run_id": str(args.run_id),
        "kind": "single_guard_budget_development_pair",
        "development_only": True,
        "statistical_confirmation": False,
        "method_change_selected_after_observing_031_results": True,
        "stream_sha256": summaries["C_fixed5"]["stream_sha256"],
        "same_input_and_labels_verified": True,
        "shared_first4_initialization_verified": True,
        "baseline_C_consistency": baseline,
        "recurrence128": recurrence,
        "primary": {
            "metric": "equal-weight mean AP(D_guard_budget)-AP(C_fixed5) across U_rec1 and V_rec1",
            "equal_weight_mean_D_minus_C_ap": mean_delta,
            "both_window_deltas_positive": both_positive,
            "pooled_recurrence_C_fixed5": pooled_c,
            "pooled_recurrence_D_guard_budget": pooled_d,
            "pooled_recurrence_normal_fpr_delta_D_minus_C": pooled_fpr_delta,
            "development_reference": criteria,
        },
        "complete_W_blocks": w_rows,
        "complete_W_equal_weight_mean_D_minus_C_ap": w_mean,
        "W_gap2_partial_secondary": partial,
        "full_stream": {
            "C_fixed5": full_c,
            "D_guard_budget": full_d,
            "D_minus_C_ap": ap_delta(full_c, full_d),
            "D_minus_C_fpr": fpr_delta(full_c, full_d),
        },
        "mechanism": mechanism,
        "cost_profile": cost,
        "interpretation_limits": {
            "independent_confirmation": False,
            "global_D_optimality_claim_allowed": False,
            "A_B_inference_allowed": False,
            "additional_seed_threshold_scenario_followup_allowed": False,
            "candidate_acceptance_alone_counts_as_reuse_mechanism": False,
            "D_cost_advantage_assumed": False,
        },
    }

    status = {
        "protocol": "032", "plan_revision": 1, "run_id": str(args.run_id),
        "completed": True,
        "full_model_replays_started": 2,
        "full_model_replay_budget": 2,
        "completed_comparators": list(ARMS),
        "A_B_runs": 0,
        "automatic_followups_started": [],
        "development_reference_all_pass": criteria["all_development_reference_pass"],
        "reuse_mechanism_established_for_registered_recurrence": mechanism["reuse_mechanism_established_for_registered_recurrence"],
        "blocker": None,
        "stop_after_this_pair": True,
    }

    (root / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (root / "mechanism_evidence.json").write_text(json.dumps(mechanism, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (root / "cost_profile.json").write_text(json.dumps(cost, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (root / "status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf8")
    compact = {
        "run_id": str(args.run_id),
        "mean_recurrence_ap_delta_D_minus_C": mean_delta,
        "U_rec1_ap_delta": recurrence[0]["D_minus_C_ap"],
        "V_rec1_ap_delta": recurrence[1]["D_minus_C_ap"],
        "pooled_recurrence_fpr_delta_D_minus_C": pooled_fpr_delta,
        "complete_W_mean_ap_delta_D_minus_C": w_mean,
        "all_development_reference_pass": criteria["all_development_reference_pass"],
        "real_dormant_before_recurrence": any_real_dormant_before_recurrence,
        "reuse_mechanism_established": mechanism["reuse_mechanism_established_for_registered_recurrence"],
        "retirements": lifecycle.get("retirements"),
        "reactivations": lifecycle.get("reactivations"),
        "D_over_C_wall_time_ratio": cost["D_over_C_wall_time_ratio"],
        "full_stream_ap_delta_D_minus_C": comparison["full_stream"]["D_minus_C_ap"],
    }
    (root / "compact_evidence.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
