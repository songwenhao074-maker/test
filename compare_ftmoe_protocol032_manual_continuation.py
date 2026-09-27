"""Compare a user-authorized Protocol-032 post-blocker diagnostic continuation.

The registered Protocol-032 run remains blocked.  This comparator never rewrites
that fact.  It compares the preserved completed C output from run 36310338191
against one subsequently authorized D_guard_budget replay on the same frozen
stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ftmoe_protocol024_eval import binary_detection_metrics

RECURRENCE = ("U_rec1", "V_rec1")
COMPLETE_W = ("W_long", "W_gap1")
EXPECTED_STREAM_SHA = "1e8b6bde1fa3f906586030547777428b28fca23c43e06868077dc0e1e46e4d1a"
EXPECTED_INIT = "7e8a7a901775b003d82d6f30d2ce448a2e7f1c100bd00019115271883c2de29b"


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def N(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def metric(prob, y, a, b):
    return binary_detection_metrics(prob[int(a):int(b)], y[int(a):int(b)], 0.5)


def ap_delta(c, d):
    if c.get("ap") is None or d.get("ap") is None:
        return None
    return float(d["ap"] - c["ap"])


def fpr_delta(c, d):
    if c.get("fpr") is None or d.get("fpr") is None:
        return None
    return float(d["fpr"] - c["fpr"])


def cursor_in_recurrence(cursor, pmap):
    if cursor is None:
        return False
    i = int(cursor)
    return any(int(pmap[n]["start"]) <= i < int(pmap[n]["end"]) for n in RECURRENCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-root", required=True)
    ap.add_argument("--d-root", required=True)
    ap.add_argument("--bundle-root", required=True)
    ap.add_argument("--formal-baseline-consistency", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    croot = Path(args.c_root)
    droot = Path(args.d_root)
    bundle = Path(args.bundle_root)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    csum = J(croot / "summary.json")
    dsum = J(droot / "summary.json")
    c = N(croot / "predictions.npz")
    d = N(droot / "predictions.npz")
    formal = J(args.formal_baseline_consistency)

    # Preserve the original registered blocker exactly; this continuation is
    # valid only if the formal run really did stop for that narrow reason.
    if formal.get("passed") is not False:
        raise AssertionError("formal Protocol032 baseline blocker is not preserved")
    if formal.get("labels_equal") is not True or formal.get("raw_labels_equal") is not True:
        raise AssertionError("formal Protocol032 blocker involved label mismatch")
    if formal.get("new_shared_first4_initialization") != EXPECTED_INIT:
        raise AssertionError("formal Protocol032 C initialization differs")
    if formal.get("baseline_shared_first4_initialization") != EXPECTED_INIT:
        raise AssertionError("Protocol031 baseline initialization differs")
    if not (float(formal.get("C_probability_max_abs_diff")) > 1e-6):
        raise AssertionError("formal blocker was not the registered probability tolerance")

    if csum.get("stream_sha256") != EXPECTED_STREAM_SHA or dsum.get("stream_sha256") != EXPECTED_STREAM_SHA:
        raise AssertionError("C/D stream identity differs from frozen Protocol032 input")
    if not np.array_equal(c["labels"], d["labels"]):
        raise AssertionError("manual continuation C/D settled labels differ")
    if not np.array_equal(c["raw_labels"], d["raw_labels"]):
        raise AssertionError("manual continuation C/D raw labels differ")
    if csum.get("initialization_shared_prefix_sha256") != EXPECTED_INIT:
        raise AssertionError("preserved C initialization differs")
    if dsum.get("initialization_shared_prefix_sha256") != EXPECTED_INIT:
        raise AssertionError("D initialization differs")

    manifest = J(bundle / "data" / "manifest.json")
    if manifest.get("stream_sha256") != EXPECTED_STREAM_SHA:
        raise AssertionError("frozen manifest stream hash differs")
    pmap = {p["name"]: p for p in manifest["timeline"]}
    y = c["labels"]

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
    both_positive = bool(len(recurrence_deltas) == 2 and all(v > 0 for v in recurrence_deltas))

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
            "phase": name,
            "intervals": [a, b],
            "C_fixed5": cm,
            "D_guard_budget": dm,
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
            "phase": "W_gap2_partial",
            "intervals": [a, b],
            "C_fixed5": cm,
            "D_guard_budget": dm,
            "D_minus_C_ap": ap_delta(cm, dm),
            "D_minus_C_fpr": fpr_delta(cm, dm),
        }

    criteria = {
        "mean_recurrence_ap_delta_gt_0": bool(mean_delta is not None and mean_delta > 0),
        "both_recurrence_windows_ap_delta_gt_0": both_positive,
        "pooled_recurrence_normal_fpr_delta_le_0p01": bool(
            pooled_fpr_delta is not None and pooled_fpr_delta <= 0.01),
        "complete_W_long_W_gap1_mean_ap_delta_ge_minus_0p02": bool(
            w_mean is not None and w_mean >= -0.02),
    }
    criteria["all_development_reference_pass"] = all(criteria.values())

    lifecycle = dsum.get("lifecycle") or {}
    candidate_records = J(droot / "candidate_records.json")
    reuse_records = J(droot / "reuse_records.json")
    snapshots = J(droot / "topology_snapshots.json")

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

    real_dormant = {
        s["label"]: s.get("real_dormant_expert_ids", [])
        for s in snapshots if s.get("label") in ("before_U_rec1", "before_V_rec1")
    }
    memory_registry = {
        s["label"]: s.get("memory_registry_ids", [])
        for s in snapshots if s.get("label") in ("before_U_rec1", "before_V_rec1")
    }
    any_real_dormant = any(bool(v) for v in real_dormant.values())
    mechanism = {
        "real_dormant_ids_before_recurrence": real_dormant,
        "memory_registry_ids_before_recurrence": memory_registry,
        "any_real_dormant_before_recurrence": any_real_dormant,
        "retirements": lifecycle.get("retirements"),
        "reactivations": lifecycle.get("reactivations"),
        "reuse_outcomes": lifecycle.get("reuse_outcomes"),
        "reuse_first_influence_cursor": lifecycle.get("reuse_first_influence_cursor"),
        "accepted_reuse_with_first_influence_in_recurrence": accepted_reuse_in_recurrence,
        "reuse_mechanism_established_for_registered_recurrence": bool(
            any_real_dormant and accepted_reuse_in_recurrence),
        "candidate_evidence": candidate_evidence,
        "reuse_evidence": reuse_evidence,
        "topology_snapshots": snapshots,
    }

    c_wall = float(csum["cost"]["wall_seconds"])
    d_wall = float(dsum["cost"]["wall_seconds"])
    cost = {
        "C_fixed5_preserved_from_run_36310338191": csum["cost"],
        "D_guard_budget_manual_continuation": dsum["cost"],
        "cross_run_D_over_C_wall_time_ratio": None if c_wall <= 0 else d_wall / c_wall,
        "same_job_cost_comparison": False,
        "equal_total_cost_claim_allowed": False,
    }

    full_c = csum["full"]["detection"]
    full_d = dsum["full"]["detection"]
    comparison = {
        "protocol": "032",
        "plan_revision": 1,
        "run_id": str(args.run_id),
        "kind": "manual_post_blocker_diagnostic_continuation",
        "manual_user_authorized_after_blocker": True,
        "registered_protocol032_completed": False,
        "registered_protocol032_terminal_run_id": 36310338191,
        "registered_protocol032_blocker": "baseline_C_consistency_failed",
        "registered_baseline_consistency_preserved": formal,
        "C_replayed_in_continuation": False,
        "C_source_run_id": 36310338191,
        "D_replayed_in_continuation": True,
        "development_only": True,
        "statistical_confirmation": False,
        "method_change_selected_after_observing_031_results": True,
        "stream_sha256": EXPECTED_STREAM_SHA,
        "same_frozen_input_and_labels_verified": True,
        "shared_first4_initialization_verified": True,
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
            "registered_protocol032_completion_claim_allowed": False,
            "independent_confirmation": False,
            "global_D_optimality_claim_allowed": False,
            "A_B_inference_allowed": False,
            "additional_seed_threshold_scenario_followup_allowed": False,
            "candidate_acceptance_alone_counts_as_reuse_mechanism": False,
            "same_job_cost_comparison": False,
            "D_cost_advantage_assumed": False,
        },
    }
    status = {
        "protocol": "032",
        "run_id": str(args.run_id),
        "kind": "manual_post_blocker_diagnostic_continuation",
        "diagnostic_continuation_completed": True,
        "registered_protocol032_completed": False,
        "formal_blocker_preserved": True,
        "C_replays_started_in_continuation": 0,
        "D_replays_started_in_continuation": 1,
        "automatic_followups_started": [],
        "development_reference_all_pass": criteria["all_development_reference_pass"],
        "reuse_mechanism_established_for_registered_recurrence": mechanism[
            "reuse_mechanism_established_for_registered_recurrence"],
        "stop_after_this_D": True,
    }

    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (out / "mechanism_evidence.json").write_text(json.dumps(mechanism, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (out / "cost_profile.json").write_text(json.dumps(cost, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (out / "status.json").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({
        "diagnostic_continuation_completed": True,
        "registered_protocol032_completed": False,
        "mean_recurrence_D_minus_C_ap": mean_delta,
        "all_development_reference_pass": criteria["all_development_reference_pass"],
        "reuse_mechanism_established": mechanism["reuse_mechanism_established_for_registered_recurrence"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
