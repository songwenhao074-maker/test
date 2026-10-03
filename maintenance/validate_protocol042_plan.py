#!/usr/bin/env python3
"""Validate Protocol-042 revision 2 registration only; does not launch science."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/ftmoe_online/protocol_042/plan.json"

def require(condition, label):
    if not condition:
        raise ValueError(label)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = json.loads(PLAN.read_text(encoding="utf-8"))
    wanted = PLAN.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    require(digest(PLAN) == wanted, "plan SHA256")
    require((p["protocol"], p["revision"]) == ("042", 2), "superseded registration")
    require(p["status"] == "instructions_registered_not_implemented_not_started", "registration status")
    require(digest(ROOT / p["instructions"]) == p["instructions_sha256"], "directive SHA256")
    arms = ["U_parent", "R_win128", "A_hist", "A_win128"]
    require(p["arms"]["order"] == p["budget"]["order"] == arms, "four arms/order")
    require(p["arms"]["primary"] == p["analysis"]["primary_arm"] == "A_win128", "primary")
    require(p["supersedes"]["revision"] == 1 and p["supersedes"]["old_budget_not_additive"], "supersession")
    require(p["authorization"]["launch_in_this_turn"] is False, "publication only")
    require(p["authorization"]["automatic_next_protocol"] is False, "stop rule")
    require(p["arms"]["all_frozen_before_U"] is True, "joint freeze")
    require(p["arms"]["A_hist"]["score"] == "cumulative_current_deployment_epoch", "history control")
    require(p["arms"]["A_hist"]["admission"] == p["arms"]["A_win128"]["admission"], "window pair")
    require(p["arms"]["R_win128"]["window"] == p["arms"]["A_win128"]["window"] == 128, "window")
    require(p["arms"]["new_arm_initialization"] == "zero_weights_bias_fresh_AdamW", "initialization")
    s = p["new_structure"]
    require([s[k] for k in ["resident_including_shadow", "active_max", "shadow_max", "ids_created_max",
                           "candidate_attempts_after_E0_max"]] == [3, 2, 1, 3, 2], "capacity")
    require(not s["renormalize"] and not s["total_clip"] and s["permanent_deletions"] == 0, "additivity")
    q = p["score"]
    require(q["fixed_window"] == q["active_epoch_mature_min"] == 128, "window/maturity")
    require(q["missing"] == "null_not_zero" and q["extra_expert_forwards"] == 0, "unknown/overhead")
    require(q["positive_rows_min"] == q["negative_rows_min"] == 16 and q["positive_intervals_min"] == 4, "support")
    require(q["positive_intervals_are_independent_events"] is False, "independence")
    require(q["eligible"] == {"overall_max": 0, "positive_max": 0, "negative_max": 0,
                            "removal_fpr_increase_max": 0.01, "removal_recall_change_min": -0.02,
                            "consecutive_due16": 3}, "retirement")
    require(q["A_victim_requires_nonuseful"] and not q["R_victim_requires_nonuseful"], "admission contrast")
    require(p["training"]["weighted_training"] is False and p["training"]["shadow_updates_per_candidate"] == 16, "training")
    b = p["budget"]
    require(b["science_sequences"] == 4 and b["total"]["optimizer_steps"] == 2576, "budget")
    for k in ["U", "each_new"]:
        require(b[k]["live_steps"] + b[k]["shadow_steps"] == b[k]["total_steps"], "per-arm arithmetic")
    for source, target in [("total_steps", "optimizer_steps"), ("deployed_forwards", "deployed_forwards"),
                           ("reuse_forwards", "reuse_forwards"), ("qualification_forwards", "qualification_forwards")]:
        require(b["U"][source] + 3 * b["each_new"][source] == b["total"][target], "total arithmetic")
    require(sum(b["total"][k] for k in ["deployed_forwards", "reuse_forwards", "qualification_forwards"]) ==
            b["total"]["prediction_expert_forwards"] == 50800, "forward total")
    for k in ["new_streams", "extra_seeds", "F_loads", "extra_arms", "permanent_deletions", "real_engineering_gradients"]:
        require(b[k] == 0, "forbidden: " + k)
    require(b["restart_from_zero"] is False and b["old_revision_budget_additional"] is False, "restart/addition")
    require(p["gate"]["prior_revision_rejected"] and p["gate"]["all_new_require_U_reproduction"], "gate")
    require(p["gate"]["missing_or_nonboolean_pass"] == "fail_closed", "missing report")
    require(len(p["engineering"]["required_fixture_ids"]) == 10 and p["engineering"]["real_event_evidence_required"], "fixtures")
    require(p["analysis"]["diagnostic_windows"] == [64, 256] and not p["analysis"]["diagnostic_windows_control"], "no extra arms")
    src = p["source041"]
    manifest = ROOT / src["manifest_path"]
    require(digest(manifest) == src["manifest_sha256"], "historical manifest")
    entries = {e["path"]: e["sha256"] for e in json.loads(manifest.read_text(encoding="utf-8"))["files"]}
    for path, expected in src["files"].items():
        require(entries.get(path) == expected, "source lock: " + path)
    require(p["source036"]["replay_seed"] == p["data"]["replay_seed"] == 3601, "data")
    require(p["runtime"]["formal_trigger"] == "workflow_dispatch_only" and
            not p["runtime"]["create_workflow_this_publication"], "workflow")
    require(p["delivery"]["stop_after_four_or_blocked"] and p["delivery"]["no_force_push"], "delivery")
    print(json.dumps({"protocol": "042", "revision": 2, "registration_valid": True,
                      "plan_sha256": wanted, "science_started": False,
                      "scientific_validity": "not_evaluated"}, indent=2))

if __name__ == "__main__":
    main()
