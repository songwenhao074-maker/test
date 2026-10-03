#!/usr/bin/env python3
"""Validate Protocol-041 registration only; does not certify execution or launch work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/ftmoe_online/protocol_041/plan.json"

def check(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = json.loads(PLAN.read_text(encoding="utf-8"))
    registered = PLAN.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    check(digest(PLAN) == registered, "plan digest mismatch")
    check(p["protocol"] == "041" and p["revision"] == 1, "registration identity")
    check(p["status"] == "instructions_registered_not_implemented_not_started", "immutable registration status")
    check(digest(ROOT / p["instructions"]) == p["instructions_sha256"], "instruction digest mismatch")
    old_path = ROOT / p["historical_controller_plan"]["path"]
    check(digest(old_path) == p["historical_controller_plan"]["sha256"], "040 registration changed")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    for name, value in p["inherited_controller"].items():
        check(value == old[name], "controller changed: " + name)
    check(p["source036"] == old["source036"], "036 input lock changed")
    check(p["data"] == old["data"], "data/causal timing changed")
    expert = dict(old["expert"])
    del expert["initialization"]
    check(p["expert"] == expert, "training kernel changed")
    check(p["arms"]["new"] == ["Z_zero", "W_parent"], "unregistered science arm")
    check(p["arms"]["sequence_order"] == p["arms"]["new"], "sequence order changed")
    check(p["arms"]["Z_zero"]["second_candidate"] == "zero", "zero control changed")
    check(p["arms"]["W_parent"]["second_candidate"] ==
          "deep_copy_current_active_parent_weights_and_bias_before_same_t_updates", "warm mapping changed")
    check(p["arms"]["Z_zero"]["optimizer"] == p["arms"]["W_parent"]["optimizer"] == "fresh_AdamW", "optimizer mapping changed")
    init = p["initialization"]
    check(init["copy_after_settlement_before_same_t_gradients"], "unsafe donor time")
    for name in ("copy_optimizer", "copy_rng", "shared_parameter_storage", "copy_B_D_lin", "add_parent_and_candidate_corrections"):
        check(init[name] is False, "forbidden initialization option: " + name)
    check(init["first_child_optimizer_step"] == 1 and init["extra_donor_training_steps"] == 0, "hidden donor/update")
    b = p["budget"]
    check(b["new_science_sequences"] == 2 and b["sequence_names"] == p["arms"]["new"], "science budget identity")
    for name in ("new_streams", "extra_seeds", "control_retraining", "F_training", "F_comparisons", "F_loads", "donor_updates", "hyperparameter_sweeps"):
        check(b[name] == 0, "forbidden budget: " + name)
    per = b["per_arm"]
    expected = {"live_optimizer_steps_max": 352, "shadow_optimizer_steps_max": 16,
                "total_optimizer_steps_max": 368, "max_updates_per_due16": 2,
                "deployed_dynamic_prediction_forwards_max": 5616, "reuse_preview_forwards_max": 2048,
                "shadow_qualification_forwards_max": 32, "total_preview_forwards_max": 2080}
    check(per == expected, "per-arm budget changed")
    for name, value in b["combined"].items():
        check(value == 2 * per[name], "combined budget mismatch: " + name)
    check(per["total_optimizer_steps_max"] == per["live_optimizer_steps_max"] + per["shadow_optimizer_steps_max"], "gradient accounting")
    check(per["total_preview_forwards_max"] == per["reuse_preview_forwards_max"] + per["shadow_qualification_forwards_max"], "preview accounting")
    check(b["real_engineering_prefixes_max"] == 1 and b["real_engineering_prefix_intervals_max"] == 256 and b["real_engineering_gradient_steps"] == 0, "real engineering budget")
    check(b["restart_from_zero_after_science_start"] is False, "unregistered restart")
    check(p["reproduction"]["prediction_arrays"] == "exact" and p["reproduction"]["AP_absolute_tolerance"] == 1e-12, "reproduction criteria changed")
    for name in ("windows", "late4", "prefix_lengths", "primary", "cumulative_preservation", "guards", "pool_exercised", "labels_priority"):
        check(p["analysis"][name] == old["analysis"][name], "quality/pool gate changed: " + name)
    check(p["analysis"]["initialization_system_signal"] == {
        "full_AP_delta_min": 0, "six_AP_delta_min": 0.0005, "positive_windows_min": 4,
        "late4_AP_delta_min": 0, "guards_vs_Z_required": True, "validity_required": True}, "initialization gate changed")
    required = [
        "zero_B_and_nonzero_gradient", "parent_copy_timing_no_alias_fresh_adam",
        "full_adam_clone_synthetic_equivalence", "expert_optimizer_and_dormant_isolation",
        "controller_competing_events_capacity_and_censoring", "disk_resume_all_substeps_to_terminal",
        "future_feature_and_label_perturbation", "actual_call_counters_terminal_zero",
        "AP_ties_single_class_confusion_and_class_copy", "complete_opportunity_accounting"]
    check(p["engineering"]["required_fixture_ids"] == required, "required fixture missing")
    check(p["engineering"]["checkpoint_semantics"] == "cursor_plus_next_unexecuted_substep", "unsafe recovery semantics")
    check(p["engineering"]["serialize_shadow_ready"] and p["engineering"]["serialize_terminal_settlement_progress"], "incomplete checkpoint")
    src = p["source040"]
    check(src["run_id"] == 37098618766 and src["artifact_id"] == 11265231995, "040 source identity")
    manifest_path = ROOT / src["manifest_path"]
    check(digest(manifest_path) == src["manifest_sha256"], "040 manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locks = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    for path, wanted in src["files"].items():
        check(locks.get(path) == wanted, "040 input mismatch: " + path)
    check(src["restore_historical_checkpoints"] is False and src["load_F_files"] is False, "unsafe source use")
    auth = p["authorization"]
    check(auth["this_publication"] == "documents_and_registration_only" and auth["launch_in_this_turn"] is False, "publication scope")
    check(auth["automatic_next_protocol"] is False and auth["F_research"] == "DEFERRED", "followup scope")
    check(p["runtime"]["formal_trigger"] == "workflow_dispatch_only" and p["runtime"]["create_workflow_this_publication"] is False, "workflow scope")
    check(p["delivery"]["stop_after_registered_two_sequences"] and p["delivery"]["no_force_push"], "stop/publication rule")
    print(json.dumps({"protocol": "041", "registration_valid": True,
                      "plan_sha256": registered, "science_started": False,
                      "scientific_validity": "not_evaluated"}, indent=2))

if __name__ == "__main__":
    main()
