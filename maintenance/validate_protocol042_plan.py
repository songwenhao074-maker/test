#!/usr/bin/env python3
"""Validate Protocol-042 registration only. Never launches or certifies science."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/ftmoe_online/protocol_042/plan.json"

def check(value, message):
    if not value:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = json.loads(PLAN.read_text(encoding="utf-8"))
    wanted = PLAN.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    check(digest(PLAN) == wanted, "plan digest")
    check(p["protocol"] == "042" and p["revision"] == 1, "identity")
    check(p["status"] == "instructions_registered_not_implemented_not_started", "immutable registration status")
    check(digest(ROOT / p["instructions"]) == p["instructions_sha256"], "instructions digest")
    prior_path = ROOT / p["historical_registration"]["path"]
    check(digest(prior_path) == p["historical_registration"]["sha256"], "historical registration digest")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    for key in ("source036", "data", "expert", "initialization", "inherited_controller"):
        check(p[key] == prior[key], "unregistered inherited change: " + key)
    check(p["arms"]["new"] == p["arms"]["sequence_order"] == ["U_uniform", "H_hard"], "arms/order")
    check(p["arms"]["U_uniform"]["shadow_BCE"] == "original_unweighted_kernel", "uniform kernel")
    check(p["arms"]["H_hard"]["shadow_BCE"] == "mature_issued_live_error_weighted", "weighted kernel")
    check(p["arms"]["live_loss_both"] == "original_unweighted_including_accepted_candidate", "live loss")
    w = p["weighting"]
    check(w["coefficient"] == 2 and w["raw_weight_bounds"] == [1, 3], "weight constants")
    check(w["error"] == "abs(y-p_live_issued)" and w["raw_weight"] == "1+2*error", "weight formula")
    check(w["probability_source"] == "own_arm_out.probability[i,h]_issued_before_label" and w["maturity"] == "i+2<=t", "causal source")
    check(w["normalization"] == "divide_by_float32_mean_over_all_batch_occurrences_and_hosts", "normalization")
    check(w["dtype"] == "float32" and w["order"] == "batch_occurrence_then_host", "weight precision/order")
    for key in ("requires_grad", "regularizer_weighted", "normalize_per_class", "normalize_per_host",
                "resample", "replace_batch", "qualification_weighted", "coefficient_sweep"):
        check(w[key] is False, "forbidden weighting option: " + key)
    check(w["extra_model_forwards"] == 0 and w["future_or_recomputed_probability_forbidden"], "hidden forwards/leakage")
    golden = w["golden_fixture"]
    raw = [1 + 2 * abs(y - prob) for y, prob in zip(golden["y"], golden["p"])]
    applied = [v / (sum(raw) / len(raw)) for v in raw]
    check(all(abs(a - b) <= 1e-6 for a, b in zip(raw, golden["expected_raw"])), "golden raw weights")
    check(all(abs(a - b) <= 1e-6 for a, b in zip(applied, golden["expected_applied"])), "golden applied weights")
    b = p["budget"]
    check(b["new_science_sequences"] == 2 and b["sequence_names"] == p["arms"]["new"], "science budget")
    check(b["per_arm"] == prior["budget"]["per_arm"], "per-arm budget changed")
    check(b["combined"] == prior["budget"]["combined"], "combined budget changed")
    check(b["combined"]["total_optimizer_steps_max"] == 736, "gradient total")
    for key, value in b["combined"].items():
        check(value == 2 * b["per_arm"][key], "combined arithmetic: " + key)
    for key in ("new_streams", "extra_seeds", "control_retraining", "F_training", "F_comparisons",
                "F_loads", "donor_updates", "hyperparameter_sweeps", "weight_computation_extra_model_forwards"):
        check(b[key] == 0, "forbidden budget: " + key)
    check(b["real_engineering_prefixes_max"] == 1 and b["real_engineering_prefix_intervals_max"] == 256
          and b["real_engineering_gradient_steps"] == 0, "engineering budget")
    check(b["restart_from_zero_after_science_start"] is False, "restart")
    for key in ("windows", "late4", "prefix_lengths", "primary", "cumulative_preservation",
                "guards", "pool_exercised", "labels_priority"):
        check(p["analysis"][key] == prior["analysis"][key], "quality gate changed: " + key)
    check(p["analysis"]["hard_weighting_system_signal"] == {
        "full_AP_delta_min": 0, "six_AP_delta_min": 0.0005, "positive_windows_min": 4,
        "late4_AP_delta_min": 0, "guards_vs_U_required": True, "validity_required": True}, "system signal")
    required = [
        "weight_formula_and_unweighted_regularizer", "issued_error_provenance_maturity_future_perturbation",
        "parent_initialization_and_expert_isolation", "eventful_disk_resume_through_gradients_and_decisions",
        "crash_injection_transaction_and_ready_integrity", "controller_competition_capacity_opportunity_terminal",
        "reproduction_schema_positive_and_negative_controls", "fail_closed_workflow_and_H_entrypoint_gate",
        "independent_metrics_and_weight_recompute"]
    check(p["engineering"]["required_fixture_ids"] == required, "fixture inventory")
    check(len(p["engineering"]["required_resume_breakpoints"]) == 12, "resume coverage")
    check(p["engineering"]["serialize_shadow_ready"] and p["engineering"]["persist_all_RNGs"], "checkpoint content")
    check(p["gate"]["enforce_in_workflow_and_H_process"] and p["gate"]["no_OR_of_optional_rc_fields"], "unsafe gate")
    check(p["gate"]["missing_empty_string_or_string_true"] == "fail_closed", "missing-value gate")
    check(len(p["gate"]["negative_gate_tests"]) == 10, "negative gate tests")
    check(p["reproduction"]["prediction_arrays"] == p["reproduction"]["qualification_arrays"] == "exact", "reproduction arrays")
    check(p["reproduction"]["metric_absolute_tolerance"] == 1e-12, "metric tolerance")
    src = p["source041"]
    check(src["run_id"] == 37110969550 and src["artifact_id"] == 11269043834, "041 source identity")
    mp = ROOT / src["manifest_path"]
    check(digest(mp) == src["manifest_sha256"], "041 immutable manifest")
    locked = {x["path"]: x["sha256"] for x in json.loads(mp.read_text(encoding="utf-8"))["files"]}
    for path, expected in src["files"].items():
        check(locked.get(path) == expected, "041 file lock: " + path)
    check(src["restore_historical_checkpoints"] is False and src["load_F_files"] is False, "source scope")
    check(p["authorization"]["launch_in_this_turn"] is False and p["authorization"]["automatic_next_protocol"] is False, "publication scope")
    check(p["runtime"]["formal_trigger"] == "workflow_dispatch_only" and p["runtime"]["create_workflow_this_publication"] is False, "workflow scope")
    check(p["delivery"]["stop_after_registered_two_sequences"] and p["delivery"]["no_force_push"], "stop/publication")
    print(json.dumps({"protocol": "042", "registration_valid": True, "plan_sha256": wanted,
                      "science_started": False, "scientific_validity": "not_evaluated"}, indent=2))

if __name__ == "__main__":
    main()
