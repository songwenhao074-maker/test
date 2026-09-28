"""Validate the Protocol034 experiment plan without training or modifying files."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(p):
    assert p["protocol"] == "034" and p["revision"] == 1
    assert p["status"] == "planned_not_implemented_not_run"
    src = p["source"]
    assert src["commit"] == "0463cd365b4cda5a50423b4fccc941fe8d074bbf"
    assert src["artifact_id"] == 10968752245 and src["run_id"] == 36417604442
    assert src["artifact_zip_sha256"] == "d92a3f9c8f1ebd292b86fbbe6e2300291135376f3564f2773975fd9203b55944"
    assert src["stream_sha256"] == "70c9283f133a5f8033e12fbc03a38af6d9ce39f1d7e6a9e825964ee9a28edbd9"
    inp = p["input"]
    assert (inp["scored_rows"], inp["guard_rows"], inp["total_rows"]) == (5968, 1, 5969)
    assert inp["new_generation"] is False
    assert (inp["replay_seed"], inp["model_seed"]) == (700, 1)
    budget = p["budget"]
    assert budget["full_training_replays"] == 3
    assert budget["training_arms"] == ["C_ref", "D_frozen_ref", "D_live"]
    assert budget["routing_arms"] == ["D_route_fragment", "D_route_full"]
    assert budget["prediction_cache_routing_passes"] == 2
    assert budget["new_simulator_streams"] == budget["extra_seeds"] == budget["threshold_sweeps"] == 0
    assert budget["additional_training_after_results"] is False
    assert p["training"]["loss_changes"] is False
    assert p["training"]["feature_changes"] is False
    assert p["training"]["quality_gate_changes"] is False
    assert p["D_live"]["active_specialist_trainable"] is True
    assert p["D_live"]["active_specialist_router_trainable"] is True
    assert p["D_live"]["dormant_specialist_trainable"] is False
    assert p["D_live"]["snapshot_observer_changes_predictions"] is False
    expected = [["U_rec1",3300,3428],["V_rec1",3808,3936],
                ["U_rec2",4316,4444],["V_rec2",4824,4952],
                ["U_rec3",5332,5460],["V_rec3",5840,5968]]
    assert p["evaluation"]["windows"] == expected
    assert all(end - start == 128 for _, start, end in expected)
    assert len({name for name, _, _ in expected}) == 6
    assert p["evaluation"]["development_signal"]["mean_AP_delta_min"] == 0.005
    assert p["H2"]["selection_uses_future_labels"] is True
    assert p["H2"]["online_claim_allowed"] is False
    assert p["H3"]["scores_use_future_labels"] is False
    assert p["H3"]["run_both_cache_passes_even_if_H2_negative"] is True
    router = p["routing"]
    assert router["score_window_matured_intervals"] == router["min_support_distinct_intervals"] == 32
    assert router["minimum_relative_loss_improvement"] == 0.01
    assert router["score_time_rule"] == "i + 2 < t"
    assert router["current_labels_allowed"] is False
    assert router["learning_feedback"] is False
    assert router["trained_parameters"] is False
    assert p["workflow"]["trigger"] == "workflow_dispatch_only"
    assert p["workflow"]["launch_in_this_turn"] is False
    assert p["evaluation"]["statistical_confirmation"] is False
    return {"protocol": "034", "plan_validation_passed": True,
            "training_replays": 3, "cache_routing_passes": 2,
            "new_generation": False, "recurrence_windows": 6,
            "implementation_verified": False, "experiments_run": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "artifacts/ftmoe_online/protocol_034/plan.json")
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.plan.read_text(encoding="utf-8"))), indent=2))
