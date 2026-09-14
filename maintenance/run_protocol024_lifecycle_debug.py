"""Exercise Protocol-024 causal lifecycle before generating the new stream.

Part A is explicitly test-only and forces the trigger timing (never acceptance)
to guarantee that candidate training and prospective validation execute. Part B
checks resume while a shadow optimizer is live. Part C enables the real loss
trigger on the old Protocol-023 seed700 stream as a debugging sanity check only;
no performance conclusion is drawn from the old task.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_lifecycle import LifecycleProtocol024Session


def make_session(bundle, frozen, anchor, out_dir, run_id, config):
    return LifecycleProtocol024Session(
        "D", 1, bundle, frozen, out_dir,
        anchor=anchor, learning_rate=1e-4, run_id=run_id,
        stream_dir=s4.DEV_STREAM,
        phase_defs=[{"name": "debug", "start": 0,
                     "end": bundle["steps"], "regime": "old_p23_debug"}],
        target_mode="tol1_regression",
        stream_sha=bundle["manifest"]["stream_sha256"],
        registration={"formal_performance_result": False,
                      "purpose": "lifecycle_mechanism_debug_only"},
        lifecycle_config=config)


def shadow_digest(session):
    bank = session.model.learner
    if bank.shadow_id is None:
        return None
    key = bank.shadow_id
    digest = hashlib.sha256()
    for name, value in sorted(bank.shadow_experts[key].state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(bank.shadow_router_weights[key].detach().cpu().contiguous().numpy().tobytes())
    digest.update(bank.shadow_router_biases[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_step_values(optimizer):
    values = []
    for state in optimizer.state.values():
        if "step" in state:
            value = state["step"]
            if torch.is_tensor(value):
                value = value.item()
            values.append(float(value))
    return sorted(values)


def run(out_path, natural_intervals=760):
    s4.configure()
    torch.use_deterministic_algorithms(True)
    budget = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
    frozen = budget["frozen_configuration"]
    bundle = s4.build_replay(s4.DEV_STREAM)
    anchor = s4.load_anchor_pool(bundle["replay"].time_scale)
    result = {"protocol": "024", "kind": "causal_lifecycle_debug",
              "formal_performance_result": False,
              "source_stream_sha256": bundle["manifest"]["stream_sha256"]}

    with tempfile.TemporaryDirectory(prefix="p24_lifecycle_debug_") as temp:
        temp = Path(temp)
        # A. Forced trigger timing only.  Candidate acceptance is still decided
        # prospectively from the paired losses; activation is never forced.
        forced_cfg = {
            "calibration_intervals": 16,
            "trigger_window": 8,
            "consecutive_abnormal_windows": 1,
            "candidate_train_intervals": 8,
            "validation_intervals": 6,
            "cooldown_intervals": 16,
            "debug_force_trigger_after": 20,
        }
        forced = make_session(bundle, frozen, anchor, temp / "forced",
                              "lifecycle_forced_toy", forced_cfg)
        for _ in range(80):
            forced.step()
        controller = forced.lifecycle_controller
        kinds = [event["kind"] for event in controller.events]
        if "candidate_created" not in kinds or "candidate_training_complete" not in kinds:
            raise AssertionError("forced toy did not exercise candidate train path")
        if not any(kind in kinds for kind in ("candidate_accepted", "candidate_rejected")):
            raise AssertionError("forced toy did not reach prospective decision")
        if set(controller.candidate_training_indices) & set(controller.candidate_validation_indices):
            raise AssertionError("candidate training/validation intervals overlap")
        if controller.extra_compute["shadow_train_steps"] < 8:
            raise AssertionError("candidate did not train on registered distinct intervals")
        if controller.extra_compute["shadow_validation_forwards"] < 6:
            raise AssertionError("candidate prospective validation did not execute")
        result["forced_toy"] = {
            "events": controller.events,
            "last_decision": controller.last_decision,
            "training_indices": controller.candidate_training_indices,
            "validation_indices": controller.candidate_validation_indices,
            "extra_compute": controller.extra_compute,
            "final_topology": forced.model.learner.topology_manifest(),
            "acceptance_forced": False,
            "trigger_timing_forced": True,
        }

        # B. Resume while the shadow optimizer is live.  Candidate is created
        # by the test harness, but the next training step is driven by a real
        # matured label through LifecycleProtocol024Session.step().
        resume_cfg = {
            "calibration_intervals": 16,
            "trigger_window": 8,
            "consecutive_abnormal_windows": 2,
            "candidate_train_intervals": 8,
            "validation_intervals": 6,
            "cooldown_intervals": 16,
            "debug_force_trigger_after": None,
        }
        live = make_session(bundle, frozen, anchor, temp / "resume_live",
                            "lifecycle_shadow_resume", resume_cfg)
        for _ in range(24):
            live.step()
        ctl = live.lifecycle_controller
        if live.model.learner.shadow_id is None:
            ctl._start_candidate(live, "test_only_resume_fixture")
        ctl.phase = "candidate_training"
        ctl._sync_session(live)
        # Let one genuine subsequent matured interval train the shadow.
        live.step()
        if live.shadow_optimizer is None or not live.shadow_optimizer.state:
            raise AssertionError("shadow optimizer has no state before checkpoint")
        checkpoint = live.save_checkpoint("candidate_inflight", live.cursor)
        clone = make_session(bundle, frozen, anchor, temp / "resume_clone",
                             "lifecycle_shadow_resume", resume_cfg)
        clone.restore_checkpoint(checkpoint["path"])
        if clone.shadow_optimizer is None:
            raise AssertionError("shadow optimizer was not reconstructed on resume")
        if optimizer_step_values(clone.shadow_optimizer) != optimizer_step_values(live.shadow_optimizer):
            raise AssertionError("shadow optimizer steps changed on restore")
        before_live = shadow_digest(live)
        before_clone = shadow_digest(clone)
        p_live, c_live = live.step()
        after_live = shadow_digest(live)
        p_clone, c_clone = clone.step()
        after_clone = shadow_digest(clone)
        if before_live != before_clone or after_live != after_clone:
            raise AssertionError("in-flight candidate tensors diverged across resume")
        if not np.array_equal(p_live, p_clone) or not np.array_equal(c_live, c_clone):
            raise AssertionError("in-flight candidate resume changed live prediction")
        result["candidate_resume"] = {
            "shadow_id": live.model.learner.shadow_id,
            "optimizer_steps_before": optimizer_step_values(live.shadow_optimizer),
            "prediction_equal": True,
            "shadow_digest_before_equal": before_live == before_clone,
            "shadow_digest_after_next_matured_train_equal": after_live == after_clone,
            "controller_phase_equal": (live.lifecycle_controller.phase ==
                                       clone.lifecycle_controller.phase),
            "training_indices_equal": (live.lifecycle_controller.candidate_training_indices ==
                                       clone.lifecycle_controller.candidate_training_indices),
        }

        # C. Natural trigger on old P23 stream. This is only a debugging sanity
        # run. It may legitimately produce zero accepted experts.
        natural_cfg = {
            "calibration_intervals": 256,
            "trigger_window": 32,
            "consecutive_abnormal_windows": 2,
            "candidate_train_intervals": 64,
            "validation_intervals": 32,
            "cooldown_intervals": 64,
            "debug_force_trigger_after": None,
        }
        natural = make_session(bundle, frozen, anchor, temp / "natural",
                               "lifecycle_old_p23_natural", natural_cfg)
        limit = min(int(natural_intervals), natural.steps)
        for _ in range(limit):
            natural.step()
        nctl = natural.lifecycle_controller
        kinds = [e["kind"] for e in nctl.events]
        result["old_p23_natural_debug"] = {
            "intervals": limit,
            "calibration_threshold": nctl.calibration_threshold,
            "phase": nctl.phase,
            "event_counts": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
            "candidate_created": kinds.count("candidate_created"),
            "candidate_accepted": kinds.count("candidate_accepted"),
            "candidate_rejected": kinds.count("candidate_rejected"),
            "active_ids": list(natural.model.learner.ids),
            "resident_count": natural.model.learner.resident_count(),
            "extra_compute": nctl.extra_compute,
            "performance_claim_allowed": False,
        }

    result["checks"] = {
        "forced_toy_candidate_training": True,
        "forced_toy_prospective_validation": True,
        "forced_toy_no_forced_acceptance": True,
        "training_validation_disjoint": True,
        "candidate_phase_resume_equal": True,
        "old_p23_natural_debug_completed": True,
    }
    result["passed"] = all(result["checks"].values())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--natural-intervals", type=int, default=760)
    args = parser.parse_args()
    result = run(args.out, args.natural_intervals)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
