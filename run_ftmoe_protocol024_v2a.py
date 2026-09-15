"""Run Protocol-024 next_round_v2a on immutable seed700 response-law stream."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_anchor import anchor_metadata, load_protocol024_raw_next_anchor
from ftmoe_protocol024_eval import (
    average_precision_report,
    binary_detection_metrics,
    positive_resource_macro_f1,
    temporal_onset_metrics,
)
from ftmoe_protocol024_session import Protocol024Session, NEXT_TARGET_MODE
from ftmoe_protocol024_v2a import (
    V2AProtocol024Session,
    V2A_DEFAULT,
    deployable_pressure,
    split_anchor_train_guard,
)


FIRST_NAMES = ("R1_first", "R2_first", "R3_first")
RECURRENCE_NAMES = ("R1_rec1", "R3_rec1", "R2_rec1",
                    "R1_rec2", "R2_rec2", "R3_rec2")
EXPECTED_STREAM_SHA = "468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946"


def _phase_defs(manifest):
    return [{"name": p["name"], "start": int(p["start"]), "end": int(p["end"]),
             "response_law": p.get("response_law")} for p in manifest["timeline"]]


def _budget():
    out = dict(json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
               ["frozen_configuration"])
    out.update({"update_every_scored_intervals": 4,
                "batch_size": 32,
                "gradient_steps_per_opportunity": 1,
                "replay_buffer_intervals": 64,
                "learning_rate": 1e-4})
    return out


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                    encoding="utf8")


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def _run_arm(arm, stream_dir, train_anchor, guard_anchor, out_root,
             v2a_config, run_id):
    bundle = s4.build_replay(stream_dir)
    if bundle["manifest"].get("stream_sha256") != EXPECTED_STREAM_SHA:
        raise AssertionError("v2a immutable stream SHA mismatch")
    out_dir = Path(out_root) / ("arm_" + arm)
    if out_dir.exists():
        raise FileExistsError("refusing to overwrite v2a arm directory %s" % out_dir)
    phases = _phase_defs(bundle["manifest"])
    registration = {
        "protocol": "024", "round": "next_round_v2", "stage": "v2a",
        "run_id": run_id, "arm": arm, "replay_seed": 700, "model_seed": 1,
        "target": NEXT_TARGET_MODE, "update_every": 4,
        "anchor": anchor_metadata(train_anchor),
        "guard": anchor_metadata(guard_anchor),
        "shadow_budget_version": v2a_config["shadow_budget_version"],
        "formal_development_result": True, "confirmation_run": False,
    }
    cls = V2AProtocol024Session if arm == "D" else Protocol024Session
    kwargs = {}
    if arm == "D":
        kwargs.update({"guard_anchor": guard_anchor, "v2a_config": v2a_config})
    session = cls(
        arm, 1, bundle, _budget(), out_dir,
        anchor=train_anchor, learning_rate=1e-4,
        run_id=run_id, stream_dir=stream_dir, phase_defs=phases,
        target_mode=NEXT_TARGET_MODE,
        stream_sha=bundle["manifest"]["stream_sha256"],
        registration=registration, **kwargs)
    started = time.perf_counter()
    for _ in range(session.steps):
        session.step()
    session.finish()
    elapsed = time.perf_counter() - started
    session.save()
    phase_metrics = session.phase_metrics()
    update_seconds = [float(x["seconds"]) for x in session.update_log]
    summary = {
        "protocol": "024", "round": "next_round_v2", "stage": "v2a",
        "arm": arm, "run_id": run_id, "target": NEXT_TARGET_MODE,
        "stream_sha256": EXPECTED_STREAM_SHA, "replay_seed": 700,
        "model_seed": 1, "update_every": 4, "updates": int(session.updates),
        "elapsed_seconds": float(elapsed),
        "prediction_mean_seconds": float(np.mean(session.predictions["prediction_seconds"])),
        "prediction_p95_seconds": float(np.percentile(session.predictions["prediction_seconds"], 95)),
        "update_mean_seconds": float(np.mean(update_seconds)) if update_seconds else None,
        "update_p95_seconds": float(np.percentile(update_seconds, 95)) if update_seconds else None,
        "full": session._metric_block(0, session.steps),
        "phase_metrics": phase_metrics,
        "frozen_base_hash": session.model.frozen_hash(),
        "anchor": anchor_metadata(train_anchor),
        "confirmation_run": False,
    }
    if arm == "D":
        controller = session.lifecycle_controller
        events = list(controller.events)
        counts = dict(Counter(x["kind"] for x in events))
        _write_jsonl(out_dir / "lifecycle.jsonl", events)
        summary["lifecycle"] = {
            "event_counts": counts,
            "proposal_opportunities": int(controller.proposal_opportunities),
            "proposal_skipped": int(controller.proposal_skipped),
            "candidate_created": int(counts.get("candidate_created", 0)),
            "candidate_accepted": int(counts.get("candidate_accepted", 0)),
            "candidate_rejected": int(counts.get("candidate_rejected", 0)),
            "capacity_blocked": int(controller.capacity_blocked),
            "births": int(counts.get("candidate_accepted", 0)),
            "retirements": 0, "reactivations": 0, "purges": 0,
            "accepted_ids": list(controller.accepted_ids),
            "rejected_ids": list(controller.rejected_ids),
            "v1_p99_diagnostic_threshold": controller.v1_diagnostic_threshold,
            "extra_compute": controller.extra_compute,
            "last_decision": controller.last_decision,
            "final_topology": session.model.learner.topology_manifest(),
            "zero_event_explanation": {
                "trigger_not_reached": int(controller.proposal_opportunities == 0),
                "candidate_created_but_rejected": int(
                    counts.get("candidate_created", 0) > 0 and
                    counts.get("candidate_accepted", 0) == 0),
                "capacity_blocked": int(controller.capacity_blocked > 0),
                "accepted": int(counts.get("candidate_accepted", 0)),
                "retired": 0, "reactivated": 0, "purged": 0,
            },
        }
    _write_json(out_dir / "summary.json", summary)
    return session, summary


def _model_metrics(session):
    return {
        "detection": binary_detection_metrics(
            session.predictions["probability"], session.predictions["labels"]),
        "onset": temporal_onset_metrics(
            session.predictions["probability"], session.predictions["raw_labels"], 1),
        "resource": positive_resource_macro_f1(
            session.predictions["class_probability"], session.predictions["labels"]),
    }


def _ap(y, score):
    return average_precision_report(np.asarray(y, dtype=np.int64).reshape(-1),
                                    np.asarray(score, dtype=np.float64).reshape(-1))


def _comparison(sessions, summaries, bundle, v2a_config):
    phases = _phase_defs(bundle["manifest"])
    by_name = {x["name"]: x for x in phases}
    switches = []
    for name in FIRST_NAMES + RECURRENCE_NAMES:
        p = by_name[name]
        start, end = p["start"], min(p["start"] + 100, p["end"])
        arms = {}
        for arm, session in sessions.items():
            arms[arm] = binary_detection_metrics(
                session.predictions["probability"][start:end],
                session.predictions["labels"][start:end])
        c_ap, d_ap = arms["C"]["ap"], arms["D"]["ap"]
        switches.append({
            "phase": name, "response_law": p.get("response_law"),
            "intervals": [start, end], "arms": arms,
            "D_minus_C_ap": None if c_ap is None or d_ap is None else float(d_ap - c_ap),
        })
    recurrence = [x for x in switches if x["phase"] in RECURRENCE_NAMES]
    deltas = [x["D_minus_C_ap"] for x in recurrence if x["D_minus_C_ap"] is not None]
    recurrence_indices = np.concatenate([
        np.arange(x["intervals"][0], x["intervals"][1], dtype=np.int64)
        for x in recurrence])
    threshold = {
        arm: binary_detection_metrics(session.predictions["probability"][recurrence_indices],
                                      session.predictions["labels"][recurrence_indices])
        for arm, session in sessions.items()}
    c_fpr, d_fpr = threshold["C"]["fpr"], threshold["D"]["fpr"]
    fpr_delta = None if c_fpr is None or d_fpr is None else float(d_fpr - c_fpr)

    arrays = bundle["arrays"]
    raw = np.asarray(arrays["raw_labels"], dtype=np.int64)
    target = raw[1:4981]
    pressure = deployable_pressure(
        arrays["host_features"][:4980], arrays["capacities"][:4980])
    delayed = np.zeros((4980, 16), dtype=np.float64)
    delayed[1:] = (raw[:4979] > 0).astype(np.float64)
    post_pressure = np.asarray(arrays["overload_ratio"][:4980]).max(-1)
    post_fault = (raw[:4980] > 0).astype(np.float64)
    baselines = {
        "deployable_pre_step_pressure": dict(
            _ap(target, pressure), availability="host_features[t] and capacities[t] before simulationStep"),
        "deployable_delayed_persistence_t_minus_1": dict(
            _ap(target[1:], delayed[1:]), availability="raw[t-1] already published before prediction t",
            excluded_prediction_t0=True),
        "post_step_pressure_reference_not_deployable": dict(
            _ap(target, post_pressure), availability="overload_ratio[t] after simulationStep"),
        "post_step_fault_reference_not_deployable": dict(
            _ap(target, post_fault), availability="raw[t] after simulationStep"),
    }
    mean_delta = float(np.mean(deltas)) if deltas else None
    positive = int(sum(x > 0 for x in deltas))
    enough = len(deltas) == len(RECURRENCE_NAMES)
    signal = None if not enough or fpr_delta is None else bool(
        mean_delta >= 0.03 and positive >= 4 and fpr_delta <= 0.01)
    return {
        "protocol": "024", "round": "next_round_v2", "stage": "v2a",
        "shadow_budget_version": v2a_config["shadow_budget_version"],
        "development_only": True, "confirmation_run": False,
        "stream_sha256": EXPECTED_STREAM_SHA, "replay_seed": 700, "model_seed": 1,
        "full": {arm: _model_metrics(session) for arm, session in sessions.items()},
        "switch_first100": switches,
        "recurrence": {
            "valid_switches": len(deltas), "total_switches": len(RECURRENCE_NAMES),
            "coverage": len(deltas) / float(len(RECURRENCE_NAMES)),
            "mean_D_minus_C_ap": mean_delta,
            "positive_D_minus_C_switches": positive,
            "threshold_metrics": threshold,
            "normal_fpr_D_minus_C": fpr_delta,
        },
        "baselines": baselines,
        "availability": {
            "prediction_order": "predict[t] -> simulationStep/observe raw[t] -> settle target[t-2]",
            "host_features_t": "available_before_prediction_t; collected before simulationStep",
            "demands_t": "available before simulationStep",
            "schedules_t": "scheduler decision available before simulationStep; not execution result",
            "raw_t": "NOT available until after prediction_t",
            "overload_ratio_t": "NOT available until after prediction_t",
            "raw_t_plus_1": "future; target for prediction t, matures no earlier than t+2",
            "pressure_feature_indices": {"cpu": 0, "ram_size": 1, "disk_size": 4},
        },
        "lifecycle": summaries["D"].get("lifecycle"),
        "development_signal": signal,
        "development_signal_rule": "six recurrence mean D-C>=0.03, >=4/6 positive, normal FPR delta<=0.01",
    }


def run(stream_dir, out_root, shadow_budget_version):
    stream_dir, out_root = Path(stream_dir), Path(out_root)
    if out_root.exists():
        raise FileExistsError("refusing to overwrite completed/partial v2a run %s" % out_root)
    out_root.mkdir(parents=True)
    bundle = s4.build_replay(stream_dir)
    if bundle["steps"] != 4980 or bundle["manifest"]["stream_sha256"] != EXPECTED_STREAM_SHA:
        raise AssertionError("v2a requires immutable registered seed700 stream")
    anchor = load_protocol024_raw_next_anchor()
    train_anchor, guard_anchor = split_anchor_train_guard(anchor, modulo=5, guard_remainder=0)
    guard_labels = guard_anchor["labels"].numpy()
    if not ((guard_labels == 0).any() and (guard_labels > 0).any()):
        raise RuntimeError("registered old-knowledge guard lacks normal or positive rows")
    v2a_config = dict(V2A_DEFAULT)
    v2a_config["shadow_budget_version"] = str(shadow_budget_version)
    if shadow_budget_version == "buffer128_4x4":
        v2a_config["candidate_train_intervals"] = 128
    elif shadow_budget_version != "distinct64_single":
        raise ValueError("unsupported v2a shadow budget version")
    run_id = "seed700_model1_v2a_%s" % shadow_budget_version
    sessions, summaries = {}, {}
    for arm in ("A", "C", "D"):
        sessions[arm], summaries[arm] = _run_arm(
            arm, stream_dir, train_anchor, guard_anchor, out_root,
            v2a_config, run_id)
    comparison = _comparison(sessions, summaries, bundle, v2a_config)
    _write_json(out_root / "comparison.json", comparison)
    status = {
        "protocol": "024", "round": "next_round_v2", "stage": "v2a",
        "run_id": run_id, "shadow_budget_version": shadow_budget_version,
        "completed": True, "confirmation_run": False,
        "candidate_created": comparison["lifecycle"]["candidate_created"],
        "candidate_accepted": comparison["lifecycle"]["candidate_accepted"],
        "candidate_rejected": comparison["lifecycle"]["candidate_rejected"],
        "capacity_blocked": comparison["lifecycle"]["capacity_blocked"],
        "development_signal": comparison["development_signal"],
        "advance_to_v2b": bool(comparison["lifecycle"]["candidate_accepted"] > 0),
        "second_budget_allowed": bool(
            shadow_budget_version == "distinct64_single" and
            comparison["lifecycle"]["candidate_accepted"] == 0),
        "seeds": {"replay": 700, "model": 1, "confirmation_used": []},
    }
    _write_json(out_root / "status.json", status)
    _write_json(out_root / "anchor_split.json", {
        "train": anchor_metadata(train_anchor), "guard": anchor_metadata(guard_anchor),
        "guard_class_counts": {str(k): int((guard_labels == k).sum()) for k in range(4)},
        "selection_depends_on_target": False,
    })
    print(json.dumps({"status": status, "recurrence": comparison["recurrence"],
                      "baselines": comparison["baselines"],
                      "lifecycle": comparison["lifecycle"]},
                     indent=2, allow_nan=False), flush=True)
    return comparison, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--shadow-budget-version", default="distinct64_single",
                        choices=("distinct64_single", "buffer128_4x4"))
    args = parser.parse_args()
    run(args.stream, args.out_root, args.shadow_budget_version)


if __name__ == "__main__":
    main()
