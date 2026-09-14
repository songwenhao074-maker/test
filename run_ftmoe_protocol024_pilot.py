"""Protocol-024 seed700 lifecycle-on A/C/D development pilot and comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_eval import (
    average_precision_report, binary_detection_metrics,
    positive_resource_macro_f1, temporal_onset_metrics)
from ftmoe_protocol024_session import Protocol024Session, NEXT_TARGET_MODE
from ftmoe_protocol024_lifecycle import LifecycleProtocol024Session


FIRST_NAMES = ("R1_first", "R2_first", "R3_first")
RECURRENCE_NAMES = ("R1_rec1", "R3_rec1", "R2_rec1",
                    "R1_rec2", "R2_rec2", "R3_rec2")
LIFECYCLE_KEYS = (
    "calibration_intervals", "trigger_window", "consecutive_abnormal_windows",
    "loss_percentile", "candidate_train_intervals", "validation_intervals",
    "accept_relative_loss_improvement", "normal_probability_allowance",
    "cooldown_intervals", "retirement_enabled", "hard_delete_enabled",
    "debug_force_trigger_after")


def _phase_defs(manifest):
    return [{"name": p["name"], "start": int(p["start"]), "end": int(p["end"]),
             "response_law": p.get("response_law")}
            for p in manifest["timeline"]]


def _budget(update_every):
    out = dict(json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))
               ["frozen_configuration"])
    out.update({"update_every_scored_intervals": int(update_every),
                "batch_size": 32, "gradient_steps_per_opportunity": 1,
                "replay_buffer_intervals": 64, "learning_rate": 1e-4})
    return out


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def _run_arm(arm, stream_dir, bundle, anchor, update_every,
             lifecycle_config, root):
    out_dir = root / ("arm_" + arm)
    phases = _phase_defs(bundle["manifest"])
    registration = {
        "kind": "protocol024_next_round_lifecycle_on_pilot",
        "formal_performance_result": True,
        "development_only": True, "confirmation_run": False,
        "target": NEXT_TARGET_MODE,
        "update_every": int(update_every), "model_seed": 1,
        "replay_seed": 700}
    cls = LifecycleProtocol024Session if arm == "D" else Protocol024Session
    kwargs = {}
    if arm == "D":
        kwargs["lifecycle_config"] = lifecycle_config
    session = cls(
        arm, 1, bundle, _budget(update_every), out_dir,
        anchor=anchor, learning_rate=1e-4,
        run_id="seed700_rawnext_lifecycle_v1",
        stream_dir=stream_dir, phase_defs=phases,
        target_mode=NEXT_TARGET_MODE,
        stream_sha=bundle["manifest"]["stream_sha256"],
        registration=registration, **kwargs)

    # Only a compact set of resumable checkpoints is retained: after each first
    # exposure and final.  Recurrence state is fully represented by final output
    # + lifecycle ledger, avoiding hundreds of MB of redundant checkpoints.
    checkpoint_boundaries = {
        p["end"]: p["name"] for p in phases if p["name"] in FIRST_NAMES}
    checkpoints = []
    started = time.perf_counter()
    for _ in range(session.steps):
        session.step()
        if session.cursor in checkpoint_boundaries:
            checkpoints.append(session.save_checkpoint(
                checkpoint_boundaries[session.cursor], session.cursor))
    session.finish()
    checkpoints.append(session.save_checkpoint("final", session.cursor))
    elapsed = time.perf_counter() - started
    predictions_path = session.save()
    metrics = session.phase_metrics()
    update_seconds = [float(x["seconds"]) for x in session.update_log]
    summary = {
        "protocol": "024", "arm": arm, "target": NEXT_TARGET_MODE,
        "development_only": True, "confirmation_run": False,
        "stream_sha256": bundle["manifest"]["stream_sha256"],
        "update_every": int(update_every), "updates": int(session.updates),
        "elapsed_seconds": float(elapsed),
        "prediction_mean_seconds": float(np.mean(session.predictions["prediction_seconds"])),
        "prediction_p95_seconds": float(np.percentile(session.predictions["prediction_seconds"], 95)),
        "update_mean_seconds": float(np.mean(update_seconds)) if update_seconds else None,
        "update_p95_seconds": float(np.percentile(update_seconds, 95)) if update_seconds else None,
        "full": session._metric_block(0, session.steps),
        "phase_metrics": metrics,
        "checkpoints": checkpoints,
        "predictions_file": predictions_path.name,
        "frozen_base_hash": session.model.frozen_hash(),
    }
    if arm == "D":
        controller = session.lifecycle_controller
        events = list(controller.events)
        counts = dict(Counter(row["kind"] for row in events))
        _write_jsonl(out_dir / "lifecycle.jsonl", events)
        accepted = int(counts.get("candidate_accepted", 0))
        summary["lifecycle"] = {
            "event_counts": counts,
            "candidate_created": int(counts.get("candidate_created", 0)),
            "candidate_accepted": accepted,
            "candidate_rejected": int(counts.get("candidate_rejected", 0)),
            "births": accepted,
            "retirements": int(counts.get("expert_retired", 0)),
            "reactivations": int(counts.get("expert_reactivated", 0)),
            "purges": int(counts.get("expert_purged", 0)),
            "lifecycle_exercised": bool(accepted > 0 or
                                       counts.get("expert_reactivated", 0) > 0 or
                                       counts.get("expert_retired", 0) > 0),
            "lifecycle_not_exercised_reason": (
                None if accepted > 0 else
                "lifecycle was ON and candidates were causally trained/qualified, but no candidate passed frozen acceptance; retirement was preregistered disabled"),
            "final_topology": session.model.learner.topology_manifest(),
            "calibration_threshold": controller.calibration_threshold,
            "extra_compute": controller.extra_compute,
            "last_decision": controller.last_decision,
            "ledger_file": "lifecycle.jsonl",
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return session, summary


def _window_detection(session, start, end):
    return binary_detection_metrics(
        session.predictions["probability"][start:end],
        session.predictions["labels"][start:end])


def _baseline_report(score, target):
    truth = (np.asarray(target) > 0).astype(np.int64).reshape(-1)
    result = average_precision_report(truth, np.asarray(score).reshape(-1))
    result["prevalence"] = float(truth.mean()) if truth.size else None
    return result


def _aggregate_law(session, phases, law):
    indices = []
    for p in phases:
        if p.get("response_law") == law:
            indices.extend(range(p["start"], p["end"]))
    idx = np.asarray(indices, dtype=np.int64)
    return binary_detection_metrics(session.predictions["probability"][idx],
                                    session.predictions["labels"][idx])


def _comparison(sessions, summaries, bundle, phases, update_every):
    by_phase = {p["name"]: p for p in phases}
    switches = []
    for name in FIRST_NAMES + RECURRENCE_NAMES:
        p = by_phase[name]
        start, end = p["start"], min(p["start"] + 100, p["end"])
        arms = {arm: _window_detection(session, start, end)
                for arm, session in sessions.items()}
        c_ap, d_ap = arms["C"]["ap"], arms["D"]["ap"]
        delta = None if c_ap is None or d_ap is None else float(d_ap - c_ap)
        switches.append({"phase": name, "response_law": p.get("response_law"),
                         "intervals": [start, end], "arms": arms,
                         "D_minus_C_ap": delta})

    first = [x for x in switches if x["phase"] in FIRST_NAMES]
    recur = [x for x in switches if x["phase"] in RECURRENCE_NAMES]
    recur_valid = [x["D_minus_C_ap"] for x in recur if x["D_minus_C_ap"] is not None]
    first_valid = [x["D_minus_C_ap"] for x in first if x["D_minus_C_ap"] is not None]

    # Threshold-FPR development check on the union of recurrence first-100 rows.
    recur_idx = []
    for item in recur:
        recur_idx.extend(range(item["intervals"][0], item["intervals"][1]))
    recur_idx = np.asarray(recur_idx, dtype=np.int64)
    recurrence_threshold = {
        arm: binary_detection_metrics(session.predictions["probability"][recur_idx],
                                      session.predictions["labels"][recur_idx])
        for arm, session in sessions.items()}
    c_fpr, d_fpr = recurrence_threshold["C"]["fpr"], recurrence_threshold["D"]["fpr"]
    fpr_delta = None if c_fpr is None or d_fpr is None else float(d_fpr - c_fpr)

    raw = np.asarray(bundle["arrays"]["raw_labels"])
    ratio = np.asarray(bundle["arrays"]["overload_ratio"])
    target = raw[1:4981]
    persistence = (raw[:4980] > 0).astype(np.float64)
    pressure = ratio[:4980].max(-1)
    baseline_full = {
        "persistence_current_fault": _baseline_report(persistence, target),
        "current_pressure": _baseline_report(pressure, target),
    }
    baseline_switches = []
    for item in switches:
        start, end = item["intervals"]
        baseline_switches.append({
            "phase": item["phase"],
            "persistence": _baseline_report(persistence[start:end], target[start:end]),
            "current_pressure": _baseline_report(pressure[start:end], target[start:end])})

    full = {}
    worst = {}
    onset = {}
    diagnosis = {}
    laws = {}
    for arm, session in sessions.items():
        full[arm] = binary_detection_metrics(session.predictions["probability"],
                                             session.predictions["labels"])
        onset[arm] = temporal_onset_metrics(session.predictions["probability"],
                                            session.predictions["raw_labels"], 1)
        diagnosis[arm] = positive_resource_macro_f1(
            session.predictions["class_probability"], session.predictions["labels"])
        laws[arm] = {law: _aggregate_law(session, phases, law)
                     for law in ("R1", "R2", "R3")}
        valid_laws = [(law, report["ap"]) for law, report in laws[arm].items()
                      if report["ap"] is not None]
        worst[arm] = (None if not valid_laws else
                      {"response_law": min(valid_laws, key=lambda x: x[1])[0],
                       "ap": float(min(valid_laws, key=lambda x: x[1])[1])})

    recurrence_mean = float(np.mean(recur_valid)) if recur_valid else None
    recurrence_positive = int(sum(x > 0 for x in recur_valid))
    enough = len(recur_valid) == len(RECURRENCE_NAMES)
    signal = (None if not enough or fpr_delta is None else bool(
        recurrence_mean >= 0.03 and recurrence_positive >= 4 and fpr_delta <= 0.01))
    d_lifecycle = summaries["D"]["lifecycle"]
    return {
        "protocol": "024", "round": "next_round_v1",
        "development_only": True, "confirmation_run": False,
        "target": NEXT_TARGET_MODE, "model_seed": 1, "replay_seed": 700,
        "selected_update_every": int(update_every),
        "switch_first100": switches,
        "first_exposure_mean_D_minus_C": (float(np.mean(first_valid)) if first_valid else None),
        "recurrence": {
            "valid_switches": len(recur_valid), "total_switches": len(RECURRENCE_NAMES),
            "coverage": len(recur_valid) / float(len(RECURRENCE_NAMES)),
            "mean_D_minus_C_ap": recurrence_mean,
            "positive_D_minus_C_switches": recurrence_positive,
            "threshold_metrics": recurrence_threshold,
            "normal_fpr_D_minus_C": fpr_delta,
        },
        "full_detection": full,
        "raw_same_host_onset": onset,
        "future_positive_resource_macro_f1": diagnosis,
        "per_response_law_detection": laws,
        "worst_response_law": worst,
        "baselines": {"full": baseline_full, "switch_first100": baseline_switches},
        "lifecycle": d_lifecycle,
        "development_signal": signal,
        "development_signal_reason": (
            "all six recurrence windows evaluable; frozen >=0.03, >=4/6 positive, FPR delta<=0.01 rule applied"
            if enough and fpr_delta is not None else
            "insufficient evaluable recurrence coverage or threshold-FPR denominator; no hard pass/fail"),
    }


def run(stream_dir, budget_path, lifecycle_path, out_root, comparison_path):
    stream_dir, out_root = Path(stream_dir), Path(out_root)
    budget_result = json.loads(Path(budget_path).read_text(encoding="utf8"))
    update_every = int(budget_result["selected_update_every"])
    lifecycle_all = json.loads(Path(lifecycle_path).read_text(encoding="utf8"))
    lifecycle_config = {k: lifecycle_all[k] for k in LIFECYCLE_KEYS}
    bundle = s4.build_replay(stream_dir)
    if bundle["steps"] != 4980:
        raise ValueError("formal pilot requires registered 4980-step stream")
    phases = _phase_defs(bundle["manifest"])
    sessions, summaries = {}, {}
    for arm in ("A", "C", "D"):
        # Fresh replay and anchor objects for each arm; model seed and raw bytes
        # stay identical.
        arm_bundle = s4.build_replay(stream_dir)
        anchor = s4.load_anchor_pool(arm_bundle["replay"].time_scale)
        sessions[arm], summaries[arm] = _run_arm(
            arm, stream_dir, arm_bundle, anchor, update_every,
            lifecycle_config, out_root)
    result = _comparison(sessions, summaries, bundle, phases, update_every)
    comparison_path = Path(comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    run(args.stream, args.budget, args.lifecycle, args.out_root, args.comparison)


if __name__ == "__main__":
    main()
