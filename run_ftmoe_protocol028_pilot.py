"""Protocol-028: exactly C_fixed5 vs D_memory_protected on seed700/model1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import run_ftmoe_protocol027_pilot as p27
import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_eval import binary_detection_metrics
from ftmoe_protocol025_session import Protocol025FixedSession
from ftmoe_protocol027_data import EXPECTED_STREAM_SHA, REVISION_ID, verify_frozen
from ftmoe_protocol028_memory_protected import Protocol028DynamicSession


COMPARATORS = ("C_fixed5", "D_memory_protected")
RECURRENCE = tuple(p27.RECURRENCE)
LATE_RECURRENCE = (
    "S4_rec1", "S2_rec2", "S6_rec1", "S1_rec2", "S5_rec1", "S3_rec2"
)


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def build_guard(bundle):
    guard = p27.build_guard(bundle)
    guard["meta"] = dict(guard["meta"])
    guard["meta"].update({
        "protocol": "028",
        "inherited_from": "Protocol027 normal-only historical guard",
        "scientific_change": "none",
    })
    return guard


def _reuse_decisions(events, phase_defs):
    pending = {}
    rows = []
    for event in events:
        kind = event.get("kind")
        if kind == "reuse_candidate_selected":
            pending[str(event.get("expert_id"))] = event
            continue
        if kind not in ("specialist_reactivated", "reuse_candidate_rejected"):
            continue
        key = str(event.get("expert_id"))
        selected = pending.pop(key, None)
        guard = event.get("guard") or {}
        rows.append({
            "phase": p27.event_phase(event.get("cursor", 0), phase_defs),
            "cursor": int(event.get("cursor", 0)),
            "matured_count": int(event.get("matured_count", 0)),
            "expert_id": key,
            "accepted": kind == "specialist_reactivated",
            "relative_improvement": event.get("relative_improvement"),
            "required_relative_improvement": event.get("required_relative_improvement"),
            "live_mean_loss": event.get("live_mean_loss"),
            "candidate_mean_loss": event.get("candidate_mean_loss"),
            "live_normal_probability": event.get("live_normal_probability"),
            "candidate_normal_probability": event.get("candidate_normal_probability"),
            "normal_safety_ok": event.get("normal_safety_ok"),
            "validation_live_fpr_0p5": event.get("validation_live_fpr_0p5"),
            "validation_candidate_fpr_0p5": event.get("validation_candidate_fpr_0p5"),
            "validation_fpr_delta_0p5": event.get("validation_fpr_delta_0p5"),
            "validation_fpr_ok": event.get("validation_fpr_ok"),
            "guard_available": guard.get("available"),
            "guard_live_loss": guard.get("live_loss"),
            "guard_candidate_loss": guard.get("candidate_loss"),
            "guard_relative_loss_increase": guard.get("relative_loss_increase"),
            "guard_live_fpr_0p5": guard.get("live_fpr_0p5"),
            "guard_candidate_fpr_0p5": guard.get("candidate_fpr_0p5"),
            "guard_fpr_delta_0p5": guard.get("fpr_delta_0p5"),
            "guard_loss_ok": event.get("guard_loss_ok"),
            "guard_fpr_ok": event.get("guard_fpr_ok"),
            "reject_reasons": list(event.get("reject_reasons") or []),
            "selection_similarity": None if selected is None else selected.get("similarity"),
            "selection_required_similarity": None if selected is None else selected.get("required_similarity"),
        })
    return rows


def _late_recurrence_reuse(events, decisions, phase_defs):
    by_phase = {name: {
        "matched": 0,
        "reuse_decisions": 0,
        "reuse_accepted": 0,
        "reuse_rejected": 0,
        "reuse_skipped_no_memory": 0,
        "reuse_skipped_no_match": 0,
        "capacity_preserve_birth_skips": 0,
        "expert_ids": [],
    } for name in LATE_RECURRENCE}
    for event in events:
        phase = p27.event_phase(event.get("cursor", 0), phase_defs)
        if phase not in by_phase:
            continue
        kind = event.get("kind")
        row = by_phase[phase]
        if kind == "reuse_candidate_selected":
            row["matched"] += 1
            if event.get("expert_id") is not None:
                row["expert_ids"].append(str(event["expert_id"]))
        elif kind == "reuse_skipped_no_memory":
            row["reuse_skipped_no_memory"] += 1
        elif kind in ("reuse_skipped_no_match", "reuse_skipped_insufficient_z"):
            row["reuse_skipped_no_match"] += 1
        elif kind == "birth_skipped_capacity_preserve_memory":
            row["capacity_preserve_birth_skips"] += 1
    for decision in decisions:
        phase = decision["phase"]
        if phase not in by_phase:
            continue
        row = by_phase[phase]
        row["reuse_decisions"] += 1
        row["reuse_accepted" if decision["accepted"] else "reuse_rejected"] += 1
    for row in by_phase.values():
        row["expert_ids"] = sorted(set(row["expert_ids"]), key=int)
    return by_phase


def run_arm(name, stream, arm_dir, method_registration, registration, run_id):
    if name not in COMPARATORS:
        raise ValueError(name)
    out = Path(arm_dir)
    out.mkdir(parents=True, exist_ok=False)
    stream = Path(stream)
    verify_frozen(stream)

    method = json.loads(Path(method_registration).read_text(encoding="utf8"))
    reg = json.loads(Path(registration).read_text(encoding="utf8"))
    if reg.get("protocol") != "028":
        raise AssertionError("Protocol028 registration mismatch")
    if reg.get("seeds") != {"replay": 700, "model": 1}:
        raise AssertionError("Protocol028 seed mismatch")
    if reg.get("comparators") != list(COMPARATORS):
        raise AssertionError("Protocol028 comparator mismatch")

    bundle = s4.build_replay(stream)
    bundle["stream_dir"] = str(stream)
    stream_sha = p27.sha(stream / "stream.npz")
    if stream_sha != EXPECTED_STREAM_SHA:
        raise AssertionError("stream hash mismatch")
    if bundle["manifest"].get("stream_sha256") != EXPECTED_STREAM_SHA:
        raise AssertionError("manifest stream hash mismatch")

    phase_defs = p27.phases(bundle["manifest"])
    guard = build_guard(bundle)
    write_json(out / "guard_manifest.json", guard["meta"])
    runtime_registration = {
        "protocol": "028",
        "comparator": name,
        "source_method_registration": method,
        "protocol028_registration": reg,
        "source_protocol027_data_revision": REVISION_ID,
        "stream_sha256": stream_sha,
        "replay_seed": 700,
        "model_seed": 1,
        "confirmation_run": False,
        "test_run": False,
    }
    common = dict(
        seed=1,
        replay_bundle=bundle,
        budget=p27.budget(),
        out_dir=out,
        run_id=f"protocol028_{run_id}_{name}",
        stream_dir=stream,
        phase_defs=phase_defs,
        stream_sha=stream_sha,
        registration=runtime_registration,
        learning_rate=1e-4,
    )
    session = (
        Protocol028DynamicSession(guard_anchor=guard, **common)
        if name == "D_memory_protected"
        else Protocol025FixedSession("C_fixed5", **common)
    )

    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    peak_param = p27.param_bytes(session)
    peak_opt = p27.opt_bytes(session)
    peak_live = 5
    peak_resident = 5

    for i in range(session.steps):
        session.step()
        if i % 64 == 0 or i + 1 == session.steps:
            peak_param = max(peak_param, p27.param_bytes(session))
            peak_opt = max(peak_opt, p27.opt_bytes(session))
        if name == "D_memory_protected":
            bank = session.model.learner
            peak_live = max(peak_live, len(bank.ids))
            peak_resident = max(peak_resident, bank.resident_count())
            if bank.resident_count() > bank.max_experts:
                raise AssertionError("Protocol028 resident capacity exceeded")

    if name == "D_memory_protected":
        session.lifecycle_controller.mark_stream_end()
    session.finish()
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    session.save()

    probability = session.predictions["probability"]
    class_probability = session.predictions["class_probability"]
    labels = session.predictions["labels"]
    if (labels < 0).any() or (session.predictions["raw_labels"] < 0).any():
        raise AssertionError("unsettled Protocol028 outputs")

    cost = {
        "wall_seconds": float(wall),
        "cpu_seconds": float(cpu),
        "max_rss_bytes": p27.rss_bytes(),
        "p95_inference_seconds": float(
            np.percentile(session.predictions["prediction_seconds"], 95)
        ),
        "online_update_opportunities_completed": int(session.updates),
        "online_sample_draws": p27.samples(session),
        "final_resident_parameter_bytes": p27.param_bytes(session),
        "peak_resident_parameter_bytes": int(peak_param),
        "final_optimizer_state_bytes": p27.opt_bytes(session),
        "peak_optimizer_state_bytes": int(peak_opt),
        "peak_live_expert_count": int(peak_live),
        "peak_resident_expert_count": int(peak_resident),
        "timing_scope": {
            "included": "step loop, online updates, lifecycle diagnostics/validation/guards executed in-loop, and finish()",
            "excluded": "session.save(), parent comparison/finalization, artifact upload and git operations",
        },
    }
    summary = {
        "protocol": "028",
        "comparator": name,
        "run_id": str(run_id),
        "completed": True,
        "development_only": True,
        "confirmation_run": False,
        "test_run": False,
        "stream_sha256": stream_sha,
        "data_revision": REVISION_ID,
        "manifest": session.comparator_manifest(),
        "full": p27.metrics(probability, class_probability, labels),
        "phases": p27.phase_metrics(probability, class_probability, labels, phase_defs),
        "cost": cost,
    }

    if name == "D_memory_protected":
        ctrl = session.lifecycle_controller
        events = list(ctrl.events)
        records = list(ctrl.candidate_records.values())
        event_counts = Counter(e["kind"] for e in events)
        by_phase = defaultdict(Counter)
        rejects = Counter()
        for record in records:
            for reason in (record.get("decision") or {}).get("reject_reasons", []):
                rejects[str(reason)] += 1
        for event in events:
            by_phase[p27.event_phase(event.get("cursor", 0), phase_defs)][event["kind"]] += 1

        accepted = sum(r.get("accepted") is True for r in records)
        rejected = sum(r.get("accepted") is False for r in records)
        pending = sum(r.get("accepted") is None for r in records)
        decisions = _reuse_decisions(events, phase_defs)
        late = _late_recurrence_reuse(events, decisions, phase_defs)
        preserve_events = [
            {
                "phase": p27.event_phase(e.get("cursor", 0), phase_defs),
                "cursor": int(e.get("cursor", 0)),
                "matured_count": int(e.get("matured_count", 0)),
                "expert_id": e.get("expert_id"),
                "preserved_expert_ids": e.get("preserved_expert_ids", []),
                "resident_count": e.get("resident_count"),
                "max_experts": e.get("max_experts"),
            }
            for e in events
            if e.get("kind") == "birth_skipped_capacity_preserve_memory"
        ]

        due = ctrl.due_conservation()
        if not all(row.get("conserved") for row in due.values()):
            raise AssertionError("Protocol028 lifecycle due accounting is not conserved")
        if ctrl.purges != 0 or event_counts.get("specialist_purged", 0) != 0:
            raise AssertionError("Protocol028 memory-protected policy performed a purge")

        extra = dict(ctrl.extra_compute)
        candidate_overhead = float(
            extra.get("shadow_train_seconds", 0.0)
            + extra.get("shadow_validation_seconds", 0.0)
            + extra.get("guard_seconds", 0.0)
        )
        reuse_overhead = float(
            extra.get("reuse_validation_seconds", 0.0)
            + extra.get("reuse_guard_seconds", 0.0)
        )
        cost.update({
            "shadow_training_steps": int(extra.get("shadow_train_steps", 0)),
            "reuse_validation_forwards": int(extra.get("reuse_validation_forwards", 0)),
            "guard_forwards": int(extra.get("guard_forwards", 0)),
            "reuse_guard_forwards": int(extra.get("reuse_guard_forwards", 0)),
            "candidate_measured_overhead_seconds": candidate_overhead,
            "reuse_measured_overhead_seconds": reuse_overhead,
            "candidate_plus_reuse_measured_overhead_seconds": candidate_overhead + reuse_overhead,
        })

        lifecycle = {
            "candidate_created": len(records),
            "candidate_accepted": accepted,
            "candidate_rejected": rejected,
            "candidate_pending": pending,
            "candidate_conservation": len(records) == accepted + rejected + pending,
            "stream_end_censored_ids": [
                r["candidate_id"] for r in records if r.get("stream_end_censored")
            ],
            "births": int(event_counts.get("candidate_accepted", 0)),
            "retirements": int(ctrl.retirements),
            "reactivations": int(ctrl.reactivations),
            "purges": int(ctrl.purges),
            "purged_bytes": int(ctrl.purged_bytes),
            "capacity_preserve_skips": int(ctrl.capacity_preserve_skips),
            "capacity_preserve_events": preserve_events,
            "would_have_purged_expert_ids_preserved": sorted(
                {str(e["expert_id"]) for e in preserve_events if e.get("expert_id") is not None},
                key=int,
            ),
            "event_counts": dict(event_counts),
            "event_counts_by_phase": {k: dict(v) for k, v in by_phase.items()},
            "candidate_reject_reasons": dict(rejects),
            "accepted_birth_ids": list(ctrl.accepted_ids),
            "resident_memory_ids": sorted(ctrl.specialist_memory.keys(), key=int),
            "opportunity_accounting": due,
            "final_topology": session.model.learner.topology_manifest(),
            "extra_compute": extra,
            "reuse_decisions": decisions,
            "late_recurrence_reuse": late,
            "reuse_observed": bool(ctrl.reactivations > 0),
            "reuse_benefit_claim_allowed": False,
        }
        summary["lifecycle"] = lifecycle
        write_jsonl(out / "lifecycle.jsonl", events)
        write_json(out / "candidate_records.json", records)
        write_json(out / "opportunity_accounting.json", due)
        write_json(out / "reuse_decisions.json", decisions)
        write_json(out / "late_recurrence_reuse.json", late)
        write_json(out / "lifecycle_summary.json", lifecycle)

    write_json(out / "summary.json", summary)
    return summary


def load_npz(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def compare(root, stream, run_id, registration):
    root = Path(root)
    summaries = {
        name: json.loads((root / name / "summary.json").read_text(encoding="utf8"))
        for name in COMPARATORS
    }
    predictions = {
        name: load_npz(root / name / "predictions.npz")
        for name in COMPARATORS
    }
    fixed = predictions["C_fixed5"]
    dynamic = predictions["D_memory_protected"]
    if not np.array_equal(fixed["labels"], dynamic["labels"]):
        raise AssertionError("comparator labels differ")
    if summaries["C_fixed5"]["stream_sha256"] != summaries["D_memory_protected"]["stream_sha256"]:
        raise AssertionError("comparators did not bind the same frozen stream")

    labels = fixed["labels"]
    phase_map = {
        p["name"]: p for p in p27.phases(
            json.loads((Path(stream) / "manifest.json").read_text(encoding="utf8"))
        )
    }
    rows = []
    deltas = []
    positive = 0
    pooled = []
    valid_all = True
    for name in RECURRENCE:
        phase = phase_map[name]
        start = phase["start"]
        end = min(phase["end"], start + 100)
        pooled.extend(range(start, end))
        yy = labels[start:end]
        cm = binary_detection_metrics(fixed["probability"][start:end], yy, .5)
        dm = binary_detection_metrics(dynamic["probability"][start:end], yy, .5)
        valid = cm["ap"] is not None and dm["ap"] is not None
        delta = float(dm["ap"] - cm["ap"]) if valid else None
        valid_all = valid_all and valid
        if valid:
            deltas.append(delta)
            positive += int(delta > 0)
        rows.append({
            "phase": name,
            "service": phase["regime"],
            "intervals": [start, end],
            "positive_rows": int((yy > 0).sum()),
            "negative_rows": int((yy == 0).sum()),
            "C_fixed5": cm,
            "D_memory_protected": dm,
            "D_minus_C_fixed5_ap": delta,
            "valid": bool(valid),
            "invalid_reason": None if valid else "AP undefined for at least one comparator",
        })

    idx = np.asarray(pooled, dtype=np.int64)
    cm = binary_detection_metrics(fixed["probability"][idx], labels[idx], .5)
    dm = binary_detection_metrics(dynamic["probability"][idx], labels[idx], .5)
    fpr_delta = (
        None if cm["fpr"] is None or dm["fpr"] is None
        else float(dm["fpr"] - cm["fpr"])
    )
    mean = float(np.mean(deltas)) if valid_all and len(deltas) == 9 else None
    signal = bool(
        mean is not None and mean >= .03
        and positive >= 6
        and fpr_delta is not None and fpr_delta <= .01
    )
    c_ap = summaries["C_fixed5"]["full"]["detection"]["ap"]
    d_ap = summaries["D_memory_protected"]["full"]["detection"]["ap"]
    full_delta = None if c_ap is None or d_ap is None else float(d_ap - c_ap)

    return {
        "protocol": "028",
        "run_id": str(run_id),
        "development_only": True,
        "confirmation_run": False,
        "test_run": False,
        "stream_sha256": EXPECTED_STREAM_SHA,
        "data_revision": REVISION_ID,
        "source_frozen_artifact": registration["source_data"],
        "same_frozen_data_verified": True,
        "recurrence_first100": rows,
        "primary": {
            "metric": "equal-weight mean D_memory_protected-C_fixed5 AP across all nine recurrence first100 windows",
            "valid_windows": sum(r["valid"] for r in rows),
            "required_valid_windows": 9,
            "equal_weight_mean_D_minus_C_fixed5_ap": mean,
            "positive_windows": positive,
            "pooled_normal_fpr_C_fixed5": cm["fpr"],
            "pooled_normal_fpr_D_memory_protected": dm["fpr"],
            "pooled_normal_fpr_delta_D_minus_C": fpr_delta,
            "numeric_lead_on_this_trajectory": None if mean is None else bool(mean > 0),
            "development_signal": signal,
            "development_reference": {
                "mean_delta_min": .03,
                "positive_windows_min": 6,
                "pooled_normal_fpr_delta_max": .01,
            },
        },
        "full_stream": {
            "C_fixed5": summaries["C_fixed5"]["full"],
            "D_memory_protected": summaries["D_memory_protected"]["full"],
            "D_minus_C_fixed5_ap": full_delta,
            "D_ap_leads": None if full_delta is None else bool(full_delta > 0),
        },
        "phase_metrics": {name: summaries[name]["phases"] for name in COMPARATORS},
        "lifecycle": summaries["D_memory_protected"].get("lifecycle"),
        "cost_profile": {name: summaries[name]["cost"] for name in COMPARATORS},
        "interpretation_limits": {
            "statistical_confirmation": False,
            "equal_total_cost": False,
            "D_extra_background_compute_allowed": True,
            "single_replay_seed": 700,
            "single_model_seed": 1,
            "memory_protection_tradeoff": "full memory can block learning a new service",
        },
    }


def parent(args):
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "status.json").exists() or any((root / n).exists() for n in COMPARATORS):
        raise FileExistsError("existing Protocol028 model run")

    verify_frozen(args.stream)
    reg = json.loads(Path(args.registration).read_text(encoding="utf8"))
    if reg.get("protocol") != "028":
        raise RuntimeError("wrong Protocol028 registration")
    source = json.loads((root / "source_frozen_artifact.json").read_text(encoding="utf8"))
    expected_source = reg["source_data"]
    for key, expected in (
        ("run_id", expected_source["frozen_run_id"]),
        ("artifact_id", expected_source["frozen_artifact_id"]),
        ("artifact_name", expected_source["frozen_artifact_name"]),
        ("artifact_digest", expected_source["frozen_artifact_digest"]),
    ):
        if source.get(key) != expected:
            raise RuntimeError(f"source frozen artifact mismatch for {key}")
    if source.get("verified") is not True:
        raise RuntimeError("source frozen artifact was not verified")

    lock = json.loads(Path(args.data_lock).read_text(encoding="utf8"))
    eligibility = json.loads(Path(args.eligibility).read_text(encoding="utf8"))
    if not lock.get("locked") or not eligibility.get("protocol027_data_eligible"):
        raise RuntimeError("frozen source data failed inherited Protocol027 eligibility")
    for item in (lock, eligibility):
        if item.get("data_revision") != REVISION_ID:
            raise RuntimeError("data revision mismatch")
        if item.get("stream_sha256") != EXPECTED_STREAM_SHA:
            raise RuntimeError("data stream mismatch")

    bundle = s4.build_replay(Path(args.stream))
    guard = build_guard(bundle)
    write_json(root / "guard_manifest.json", guard["meta"])
    write_json(root / "normal_guard_audit.json", {
        "protocol": "028",
        "valid": guard["meta"]["normal_rows"] > 0,
        **guard["meta"],
        "birth_and_reuse_guard_implementation": "inherited Protocol027 normal_detection_guard_report",
        "positive_rows_zero_policy": "allowed",
    })
    del bundle, guard

    completed = []
    failed = []
    failure_details = {}
    for name in COMPARATORS:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-arm", name,
            "--stream", args.stream,
            "--arm-dir", str(root / name),
            "--method-registration", args.method_registration,
            "--registration", args.registration,
            "--run-id", str(args.run_id),
        ]
        stdout = root / f"{name}.stdout.log"
        stderr = root / f"{name}.stderr.log"
        with stdout.open("w") as out, stderr.open("w") as err:
            proc = subprocess.run(cmd, text=True, stdout=out, stderr=err)
        if (
            proc.returncode == 0
            and (root / name / "summary.json").is_file()
            and (root / name / "predictions.npz").is_file()
        ):
            completed.append(name)
        else:
            failed.append(name)
            failure_details[name] = {
                "returncode": proc.returncode,
                "stderr_tail": stderr.read_text(errors="replace")[-8000:],
            }
            write_json(root / f"{name}.failure.json", failure_details[name])

    comparison = (
        compare(root, args.stream, args.run_id, reg)
        if completed == list(COMPARATORS) and not failed
        else None
    )
    if comparison is not None:
        write_json(root / "comparison.json", comparison)
        write_json(root / "cost_profile.json", comparison["cost_profile"])

    status = {
        "protocol": "028",
        "data_revision": REVISION_ID,
        "source_frozen_artifact": source,
        "run_id": str(args.run_id),
        "completed": comparison is not None,
        "data_restored": True,
        "same_frozen_data_verified": True,
        "eligibility_verified": True,
        "normal_guard_valid": True,
        "completed_comparators": completed,
        "failed_comparators": failed,
        "failure_details": failure_details,
        "development_signal": None if comparison is None else bool(
            comparison["primary"]["development_signal"]
        ),
        "confirmation_run": False,
        "test_run": False,
        "automatic_followups_started": [],
        "blocker": None if comparison is not None else "one_or_more_registered_arms_failed",
    }
    if comparison is not None:
        life = comparison.get("lifecycle") or {}
        status.update({
            "primary_mean_D_minus_C_fixed5_ap":
                comparison["primary"]["equal_weight_mean_D_minus_C_fixed5_ap"],
            "positive_recurrence_windows": comparison["primary"]["positive_windows"],
            "full_stream_D_minus_C_fixed5_ap":
                comparison["full_stream"]["D_minus_C_fixed5_ap"],
            "D_reactivations": int(life.get("reactivations", 0)),
            "D_purges": int(life.get("purges", 0)),
            "D_capacity_preserve_skips": int(life.get("capacity_preserve_skips", 0)),
        })
    write_json(root / "status.json", status)
    print(json.dumps({
        "status": status,
        "primary": None if comparison is None else comparison["primary"],
    }, indent=2, allow_nan=False))


def child(args):
    try:
        run_arm(
            args.child_arm, args.stream, args.arm_dir,
            args.method_registration, args.registration, args.run_id
        )
    except Exception:
        Path(args.arm_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.arm_dir) / "failure.json", {
            "protocol": "028",
            "comparator": args.child_arm,
            "completed": False,
            "traceback": traceback.format_exc(),
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True)
    parser.add_argument("--out-root")
    parser.add_argument("--arm-dir")
    parser.add_argument("--method-registration", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--data-lock")
    parser.add_argument("--eligibility")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child-arm", choices=COMPARATORS)
    args = parser.parse_args()

    if args.child_arm:
        if not args.arm_dir:
            parser.error("--arm-dir required")
        child(args)
    else:
        if not all((args.out_root, args.data_lock, args.eligibility)):
            parser.error("--out-root/--data-lock/--eligibility required")
        parent(args)


if __name__ == "__main__":
    main()
