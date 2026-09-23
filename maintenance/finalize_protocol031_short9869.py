"""User-requested Protocol-031 short-horizon amendment: freeze prefix row target 9869.

This does not mutate the original revision002 registration or scientific source.
It resumes the exact seed700 simulator checkpoint when needed, stops once at least
9869 rows are available, and freezes the deterministic 0:9869 prefix as an
explicit post-start amendment for a shorter C/D development comparison.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

# When executed as `python maintenance/<script>.py`, Python places maintenance/
# rather than the repository root on sys.path. Add only the repo root; this is
# an import-path repair and does not alter simulator/data/model behavior.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_ftmoe_protocol031_stream as P
import maintenance.run_protocol031_segment as SEG

SHORT_ROWS = 9869
SHORT_STEPS = 9868
SHORT_RECURRENCE = ("U_rec1", "V_rec1")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def short_registration(stream_sha=None):
    full = P.registration()
    reg = copy.deepcopy(full)
    original_rows = int(reg["scored_intervals"]) + int(reg["guard_intervals"])
    reg["status"] = "user_amended_short_horizon"
    reg["registered_before_generation"] = False
    reg["scored_intervals"] = SHORT_STEPS
    reg["guard_intervals"] = 1
    timeline = []
    for item in reg["timeline"]:
        start, end = int(item["start"]), int(item["end"])
        if start >= SHORT_STEPS:
            break
        row = copy.deepcopy(item)
        row["end"] = min(end, SHORT_STEPS)
        row["length"] = int(row["end"]) - start
        timeline.append(row)
    reg["timeline"] = timeline
    reg["evaluation"]["primary_windows"] = list(SHORT_RECURRENCE)
    reg["evaluation"]["primary_metric"] = (
        "equal-weight mean AP(D_nonblocking_reuse)-AP(C_fixed5) across the two "
        "U/V recurrence128 windows available before the user-amended 9869-row horizon"
    )
    ref = reg["evaluation"]["development_reference"]
    ref["required_valid_windows"] = 2
    ref["positive_windows_min"] = 2
    reg["generation"]["expected_stream_sha256"] = stream_sha
    reg["generation"]["frozen_before_model_runs"] = bool(stream_sha)
    reg["generation"]["short_horizon_rows"] = SHORT_ROWS
    reg["generation"]["prefix_of_original_registered_stream"] = True
    reg["amendment"] = {
        "kind": "user_requested_short_horizon",
        "requested_row_target": SHORT_ROWS,
        "requested_scored_intervals": SHORT_STEPS,
        "original_row_target": original_rows,
        "original_scored_intervals": int(full["scored_intervals"]),
        "requested_after_generation_started": True,
        "changes_simulator_physics_before_target": False,
        "changes_seed": False,
        "changes_model_hyperparameters": False,
        "available_primary_windows": list(SHORT_RECURRENCE),
        "not_equivalent_to_original_six_window_preregistered_primary": True,
        "interpretation": "development-only shortened-horizon comparison; do not present as the original six-window preregistered result",
    }
    return reg


def ensure_prefix(data_root):
    root = Path(data_root)
    _, manifest_path = P._checkpoint_paths(root)
    if not manifest_path.is_file():
        raise FileNotFoundError("Protocol031 checkpoint manifest missing")
    state = json.loads(manifest_path.read_text(encoding="utf8"))
    next_t = int(state["next_t"])
    if next_t < SHORT_ROWS:
        SEG.collect_segment(root, SHORT_ROWS - next_t)
        state = json.loads(manifest_path.read_text(encoding="utf8"))
        next_t = int(state["next_t"])
    if next_t < SHORT_ROWS:
        raise RuntimeError(f"short horizon not reached: next_t={next_t}, target={SHORT_ROWS}")
    return next_t


def load_prefix(data_root):
    reg = P.registration()
    reg_sha = P.json_sha(P.REGISTRATION_PATH)
    count = int(reg["scored_intervals"]) + int(reg["guard_intervals"])
    arrays = P._allocate(count)
    payload, chunks, _ = SEG.read_checkpoint(Path(data_root), reg_sha, arrays)
    if int(payload["next_t"]) < SHORT_ROWS:
        raise RuntimeError("checkpoint does not contain requested short prefix")
    prefix = {k: np.asarray(v[:SHORT_ROWS]).copy() for k, v in arrays.items()}
    return prefix, payload, chunks


def audit_prefix(reg, phases, arrays, source_next_t):
    raw = np.asarray(arrays["raw_labels"], np.int64)
    ratio = np.asarray(arrays["overload_ratio"], np.float64)
    physical = np.where((ratio > 1.0).any(-1), ratio.argmax(-1) + 1, 0)
    common = P._common_features(arrays["host_features"], arrays["capacities"])
    per_phase, recurrence = {}, {}
    phase_ok = True
    for phase in phases:
        block = P.summary_block(raw, ratio, arrays["audit_host_event_any"], phase["start"], phase["end"])
        block["logical_service"] = phase["logical_service"]
        block["source_service"] = phase["service"]
        per_phase[phase["name"]] = block
        if phase["logical_service"] is not None:
            phase_ok = phase_ok and block["positive_host_steps"] >= 32 and block["negative_host_steps"] >= 32
        if phase["name"] in SHORT_RECURRENCE:
            y = raw[phase["start"]:phase["end"]]
            recurrence[phase["name"]] = {
                "intervals": [int(phase["start"]), int(phase["end"])],
                "positive_host_steps": int((y > 0).sum()),
                "negative_host_steps": int((y == 0).sum()),
                "ap_defined": bool((y > 0).any() and (y == 0).any()),
            }
    rec_ok = set(recurrence) == set(SHORT_RECURRENCE) and all(
        r["positive_host_steps"] >= 32 and r["negative_host_steps"] >= 32 and r["ap_defined"]
        for r in recurrence.values()
    )
    schedules = np.asarray(arrays["schedules"][:SHORT_STEPS], np.float64)
    caps = np.asarray(arrays["capacities"][:SHORT_STEPS], np.float64)
    scheduler_ok = bool(np.isfinite(schedules).all() and np.allclose(schedules.sum(-1), 1.0, atol=1e-5))
    caps_ok = bool(np.isfinite(caps).all() and (caps > 0).all())
    guard_idx = np.arange(0, 299, dtype=np.int64)
    guard_idx = guard_idx[guard_idx % 5 == 0]
    guard_y = raw[guard_idx + 1].reshape(-1)
    guard_normal = int((guard_y == 0).sum())
    event_ids = np.asarray(arrays["audit_event_service_ids"][:SHORT_STEPS])
    event_counts = {}
    for logical, source in P.SOURCE_SERVICE.items():
        mid = int(P.SERVICE_MECHANISM_IDS[source])
        event_counts[logical] = int((event_ids == mid).sum())
    gates = {
        "prefix_rows_exact": int(raw.shape[0]) == SHORT_ROWS,
        "source_checkpoint_reached_target": int(source_next_t) >= SHORT_ROWS,
        "physical_label_recompute_exact": bool(np.array_equal(raw, physical)),
        "timeline_scored_intervals_9868": int(sum(p["length"] for p in phases)) == SHORT_STEPS,
        "events_U_V_W_observed_before_target": all(event_counts[x] > 0 for x in ("U", "V", "W")),
        "all_nonbaseline_phases_min32_positive_and_negative": bool(phase_ok),
        "two_available_recurrence_windows_valid": bool(rec_ok),
        "common_features_shape_finite": bool(common.shape == (SHORT_ROWS, 16, 9) and np.isfinite(common).all()),
        "scheduler_rows_valid": scheduler_ok,
        "capacities_valid": caps_ok,
        "normal_guard_available": guard_normal > 0,
    }
    audit = {
        "protocol": "031", "plan_revision": 2,
        "kind": "post_start_user_amended_short_horizon_model_free_audit",
        "model_results_seen": False,
        "short_horizon_rows": SHORT_ROWS, "short_scored_intervals": SHORT_STEPS,
        "source_checkpoint_next_t": int(source_next_t),
        "prefix_of_original_registered_stream": True,
        "original_six_window_primary_preserved": False,
        "per_phase": per_phase,
        "recurrence_first128_class_coverage": recurrence,
        "service_event_observation_counts": event_counts,
        "normal_guard": {"normal_rows": guard_normal, "positive_rows": int((guard_y > 0).sum())},
        "gates": gates,
        "audit_pass": bool(all(gates.values())),
        "reroll_after_failure_allowed": False,
    }
    return audit, common


def freeze(data_root, bundle_root):
    source_next_t = ensure_prefix(data_root)
    arrays, payload, chunks = load_prefix(data_root)
    reg = short_registration(None)
    phases = P.phase_table(reg)
    audit, common = audit_prefix(reg, phases, arrays, source_next_t)
    bundle = Path(bundle_root)
    data = bundle / "data"
    ev = bundle / "evidence"
    if bundle.exists():
        shutil.rmtree(bundle)
    data.mkdir(parents=True)
    ev.mkdir(parents=True)

    stream = data / "stream.npz"
    np.savez_compressed(stream, **arrays, overload_mask=(arrays["overload_ratio"] > 1.0).astype(np.uint8))
    stream_sha = sha256(stream)
    reg = short_registration(stream_sha)
    reg_path = ev / "scenario_registration_frozen.json"
    write_json(reg_path, reg)

    common_path = data / "common_observable_features.npz"
    np.savez_compressed(common_path, features=common)
    write_json(data / "data_audit.json", audit)

    observed_events = []
    eids = np.asarray(arrays["audit_event_ids"][:SHORT_STEPS])
    sids = np.asarray(arrays["audit_event_service_ids"][:SHORT_STEPS])
    for eid in np.unique(eids[eids >= 0]):
        vals = sids[eids == eid]
        vals = vals[vals >= 0]
        observed_events.append({"event_id": int(eid), "service_mechanism_id": None if vals.size == 0 else int(vals[0])})
    write_json(data / "events.json", observed_events)

    source_law = P.ROOT / "simulator/workload/BitbrainWorkloadProtocol025.py"
    manifest = {
        "protocol": "031", "plan_revision": 2, "scenario_id": P.SCENARIO_ID,
        "data_revision": P.DATA_REVISION, "seed": 700,
        "steps": SHORT_STEPS, "guard_rows": 1,
        "stream_file": "stream.npz", "stream_sha256": stream_sha,
        "registration_sha256": sha256(reg_path), "timeline": phases,
        "logical_to_source_service": P.SOURCE_SERVICE,
        "event_probability": float(reg["generation"]["event_probability"]),
        "model_input_keys": ["host_features", "demands", "schedules", "capacities", "creation_ids", "before_placement"],
        "common_observable_features_file": "common_observable_features.npz",
        "common_observable_feature_order": P.COMMON_FEATURE_ORDER,
        "label_key": "raw_labels",
        "source_law_file": str(source_law.relative_to(P.ROOT)),
        "source_law_sha256": sha256(source_law),
        "prefix_source_checkpoint_next_t": int(source_next_t),
        "prefix_rows_frozen": SHORT_ROWS,
        "post_start_user_amendment": True,
        "audit_pass": bool(audit["audit_pass"]),
    }
    write_json(data / "manifest.json", manifest)

    eligibility = {
        "protocol": "031", "plan_revision": 2,
        "kind": "short9869_model_free_eligibility",
        "protocol031_data_eligible": bool(audit["audit_pass"]),
        "model_results_seen": False,
        "stream_sha256": stream_sha,
        "shape": list(np.asarray(arrays["raw_labels"]).shape),
        "source_checkpoint_next_t": int(source_next_t),
        "available_primary_windows": list(SHORT_RECURRENCE),
        "gates": audit["gates"],
        "post_start_user_amendment": True,
    }
    write_json(ev / "eligibility.json", eligibility)

    lock = {
        "protocol": "031", "plan_revision": 2,
        "locked": bool(audit["audit_pass"]),
        "model_free_eligibility_passed": bool(audit["audit_pass"]),
        "stream_sha256": stream_sha,
        "steps": SHORT_STEPS, "guard_rows": 1,
        "hosts": 16, "replay_seed": 700, "model_seed": 1,
        "generation_budget_used": 1,
        "short_horizon_rows": SHORT_ROWS,
        "available_recurrence_windows": list(SHORT_RECURRENCE),
        "post_start_user_amendment": True,
        "not_original_six_window_preregistered_primary": True,
    }
    write_json(ev / "data_lock.json", lock)

    status = {
        "protocol": "031", "complete": bool(audit["audit_pass"]),
        "next_t": SHORT_ROWS, "scored_target": SHORT_STEPS, "row_target": SHORT_ROWS,
        "source_checkpoint_next_t": int(source_next_t),
        "stream_sha256": stream_sha,
        "audit_pass": bool(audit["audit_pass"]),
        "post_start_user_amendment": True,
    }
    write_json(ev / "generation_status.json", status)
    receipt = Path(data_root) / "pre_generation_receipt.json"
    if receipt.is_file():
        shutil.copy2(receipt, ev / "pre_generation_receipt.json")
    write_json(ev / "short_horizon_amendment.json", reg["amendment"])
    frozen_manifest = {
        "protocol": "031", "stream_sha256": stream_sha,
        "stream_bytes": stream.stat().st_size,
        "manifest_sha256": sha256(data / "manifest.json"),
        "data_audit_sha256": sha256(data / "data_audit.json"),
        "common_features_sha256": sha256(common_path),
        "registration_sha256": sha256(reg_path),
        "frozen_before_model_runs": True,
        "short_horizon_rows": SHORT_ROWS,
    }
    write_json(data / "frozen_data_manifest.json", frozen_manifest)
    if not audit["audit_pass"]:
        raise SystemExit("short9869 model-free audit failed")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--bundle-root", required=True)
    a = ap.parse_args()
    print(json.dumps(freeze(a.data_root, a.bundle_root), indent=2))


if __name__ == "__main__":
    main()
