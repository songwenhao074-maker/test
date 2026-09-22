"""Protocol-027 eligibility for explicitly registered data revision 002.

The original Protocol-025 two-class guard failure stays historical evidence.
This independent audit uses the existing normal-only Protocol-027 guard and
checks the selected recovered dataset before either comparator can run.
Equivalence to the unavailable original complete stream is not asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_ftmoe_protocol025_revision1 import audit as audit_protocol025
import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol025_model import causal_common_features

from ftmoe_protocol027_data import (EXPECTED_STREAM_SHA, EXPECTED_FINAL_CHUNK_SHA,
    REVISION_ID, REVISION_PATH, verify_frozen)
EXPECTED_STEPS = 5520
EXPECTED_ROWS = 5521
EXPECTED_HOSTS = 16
RECURRENCE = ("S1_rec1", "S3_rec1", "S2_rec1", "S4_rec1", "S2_rec2",
              "S6_rec1", "S1_rec2", "S5_rec1", "S3_rec2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf8")


def verify_recovery(data_root, output_root):
    """Record immutable recovery checks before a failed check stops execution."""
    root = Path(data_root)
    def read(name):
        p = root / name
        return json.loads(p.read_text(encoding="utf8")) if p.is_file() else {}
    manifest = read("manifest.json")
    resume = read("resume_manifest.json")
    stream = root / "stream.npz"
    actual_stream = sha256(stream) if stream.is_file() else None
    chunks = manifest.get("chunk_manifest", [])
    last = chunks[-1] if chunks else {}
    chunk = root / "chunks" / last.get("file", "")
    actual_chunk = sha256(chunk) if chunk.is_file() else None
    gates = {
        "resume_next_t_5521": resume.get("next_t") == EXPECTED_ROWS,
        "stream_sha256_matches_registered": actual_stream == EXPECTED_STREAM_SHA == manifest.get("stream_sha256"),
        "final_chunk_interval_5400_5521": (last.get("start"), last.get("end")) == (5400, EXPECTED_ROWS),
        "final_chunk_hash_matches_registered": actual_chunk == EXPECTED_FINAL_CHUNK_SHA == last.get("sha256"),
    }
    result = {
        "protocol": "027", "data_revision": REVISION_ID, "kind": "immutable_recovery_verification",
        "passed": all(gates.values()), "gates": gates,
        "next_t": resume.get("next_t"),
        "stream_sha256": actual_stream, "expected_stream_sha256": EXPECTED_STREAM_SHA,
        "final_chunk_sha256": actual_chunk, "expected_final_chunk_sha256": EXPECTED_FINAL_CHUNK_SHA,
        "model_steps_run": 0,
    }
    write_json(Path(output_root) / "recovery_verification.json", result)
    return result


def _runtime_feature_consistency(data_root):
    bundle = s4.build_replay(data_root)
    stored = np.load(Path(data_root) / "common_observable_features.npz")["features"]
    if stored.shape != (EXPECTED_ROWS, EXPECTED_HOSTS, 9) or not np.isfinite(stored).all():
        return {"passed": False, "reason": "stored_common_feature_shape_or_finite", "shape": list(stored.shape)}
    max_abs = 0.0
    checked = 0
    failing_intervals = []
    channel_max = np.zeros(9, dtype=np.float64)
    early_max = 0.0
    later_max = 0.0
    for left in range(0, EXPECTED_STEPS, 64):
        indices = list(range(left, min(EXPECTED_STEPS, left + 64)))
        x, _sched, _graph, context = s4.window_batch(bundle["replay"], indices)
        got = causal_common_features(
            x, context, bundle["replay"].time_scale, bundle["graph_scale"]
        ).detach().cpu().numpy()
        want = stored[np.asarray(indices)]
        if not np.isfinite(got).all():
            return {"passed": False, "reason": "runtime_common_feature_nonfinite", "first_index": left}
        diff = float(np.max(np.abs(got - want))) if got.size else 0.0
        max_abs = max(max_abs, diff)
        errors = np.abs(got - want)
        channel_max = np.maximum(channel_max, errors.max(axis=(0, 1)))
        row_max = errors.max(axis=(1, 2))
        failing_intervals.extend(int(i) for i, error in zip(indices, row_max) if error > 5e-5)
        for i, error in zip(indices, row_max):
            if i < 4:
                early_max = max(early_max, float(error))
            else:
                later_max = max(later_max, float(error))
        checked += len(indices)
    return {"passed": bool(max_abs <= 5e-5), "checked_intervals": checked,
            "max_abs_difference": max_abs, "tolerance": 5e-5,
            "includes_initial_padding_boundary": True,
            "failing_intervals": failing_intervals,
            "max_abs_difference_by_channel": channel_max.tolist(),
            "first_four_max_abs_difference": early_max,
            "remaining_max_abs_difference": later_max}


def audit(data_root, output_root):
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if (data_root / "frozen_data_manifest.json").is_file():
        verify_frozen(data_root)
    else:
        try:
            audit_protocol025(data_root)
        except SystemExit as exc:
            if int(exc.code or 0) != 2:
                raise

    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf8"))
    source_audit = json.loads((data_root / "data_audit.json").read_text(encoding="utf8"))
    actual_stream_sha = sha256(data_root / "stream.npz")
    failed_source_gates = sorted(k for k, v in source_audit.get("gates", {}).items() if not bool(v))

    with np.load(data_root / "stream.npz") as data:
        raw = np.asarray(data["raw_labels"], dtype=np.int64)
        ratio = np.asarray(data["overload_ratio"], dtype=np.float64)
        physical = np.where((ratio > 1.0).any(-1), ratio.argmax(-1) + 1, 0)
        labels_match = bool(np.array_equal(raw, physical))
        raw_shape = tuple(raw.shape)

    guard_indices = np.arange(0, 299, dtype=np.int64)
    guard_indices = guard_indices[guard_indices % 5 == 0]
    guard_y = raw[guard_indices + 1].reshape(-1)
    guard_normal = int((guard_y == 0).sum())
    guard_positive = int((guard_y > 0).sum())

    chunks = list(manifest.get("chunk_manifest", []))
    last = chunks[-1] if chunks else {}
    last_path = data_root / "chunks" / str(last.get("file", ""))
    actual_final_chunk_sha = sha256(last_path) if last_path.is_file() else None
    final_chunk_ok = bool(last.get("start") == 5400 and last.get("end") == 5521 and
                          last.get("sha256") == EXPECTED_FINAL_CHUNK_SHA and
                          actual_final_chunk_sha == EXPECTED_FINAL_CHUNK_SHA)

    feature_check = _runtime_feature_consistency(data_root)
    recurrence = source_audit.get("recurrence_first100_class_coverage", {})
    recurrence_ok = (set(recurrence) == set(RECURRENCE) and
                     all(bool(recurrence[name].get("ap_defined")) for name in RECURRENCE))

    assembly_path = data_root / "assembly_verification.json"
    assembly = json.loads(assembly_path.read_text()) if assembly_path.is_file() else {}
    gates = {
        "revision002_preserved_prefix_verified": assembly.get("passed") is True and assembly.get("data_revision") == REVISION_ID and assembly.get("candidate_stream_sha256") == actual_stream_sha,
        "source_protocol025_audit_remains_false": source_audit.get("audit_pass") is False,
        "source_failure_is_only_old_two_class_guard": failed_source_gates == ["F0_guard_has_positive_and_negative"],
        "stream_sha256_matches_registered": actual_stream_sha == EXPECTED_STREAM_SHA == manifest.get("stream_sha256"),
        "shape_5521x16": raw_shape == (EXPECTED_ROWS, EXPECTED_HOSTS),
        "physical_capacity_labels_recompute": labels_match,
        "final_chunk_hash_matches_registered": final_chunk_ok,
        "all_non_guard_source_integrity_gates_pass": all(
            bool(v) for k, v in source_audit.get("gates", {}).items()
            if k != "F0_guard_has_positive_and_negative"
        ),
        "normal_guard_has_known_normal_rows": guard_normal > 0,
        "nine_recurrence_windows_have_both_classes": recurrence_ok,
        "runtime_9d_features_match_stored_causal_features": bool(feature_check["passed"]),
    }
    eligible = bool(all(gates.values()))
    eligibility = {
        "protocol": "027",
        "kind": "independent_data_eligibility", "data_revision": REVISION_ID,
        "source_protocol": "025",
        "source_protocol025_audit_pass": False,
        "source_failed_gates": failed_source_gates,
        "protocol027_data_eligible": eligible,
        "stream_sha256": actual_stream_sha,
        "expected_stream_sha256": EXPECTED_STREAM_SHA,
        "final_chunk_sha256": actual_final_chunk_sha,
        "expected_final_chunk_sha256": EXPECTED_FINAL_CHUNK_SHA,
        "shape": list(raw_shape),
        "guard": {
            "role": "historical_normal_regression_guard",
            "prediction_indices": guard_indices.tolist(),
            "target": "same-host raw[t+1] inside F0",
            "normal_rows": guard_normal,
            "positive_rows": guard_positive,
            "positive_rows_required": False,
        },
        "feature_consistency": feature_check,
        "gates": gates,
    }
    write_json(output_root / "eligibility.json", eligibility)
    if not eligible:
        write_json(output_root / "data_lock.json", {
            "protocol": "027", "locked": False, "reason": "eligibility_failed",
            "stream_sha256": actual_stream_sha, "gates": gates,
        })
        print(json.dumps(eligibility, indent=2, allow_nan=False))
        raise SystemExit(3)

    lock = {
        "protocol": "027", "locked": True,
        "source_protocol": "025", "source_data_revision": "data_revision_001",
        "data_revision": REVISION_ID, "data_revision_registration_sha256": sha256(REVISION_PATH),
        "stream_sha256": actual_stream_sha,
        "final_chunk_sha256": actual_final_chunk_sha,
        "steps": EXPECTED_STEPS, "guard_rows": 1, "hosts": EXPECTED_HOSTS,
        "replay_seed": 700, "model_seed": 1,
        "confirmation_seeds_used": [], "test_seeds_used": [],
        "scientific_data_changed": True,
        "equivalence_to_original_complete_stream": "not_established",
        "common_features_sha256": sha256(data_root / "common_observable_features.npz"),
    }
    write_json(output_root / "data_lock.json", lock)
    print(json.dumps({"eligible": True, "stream_sha256": actual_stream_sha,
                      "guard_normal_rows": guard_normal,
                      "feature_max_abs_difference": feature_check["max_abs_difference"]}, indent=2))
    return eligibility, lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recovery-only", action="store_true")
    args = parser.parse_args()
    if args.recovery_only:
        result = verify_recovery(args.data_root, args.output_root)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 3)
    audit(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
