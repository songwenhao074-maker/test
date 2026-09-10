"""Analyze the eight registered Protocol-020 R1 development runs.

The analyzer reads each run's ``configuration.json``, ``summary.json`` and
``predictions.npz`` and produces a machine-readable report plus a compact
Markdown table.  It validates the paired-input contract before comparing
methods.  In particular, C-residual-off and C-residual-on must have the same
learner final state, update count and sampling exposure on a given stream;
otherwise a protective-deployment effect cannot be attributed to protection.

Registered methods are A, C-legacy, C-residual-off and C-residual-on on
replay seeds 500 and 501, all with model seed 1 and 2000 scored steps.  The
streams are development data.  The report therefore stays descriptive and
does not claim generalization, D effectiveness or final confirmation.

Examples:

    python analyze_ftmoe_protocol020_r1.py \
        --runs artifacts/ftmoe_online/protocol_020/revision_20260909/r1

    python analyze_ftmoe_protocol020_r1.py \
        --runs on500=.../C_residual_on_dev500,off500=.../C_residual_off_dev500

Missing or inconsistent runs are retained in the JSON report with an
``INCOMPLETE``/``INVALID`` status.  The analyzer never turns a partial grid
into a PASS result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_ftmoe_online import summarize_arrays
from run_ftmoe_protocol019 import extra_detection_metrics
from train_ftmoe_end_to_end import metric_arrays

ROOT = Path(__file__).resolve().parent
EXPECTED_METHODS = ("A", "C-legacy", "C-residual-off", "C-residual-on")
EXPECTED_REPLAY_SEEDS = (500, 501)
EXPECTED_KEYS = frozenset((method, seed)
                          for method in EXPECTED_METHODS
                          for seed in EXPECTED_REPLAY_SEEDS)
REQUIRED_FILES = (
    "configuration.json", "summary.json", "predictions.npz",
    "predictions.jsonl", "updates.jsonl", "r1_ledger.jsonl",
    "r1_events.jsonl", "reference.json", "resume.pt",
)


def _jsonable(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n",
                          encoding="utf8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _method_of(config):
    method = config.get("method")
    if method in EXPECTED_METHODS:
        return method
    # Keep the analyzer tolerant of an R1 state schema that calls the user
    # facing label r1_method while retaining ``method=C`` for S7 internals.
    method = config.get("r1_method") or config.get("method_label")
    return method if method in EXPECTED_METHODS else None


def _replay_seed(config):
    value = config.get("replay_seed")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _discover_run_dirs(specs):
    """Resolve a root or explicit ``label=directory`` specification."""
    directories = []
    for raw in specs:
        # Windows paths contain a colon but not an equals sign, so this split
        # remains unambiguous for the documented explicit form.
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                _, item = item.split("=", 1)
            path = Path(item)
            if (path / "configuration.json").is_file():
                directories.append(path)
                continue
            if not path.exists():
                directories.append(path)
                continue
            candidates = sorted(path.rglob("configuration.json"))
            directories.extend(candidate.parent for candidate in candidates)
    # Keep one record per directory while preserving deterministic order.
    return list(dict.fromkeys(directory.resolve() for directory in directories))


def _load_records(specs):
    records = []
    for folder in _discover_run_dirs(specs):
        record = {"folder": str(folder), "files": {}, "issues": []}
        for name in REQUIRED_FILES:
            path = folder / name
            record["files"][name] = path.is_file()
        if not folder.is_dir():
            record["issues"].append("run directory does not exist")
            records.append(record)
            continue
        if not record["files"]["configuration.json"]:
            record["issues"].append("missing configuration.json")
            records.append(record)
            continue
        try:
            record["configuration"] = _read_json(folder / "configuration.json")
        except Exception as exc:
            record["issues"].append("invalid configuration.json: " + str(exc))
            records.append(record)
            continue
        record["method"] = _method_of(record["configuration"])
        record["replay_seed"] = _replay_seed(record["configuration"])
        if record["method"] is None:
            record["issues"].append("unknown R1 method label")
        if record["replay_seed"] not in EXPECTED_REPLAY_SEEDS:
            record["issues"].append("unexpected replay seed")
        for name in REQUIRED_FILES:
            if not record["files"][name]:
                record["issues"].append("missing " + name)
        if record["files"]["summary.json"]:
            try:
                record["summary"] = _read_json(folder / "summary.json")
            except Exception as exc:
                record["issues"].append("invalid summary.json: " + str(exc))
        if record.get("summary", {}).get("configuration") not in (None,
                                                                      record.get("configuration")):
            record["issues"].append("summary/configuration mismatch")
        if record["files"]["predictions.npz"]:
            try:
                with np.load(folder / "predictions.npz", allow_pickle=False) as data:
                    record["predictions"] = {key: data[key].copy() for key in data.files}
            except Exception as exc:
                record["issues"].append("invalid predictions.npz: " + str(exc))
        records.append(record)
    return records


def _updates(record):
    path = Path(record["folder"]) / "updates.jsonl"
    if not path.is_file():
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except Exception as exc:
        record["issues"].append("invalid updates.jsonl: " + str(exc))
    return rows


def _jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except Exception:
        return []
    return rows


def _state_digest(value):
    """Hash a checkpoint tree without serializing model/optimizer contents."""
    digest = hashlib.sha256()

    def add(item):
        import torch
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda value: repr(value)):
                add(repr(key))
                add(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"list" if isinstance(item, list) else b"tuple")
            for child in item:
                add(child)
            return
        if item is None:
            digest.update(b"none")
            return
        digest.update(type(item).__name__.encode("ascii"))
        digest.update(json.dumps(_jsonable(item), sort_keys=True,
                                 ensure_ascii=False, allow_nan=False).encode("utf8"))

    add(value)
    return digest.hexdigest()


def _hash_value(value):
    canonical = json.dumps(_jsonable(value), sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf8")).hexdigest()


def _draw_sequence(rows):
    return [{"buffer_indices": row.get("buffer_indices", []),
             "anchor_indices": row.get("anchor_indices", [])}
            for row in rows if isinstance(row, dict)]


def _artifact_audit(record):
    """Audit final resume state and exact sampler state for off/on pairing."""
    path = Path(record["folder"]) / "resume.pt"
    audit = {"status": "unavailable", "issues": []}
    if not path.is_file():
        audit["issues"].append("missing resume.pt")
        return audit
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        audit["issues"].append("invalid resume.pt: " + str(exc))
        return audit
    session = state.get("session") if isinstance(state, dict) else None
    if not isinstance(session, dict):
        audit["issues"].append("resume.pt has no session state")
        return audit
    expected_steps = record.get("configuration", {}).get("steps")
    cursor = state.get("cursor", session.get("cursor"))
    audit["resume_status"] = state.get("status")
    audit["cursor"] = cursor
    if state.get("status") != "complete":
        audit["issues"].append("resume.pt is not a completed state")
    if expected_steps is not None and int(cursor or -1) != int(expected_steps):
        audit["issues"].append("resume.pt cursor is not the registered horizon")
    if "optimizer" not in session:
        audit["issues"].append("resume.pt missing optimizer state")
    if "updates" not in session:
        audit["issues"].append("resume.pt missing updates state")
    if "exposure" not in session:
        audit["issues"].append("resume.pt missing exposure state")
    if "rare_exposure" not in session:
        audit["issues"].append("resume.pt missing rare_exposure state")
    file_updates = _updates(record)
    state_updates = session.get("updates", [])
    file_sequence = _draw_sequence(file_updates)
    state_sequence = _draw_sequence(state_updates)
    audit["optimizer_state_hash"] = (
        _state_digest(session["optimizer"]) if "optimizer" in session else None
    )
    audit["updates_sequence_hash"] = _hash_value(file_sequence)
    audit["state_updates_sequence_hash"] = _hash_value(state_sequence)
    audit["sampling_exposure_hash"] = (
        _state_digest(session["exposure"]) if "exposure" in session else None
    )
    audit["rare_exposure_hash"] = (
        _state_digest(session["rare_exposure"])
        if "rare_exposure" in session else None
    )
    audit["update_count"] = session.get("update_number")
    if audit["updates_sequence_hash"] != audit["state_updates_sequence_hash"]:
        audit["issues"].append("updates.jsonl differs from resume.pt update sequence")
    if session.get("update_number") is None:
        audit["issues"].append("resume.pt missing update_number")
    audit["status"] = "complete" if not audit["issues"] else "invalid"
    return audit


def _nested_values(record, names):
    summary = record.get("summary", {})
    config = record.get("configuration", {})
    containers = [summary, config]
    for key in ("r1_summary", "r1_statistics", "protection_summary", "s7_statistics"):
        value = summary.get(key)
        if isinstance(value, dict):
            containers.append(value)
    values = []
    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                values.append((name, container[name]))
    return values


def _learner_state_hash(record):
    values = _nested_values(record, ("learner_state_hash", "learner_final_state_hash",
                                     "learner_hash"))
    return values[0][1] if values else None


def _update_count(record):
    audit = record.get("_artifact_audit")
    if isinstance(audit, dict) and audit.get("status") == "complete":
        value = audit.get("update_count")
        if isinstance(value, (int, float)):
            return int(value)
    values = _nested_values(record, ("update_count", "updates", "update_number"))
    for _, value in values:
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _sampling_signature(record):
    """Return exact final-checkpoint sampler hashes for an off/on pair."""
    audit = record.get("_artifact_audit")
    if not isinstance(audit, dict) or audit.get("status") != "complete":
        return None
    return {
        "source": "resume.pt+updates.jsonl",
        "updates_sequence_hash": audit.get("updates_sequence_hash"),
        "sampling_exposure_hash": audit.get("sampling_exposure_hash"),
        "rare_exposure_hash": audit.get("rare_exposure_hash"),
    }


def _fairness(records):
    pairs = []
    issues = []
    by_key = {(r.get("method"), r.get("replay_seed")): r for r in records
              if r.get("method") in EXPECTED_METHODS}
    for replay_seed in EXPECTED_REPLAY_SEEDS:
        off = by_key.get(("C-residual-off", replay_seed))
        on = by_key.get(("C-residual-on", replay_seed))
        pair = {"replay_seed": replay_seed, "status": "incomplete"}
        if off is None or on is None:
            pair["issues"] = ["missing residual off/on pair"]
            pairs.append(pair)
            issues.extend(pair["issues"])
            continue
        checks = {}
        off_hash, on_hash = _learner_state_hash(off), _learner_state_hash(on)
        off_updates, on_updates = _update_count(off), _update_count(on)
        off_sampling, on_sampling = _sampling_signature(off), _sampling_signature(on)
        off_audit = off.get("_artifact_audit", {})
        on_audit = on.get("_artifact_audit", {})
        checks["learner_state_hash"] = {
            "off": off_hash, "on": on_hash,
            "equal": off_hash is not None and on_hash is not None and off_hash == on_hash,
        }
        checks["update_count"] = {
            "off": off_updates, "on": on_updates,
            "equal": off_updates is not None and on_updates is not None and off_updates == on_updates,
        }
        checks["sampling_exposure"] = {
            "off": off_sampling, "on": on_sampling,
            "equal": off_sampling is not None and on_sampling is not None and
            _hash_value(off_sampling) == _hash_value(on_sampling),
        }
        checks["optimizer_state"] = {
            "off_hash": off_audit.get("optimizer_state_hash"),
            "on_hash": on_audit.get("optimizer_state_hash"),
            "equal": off_audit.get("optimizer_state_hash") is not None and
            off_audit.get("optimizer_state_hash") == on_audit.get("optimizer_state_hash"),
        }
        checks["updates_sequence"] = {
            "off_hash": off_audit.get("updates_sequence_hash"),
            "on_hash": on_audit.get("updates_sequence_hash"),
            "equal": off_audit.get("updates_sequence_hash") is not None and
            off_audit.get("updates_sequence_hash") == on_audit.get("updates_sequence_hash"),
        }
        checks["rare_exposure"] = {
            "off_hash": off_audit.get("rare_exposure_hash"),
            "on_hash": on_audit.get("rare_exposure_hash"),
            "equal": off_audit.get("rare_exposure_hash") is not None and
            off_audit.get("rare_exposure_hash") == on_audit.get("rare_exposure_hash"),
        }
        pair["checks"] = checks
        pair_issues = [name for name, value in checks.items() if not value["equal"]]
        pair["issues"] = pair_issues
        pair["status"] = "eligible" if not pair_issues else "not_eligible"
        pairs.append(pair)
        if pair_issues:
            issues.append(f"seed {replay_seed}: " + ", ".join(pair_issues))
    return {
        "pairs": pairs,
        "status": "eligible" if len(pairs) == len(EXPECTED_REPLAY_SEEDS) and not issues
        else ("incomplete" if any(p["status"] == "incomplete" for p in pairs)
              else "not_eligible"),
        "protective_attribution_eligible": not issues and
        len(pairs) == len(EXPECTED_REPLAY_SEEDS),
        "issues": issues,
    }


def _metric_segment(predictions, start, end):
    from sklearn.metrics import f1_score
    probability = predictions["probability"][start:end]
    classes = predictions["class_probability"][start:end]
    labels = predictions["labels"][start:end]
    raw = predictions["raw_labels"][start:end]
    valid = labels >= 0
    raw_valid = raw >= 0
    if not valid.any() or not raw_valid.any():
        return {"status": "unfinalized", "tolerance": None, "raw": None}
    tolerance_metrics = metric_arrays(probability[valid].reshape(-1),
                                       classes[valid].reshape(-1, 3),
                                       labels[valid].reshape(-1))
    raw_metrics = metric_arrays(probability[raw_valid].reshape(-1),
                                classes[raw_valid].reshape(-1, 3),
                                raw[raw_valid].reshape(-1))
    def end_to_end(probability_slice, classes_slice, labels_slice):
        flat_p = probability_slice.reshape(-1)
        flat_c = classes_slice.reshape(-1, 3)
        flat_y = labels_slice.reshape(-1)
        predicted_label = np.where(flat_p >= .5, flat_c.argmax(axis=1) + 1, 0)
        values = f1_score(flat_y, predicted_label, labels=[1, 2, 3],
                          average=None, zero_division=0)
        return {
            "macro_f1": float(f1_score(
                flat_y, predicted_label, labels=[1, 2, 3],
                average="macro", zero_division=0)),
            "class_f1": {str(index): float(value)
                          for index, value in zip((1, 2, 3), values)},
            "definition": "predicted_label=np.where(p>=0.5,argmax(classp)+1,0)",
        }
    tolerance_extra = extra_detection_metrics(probability[valid], labels[valid])
    raw_extra = extra_detection_metrics(probability[raw_valid], raw[raw_valid])
    return {"status": "complete",
            "tolerance": {"metrics": _jsonable(tolerance_metrics),
                          "extra_detection": _jsonable(tolerance_extra),
                          "end_to_end": end_to_end(
                              probability[valid], classes[valid], labels[valid])},
            "raw": {"metrics": _jsonable(raw_metrics),
                    "extra_detection": _jsonable(raw_extra),
                    "end_to_end": end_to_end(
                        probability[raw_valid], classes[raw_valid], raw[raw_valid])}}


def _diagnostics(predictions):
    probability = predictions["probability"].reshape(-1)
    classes = predictions["class_probability"].reshape(-1, 3)
    labels = predictions["labels"].reshape(-1)
    valid = labels >= 0
    if not valid.any():
        return None
    return _metric_segment({key: value for key, value in predictions.items()
                            if key in ("probability", "class_probability",
                                       "labels", "raw_labels")},
                          0, len(predictions["labels"]))


def _reference(record):
    path = Path(record["folder"]) / "reference.json"
    if not path.is_file():
        return {"status": "missing", "source": "same_domain_train_anchor",
                "records": []}
    try:
        value = _read_json(path)
    except Exception as exc:
        return {"status": "invalid", "error": str(exc), "records": []}
    if isinstance(value, dict):
        rows = value.get("records", value.get("reference", []))
        if not isinstance(rows, list):
            rows = [value]
    else:
        rows = value if isinstance(value, list) else []
    sources = sorted({row.get("source") for row in rows
                      if isinstance(row, dict) and row.get("source") is not None})
    source_ok = bool(rows) and sources == ["same_domain_train_anchor"]
    score_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = row.get("mean", row)
        if isinstance(score, dict):
            score_rows.append({"step": row.get("step"),
                               "f1": score.get("f1"),
                               "pr_auc": score.get("pr_auc"),
                               "positive_class_macro_f1": score.get("positive_class_macro_f1")})
    return {"status": "complete" if source_ok else ("missing_source" if rows else "missing"),
            "source": "same_domain_train_anchor", "observed_sources": sources,
            "independent_holdout": False, "records": len(rows),
            "first": score_rows[0] if score_rows else None,
            "last": score_rows[-1] if score_rows else None,
            "trajectory": score_rows}


def _event_counts(record):
    rows = _jsonl(Path(record["folder"]) / "r1_events.jsonl")
    counts = {"candidate_snapshot_evaluated": 0,
              "candidate_snapshot_accepted": 0,
              "candidate_snapshot_rejected": 0,
              "candidate_snapshot_protection_disabled": 0,
              "deployment_enabled": 0,
              "fallback": 0}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event, decision = row.get("event"), row.get("decision")
        if event == "assessment_evaluated":
            counts["candidate_snapshot_evaluated"] += 1
            if decision == "accepted_live_replaced":
                counts["candidate_snapshot_accepted"] += 1
                counts["deployment_enabled"] += 1
            elif decision == "rejected_live_unchanged":
                counts["candidate_snapshot_rejected"] += 1
            elif decision == "protection_disabled_no_admission":
                counts["candidate_snapshot_protection_disabled"] += 1
        elif (event == "live_evidence_evaluated" and
              decision == "rollback_to_frozen_A"):
            counts["fallback"] += 1
    return {"counts": counts,
            "status": "complete" if rows else "unavailable",
            "event_schema": {
                "assessment": "assessment_evaluated + decision accepted_live_replaced/rejected_live_unchanged",
                "fallback": "live_evidence_evaluated + decision rollback_to_frozen_A",
            },
            "candidate_is_evaluated_snapshot": True,
            "candidate_is_new_expert": False}


def _deployment_usage(record):
    method = record.get("method")
    if method not in ("C-residual-off", "C-residual-on"):
        return {"applicable": False, "deployment_enabled_ratio": None,
                "fallback_ratio": None, "status": "not_applicable"}
    ledger = _jsonl(Path(record["folder"]) / "r1_ledger.jsonl")
    event = _event_counts(record)
    alphas = [float(row["deployment_alpha"]) for row in ledger
              if isinstance(row, dict) and
              isinstance(row.get("deployment_alpha"), (int, float))]
    if not alphas:
        return {"applicable": True, "deployment_enabled_ratio": None,
                "fallback_ratio": None, "status": "unavailable",
                "event_counts": event}
    alpha = np.asarray(alphas, dtype=np.float64)
    return {"applicable": True,
            "deployment_enabled_ratio": float((alpha > 0.0).mean()),
            "fallback_ratio": float((alpha <= 0.0).mean()),
            "deployment_alpha_mean": float(alpha.mean()),
            "deployment_alpha_min": float(alpha.min()),
            "deployment_alpha_max": float(alpha.max()),
            "deployment_alpha_records": int(len(alpha)),
            "status": "complete", "event_counts": event}


def _cost(record):
    summary = record.get("summary", {})
    r1 = summary.get("r1_summary", {})
    if not isinstance(r1, dict):
        r1 = {}
    values = {
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "rss_gib": summary.get("rss_gib"),
        "prediction_mean_seconds": summary.get("prediction_mean_seconds"),
        "prediction_p95_seconds": summary.get("prediction_p95_seconds"),
        "update_total_seconds": summary.get("update_total_seconds"),
    }
    for name in ("parameter_counts", "parameter_memory_bytes", "prediction_cost",
                 "correction_memory_bytes", "parameters", "snapshot_storage_bytes",
                 "snapshot_storage", "shadow_forward_seconds",
                 "additional_forward_seconds", "actual_expert_forward_count",
                 "deployment_forward_count", "trainable_parameter_count",
                 "total_parameter_count"):
        if name in r1:
            values[name] = r1[name]
        elif name in summary:
            values[name] = summary[name]
    return _jsonable(values)


def _run_report(record, manifest_cache):
    config = record.get("configuration", {})
    summary = record.get("summary", {})
    predictions = record.get("predictions")
    if predictions is None:
        return {"folder": record["folder"], "method": record.get("method"),
                "replay_seed": record.get("replay_seed"),
                "status": "incomplete", "issues": record["issues"]}
    required_keys = ("probability", "class_probability", "labels", "raw_labels")
    missing = [key for key in required_keys if key not in predictions]
    if missing:
        record["issues"].append("predictions missing " + ", ".join(missing))
        return {"folder": record["folder"], "method": record.get("method"),
                "replay_seed": record.get("replay_seed"), "status": "invalid",
                "issues": record["issues"]}
    try:
        metrics = summarize_arrays(
            predictions["probability"], predictions["class_probability"],
            predictions["labels"], predictions["raw_labels"])
    except Exception as exc:
        record["issues"].append("metric recomputation failed: " + str(exc))
        return {"folder": record["folder"], "method": record.get("method"),
                "replay_seed": record.get("replay_seed"), "status": "invalid",
                "issues": record["issues"]}
    if summary.get("metrics") not in (None, metrics):
        record["issues"].append("summary metrics differ from independent recomputation")
    if summary.get("configuration") not in (None, config):
        record["issues"].append("summary configuration differs from run configuration")
    artifact_audit = _artifact_audit(record)
    record["_artifact_audit"] = artifact_audit
    record["issues"].extend(artifact_audit.get("issues", []))
    anchor_retention = _reference(record)
    if anchor_retention.get("status") != "complete":
        record["issues"].append(
            "reference.json is missing or is not same_domain_train_anchor")
    deployment = _deployment_usage(record)
    if (deployment.get("applicable") and
            deployment.get("status") != "complete"):
        record["issues"].append("R1 deployment_alpha ledger is unavailable")
    n = len(predictions["labels"])
    manifest = manifest_cache.get(record.get("replay_seed"))
    if manifest is None:
        record["issues"].append("stream manifest unavailable")
        registered, blocks = [], []
    else:
        registered, blocks = _phase_ranges(manifest, n)
    phase_metrics = []
    for item in registered:
        phase_metrics.append({**{key: item[key] for key in ("index", "name", "start", "end", "kind")},
                              "metrics": _metric_segment(predictions, item["start"], item["end"])})
    block_metrics = []
    for item in blocks:
        block_metrics.append({**{key: item[key] for key in ("index", "name", "start", "end", "kind")},
                              "metrics": _metric_segment(predictions, item["start"], item["end"])})
    worst = None
    complete_phases = [item for item in phase_metrics
                       if item["metrics"].get("tolerance") is not None]
    if complete_phases:
        def f1(item):
            return item["metrics"]["tolerance"]["metrics"].get("f1")
        finite = [item for item in complete_phases if f1(item) is not None]
        if finite:
            worst = min(finite, key=lambda item: (float(f1(item)), item["index"]))
            worst = {"index": worst["index"], "name": worst["name"],
                     "start": worst["start"], "end": worst["end"],
                     "f1": f1(worst),
                     "pr_auc": worst["metrics"]["tolerance"]["metrics"].get("pr_auc")}
    return {
        "folder": record["folder"],
        "method": record.get("method"),
        "replay_seed": record.get("replay_seed"),
        "model_seed": config.get("model_seed"),
        "status": "complete" if not record["issues"] else "invalid",
        "issues": list(record["issues"]),
        "configuration": {
            "source_checkpoint": config.get("source_checkpoint"),
            "stream": config.get("stream"),
            "stream_sha256": config.get("stream_sha256"),
            "learning_rate": config.get("learning_rate"),
            "input_contract_version": config.get("input_contract_version"),
            "normalization_version": config.get("normalization_version"),
            "graph_semantics_version": config.get("graph_semantics_version"),
            "model_class": config.get("model_class"),
            "protection_enabled": config.get("protection_enabled"),
        },
        "overall": metrics,
        "overall_extra_detection": _metric_segment(predictions, 0, n),
        "registered_phase_metrics": phase_metrics,
        "time_block_metrics_400": block_metrics,
        "worst_registered_phase": worst,
        "anchor_retention": anchor_retention,
        "deployment": deployment,
        "cost": _cost(record),
        "learner_state_hash": _learner_state_hash(record),
        "update_count": _update_count(record),
        "sampling_signature": _sampling_signature(record),
        "artifact_audit": {key: value for key, value in artifact_audit.items()
                           if key != "issues"},
    }


def _phase_ranges(manifest, steps, block_size=400):
    phases = manifest.get("phases") or []
    phase_len = int(manifest.get("phase_len") or steps)
    registered = []
    for index, phase in enumerate(phases):
        start = index * phase_len
        end = min(steps, (index + 1) * phase_len)
        if start >= steps:
            break
        name = phase.get("name", f"phase_{index}") if isinstance(phase, dict) else str(phase)
        registered.append({"index": index, "name": name, "start": start,
                           "end": end, "kind": "registered_phase"})
    if not registered:
        registered = [{"index": 0, "name": "registered_stream", "start": 0,
                       "end": steps, "kind": "registered_phase"}]
    blocks = []
    for start in range(0, steps, block_size):
        index = len(blocks)
        blocks.append({"index": index, "name": f"time_block_{index}",
                       "start": start, "end": min(steps, start + block_size),
                       "kind": "time_block_400"})
    return registered, blocks


def _load_manifests(records):
    manifests = {}
    issues = []
    for record in records:
        config = record.get("configuration", {})
        seed = record.get("replay_seed")
        stream = config.get("stream")
        if seed in manifests:
            continue
        if not stream:
            issues.append(f"seed {seed}: configuration has no stream")
            continue
        path = Path(stream)
        try:
            manifest = _read_json(path / "manifest.json")
            if manifest.get("protocol") != "020":
                issues.append(f"seed {seed}: stream protocol is not 020")
            if manifest.get("seed") != seed:
                issues.append(f"seed {seed}: stream manifest seed mismatch")
            if manifest.get("steps") != 2000:
                issues.append(f"seed {seed}: stream horizon is not 2000")
            stream_hash = config.get("stream_sha256")
            if stream_hash and (path / "stream.npz").is_file():
                digest = hashlib.sha256((path / "stream.npz").read_bytes()).hexdigest()
                if digest != stream_hash:
                    issues.append(f"seed {seed}: stream hash mismatch")
            manifests[seed] = manifest
        except Exception as exc:
            issues.append(f"seed {seed}: cannot read stream manifest: {exc}")
    return manifests, issues


def _fairness_input_checks(reports):
    issues = []
    complete = [r for r in reports if r.get("status") == "complete"]
    for seed in EXPECTED_REPLAY_SEEDS:
        group = [r for r in complete if r.get("replay_seed") == seed]
        if not group:
            issues.append(f"seed {seed}: no complete run")
            continue
        for key, label in (("model_seed", "model seed"), ("stream_sha256", "stream hash"),
                           ("stream_manifest_sha256", "stream manifest hash"),
                           ("source_checkpoint", "source checkpoint"),
                           ("input_contract_version", "input contract"),
                           ("normalization_version", "normalization"),
                           ("graph_semantics_version", "graph semantics")):
            values = {json.dumps(r["configuration"].get(key), sort_keys=True)
                      for r in group}
            if len(values) != 1:
                issues.append(f"seed {seed}: unequal {label} across methods")
        expected_lr = {0.0 if r.get("method") == "A" else 1e-4 for r in group}
        for r in group:
            expected = 0.0 if r.get("method") == "A" else 1e-4
            if r["configuration"].get("learning_rate") != expected:
                issues.append(f"{r.get('method')} seed {seed}: learning rate is not registered")
    return issues


def _comparisons(reports):
    by_key = {(r.get("method"), r.get("replay_seed")): r for r in reports
              if r.get("status") == "complete"}
    output = {}
    for method in ("C-legacy", "C-residual-off", "C-residual-on"):
        rows = []
        for seed in EXPECTED_REPLAY_SEEDS:
            candidate = by_key.get((method, seed))
            baseline = by_key.get(("A", seed))
            if candidate is None or baseline is None:
                rows.append({"replay_seed": seed, "status": "incomplete"})
                continue
            c = candidate["overall"].get("second_half", {})
            a = baseline["overall"].get("second_half", {})
            rows.append({"replay_seed": seed, "status": "complete",
                         "f1_delta_vs_A": _delta(c.get("f1"), a.get("f1")),
                         "pr_auc_delta_vs_A": _delta(c.get("pr_auc"), a.get("pr_auc")),
                         "precision_delta_vs_A": _delta(c.get("precision"), a.get("precision")),
                         "recall_delta_vs_A": _delta(c.get("recall"), a.get("recall")),
                         "fp_delta_vs_A": _delta(c.get("fp"), a.get("fp")),
                         "fn_delta_vs_A": _delta(c.get("fn"), a.get("fn"))})
        output[method] = rows
    return output


def _delta(left, right):
    if left is None or right is None:
        return None
    return float(left - right)


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report):
    lines = [
        "# Protocol 020 R1 development analysis",
        "",
        "Status: **%s**. The grid is development data only; this report does not claim generalization, D effectiveness, or final confirmation." % report["status"],
        "",
        "Protective attribution eligibility: **%s**." % report["fairness"]["protective_attribution_eligible"],
        "",
        "| replay | method | status | full F1 | full PR-AUC | worst registered F1 | anchor F1 | conditional resource F1 | end-to-end resource F1 | deploy ratio | base-only ratio | accepted / rejected | elapsed s | RSS GiB |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["runs"], key=lambda item: (item.get("replay_seed") or 0,
                                                           item.get("method") or "")):
        overall = row.get("overall", {}).get("full", {})
        worst = row.get("worst_registered_phase") or {}
        anchor = row.get("anchor_retention", {}).get("last") or {}
        cond = row.get("overall_extra_detection", {}).get("tolerance", {})
        # The extra-detection bundle is the common metric source; conditional
        # and endpoint resource values come from metric_arrays in its metrics.
        diag = cond.get("metrics", {}) if isinstance(cond, dict) else {}
        conditional = diag.get("positive_class_macro_f1")
        endpoint = (row.get("overall_extra_detection", {}).get("tolerance", {})
                    .get("end_to_end", {}) if isinstance(row.get("overall_extra_detection"), dict)
                    else {})
        deployment = row.get("deployment", {})
        events = deployment.get("event_counts", {}).get("counts", {})
        accepted = events.get("candidate_snapshot_accepted", "—")
        rejected = events.get("candidate_snapshot_rejected", "—")
        cost = row.get("cost", {})
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s | %s |" % (
            row.get("replay_seed", "—"), row.get("method", "—"), row.get("status", "—"),
            _fmt(overall.get("f1")), _fmt(overall.get("pr_auc")), _fmt(worst.get("f1")),
            _fmt(anchor.get("f1")),
            _fmt(conditional), _fmt(endpoint.get("macro_f1")),
            _fmt(deployment.get("deployment_enabled_ratio")),
            _fmt(deployment.get("fallback_ratio")), accepted, rejected,
            _fmt(cost.get("elapsed_seconds"), 2), _fmt(cost.get("rss_gib"), 3)))
    lines.extend(["", "Registered phases use the stream manifest. A single-phase dev501 stream remains one phase; its 400-step slices are reported separately as time blocks.", "",
                  "Candidate counts refer to evaluated residual assessment snapshots, not new experts.", ""])
    if report["issues"]:
        lines.append("Input or completeness issues:")
        for issue in report["issues"]:
            lines.append("- " + issue)
    else:
        lines.append("Input/completeness issues: none detected.")
    return "\n".join(lines) + "\n"


def analyze(specs, output=None, markdown=None):
    records = _load_records(specs)
    manifests, manifest_issues = _load_manifests(records)
    reports = [_run_report(record, manifests) for record in records]
    by_key = {}
    duplicate_keys = []
    for row in reports:
        key = (row.get("method"), row.get("replay_seed"))
        if key in by_key:
            duplicate_keys.append(key)
        by_key[key] = row
    missing_keys = sorted(EXPECTED_KEYS - set(by_key))
    unexpected_keys = sorted(set(by_key) - EXPECTED_KEYS)
    input_issues = manifest_issues + _fairness_input_checks(reports)
    if duplicate_keys:
        input_issues.append("duplicate run keys: " + repr(duplicate_keys))
    if missing_keys:
        input_issues.append("missing run keys: " + repr(missing_keys))
    if unexpected_keys:
        input_issues.append("unexpected run keys: " + repr(unexpected_keys))
    fairness = _fairness(records)
    input_issues.extend(fairness["issues"])
    all_complete = len(reports) == len(EXPECTED_KEYS) and not missing_keys and not unexpected_keys and \
        all(row.get("status") == "complete" for row in reports)
    status = "COMPLETE" if all_complete and not input_issues else (
        "INCOMPLETE" if missing_keys or any("missing" in item.lower() for item in input_issues)
        else "INVALID")
    report = {
        "schema_version": 1,
        "protocol": "020",
        "revision": "20260909-r1",
        "status": status,
        "data_scope": "development_only",
        "claims": {"generalization": False, "dynamic_D_effectiveness": False,
                   "final_confirmation": False},
        "expected_grid": {"methods": list(EXPECTED_METHODS),
                          "replay_seeds": list(EXPECTED_REPLAY_SEEDS),
                          "model_seed": 1, "steps": 2000, "run_count": 8},
        "runs": reports,
        "fairness": fairness,
        "comparisons_vs_A": _comparisons(reports),
        "issues": sorted(set(input_issues)),
        "performance_gate": {"status": "descriptive_only",
                              "pass": None,
                              "reason": "R1 report does not define a success PASS gate"},
    }
    if output is None:
        roots = [Path(spec.split("=", 1)[-1].split(",", 1)[0]) for spec in specs]
        root = roots[0] if roots else ROOT
        if root.is_file():
            root = root.parent
        output = root / "r1_analysis.json"
    output = Path(output)
    if output.suffix.lower() != ".json":
        output = output / "r1_analysis.json"
    if markdown is None:
        markdown = output.with_suffix(".md")
    _atomic_json(output, report)
    markdown = Path(markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(report), encoding="utf8")
    return report, output, markdown


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=str, nargs="+", required=True,
                        help="R1 root directory, or comma-separated label=run_dir entries")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON report path (default: <runs>/r1_analysis.json)")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Markdown report path (default: alongside JSON report)")
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    report, output, markdown = analyze(args.runs, args.output, args.markdown)
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "output": str(output), "markdown": str(markdown),
                      "protective_attribution_eligible":
                          report["fairness"]["protective_attribution_eligible"]},
                     ensure_ascii=False))
