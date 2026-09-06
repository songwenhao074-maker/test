"""Protocol 019 S2 — data-only distribution audit of a development stream.

Pure stream statistics (no model, no labels future).  Mirrors the plan
§7.1/§17 audit categories:

- Resources: per-resource p10/p50/p90/p95/p99, zero ratio, training-range
  exceed ratio (range = protocol_004_physical training block quantiles)
- Host aggregation: mean active containers, P(host>=2), P(host>=3), max
  occupancy
- Events: anomaly prevalence, CPU/RAM/Disk counts, mean/p95 anomaly duration,
  recovery duration
- Schedule: proposed vs executed migrations, rejection rate, proposed!=actual
- Temporal structure: lag-1/12/30 autocorrelation of per-host aggregate
  demand
- Normalization v2 alarm report (normalized abs max < 50)
- Identity audit: how many slot changes are identity replacements vs real
  migrations

Writes <stream_dir>/distribution.json. Exit code 0 on success.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_DATA = ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical"


def quantiles(values, qs=(10, 50, 90, 95, 99)):
    return {f"p{q}": float(np.percentile(values, q)) for q in qs}


def autocorr(series, lag):
    if len(series) <= lag:
        return float("nan")
    x = series[lag:]
    y = series[: len(series) - lag]
    if x.std() == 0 or y.std() == 0:
        return 1.0 if float(x.mean()) != 0 else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def audit(stream_dir: Path) -> dict:
    if not (stream_dir / "manifest.json").exists():
        raise FileNotFoundError(stream_dir / "manifest.json")
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    with np.load(stream_dir / "stream.npz") as data:
        arrays = {key: data[key] for key in data.files}
    steps = manifest["steps"]
    host = arrays["host_features"][:steps]          # (T,16,7)
    demands = arrays["demands"][:steps]
    labels = arrays["raw_labels"][:steps]
    capacities = arrays["capacities"]               # (16,3)
    creation = arrays["creation_ids"][: steps + 1]  # includes guard for last label
    before = arrays["before_placement"][:steps]
    schedules = arrays["schedules"][:steps]
    events = json.loads((stream_dir / "events.json").read_text(encoding="utf8"))

    # ---- resource columns cpu=0, ram=1, disk=4 ----
    resource_report = {}
    training = np.load(TRAIN_DATA / "container_demand_series.npy")
    with open(TRAIN_DATA / "manifest.json", encoding="utf8") as handle:
        train_manifest = json.load(handle)
    train_demands = training[train_manifest["train_blocks"]].reshape(-1, 16, 7)
    for name, index in (("cpu", 0), ("ram", 1), ("disk", 4)):
        online = host[:, :, index]  # per-host aggregate demand
        container_demand = demands[:, :, index]  # per-container demand
        train = train_demands[:, :, index]
        resource_report[name] = {
            **quantiles(online.ravel()),
            "zero_ratio": float((online == 0).mean()),
            "mean": float(online.mean()),
            "container_p95": float(np.percentile(container_demand, 95)),
            "container_train_p95": float(np.percentile(train, 95)),
            "container_train_p99": float(np.percentile(train, 99)),
            "container_train_max": float(train.max()),
            "container_train_range_exceed_ratio": float(
                (container_demand > train.max()).mean()),
            "container_train_p95_exceed_ratio": float(
                (container_demand > np.percentile(train, 95)).mean()),
            "host_aggregate_acf_lag1": autocorr(online.sum(1), 1),
            "host_aggregate_acf_lag12": autocorr(online.sum(1), 12),
            "host_aggregate_acf_lag30": autocorr(online.sum(1), 30),
        }
    # ---- host aggregation ----
    occupancy = (creation[:steps] >= 0).sum(1)  # active containers per interval
    host_matrix = np.zeros((steps, 16), int)
    for t in range(steps):
        for c in range(16):
            if creation[t, c] >= 0 and before[t, c] >= 0:
                host_matrix[t, before[t, c]] += 1
    host_report = {
        "mean_active_containers": float(occupancy.mean()),
        "p_host_ge_2": float((host_matrix >= 2).mean()),
        "p_host_ge_3": float((host_matrix >= 3).mean()),
        "max_host_occupancy": int(host_matrix.max()),
        "mean_host_occupancy": float(host_matrix.mean()),
    }
    # ---- events ----
    anomaly = labels > 0
    prevalence = float(anomaly.mean())
    event_report = {
        "anomaly_prevalence": prevalence,
        "host_steps": int(labels.size),
        "anomaly_host_steps": int(anomaly.sum()),
        "counts": {
            "cpu": int((labels == 1).sum()),
            "ram": int((labels == 2).sum()),
            "disk": int((labels == 3).sum()),
        },
    }
    # per-host event duration (consecutive anomalous intervals)
    durations = []
    for h in range(16):
        mask = (labels[:, h] > 0).astype(int)
        changes = np.diff(np.concatenate([[0], mask, [0]]))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        durations.extend((ends - starts).tolist())
    durations = np.asarray(durations) if durations else np.zeros(0, int)
    event_report["mean_anomaly_duration"] = float(durations.mean()) if durations.size else 0.0
    event_report["p95_anomaly_duration"] = float(np.percentile(durations, 95)) if durations.size else 0.0
    event_report["n_anomaly_events"] = int(len(durations))
    # recovery duration: length of a normal run that directly follows an
    # anomalous run on the same host (return to baseline).
    recovery = []
    for h in range(16):
        normal = (labels[:, h] == 0).astype(int)
        changes = np.diff(np.concatenate([[0], normal, [0]]))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if start > 0 and labels[start - 1, h] > 0:
                recovery.append(int(end - start))
    event_report["mean_recovery_duration"] = (
        float(np.mean(recovery)) if recovery else 0.0)
    # ---- schedule ----
    proposed_total = 0
    executed_total = 0
    identity_ok = 0
    replacement = 0
    real_migration = 0
    proposed_but_actual_diff = 0
    rejected = 0
    same_identity_edges = 0
    for entry in events[:steps]:
        decision = entry["proposed_decision"]
        actual = entry["actual_migrations"]
        proposed_total += len(decision)
        executed_total += len(actual)
        proposed_set = set(tuple(x) for x in decision)
        actual_set = set(tuple(x) for x in actual)
        proposed_but_actual_diff += len(proposed_set - actual_set)
        rejected += len([d for d in decision if tuple(d) not in actual_set])
    # identity audit over transitions t-1 -> t (window of the whole stream)
    for t in range(1, steps):
        for c in range(16):
            pid, cid = creation[t - 1, c], creation[t, c]
            if pid >= 0 and cid >= 0 and pid == cid:
                if before[t - 1, c] >= 0 and before[t, c] >= 0 and before[t - 1, c] != before[t, c]:
                    real_migration += 1
            elif pid >= 0 and cid >= 0 and pid != cid:
                if before[t - 1, c] >= 0 and before[t, c] >= 0 and before[t - 1, c] != before[t, c]:
                    replacement += 1
            elif pid >= 0 and cid < 0:
                pass  # task ended
            elif pid < 0 and cid >= 0:
                pass  # task arrival
    schedule_report = {
        "proposed_migrations": proposed_total,
        "executed_migrations": executed_total,
        "rejection_rate": float(rejected / max(proposed_total, 1)),
        "proposed_but_not_executed": proposed_but_actual_diff,
        "identity_audit": {
            "real_same_identity_migrations": real_migration,
            "identity_replacements_misread_as_migration": replacement,
            "slot_changes_total": real_migration + replacement,
        },
    }
    # ---- normalization alarm (v1 vs v2) ----
    v2_artifact = ROOT / "artifacts/ftmoe_online/protocol_019/normalization_v2_time_scale.json"
    norm_report = {}
    if v2_artifact.exists():
        import sys
        sys.path.insert(0, str(ROOT))
        from recovery.PreGANSrc.src.ftmoe_normalization import normalized_abs_max_report
        data = json.loads(v2_artifact.read_text(encoding="utf8"))
        norm_report["v1"] = normalized_abs_max_report(
            host, np.asarray(data["time_scale_v1_16x7"]))
        norm_report["v2"] = normalized_abs_max_report(
            host, np.asarray(data["time_scale_v2_16x7"]))
    report = {
        "stream": str(stream_dir.resolve()),
        "manifest_seed": manifest.get("seed"),
        "manifest_steps": steps,
        "resources": resource_report,
        "host_aggregation": host_report,
        "events": event_report,
        "schedule": schedule_report,
        "temporal": {
            "mean_cpu_lag1": resource_report["cpu"]["host_aggregate_acf_lag1"],
        },
        "normalization": norm_report,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.stream)
    out = args.stream / "distribution.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])
    print(f"written: {out}")
