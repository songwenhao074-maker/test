"""Protocol 020 S4 — analyze the data-only capacity scan candidates.

Reads every registered (steps=400, non-smoke) candidate directory under
artifacts/ftmoe_online/protocol_020/capacity_scan/<axis>_<value>/ and
emits:

- per-candidate statistics following the plan §9 schema (raw / tolerance
  class counts, pure / multi-resource overload counts, anomaly prevalence,
  event counts + duration stats, deployment/migration rejection rates,
  active containers mean, host occupancy p95);
- the pre-registered data gate checklist (§11): normal share 80-95%,
  CPU/RAM/Disk dominant host-steps >= 150 and >= 1%, >= 30 independent
  events per fault class, deployment rejection < 20%, migration rejection
  < 40%.

Outputs capacity_scan_report.json and prints per-axis tables.

Usage:
    python analyze_ftmoe_protocol020_capacity_scan.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCAN = ROOT / "artifacts/ftmoe_online/protocol_020/capacity_scan"
REPORT = SCAN / "capacity_scan_report.json"

STEP_FLOOR_REF = 150          # dominant host-steps per fault class (plan §11, @32000)
EVENT_FLOOR_REF = 30          # independent events per fault class (plan §11, @32000)
REF_HOST_STEPS = 32000        # plan §11 reference horizon (2000 x 16)
PHASE_HOST_STEPS = 8000       # one drift phase (500 x 16), plan §13 floor = 100
PHASE_DOMINANT_FLOOR = 100
NORMAL_MIN, NORMAL_MAX = 0.80, 0.95
DEPLOY_REJ_MAX, MIGRATE_REJ_MAX = 0.20, 0.40


def tolerance_counts(labels, steps):
    """±1 interval tolerance per host: prev label first, then next (raw)."""
    tol = labels[:steps].copy()
    prev = np.zeros_like(tol)
    prev[1:] = labels[:steps - 1]
    nxt = np.zeros_like(tol)
    nxt[:-1] = labels[1:steps]
    filled = np.where(tol > 0, tol, 0)
    filled = np.where(filled == 0, prev, filled)
    filled = np.where(filled == 0, nxt, filled)
    return np.bincount(filled.ravel(), minlength=4)


def analyze_candidate(path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("smoke") or manifest["steps"] != 400:
        return None
    with np.load(path / "stream.npz") as data:
        labels = data["raw_labels"]
        ratio = data["overload_ratio"]
        mask = data["overload_mask"]
        caps = data["capacities"]
        creation = data["creation_ids"]
        deploy_a = data["deploy_attempts"]
        deploy_r = data["deploy_rejected"]
        migrate_a = data["migrate_attempts"]
        migrate_r = data["migrate_rejected"]
    steps = manifest["steps"]
    scored = labels[:steps]
    n = steps * 16
    overloaded = scored > 0
    masks = mask[:steps]
    multi = (masks.sum(-1) > 1) & overloaded
    triple = (masks.sum(-1) == 3) & overloaded
    raw_counts = np.bincount(scored.ravel(), minlength=4)
    active = (creation[:steps] >= 0).sum(-1)  # live containers per interval
    occupied = (creation[:steps] >= 0).any(-1)  # hosts with >=1 container
    active_mask = creation[:steps] >= 0
    host_count = (creation[:steps].sum(-1) > -16)  # placeholder unused
    occupancy = active_mask.sum(-1) / 16.0
    deploy_attempts = int(deploy_a.sum())
    deploy_rejected = int(deploy_r.sum())
    migrate_attempts = int(migrate_a.sum())
    migrate_rejected = int(migrate_r.sum())
    # pure fault host-steps: dominant class on a single overloaded resource
    pure = np.zeros(4, dtype=np.int64)
    for r, name in ((1, "cpu"), (2, "ram"), (3, "disk")):
        pure[r] = int(((scored == r) & (masks.sum(-1) == 1)).sum())
    events = manifest.get("events", {})
    return {
        "candidate": path.name,
        "profile": manifest["profile"],
        "raw_class_counts": [int(x) for x in raw_counts.tolist()],
        "tolerance_class_counts": [int(x) for x in tolerance_counts(labels, steps).tolist()],
        "pure_fault_counts": [int(x) for x in pure.tolist()],
        "multi_resource_overload_counts": {
            "cpu_ram": int(((masks[:, :, 0] == 1) & (masks[:, :, 1] == 1)
                            & (masks[:, :, 2] == 0) & overloaded).sum()),
            "cpu_disk": int(((masks[:, :, 0] == 1) & (masks[:, :, 1] == 0)
                             & (masks[:, :, 2] == 1) & overloaded).sum()),
            "ram_disk": int(((masks[:, :, 0] == 0) & (masks[:, :, 1] == 1)
                             & (masks[:, :, 2] == 1) & overloaded).sum()),
            "triple": int(triple.sum()),
            "any_multi": int(multi.sum()),
        },
        "anomaly_prevalence": float(overloaded.mean()),
        "normal_share": float((scored == 0).mean()),
        "event_counts": {name: events.get(name, {}).get("count", 0)
                         for name in ("1", "2", "3")},
        "mean_event_duration": {name: events.get(name, {}).get("mean_duration", 0.0)
                                for name in ("1", "2", "3")},
        "p95_event_duration": {name: events.get(name, {}).get("p95_duration", 0.0)
                               for name in ("1", "2", "3")},
        "deployment_attempts": deploy_attempts,
        "deployment_rejected": deploy_rejected,
        "deployment_rejection_rate": float(deploy_rejected / max(deploy_attempts, 1)),
        "migration_attempts": migrate_attempts,
        "migration_rejected": migrate_rejected,
        "migration_rejection_rate": float(migrate_rejected / max(migrate_attempts, 1)),
        "active_containers_mean": float(active[overloaded.any(-1)].mean())
        if overloaded.any() else float(active.mean()),
        "host_occupancy_p95": float(np.percentile(occupancy, 95)),
        "scored_host_steps": n,
    }


def gate_check(entry, horizon):
    counts = entry["raw_class_counts"]
    normal_share = entry["normal_share"]
    fraction = horizon / REF_HOST_STEPS
    step_floor = max(1, int(round(STEP_FLOOR_REF * fraction)))
    event_floor = max(1, int(round(EVENT_FLOOR_REF * fraction)))
    phase_projection = {name: int(round(counts[i] / horizon * PHASE_HOST_STEPS))
                        for i, name in ((1, "cpu"), (2, "ram"), (3, "disk"))}
    checks = {
        "normal_share_in_[0.80,0.95]": NORMAL_MIN <= normal_share <= NORMAL_MAX,
        "cpu_hoststeps>=floor": counts[1] >= step_floor,
        "ram_hoststeps>=floor": counts[2] >= step_floor,
        "disk_hoststeps>=floor": counts[3] >= step_floor,
        "cpu_events>=floor": entry["event_counts"]["1"] >= event_floor,
        "ram_events>=floor": entry["event_counts"]["2"] >= event_floor,
        "disk_events>=floor": entry["event_counts"]["3"] >= event_floor,
        "deployment_rejection<0.20": entry["deployment_rejection_rate"] < DEPLOY_REJ_MAX,
        "migration_rejection<0.40": entry["migration_rejection_rate"] < MIGRATE_REJ_MAX,
    }
    per_phase = {name: (projection >= PHASE_DOMINANT_FLOOR)
                 for name, projection in phase_projection.items()}
    return checks, all(checks.values()), phase_projection, per_phase


def axis_table(entries):
    lines = []
    lines.append("axis value  | norm%  CPU RAM Disk | ev C/R/D | phase-proj C/R/D | dep-rej mig-rej | gate")
    for e in entries:
        c = e["raw_class_counts"]
        ev = e["event_counts"]
        checks, ok, proj, per_phase = gate_check(e, e["scored_host_steps"])
        lines.append("%-5s %-5s | %5.1f%% %4d %3d %4d | %3d/%3d/%3d | %4d/%4d/%4d | %6.1f%% %6.1f%% | %s"
                     % (e["profile"]["axis"], str(e["profile"]["value"]),
                        e["normal_share"] * 100, c[1], c[2], c[3],
                        ev["1"], ev["2"], ev["3"],
                        proj["cpu"], proj["ram"], proj["disk"],
                        e["deployment_rejection_rate"] * 100,
                        e["migration_rejection_rate"] * 100,
                        "PASS" if ok else "FAIL"))
    return "\n".join(lines)


def main():
    entries = []
    for directory in sorted(SCAN.iterdir()):
        if not directory.is_dir():
            continue
        for sub in sorted(directory.iterdir()):
            if not (sub / "manifest.json").is_file():
                continue
            entry = analyze_candidate(sub)
            if entry is not None:
                profile = entry["profile"]
                axis = "cpu" if "cpu" in profile and profile.get("cpu") != 1.0 \
                    else "ram" if profile.get("ram") != 1.0 \
                    else "disk"
                entry["profile"]["axis"] = axis
                entry["profile"]["value"] = profile.get(axis, 1.0)
                horizon = entry["scored_host_steps"]
                checks, ok, proj, per_phase = gate_check(entry, horizon)
                entry["gates"] = checks
                entry["gate_pass"] = ok
                entry["phase_projection_per_8000"] = proj
                entry["phase_projection_pass"] = per_phase
                entries.append(entry)
    entries.sort(key=lambda e: (e["profile"]["axis"], e["profile"]["value"]))
    report = {"schema_version": 1, "protocol": "020", "phase": "S4",
              "gate_rules": {"normal_share": [NORMAL_MIN, NORMAL_MAX],
                             "fault_hoststep_floor_reference": STEP_FLOOR_REF,
                             "event_floor_reference": EVENT_FLOOR_REF,
                             "reference_horizon_host_steps": REF_HOST_STEPS,
                             "phase_horizon_host_steps": PHASE_HOST_STEPS,
                             "phase_dominant_floor": PHASE_DOMINANT_FLOOR,
                             "deployment_rejection_max": DEPLOY_REJ_MAX,
                             "migration_rejection_max": MIGRATE_REJ_MAX},
              "candidates": entries}
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf8")
    for axis in ("cpu", "ram", "disk"):
        axis_entries = [e for e in entries if e["profile"]["axis"] == axis]
        if axis_entries:
            print("=== %s axis ===" % axis)
            print(axis_table(axis_entries))
    print("wrote %s (%d candidates)" % (REPORT, len(entries)))


if __name__ == "__main__":
    main()
