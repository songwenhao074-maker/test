"""Protocol 020 S5 — drift phase dominance analyzer (data-only, §13 gate).

For every registered drift stream directory under
artifacts/ftmoe_online/protocol_020/drift_streams/<cohort>_seed*/ it checks:

- per-phase raw dominant-fault class counts over the scored horizon;
- phase naming gate (§13): the target fault of phase p must be dominant:
      target_dominant_count >= 100
      target_count > cpu_count, target_count > ram_count, target_count > disk_count
      target share among anomalous host-steps >= 50%
- rejection gates across the whole stream (deployment < 20%, migration < 40%)
  and phase-local tolerance counts.

Outputs drift_analysis.json next to each stream + prints a table.

Usage:
    python analyze_ftmoe_protocol020_drift.py [stream_dir ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DRIFT = ROOT / "artifacts/ftmoe_online/protocol_020/drift_streams"

TARGET = {"cpu_fault": 1, "ram_fault": 2, "disk_fault": 3}
DOMINANT_FLOOR = 100
SHARE_FLOOR = 0.50


def per_phase_counts(labels, phase_len, steps, n_phases):
    out = []
    for p in range(n_phases):
        seg = labels[p * phase_len:(p + 1) * phase_len]
        counts = np.bincount(seg.ravel(), minlength=4)
        out.append(counts)
    return out


def tolerance_counts_segment(labels, start, end):
    """±1-interval per-host tolerance on the segment (prev first, then next,
    no cross-phase boundary borrowing)."""
    seg = labels[start:end].copy()
    filled = np.where(seg > 0, seg, 0)
    prev = np.zeros_like(seg)
    prev[1:] = seg[:-1]
    nxt = np.zeros_like(seg)
    nxt[:-1] = seg[1:]
    filled = np.where(filled == 0, prev, filled)
    filled = np.where(filled == 0, nxt, filled)
    return np.bincount(filled.ravel(), minlength=4)


def analyze(stream_dir: Path):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    with np.load(stream_dir / "stream.npz") as data:
        labels = data["raw_labels"]
        deploy_a = data["deploy_attempts"]
        deploy_r = data["deploy_rejected"]
        migrate_a = data["migrate_attempts"]
        migrate_r = data["migrate_rejected"]
    steps = manifest["steps"]
    phase_len = int(manifest["phase_len"])
    phases = manifest["phases"]
    scored = labels[:steps]
    seg_counts = per_phase_counts(labels, phase_len, steps, len(phases))
    checks = []
    for p, phase in enumerate(phases):
        name = phase["name"]
        counts = seg_counts[p]
        anomalous = int(counts[1:].sum())
        target = TARGET.get(name)
        tol = tolerance_counts_segment(labels, p * phase_len,
                                       (p + 1) * phase_len)
        entry = {"phase": name, "interval": [p * phase_len, (p + 1) * phase_len],
                 "raw_class_counts": [int(x) for x in counts.tolist()],
                 "tolerance_class_counts": [int(x) for x in tol.tolist()],
                 "anomalous_hoststeps": anomalous}
        if target is None:
            entry["gate"] = None  # baseline phase: no dominance requirement
            entry["gate_pass"] = True
        else:
            target_count = int(counts[target])
            others = {k: int(counts[v]) for k, v in TARGET.items() if v != target}
            share = float(target_count / max(anomalous, 1))
            gate_pass = (target_count >= DOMINANT_FLOOR
                         and target_count > max(others.values(), default=0)
                         and share >= SHARE_FLOOR)
            entry.update({
                "target_count": target_count,
                "other_counts": others,
                "target_share_of_anomalous": share,
                "gate_pass": gate_pass,
                "gate": {"dominant_floor": DOMINANT_FLOOR,
                         "share_floor": SHARE_FLOOR},
            })
        checks.append(entry)
    deploy_attempts = int(deploy_a.sum())
    migrate_attempts = int(migrate_a.sum())
    result = {
        "stream": str(stream_dir),
        "stream_sha256": manifest["stream_sha256"],
        "cohort": manifest["cohort"], "seed": manifest["seed"],
        "per_phase": checks,
        "all_phase_gates_pass": all(c["gate_pass"] for c in checks),
        "deployment_rejection_rate":
            float(deploy_r.sum() / max(deploy_attempts, 1)),
        "migration_rejection_rate":
            float(migrate_r.sum() / max(migrate_attempts, 1)),
        "whole_stream_raw_counts":
            [int(x) for x in np.bincount(scored.ravel(), minlength=4).tolist()],
    }
    result["rejection_gates_pass"] = (
        result["deployment_rejection_rate"] < 0.20
        and result["migration_rejection_rate"] < 0.40)
    write_json(stream_dir / "drift_analysis.json", result)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf8")


def main():
    targets = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else \
        sorted(DRIFT.glob("*_seed*_steps*"))
    for stream_dir in targets:
        if not (stream_dir / "manifest.json").is_file():
            continue
        try:
            result = analyze(stream_dir)
        except Exception as exc:  # pragma: no cover
            print(json.dumps({"stream": str(stream_dir), "error": str(exc)}))
            continue
        print(json.dumps({"stream": stream_dir.name,
                          "all_phase_gates_pass": result["all_phase_gates_pass"],
                          "rejection_gates_pass": result["rejection_gates_pass"],
                          "per_phase": [
                              {"phase": c["phase"],
                               "raw_counts": c["raw_class_counts"],
                               "target_count": c.get("target_count"),
                               "share": c.get("target_share_of_anomalous"),
                               "pass": c["gate_pass"]} for c in result["per_phase"]]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
