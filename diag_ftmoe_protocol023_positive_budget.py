"""Protocol 023 round 2A — read-only diagnostic: where do the h=1 positives live?

The round-2A directive (§5) requires >= 30 h=1 within-regime test positives per
regime (io_first currently has 5).  Before changing any generator parameter this
script measures, on the *existing collected streams*, the quantities that decide
how many positives a regime can produce at all:

    * host-cell label counts per regime window and per phase
    * h=1 onset-target positives (the probe's own target, regime onset resource)
    * task-level cascade counts per regime cohort and per phase
    * per-task onset-resource excursion above the registered tau
    * the lifetime distribution of the onset-resource label on cascade tasks

It only reads streams and writes a JSON report; nothing is regenerated.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_ftmoe_protocol022_learnability as probe
from simulator.workload.BitbrainWorkloadProtocol023 import (REGIMES_V3,
                                                            REGISTERED_ONSET_TAU)

STREAMS = ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/diagnostics"

CLASS_OF_RESOURCE = {"cpu": 1, "ram": 2, "disk": 3}
RES_OF_INDEX = {0: "cpu", 1: "ram", 2: "disk"}

SINGLE = {"compute_first": "single_compute_first_seed700_steps1200",
          "memory_first": "single_memory_first_seed700_steps1200",
          "io_first": "single_io_first_seed700_steps1200"}
SINGLE_WINDOW = (150, 1200)          # familiar 150 + regime-only 1050

DEV = "dev_seed700_steps2880"
DEV_PHASES = {"A": (("A1_compute", 300, 720), ("A2_recur", 1800, 2160)),
              "B": (("B1_memory", 720, 1140), ("B2_recur", 2520, 2880)),
              "C": (("C1_io", 1140, 1560), ("C2_recur", 2160, 2520))}


def label_durations(mask, steps):
    """Length of each contiguous True run, row-major over cells."""
    out = []
    for host in range(mask.shape[1]):
        col = mask[:, host]
        run = 0
        for t in range(steps):
            if col[t]:
                run += 1
            elif run:
                out.append(run)
                run = 0
        if run:
            out.append(run)
    return out


def windowed_onset_positives(labels, steps, klass, window):
    """h=1 positives of the probe's onset target restricted to one window."""
    in_class = (labels == klass).astype(float)
    future = np.zeros((steps, labels.shape[1]), dtype=float)
    future[:steps - 1] = in_class[1:steps]
    y = future
    valid = (in_class[:steps] == 0)
    y = np.where(valid, y, np.nan)
    start, end = window
    cell = y[start:end]
    return {"positives": int(np.nansum(cell)),
            "valid_cells": int(np.isfinite(cell).sum()),
            "in_class_cells": int((in_class[start:end] > 0).sum()),
            "prevalence": float(np.nanmean(cell)) if np.isfinite(cell).any() else None}


def task_timeline(stream_dir):
    with np.load(stream_dir / "task_timeline.npz", allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def cohort_report(stream_dir, regime, onset_resource, klass, window):
    data = probe.load_stream(stream_dir)
    steps = data["steps"]
    labels = data["labels"]
    tl = task_timeline(stream_dir)
    demand_index = {"cpu": 0, "ram": 1, "disk": 2}[onset_resource]
    tau = float(REGISTERED_ONSET_TAU[onset_resource])

    out = {"stream": stream_dir.name, "steps": steps,
           "onset_resource": onset_resource, "onset_tau": tau,
           "window": list(window)}

    # --- host-cell level -------------------------------------------------
    win = np.zeros(steps, dtype=bool)
    win[window[0]:window[1]] = True
    per_class = {int(k): int(((labels[:steps] == k) & win[:, None]).sum())
                 for k in range(4)}
    out["host_cells_in_window"] = int(win.sum() * labels.shape[1])
    out["host_cell_label_counts"] = per_class
    out["onset_positives_h1"] = windowed_onset_positives(
        labels, steps, klass, window)

    # --- the regime's own tasks, by phase ---------------------------------
    phase_ids = tl["phase_ids"] if "phase_ids" in tl else None
    creation = tl["creation_id"]
    intervals = tl["time"]
    demand = tl["demand"]
    tasks = {}
    for i, cid in enumerate(creation):
        cid = int(cid)
        rec = tasks.setdefault(cid, {"first_interval": int(intervals[i]),
                                     "last_interval": int(intervals[i]),
                                     "peak": -1.0, "obs": 0})
        rec["first_interval"] = min(rec["first_interval"], int(intervals[i]))
        rec["last_interval"] = max(rec["last_interval"], int(intervals[i]))
        rec["peak"] = max(rec["peak"], float(demand[i, demand_index]))
        rec["obs"] += 1
    born_in_window = [c for c, r in tasks.items()
                      if window[0] <= r["first_interval"] < window[1]]
    peaks = np.array([tasks[c]["peak"] for c in born_in_window]) if born_in_window else np.zeros(0)
    out["tasks_born_in_window"] = len(born_in_window)
    out["tasks_born_in_window_above_tau"] = int((peaks > tau).sum())
    out["tasks_born_in_window_peak_median"] = float(np.median(peaks)) if peaks.size else None
    out["tasks_born_in_window_peak_p90"] = float(np.percentile(peaks, 90)) if peaks.size else None
    out["tasks_born_in_window_peak_max"] = float(peaks.max()) if peaks.size else None

    # lifetime of the onset-resource label on the host that carries it
    runs = label_durations((labels[:steps] == klass), steps)
    out["onset_label_run_lengths"] = {
        "n": len(runs),
        "median": float(np.median(runs)) if runs else None,
        "mean": float(np.mean(runs)) if runs else None,
        "p90": float(np.percentile(runs, 90)) if runs else None,
        "max": int(max(runs)) if runs else None,
        "hist": {int(k): int(sum(1 for r in runs if r == k))
                 for k in sorted(set(runs))} if runs else {},
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"kind": "round2a_positive_budget_diagnostic", "streams": {}}
    for regime in ("compute_first", "memory_first", "io_first"):
        resource = REGIMES_V3[regime]["onset_resource"]
        klass = CLASS_OF_RESOURCE[resource]
        single = STREAMS / SINGLE[regime]
        entry = {"single": cohort_report(single, regime, resource, klass,
                                         SINGLE_WINDOW),
                 "dev_phases": {}}
        dev = STREAMS / DEV
        for name, start, end in DEV_PHASES[{"compute_first": "A",
                                            "memory_first": "B",
                                            "io_first": "C"}[regime]]:
            entry["dev_phases"][name] = cohort_report(dev, regime, resource,
                                                      klass, (start, end))
        report["streams"][regime] = entry

    path = args.out / "positive_budget.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf8")
    for regime, entry in report["streams"].items():
        s = entry["single"]
        print("%-14s single[%d,%d): tasks_born=%4d above_tau=%4d h1_pos=%3d "
              "in_class_cells=%5d prev=%.4f"
              % (regime, s["window"][0], s["window"][1],
                 s["tasks_born_in_window"], s["tasks_born_in_window_above_tau"],
                 s["onset_positives_h1"]["positives"],
                 s["onset_positives_h1"]["in_class_cells"],
                 s["onset_positives_h1"]["prevalence"] or 0.0))
        for name, d in entry["dev_phases"].items():
            print("    %-12s tasks_born=%4d above_tau=%4d h1_pos=%3d "
                  "in_class_cells=%5d prev=%.4f"
                  % (name, d["tasks_born_in_window"],
                     d["tasks_born_in_window_above_tau"],
                     d["onset_positives_h1"]["positives"],
                     d["onset_positives_h1"]["in_class_cells"],
                     d["onset_positives_h1"]["prevalence"] or 0.0))
    print("written:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
