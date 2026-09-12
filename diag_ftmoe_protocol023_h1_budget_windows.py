"""Protocol 023 round 2A — h=1 positive budget vs test-window definition.

Directive §7 (H2-v2) requires ``within-regime positives >= 30`` per regime at
h=1.  The round-1 probe's test segment is the recurrence block alone (360
intervals), where io-first has 5 positives.  This script measures the same
quantity under every test-window definition that is *prequential-legal* (a test
window may never overlap the training segment), so the reviewer can see whether
the deficit is a definition artefact or a data property:

    recurrence   : the regime's ``*_recur`` block only (round-1 definition)
    tail         : every interval after the regime's own training block
    heldout_tail : tail, minus any interval belonging to another regime's
                   training block (i.e. only intervals the model has never
                   trained on for any regime)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import probe_ftmoe_protocol022_learnability as probe

STREAM = (ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
          "/dev_seed700_steps2880")
OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/data_calibration"
CLASS_OF = {"compute_first": 1, "memory_first": 2, "io_first": 3}

# registered timeline (name, start, end, kind)
PHASES = (("F0_baseline", 0, 300, "familiar"),
          ("A1_compute", 300, 720, "compute_first"),
          ("B1_memory", 720, 1140, "memory_first"),
          ("C1_io", 1140, 1560, "io_first"),
          ("F1_baseline", 1560, 1800, "familiar"),
          ("A2_recur", 1800, 2160, "compute_first"),
          ("C2_recur", 2160, 2520, "io_first"),
          ("B2_recur", 2520, 2880, "memory_first"))


def onset_matrix(labels, steps, klass):
    """The probe's own h=1 onset target (NaN where not valid)."""
    in_class = (labels == klass).astype(float)
    future = np.zeros((steps, labels.shape[1]), dtype=float)
    future[:steps - 1] = in_class[1:steps]
    valid = (in_class[:steps] == 0)
    return np.where(valid, future, np.nan)


def masks_for(regime, steps):
    first = next(p for p in PHASES if p[3] == regime)
    train = np.zeros(steps, dtype=bool)
    train[first[1]:first[2]] = True
    recur = np.zeros(steps, dtype=bool)
    for p in PHASES:
        if p[3] == regime and p[1] >= first[2]:
            recur[p[1]:p[2]] = True
    tail = np.zeros(steps, dtype=bool)
    tail[first[2]:] = True
    others_train = np.zeros(steps, dtype=bool)
    for p in PHASES:
        if p[3] != regime and p[3] != "familiar" and p[1] < first[2]:
            others_train[p[1]:p[2]] = True
    heldout = tail & ~others_train
    return {"train": train, "recurrence": recur, "tail": tail,
            "heldout_tail": heldout, "other_regime_train_blocks": others_train}


def measure(data, regime, masks, windows):
    steps = data["steps"]
    labels = data["labels"]
    y = onset_matrix(labels, steps, CLASS_OF[regime])[:steps]
    out = {}
    for name, m in windows.items():
        cell = y[m]
        pos = int(np.nansum(cell))
        valid = int(np.isfinite(cell).sum())
        # positives whose future onset actually falls inside the same window
        out[name] = {
            "intervals": int(m.sum()),
            "host_cells": int(valid),
            "positives": pos,
            "prevalence": float(np.nanmean(cell)) if valid else None,
        }
    # overlap check: a test cell whose onset lands at t+1 outside the window
    for name in ("recurrence", "heldout_tail"):
        m = windows[name]
        outside = 0
        idx = np.nonzero(m)[0]
        for t in idx:
            if t + 1 >= steps:
                continue
            target = labels[t + 1] == CLASS_OF[regime]
            if (labels[t] == CLASS_OF[regime]).any():
                continue
            if target.any() and not m[t + 1]:
                outside += int(target.sum())
        out[name]["onsets_landing_outside_window"] = outside
    return out


def main():
    data = probe.load_stream(STREAM)
    steps = data["steps"]
    report = {"kind": "round2a_h1_positive_budget_by_window",
              "stream": STREAM.name, "steps": steps,
              "definition_note": ("the round-1 probe counts positives over "
                                  "flattened (interval x host) cells of the "
                                  "regime's onset target at h=1"),
              "regimes": {}}
    for regime in ("compute_first", "memory_first", "io_first"):
        masks = masks_for(regime, steps)
        windows = {k: masks[k] for k in ("recurrence", "tail", "heldout_tail")}
        res = measure(data, regime, masks, windows)
        res["train_intervals"] = int(masks["train"].sum())
        report["regimes"][regime] = res
        print("=== %s (train %d intervals) ===" % (regime, masks["train"].sum()))
        for name, e in res.items():
            if not isinstance(e, dict):
                continue
            print("  %-14s intervals=%4d host_cells=%5d positives=%3d "
                  "prevalence=%.5f%s"
                  % (name, e["intervals"], e["host_cells"], e["positives"],
                     e["prevalence"] or 0.0,
                     ("  onsets_outside=%d" % e["onsets_landing_outside_window"])
                     if "onsets_landing_outside_window" in e else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h1_positive_budget_by_window.json").write_text(
        json.dumps(report, indent=2), encoding="utf8")
    print("written:", (OUT / "h1_positive_budget_by_window.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
