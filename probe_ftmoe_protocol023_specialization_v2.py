"""Protocol 023 round 2A — H2-v2 specialization probe (directive §7).

What H2-v2 asks
---------------
Train the instrument on one mechanism's *first exposure*, evaluate it on the
same mechanism's later intervals, and require:

    within AP - mean cross AP >= 0.05        (the registered gap rule)
    within-regime positives >= 30            (the new reliability condition)

for at least 2 of 3 regimes (3 of 3 is the ideal).

How "within" and "cross" are defined here, and why
--------------------------------------------------
The known blocker is one-directional: regime C's disk onset needs at least two
co-located disk phases, so its onset rate is far lower than A's or B's
(diagnostics/disk_onset_deficit.json).  A "cross" cell for this probe is
therefore the same target scored on intervals the model never trained on but
*without* that mechanism active -- which a single-regime stream does not
contain.

So this probe separates the two halves of the gate and never conflates them:

* **reliability** (positives >= 30) is measured on the calibrated
  single-regime streams, where the mechanism is dense: equal split of the
  regime window into a training exposure and a held-out tail, target = the
  mechanism's own onset resource, model = the closed-form logistic instrument
  of Protocol 022/023 (identical features, standardisation, Newton fit and AP).
  This is the measurement that decides whether a third specialist claim is
  supportable at all.

* **gap** (within AP - mean cross AP) is measured on the *development* stream,
  which contains all three mechanisms in separate blocks.  The train segment is
  a mechanism's first-exposure block; the within test segment is that
  mechanism's recurrence block; the cross test segments are the other two
  mechanisms' recurrence blocks.  No test block overlaps any train block, and
  no row is ever trained on and tested at once.

Both halves are computed and reported per regime; a regime passes only when
both hold.  The round-1 probe (``cross_regime_probe.json``) is not modified or
overwritten; this file is written next to it as ``cross_regime_probe_v2.json``.

Usage
-----
    python probe_ftmoe_protocol023_specialization_v2.py
    python probe_ftmoe_protocol023_specialization_v2.py --tag-suffix none
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_ftmoe_protocol022_learnability as base
import probe_ftmoe_protocol023_specialization as spec1
from simulator.workload.BitbrainWorkloadProtocol023 import (REGIMES_V3,
                                                            REGIME_IDS)

STREAMS = ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
DEV_TAG = "dev_seed700_steps2880%s"
OUT = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/specialization_v2"

CLASS_OF_RESOURCE = {"cpu": 1, "ram": 2, "disk": 3}
GATE = {"min_within_minus_cross_ap": 0.05,
        "min_regimes_passing": 2,
        "min_within_positives": 30,
        "ideal_regimes_passing": 3}
HORIZONS = (1,)
#: Reliability splits measured for every regime (see ``reliability_cells``).
RELIABILITY_SPLITS = (0.50, 0.30)
#: The registered primary split: the balanced exposure/tail reading.
PRIMARY_RELIABILITY_SPLIT = 0.50
#: Registered dev timeline (name, start, end, regime) of round 1 and round 2A.
DEV_PHASES = (("F0_baseline", 0, 300, None),
              ("A1_compute", 300, 720, "compute_first"),
              ("B1_memory", 720, 1140, "memory_first"),
              ("C1_io", 1140, 1560, "io_first"),
              ("F1_baseline", 1560, 1800, None),
              ("A2_recur", 1800, 2160, "compute_first"),
              ("C2_recur", 2160, 2520, "io_first"),
              ("B2_recur", 2520, 2880, "memory_first"))
FORBIDDEN_TOKENS = ("regime_id", "phase_id", "mechanism_id", "cascade_flag",
                    "event_id", "future", "unmatured")


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def single_stream(regime, tag_suffix):
    return STREAMS / ("single_%s_seed700_steps1200%s" % (regime, tag_suffix))


def dev_stream(tag_suffix):
    return STREAMS / (DEV_TAG % tag_suffix)


# --------------------------------------------------------------------------
# features (the round-1 instrument, reused verbatim)
# --------------------------------------------------------------------------
def build_features(data):
    groups = base.build_groups(data)
    groups.update(spec1.cross_lag_groups(data))
    keys = base.all_keys(groups)
    X, names = base.assemble(groups, keys)
    hits = [name for name in names
            if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if hits:
        raise SystemExit("a forbidden token reached the feature set: %r" % hits)
    return X, names


def fit_and_score(X, y, fit_rows, test_rows, label):
    """One cell: fit on fit_rows (flattened), score test_rows."""
    cell = {"cell": label, "features": int(X.shape[1]),
            "train_rows": int(fit_rows.sum()),
            "test_rows": int(test_rows.sum())}
    both = np.isfinite(y)
    fit_rows = fit_rows & both
    test_rows = test_rows & both
    if fit_rows.sum() < 50 or test_rows.sum() < 50:
        cell["skipped"] = "too few usable rows"
        return cell
    finite = np.isfinite(X).all(axis=1)
    fit_rows = fit_rows & finite
    test_rows = test_rows & finite
    cell["train_rows_finite"] = int(fit_rows.sum())
    cell["test_rows_finite"] = int(test_rows.sum())
    cell["dropped_nonfinite_train"] = int((fit_rows & both).sum()
                                          - fit_rows.sum())
    if y[fit_rows].min() == y[fit_rows].max() or \
            y[test_rows].min() == y[test_rows].max():
        cell["skipped"] = "single-class fit or test window"
        return cell
    mu, sd = base.standardize(X, fit_rows)
    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        raise ValueError("standardizer is not finite (NaN-poisoning guard)")
    Xs = (X - mu) / sd
    w = base.fit_logistic(Xs, y, fit_rows)
    if not np.isfinite(w).all():
        raise ValueError("fitted weights are not finite")
    score = base.predict(X, mu, sd, w)[test_rows]
    y_test = y[test_rows]
    cell["metrics"] = base.metrics(y_test, score)
    cell["prevalence"] = float(y_test.mean())
    cell["positives"] = int(y_test.sum())
    return cell


# --------------------------------------------------------------------------
# half 1: reliability on the calibrated single-regime streams
# --------------------------------------------------------------------------
def reliability_cells(data, regime, horizon, split=0.5):
    """Split one single-regime stream into an exposure head and a held-out tail.

    ``split`` is the fraction of the regime window used for training; the rest is
    the held-out test segment.  Two splits are measured for every regime and both
    are reported, because the reliability condition is a *count* condition and a
    count depends on the length of the window it is counted over:

        0.50  the balanced reading (equal exposure and evaluation)
        0.30  the "long held-out tail" reading (70 % of the window never trained
              on, which is the reading under which a rare onset resource can
              accumulate 30 positives at all)
    """
    steps = data["steps"]
    manifest = data["manifest"]
    phases = [p for p in manifest["phases"] if p.get("regime_id")]
    if len(phases) != 1:
        raise SystemExit("single-regime stream declares %d regime phases"
                         % len(phases))
    start = int(phases[0]["start"])
    end = int(min(phases[0]["end"], steps))
    middle = start + int(round((end - start) * float(split)))
    target = np.zeros(steps, dtype=bool)
    target[start:middle] = True
    test = np.zeros(steps, dtype=bool)
    test[middle:end] = True
    X, names = build_features(data)
    label = CLASS_OF_RESOURCE[REGIMES_V3[regime]["onset_resource"]]
    y = base.onset_target(data, label, horizon).reshape(-1)
    rows = y.shape[0]
    hosts = data["labels"].shape[1]
    fit_rows = np.repeat(target, hosts)[:rows]
    test_rows = np.repeat(test, hosts)[:rows]
    cell = fit_and_score(X[:rows], y, fit_rows, test_rows,
                         "reliability/%s/h=%d/split=%.2f/%s_onset"
                         % (regime, horizon, split,
                            REGIMES_V3[regime]["onset_resource"]))
    cell["window"] = {"train": [start, middle], "test": [middle, end]}
    cell["split"] = float(split)
    cell["features"] = len(names)
    return cell, {"train": [start, middle], "test": [middle, end]}


# --------------------------------------------------------------------------
# half 2: the gap on the development stream (round-1 segmentation)
# --------------------------------------------------------------------------
def dev_segments(steps):
    out = {}
    for regime in REGIME_IDS:
        blocks = [p for p in DEV_PHASES if p[3] == regime]
        if len(blocks) < 2:
            raise SystemExit("regime %s has %d block(s) in the dev timeline"
                             % (regime, len(blocks)))
        out[regime] = {"train": blocks[0], "test": blocks[1:]}
    return out


def interval_mask(steps, blocks):
    mask = np.zeros(steps, dtype=bool)
    for block in blocks:
        mask[block[1]:min(block[2], steps)] = True
    return mask


def gap_cells(data, regime, horizon):
    """Train on the first exposure, test on every recurrence block."""
    steps = data["steps"]
    segments = dev_segments(steps)
    X, names = build_features(data)
    label = CLASS_OF_RESOURCE[REGIMES_V3[regime]["onset_resource"]]
    y = base.onset_target(data, label, horizon).reshape(-1)
    hosts = data["labels"].shape[1]
    rows = min(X.shape[0], y.shape[0])
    train_mask = np.repeat(interval_mask(steps, [segments[regime]["train"]]),
                           hosts)[:rows]
    cells = {}
    for other in REGIME_IDS:
        test_mask = np.repeat(interval_mask(steps, segments[other]["test"]),
                              hosts)[:rows]
        cells[other] = fit_and_score(
            X[:rows], y[:rows], train_mask, test_mask,
            "gap/train=%s/test=%s/h=%d/%s_onset"
            % (regime, other, horizon, REGIMES_V3[regime]["onset_resource"]))
        cells[other]["train_regime"] = regime
        cells[other]["test_regime"] = other
        cells[other]["within_regime"] = bool(other == regime)
    return cells, names


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------
def run(tag_suffix, horizons=HORIZONS):
    started = time.perf_counter()
    report = {
        "protocol": "023", "round": "2A", "stage": "H2-v2",
        "instrument": ("probe_ftmoe_protocol022_learnability (Newton/IRLS "
                       "logistic, P22 feature groups) + "
                       "probe_ftmoe_protocol023_specialization cross-lag groups"),
        "tag_suffix": tag_suffix,
        "horizons": list(horizons), "primary_horizon": horizons[0],
        "gate_registration": {
            "rule": ("within AP - mean cross AP >= %.2f AND within positives "
                     ">= %d, for >= %d of %d regimes"
                     % (GATE["min_within_minus_cross_ap"],
                        GATE["min_within_positives"],
                        GATE["min_regimes_passing"], len(REGIME_IDS))),
            "directive": "§7 H2-v2",
            "reliability_source": ("calibrated single-regime streams, exposure/"
                                   "tail split of the regime window"),
            "gap_source": ("development stream: train on the mechanism's first "
                           "exposure, test on each mechanism's recurrence block"),
        },
        "reliability": {}, "gap": {}, "rows": [],
    }
    dev_path = dev_stream(tag_suffix)
    if not dev_path.is_dir():
        raise SystemExit("missing development stream %s" % dev_path)
    dev = base.load_stream(dev_path)

    for regime in REGIME_IDS:
        path = single_stream(regime, tag_suffix)
        if not path.is_dir():
            raise SystemExit("missing stream %s" % path)
        data = base.load_stream(path)
        regime_block = {"stream": path.name, "steps": data["steps"],
                        "cells": {}, "splits": {},
                        "onset_resource":
                        REGIMES_V3[regime]["onset_resource"]}
        gap_block = {"stream": dev_path.name, "steps": dev["steps"],
                     "cells": {}}
        for horizon in horizons:
            for split in RELIABILITY_SPLITS:
                cell, window = reliability_cells(data, regime, horizon, split)
                regime_block["cells"]["h=%d/split=%.2f" % (horizon, split)] = cell
                regime_block["splits"]["%.2f" % split] = window
            cells, names = gap_cells(dev, regime, horizon)
            gap_block["cells"]["h=%d" % horizon] = cells
            gap_block["n_features"] = len(names)
            within = cells[regime]
            cross = [cell for name, cell in cells.items() if name != regime]
            within_ap = (within.get("metrics") or {}).get("ap")
            cross_aps = [c.get("metrics", {}).get("ap") for c in cross]
            cross_aps = [v for v in cross_aps if v is not None]
            # reliability: the primary reading is the registered one, the other
            # is reported next to it (never instead of it)
            primary_cell = regime_block["cells"][
                "h=%d/split=%.2f" % (horizon, PRIMARY_RELIABILITY_SPLIT)]
            positives = primary_cell.get("positives")
            row = {
                "horizon": horizon, "regime": regime,
                "within_ap": within_ap,
                "mean_cross_ap": (float(np.mean(cross_aps)) if cross_aps
                                  else None),
                "within_positives": positives,
                "within_prevalence": primary_cell.get("prevalence"),
                "reliability_split": PRIMARY_RELIABILITY_SPLIT,
                "reliability_window": regime_block["splits"][
                    "%.2f" % PRIMARY_RELIABILITY_SPLIT],
                "reliability_positives_by_split": {
                    "%.2f" % split: regime_block["cells"][
                        "h=%d/split=%.2f" % (horizon, split)].get("positives")
                    for split in RELIABILITY_SPLITS},
                "reliability_ap_by_split": {
                    "%.2f" % split: (regime_block["cells"][
                        "h=%d/split=%.2f" % (horizon, split)].get("metrics")
                        or {}).get("ap")
                    for split in RELIABILITY_SPLITS},
                "cross_ap": {name: (c.get("metrics") or {}).get("ap")
                             for name, c in cells.items() if name != regime},
            }
            row["gap"] = (float(within_ap) - float(np.mean(cross_aps))
                          if within_ap is not None and cross_aps else None)
            row["gap_ok"] = bool(row["gap"] is not None
                                 and row["gap"]
                                 >= GATE["min_within_minus_cross_ap"])
            row["positives_ok"] = bool(positives is not None
                                       and int(positives)
                                       >= GATE["min_within_positives"])
            row["passes"] = bool(row["gap_ok"] and row["positives_ok"])
            report["rows"].append(row)
        report["reliability"][regime] = regime_block
        report["gap"][regime] = gap_block

    primary = [row for row in report["rows"] if row["horizon"] == horizons[0]]
    n_passing = sum(1 for row in primary if row["passes"])
    gate = {
        "thresholds": dict(GATE),
        "registration": report["gate_registration"],
        "n_regimes_passing": n_passing,
        "n_regimes_gap_ok": sum(1 for row in primary if row["gap_ok"]),
        "n_regimes_positives_ok": sum(1 for row in primary
                                      if row["positives_ok"]),
        "passed": bool(n_passing >= GATE["min_regimes_passing"]),
        "ideal_three_of_three": bool(n_passing == len(REGIME_IDS)),
        "verdict": ("specialization opportunity present (H2-v2 PASS)"
                    if n_passing >= GATE["min_regimes_passing"] else
                    "STOP-MR: the three mechanisms are not separable enough to "
                    "justify expert specialization"),
    }
    report["gate"] = gate
    report["verdict"] = gate["verdict"]
    report["stop_conditions_hit"] = [] if gate["passed"] else ["STOP-MR"]
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def write_report(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cross_regime_probe_v2.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf8")
    rows = [row for row in report["rows"]
            if row["horizon"] == report["primary_horizon"]]
    lines = ["# Protocol 023 round 2A — H2-v2 specialization probe", "",
             "Streams: `%s` (reliability) and `%s` (gap) · h=%d"
             % (report["reliability"][REGIME_IDS[0]]["stream"],
                report["gap"][REGIME_IDS[0]]["stream"],
                report["primary_horizon"]), "",
             "| regime | within AP | mean cross AP | gap | gap ok |"
             " positives@%.2f | positives@%.2f | positives ok | passes |"
             % (RELIABILITY_SPLITS[0], RELIABILITY_SPLITS[1]),
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        by_split = row["reliability_positives_by_split"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (row["regime"], _fmt(row["within_ap"]),
                        _fmt(row["mean_cross_ap"]), _fmt(row["gap"]),
                        row["gap_ok"],
                        by_split.get("%.2f" % RELIABILITY_SPLITS[0]),
                        by_split.get("%.2f" % RELIABILITY_SPLITS[1]),
                        row["positives_ok"], row["passes"]))
    gate = report["gate"]
    lines += ["", "**%d/%d regimes pass both conditions** (gap ok %d/3, "
              "positives ok %d/3); ideal 3/3. %s"
              % (gate["n_regimes_passing"], len(REGIME_IDS),
                 gate["n_regimes_gap_ok"], gate["n_regimes_positives_ok"],
                 gate["verdict"]), "",
              "Rule: %s" % gate["registration"]["rule"], "",
              "Reliability is measured on the calibrated single-regime streams; "
              "the gap is measured on the development stream. See the JSON for "
              "the per-cell windows and prevalences.", ""]
    (out_dir / "cross_regime_probe_v2.md").write_text("\n".join(lines),
                                                      encoding="utf8")
    return path


def _fmt(value):
    if value is None:
        return "—"
    try:
        return "%.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-suffix", default="_calibrated")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.tag_suffix == "none":
        args.tag_suffix = ""
    report = run(args.tag_suffix)
    path = write_report(report, args.out)
    gate = report["gate"]
    print(json.dumps({
        "written": rel(path),
        "n_regimes_passing": gate["n_regimes_passing"],
        "ideal_three_of_three": gate["ideal_three_of_three"],
        "verdict": gate["verdict"],
        "rows": [{k: row[k] for k in ("regime", "within_ap", "mean_cross_ap",
                                      "gap", "within_positives", "gap_ok",
                                      "positives_ok", "passes")}
                 for row in report["rows"]
                 if row["horizon"] == report["primary_horizon"]],
        "stop_conditions_hit": report["stop_conditions_hit"]},
        ensure_ascii=False, indent=2), flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
