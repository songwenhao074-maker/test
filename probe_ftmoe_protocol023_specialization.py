"""Protocol 023 — S3 multi-regime specialization / learnability probe (plan §13).

The question this answers
------------------------
Plan §13 registers the *specialization opportunity* gate that has to be passed
before dynamic experts (D) may even be implemented:

    train on regime R  ->  evaluate on regimes A, B, C

If a learner trained on one regime transfers to the other two almost as well as
it does within its own regime, the three mechanisms are too similar for expert
specialization to mean anything, and the honest outcome is STOP-MR rather than
a dynamic-expert experiment.

    within-regime AP - mean cross-regime AP >= 0.05   for at least 2 of 3 regimes

What is measured, and on what
-----------------------------
The probe is the closed-form logistic instrument already used by Protocol 022
(``probe_ftmoe_protocol022_learnability``): the same feature groups, the same
NaN discipline, the same Newton/IRLS fit, the same average-precision
implementation.  Reusing it is deliberate — the S3 numbers are then comparable
with Protocol 022's S4 numbers instead of being a new instrument's numbers.
Two things are added:

1. **Symmetric temporal features.**  P22's feature set encodes the compute-first
   law (RAM/disk response to a CPU rise).  A probe carrying only that law could
   not represent the memory-first or io-first orders, and its failure on B/C
   would say nothing about specialization.  ``cross_lag`` therefore adds, for
   every ordered resource pair and every registered lag of the family, the
   deviation ``ratio[lead](t - lag) - ratio[follow](t)`` — a resource whose
   earlier rise predicts the other's later rise.  All three registered orders
   are representable, and the *probe* still has to discover which one is active
   from the data, because no regime identifier is ever a feature
   (``FORBIDDEN_INPUT_TOKENS``; the probe asserts this).

2. **Per-regime onset targets.**  Regime A's onset is a CPU onset, B's is a RAM
   onset, C's is a disk onset (``onset_resource`` in the registered family).
   The target of a regime is the onset of *its own* resource: a learner that
   only ever warns about CPU rises is not specializing in B.

Registered segmentation (dev stream)
------------------------------------
The dev stream's phases map onto the three mechanisms, and each mechanism
appears twice, so the recurrence half is a genuine held-out repetition:

    A: phases A1_compute + A2_recur      (first exposure and recurrence)
    B: phases B1_memory  + B2_recur
    C: phases C1_io      + C2_recur

For each regime the *first* exposure is the training segment and the recurrence
block is the test segment (``<regime>_recur``), which keeps the evaluation
prequential in spirit: the model is tested on a later repetition of the same
mechanism and on the other two mechanisms.

Honesty rules
-------------
- A feature column with a single non-finite value is dropped by the row mask and
  the dropped count is reported per fit (P22's P22-07: a NaN-poisoned
  standardizer silently turns the probe into a constant predictor).
- Every cell reports its own prevalence and positive count, because an AP
  without the prevalence next to it is not interpretable.
- Cells that cannot be fit (no positives, no variation) are reported as such
  rather than silently omitted.
- The gate is evaluated exactly as registered, and a STOP-MR outcome is written
  as STOP-MR rather than softened.

Usage
-----
    python probe_ftmoe_protocol023_specialization.py --quick
    python probe_ftmoe_protocol023_specialization.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_ftmoe_protocol022_learnability as probe

from simulator.workload.BitbrainWorkloadProtocol023 import (REGIMES_V3,
                                                            REGIME_IDS)

STREAM_DIR = (ROOT / "artifacts/ftmoe_online/protocol_023/development_streams"
                     "/dev_seed700_steps2880")
OUT_DIR = ROOT / "artifacts/ftmoe_online/protocol_023/specialization"

# label index per resource, as the replay labels them (1 CPU, 2 RAM, 3 disk)
CLASS_OF_RESOURCE = {"cpu": 1, "ram": 2, "disk": 3}
RESOURCE_OF_LABEL = {1: "cpu", 2: "ram", 3: "disk"}

# Registered phase plan of the dev stream (name, start, end, kind).
# ``kind`` is "familiar" or the regime the phase activates.
PHASE_PLAN = ((("F0_baseline", 300, "familiar"), 0),
              (("A1_compute", 420, "compute_first"), 300),
              (("B1_memory", 420, "memory_first"), 720),
              (("C1_io", 420, "io_first"), 1140),
              (("F1_baseline", 240, "familiar"), 1560),
              (("A2_recur", 360, "compute_first"), 1800),
              (("C2_recur", 360, "io_first"), 2160),
              (("B2_recur", 360, "memory_first"), 2520))

REGISTERED_STEPS = 2880
HORIZONS = (1, 4)
GATE = {
    "min_within_minus_cross_ap": 0.05,
    "min_regimes_passing": 2,
}

FORBIDDEN_TOKENS = ("regime_id", "phase_id", "mechanism_id", "cascade_flag",
                    "event_id", "future", "unmatured")


# --------------------------------------------------------------------------
# stream / segmentation
# --------------------------------------------------------------------------
def phase_bounds():
    out = []
    for (name, length, kind), start in PHASE_PLAN:
        out.append({"name": name, "kind": kind, "start": int(start),
                    "end": int(start + length), "length": int(length)})
    return out


def regime_segments(phases):
    """regime -> {"train": [phases], "test": [phases]} (first vs recurrence)."""
    out = {}
    for regime in REGIME_IDS:
        blocks = [p for p in phases if p["kind"] == regime]
        if len(blocks) < 2:
            raise ValueError("regime %s has %d block(s); the registered dev "
                             "timeline needs a first exposure and a recurrence"
                             % (regime, len(blocks)))
        out[regime] = {"train": [blocks[0]], "test": blocks[1:]}
    return out


def cross_lag_groups(data):
    """Registered-order features: ratio(lead) at t-lag minus ratio(follow) at t.

    Every ordered pair of distinct resources and every lag the family registers
    is included, so each of the three registered orders is representable.
    """
    ratios = probe.ratios_of(data)
    index = {"cpu": 0, "ram": 1, "disk": 2}
    lags = set()
    for regime in REGIME_IDS:
        for resource, lag in REGIMES_V3[regime]["sequence"]:
            if lag > 0:
                lags.add(int(lag))
        for key, window in REGIMES_V3[regime]["response_windows"].items():
            lags.add(int(window[0]))
    out = {}
    for lead in index:
        for follow in index:
            if lead == follow:
                continue
            for lag in sorted(lags):
                if lag <= 0:
                    continue
                lead_ratio = ratios[:, :, index[lead]]
                follow_ratio = ratios[:, :, index[follow]]
                delta = np.full(lead_ratio.shape, np.nan)
                delta[lag:] = lead_ratio[:-lag] - follow_ratio[lag:]
                out[("cross_lag", "%s_lead%d_%s" % (lead, lag, follow))] = delta
    return out


def mask_rows(*arrays):
    """(kept mask, dropped count) over rows where every array is finite."""
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for value in arrays:
        mask &= np.isfinite(np.asarray(value, dtype=np.float64))
    return mask, int((~mask).sum())


def interval_mask(steps, phases):
    out = np.zeros(steps, dtype=bool)
    for phase in phases:
        out[phase["start"]:phase["end"]] = True
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_cell(X, y, names, train_mask, test_mask, label):
    """Fit on ``train_mask`` (rows), test on ``test_mask``; report the cell."""
    both = np.isfinite(y)
    fit_rows = train_mask & both
    test_rows = test_mask & both
    cell = {"cell": label, "features": len(names),
            "train_rows": int(fit_rows.sum()), "test_rows": int(test_rows.sum())}
    if fit_rows.sum() < 50 or test_rows.sum() < 50:
        cell["skipped"] = "too few usable rows"
        return cell
    finite = np.isfinite(X).all(axis=1)
    fit_rows = fit_rows & finite
    test_rows = test_rows & finite
    cell["train_rows_finite"] = int(fit_rows.sum())
    cell["test_rows_finite"] = int(test_rows.sum())
    cell["dropped_nonfinite_train"] = int((train_mask & both).sum()
                                          - fit_rows.sum())
    ys = y[fit_rows]
    if ys.min() == ys.max() or y[test_rows].min() == y[test_rows].max():
        cell["skipped"] = "single-class fit or test window"
        return cell
    mu, sd = probe.standardize(X, fit_rows)
    if not (np.isfinite(mu).all() and np.isfinite(sd).all()):
        raise ValueError("standardizer is not finite (NaN-poisoning guard)")
    # exactly the Protocol-022 convention: standardize once from the fit split,
    # fit on the standardized matrix, predict through (X - mu)/sd
    Xs = (X - mu) / sd
    w = probe.fit_logistic(Xs, y, fit_rows)
    if not np.isfinite(w).all():
        raise ValueError("fitted weights are not finite")
    score = probe.predict(X, mu, sd, w)[test_rows]
    y_test = y[test_rows]
    metrics = probe.metrics(y_test, score)
    cell["metrics"] = metrics
    cell["prevalence"] = float(y_test.mean())
    cell["positives"] = int(y_test.sum())
    return cell


def regime_targets(data):
    """regime -> (resource, 4-class label index, onset target builder)."""
    out = {}
    for regime in REGIME_IDS:
        resource = REGIMES_V3[regime]["onset_resource"]
        out[regime] = {"resource": resource,
                       "label": CLASS_OF_RESOURCE[resource]}
    return out


def run_probe(stream_dir, horizons=HORIZONS, quick=False):
    started = time.perf_counter()
    data = probe.load_stream(stream_dir)
    steps = data["steps"]
    if steps != REGISTERED_STEPS:
        raise ValueError("unregistered dev-stream horizon: %d (registered %d)"
                         % (steps, REGISTERED_STEPS))
    phases = phase_bounds()
    segments = regime_segments(phases)
    targets = regime_targets(data)

    groups = probe.build_groups(data)
    groups.update(cross_lag_groups(data))
    keys = probe.all_keys(groups)
    X, names = probe.assemble(groups, keys)
    hits = [name for name in names
            if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if hits:
        raise ValueError("a forbidden token reached the feature set: %r" % hits)

    report = {
        "protocol": "023",
        "stage": "P23-S3",
        "kind": "multi_regime_specialization_probe",
        "instrument": "probe_ftmoe_protocol022_learnability (closed-form "
                      "logistic, reused for comparability)",
        "stream": str(stream_dir.relative_to(ROOT)).replace("\\", "/"),
        "steps": steps,
        "horizons": list(horizons),
        "n_features": len(names),
        "features": names,
        "cross_lag_features": [n for n in names
                               if n.startswith(("cpu_lead", "ram_lead",
                                                "disk_lead"))],
        "phases": phases,
        "segments": {r: {"train": [p["name"] for p in v["train"]],
                         "test": [p["name"] for p in v["test"]]}
                     for r, v in segments.items()},
        "onset_resource_by_regime": {r: t["resource"]
                                     for r, t in targets.items()},
        "forbidden_input_check": "passed (no forbidden token in a feature name)",
        "quick": bool(quick),
        "cells": [],
    }

    for regime in REGIME_IDS:
        train_mask = interval_mask(steps, segments[regime]["train"])
        test_masks = {other: interval_mask(steps, segments[other]["test"])
                      for other in REGIME_IDS}
        for horizon in horizons:
            klass = targets[regime]["label"]
            y = probe.onset_target(data, klass, horizon)
            y_rows = y.reshape(-1)
            X_rows, _ = probe.assemble(groups, keys)
            y_fit_mask = np.repeat(train_mask, y.shape[1])
            for other in REGIME_IDS:
                test_mask = np.repeat(test_masks[other], y.shape[1])
                cell = evaluate_cell(
                    X_rows, y_rows, names, y_fit_mask, test_mask,
                    "train=%s/test=%s/h=%d/%s_onset"
                    % (regime, other, horizon, targets[regime]["resource"]))
                cell["train_regime"] = regime
                cell["test_regime"] = other
                cell["horizon"] = horizon
                cell["target"] = "%s_onset" % targets[regime]["resource"]
                cell["within_regime"] = bool(other == regime)
                report["cells"].append(cell)

    gate = specialization_gate(report, horizons)
    report["specialization_gate"] = gate
    report["elapsed_seconds"] = time.perf_counter() - started
    report["stop_conditions_hit"] = [] if gate["passed"] else ["STOP-MR"]
    return report


def _ap(cell):
    metrics = cell.get("metrics") or {}
    value = metrics.get("ap")
    return None if value is None else float(value)


def specialization_gate(report, horizons=HORIZONS):
    """Plan §13: within-regime AP minus mean cross-regime AP >= 0.05, 2 of 3."""
    per_regime, gaps = {}, []
    for horizon in horizons:
        for regime in REGIME_IDS:
            cells = [c for c in report["cells"]
                     if c.get("train_regime") == regime
                     and c.get("horizon") == horizon]
            within = [c for c in cells if c.get("within_regime")]
            cross = [c for c in cells if not c.get("within_regime")]
            within_ap = _ap(within[0]) if within else None
            cross_aps = [_ap(c) for c in cross]
            cross_aps = [v for v in cross_aps if v is not None]
            entry = {"horizon": horizon, "regime": regime,
                     "within_ap": within_ap,
                     "cross_ap": {c.get("test_regime"): _ap(c) for c in cross},
                     "mean_cross_ap": (float(np.mean(cross_aps))
                                       if cross_aps else None),
                     "within_skipped": (within[0].get("skipped")
                                        if within else "no cell")}
            if within_ap is not None and cross_aps:
                entry["gap"] = float(within_ap - np.mean(cross_aps))
                entry["passes"] = bool(entry["gap"]
                                       >= GATE["min_within_minus_cross_ap"])
            else:
                entry["gap"] = None
                entry["passes"] = False
            per_regime[(horizon, regime)] = entry
            gaps.append(entry)
    for horizon in horizons:
        passing = [e for e in gaps
                   if e["horizon"] == horizon and e.get("passes")]
        report_key = "horizon_%d" % horizon
        if report_key not in report:
            report[report_key] = {}
        report[report_key]["n_regimes_passing"] = len(passing)
        report[report_key]["regimes_passing"] = sorted(e["regime"]
                                                       for e in passing)
    primary = [e for e in gaps if e["horizon"] == horizons[0]]
    n_passing = sum(1 for e in primary if e.get("passes"))
    return {
        "registration": {"gate": "plan §13",
                         "rule": "within-regime AP - mean cross-regime AP >= "
                                 "%.2f for at least %d of %d regimes"
                                 % (GATE["min_within_minus_cross_ap"],
                                    GATE["min_regimes_passing"],
                                    len(REGIME_IDS)),
                         "primary_horizon": horizons[0]},
        "thresholds": dict(GATE),
        "per_regime": [{"horizon": e["horizon"], "regime": e["regime"],
                        "within_ap": e["within_ap"],
                        "mean_cross_ap": e["mean_cross_ap"],
                        "gap": e["gap"], "passes": e["passes"]}
                       for e in gaps],
        "n_regimes_passing": int(n_passing),
        "passed": bool(n_passing >= GATE["min_regimes_passing"]),
        "verdict": ("specialization opportunity present"
                    if n_passing >= GATE["min_regimes_passing"]
                    else "STOP-MR: the three mechanisms are not separable "
                         "enough to justify expert specialization"),
    }


def write_report(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cross_regime_probe.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                         + "\n", encoding="utf8")
    lines = ["# Protocol 023 — S3 cross-regime specialization probe", "",
             "Stream: `%s` · steps %d · features %d · horizon(s) %s"
             % (report["stream"], report["steps"], report["n_features"],
                report["horizons"]), "",
             "Onset resource per regime: `%s`" % report["onset_resource_by_regime"],
             "", "## Cells (train -> test)", "",
             "| train | test | h | target | AP | ROC-AUC | FPR | prevalence | positives |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        metrics = cell.get("metrics") or {}
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |"
                     % (cell.get("train_regime"), cell.get("test_regime"),
                        cell.get("horizon", 0), cell.get("target"),
                        _fmt(metrics.get("ap")), _fmt(metrics.get("roc_auc")),
                        _fmt(metrics.get("fpr_at_prevalence")),
                        _fmt(cell.get("prevalence")), cell.get("positives")))
    gate = report["specialization_gate"]
    lines += ["", "## §13 specialization gate", "",
              "Rule: %s" % gate["registration"]["rule"], "",
              "| regime | within AP | mean cross AP | gap | passes |",
              "|---|---:|---:|---:|---|"]
    for entry in gate["per_regime"]:
        if entry["horizon"] != report["horizons"][0]:
            continue
        lines.append("| %s | %s | %s | %s | %s |"
                     % (entry["regime"], _fmt(entry["within_ap"]),
                        _fmt(entry["mean_cross_ap"]), _fmt(entry["gap"]),
                        entry["passes"]))
    lines += ["", "**%d/%d regimes pass at h=%d.** %s"
              % (gate["n_regimes_passing"], len(REGIME_IDS),
                 report["horizons"][0], gate["verdict"]), ""]
    (out_dir / "cross_regime_probe.md").write_text("\n".join(lines),
                                                   encoding="utf8")
    return json_path


def _fmt(value):
    if value is None:
        return "—"
    try:
        return "%.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=STREAM_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="horizon 1 only (the registered primary horizon)")
    parser.add_argument("--horizons", type=str, default=None)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "3"
    horizons = HORIZONS
    if args.horizons:
        horizons = tuple(int(v) for v in args.horizons.split(","))
    elif args.quick:
        horizons = (1,)
    report = run_probe(args.stream, horizons=horizons, quick=args.quick)
    path = write_report(report, args.out)
    gate = report["specialization_gate"]
    print(json.dumps({"written": str(path.relative_to(ROOT)).replace("\\", "/"),
                      "n_regimes_passing": gate["n_regimes_passing"],
                      "verdict": gate["verdict"],
                      "stop_conditions_hit": report["stop_conditions_hit"],
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
