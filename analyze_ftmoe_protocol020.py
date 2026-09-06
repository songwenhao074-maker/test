"""Protocol 020 S7-S9 — online run analysis (phase-aware) + adaptation lag.

Reads runner predictions/updates + the protocol-020 stream manifest and
reports, per method run:

- per-phase detection metrics (tolerance labels, threshold 0.5) incl. PR-AUC,
  raw class counts, expert count / routing stats;
- rolling (window 100, stride 10) F1 and PR-AUC series;
- adaptation lag per phase transition (plan §34): first rolling end whose F1
  reaches 95% of the late-phase median and holds 3 of the next 5 samples;
- area under the rolling F1/PR-AUC series (first/second half) for online
  prequential comparisons (plan §36);
- topology events (D): expert additions / dormant / reactivated with steps.

Stationary streams (manifest kind == stationary) have no phase transitions;
their per-phase block equals the whole stream.

Usage:
    python analyze_ftmoe_protocol020.py --stream <stream_dir> \
        --runs A=<run_dir>,C=<run_dir>[,D=<run_dir>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from train_ftmoe_end_to_end import metric_arrays  # noqa: E402

WINDOW = 100
STRIDE = 10


def rolling_metrics(probability, labels):
    rows = []
    for end in range(WINDOW, len(labels) + 1, STRIDE):
        p = probability[end - WINDOW:end].reshape(-1)
        y = labels[end - WINDOW:end].reshape(-1)
        anomaly = y > 0
        pred = p >= .5
        tp = int((pred & anomaly).sum())
        fp = int((pred & ~anomaly).sum())
        fn = int((~pred & anomaly).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        pr_auc = float(average_precision_score(anomaly, p)) if anomaly.any() else 0.0
        rows.append({"end": end, "f1": f1, "pr_auc": pr_auc})
    return rows


def adaptation_lag(rolling, boundary, target_window=(150, 250), ratio=0.95):
    """Plan §34: first rolling end >= boundary reaching 95% of the
    late-phase median (windows ending within boundary+[150,250]) and holding
    for 3 of the next 5 rolling points."""
    late = [r["f1"] for r in rolling
            if boundary + target_window[0] <= r["end"] <= boundary + target_window[1]]
    if not late:
        return None, None
    target = float(np.median(late))
    level = ratio * target
    after = [r for r in rolling if r["end"] >= boundary]
    for i, r in enumerate(after):
        horizon = after[i:i + 5]
        if len(horizon) >= 3 and sum(1 for h in horizon if h["f1"] >= level) >= 3:
            return r["end"], target
    return None, target


def area_under(rolling, start, stop):
    values = [r for r in rolling if start <= r["end"] <= stop]
    if not values:
        return None
    return {"f1": float(np.trapz([v["f1"] for v in values], dx=STRIDE) / max(len(values), 1)),
            "pr_auc": float(np.trapz([v["pr_auc"] for v in values], dx=STRIDE)
                            / max(len(values), 1))}


def analyze(stream_dir: Path, run_dirs: dict):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    steps = manifest["steps"]
    kind = manifest.get("kind", "drift")
    phase_len = int(manifest["phase_len"])
    phases = manifest["phases"]
    transitions = manifest.get("transitions", [])
    boundaries = [t["interval"] for t in transitions]
    phase_rows = []
    for p, phase in enumerate(phases):
        a, b = p * phase_len, min((p + 1) * phase_len, steps)
        phase_rows.append({"phase": p, "name": phase["name"],
                           "scales": {k: phase[k] for k in
                                      ("cpu_scale", "ram_scale", "disk_scale")},
                           "start": a, "end": b})
    result = {"stream": str(stream_dir), "kind": kind,
              "manifest_seed": manifest["seed"], "cohort": manifest["cohort"],
              "phase_len": phase_len, "transitions": transitions,
              "per_phase_definitions": phase_rows, "runs": {}}
    for label, folder in run_dirs.items():
        with np.load(folder / "predictions.npz") as pred:
            probability = pred["probability"]
            class_probability = pred["class_probability"]
            labels = pred["labels"]
            raw_labels = pred["raw_labels"]
            expert_count = pred["expert_count"]
            unmatched = pred["unmatched_ratio"]
        rolling = rolling_metrics(probability[:steps], labels[:steps])
        run = {"per_phase": [], "rolling": rolling,
               "expert_count_min": int(expert_count.min()),
               "expert_count_max": int(expert_count.max()),
               "unmatched_ratio_mean": float(unmatched.mean()),
               "unmatched_ratio_max": float(unmatched.max())}
        for row in phase_rows:
            a, b = row["start"], row["end"]
            metrics = metric_arrays(probability[a:b].reshape(-1),
                                    class_probability[a:b].reshape(-1, 3),
                                    labels[a:b].reshape(-1))
            raw_counts = np.bincount(raw_labels[a:b].ravel(), minlength=4)
            run["per_phase"].append({
                "phase": row["phase"], "name": row["name"], "start": a, "end": b,
                "f1": metrics["f1"], "precision": metrics["precision"],
                "recall": metrics["recall"], "pr_auc": metrics["pr_auc"],
                "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
                "tn": metrics["tn"], "positives": int((labels[a:b] > 0).sum()),
                "raw_class_counts": raw_counts.tolist(),
                "expert_count_mean": float(expert_count[a:b].mean()),
                "unmatched_mean": float(unmatched[a:b].mean()),
            })
        lag_rows = []
        for boundary in boundaries:
            lag, target = adaptation_lag(rolling, boundary)
            lag_rows.append({"boundary": boundary, "lag_interval": lag,
                             "target_median_f1": (round(target, 4)
                                                  if target is not None else None)})
        run["adaptation_lags"] = lag_rows
        half = steps // 2
        run["area_under_rolling"] = {
            "first_half": area_under(rolling, WINDOW, half),
            "second_half": area_under(rolling, half, steps),
            "full": area_under(rolling, WINDOW, steps),
        }
        updates = json.loads((folder / "updates.json").read_text(encoding="utf8"))
        events = [u for u in updates if u.get("topology_event")]
        run["topology_events"] = [{
            "step": u["step"], "update_number": u["update_number"],
            "before": e.get("before"), "after": e.get("after"),
            "added": e.get("added", []), "dormant": e.get("dormant", []),
            "reactivated": e.get("reactivated", []), "decisions": e.get("decisions", []),
            "unmatched": e.get("unmatched"), "ema": round(float(e.get("ema", 0)), 4)}
            for u in events for e in [u["topology_event"]]]
        result["runs"][label] = run
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--runs", type=str, required=True,
                        help="comma-separated label=dir pairs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_dirs = {}
    for pair in args.runs.split(","):
        label, folder = pair.split("=", 1)
        run_dirs[label] = Path(folder)
    report = analyze(args.stream, run_dirs)
    out = args.out or (args.stream / "p20_analysis.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf8")
    for label, run in report["runs"].items():
        print(f"== {label}  experts {run['expert_count_min']}-{run['expert_count_max']} "
              f"unmatched mean {run['unmatched_ratio_mean']:.4f}")
        for row in run["per_phase"]:
            print("   phase %d %-5s F1 %.4f PR %.4f pos %d raw%s" % (
                row["phase"], row["name"], row["f1"], row["pr_auc"],
                row["positives"], row["raw_class_counts"]))
        for lag in run["adaptation_lags"]:
            print("   boundary %d lag %s (target %.4f)" % (
                lag["boundary"], lag["lag_interval"], lag["target_median_f1"]))
        area = run["area_under_rolling"]
        print("   area-under-rolling 2nd-half f1 %s pr %s" % (
            area["second_half"]["f1"] if area["second_half"] else None,
            area["second_half"]["pr_auc"] if area["second_half"] else None))
        print("   topology events:", len(run["topology_events"]))
    print("written:", out)
