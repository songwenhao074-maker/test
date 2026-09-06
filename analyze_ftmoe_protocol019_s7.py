"""Protocol 019 S7 — drift-stream analysis (per-phase metrics + adaptation lag).

Reads run predictions/updates + stream phase_ids and reports:
- per-phase detection metrics (tolerance labels, threshold 0.5) and class
  event counts for every method run
- rolling-100 F1 series (stride 10) per method
- adaptation lag per phase transition: first rolling point that reaches
  95% of that phase's late-phase median rolling F1
- expert-count / unmatched-ratio trajectory and topology events for D
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from train_ftmoe_end_to_end import metric_arrays  # noqa: E402


def rolling_f1(probability, labels, window=100, stride=10):
    values = []
    for end in range(window, len(labels) + 1, stride):
        p = probability[end - window:end].reshape(-1)
        y = labels[end - window:end].reshape(-1)
        anomaly = y > 0
        pred = p >= .5
        tp = int((pred & anomaly).sum())
        fp = int((pred & ~anomaly).sum())
        fn = int((~pred & anomaly).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1) if (tp + fp + fn) else 0.0
        values.append({"end": end, "f1": f1})
    return values


def adaptation_lag(rolling, boundary, target_median_window=(150, 250),
                   threshold_ratio=0.95):
    """First rolling end >= boundary+window whose f1 reaches 95% of the
    late-phase median and stays there for 3 of the next 5 samples."""
    after = [r for r in rolling if r["end"] >= boundary]
    late = [r["f1"] for r in rolling if boundary + target_median_window[0] <= r["end"]
            <= boundary + target_median_window[1]]
    if not late:
        return None, None
    target = float(np.median(late))
    level = threshold_ratio * target
    for i, r in enumerate(after):
        horizon = after[i:i + 5]
        if len(horizon) >= 3 and sum(1 for h in horizon if h["f1"] >= level) >= 3:
            return r["end"], target
    return None, target


def analyze(stream_dir: Path, run_dirs: dict):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    with np.load(stream_dir / "stream.npz") as data:
        phase_ids = data["phase_ids"][: manifest["steps"]]
    phase_len = manifest["phase_len"]
    schedule = manifest["schedule"]
    transitions = manifest.get("phase_transitions", [])
    boundaries = [t["interval"] for t in transitions]
    per_phase_rows = []
    for p, name in enumerate(schedule):
        a, b = p * phase_len, min((p + 1) * phase_len, manifest["steps"])
        per_phase_rows.append({"phase": p, "name": name, "start": a, "end": b})

    result = {"stream": str(stream_dir), "manifest_seed": manifest["seed"],
              "schedule": schedule, "phase_len": phase_len,
              "transitions": transitions, "runs": {}}
    for label, folder in run_dirs.items():
        with np.load(folder / "predictions.npz") as pred:
            probability = pred["probability"]
            class_probability = pred["class_probability"]
            labels = pred["labels"]
            raw_labels = pred["raw_labels"]
            expert_count = pred["expert_count"]
            unmatched = pred["unmatched_ratio"]
        run = {"per_phase": [], "rolling": rolling_f1(probability, labels),
               "expert_count_min": int(expert_count.min()),
               "expert_count_max": int(expert_count.max()),
               "unmatched_ratio_mean": float(unmatched.mean()),
               "unmatched_ratio_max": float(unmatched.max())}
        for row in per_phase_rows:
            a, b = row["start"], row["end"]
            metrics = metric_arrays(probability[a:b].reshape(-1),
                                    class_probability[a:b].reshape(-1, 3),
                                    labels[a:b].reshape(-1))
            raw_counts = np.bincount(raw_labels[a:b].ravel(), minlength=4)
            phase_slice = {
                "phase": row["phase"], "name": row["name"],
                "f1": metrics["f1"], "precision": metrics["precision"],
                "recall": metrics["recall"], "pr_auc": metrics["pr_auc"],
                "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
                "tn": metrics["tn"],
                "positives": int((labels[a:b] > 0).sum()),
                "raw_class_counts": raw_counts.tolist(),
                "expert_count_mean": float(expert_count[a:b].mean()),
                "unmatched_mean": float(unmatched[a:b].mean()),
            }
            run["per_phase"].append(phase_slice)
        # adaptation lag per transition boundary (requires 100+ interval lookback)
        lag_rows = []
        for boundary in boundaries:
            lag, target = adaptation_lag(run["rolling"], boundary)
            lag_rows.append({"boundary": boundary, "lag_interval": lag,
                             "target_median_f1": (round(target, 4) if target is not None else None)})
        run["adaptation_lags"] = lag_rows
        # topology events (D)
        updates = json.loads((folder / "updates.json").read_text(encoding="utf8"))
        events = [u for u in updates if u.get("topology_event")]
        run["topology_events"] = [{
            "step": u["step"], "update_number": u["update_number"],
            "before": e["before"], "after": e["after"],
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
    args = parser.parse_args()
    run_dirs = {}
    for pair in args.runs.split(","):
        label, folder = pair.split("=", 1)
        run_dirs[label] = Path(folder)
    report = analyze(args.stream, run_dirs)
    out = args.stream / "s7_analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    for label, run in report["runs"].items():
        print(f"== {label}  experts {run['expert_count_min']}-{run['expert_count_max']} "
              f"unmatched mean {run['unmatched_ratio_mean']:.4f}")
        for row in run["per_phase"]:
            print("   phase %d %-5s F1 %.4f PR %.4f pos %d raw%s" % (
                row["phase"], row["name"], row["f1"], row["pr_auc"], row["positives"],
                row["raw_class_counts"]))
        for lag in run["adaptation_lags"]:
            print("   boundary %d lag %s (target %.4f)" % (
                lag["boundary"], lag["lag_interval"], lag["target_median_f1"]))
        print("   topology events:", len(run["topology_events"]))
    print("written:", out)
