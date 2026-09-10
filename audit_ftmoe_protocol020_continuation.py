"""Audit P20 sampling, actual topology events, and comparable run metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from analyze_ftmoe_protocol020 import rolling_metrics, area_under, adaptation_lag
from run_ftmoe_online import sha, write_json


def detection(probability, labels):
    p, target = probability.reshape(-1), labels.reshape(-1) > 0
    positive = p >= .5
    tp = int((positive & target).sum())
    fp = int((positive & ~target).sum())
    fn = int((~positive & target).sum())
    tn = int((~positive & ~target).sum())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "pr_auc": float(average_precision_score(target, p)) if target.any() else None,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "fpr": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn,
            "positives": int(target.sum()), "samples": len(target)}


def audit(folder):
    summary = json.loads((folder / "summary.json").read_text())
    config = summary["configuration"]
    if sha(Path(config["source_checkpoint"]["path"])) != config["source_checkpoint"]["sha256"]:
        raise AssertionError("Source checkpoint changed after the run")
    manifest = json.loads((Path(config["stream"]) / "manifest.json").read_text())
    if sha(Path(config["stream"]) / "stream.npz") != config["stream_sha256"]:
        raise AssertionError("Stream changed after the run")
    with np.load(folder / "predictions.npz") as data:
        pred = {key: data[key] for key in data.files}
    updates = json.loads((folder / "updates.json").read_text())
    reference = json.loads((folder / "reference.json").read_text())
    p, labels = pred["probability"], pred["labels"]
    phase_len = manifest["phase_len"]
    positive_exposures, total_exposures = 0, 0
    for update in updates:
        ix = update["buffer_indices"]
        sampled = labels[ix]
        positive_exposures += int((sampled > 0).sum())
        total_exposures += sampled.size
        if ix and max(ix) > update["step"] - 2:
            raise AssertionError("Future label in replay")
    training_rate = positive_exposures / max(total_exposures, 1)
    weight = config["loss_v3"]["detection_weight"][1]
    event_rows = [u["topology_event"] for u in updates if u.get("topology_event")]
    result = {"method": config["method"], "folder": str(folder),
        "stream_sha256": config["stream_sha256"],
        "checkpoint_sha256": config["source_checkpoint"]["sha256"],
        "full": detection(p, labels), "raw": detection(p, pred["raw_labels"]),
        "second_half": detection(p[len(p) // 2:], labels[len(p) // 2:]),
        "phases": [], "sampling": {
            "stream_positive_rate": float((labels > 0).mean()),
            "sampled_positive_rate": training_rate, "positive_weight": weight,
            "weighted_positive_mass_proxy": weight * training_rate / max(1 - training_rate + weight * training_rate, 1e-9)},
        "topology": {"peak_experts": int(pred["expert_count"].max()),
            "additions": sum(len(e.get("added", [])) for e in event_rows),
            "retirements": sum(len(e.get("dormant", [])) for e in event_rows),
            "reactivations": sum(len(e.get("reactivated", [])) for e in event_rows),
            "decisions": [e for e in event_rows if e.get("decisions") or e.get("dormant") or e.get("reactivated")]},
        "reference": reference,
        "reference_source": reference[0].get("source", "legacy_protocol004_reference"),
        "reference_f1_drop": reference[0].get("mean", reference[0])["f1"] - reference[-1].get("mean", reference[-1])["f1"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "prediction_mean_seconds": summary["prediction_mean_seconds"],
        "frozen_parameters_unchanged": summary["frozen_parameters_unchanged"]}
    rolling = rolling_metrics(p, labels)
    result["rolling_mean"] = area_under(rolling, 100, len(labels))
    with np.load(Path(config["stream"]) / "stream.npz") as stream:
        proposed = np.einsum("tch,tcf->thf", stream["schedules"][:-1],
                             stream["demands"][:-1][..., [0, 1, 4]])
        proposed_overload = (proposed / stream["capacities"][:-1] > 1).any(-1)
        raw_normal = pred["raw_labels"] == 0
        raw_fp = (p >= .5) & raw_normal
        result["placement_diagnostic"] = {
            "proposed_overload_but_raw_normal": int((proposed_overload & raw_normal).sum()),
            "raw_false_positives": int(raw_fp.sum()),
            "raw_fp_with_proposed_overload": int((raw_fp & proposed_overload).sum())}
    for i, phase in enumerate(manifest["phases"]):
        start, end = i * phase_len, min((i + 1) * phase_len, len(labels))
        result["phases"].append({"name": phase["name"], "start": start, "end": end,
            **detection(p[start:end], labels[start:end]),
            "descriptive_lag": adaptation_lag(rolling, start, end)[0] if i else None})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"interpretation": "Development comparisons; no claim of independent statistical confirmation.",
              "runs": [audit(folder) for folder in args.runs]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    lines = ["# Protocol 020 continuation comparisons", "",
        "All numbers are development results from the common frozen S6 checkpoint. "
        "No independent confirmation has been performed. Labels use the registered one-step tolerance.", "",
        "Reference drops from legacy Protocol 004 and current P20 anchors must not be compared directly.", ""]
    for stream_name in ("drift", "stationary"):
        rows = [r for r in result["runs"] if (len(r["phases"]) > 1) == (stream_name == "drift")]
        lines.extend(["## " + stream_name, "",
            "| Run | F1 | PR-AUC | FP | FN | Births | Reference F1 drop | Reference source |",
            "|---|---:|---:|---:|---:|---:|---:|---|"])
        for row in rows:
            full = row["full"]
            lines.append("| %s | %.4f | %.4f | %d | %d | %d | %+.4f | %s |" % (
                Path(row["folder"]).name, full["f1"], full["pr_auc"], full["fp"], full["fn"],
                row["topology"]["additions"], row["reference_f1_drop"], row["reference_source"]))
        lines.append("")
    (args.out.parent / "comparison_tables.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    for run in result["runs"]:
        print(json.dumps({"run": run["folder"], "f1": run["full"]["f1"],
            "pr_auc": run["full"]["pr_auc"], "fp": run["full"]["fp"],
            "sampled_positive": run["sampling"]["sampled_positive_rate"],
            "weighted_positive_mass": run["sampling"]["weighted_positive_mass_proxy"],
            "births": run["topology"]["additions"],
            "reference_drop": run["reference_f1_drop"]}))
