"""Read-only reanalysis of saved P23 predictions; never changes old artifacts."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/ftmoe_online/protocol_023/round2a/fixed_c_prequential"
OUTPUT = ROOT / "artifacts/ftmoe_online/analysis_20260914/results_audit.json"


def ap(y, scores):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1)
    if not y.size or y.min() == y.max():
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = y[order]
    return float((np.cumsum(ordered) / np.arange(1, len(y) + 1) * ordered).sum() / y.sum())


def temporal_onset(probability, labels):
    # h=1; only same-host transitions with both endpoints inside the phase.
    assert probability.ndim == labels.ndim == 2
    assert probability.shape == labels.shape
    valid = (labels[:-1] == 0) & (labels[1:] >= 0)
    target = labels[1:] > 0
    return {"ap": ap(target[valid], probability[:-1][valid]),
            "positives": int(target[valid].sum()), "rows": int(valid.sum())}


def legacy_flat_onset(probability, labels):
    known = labels.reshape(-1) >= 0
    probability = probability.reshape(-1)[known]
    binary = (labels.reshape(-1)[known] > 0).astype(np.int64)
    future = np.zeros_like(binary)
    future[:-1] = binary[1:]
    valid = binary == 0
    return ap(future[valid], probability.reshape(-1)[valid])


def main():
    report = {"source": str(SOURCE.relative_to(ROOT)),
              "scope": "Saved development seed 700 predictions only; no training or model selection",
              "onset_correction": "h=1 along time for the same host; omit unobserved phase-tail future; report raw and tolerance labels separately",
              "arms": {}, "late_detection_gain_C_minus_A": []}
    for arm in ("A", "C"):
        directory = SOURCE / ("arm_" + arm)
        with np.load(directory / "predictions.npz", allow_pickle=False) as data:
            p, labels, raw = (data[key] for key in ("probability", "labels", "raw_labels"))
        metrics = json.loads((directory / "phase_metrics.json").read_text(encoding="utf8"))
        rows = []
        for phase in metrics["phases"]:
            start, end = phase["intervals"]
            lo, hi = phase["late"]["intervals"]
            known = labels[lo:hi] >= 0
            measured = ap(labels[lo:hi][known] > 0, p[lo:hi][known])
            stored = phase["late"]["detection"]["pr_auc"]
            assert measured == stored or abs(measured - stored) < 1e-12
            legacy = legacy_flat_onset(p[start:end], labels[start:end])
            stored_onset = phase["whole"]["onset"]["ap"]
            assert legacy == stored_onset or abs(legacy - stored_onset) < 1e-12
            rows.append({"phase": phase["phase"], "late_detection_ap": measured,
                         "whole_detection_ap": ap(labels[start:end][labels[start:end] >= 0] > 0, p[start:end][labels[start:end] >= 0]),
                         "legacy_onset_ap_reproduced": legacy,
                         "corrected_onset_tolerance": temporal_onset(p[start:end], labels[start:end]),
                         "corrected_onset_raw": temporal_onset(p[start:end], raw[start:end])})
        report["arms"][arm] = rows
        report.setdefault("unsettled_rows", {})[arm] = {
            "interval_indices": np.where((labels < 0).any(axis=1))[0].tolist(),
            "host_step_count": int((labels < 0).sum())}
    for a, c in zip(report["arms"]["A"], report["arms"]["C"]):
        if a["late_detection_ap"] is not None:
            report["late_detection_gain_C_minus_A"].append({
                "phase": a["phase"], "A": a["late_detection_ap"],
                "C": c["late_detection_ap"],
                "gain": c["late_detection_ap"] - a["late_detection_ap"]})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite audit")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
