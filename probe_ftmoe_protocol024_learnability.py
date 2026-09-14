"""Registered Protocol-024 learnability probe for response_law_v1.

The split is deliberately separated by HISTORY+h = 12+1 = 13 complete
intervals so no validation 12-step input history shares rows with the training
side and no training raw[t+1] target touches the validation history.

Train prediction times: [0,3887)
Isolation gap:          [3887,3900)
Validation times:       [3900,4980) (all six recurrences)

Every method predicts exactly same-host raw_label[t+1] > 0 and uses no audit
law/phase/event ID.

Registered decision rule:
- raw/history is learnable iff validation AP >= prevalence + 0.05;
- frozen z retains the signal iff its AP is no more than 0.05 below history.
Persistence/current-pressure are reported required baselines, not a requirement
that history beat current overload pressure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_eval import average_precision_report
from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE


TRAIN_END = 3887
VALID_START = 3900
VALID_END = 4980
HISTORY = 12
HORIZON = 1
ISOLATION_GAP = HISTORY + HORIZON
MIN_AP_OVER_PREVALENCE = 0.05
MAX_Z_AP_GAP = 0.05


def _history_features(host):
    """[time,host,12*7], with the same left-edge repetition as ReplayV3."""
    t_count, hosts, dims = host.shape
    out = np.empty((t_count, hosts, HISTORY * dims), dtype=np.float32)
    for t in range(t_count):
        pos = np.maximum(np.arange(t - HISTORY + 1, t + 1), 0)
        out[t] = host[pos].transpose(1, 0, 2).reshape(hosts, -1)
    return out


def _fit_score(train_x, train_y, valid_x):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    valid_scaled = scaler.transform(valid_x)
    model = LogisticRegression(
        max_iter=300, solver="lbfgs", class_weight="balanced", C=1.0,
        random_state=24024)
    model.fit(train_scaled, train_y)
    return model.predict_proba(valid_scaled)[:, 1], {
        "iterations": int(np.max(model.n_iter_)),
        "features": int(train_x.shape[1]),
        "train_rows": int(train_x.shape[0])}


def _report(y, score):
    result = average_precision_report(y.astype(np.int64), score.astype(np.float64))
    result["prevalence"] = float(y.mean()) if y.size else None
    return result


def _extract_z(bundle):
    model = FrozenResidualFTMoE(bundle["checkpoint"], "C", 1)
    model.eval()
    chunks = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, VALID_END, 32):
            indices = list(range(start, min(VALID_END, start + 32)))
            x, sched, graph, context = s4.window_batch(bundle["replay"], indices)
            _, z = model._base_forward(x, sched, graph, graph_context=context,
                                       record=False)
            chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0), time.perf_counter() - started


def run(stream_dir, out_path):
    stream_dir = Path(stream_dir)
    out_path = Path(out_path)
    bundle = s4.build_replay(stream_dir)
    if bundle["steps"] != VALID_END:
        raise ValueError("learnability probe requires registered 4980-step stream")
    if VALID_START - TRAIN_END < ISOLATION_GAP:
        raise AssertionError("learnability split lost the registered 13-interval gap")

    arrays = bundle["arrays"]
    raw = np.asarray(arrays["raw_labels"], dtype=np.int64)
    host = np.asarray(arrays["host_features"][:VALID_END], dtype=np.float32)
    ratio = np.asarray(arrays["overload_ratio"][:VALID_END], dtype=np.float64)
    if raw.shape[0] < VALID_END + 1:
        raise ValueError("future-fault target requires one guard row")
    target = (raw[1:VALID_END + 1] > 0).astype(np.int64)

    train_t = np.arange(0, TRAIN_END)
    valid_t = np.arange(VALID_START, VALID_END)
    train_y = target[train_t].reshape(-1)
    valid_y = target[valid_t].reshape(-1)
    if train_y.sum() == 0 or valid_y.sum() == 0:
        raise RuntimeError("future-fault learnability split has no positives")
    if train_y.sum() == train_y.size or valid_y.sum() == valid_y.size:
        raise RuntimeError("future-fault learnability split has no negatives")

    train_last_target_row = TRAIN_END
    valid_first_history_row = VALID_START - HISTORY + 1
    if train_last_target_row >= valid_first_history_row:
        raise AssertionError(
            "train future target overlaps the first validation input history")

    history = _history_features(host)
    current_x = host.reshape(VALID_END * 16, 7)
    history_x = history.reshape(VALID_END * 16, HISTORY * 7)
    train_rows = np.concatenate([np.arange(t * 16, (t + 1) * 16)
                                 for t in train_t])
    valid_rows = np.concatenate([np.arange(t * 16, (t + 1) * 16)
                                 for t in valid_t])

    started = time.perf_counter()
    current_score, current_meta = _fit_score(
        current_x[train_rows], train_y, current_x[valid_rows])
    history_score, history_meta = _fit_score(
        history_x[train_rows], train_y, history_x[valid_rows])
    probe_fit_seconds = time.perf_counter() - started

    z, z_extract_seconds = _extract_z(bundle)
    z_x = z.reshape(VALID_END * 16, -1)
    z_score, z_meta = _fit_score(z_x[train_rows], train_y, z_x[valid_rows])

    persistence = (raw[VALID_START:VALID_END] > 0).astype(np.float64).reshape(-1)
    current_pressure = ratio[VALID_START:VALID_END].max(axis=-1).reshape(-1)

    result = {
        "protocol": "024", "kind": "future_fault_learnability",
        "target": "raw_label[t+1] > 0 same host",
        "audit_ids_used_as_features": False,
        "history_intervals": HISTORY,
        "target_horizon": HORIZON,
        "required_isolation_gap_intervals": ISOLATION_GAP,
        "actual_isolation_gap_intervals": int(VALID_START - TRAIN_END),
        "train_prediction_intervals": [0, TRAIN_END],
        "isolation_gap_prediction_intervals": [TRAIN_END, VALID_START],
        "validation_prediction_intervals": [VALID_START, VALID_END],
        "train_last_future_raw_row": int(train_last_target_row),
        "validation_first_history_raw_row": int(valid_first_history_row),
        "no_history_or_future_target_overlap": True,
        "train_positive_host_steps": int(train_y.sum()),
        "validation_positive_host_steps": int(valid_y.sum()),
        "validation_prevalence": float(valid_y.mean()),
        "gate": {"min_ap_over_prevalence": MIN_AP_OVER_PREVALENCE,
                 "max_frozen_z_ap_gap_to_history": MAX_Z_AP_GAP,
                 "registered_before_valid_model_result": True},
        "methods": {
            "persistence_current_fault": _report(valid_y, persistence),
            "current_pressure": _report(valid_y, current_pressure),
            "current_raw_7d_logistic": dict(_report(valid_y, current_score),
                                             **current_meta),
            "raw_history_12x7_logistic": dict(_report(valid_y, history_score),
                                                **history_meta),
            "frozen_z_64d_logistic": dict(_report(valid_y, z_score), **z_meta),
        },
        "history_gain_over_current_raw": None,
        "history_gain_over_persistence": None,
        "history_gain_over_current_pressure": None,
        "z_gap_to_history": None,
        "probe_fit_seconds_excluding_z": probe_fit_seconds,
        "z_extraction_seconds": z_extract_seconds,
    }
    h_ap = result["methods"]["raw_history_12x7_logistic"]["ap"]
    c_ap = result["methods"]["current_raw_7d_logistic"]["ap"]
    p_ap = result["methods"]["persistence_current_fault"]["ap"]
    pressure_ap = result["methods"]["current_pressure"]["ap"]
    z_ap = result["methods"]["frozen_z_64d_logistic"]["ap"]
    prevalence = result["validation_prevalence"]
    if h_ap is not None and c_ap is not None:
        result["history_gain_over_current_raw"] = float(h_ap - c_ap)
    if h_ap is not None and p_ap is not None:
        result["history_gain_over_persistence"] = float(h_ap - p_ap)
    if h_ap is not None and pressure_ap is not None:
        result["history_gain_over_current_pressure"] = float(h_ap - pressure_ap)
    if h_ap is not None and z_ap is not None:
        result["z_gap_to_history"] = float(h_ap - z_ap)

    result["raw_history_learnable"] = bool(
        h_ap is not None and h_ap >= prevalence + MIN_AP_OVER_PREVALENCE)
    result["z_retains_history_signal"] = bool(
        z_ap is not None and h_ap is not None and z_ap >= h_ap - MAX_Z_AP_GAP)
    result["proceed_to_budget_and_pilot"] = bool(
        result["raw_history_learnable"] and result["z_retains_history_signal"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.stream, args.out)


if __name__ == "__main__":
    main()
