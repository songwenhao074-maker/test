"""Registered Protocol-024 learnability probe for response_law_v1.

Train split: [0,3900) = familiar prefix + all three long first exposures.
Validation split: [3900,4980) = the six registered recurrences.  Every method
predicts exactly raw_label[t+1] > 0 and uses no audit law/phase/event ID.
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


TRAIN_END = 3900
VALID_END = 4980
HISTORY = 12


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
    arrays = bundle["arrays"]
    raw = np.asarray(arrays["raw_labels"], dtype=np.int64)
    host = np.asarray(arrays["host_features"][:VALID_END], dtype=np.float32)
    ratio = np.asarray(arrays["overload_ratio"][:VALID_END], dtype=np.float64)
    if raw.shape[0] < VALID_END + 1:
        raise ValueError("future-fault target requires one guard row")
    target = (raw[1:VALID_END + 1] > 0).astype(np.int64)

    train_t = np.arange(0, TRAIN_END)
    valid_t = np.arange(TRAIN_END, VALID_END)
    train_y = target[train_t].reshape(-1)
    valid_y = target[valid_t].reshape(-1)
    if train_y.sum() == 0 or valid_y.sum() == 0:
        raise RuntimeError("future-fault learnability split has no positives")

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

    # Baselines use only current information at t.
    persistence = (raw[TRAIN_END:VALID_END] > 0).astype(np.float64).reshape(-1)
    current_pressure = ratio[TRAIN_END:VALID_END].max(axis=-1).reshape(-1)

    result = {
        "protocol": "024", "kind": "future_fault_learnability",
        "target": "raw_label[t+1] > 0 same host",
        "audit_ids_used_as_features": False,
        "train_intervals": [0, TRAIN_END],
        "validation_intervals": [TRAIN_END, VALID_END],
        "train_positive_host_steps": int(train_y.sum()),
        "validation_positive_host_steps": int(valid_y.sum()),
        "validation_prevalence": float(valid_y.mean()),
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
        "z_gap_to_history": None,
        "probe_fit_seconds_excluding_z": probe_fit_seconds,
        "z_extraction_seconds": z_extract_seconds,
    }
    h_ap = result["methods"]["raw_history_12x7_logistic"]["ap"]
    c_ap = result["methods"]["current_raw_7d_logistic"]["ap"]
    p_ap = result["methods"]["persistence_current_fault"]["ap"]
    z_ap = result["methods"]["frozen_z_64d_logistic"]["ap"]
    if h_ap is not None and c_ap is not None:
        result["history_gain_over_current_raw"] = float(h_ap - c_ap)
    if h_ap is not None and p_ap is not None:
        result["history_gain_over_persistence"] = float(h_ap - p_ap)
    if h_ap is not None and z_ap is not None:
        result["z_gap_to_history"] = float(h_ap - z_ap)

    # This is a diagnostic, not a post-hoc scenario tuner.  It marks whether
    # raw/history has any meaningful ranking signal beyond the simplest current
    # baselines; the response-law revision rule is handled by status/reporting.
    best_current = max(x for x in (p_ap, c_ap) if x is not None)
    result["raw_history_has_incremental_signal"] = bool(
        h_ap is not None and h_ap >= best_current + 0.01)
    result["z_retains_history_signal"] = bool(
        z_ap is not None and h_ap is not None and z_ap >= h_ap - 0.05)

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
