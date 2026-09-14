"""Protocol-024 evaluation helpers.

These helpers intentionally live outside Protocol-023 so historical registered
artifacts remain immutable. They fix three issues found in the 2026-09-14 audit:
1) onset must shift along time for the SAME host, never over a flattened host axis;
2) all scored rows must be settled, including steps-2 and steps-1 at finalization;
3) resource-class macro F1 is evaluated on true-fault rows only.

No model or simulator state is modified here.
"""
from __future__ import annotations

import numpy as np


def average_precision(y, score):
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if y.size == 0 or y.max() == y.min():
        return None
    order = np.argsort(-score, kind="stable")
    y = y[order]
    positives = y.sum()
    if positives <= 0:
        return None
    precision = np.cumsum(y) / np.arange(1, y.size + 1)
    return float((precision * y).sum() / positives)


def temporal_onset_metrics(probability, labels, horizon=1):
    """Same-host future onset AP over [time, host] arrays.

    Target(t,h)=1 iff host h is normal at t and becomes faulty in one of
    t+1..t+horizon. The last ``horizon`` time rows are excluded because the
    complete registered future is outside the phase. Any position whose
    current/future label is negative (unsettled/unknown) is excluded.
    """
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probability.shape != labels.shape or labels.ndim != 2:
        raise ValueError("probability and labels must both be [time, host]")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    T, H = labels.shape
    usable_T = T - horizon
    if usable_T <= 0:
        return {"ap": None, "positives": 0, "rows": 0}

    current = labels[:usable_T]
    known = current >= 0
    future_fault = np.zeros((usable_T, H), dtype=bool)

    for offset in range(1, horizon + 1):
        future = labels[offset:offset + usable_T]
        known &= future >= 0
        future_fault |= future > 0

    valid = known & (current == 0)
    target = future_fault[valid].astype(np.int64)
    score = probability[:usable_T][valid]
    return {"ap": average_precision(target, score),
            "positives": int(target.sum()),
            "rows": int(target.size)}


def positive_resource_macro_f1(class_probability, labels):
    """CPU/RAM/Disk macro-F1 only on true-fault rows.

    labels: 0=normal, 1=CPU, 2=RAM, 3=Disk.
    class_probability: [..., 3].
    """
    class_probability = np.asarray(class_probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if class_probability.shape[:-1] != labels.shape or class_probability.shape[-1] != 3:
        raise ValueError("class_probability must be labels.shape + (3,)")

    mask = labels > 0
    if not mask.any():
        return {"macro_f1": None, "rows": 0, "per_class": []}

    truth = labels[mask].reshape(-1)
    pred = class_probability[mask].reshape(-1, 3).argmax(-1) + 1
    per_class = []
    for klass in (1, 2, 3):
        t = truth == klass
        p = pred == klass
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": klass, "precision": precision,
                          "recall": recall, "f1": f1,
                          "support": int(t.sum())})
    return {"macro_f1": float(np.mean([x["f1"] for x in per_class])),
            "rows": int(mask.sum()), "per_class": per_class}


def tolerance_label(raw_seen, index, observed_until):
    """Protocol-023-compatible +/-1 tolerance label for one scored row."""
    raw_seen = np.asarray(raw_seen, dtype=np.int64)
    if index < 0 or index + 1 > observed_until:
        raise ValueError("tolerance label has not matured")
    local = raw_seen[:index + 2]
    result = local[index].copy()
    if (local[max(0, index - 1):index + 2] < 0).any():
        raise ValueError("unobserved raw label around index")
    if index:
        fill = (result == 0) & (local[index - 1] > 0)
        result[fill] = local[index - 1][fill]
    fill = (result == 0) & (local[index + 1] > 0)
    result[fill] = local[index + 1][fill]
    return result


def finalize_two_pending(raw_seen, labels, raw_labels, settled, settled_at, steps):
    """Settle BOTH tail rows left pending by the t-2 online settlement rule."""
    raw_seen = np.asarray(raw_seen)
    labels = np.asarray(labels)
    raw_labels = np.asarray(raw_labels)
    settled = np.asarray(settled)
    settled_at = np.asarray(settled_at)

    for idx in (steps - 2, steps - 1):
        if idx < 0:
            continue
        raw_labels[idx] = raw_seen[idx]
        labels[idx] = tolerance_label(raw_seen, idx, steps)
        settled[idx] = True
        settled_at[idx] = steps

    missing = np.flatnonzero(~settled)
    unknown = np.flatnonzero((labels < 0).any(axis=1))
    if missing.size or unknown.size:
        raise AssertionError("finalization incomplete: unsettled=%s unknown_labels=%s"
                             % (missing.tolist(), unknown.tolist()))
    return labels, raw_labels, settled, settled_at
