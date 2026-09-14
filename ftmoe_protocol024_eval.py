"""Protocol-024 evaluation helpers used by the *actual* next-round runner.

Historical Protocol-023 artifacts are immutable.  This module is the single
implementation for Protocol-024 temporal/onset and positive-resource metrics.
The rules here are deliberately strict because the next-round directive uses
these functions to produce the final JSON, not merely as diagnostic helpers.
"""
from __future__ import annotations

import numpy as np


def _as_finite_float(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("%s contains NaN/inf" % name)
    return array


def average_precision_report(y, score):
    """Tie-invariant non-interpolated AP plus an explicit degenerate reason.

    Scores are accumulated by *score group* rather than by input order.  Hence
    equal-score samples produce AP equal to prevalence and shuffling equal-score
    rows cannot change the result.  AP is undefined when only one class is
    present; Protocol-024 reports null plus a reason instead of inventing 0.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    score = _as_finite_float(score, "score").reshape(-1)
    if y.shape != score.shape:
        raise ValueError("y and score must have the same number of elements")
    if y.size == 0:
        return {"ap": None, "reason": "no_valid_rows", "rows": 0,
                "positives": 0, "negatives": 0}
    if not np.isin(y, [0, 1]).all():
        raise ValueError("average precision target must be binary 0/1")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0:
        return {"ap": None, "reason": "no_positive", "rows": int(y.size),
                "positives": 0, "negatives": negatives}
    if negatives == 0:
        return {"ap": None, "reason": "no_negative", "rows": int(y.size),
                "positives": positives, "negatives": 0}

    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    score_sorted = score[order]
    # End index (exclusive) of every equal-score group.
    end = np.r_[np.flatnonzero(score_sorted[1:] != score_sorted[:-1]) + 1,
                y_sorted.size]
    start = np.r_[0, end[:-1]]
    tp = 0
    seen = 0
    ap = 0.0
    for left, right in zip(start.tolist(), end.tolist()):
        group_pos = int(y_sorted[left:right].sum())
        tp += group_pos
        seen += int(right - left)
        if group_pos:
            ap += (group_pos / float(positives)) * (tp / float(seen))
    return {"ap": float(ap), "reason": None, "rows": int(y.size),
            "positives": positives, "negatives": negatives}


def average_precision(y, score):
    """Compatibility scalar view used by older Protocol-024 tests."""
    return average_precision_report(y, score)["ap"]


def temporal_onset_metrics(probability, labels, horizon=1):
    """Same-host future onset AP over [time, host] arrays.

    Target(t,h)=1 iff host h is known-normal at t and becomes faulty in one of
    t+1..t+horizon.  Unknown current/future labels are excluded.  The last
    ``horizon`` rows are excluded because their complete future is outside the
    supplied block.  Probability is always aligned to the *current* row; a
    caller using a future-shifted training target must not shift this again.
    """
    probability = _as_finite_float(probability, "probability")
    labels = np.asarray(labels, dtype=np.int64)
    if probability.shape != labels.shape or labels.ndim != 2:
        raise ValueError("probability and labels must both be [time, host]")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    horizon = int(horizon)

    T, H = labels.shape
    usable_T = T - horizon
    if usable_T <= 0:
        report = average_precision_report([], [])
        report["horizon"] = horizon
        return report

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
    report = average_precision_report(target, score)
    report["horizon"] = horizon
    return report


def positive_resource_macro_f1(class_probability, labels):
    """CPU/RAM/Disk macro-F1 on known true-fault rows only.

    labels: -1=unknown, 0=normal, 1=CPU, 2=RAM, 3=Disk.
    class_probability: labels.shape + (3,).
    """
    class_probability = _as_finite_float(class_probability, "class_probability")
    labels = np.asarray(labels, dtype=np.int64)
    if (class_probability.shape[:-1] != labels.shape
            or class_probability.ndim != labels.ndim + 1
            or class_probability.shape[-1] != 3):
        raise ValueError("class_probability must be labels.shape + (3,)")
    if ((labels < -1) | (labels > 3)).any():
        raise ValueError("labels must use -1/0/1/2/3")

    mask = labels > 0
    if not mask.any():
        return {"macro_f1": None, "reason": "no_fault_positive",
                "rows": 0, "per_class": []}

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
            "reason": None, "rows": int(mask.sum()), "per_class": per_class}


def tolerance_label(raw_seen, index, observed_until):
    """Protocol-023-compatible +/-1 tolerance label for a regression run."""
    raw_seen = np.asarray(raw_seen, dtype=np.int64)
    if index < 0 or index + 1 > observed_until:
        raise ValueError("tolerance label has not matured")
    result = raw_seen[index].copy()
    if (raw_seen[max(0, index - 1):index + 2] < 0).any():
        raise ValueError("unobserved raw label around index")
    if index:
        fill = (result == 0) & (raw_seen[index - 1] > 0)
        result[fill] = raw_seen[index - 1][fill]
    fill = (result == 0) & (raw_seen[index + 1] > 0)
    result[fill] = raw_seen[index + 1][fill]
    return result


def assert_complete_settlement(labels, raw_labels, settled, scored_steps=None):
    """Fail if *any* scored row is unsettled/unknown, not only the tail."""
    labels = np.asarray(labels)
    raw_labels = np.asarray(raw_labels)
    settled = np.asarray(settled, dtype=bool)
    steps = int(labels.shape[0] if scored_steps is None else scored_steps)
    if labels.shape[0] < steps or raw_labels.shape[0] < steps or settled.size < steps:
        raise ValueError("settlement arrays are shorter than scored_steps")
    missing = np.flatnonzero(~settled[:steps])
    unknown_labels = np.flatnonzero((labels[:steps] < 0).any(axis=1))
    unknown_raw = np.flatnonzero((raw_labels[:steps] < 0).any(axis=1))
    if missing.size or unknown_labels.size or unknown_raw.size:
        raise AssertionError(
            "finalization incomplete: unsettled=%s unknown_labels=%s unknown_raw=%s"
            % (missing.tolist(), unknown_labels.tolist(), unknown_raw.tolist()))
    return {"scored_steps": steps, "settled": steps,
            "unknown_labels": 0, "unknown_raw": 0}


def finalize_two_pending(raw_seen, labels, raw_labels, settled, settled_at, steps):
    """Settle the two tail rows left by the historical t-2 tolerance rule.

    This function intentionally also verifies *all earlier rows*.  A hidden
    earlier settlement hole is therefore a hard failure, which is required by
    the next-round directive.
    """
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
    assert_complete_settlement(labels, raw_labels, settled, steps)
    return labels, raw_labels, settled, settled_at


def raw_next_target(raw_seen, index, observed_until):
    """Future target raw_label[index+1] with one publication-delay gate.

    ``observed_until`` is the current scored cursor whose raw row has already
    been physically observed.  A prediction at ``index`` may enter training
    only when ``observed_until >= index + 2``; this enforces the directive's
    t -> t+1 future event -> t+2 earliest-training order.
    """
    raw_seen = np.asarray(raw_seen, dtype=np.int64)
    if index < 0 or observed_until < index + 2:
        raise ValueError("raw-next target has not matured through publication delay")
    if index + 1 >= raw_seen.shape[0] or (raw_seen[index + 1] < 0).any():
        raise ValueError("raw-next future label is not physically observed")
    return raw_seen[index + 1].copy()
