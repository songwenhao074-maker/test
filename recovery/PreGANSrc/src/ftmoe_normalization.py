"""Protocol 019 — normalization v2 with safe same-group fallback.

Repairs the Protocol 014 defect where a (host, feature) training maximum of
~0 (or extremely low coverage) amplified ordinary online values to 1e8-1e11
after the 1e-8 clamp in legacy ``load_data``.

Rules (input contract v2, see 指令/FTMOE_ONLINE_TUNING_REVIEW_AND_SOLUTION_PLAN.md §5.2):

- A (host, feature) column is *under-covered* when any of:
    training scale <= 1e-8 (legacy zero clamp)      ["training_max_zero"]
    count_nonzero  < 12                             ["count_nonzero_lt_12"]
    training max   < 1e-4 * same_group_p95          ["max_below_group_p95_fraction"]
- For under-covered columns the divisor falls back to the *same hardware
  group* statistic of the same feature (median of healthy peer maxima).
- Only training statistics may drive the fallback.  Online labels and online
  future statistics are forbidden inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ftmoe_input_contract import HOST_FEATURES, HARDWARE_GROUPS, host_group

# Legacy load_data clamps zero training maxima with np.maximum(max, 1e-8);
# on disk that clamp surfaces as 1e-8 or the float32-roundtrip 9.9999999e-9,
# so the zero-coverage rule must use <= rather than <.
ZERO_CLAMP_EPS = 1e-8
LOW_COVERAGE_COUNT = 12
LOW_COVERAGE_MAX_FRACTION = 1e-4
# Only an alarm threshold for the *normalized* stream; never a clip.
NORMALIZED_ABS_MAX_WARNING = 50.0

GROUPS = {"rpi4gb": list(range(0, 8)), "rpi8gb": list(range(8, 16))}
GROUP_FEATURE_P95 = {"rpi4gb": {}, "rpi8gb": {}}


def _quantiles(values: np.ndarray) -> dict:
    return {
        "max": float(values.max()) if values.size else 0.0,
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()) if values.size else 0.0,
        "zero_ratio": float((values == 0).mean()) if values.size else 1.0,
    }


def compute_column_stats(train_series: np.ndarray) -> dict:
    """Per (host, feature) training statistics.

    Args:
        train_series: float array of training intervals shaped
            (..., 16, 7) — every host-feature column of the training data.
    Returns:
        {"hosts": [ {feature: {count_nonzero, max, p50, p95, p99, mean,
        zero_ratio}}, ... ] } with 16 host entries of 7 features each.
    """
    series = np.asarray(train_series)
    if series.shape[-2:] != (16, 7):
        raise ValueError(f"expected (...,16,7) shaped training series, got {series.shape}")
    flat = series.reshape(-1, 16, 7).transpose(1, 2, 0)  # (host, feat, interval)
    hosts = []
    for h in range(16):
        feature_stats = {}
        for f, name in enumerate(HOST_FEATURES):
            values = flat[h, f]
            nonzero = values[values > 0]
            stats = {"count_nonzero": int((values > 0).sum())}
            stats.update(_quantiles(values))
            stats["nonzero_p50"] = float(np.percentile(nonzero, 50)) if nonzero.size else 0.0
            stats["nonzero_max"] = float(nonzero.max()) if nonzero.size else 0.0
            feature_stats[name] = stats
        hosts.append(feature_stats)
    return {"hosts": hosts}


def _group_p95_of_peer_max(stats: dict, group: str, feature: str) -> float:
    maxima = [
        stats["hosts"][h][feature]["max"]
        for h in GROUPS[group]
    ]
    return float(np.percentile(maxima, 95))


def build_fallback_map(stats: dict) -> dict:
    """Decide which (host, feature) columns are under-covered and their
    same-group fallback divisor (median of healthy peer maxima)."""
    fallback = {}
    for h in range(16):
        group = host_group(h)
        for feature in HOST_FEATURES:
            entry = stats["hosts"][h][feature]
            scale = entry["max"]
            group_p95 = _group_p95_of_peer_max(stats, group, feature)
            reasons = []
            if scale <= ZERO_CLAMP_EPS:
                reasons.append("training_max_zero")
            if entry["count_nonzero"] < LOW_COVERAGE_COUNT:
                reasons.append(
                    f"count_nonzero_lt_{LOW_COVERAGE_COUNT}({entry['count_nonzero']})"
                )
            if scale < LOW_COVERAGE_MAX_FRACTION * group_p95:
                reasons.append("max_below_group_p95_fraction")
            if not reasons:
                continue
            peers = [p for p in GROUPS[group] if p != h]
            healthy = [
                stats["hosts"][p][feature]["max"]
                for p in peers
                if stats["hosts"][p][feature]["max"] >= ZERO_CLAMP_EPS
            ]
            if healthy:
                fallback_scale = float(np.median(healthy))
            else:
                raise ValueError(
                    f"no healthy peer for host {h} feature {feature} in {group}"
                )
            fallback[(h, feature)] = {
                "group": group,
                "reason": reasons,
                "fallback_scale": fallback_scale,
                "group_p95": float(group_p95),
            }
    return fallback


def apply_v2_time_scale(time_scale: np.ndarray, stats: dict) -> tuple[np.ndarray, list]:
    """Build the v2 (16,7) time scale from the legacy scale and training stats.

    Legacy scale entries that pass all coverage rules are kept untouched so
    that v2 is exactly equal to v1 on healthy columns.

    Returns (v2_scale (16,7) float64, fallback_columns list for logging).
    """
    scale = np.asarray(time_scale, dtype=np.float64).reshape(16, 7)
    if scale.shape != (16, 7):
        raise ValueError(f"time_scale must reshape to (16,7), got {scale.shape}")
    fallback_map = build_fallback_map(stats)
    fallback_columns = []
    v2 = scale.copy()
    for (h, feature), decision in sorted(fallback_map.items()):
        f = HOST_FEATURES.index(feature)
        if not np.isclose(scale[h, f], decision["fallback_scale"]):
            fallback_columns.append(
                {
                    "host": h,
                    "feature": feature,
                    "training_scale": float(scale[h, f]),
                    "reason": decision["reason"],
                    "fallback_scale": decision["fallback_scale"],
                    "group": decision["group"],
                }
            )
        v2[h, f] = decision["fallback_scale"]
    return v2, fallback_columns


def normalized_abs_max_report(raw_host_features: np.ndarray,
                              time_scale: np.ndarray) -> dict:
    """Distribution report of |normalized| values for one stream (16,7 layout).

    The abs max is an alarm threshold (default 50) — never a clip.  For
    non-finite inputs the report flags them instead of silently passing an
    isfinite check.
    """
    features = np.asarray(raw_host_features, dtype=np.float64)
    if features.ndim != 3 or features.shape[1:] != (16, 7):
        raise ValueError(f"expected (T,16,7), got {features.shape}")
    scale = np.asarray(time_scale, dtype=np.float64).reshape(16, 7)
    normalized = features / scale
    if not np.isfinite(normalized).all():
        nonfinite = int((~np.isfinite(normalized)).sum())
        return {"nonfinite": nonfinite, "alarm": True,
                "alarm_reason": f"{nonfinite} non-finite normalized values"}
    flat = np.abs(normalized)
    report = {
        "nonfinite": 0,
        "max": float(flat.max()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "fraction_abs_gt_5": float((flat > 5).mean()),
        "fraction_abs_gt_10": float((flat > 10).mean()),
        "abs_max_warning_threshold": NORMALIZED_ABS_MAX_WARNING,
        "alarm": bool(flat.max() > NORMALIZED_ABS_MAX_WARNING),
    }
    return report


def save_stats(path: Path | str, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf8"
    )
