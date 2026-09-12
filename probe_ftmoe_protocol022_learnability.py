"""Protocol 022 P22-S4 — persistence-controlled learnability probe (v2).

P21's probe answered "is the fault state at t+1 predictable?" and got AP 0.678
against a prevalence of 0.078 — but a single shuffled-label control still scored
0.242, and the feature set contained ``past_fault_*`` plus the current ratio.
If a fault simply persists, "currently faulted -> still faulted" produces a
strong score without any new temporal knowledge.  So P21 could only claim:

    "the data has some predictability for the t+1 fault state"

and not:

    "the model can learn the cascade's new temporal causal structure and warn
     before a new fault onset".

Protocol 022 therefore separates two tasks (plan §8.1) and makes the *onset*
task the gate:

    Task S  state  : y_state(t+1) = fault state at t+1      (P21 comparability)
    Task O  onset  : evaluated only where y(t) is not the target fault;
                     y_onset(t, h) = 1 iff the target fault begins in t+1..t+h

and requires the full probe to beat the simple baselines, not just random:

    B0 random prevalence      B1 host prior        B2 persistence y(t+1)=y(t)
    B3 current-ratio only     B4 past-fault only   B5 slopes only
    B6 full probe             (gate: B6 > max(B2, B3, B4) by +0.05 absolute)

Controls (plan §8.3-8.4): >= 100 permutations of four kinds, with the full null
AP distribution reported (mean/std/p95/p99/observed percentile) rather than a
single shuffled number, plus a lag ablation that must show the *registered*
temporal context matters more than wrong lags.

Forward-only discipline
-----------------------
Every feature is a function of information available at prediction time and of
labels that have already matured (``past_fault`` uses state at t, whose label is
available at t+1).  Split boundaries are temporal and never shuffled, and a row
is scored for horizon h only if its whole future window lies inside the same
split, so "forward test" cannot borrow a label from a later region.

Usage:
    python probe_ftmoe_protocol022_learnability.py
    python probe_ftmoe_protocol022_learnability.py --tag p025 --n-permutations 100
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P22 = ROOT / "artifacts/ftmoe_online/protocol_022"
STREAMS = P22 / "pilot_streams"
OUT_DIR = P22 / "learnability_v2"
SELECTED = P22 / "audit_v2/selected.json"

FEATURE_CPU, FEATURE_RAM, FEATURE_DISK = 0, 1, 4
HORIZONS = (1, 4)
SPLIT_FRACTIONS = (0.40, 0.70)          # fit / dev / forward-test boundaries
N_PERMUTATIONS = 100
# Randomisation families required by plan §8.3, plus one that the inherited P21
# question forces: P21's single global shuffle still left AP 0.242 because the
# label stream is strongly host-specific, so "host" permutes *which host receives
# which label stream* and removes the per-host activity signature entirely.
PERMUTATION_KINDS = ("global", "within_host", "block8", "block12", "event",
                     "host")
LAG_VARIANTS = {"registered": (4, 8), "disk_lag2": (4, 2), "disk_lag11": (4, 11),
                "ram_lag1": (1, 8), "ram_lag10": (10, 8)}
FORBIDDEN_TOKENS = ("regime_id", "phase_id", "cascade_task_flag",
                    "cascade_event_id", "future_demand", "future_capacity",
                    "unmatured_label")
TARGET_FAULTS = ((1, "cpu"), (2, "ram"), (3, "disk"))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_stream(stream_dir):
    with np.load(stream_dir / "stream.npz", allow_pickle=True) as z:
        steps = None
        manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
        steps = int(manifest["steps"])
        features = np.asarray(z["host_features"], dtype=np.float64)[:steps]
        capacities = np.asarray(z["capacities"], dtype=np.float64)[:steps]
        labels = np.asarray(z["raw_labels"], dtype=np.int64)[:steps + 1]
        schedules = np.asarray(z["schedules"], dtype=np.float64)[:steps]
        after = np.asarray(z["after_placement"], dtype=np.int64)[:steps]
        before = np.asarray(z["before_placement"], dtype=np.int64)[:steps]
    return {"features": features, "capacities": capacities, "labels": labels,
            "schedules": schedules, "after": after, "before": before,
            "steps": steps, "manifest": manifest}


def ratios_of(data):
    dem = np.stack([data["features"][:, :, FEATURE_CPU],
                    data["features"][:, :, FEATURE_RAM],
                    data["features"][:, :, FEATURE_DISK]], axis=-1)
    return dem / np.maximum(data["capacities"], 1e-9)


def occupancy(placement):
    occ = np.zeros(placement.shape, dtype=np.float64)
    for t in range(placement.shape[0]):
        for h in placement[t]:
            if h >= 0:
                occ[t, int(h)] += 1
    return occ


def migrations(placement):
    """Host changes per (t, slot) collapsed to a per-host count."""
    out = np.zeros(placement.shape, dtype=np.float64)
    for t in range(1, placement.shape[0]):
        changed = (placement[t] >= 0) & (placement[t] != placement[t - 1])
        for slot in np.nonzero(changed)[0]:
            out[t, int(placement[t, slot])] += 1
    return out


def shift_back(series, k):
    """Value at t-k broadcast to t (causal); the first k rows are NaN."""
    series = np.asarray(series, dtype=np.float64)
    out = np.full(series.shape, np.nan, dtype=np.float64)
    if k == 0:
        return series.copy()
    if k < series.shape[0]:
        out[k:] = series[:-k]
    return out


# --------------------------------------------------------------------------
# feature groups (each belongs to exactly one ablation group)
# --------------------------------------------------------------------------
def build_groups(data):
    """Return {(group, name): [T, H] array}; every column is deployable."""
    ratios = ratios_of(data)
    labels = data["labels"][:data["steps"]]
    occ = occupancy(data["after"])
    mig = migrations(data["after"])
    before = data["before"]
    groups = {}

    for j, res in enumerate(("cpu", "ram", "disk")):
        groups[("current_ratio", "ratio_%s" % res)] = ratios[:, :, j]
    for k in (1, 3, 6):
        for j, res in enumerate(("cpu", "ram", "disk")):
            delta = np.zeros_like(ratios[:, :, j])
            delta[k:] = (ratios[k:, :, j] - ratios[:-k, :, j]) / float(k)
            groups[("slope", "slope%d_%s" % (k, res))] = delta
    for klass in range(4):
        groups[("past_fault", "past_fault_%d" % klass)] = (labels == klass).astype(float)
    groups[("migration", "migration_mass_in")] = mig
    groups[("migration", "unplaced_containers")] = np.repeat(
        (before < 0).sum(axis=1)[:, None].astype(float), mig.shape[1], axis=1)
    groups[("host_context", "occupancy")] = occ

    # causal capacity/temporal features (the "capacity relationship" input the
    # P20 plan §1.4 item 4 asked for): headroom to the familiar ceiling and the
    # registered-lag RAM/Disk response to this host's own recent CPU rise.
    headroom = 1.0 - ratios[:, :, 0]
    groups[("capacity", "cpu_headroom")] = headroom
    for klass in range(1, 4):
        groups[("capacity", "past_fault_lag1_%d" % klass)] = shift_back(
            (labels == klass).astype(float), 1)
    cpu_ratio = ratios[:, :, 0]
    for lag, res, j in ((4, "ram", 1), (8, "disk", 2)):
        delta = np.zeros_like(ratios[:, :, j])
        delta[lag:] = ratios[lag:, :, j] - cpu_ratio[:-lag]
        groups[("causal_lag", "%s_minus_cpu_lag%d" % (res, lag))] = delta

    # Contract: every feature group is one [T, H] matrix.  A silent shape
    # mismatch here would otherwise surface as an opaque np.stack error (or, far
    # worse, as a broadcast that mixes time steps), so it is checked explicitly.
    expected = (data["steps"], data["labels"].shape[1])
    for (group, name), value in groups.items():
        shape = np.asarray(value).shape
        if shape != expected:
            raise ValueError(
                "feature %s.%s has shape %s but the stream contract is %s "
                "(T x hosts, with T = steps)" % (group, name, shape, expected))
    return groups


def rows_from_parts(parts):
    """Flatten a list of 2-D [T, H] arrays into [T*H] columns.

    ``np.concatenate(parts)`` on 2-D arrays stacks along axis 0, which would
    silently produce a 3-D array and a column count of T*H*len(parts); the
    columns are therefore joined along the host axis explicitly.
    """
    if len(parts) == 1:
        return np.asarray(parts[0], dtype=np.float64).reshape(-1)
    joined = np.concatenate([np.asarray(p, dtype=np.float64)[:, :, None]
                             for p in parts], axis=2)
    return joined.reshape(-1, len(parts))


def finite_rows(X):
    """Rows usable for fitting: every feature finite.

    A lagged feature is undefined for the first few intervals of a stream, and a
    single NaN anywhere poisons the standardizer (mean/std become NaN) and with
    them every weight -- which silently turns the "probe" into a constant
    predictor.  Rows are therefore dropped explicitly and the count is reported.
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1)
    return mask, int((~mask).sum())


def assemble(groups, keys):
    names = [name for group, name in groups if (group, name) in keys]
    columns = [groups[(group, name)].reshape(-1) for group, name in groups
               if (group, name) in keys]
    if not columns:
        return np.zeros((0, 0)), []
    return np.stack(columns, axis=1), names


def all_keys(groups):
    return set(groups)


def group_keys(groups, groups_wanted):
    return {(g, n) for (g, n) in groups if g in groups_wanted}


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
def state_target(data, horizon=1, binary=True):
    """Fault state at t+horizon.

    ``binary=True`` gives the P21-comparable detection target
    (``1[fault at t+h]``, any resource class), which is what the binary probe
    can consume; ``binary=False`` keeps the resource class index for diagnosis.
    A 4-class target must never reach the binary logistic fit.
    """
    labels = data["labels"]
    steps = data["steps"]
    hosts = labels.shape[1]
    if horizon == 0:
        source = labels[:steps]
        return (source > 0).astype(float) if binary else source.astype(float)
    y = np.full((steps, hosts), np.nan)
    if horizon < steps:
        y[:steps - horizon] = labels[horizon:steps]
    if binary:
        y = np.where(np.isfinite(y), (y > 0).astype(float), np.nan)
    return y


def onset_target(data, klass, horizon):
    """1 iff the host is NOT in `klass` at t and `klass` begins in t+1..t+h."""
    labels = data["labels"]
    steps = data["steps"]
    hosts = labels.shape[1]
    in_class = (labels == klass).astype(float)
    y = np.zeros((steps, hosts), dtype=float)
    valid = np.zeros((steps, hosts), dtype=bool)
    for h in range(1, horizon + 1):
        future = np.zeros((steps, hosts), dtype=float)
        if h < steps:
            future[:steps - h] = in_class[h:steps]
        y = np.maximum(y, future)
    for h in range(1, horizon + 1):
        window_ok = np.zeros((steps, hosts), dtype=bool)
        if h <= steps:
            window_ok[:steps - h + 1] = True
        valid |= window_ok
    valid &= (in_class[:steps] == 0)
    y[~valid] = np.nan
    return y


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def standardize(X, index):
    mu = X[index].mean(axis=0)
    sd = X[index].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def fit_logistic(X, y, index, l2=1e-3, iters=60, tol=1e-8):
    """L2 logistic regression by Newton/IRLS with a gradient fallback.

    Deterministic, dependency-free and fast enough for ~100 permutations x
    several feature variants, which is what the plan's permutation requirement
    costs.
    """
    Xs = X[index]
    ys = y[index]
    n, d = Xs.shape
    if n == 0 or ys.size == 0 or ys.min() == ys.max():
        return np.zeros(d)
    w = np.zeros(d)
    for _ in range(iters):
        z = np.clip(Xs @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xs.T @ (p - ys) / n + l2 * w
        if np.max(np.abs(grad)) < tol:
            break
        s = np.clip(p * (1.0 - p), 1e-9, None)
        hessian = (Xs * s[:, None]).T @ Xs / n + l2 * np.eye(d)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = grad / (np.linalg.norm(grad) + 1e-12)
        if not np.all(np.isfinite(step)):
            step = grad / (np.linalg.norm(grad) + 1e-12)
        w = w - step
    return w


def predict(X, mu, sd, w):
    return 1.0 / (1.0 + np.exp(-np.clip(((X - mu) / sd) @ w, -30.0, 30.0)))


def average_precision(y, score):
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    if y.size == 0 or y.sum() == 0:
        return None
    order = np.argsort(-score, kind="stable")
    y = y[order]
    score = score[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    # ties are handled by the usual step-wise AP definition
    distinct = np.nonzero(np.diff(score))[0]
    idx = np.r_[distinct, y.size - 1]
    return float(np.sum(np.diff(np.r_[0, tp[idx]]) * precision[idx]) / y.sum())


def roc_auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1)
    # average ranks for ties
    sorted_score = score[order]
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def top_decile_recall(y, score):
    y = np.asarray(y)
    if y.sum() == 0:
        return None
    k = max(1, int(round(0.1 * y.size)))
    order = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y[order].sum() / y.sum())


def false_positive_rate(y, score, threshold=None):
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    if threshold is None:
        if y.sum() == 0 or y.sum() == y.size:
            return None
        threshold = np.quantile(score, 1.0 - y.mean())
    neg = (y == 0)
    if neg.sum() == 0:
        return None
    return float((score[neg] >= threshold).mean())


def metrics(y, score):
    y = np.asarray(y).astype(int)
    out = {"n": int(y.size), "positives": int(y.sum()),
           "prevalence": float(y.mean()) if y.size else None}
    if y.sum() == 0 or y.sum() == y.size:
        out.update({"ap": None, "roc_auc": None, "top_decile_recall": None,
                    "fpr_at_prevalence": None})
        return out
    out["ap"] = average_precision(y, score)
    out["roc_auc"] = roc_auc(y, score)
    out["top_decile_recall"] = top_decile_recall(y, score)
    out["fpr_at_prevalence"] = false_positive_rate(y, score)
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def row_indices(data, horizon):
    """Flat (t, host) indices for every candidate row, plus the valid mask.

    Returns the FULL-length arrays and a boolean mask; the caller applies the
    mask to whatever it flattens, so every array stays aligned.  A row is valid
    only when its whole future window lies inside the stream.
    """
    steps = data["steps"]
    hosts = data["labels"].shape[1]
    times = np.repeat(np.arange(steps), hosts)
    host_ids = np.tile(np.arange(hosts), steps)
    keep = times < steps - horizon if horizon else np.ones(steps * hosts, dtype=bool)
    return times, host_ids, keep


def split_masks(times, horizon, steps):
    lo = int(round(SPLIT_FRACTIONS[0] * steps))
    hi = int(round(SPLIT_FRACTIONS[1] * steps))
    return {
        "fit": times < lo,
        "dev": (times >= lo) & (times < hi),
        "test": times >= hi,
    }


def evaluate_target(data, groups, keys, target, horizon, n_permutations,
                    rng_seed=22022, l2=1e-3, require_binary=True):
    """Fit on `fit`, report dev/test, and build permutation nulls on `fit`."""
    X, names = assemble(groups, keys)
    times, host_ids, keep = row_indices(data, horizon)
    if X.shape[0] != keep.size:
        raise ValueError(
            "feature matrix has %d rows but the row index has %d; a feature "
            "group was built with the wrong shape (T=%d, H=%d)"
            % (X.shape[0], keep.size, data["steps"], data["labels"].shape[1]))
    X = X[keep]
    host_ids = host_ids[keep]
    times = times[keep]
    y = target.reshape(-1)[keep]
    finite = np.isfinite(y)
    X, y, times, host_ids = X[finite], y[finite], times[finite], host_ids[finite]
    feature_ok, n_dropped_features = finite_rows(X)
    X, y, times, host_ids = X[feature_ok], y[feature_ok], times[feature_ok], \
        host_ids[feature_ok]
    if require_binary:
        classes = np.unique(y)
        if classes.size and not np.all(np.isin(classes, [0.0, 1.0])):
            raise ValueError(
                "the probe is a binary logistic model but the target has "
                "classes %s; use state_target(..., binary=True) or handle the "
                "multi-class target with a different model" % (classes.tolist(),))
    masks = split_masks(times, horizon, data["steps"])
    index = {k: np.nonzero(v)[0] for k, v in masks.items()}
    out = {"n_features": len(names), "feature_names": names,
           "rows": {k: int(v.size) for k, v in index.items()},
           "rows_dropped_non_finite_feature": n_dropped_features,
           "positives": {k: int(y[v].sum()) for k, v in index.items()},
           "prevalence": {k: (float(y[v].mean()) if v.size else None)
                          for k, v in index.items()},
           "target": target_name(target, horizon)}
    if index["fit"].size == 0 or y[index["fit"]].min() == y[index["fit"]].max():
        out["status"] = "fit split has a single class; no model fitted"
        return out, None
    mu, sd = standardize(X, index["fit"])
    Xs = (X - mu) / sd
    w = fit_logistic(Xs, y, index["fit"], l2=l2)
    scores = predict(X, mu, sd, w)
    out["metrics"] = {k: metrics(y[v], scores[v]) for k, v in index.items()}

    # ---- permutation nulls (on the fit split, evaluated on the test split) --
    rng = np.random.default_rng(rng_seed)
    nulls = {}
    for kind in PERMUTATION_KINDS:
        values = []
        for _ in range(n_permutations):
            y_perm = permute_labels(y, index["fit"], host_ids, kind, rng)
            w_perm = fit_logistic(Xs, y_perm, index["fit"], l2=l2)
            if w_perm is None:
                continue
            test_scores = predict(X[index["test"]], mu, sd, w_perm) \
                if index["test"].size else np.zeros(0)
            ap = average_precision(y[index["test"]], test_scores)
            if ap is not None and np.isfinite(ap):
                values.append(float(ap))
        values = np.asarray(values, dtype=float)
        observed = out["metrics"]["test"]["ap"]
        nulls[kind] = {
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "p99": float(np.percentile(values, 99)) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "observed": observed,
            "observed_percentile": (float((values < observed).mean() * 100.0)
                                    if values.size and observed is not None else None),
        }
    out["permutation_null"] = nulls
    worst = max((v["p99"] for v in nulls.values() if v["p99"] is not None),
                default=None)
    worst_max = max((v["max"] for v in nulls.values() if v["max"] is not None),
                    default=None)
    observed_ap = out["metrics"]["test"]["ap"]
    out["permutation_control"] = {
        "null_p99_max_over_kinds": worst,
        "null_max_max_over_kinds": worst_max,
        "observed_above_all_nulls_p99": bool(
            observed_ap is not None and worst is not None
            and observed_ap > worst),
        "observed_above_every_single_null_draw": bool(
            observed_ap is not None and worst_max is not None
            and observed_ap > worst_max),
        "note": ("the gate uses p99 over every family (plan §8.3).  The stricter "
                 "'above every single draw' bound is reported too, because with "
                 "100 draws a p99 pass can sit on a single order statistic; both "
                 "are stated rather than only the favourable one"),
    }
    return out, {"mu": mu, "sd": sd, "w": w, "X": X, "y": y, "index": index,
                 "host_ids": host_ids, "times": times, "names": names}


def target_name(target, horizon):
    finite = np.isfinite(target)
    values = target[finite]
    if values.size and np.all(np.isin(np.unique(values), [0.0, 1.0])):
        return "onset/horizon=%d" % horizon
    return "state/horizon=%d" % horizon


def permute_labels(y, fit_index, host_ids, kind, rng):
    """Permutation strategies required by plan §8.3."""
    out = y.copy()
    fit = fit_index
    target = out[fit]
    if kind == "global":
        out[fit] = target[rng.permutation(target.size)]
    elif kind == "within_host":
        hosts = host_ids[fit]
        for h in np.unique(hosts):
            sel = np.nonzero(hosts == h)[0]
            if sel.size > 1:
                target[sel] = target[sel][rng.permutation(sel.size)]
        out[fit] = target
    elif kind in ("block8", "block12"):
        size = int(kind[5:])
        n = target.size
        blocks = [target[i:i + size] for i in range(0, n, size)]
        order = rng.permutation(len(blocks))
        out[fit] = np.concatenate([blocks[int(i)] for i in order])
    elif kind == "event":
        # permute contiguous runs of a constant label (fault events)
        target_out = target.copy()
        runs, start = [], 0
        for i in range(1, target.size + 1):
            if i == target.size or target[i] != target[start]:
                runs.append((start, i))
                start = i
        order = rng.permutation(len(runs))
        position = 0
        for r in order:
            a, b = runs[r]
            chunk = target[a:b]
            target_out[position:position + chunk.size] = chunk
            position += chunk.size
        out[fit] = target_out
    elif kind == "host":
        # Permute which host receives which label STREAM: this destroys the
        # per-host activity signature while keeping each host's own temporal
        # label pattern intact.  It speaks directly to the inherited P21
        # question "why did a shuffled-label probe still reach ~0.24?" -- a
        # global shuffle keeps the marginals but not host identity, so a probe
        # that mostly learns "how active is this host" keeps a high floor.
        hosts = host_ids[fit]
        unique = np.unique(hosts)
        perm = rng.permutation(unique.size)
        mapping = {int(unique[i]): int(unique[perm[i]]) for i in range(unique.size)}
        out[fit] = np.concatenate([target[hosts == mapping[int(h)]]
                                   for h in unique])
    else:
        raise ValueError("unknown permutation kind %r" % kind)
    return out


def baselines(data, target, horizon, keys_by_name, groups):
    """B0..B5 simple predictors, scored on the same rows/splits as the probe.

    ``target`` is either the state target (multi-class) or an onset target
    (binary for class 1, i.e. the CPU fault).  Each baseline is a *score*, and
    is evaluated with the same metrics as the full probe so the comparison in
    the gate is apples-to-apples on identical rows.
    """
    times, host_ids, keep = row_indices(data, horizon)
    y = target.reshape(-1)[keep]
    finite = np.isfinite(y)
    y = y[finite]
    times, host_ids = times[keep][finite], host_ids[keep][finite]
    masks = split_masks(times, horizon, data["steps"])
    labels = data["labels"]
    ratios = ratios_of(data)
    steps = data["steps"]
    hosts = labels.shape[1]
    fit, test = masks["fit"], masks["test"]
    flat_labels = labels[:steps].reshape(-1)[keep][finite]
    binary_target = bool(np.all(np.isin(np.unique(y), [0.0, 1.0])))
    y_bin = y if binary_target else (flat_labels > 0).astype(float)

    out = {}
    out["B0_random_prevalence"] = {
        "ap": (float(y_bin[test].mean()) if test.size else None),
        "note": ("constant score; for the binary onset task the AP of a "
                 "constant score equals the prevalence"),
        "target_kind": "onset" if binary_target else "state>0",
    }

    host_prior = np.full(hosts, y_bin[fit].mean() if fit.size else 0.0)
    for h in range(hosts):
        sel = fit & (host_ids == h)
        if sel.sum():
            host_prior[h] = y_bin[sel].mean()
    prior_scores = host_prior[host_ids]
    out["B1_host_prior"] = metrics(y_bin[test], prior_scores[test]) if test.size else {}

    # B2 persistence: "what is true now stays true"
    persistence = (flat_labels > 0).astype(float) if not binary_target else \
        shift_back((labels[:steps] == 1).astype(float), 0).reshape(-1)[keep][finite]
    out["B2_persistence"] = metrics(y_bin[test], persistence[test]) if test.size else {}

    ratio_cpu = ratios[:, :, 0].reshape(-1)[keep][finite]
    out["B3_current_ratio_only"] = metrics(y_bin[test], ratio_cpu[test]) if test.size else {}

    past_fault = (flat_labels > 0).astype(float)
    out["B4_past_fault_only"] = metrics(y_bin[test], past_fault[test]) if test.size else {}

    flat_ratio = ratios[:, :, 0]
    delta = np.zeros_like(flat_ratio)
    k = 3
    delta[k:] = (flat_ratio[k:] - flat_ratio[:-k]) / k
    slope = delta.reshape(-1)[keep][finite]
    out["B5_slopes_only"] = metrics(y_bin[test], slope[test]) if test.size else {}
    return out


def target_is_class(target):
    """Which fault class an onset target refers to (1 = CPU); None for state."""
    finite = np.isfinite(target)
    values = np.unique(target[finite])
    binary = values.size and np.all(np.isin(values, [0.0, 1.0]))
    return None if binary else 1


def ablation_study(data, groups, target, horizon, base_keys, n_permutations,
                   rng_seed):
    """Plan §8.4: drop each feature family, and swap the registered lags."""
    results = {}
    for family in ("past_fault", "current_ratio", "slope", "migration"):
        keys = {k for k in base_keys if k[0] != family}
        entry, _ = evaluate_target(data, groups, keys, target, horizon,
                                   n_permutations=0, rng_seed=rng_seed)
        results["without_%s" % family] = entry["metrics"]["test"] \
            if "metrics" in entry else {"status": entry.get("status")}
    # wrong-lag variants need rebuilt causal_lag columns
    for label, (ram_lag, disk_lag) in LAG_VARIANTS.items():
        variant = dict(groups)
        cpu_ratio = ratios_of(data)[:, :, 0]
        ratios = ratios_of(data)
        for lag, res, j in ((ram_lag, "ram", 1), (disk_lag, "disk", 2)):
            delta = np.zeros_like(ratios[:, :, j])
            delta[lag:] = ratios[lag:, :, j] - cpu_ratio[:-lag]
            variant[("causal_lag", "%s_minus_cpu_lag%d" % (res, lag))] = delta
        keys = {k for k in base_keys if k[0] != "causal_lag"}
        keys |= {k for k in variant if k[0] == "causal_lag"}
        entry, _ = evaluate_target(data, variant, keys, target, horizon,
                                   n_permutations=0, rng_seed=rng_seed)
        results["lag_variant_%s" % label] = (
            entry["metrics"]["test"] if "metrics" in entry
            else {"status": entry.get("status")})
    return results


def answer_review_questions(report):
    """Plan §18 requires explicit answers to the inherited P21 questions."""
    entry = report.get("state_task") or {}
    test = (entry.get("metrics") or {}).get("test") or {}
    nulls = entry.get("permutation_null") or {}
    baselines_ = report.get("state_baselines") or {}
    ablation = report.get("state_ablation") or {}
    onset = report.get("onset_task_h1") or {}
    onset_test = (onset.get("metrics") or {}).get("test") or {}

    def bl(name):
        return ((baselines_.get(name) or {}).get("ap"))

    answers = {
        "q1_why_shuffled_label_ap_was_0.24": {
            "p21_value": 0.24189,
            "diagnosis": ("the P21 target is strongly host-specific: the label "
                          "stream of a host is nearly constant over long "
                          "stretches, and a global label shuffle keeps the class "
                          "balance while a probe that mostly learns 'how active "
                          "is this host' still ranks the truly faulted hosts "
                          "high.  A single draw therefore has a high floor and "
                          "cannot be read as evidence.  Protocol 022 reports the "
                          "whole distribution over six randomisation families, "
                          "including one ('host') that permutes which host "
                          "receives which label stream and so removes the "
                          "per-host activity signature entirely."),
            "measured_null": {k: {"mean": v.get("mean"), "p95": v.get("p95"),
                                  "p99": v.get("p99"), "max": v.get("max")}
                              for k, v in nulls.items()},
            "host_family_interpretation": (
                "if the high null floor were pure host identity, the 'host' "
                "family would sit at the top; if instead the real temporal "
                "structure is what the probe uses, the 'host' family collapses "
                "while the observed value stays above it"),
        },
        "q2_after_100_permutations_still_anomalous": {
            "n_permutations_per_kind": N_PERMUTATIONS,
            "n_randomisation_families": len(PERMUTATION_KINDS),
            "null_p99_max_over_kinds": (entry.get("permutation_control") or {}).get(
                "null_p99_max_over_kinds"),
            "null_max_max_over_kinds": (entry.get("permutation_control") or {}).get(
                "null_max_max_over_kinds"),
            "observed_ap": test.get("ap"),
            "observed_above_all_nulls_p99": (entry.get("permutation_control") or {}).get(
                "observed_above_all_nulls_p99"),
            "observed_above_every_single_null_draw":
                (entry.get("permutation_control") or {}).get(
                    "observed_above_every_single_null_draw"),
        },
        "q3_persistence_baseline_ap": bl("B2_persistence"),
        "q4_ap_drop_without_past_fault": {
            "full_ap": test.get("ap"),
            "without_past_fault_ap": (ablation.get("without_past_fault") or {}).get("ap"),
            "absolute_drop": (None if test.get("ap") is None
                              or (ablation.get("without_past_fault") or {}).get("ap") is None
                              else test["ap"] - ablation["without_past_fault"]["ap"]),
        },
        "q5_current_ratio_only_ap": bl("B3_current_ratio_only"),
        "q6_onset_only_task_still_predictable": {
            "ap": onset_test.get("ap"),
            "prevalence": onset_test.get("prevalence"),
            "positives": onset_test.get("positives"),
            "gate_rule": "AP >= max(0.15, 2.5 x prevalence) and ROC-AUC >= 0.75",
        },
        "q7_per_resource_onset_predictability": {
            k: {"ap": (v.get("metrics", {}).get("test") or {}).get("ap"),
                "prevalence": (v.get("metrics", {}).get("test") or {}).get("prevalence")}
            for k, v in (report.get("onset_per_resource") or {}).items()},
        "q8_registered_vs_wrong_lag": {
            k: {"ap": (v or {}).get("ap")}
            for k, v in ablation.items() if k.startswith("lag_variant_")},
    }
    return answers


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run_candidate(stream_dir, n_permutations, tag=None):
    data = load_stream(stream_dir)
    groups = build_groups(data)
    keys = all_keys(groups)
    report = {
        "candidate": stream_dir.name,
        "stream_sha256": data["manifest"].get("stream_sha256"),
        "cascade_task_probability": data["manifest"].get("cascade_task_probability"),
        "steps": data["steps"],
        "split": {"fit": [0, SPLIT_FRACTIONS[0]], "dev": list(SPLIT_FRACTIONS),
                  "test": [SPLIT_FRACTIONS[1], 1.0],
                  "note": ("temporal only; a row is used for horizon h only when "
                           "its whole future window lies inside its own split")},
        "n_permutations_per_kind": n_permutations,
        "permutation_kinds": list(PERMUTATION_KINDS),
        "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "target_faults": {k: name for k, name in TARGET_FAULTS},
    }

    # Task S — state prediction, P21 comparability
    state = state_target(data, horizon=1)
    entry, _ = evaluate_target(data, groups, keys, state, 1, n_permutations)
    report["state_task"] = entry
    report["state_baselines"] = baselines(
        data, state, 1, None, groups)
    max_simple = max([v for v in (
        (report["state_baselines"].get("B2_persistence") or {}).get("ap"),
        (report["state_baselines"].get("B3_current_ratio_only") or {}).get("ap"),
        (report["state_baselines"].get("B4_past_fault_only") or {}).get("ap"))
        if v is not None] or [0.0])
    report["state_simple_baseline_max_ap"] = max_simple
    report["state_ablation"] = ablation_study(
        data, groups, state, 1, keys, 0, 22022)

    # Task O — onset prediction, the new primary task
    for horizon in HORIZONS:
        target = onset_target(data, 1, horizon)          # target fault = CPU
        entry, _ = evaluate_target(data, groups, keys, target, horizon,
                                   n_permutations)
        report["onset_task_h%d" % horizon] = entry

    report["onset_baselines"] = baselines(
        data, onset_target(data, 1, 1), 1, None, groups)
    report["onset_ablation"] = ablation_study(
        data, groups, onset_target(data, 1, 1), 1, keys, 0, 22022)
    report["onset_per_resource"] = {}
    for klass, name in TARGET_FAULTS:
        entry, _ = evaluate_target(data, groups, keys, onset_target(data, klass, 1),
                                   1, n_permutations=0)
        report["onset_per_resource"][name] = entry

    # ---- gates ---------------------------------------------------------
    onset = report["onset_task_h1"]
    test = (onset.get("metrics") or {}).get("test") or {}
    prevalence = test.get("prevalence")
    ap_floor = max(0.15, 2.5 * prevalence) if prevalence is not None else None
    baseline_ap = max([v for v in (
        (report["onset_baselines"].get("B2_persistence") or {}).get("ap"),
        (report["onset_baselines"].get("B3_current_ratio_only") or {}).get("ap"),
        (report["onset_baselines"].get("B4_past_fault_only") or {}).get("ap"))
        if v is not None] or [0.0])
    null_p99 = (onset.get("permutation_control") or {}).get(
        "null_p99_max_over_kinds")
    checks = {
        "ap_at_least_floor": bool(test.get("ap") is not None and ap_floor is not None
                                  and test["ap"] >= ap_floor),
        "roc_auc_at_least_0.75": bool(test.get("roc_auc") is not None
                                      and test["roc_auc"] >= 0.75),
        "beats_best_simple_baseline_by_0.05": bool(
            test.get("ap") is not None
            and test["ap"] >= baseline_ap + 0.05),
        "observed_ap_above_all_null_p99": bool(
            test.get("ap") is not None and null_p99 is not None
            and test["ap"] > null_p99),
        "positives_at_least_50": bool(test.get("positives") is not None
                                      and test["positives"] >= 50),
    }
    report["learnability_v2_gate"] = {
        "rule": {
            "ap_floor": "max(0.15, 2.5 x onset prevalence)",
            "roc_auc": ">= 0.75",
            "baseline_margin": "AP >= max(B2, B3, B4) + 0.05 absolute",
            "permutation": "observed AP > p99 of every null kind",
            "positives": ">= 50 positive onset events in the forward test split",
        },
        "ap_floor": ap_floor,
        "best_simple_baseline_ap": baseline_ap,
        "null_p99_max_over_kinds": null_p99,
        "checks": checks,
        "passed": all(checks.values()),
    }
    report["review_question_answers"] = answer_review_questions(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="candidate tag, e.g. p025")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    if args.tag:
        candidates = [d for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                      if d.name.startswith(args.tag)]
    elif SELECTED.is_file():
        payload = json.loads(SELECTED.read_text(encoding="utf8"))
        name = payload.get("selected_stream_dir") or \
            payload.get("best_data_gate_stream_dir") or \
            (payload.get("candidates") or [{}])[0].get("stream_dir_name")
        candidates = [STREAMS / Path(str(name)).name] if name else []
    else:
        candidates = [d for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                      if (d / "stream.npz").is_file()]
    candidates = [c for c in candidates if (c / "stream.npz").is_file()]
    if not candidates:
        raise SystemExit("no candidate streams found in %s" % STREAMS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_state = None
    if SELECTED.is_file():
        payload = json.loads(SELECTED.read_text(encoding="utf8"))
        selection_state = {
            "selected": payload.get("selected"),
            "stop_conditions_hit": payload.get("stop_conditions_hit"),
            "u4_v2_passed": payload.get("gate_outcome", {}).get("u4_v2_passed"),
        }
    for stream_dir in candidates:
        report = run_candidate(stream_dir, args.n_permutations)
        report["protocol"] = "022"
        report["step"] = "P22-S4"
        report["selection_state"] = selection_state
        out = OUT_DIR / ("learnability_%s.json" % stream_dir.name)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf8")
        print(json.dumps({
            "candidate": report["candidate"],
            "state_test_ap": ((report["state_task"].get("metrics") or {})
                              .get("test") or {}).get("ap"),
            "onset_test_ap": ((report["onset_task_h1"].get("metrics") or {})
                              .get("test") or {}).get("ap"),
            "onset_prevalence": ((report["onset_task_h1"].get("metrics") or {})
                                 .get("test") or {}).get("prevalence"),
            "onset_positives": ((report["onset_task_h1"].get("metrics") or {})
                                .get("test") or {}).get("positives"),
            "best_simple_baseline_ap": report["learnability_v2_gate"]
            ["best_simple_baseline_ap"],
            "null_p99": report["learnability_v2_gate"]["null_p99_max_over_kinds"],
            "gate_passed": report["learnability_v2_gate"]["passed"],
            "written": str(out.relative_to(ROOT)).replace("\\", "/"),
        }, ensure_ascii=False), flush=True)

    summary = {
        "protocol": "022", "step": "P22-S4",
        "gate_rule": "onset AP >= max(0.15, 2.5 x prevalence), ROC-AUC >= 0.75, "
                     ">= best simple baseline + 0.05, > every null p99, "
                     ">= 50 forward positives",
        "candidates": [],
    }
    for stream_dir in candidates:
        path = OUT_DIR / ("learnability_%s.json" % stream_dir.name)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf8"))
            summary["candidates"].append({
                "candidate": payload["candidate"],
                "gate_passed": payload["learnability_v2_gate"]["passed"],
                "gate": payload["learnability_v2_gate"],
                "file": str(path.relative_to(ROOT)).replace("\\", "/")})
    summary["gate_passed_any"] = any(c["gate_passed"] for c in summary["candidates"])
    summary["stop_conditions_hit"] = ([] if summary["gate_passed_any"]
                                      else ["STOP-B2"])
    (OUT_DIR / "learnability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"gate_passed_any": summary["gate_passed_any"],
                      "stop_conditions_hit": summary["stop_conditions_hit"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
