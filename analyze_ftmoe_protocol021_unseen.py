"""Protocol 021 P21-S4 — quantitative U4 audit of the cascade pilot candidates.

Answers, with numbers instead of prose, whether the registered cascade regime is
*distributionally* novel (U4) relative to every auditable offline corpus, and
whether it is exactly disjoint from the Protocol-020 material.

Compared against (§8):
    P014  train/dev physical replays            artifacts/ftmoe_end_to_end/data/protocol_004_physical/*.npy
    P019  adaptation train/dev + stationary/drift streams
    P020  S6 adaptation train/dev episodes
    P020  S5/S7 development streams (dev500/dev501)

Reported per candidate:
    * resource statistics      demand percentiles, zero ratio, >familiar-ceiling share
    * temporal structure       host-aggregate ACF lag 1..11
    * cross-resource lag       Corr(CPU_t, RAM_t+k), Corr(CPU_t, Disk_t+k),
                               Corr(RAM_t, Disk_t+k) for k = 0..11
    * event structure          independent events, duration percentiles, inter-event
                               interval, burstiness, CPU->RAM / RAM->Disk transition
                               probabilities, transition matrix
    * exact overlap            12-step scoring-window hash overlap with P20 S6 == 0
    * source overlap           pilot cohort vs P20 S6 train/dev == empty
    * U4 gate                  §8.5, evaluated against the offline 95% range

Outputs:
    artifacts/ftmoe_online/protocol_021/unseen_data_audit/candidate_p0NN.json
    artifacts/ftmoe_online/protocol_021/unseen_data_audit/offline_reference.json
    artifacts/ftmoe_online/protocol_021/unseen_data_audit/selected.json

Usage:
    python analyze_ftmoe_protocol021_unseen.py
    python analyze_ftmoe_protocol021_unseen.py --candidates p015_seed600_steps1200
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P21 = ROOT / "artifacts/ftmoe_online/protocol_021"
STREAMS = P21 / "pilot_streams"
AUDIT = P21 / "unseen_data_audit"
P14 = ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical"
P19 = ROOT / "artifacts/ftmoe_online/protocol_019"
P20 = ROOT / "artifacts/ftmoe_online/protocol_020"
SPLIT_PATH = P20 / "vm_split.json"

FEATURE_CPU, FEATURE_RAM, FEATURE_DISK = 0, 1, 4
# Familiar (Protocol-020 baseline) adapter ceilings — a single task cannot
# exceed these anywhere in the offline chain.
FAMILIAR_CEILING = {"cpu": 1860.0, "ram": 1400.0, "disk": 9000.0}
LAGS = list(range(0, 12))
WINDOW = 12
OFFLINE_CORPORA_MAX = 40          # cap per-corpus host samples for speed


def as_host_features(arr, label="array"):
    """Normalise to [T, H, 7]; raise loudly on anything else."""
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 3 and out.shape[2] == 7:
        return out
    if out.ndim == 2 and out.shape[1] % 7 == 0:
        return out.reshape(out.shape[0], out.shape[1] // 7, 7)
    raise ValueError("%s is not a host-feature array: shape=%s" % (label, out.shape))


def load_host_features(path, prefer=("host_features", "time", "time_series")):
    """Return [T, H, 7] host-level features from a stream/episode file."""
    z = np.load(path, allow_pickle=True)
    for key in prefer:
        if key in z.files:
            arr = np.asarray(z[key], dtype=np.float64)
            if arr.ndim == 2:                      # [T, H*7]
                hosts = arr.shape[1] // 7
                arr = arr.reshape(arr.shape[0], hosts, 7)
            if arr.ndim == 3 and arr.shape[2] == 7:
                return arr
    if "demands" in z.files:                       # last resort: per-container
        arr = np.asarray(z["demands"], dtype=np.float64)
        if arr.ndim == 3 and arr.shape[2] == 7:
            return arr
    raise KeyError("no host-feature array in %s (keys=%s)" % (path, z.files))


def offline_corpora():
    """Return {name: [array[T,H,7], ...]} for every auditable offline corpus."""
    corpora = {}

    ts = P14 / "time_series.npy"
    if ts.is_file():
        arr = np.load(ts)
        corpora["p014_physical_train_dev"] = [
            as_host_features(arr[i], "p014 replay %d" % i)
            for i in range(arr.shape[0])]

    p19_raw = []
    for seed_dir in sorted((P19 / "adaptation_data/raw").glob("seed*_steps*")):
        f = seed_dir / "stream.npz"
        if f.is_file():
            p19_raw.append(load_host_features(f))
    for sub, name in (("stationary_streams", "p019_stationary"),
                      ("drift_streams", "p019_drift"),
                      ("dev_streams", "p019_dev")):
        for seed_dir in sorted((P19 / sub).glob("seed*_steps*")):
            f = seed_dir / "stream.npz"
            if f.is_file():
                p19_raw.append(load_host_features(f))
    if p19_raw:
        corpora["p019_adaptation_and_development"] = p19_raw

    eps = []
    for f in sorted((P20 / "adaptation_data/v1/episodes").glob("*.npz")):
        eps.append(load_host_features(f))
    if eps:
        corpora["p020_s6_adaptation_train_dev"] = eps

    streams = []
    for f in sorted((P20 / "drift_streams").glob("*/stream.npz")):
        streams.append(load_host_features(f))
    if streams:
        corpora["p020_s5_s7_development_streams"] = streams

    return corpora


def onset_response(features, min_jump=200.0, ram_window=(4, 14), disk_window=(8, 18)):
    """Mechanism-level response to a task's own CPU onset.

    POST-HOC DIAGNOSTIC — explicitly NOT part of the pre-registered U4 gate.

    Two instruments are reported side by side, because the registered ramp
    starts *at the familiar level* at its onset age (progressive onset, no
    instantaneous jump), so an exact-lag difference is biased against it:

        at_lag         RAM(t+4) - RAM(t-1)              (exact registered lag)
        window_max     max RAM over [t+4, t+14) - baseline
                                                    (shape-consistent window)

    The same detector and windows are applied to the offline corpora.
    A sustained burst counts once (a step whose previous step already jumped
    by the same margin is not a new onset).
    """
    arr = as_host_features(features)
    cpu = arr[:, :, FEATURE_CPU]
    ram = arr[:, :, FEATURE_RAM]
    disk = arr[:, :, FEATURE_DISK]
    T, H = cpu.shape
    horizon = max(disk_window[1], ram_window[1])
    events = []
    for h in range(H):
        for t in range(1, T - horizon - 1):
            jump = cpu[t, h] - cpu[t - 1, h]
            if jump < min_jump:
                continue
            if t >= 2 and (cpu[t - 1, h] - cpu[t - 2, h]) >= min_jump:
                continue                      # still the same onset
            events.append((t, h))
    if not events:
        return {"n_onsets": 0, "min_jump": min_jump,
                "note": "no CPU onset of this size exists in the corpus"}
    baseline_ram = np.array([ram[t - 1, h] for t, h in events])
    baseline_disk = np.array([disk[t - 1, h] for t, h in events])
    ram_at = np.array([ram[t + 4, h] for t, h in events]) - baseline_ram
    ram_win = np.array([ram[t + ram_window[0]:t + ram_window[1], h].max()
                        for t, h in events]) - baseline_ram
    disk_at = np.array([disk[t + 8, h] for t, h in events]) - baseline_disk
    disk_win = np.array([disk[t + disk_window[0]:t + disk_window[1], h].max()
                         for t, h in events]) - baseline_disk
    return {
        "n_onsets": len(events),
        "min_jump": min_jump,
        "ram_at_lag4_p50": float(np.percentile(ram_at, 50)),
        "ram_window_max_p50": float(np.percentile(ram_win, 50)),
        "ram_window_max_p90": float(np.percentile(ram_win, 90)),
        "ram_window_max_mean": float(ram_win.mean()),
        "share_ram_window_positive": float((ram_win > 0).mean()),
        "disk_at_lag8_p50": float(np.percentile(disk_at, 50)),
        "disk_window_max_p50": float(np.percentile(disk_win, 50)),
        "disk_window_max_p90": float(np.percentile(disk_win, 90)),
        "share_disk_window_positive": float((disk_win > 0).mean()),
        "ram_window": list(ram_window),
        "disk_window": list(disk_window),
    }


def resource_stats(features):
    """Per-container demand statistics from the [T,H,7] host aggregate is not
    possible; use the raw demand array when available (candidate streams)."""
    out = {}
    for name, idx in (("cpu", FEATURE_CPU), ("ram", FEATURE_RAM),
                      ("disk", FEATURE_DISK)):
        v = np.asarray(features)[:, :, idx]
        live = v[v > 0]
        if live.size == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(live.size),
            "p10": float(np.percentile(live, 10)),
            "p50": float(np.percentile(live, 50)),
            "p90": float(np.percentile(live, 90)),
            "p95": float(np.percentile(live, 95)),
            "p99": float(np.percentile(live, 99)),
            "max": float(live.max()),
            "zero_ratio": float((v <= 0).mean()),
            "share_above_familiar_ceiling":
                float((live > FAMILIAR_CEILING[name]).mean()),
        }
    return out


def demand_stats(demands):
    """Statistics over live per-container demands (the physically meaningful
    scale for what one task asks for)."""
    out = {}
    for name, idx in (("cpu", FEATURE_CPU), ("ram", FEATURE_RAM),
                      ("disk", FEATURE_DISK)):
        v = np.asarray(demands)[:, :, idx]
        live = v[v > 0]
        out[name] = {
            "n": int(live.size),
            "p10": float(np.percentile(live, 10)) if live.size else None,
            "p50": float(np.percentile(live, 50)) if live.size else None,
            "p90": float(np.percentile(live, 90)) if live.size else None,
            "p95": float(np.percentile(live, 95)) if live.size else None,
            "p99": float(np.percentile(live, 99)) if live.size else None,
            "max": float(live.max()) if live.size else None,
            "zero_ratio": float((v <= 0).mean()),
            "share_above_familiar_ceiling":
                float((live > FAMILIAR_CEILING[name]).mean()) if live.size else None,
        }
    return out


def acf(series, max_lag):
    """Autocorrelation (biased, denominator n) for lags 0..max_lag."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return [float("nan")] * (max_lag + 1)
    return [float(np.dot(x[:len(x) - k], x[k:]) / denom) for k in range(max_lag + 1)]


def host_series(features, idx):
    """Host-aggregate series per host -> list of 1-D arrays."""
    arr = np.asarray(features, dtype=np.float64)[:, :, idx]
    return [arr[:, h] for h in range(arr.shape[1])]


def cross_lag(features, idx_a, idx_b, lags=LAGS):
    """Mean over hosts of Corr(A_t, B_{t+k}) for each lag k."""
    a_all = np.asarray(features, dtype=np.float64)[:, :, idx_a]
    b_all = np.asarray(features, dtype=np.float64)[:, :, idx_b]
    per_lag = {k: [] for k in lags}
    for h in range(a_all.shape[1]):
        a = a_all[:, h]
        b = b_all[:, h]
        if a.std() <= 0 or b.std() <= 0:
            continue
        for k in lags:
            if k == 0:
                x, y = a, b
            else:
                x, y = a[:len(a) - k], b[k:]
            if x.std() <= 0 or y.std() <= 0:
                continue
            per_lag[k].append(float(np.corrcoef(x, y)[0, 1]))
    return {str(k): (float(np.mean(v)) if v else None) for k, v in per_lag.items()}


def event_structure(labels, steps):
    runs = []
    for h in range(labels.shape[1]):
        current, length, start = 0, 0, 0
        for t in range(steps):
            lab = int(labels[t, h])
            if lab > 0 and lab == current:
                length += 1
            else:
                if current > 0 and length:
                    runs.append({"host": h, "class": current, "start": start,
                                 "duration": length})
                current, length, start = lab, (1 if lab > 0 else 0), t
        if current > 0 and length:
            runs.append({"host": h, "class": current, "start": start,
                         "duration": length})
    durations = np.array([r["duration"] for r in runs], dtype=float)
    starts = sorted(r["start"] for r in runs)
    gaps = np.diff(starts) if len(starts) > 1 else np.array([])
    if durations.size:
        mean_d, std_d = durations.mean(), durations.std()
        burstiness = float((std_d - mean_d) / (std_d + mean_d)) if (std_d + mean_d) else None
    else:
        burstiness = None
    matrix = np.zeros((4, 4), dtype=int)
    for h in range(labels.shape[1]):
        seq = labels[:steps, h]
        for t in range(steps - 1):
            matrix[int(seq[t]), int(seq[t + 1])] += 1
    return {
        "independent_events": len(runs),
        "runs": runs,
        "duration_p50": float(np.percentile(durations, 50)) if durations.size else None,
        "duration_p90": float(np.percentile(durations, 90)) if durations.size else None,
        "duration_p95": float(np.percentile(durations, 95)) if durations.size else None,
        "inter_event_interval_mean": float(gaps.mean()) if gaps.size else None,
        "burstiness": burstiness,
        "transition_matrix_normal_to_fault": matrix[0].tolist(),
        "transition_matrix": matrix.tolist(),
    }


def elevation_transitions(features, steps):
    """P(elevated B at t+lag | elevated A at t) with familiar-ceiling thresholds."""
    out = {}
    a_cpu = np.asarray(features, dtype=np.float64)[:steps, :, FEATURE_CPU]
    a_ram = np.asarray(features, dtype=np.float64)[:steps, :, FEATURE_RAM]
    a_disk = np.asarray(features, dtype=np.float64)[:steps, :, FEATURE_DISK]
    pairs = (("cpu_to_ram", a_cpu, a_ram, 4, "cpu", "ram"),
             ("ram_to_disk", a_ram, a_disk, 8, "ram", "disk"),
             ("cpu_to_ram_k0", a_cpu, a_ram, 0, "cpu", "ram"))
    for name, a, b, lag, ka, kb in pairs:
        src = a[:-lag] if lag else a
        dst = b[lag:] if lag else b
        mask = src > FAMILIAR_CEILING[ka]
        total = int(mask.sum())
        hits = int((dst[mask] > FAMILIAR_CEILING[kb]).sum()) if total else 0
        out[name] = {"lag": lag, "conditioned_on": ka, "target": kb,
                     "n_source_elevated": total,
                     "n_target_elevated": hits,
                     "probability": (hits / total) if total else None}
    return out


def window_hashes(demands, window=WINDOW):
    """Exact content hashes of every sliding host window over the demand array."""
    arr = np.asarray(demands, dtype=np.float64)
    out = set()
    for t in range(arr.shape[0] - window + 1):
        block = np.ascontiguousarray(arr[t:t + window])
        out.add(hashlib.sha256(block.tobytes()).hexdigest())
    return out


def cohort_ids_of(path):
    z = np.load(path, allow_pickle=True)
    return z


def p20_s6_cohorts():
    """VM ids used by the P20 S6 adaptation episodes (train and dev)."""
    train, dev = set(), set()
    for split, target in (("train", train), ("dev", dev)):
        for man in sorted((P20 / "adaptation_data/raw" / split).glob("*/seed*_steps*/manifest.json")):
            payload = json.loads(man.read_text(encoding="utf8"))
            ids = payload.get("cohort_vm_ids")
            if ids:
                target.update(int(i) for i in ids)
    return train, dev


def analyse_candidate(stream_dir, offline, reference):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    audit = json.loads((stream_dir / "unseen_data_audit.json").read_text(encoding="utf8"))
    z = np.load(stream_dir / "stream.npz", allow_pickle=True)
    steps = int(manifest["steps"])
    features = np.asarray(z["host_features"], dtype=np.float64)[:steps]
    demands = np.asarray(z["demands"], dtype=np.float64)[:steps]
    labels = np.asarray(z["raw_labels"], dtype=np.int64)
    ratio = np.asarray(z["overload_ratio"], dtype=np.float64)[:steps]

    result = {
        "candidate": manifest["name"],
        "stream_dir": str(stream_dir.relative_to(ROOT)).replace("\\", "/"),
        "stream_sha256": manifest["stream_sha256"],
        "generator_sha256": manifest["source_sha256"].get(
            "simulator/workload/BitbrainWorkloadProtocol021.py"),
        "cascade_task_probability": manifest["cascade_task_probability"],
        "cohort": manifest["cohort"],
        "cohort_size": len(manifest["cohort_vm_ids"]),
        "steps": steps,
        # the collector writes the Pilot Data Gate object itself (no wrapper)
        "data_gate": audit,
        "resource_stats_host_aggregate": resource_stats(features),
        "resource_stats_per_container": demand_stats(demands),
        "ratio_tail": {
            name: {
                "share_ratio_gt_1": float((ratio[:, :, i] > 1).mean()),
                "max_ratio": float(ratio[:, :, i].max()),
            } for i, name in enumerate(("cpu", "ram", "disk"))
        },
        "acf_host_aggregate": {},
        "cross_lag_correlation": {},
        "event_structure": event_structure(labels, steps),
        "elevation_transitions": elevation_transitions(features, steps),
    }
    for name, idx in (("cpu", FEATURE_CPU), ("ram", FEATURE_RAM),
                      ("disk", FEATURE_DISK)):
        curves = [acf(s, 11) for s in host_series(features, idx)]
        curves = [c for c in curves if not np.isnan(c).any()]
        result["acf_host_aggregate"][name] = (
            [float(np.mean([c[k] for c in curves])) for k in range(12)]
            if curves else None)
    for name, (ia, ib) in (("cpu_ram", (FEATURE_CPU, FEATURE_RAM)),
                           ("cpu_disk", (FEATURE_CPU, FEATURE_DISK)),
                           ("ram_disk", (FEATURE_RAM, FEATURE_DISK))):
        result["cross_lag_correlation"][name] = cross_lag(features, ia, ib)

    # ---- exact overlap -------------------------------------------------
    cand_hashes = window_hashes(demands)
    s6_hashes = set()
    for f in sorted((P20 / "adaptation_data/v1/episodes").glob("*.npz")):
        with np.load(f, allow_pickle=True) as ep:
            s6_hashes |= window_hashes(np.asarray(ep["demands"], dtype=np.float64))
    p20_dev_hashes = set()
    for f in sorted((P20 / "drift_streams").glob("*/stream.npz")):
        with np.load(f, allow_pickle=True) as st:
            p20_dev_hashes |= window_hashes(np.asarray(st["demands"], dtype=np.float64))
    train_ids, dev_ids = p20_s6_cohorts()
    cohort = set(int(i) for i in manifest["cohort_vm_ids"])
    result["overlap"] = {
        "scoring_window_exact_overlap_p20_s6": len(cand_hashes & s6_hashes),
        "scoring_window_exact_overlap_p20_dev_streams": len(cand_hashes & p20_dev_hashes),
        "cohort_intersection_p20_s6_train": sorted(cohort & train_ids),
        "cohort_intersection_p20_s6_dev": sorted(cohort & dev_ids),
        "cohort_vs_other_p21_cohorts": "online cohort is disjoint from train/dev by vm_split construction",
    }

    # ---- U4 gate -------------------------------------------------------
    ref = reference
    lag_checks = {}
    for pair, lag in (("cpu_ram", "4"), ("cpu_disk", "8")):
        value = result["cross_lag_correlation"][pair].get(lag)
        hist = ref["cross_lag_95"][pair].get(lag, {})
        exceeds = (value is not None and hist.get("p97_5") is not None
                   and value > hist["p97_5"])
        lag_checks[pair] = {"lag": int(lag), "candidate": value,
                            "offline_p97_5": hist.get("p97_5"),
                            "offline_max": hist.get("max"),
                            "exceeds_offline_95_range": bool(exceeds)}
    peak_checks = {}
    for pair in ("cpu_ram", "cpu_disk", "ram_disk"):
        values = result["cross_lag_correlation"][pair]
        numeric = {int(k): v for k, v in values.items() if v is not None}
        if not numeric:
            continue
        peak_lag = max(numeric, key=lambda k: abs(numeric[k]))
        peak_checks[pair] = {"peak_lag": peak_lag, "peak_value": numeric[peak_lag]}
    trans = result["elevation_transitions"]
    offline_trans = ref["elevation_transitions"]
    trans_checks = {}
    for key in ("cpu_to_ram", "ram_to_disk"):
        cand = trans[key]["probability"]
        off = offline_trans.get(key, {}).get("max_probability")
        delta = (cand - off) if (cand is not None and off is not None) else None
        ratio = (cand / off) if (cand is not None and off not in (None, 0)) else None
        trans_checks[key] = {
            "candidate": cand, "offline_max": off,
            "absolute_difference": delta, "ratio": ratio,
            "passes_rule": bool(delta is not None and (delta >= 0.15 or
                                                       (ratio is not None and ratio >= 1.5))),
        }
    gate = {
        "lag_peak_exceeds_offline_95_range": bool(
            lag_checks["cpu_ram"]["exceeds_offline_95_range"]
            or lag_checks["cpu_disk"]["exceeds_offline_95_range"]),
        "transition_probability_differs": bool(
            trans_checks["cpu_to_ram"]["passes_rule"]
            or trans_checks["ram_to_disk"]["passes_rule"]),
        "window_overlap_zero": (result["overlap"]["scoring_window_exact_overlap_p20_s6"] == 0
                                and result["overlap"]["scoring_window_exact_overlap_p20_dev_streams"] == 0),
        "p20_s6_source_overlap_empty": (not result["overlap"]["cohort_intersection_p20_s6_train"]
                                        and not result["overlap"]["cohort_intersection_p20_s6_dev"]),
        "data_gate_passed": bool(audit["passed"]),
    }
    gate["passed"] = all(gate.values())
    result["u4_gate"] = {"checks": gate, "lag_checks": lag_checks,
                         "peak_checks": peak_checks,
                         "transition_checks": trans_checks,
                         "reference": "offline_reference.json"}
    result["claim_level"] = {
        "U1_source_disjoint": gate["p20_s6_source_overlap_empty"],
        "U2_parameter_combo_unseen": True,
        "U3_generator_mechanism_unseen": True,
        "U4_distributional_novelty": gate["passed"],
        "note": "reported separately per protocol §8.5; never collapsed into a single unseen flag",
    }
    # mechanism-level diagnostic: per-slot response to the task's own CPU onset
    # (post-hoc explanation of the aggregate gate, never used to override it)
    result["mechanism_diagnostic"] = {
        "status": "POST-HOC DIAGNOSTIC, NOT PART OF THE PRE-REGISTERED U4 GATE",
        "candidate": onset_response(features),
        "offline": reference.get("mechanism_diagnostic_offline"),
        "interpretation_rule": "if the candidate's per-slot +4 RAM / +8 Disk response clearly exceeds the offline corpora under the identical detector, the mechanism exists and the aggregate gate failed as an instrument; if it does not, the mechanism itself is not visible",
    }
    return result


def build_reference(offline):
    """Offline distributions used by the U4 gate."""
    reference = {"corpora": {}, "cross_lag_95": {}, "elevation_transitions": {}}
    pair_indices = {"cpu_ram": (FEATURE_CPU, FEATURE_RAM),
                    "cpu_disk": (FEATURE_CPU, FEATURE_DISK),
                    "ram_disk": (FEATURE_RAM, FEATURE_DISK)}
    pooled = {pair: {k: [] for k in LAGS} for pair in pair_indices}
    trans_pool = {"cpu_to_ram": [], "ram_to_disk": []}
    for name, arrays in offline.items():
        stats = {"n_streams": len(arrays),
                 "steps_total": int(sum(a.shape[0] for a in arrays)),
                 "hosts": int(arrays[0].shape[1]),
                 "cross_lag": {}, "elevation_transitions": {}}
        for pair, (ia, ib) in pair_indices.items():
            per_lag = {k: [] for k in LAGS}
            for arr in arrays[:OFFLINE_CORPORA_MAX]:
                res = cross_lag(arr, ia, ib)
                for k, v in res.items():
                    if v is not None:
                        per_lag[int(k)].append(v)
                        pooled[pair][int(k)].append(v)
            stats["cross_lag"][pair] = {
                str(k): (float(np.mean(v)) if v else None)
                for k, v in per_lag.items()}
        for arr in arrays[:OFFLINE_CORPORA_MAX]:
            tr = elevation_transitions(arr, arr.shape[0])
            for key in ("cpu_to_ram", "ram_to_disk"):
                p = tr[key]["probability"]
                if p is not None:
                    trans_pool[key].append(p)
                    stats["elevation_transitions"].setdefault(key, []).append(p)
        reference["corpora"][name] = stats
    for pair, per_lag in pooled.items():
        reference["cross_lag_95"][pair] = {}
        for k, values in per_lag.items():
            if not values:
                reference["cross_lag_95"][pair][str(k)] = {"n": 0}
                continue
            arr = np.asarray(values, dtype=float)
            reference["cross_lag_95"][pair][str(k)] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "p2_5": float(np.percentile(arr, 2.5)),
                "p97_5": float(np.percentile(arr, 97.5)),
                "max": float(arr.max()),
                "min": float(arr.min()),
            }
    for key, values in trans_pool.items():
        if values:
            arr = np.asarray(values, dtype=float)
            reference["elevation_transitions"][key] = {
                "n": int(arr.size), "mean": float(arr.mean()),
                "p97_5": float(np.percentile(arr, 97.5)),
                "max_probability": float(arr.max())}
        else:
            reference["elevation_transitions"][key] = {"n": 0, "max_probability": None}

    # mechanism-level (post-hoc) reference: same onset detector on offline data
    mech = {"status": "POST-HOC DIAGNOSTIC, NOT PART OF THE PRE-REGISTERED U4 GATE",
            "per_corpus": {}, "ram_window_max_p50_by_stream": [],
            "disk_window_max_p50_by_stream": [], "ram_at_lag4_p50_by_stream": [],
            "streams_with_onsets": 0}
    for name, arrays in offline.items():
        ram_p50, disk_p50, ram_at4, onsets = [], [], [], 0
        for arr in arrays[:OFFLINE_CORPORA_MAX]:
            res = onset_response(arr)
            onsets += res["n_onsets"]
            if res["n_onsets"]:
                ram_p50.append(res["ram_window_max_p50"])
                disk_p50.append(res["disk_window_max_p50"])
                ram_at4.append(res["ram_at_lag4_p50"])
        mech["per_corpus"][name] = {
            "streams": len(arrays), "onsets": int(onsets),
            "ram_window_max_p50_median": float(np.median(ram_p50)) if ram_p50 else None,
            "ram_window_max_p50_max": float(np.max(ram_p50)) if ram_p50 else None,
            "ram_at_lag4_p50_median": float(np.median(ram_at4)) if ram_at4 else None,
            "disk_window_max_p50_median": float(np.median(disk_p50)) if disk_p50 else None,
            "disk_window_max_p50_max": float(np.max(disk_p50)) if disk_p50 else None,
        }
        mech["ram_window_max_p50_by_stream"].extend(ram_p50)
        mech["disk_window_max_p50_by_stream"].extend(disk_p50)
        mech["ram_at_lag4_p50_by_stream"].extend(ram_at4)
        mech["streams_with_onsets"] += len(ram_p50)
    for key in ("ram_window_max", "disk_window_max", "ram_at_lag4"):
        values = mech["%s_p50_by_stream" % key]
        if values:
            arr = np.asarray(values, dtype=float)
            mech["%s_offline_summary" % key] = {
                "n_streams": int(arr.size), "p50": float(np.percentile(arr, 50)),
                "p97_5": float(np.percentile(arr, 97.5)), "max": float(arr.max())}
        else:
            mech["%s_offline_summary" % key] = {"n_streams": 0}
    reference["mechanism_diagnostic_offline"] = mech
    reference["definitions"] = {
        "host_aggregate": "per-host sum of live container demands (feature 0/1/4)",
        "cross_lag": "mean over hosts of Corr(A_t, B_t+k); the historical 95% range is the pooled distribution over all audited offline corpora and hosts",
        "elevated": "a single task's demand above the familiar Protocol-020 ceiling (cpu 1860, ram 1400, disk 9000) — unreachable offline for CPU by construction",
        "transition": "P(target elevated at t+lag | source elevated at t)",
    }
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()

    AUDIT.mkdir(parents=True, exist_ok=True)
    ref_path = AUDIT / "offline_reference.json"
    if args.skip_offline and ref_path.is_file():
        reference = json.loads(ref_path.read_text(encoding="utf8"))
    else:
        offline = offline_corpora()
        reference = build_reference(offline)
        ref_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf8")
    print(json.dumps({"offline_corpora": {k: v["n_streams"]
                                          for k, v in reference["corpora"].items()}},
                     ensure_ascii=False))

    names = args.candidates or [d.name for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                                if (d / "stream.npz").is_file()]
    results = []
    for name in names:
        stream_dir = STREAMS / name
        if not (stream_dir / "stream.npz").is_file():
            print("skip (incomplete):", name)
            continue
        result = analyse_candidate(stream_dir, None, reference)
        tag = "p%03d" % round(result["cascade_task_probability"] * 100)
        out = AUDIT / ("candidate_%s.json" % tag)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf8")
        results.append(result)
        print(json.dumps({"candidate": name, "data_gate": result["data_gate"]["passed"],
                          "u4_gate": result["u4_gate"]["checks"]["passed"],
                          "prevalence": result["data_gate"]["metrics"]["prevalence"],
                          "cascade_positives":
                              result["data_gate"]["metrics"]["cascade_related_positive_hoststeps"],
                          "independent_cascade_events":
                              result["data_gate"]["metrics"]["independent_cascade_fault_events"]},
                         ensure_ascii=False))

    if results:
        # selection uses ONLY data physics / event structure (protocol §7.1).
        # Gate outcomes dominate the ordering: a candidate that failed a gate
        # can never be "selected", however many events it produced.
        def score(r):
            m = r["data_gate"]["metrics"]
            return (1 if r["data_gate"]["passed"] else 0,
                    1 if r["u4_gate"]["checks"]["passed"] else 0,
                    m["independent_cascade_fault_events"],
                    -abs(m["prevalence"] - 0.05))
        ranked = sorted(results, key=score, reverse=True)
        best = ranked[0]
        gates_ok = bool(best["data_gate"]["passed"]
                        and best["u4_gate"]["checks"]["passed"])
        data_gate_passers = [r for r in ranked if r["data_gate"]["passed"]]
        stop = {}
        if not any(r["u4_gate"]["checks"]["passed"] for r in results):
            stop["STOP-A"] = ("U4 distributional novelty not established for any "
                              "candidate: the pre-registered host-aggregate lag / "
                              "transition gate is not exceeded; models may not be "
                              "run under the unseen name (protocol §39)")
        if not any(r["data_gate"]["passed"] for r in results):
            stop["STOP-data-gate"] = ("no candidate passed the Pilot Data Gate; "
                                      "register a second-round mechanism "
                                      "(cascade_v2) and repeat the full pilot")
        payload = {
            "protocol": "021", "step": "P21-S3/S4",
            "selection_rule": "data physics and event structure only (protocol §7.1); model scores were never consulted",
            "gate_outcome": {
                "data_gate_passed": [r["candidate"] for r in results if r["data_gate"]["passed"]],
                "data_gate_failed": [r["candidate"] for r in results if not r["data_gate"]["passed"]],
                "u4_gate_passed": [r["candidate"] for r in results if r["u4_gate"]["checks"]["passed"]],
                "u4_gate_failed": [r["candidate"] for r in results if not r["u4_gate"]["checks"]["passed"]],
            },
            "candidates": [
                {"candidate": r["candidate"],
                 "stream_dir_name": Path(r["stream_dir"]).name,
                 "cascade_task_probability": r["cascade_task_probability"],
                 "data_gate_passed": r["data_gate"]["passed"],
                 "u4_gate_passed": r["u4_gate"]["checks"]["passed"],
                 "u4_checks": r["u4_gate"]["checks"],
                 "prevalence": r["data_gate"]["metrics"]["prevalence"],
                 "cascade_related_positive_hoststeps":
                     r["data_gate"]["metrics"]["cascade_related_positive_hoststeps"],
                 "independent_cascade_fault_events":
                     r["data_gate"]["metrics"]["independent_cascade_fault_events"],
                 "deployment_rejection_rate":
                     r["data_gate"]["metrics"]["deployment_rejection_rate"],
                 "migration_rejection_rate":
                     r["data_gate"]["metrics"]["migration_rejection_rate"],
                 "worst_event_share":
                     r["data_gate"]["metrics"]["worst_event_share_of_positives"]}
                for r in results],
            "selected": best["candidate"] if gates_ok else None,
            "selected_tag": ("p%03d" % round(best["cascade_task_probability"] * 100)
                             if gates_ok else None),
            "selected_stream_dir": best["stream_dir"] if gates_ok else None,
            "selected_stream_sha256": best["stream_sha256"] if gates_ok else None,
            "selected_reason": ("both the Pilot Data Gate and the U4 gate passed"
                                if gates_ok else
                                "no candidate passed both gates; STOP-A applies"),
            "best_data_gate_candidate": (data_gate_passers[0]["candidate"]
                                         if data_gate_passers else None),
            "best_data_gate_stream_dir": (Path(data_gate_passers[0]["stream_dir"]).name
                                          if data_gate_passers else None),
            "stop_conditions_hit": [k for k in sorted(stop)],
            "stop_details": stop,
            "next_step_if_pass": "protocol §10: build the 3400-step development stream (familiar 500 / unseen 1200 / familiar return 500 / unseen recurrence 1200) and run frozen A vs C-residual-off/on; D remains blocked until fixed C shows a positive late-unseen signal",
            "next_step_if_fail": "STOP: register a second-round mechanism (cascade_v2) and/or a mechanism-level novelty measurement, then repeat the full 3-candidate pilot; no silent search expansion and no gate redefinition after seeing results",
        }
        (AUDIT / "selected.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"selected": payload["selected"],
                          "stop_conditions_hit": payload["stop_conditions_hit"]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
