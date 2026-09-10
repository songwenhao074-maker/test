"""Protocol 021 P21-S5 — causal learnability probe for the cascade regime.

"Unseen" does not imply "learnable" (protocol §9).  Before any FT-MoE run this
script fits a deliberately simple, forward-only probe on deployable features and
checks whether the regime's fault state is predictable at all.

Two targets are reported, because the protocol and the existing FT-MoE runner
use different conventions and conflating them would be sloppy:

    target=next  (gate target, protocol §9.2 "t+1 detection")
                 features measured at t  ->  fault state with index t+1
                 past-fault feature       = matured label at t
    target=same  (comparability with the FT-MoE online runner)
                 pre-action features at t -> fault state after the action at t
                 past-fault feature       = matured label at t-1

The matured-label feature is always offset so that it can never contain the
target.  A dedicated test asserts that offset.

Inputs (only what is available at prediction time, §9.1):
    * current CPU/RAM/Disk demand/capacity ratio (host aggregate)
    * 1-step delta, 3-step slope, 6-step slope of each ratio
    * host occupancy (live containers on the host)
    * proposed migration mass from the scheduler matrix
    * past observed fault state (matured label, offset as above)

Forbidden (§9.1): phase id, regime id, cascade flag, cascade event id, future
demand, future capacity, unmatured labels.  The feature matrix is assembled from
an explicit column list, and build_matrix() refuses any forbidden name.

Split (§9.3): temporal only — 0-399 fit, 400-799 dev, 800-1199 held-forward test
for a 1200-step stream.  Never shuffled across time.

Gate (§9.4, evaluated on target=next):  AP >= max(0.20, 3 x prevalence)
AND (ROC-AUC >= 0.75 OR top-decile recall >= 0.40).

Integrity controls (diagnostic only, never used for the gate):
    * time-shuffled labels inside the fit split -> the temporal split must matter
    * leaky next-step capacity/ratio feature    -> the probe must react to leakage

Usage:
    python probe_ftmoe_protocol021_learnability.py
    python probe_ftmoe_protocol021_learnability.py --tag p025
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
P21 = ROOT / "artifacts/ftmoe_online/protocol_021"
STREAMS = P21 / "pilot_streams"
OUT_DIR = P21 / "learnability"
SELECTED = P21 / "unseen_data_audit/selected.json"

SPLITS = (("fit", 0, 400), ("dev", 400, 800), ("test", 800, 1200))
FEATURE_CPU, FEATURE_RAM, FEATURE_DISK = 0, 1, 4
FORBIDDEN = ("phase_id", "regime_id", "cascade_task_flag", "cascade_event_id",
             "future_demand", "future_capacity", "unmatured_label")


def ratio_series(features, capacities):
    """[T,H,3] load ratios from host-aggregate demand and host capacity."""
    dem = np.stack([features[:, :, FEATURE_CPU], features[:, :, FEATURE_RAM],
                    features[:, :, FEATURE_DISK]], axis=-1)
    return dem / np.maximum(capacities, 1e-9)


def occupancy(placement):
    occ = np.zeros((placement.shape[0], placement.shape[1]), dtype=np.float64)
    for t in range(placement.shape[0]):
        for h in placement[t]:
            if h >= 0:
                occ[t, int(h)] += 1
    return occ


def build_matrix(stream_path, steps, target="next"):
    """Assemble the probe matrix with an explicit, leak-checked column list."""
    with np.load(stream_path, allow_pickle=True) as z:
        features = np.asarray(z["host_features"], dtype=np.float64)[:steps]
        capacities = np.asarray(z["capacities"], dtype=np.float64)[:steps]
        labels = np.asarray(z["raw_labels"], dtype=np.int64)[:steps + 1]
        schedules = np.asarray(z["schedules"], dtype=np.float64)[:steps]
        after = np.asarray(z["after_placement"], dtype=np.int64)[:steps]
    ratios = ratio_series(features, capacities)
    occ = occupancy(after)
    mig = schedules.sum(axis=1)
    T, H, _ = ratios.shape

    names, columns = [], []

    def add(name, data):
        if any(token in name for token in FORBIDDEN):
            raise ValueError("forbidden feature name: %s" % name)
        names.append(name)
        columns.append(np.asarray(data, dtype=np.float64).reshape(-1))

    add("ratio_cpu", ratios[:, :, 0])
    add("ratio_ram", ratios[:, :, 1])
    add("ratio_disk", ratios[:, :, 2])
    for k in (1, 3, 6):
        for j, res in enumerate(("cpu", "ram", "disk")):
            delta = np.zeros_like(ratios[:, :, j])
            delta[k:] = (ratios[k:, :, j] - ratios[:-k, :, j]) / float(k)
            add("slope%d_%s" % (k, res), delta)
    add("occupancy", occ)
    add("migration_mass_in", mig)

    # matured past-fault state, always offset away from the target
    offset = 1 if target == "same" else 0
    past = np.zeros_like(labels[:T])
    past[offset:] = labels[:T - offset] if offset else labels[:T]
    if target == "same":
        past[0] = 0                      # no matured history for the first row
    for klass in range(4):
        add("past_fault_%d" % klass, (past == klass).astype(np.float64))

    X = np.stack(columns, axis=1)
    row_time = np.repeat(np.arange(T), H)
    row_host = np.tile(np.arange(H), T)
    y_all = labels[1:T + 1] if target == "next" else labels[:T]
    y = y_all.reshape(-1)
    X = X.reshape(T * H, -1)
    if target == "next":
        keep = row_time < steps - 1
    else:
        keep = np.ones_like(row_time, dtype=bool)
        keep &= row_time >= 1            # need a matured past label
    return {"X": X[keep], "y": y[keep], "t": row_time[keep],
            "host": row_host[keep], "names": names, "target": target}


def metrics(y_true, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    out = {"n": int(y_true.size), "prevalence": float(y_true.mean())}
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        out.update({"ap": None, "roc_auc": None, "top_decile_recall": None})
        return out
    out["ap"] = float(average_precision_score(y_true, scores))
    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    k = max(1, int(round(0.1 * y_true.size)))
    order = np.argsort(-scores)[:k]
    out["top_decile_recall"] = float(y_true[order].sum() / max(y_true.sum(), 1))
    return out


def fit_and_score(dataset):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = dataset["X"]
    y = (dataset["y"] > 0).astype(int)
    t = dataset["t"]
    masks = {name: (t >= lo) & (t < hi) for name, lo, hi in SPLITS}
    scaler = StandardScaler().fit(X[masks["fit"]])
    model = LogisticRegression(max_iter=4000, random_state=21021)
    model.fit(scaler.transform(X[masks["fit"]]), y[masks["fit"]])
    scores = model.predict_proba(scaler.transform(X))[:, 1]
    result = {name: metrics(y[masks[name]], scores[masks[name]])
              for name, _, _ in SPLITS}
    return model, scaler, scores, masks, result, y


def shuffled_label_control(dataset, masks, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(21021)
    y_perm = y.copy()
    fit = masks["fit"]
    y_perm[fit] = y[fit][rng.permutation(int(fit.sum()))]
    scaler = StandardScaler().fit(dataset["X"][fit])
    model = LogisticRegression(max_iter=4000, random_state=21021)
    model.fit(scaler.transform(dataset["X"][fit]), y_perm[fit])
    scores = model.predict_proba(scaler.transform(dataset["X"][masks["test"]]))[:, 1]
    return metrics(y[masks["test"]], scores)


def leaky_feature_control(dataset, masks, y):
    """Append the *next-step* CPU ratio: a probe that cannot exploit this would
    be broken, so this control validates the probe rather than the regime."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = dataset["X"]
    t = dataset["t"]
    leak = np.zeros((X.shape[0], 1))
    # reconstruct the next-step ratio per (t, host) from the stored columns:
    # ratio_cpu is column 0, so shift it forward by one time step within a host.
    ratio_cpu = X[:, 0]
    order = np.lexsort((t, dataset["host"]))
    shifted = np.zeros_like(ratio_cpu)
    hosts = dataset["host"]
    times = t
    index = {(int(h), int(tt)): i for i, (h, tt) in enumerate(zip(hosts, times))}
    for i, (h, tt) in enumerate(zip(hosts, times)):
        j = index.get((int(h), int(tt) + 1))
        shifted[i] = ratio_cpu[j] if j is not None else ratio_cpu[i]
    leak[:, 0] = shifted
    X_leak = np.concatenate([X, leak], axis=1)
    fit = masks["fit"]
    scaler = StandardScaler().fit(X_leak[fit])
    model = LogisticRegression(max_iter=4000, random_state=21021)
    model.fit(scaler.transform(X_leak[fit]), y[fit])
    scores = model.predict_proba(scaler.transform(X_leak[masks["test"]]))[:, 1]
    return metrics(y[masks["test"]], scores)


def diagnosis_accuracy(dataset, masks):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = dataset["X"]
    y_cls = dataset["y"]
    fit = masks["fit"]
    scaler = StandardScaler().fit(X[fit])
    clf = LogisticRegression(max_iter=4000, multi_class="multinomial",
                             random_state=21021)
    clf.fit(scaler.transform(X[fit]), y_cls[fit])
    pred = clf.predict(scaler.transform(X[masks["test"]]))
    truth = y_cls[masks["test"]]
    fault = truth > 0
    return {"multiclass_accuracy": float((pred == truth).mean()),
            "conditional_fault_class_accuracy":
                float((pred[fault] == truth[fault]).mean()) if fault.any() else None}


def run_candidate(stream_dir, target="next"):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    steps = int(manifest["steps"])
    dataset = build_matrix(stream_dir / "stream.npz", steps, target=target)
    model, scaler, scores, masks, result, y = fit_and_score(dataset)
    test = result["test"]
    prevalence = test["prevalence"]
    ap_min = max(0.20, 3.0 * prevalence)
    gate = {
        "ap_at_least_min": bool(test["ap"] is not None and test["ap"] >= ap_min),
        "ranking_ok": bool((test["roc_auc"] is not None and test["roc_auc"] >= 0.75)
                           or (test["top_decile_recall"] is not None
                               and test["top_decile_recall"] >= 0.40)),
    }
    gate["passed"] = all(gate.values())
    return {
        "candidate": stream_dir.name,
        "stream_sha256": manifest["stream_sha256"],
        "cascade_task_probability": manifest["cascade_task_probability"],
        "steps": steps,
        "target": target,
        "n_features": len(dataset["names"]),
        "feature_names": dataset["names"],
        "metrics": result,
        "random_baseline_ap": float(prevalence),
        "diagnosis": diagnosis_accuracy(dataset, masks),
        "gate": {"ap_min": ap_min, **gate},
        "controls": {
            "time_shuffled_label_ap": shuffled_label_control(dataset, masks, y)["ap"],
            "leaky_next_ratio_ap": leaky_feature_control(dataset, masks, y)["ap"],
            "note": "controls are diagnostics only; never used for the gate",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None,
                        help="candidate tag, e.g. p025; default = selected.json")
    parser.add_argument("--target", default="next", choices=("next", "same"))
    args = parser.parse_args()

    selection_state = None
    if args.tag:
        candidates = [d for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                      if d.name.startswith(args.tag)]
    elif SELECTED.is_file():
        payload = json.loads(SELECTED.read_text(encoding="utf8"))
        name = (payload.get("selected_stream_dir")
                or payload.get("best_data_gate_stream_dir"))
        if name:
            name = Path(str(name)).name
        selection_state = {
            "selected": payload.get("selected"),
            "stop_conditions_hit": payload.get("stop_conditions_hit"),
            "probe_run_as": ("post-STOP-A data-layer diagnostic"
                             if not payload.get("selected") else
                             "registered P21-S5 probe on the selected candidate"),
            "candidate_dir_used": name,
            "note": ("the probe is part of the required evidence set (§42) and "
                     "does not authorise the model stage; STOP-A still applies"),
        }
        if name:
            candidates = [STREAMS / name]
        else:
            candidates = [d for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                          if (d / "stream.npz").is_file()]
    else:
        candidates = [d for d in sorted(STREAMS.glob("p*_seed*_steps*"))
                      if (d / "stream.npz").is_file()]
    if not candidates:
        raise SystemExit("no candidate streams found in %s" % STREAMS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "021", "step": "P21-S5",
        "split": {"fit": [0, 400], "dev": [400, 800], "test": [800, 1200],
                  "note": "temporal only, never shuffled (protocol §9.3)"},
        "forbidden_inputs": list(FORBIDDEN),
        "gate_rule": {"ap_min": "max(0.20, 3 x prevalence)",
                      "also": "ROC-AUC >= 0.75 or top-decile recall >= 0.40",
                      "evaluated_on": "target=%s" % args.target},
        "selection_state": selection_state,
        "candidates": [],
    }
    for stream_dir in candidates:
        entry = run_candidate(stream_dir, target=args.target)
        report["candidates"].append(entry)
        print(json.dumps({"candidate": entry["candidate"], "target": args.target,
                          "test_ap": entry["metrics"]["test"]["ap"],
                          "ap_min": entry["gate"]["ap_min"],
                          "roc_auc": entry["metrics"]["test"]["roc_auc"],
                          "top_decile_recall":
                              entry["metrics"]["test"]["top_decile_recall"],
                          "gate_passed": entry["gate"]["passed"],
                          "shuffled_control_ap":
                              entry["controls"]["time_shuffled_label_ap"],
                          "leaky_control_ap":
                              entry["controls"]["leaky_next_ratio_ap"]},
                         ensure_ascii=False))

    report["gate_passed_any"] = any(c["gate"]["passed"] for c in report["candidates"])
    report["gate_passed_all"] = all(c["gate"]["passed"] for c in report["candidates"])
    suffix = "" if args.target == "next" else "_same"
    result_path = OUT_DIR / ("probe_result%s.json" % suffix)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf8")
    print(json.dumps({"written": str(result_path),
                      "gate_passed_any": report["gate_passed_any"],
                      "gate_passed_all": report["gate_passed_all"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
