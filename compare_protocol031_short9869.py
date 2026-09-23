"""Compare the single C/D pair for the user-amended Protocol-031 9869-row horizon."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from run_ftmoe_protocol031_pilot import binary_detection_metrics, load_npz, metric_window, phases, write_json

COMPARATORS = ("C_fixed5", "D_nonblocking_reuse")
RECURRENCE = ("U_rec1", "V_rec1")


def compare(root, stream, scenario_registration, run_id):
    root = Path(root); stream = Path(stream)
    reg = json.loads(Path(scenario_registration).read_text(encoding="utf8"))
    sums = {n: json.loads((root / n / "summary.json").read_text(encoding="utf8")) for n in COMPARATORS}
    pred = {n: load_npz(root / n / "predictions.npz") for n in COMPARATORS}
    c, d = pred["C_fixed5"], pred["D_nonblocking_reuse"]
    if not np.array_equal(c["labels"], d["labels"]):
        raise AssertionError("Protocol031 short9869 comparator labels differ")
    if sums["C_fixed5"]["stream_sha256"] != sums["D_nonblocking_reuse"]["stream_sha256"]:
        raise AssertionError("Protocol031 short9869 stream identity differs")
    if sums["C_fixed5"]["initialization_shared_prefix_sha256"] != sums["D_nonblocking_reuse"]["initialization_shared_prefix_sha256"]:
        raise AssertionError("Protocol031 short9869 shared first4 initialization differs")
    y = c["labels"]
    manifest = json.loads((stream / "manifest.json").read_text(encoding="utf8"))
    pmap = {p["name"]: p for p in phases(manifest)}
    if not all(name in pmap for name in RECURRENCE):
        raise AssertionError("short9869 recurrence windows missing")

    rows, deltas, pooled = [], [], []
    positive = 0
    for name in RECURRENCE:
        p = pmap[name]; a, b = int(p["start"]), int(p["end"]); pooled.extend(range(a, b))
        cm = metric_window(c["probability"], y, a, b)
        dm = metric_window(d["probability"], y, a, b)
        valid = cm["ap"] is not None and dm["ap"] is not None
        delta = float(dm["ap"] - cm["ap"]) if valid else None
        if valid:
            deltas.append(delta); positive += int(delta > 0)
        prefixes = {}
        for width in (32, 64):
            cc = metric_window(c["probability"], y, a, min(b, a + width))
            dd = metric_window(d["probability"], y, a, min(b, a + width))
            prefixes[f"first{width}"] = {
                "C_fixed5": cc, "D_nonblocking_reuse": dd,
                "D_minus_C_ap": None if cc["ap"] is None or dd["ap"] is None else float(dd["ap"] - cc["ap"]),
                "D_minus_C_fpr": None if cc["fpr"] is None or dd["fpr"] is None else float(dd["fpr"] - cc["fpr"]),
            }
        rows.append({
            "phase": name, "service": p["regime"], "intervals": [a, b],
            "positive_rows": int((y[a:b] > 0).sum()), "negative_rows": int((y[a:b] == 0).sum()),
            "C_fixed5": cm, "D_nonblocking_reuse": dm,
            "D_minus_C_ap": delta,
            "D_minus_C_fpr": None if cm["fpr"] is None or dm["fpr"] is None else float(dm["fpr"] - cm["fpr"]),
            "valid": bool(valid), "prefixes": prefixes,
        })

    idx = np.asarray(pooled, dtype=np.int64)
    pcm = binary_detection_metrics(c["probability"][idx], y[idx], .5)
    pdm = binary_detection_metrics(d["probability"][idx], y[idx], .5)
    pooled_fpr_delta = None if pcm["fpr"] is None or pdm["fpr"] is None else float(pdm["fpr"] - pcm["fpr"])
    mean = float(np.mean(deltas)) if len(deltas) == len(RECURRENCE) else None

    wnames = [p["name"] for p in phases(manifest) if p["name"].startswith("W_")]
    wrows, wdeltas = [], []
    for name in wnames:
        p = pmap[name]; a, b = int(p["start"]), int(p["end"])
        cm = metric_window(c["probability"], y, a, b); dm = metric_window(d["probability"], y, a, b)
        delta = None if cm["ap"] is None or dm["ap"] is None else float(dm["ap"] - cm["ap"])
        if delta is not None: wdeltas.append(delta)
        wrows.append({"phase": name, "intervals": [a, b], "C_fixed5": cm, "D_nonblocking_reuse": dm, "D_minus_C_ap": delta})
    wmean = float(np.mean(wdeltas)) if len(wdeltas) == len(wnames) else None

    ref = reg["evaluation"]["development_reference"]
    signal = bool(
        mean is not None and mean >= float(ref["mean_AP_delta_min"]) and
        positive >= int(ref["positive_windows_min"]) and
        pooled_fpr_delta is not None and pooled_fpr_delta <= float(ref["pooled_recurrence_normal_FPR_delta_max"]) and
        wmean is not None and wmean >= float(ref["mean_W_block_AP_delta_min"])
    )
    cf = sums["C_fixed5"]["full"]["detection"]; df = sums["D_nonblocking_reuse"]["full"]["detection"]
    return {
        "protocol": "031", "plan_revision": 2, "run_id": str(run_id),
        "development_only": True, "confirmation_run": False, "test_run": False,
        "post_start_user_amendment": True, "short_horizon_rows": 9869,
        "not_original_six_window_preregistered_primary": True,
        "stream_sha256": sums["C_fixed5"]["stream_sha256"],
        "same_frozen_data_verified": True, "shared_first4_initialization_verified": True,
        "recurrence128": rows,
        "primary": {
            "metric": reg["evaluation"]["primary_metric"],
            "valid_windows": sum(r["valid"] for r in rows),
            "required_valid_windows": int(ref["required_valid_windows"]),
            "equal_weight_mean_D_minus_C_ap": mean,
            "positive_windows": positive,
            "pooled_recurrence_normal_fpr_C_fixed5": pcm["fpr"],
            "pooled_recurrence_normal_fpr_D_nonblocking_reuse": pdm["fpr"],
            "pooled_recurrence_normal_fpr_delta_D_minus_C": pooled_fpr_delta,
            "equal_weight_mean_W_block_D_minus_C_ap": wmean,
            "development_signal": signal,
            "development_reference": ref,
        },
        "W_blocks": wrows,
        "full_stream": {
            "C_fixed5": sums["C_fixed5"]["full"],
            "D_nonblocking_reuse": sums["D_nonblocking_reuse"]["full"],
            "D_minus_C_ap": None if cf["ap"] is None or df["ap"] is None else float(df["ap"] - cf["ap"]),
            "D_minus_C_fpr": None if cf["fpr"] is None or df["fpr"] is None else float(df["fpr"] - cf["fpr"]),
        },
        "phase_metrics": {n: sums[n]["phases"] for n in COMPARATORS},
        "lifecycle": sums["D_nonblocking_reuse"].get("lifecycle"),
        "cost_profile": {n: sums[n]["cost"] for n in COMPARATORS},
        "interpretation_limits": {
            "statistical_confirmation": False,
            "single_replay_seed": 700, "single_model_seed": 1,
            "shortened_after_generation_started": True,
            "original_six_window_claim_allowed": False,
            "comparison_scope": "C_fixed5 vs D_nonblocking_reuse on the 9869-row deterministic prefix only",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--stream", required=True)
    ap.add_argument("--scenario-registration", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--cost-output", required=True)
    a = ap.parse_args()
    comp = compare(a.root, a.stream, a.scenario_registration, a.run_id)
    write_json(a.output, comp); write_json(a.cost_output, comp["cost_profile"])
    print(json.dumps({"primary": comp["primary"], "short_horizon_rows": 9869}, indent=2))


if __name__ == "__main__":
    main()
