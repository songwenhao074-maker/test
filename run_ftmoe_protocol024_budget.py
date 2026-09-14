"""Bounded Protocol-024 fixed-C update budget comparison (u4 versus u1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_session import Protocol024Session, NEXT_TARGET_MODE


FIRST_EXPOSURES = ("R1_first", "R2_first", "R3_first")


def _phases(manifest):
    return [{"name": p["name"], "start": int(p["start"]), "end": int(p["end"]),
             "response_law": p.get("response_law")}
            for p in manifest["timeline"]]


def _budget(update_every):
    source = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))["frozen_configuration"]
    out = dict(source)
    out["update_every_scored_intervals"] = int(update_every)
    out["batch_size"] = 32
    out["gradient_steps_per_opportunity"] = 1
    out["replay_buffer_intervals"] = 64
    out["learning_rate"] = 1e-4
    return out


def _run_one(bundle, anchor, stream_dir, out_dir, update_every):
    session = Protocol024Session(
        "C", 1, bundle, _budget(update_every), out_dir,
        anchor=anchor, learning_rate=1e-4,
        run_id="budget_u%d" % update_every,
        stream_dir=stream_dir, phase_defs=_phases(bundle["manifest"]),
        target_mode=NEXT_TARGET_MODE,
        stream_sha=bundle["manifest"]["stream_sha256"],
        registration={"kind": "bounded_C_budget_development",
                      "update_every": int(update_every),
                      "formal_performance_result": False})
    started = time.perf_counter()
    for _ in range(session.steps):
        session.step()
    session.finish()
    elapsed = time.perf_counter() - started
    metrics = session.phase_metrics()
    by_name = {x["phase"]: x for x in metrics}
    first_aps = [by_name[name]["first_100"]["detection"]["ap"]
                 for name in FIRST_EXPOSURES]
    valid = [x for x in first_aps if x is not None]
    update_seconds = [row["seconds"] for row in session.update_log]
    prediction_seconds = session.predictions["prediction_seconds"]
    summary = {
        "update_every": int(update_every),
        "updates": int(session.updates),
        "elapsed_seconds": float(elapsed),
        "prediction_mean_seconds": float(np.mean(prediction_seconds)),
        "prediction_p95_seconds": float(np.percentile(prediction_seconds, 95)),
        "update_mean_seconds": (float(np.mean(update_seconds)) if update_seconds else None),
        "update_p95_seconds": (float(np.percentile(update_seconds, 95)) if update_seconds else None),
        "first_exposure_first100_ap": dict(zip(FIRST_EXPOSURES, first_aps)),
        "first_exposure_mean_ap": (float(np.mean(valid)) if valid else None),
        "full_detection": session._metric_block(0, session.steps)["detection"],
        "phase_metrics": metrics,
        "frozen_base_unchanged": True,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return summary


def run(stream_dir, out_path):
    stream_dir, out_path = Path(stream_dir), Path(out_path)
    bundle = s4.build_replay(stream_dir)
    if bundle["steps"] != 4980:
        raise ValueError("budget comparison requires registered 4980-step stream")
    anchor = s4.load_anchor_pool(bundle["replay"].time_scale)
    root = out_path.parent / "runs"
    u4 = _run_one(bundle, anchor, stream_dir, root / "C_u4", 4)
    # rebuild replay/anchor session state from immutable data; the C model seed is
    # reset inside Protocol024Session so the only difference is update cadence.
    bundle_u1 = s4.build_replay(stream_dir)
    anchor_u1 = s4.load_anchor_pool(bundle_u1["replay"].time_scale)
    u1 = _run_one(bundle_u1, anchor_u1, stream_dir, root / "C_u1", 1)
    gain = (None if u4["first_exposure_mean_ap"] is None or
                    u1["first_exposure_mean_ap"] is None
            else float(u1["first_exposure_mean_ap"] - u4["first_exposure_mean_ap"]))
    choose_u1 = bool(gain is not None and gain > 0.05 and
                     u1["update_p95_seconds"] is not None and
                     np.isfinite(u1["update_p95_seconds"]))
    selected = 1 if choose_u1 else 4
    result = {
        "protocol": "024", "kind": "bounded_C_budget_comparison",
        "target": "raw_next_fault", "model_seed": 1, "replay_seed": 700,
        "candidate_u4": u4, "candidate_u1": u1,
        "u1_minus_u4_first_exposure_mean_ap": gain,
        "selection_rule": "select u1 only if first-exposure mean AP gain >0.05 and p95 update latency is finite; otherwise u4",
        "selected_update_every": selected,
        "selected_name": "bounded_fast_u1" if selected == 1 else "deployment_default_u4",
        "confirmation_run": False,
    }
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
