"""Pre-pilot Protocol-024 lifecycle loss-threshold calibration.

This is a development-only pass, not a performance arm.  It uses the selected
common live-update budget, the same independent ``raw_next_fault`` anchor, and a
fixed 600-interval prefix (F0=300 + first 300 intervals of R1).  The prefix is
chosen before looking at D-C performance specifically so calibration cannot be
restricted to the zero-fault F0 block.

The pass runs D with lifecycle OFF.  Once the first 600 predictions have
physically matured (cursor 602 under the publication-delay convention), it
computes the registered p99 of 32-distinct-interval mean supervised losses.
That numeric value is written into the runtime lifecycle config consumed by the
formal pilot.  The formal controller still waits through the same 600 matured
intervals before monitoring, so the precomputed value is never actionable
before all calibration observations would causally have arrived.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_anchor import (
    anchor_metadata, load_protocol024_raw_next_anchor)
from ftmoe_protocol024_lifecycle import _loss_from_logits
from ftmoe_protocol024_session import Protocol024Session, NEXT_TARGET_MODE


CALIBRATION_INTERVALS = 600
TRIGGER_WINDOW = 32
LOSS_PERCENTILE = 99.0


def _phases(manifest):
    return [{"name": p["name"], "start": int(p["start"]), "end": int(p["end"]),
             "response_law": p.get("response_law")}
            for p in manifest["timeline"]]


def _budget(update_every):
    source = json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))["frozen_configuration"]
    out = dict(source)
    out.update({"update_every_scored_intervals": int(update_every),
                "batch_size": 32,
                "gradient_steps_per_opportunity": 1,
                "replay_buffer_intervals": 64,
                "learning_rate": 1e-4})
    return out


def run(stream_dir, budget_path, lifecycle_path, out_path):
    stream_dir = Path(stream_dir)
    lifecycle_path = Path(lifecycle_path)
    out_path = Path(out_path)
    selected = json.loads(Path(budget_path).read_text(encoding="utf8"))
    update_every = int(selected["selected_update_every"])
    config = json.loads(lifecycle_path.read_text(encoding="utf8"))
    if int(config["calibration_intervals"]) != CALIBRATION_INTERVALS:
        raise ValueError("lifecycle config must preregister 600 calibration intervals")
    if int(config["trigger_window"]) != TRIGGER_WINDOW:
        raise ValueError("lifecycle config trigger window drifted")
    if float(config["loss_percentile"]) != LOSS_PERCENTILE:
        raise ValueError("lifecycle config percentile drifted")

    bundle = s4.build_replay(stream_dir)
    if int(bundle["steps"]) != 4980:
        raise ValueError("calibration requires the registered 4980-step stream")
    anchor = load_protocol024_raw_next_anchor()
    out_dir = out_path.parent / "calibration_session"
    session = Protocol024Session(
        "D", 1, bundle, _budget(update_every), out_dir,
        anchor=anchor, learning_rate=1e-4,
        run_id="seed700_loss_threshold_calibration_v1",
        stream_dir=stream_dir, phase_defs=_phases(bundle["manifest"]),
        target_mode=NEXT_TARGET_MODE,
        stream_sha=bundle["manifest"]["stream_sha256"],
        registration={"kind": "lifecycle_threshold_calibration",
                      "development_only": True,
                      "formal_performance_result": False,
                      "calibration_intervals": CALIBRATION_INTERVALS,
                      "trigger_window": TRIGGER_WINDOW,
                      "loss_percentile": LOSS_PERCENTILE,
                      "anchor": anchor_metadata(anchor)})

    # Prediction t's raw[t+1] target is available for training/settlement at
    # t+2, so cursor 602 is required to settle indices 0..599.
    while session.cursor < CALIBRATION_INTERVALS + 2:
        session.step()
    targets = np.asarray(session.predictions["labels"][:CALIBRATION_INTERVALS],
                         dtype=np.int64)
    if (targets < 0).any():
        raise AssertionError("calibration prefix contains an unsettled target")
    positives = int((targets > 0).sum())
    normals = int((targets == 0).sum())
    if positives <= 0 or normals <= 0:
        raise RuntimeError(
            "calibration prefix must contain both normal and fault host-steps; "
            "got positives=%d normals=%d" % (positives, normals))

    losses = []
    for index in range(CALIBRATION_INTERVALS):
        losses.append(_loss_from_logits(
            session.predictions["detection_logits"][index],
            session.predictions["class_logits"][index], targets[index]))
    losses = np.asarray(losses, dtype=np.float64)
    means = np.asarray([
        losses[i:i + TRIGGER_WINDOW].mean()
        for i in range(0, len(losses) - TRIGGER_WINDOW + 1)
    ], dtype=np.float64)
    threshold = float(np.percentile(means, LOSS_PERCENTILE))
    if not np.isfinite(threshold):
        raise RuntimeError("calibrated lifecycle loss threshold is non-finite")

    config["fixed_loss_threshold"] = threshold
    config["calibration_source"] = {
        "kind": "separate_development_lifecycle_off_pass",
        "stream_sha256": bundle["manifest"]["stream_sha256"],
        "model_seed": 1,
        "replay_seed": 700,
        "selected_update_every": update_every,
        "prefix_intervals": [0, CALIBRATION_INTERVALS],
        "target": NEXT_TARGET_MODE,
        "positive_host_steps": positives,
        "normal_host_steps": normals,
        "loss_window_count": int(len(means)),
        "anchor": anchor_metadata(anchor),
        "D_C_performance_not_read_for_calibration": True,
    }
    lifecycle_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n",
                              encoding="utf8")

    result = {
        "protocol": "024",
        "kind": "lifecycle_loss_threshold_calibration",
        "development_only": True,
        "formal_performance_result": False,
        "target": NEXT_TARGET_MODE,
        "stream_sha256": bundle["manifest"]["stream_sha256"],
        "model_seed": 1,
        "replay_seed": 700,
        "selected_update_every": update_every,
        "calibration_intervals": CALIBRATION_INTERVALS,
        "trigger_window": TRIGGER_WINDOW,
        "loss_percentile": LOSS_PERCENTILE,
        "fixed_loss_threshold": threshold,
        "positive_host_steps": positives,
        "normal_host_steps": normals,
        "interval_loss_mean": float(losses.mean()),
        "interval_loss_min": float(losses.min()),
        "interval_loss_max": float(losses.max()),
        "window_loss_mean": float(means.mean()),
        "window_loss_min": float(means.min()),
        "window_loss_max": float(means.max()),
        "anchor": anchor_metadata(anchor),
        "lifecycle_actions_enabled": False,
        "D_C_performance_not_read": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--lifecycle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.stream, args.budget, args.lifecycle, args.out)


if __name__ == "__main__":
    main()
