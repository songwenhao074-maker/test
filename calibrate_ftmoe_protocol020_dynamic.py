"""Calibrate D's novelty/error signals on a separate stationary dev stream.

No weights are trained. Quantiles are computed on the first 400 intervals;
the configuration is frozen before drift comparisons.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_ftmoe_protocol020 import ReplayV3, load_v2_time_scale_p20
from run_ftmoe_online import sha, write_json
from run_ftmoe_protocol019 import resources
from recovery.PreGANSrc.src.ftmoe_online_s8 import OnlineFTMoEV3


def calibrate(checkpoint_path, stream, output):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    resources()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads((stream / "manifest.json").read_text())
    if manifest.get("kind") != "stationary":
        raise ValueError("Calibration must use a stationary stream")
    if manifest["stream_sha256"] != sha(stream / "stream.npz"):
        raise ValueError("Calibration stream hash mismatch")
    with np.load(stream / "stream.npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["capacities_per_interval"] = arrays.pop("capacities")
    arrays["capacities"] = arrays["capacities_per_interval"][0]
    scale, _ = load_v2_time_scale_p20(checkpoint["normalization"])
    replay = ReplayV3(arrays, scale, checkpoint["normalization"]["graph_scale"], manifest["steps"])
    model = OnlineFTMoEV3(checkpoint, "A", checkpoint["seed"])
    entropy, margin, losses = [], [], []
    for t in range(min(400, replay.steps)):
        x, s, g, ids, before, caps = replay.window_v3(t)
        out = model.predict_online(x[None], s[None], g[None], graph_context={
            "creation_ids": ids[None], "before_placement": before[None], "capacities": caps[None]})
        entropy.append(model.eagate.last_observation["entropy"].numpy().ravel())
        margin.append(model.eagate.last_observation["margin"].numpy().ravel())
        p = out["detection_logits"].softmax(-1)[0, :, 1].numpy().clip(1e-7, 1 - 1e-7)
        raw = arrays["raw_labels"]
        positive = (raw[max(0, t - 1):t + 2] > 0).any(0)
        losses.append(float(np.mean(-np.where(positive, np.log(p), np.log1p(-p)))))
    entropy, margin = np.asarray(entropy), np.asarray(margin)
    ep, mp = float(np.percentile(entropy, 95)), float(np.percentile(margin, 5))
    novelty = ((entropy > ep) | (margin < mp)).mean(1)
    windows = [float(novelty[t:t + 10].mean()) for t in range(0, len(novelty), 10)]
    config = {"version": 3, "capacity_aware": False, "shadow_lr": 1e-4,
        "calibration": {"entropy_p95": ep, "margin_p05": mp,
            "novelty_mean": float(novelty.mean()),
            "novelty_threshold": max(float(np.percentile(windows, 95)), .10),
            "loss_baseline": float(np.mean(losses)),
            "stream": str(stream.resolve()), "stream_sha256": manifest["stream_sha256"],
            "checkpoint_sha256": sha(checkpoint_path), "intervals": len(entropy)},
        "birth": {"minimum_novel_states": 128, "consecutive_windows": 3, "loss_ratio": 1.25},
        "shadow": {"training_updates": 10, "validation_after_updates": 3,
                   "minimum_future_loss_improvement": .005, "ramp_updates": 10}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, config)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.checkpoint, args.stream, args.output)
