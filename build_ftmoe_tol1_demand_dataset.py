"""Attach causal container-demand features to the existing tol1 dataset.

The labels, schedules and temporal input remain byte-identical to
``qos_overload_ms_tol1``.  Container demands come from deterministic fixed-
schedule replay, so the graph branch can aggregate the workload actually
placed by each recorded schedule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("recovery/PreGANSrc/data")
SOURCE = ROOT / "qos_overload_ms_tol1"
OUTPUT = ROOT / "ftmoe_tol1_with_demands"
SEEDS = (42, 1, 6)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    demands = []
    replay_sources = []
    for seed in SEEDS:
        directory = ROOT / f"qos_overload_fixed_rs35_s{seed}"
        replay_path = directory / "replay_log.npz"
        replay = np.load(replay_path)
        demand = replay["container_demands"].astype(np.float32)
        recorded_schedule = np.load(directory / "schedule_series.npy")
        source_schedule = np.load(
            ROOT / f"qos_overload_rs35_s{seed}" / "schedule_series.npy"
        )
        if not np.array_equal(recorded_schedule, source_schedule):
            raise ValueError(f"fixed replay schedule mismatch for seed {seed}")
        recorded_label = np.load(directory / "labels_overload.npy")
        source_label = np.load(
            ROOT / f"qos_overload_rs35_s{seed}" / "labels_overload.npy"
        )
        if not np.array_equal(recorded_label, source_label):
            raise ValueError(f"fixed replay label mismatch for seed {seed}")
        demands.append(demand.reshape(202, 112))
        replay_sources.append({
            "seed": seed, "directory": str(directory),
            "replay_sha256": digest(replay_path),
        })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ["time_series.npy", "schedule_series.npy",
                 "labels_overload.npy", "labels_overload_class.npy"]:
        np.save(OUTPUT / name, np.load(SOURCE / name))
    demand_array = np.concatenate(demands)
    np.save(OUTPUT / "container_demand_series.npy", demand_array)
    manifest = {
        "name": "ftmoe_tol1_with_demands",
        "base_dataset": str(SOURCE),
        "base_file_sha256": {
            name: digest(SOURCE / name) for name in [
                "time_series.npy", "schedule_series.npy",
                "labels_overload.npy", "labels_overload_class.npy",
            ]
        },
        "container_demand_shape": list(demand_array.shape),
        "fixed_schedule_replays": replay_sources,
        "simulator_configuration": {
            "RAM_SCALE": 3.5, "DISK_SCALE": 1.0,
            "DISK_CAP_SCALE": 1.0, "LOAD_SCALE": 1.5,
        },
        "split": {
            "train_positions_per_development_block": [0, 162],
            "validation_positions_per_development_block": [162, 202],
            "locked_test_block": 6,
        },
        "causality": (
            "Graph features are container demands sampled before execution. "
            "Schedules are the original decisions. Exact replay labels match "
            "the original source before the existing tol1 dilation."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
