"""Build the simulator-derived FT-MoE dataset with causal container demands.

The original replay export overwrote model input with an external trace after
each step.  That removed the container demands which the schedule actually
places.  ``dump_replay.py`` now records those demands before execution.  This
builder concatenates three deterministic replays without deriving features
from labels or post-hoc predictions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 1, 6)
ROOT = Path("recovery/PreGANSrc/data")
OUTPUT = ROOT / "ftmoe_formal_hybrid_multi600"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    features, container_demands, schedules, labels = [], [], [], []
    sources = []
    for seed in SEEDS:
        source = ROOT / f"qos_overload_formal_multi600_s{seed}"
        replay_path = source / "replay_log.npz"
        replay = np.load(replay_path)
        demand = replay["container_demands"].astype(np.float32)
        schedule = np.load(source / "schedule_series.npy").astype(np.float32)
        label = np.load(source / "labels_overload_class.npy").astype(np.int64)
        if demand.shape != (202, 16, 7):
            raise ValueError(f"unexpected demand shape for seed {seed}")
        if schedule.shape != (202, 16, 16) or label.shape != (202, 16):
            raise ValueError(f"unexpected replay shape for seed {seed}")
        features.append(np.load(source / "time_series.npy").astype(np.float32))
        container_demands.append(demand.reshape(202, 112))
        schedules.append(schedule)
        labels.append(label)
        sources.append({
            "seed": seed,
            "directory": str(source),
            "replay_sha256": sha256(replay_path),
            "positive_host_steps": int((label > 0).sum()),
            "class_counts": {
                str(cls): int((label == cls).sum()) for cls in (1, 2, 3)
            },
        })

    feature_array = np.concatenate(features)
    demand_array = np.concatenate(container_demands)
    schedule_array = np.concatenate(schedules)
    label_array = np.concatenate(labels)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "time_series.npy", feature_array)
    np.save(OUTPUT / "container_demand_series.npy", demand_array)
    np.save(OUTPUT / "schedule_series.npy", schedule_array)
    np.save(OUTPUT / "labels_overload_class.npy", label_array)
    np.save(OUTPUT / "labels_overload.npy", (label_array > 0).astype(np.int64))
    manifest = {
        "name": "ftmoe_formal_hybrid_multi600",
        "shape": {
            "time_series": list(feature_array.shape),
            "container_demand_series": list(demand_array.shape),
            "schedule_series": list(schedule_array.shape),
            "labels": list(label_array.shape),
        },
        "graph_feature_semantics": [
            "container_base_ips", "container_ram_size", "container_ram_read",
            "container_ram_write", "container_disk_size",
            "container_disk_read", "container_disk_write",
        ],
        "label_semantics": {
            "0": "normal", "1": "CPU capacity violation",
            "2": "RAM capacity violation", "3": "disk capacity violation",
        },
        "simulator_configuration": {
            "RAM_SCALE": 3.5, "DISK_SCALE": 600.0,
            "DISK_CAP_SCALE": 0.15, "LOAD_SCALE": 1.5,
        },
        "replay_order": list(SEEDS),
        "validation_protocol": {
            "development_blocks": [42, 1],
            "train_positions_per_block": [0, 162],
            "validation_positions_per_block": [162, 202],
            "locked_test_block": 6,
        },
        "sources": sources,
        "warning": (
            "Disk capacity and demand scales are simulator configuration, not "
            "post-hoc label edits. Features contain pre-execution container "
            "demands; labels contain post-placement physical capacity outcomes."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
