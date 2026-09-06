"""Build a nine-replay FT-MoE dataset with three unseen final blocks.

Blocks 42, 1, and 6 replay the repository's original schedules exactly and
add causal per-container demand features.  Later blocks are fresh simulator
runs.  Profile-class thresholds use only five training blocks.  Temporal
tolerance is applied inside each replay block, never across a boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("recovery/PreGANSrc/data")
OUTPUT = ROOT / "ftmoe_replay_holdout_final"
BLOCK = 202
HOSTS = 16
TOLERANCE = 1
RAM_AMPLIFICATION = 1.5
DISK_AMPLIFICATION = 1.5
SEEDS = (42, 1, 6, 17, 23, 31, 37, 41, 47)
TRAINING_SEEDS = (42, 1, 6, 17, 23)
SOURCES = {
    42: ROOT / "qos_overload_fixed_rs35_s42",
    1: ROOT / "qos_overload_fixed_rs35_s1",
    6: ROOT / "qos_overload_fixed_rs35_s6",
    17: ROOT / "qos_overload_formal_rs35_s17",
    23: ROOT / "qos_overload_formal_rs35_s23",
    31: ROOT / "qos_overload_formal_rs35_s31",
    37: ROOT / "qos_overload_formal_rs35_s37",
    41: ROOT / "qos_overload_formal_rs35_s41",
    47: ROOT / "qos_overload_formal_rs35_s47",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_classes(features: np.ndarray, overload: np.ndarray,
                    thresholds: np.ndarray) -> np.ndarray:
    classes = np.zeros_like(overload, dtype=np.int64)
    capacities = np.array([4029.0, 4295.0, 32212.0])
    for t in range(len(features)):
        for host in range(HOSTS):
            if not overload[t, host]:
                continue
            row = features[t, host * 7:(host + 1) * 7]
            if row[0] > thresholds[host, 0]:
                cls = 1
            elif row[1] > thresholds[host, 1] / RAM_AMPLIFICATION:
                cls = 2
            elif row[4] > thresholds[host, 4] / DISK_AMPLIFICATION:
                cls = 3
            else:
                ratios = [
                    row[0] * 18.6 / capacities[0],
                    row[1] * 1.4 / capacities[1],
                    min(row[4], 9.0) / capacities[2],
                ]
                cls = int(np.argmax(ratios)) + 1
            classes[t, host] = cls
    return classes


def dilate_one_block(overload: np.ndarray,
                     classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dilated = np.zeros_like(overload, dtype=np.int64)
    dilated_classes = np.zeros_like(classes, dtype=np.int64)
    for host in range(HOSTS):
        column = overload[:, host]
        for t in range(BLOCK):
            lo = max(0, t - TOLERANCE)
            hi = min(BLOCK, t + TOLERANCE + 1)
            candidates = np.flatnonzero(column[lo:hi])
            if candidates.size:
                nearest = candidates[np.argmin(np.abs(candidates - (t - lo)))] + lo
                dilated[t, host] = 1
                dilated_classes[t, host] = classes[nearest, host]
    return dilated, dilated_classes


def main() -> None:
    features_by_seed: dict[int, np.ndarray] = {}
    schedules_by_seed: dict[int, np.ndarray] = {}
    overload_by_seed: dict[int, np.ndarray] = {}
    demands_by_seed: dict[int, np.ndarray] = {}
    provenance = []
    for seed in SEEDS:
        directory = SOURCES[seed]
        features = np.load(directory / "time_series.npy").astype(np.float32)
        schedules = np.load(directory / "schedule_series.npy").astype(np.float32)
        overload = np.load(directory / "labels_overload.npy").astype(np.int64)
        replay_path = directory / "replay_log.npz"
        replay = np.load(replay_path)
        demands = replay["container_demands"].reshape(BLOCK, -1).astype(np.float32)
        if features.shape != (BLOCK, 112):
            raise ValueError(f"bad feature shape for seed {seed}: {features.shape}")
        if schedules.shape != (BLOCK, 16, 16):
            raise ValueError(f"bad schedule shape for seed {seed}: {schedules.shape}")
        if overload.shape != (BLOCK, 16) or demands.shape != (BLOCK, 112):
            raise ValueError(f"bad label/demand shape for seed {seed}")
        features_by_seed[seed] = features
        schedules_by_seed[seed] = schedules
        overload_by_seed[seed] = overload
        demands_by_seed[seed] = demands
        provenance.append({
            "seed": seed,
            "directory": str(directory),
            "replay_sha256": digest(replay_path),
        })

    threshold_source = np.concatenate([
        features_by_seed[seed] for seed in TRAINING_SEEDS
    ])
    thresholds = np.percentile(threshold_source, 98, axis=0).reshape(HOSTS, 7)
    labels = []
    for seed in SEEDS:
        classes = profile_classes(
            features_by_seed[seed], overload_by_seed[seed], thresholds,
        )
        _, dilated_classes = dilate_one_block(overload_by_seed[seed], classes)
        labels.append(dilated_classes)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays = {
        "time_series.npy": np.concatenate([features_by_seed[s] for s in SEEDS]),
        "schedule_series.npy": np.concatenate([schedules_by_seed[s] for s in SEEDS]),
        "labels_overload.npy": (np.concatenate(labels) > 0).astype(np.int64),
        "labels_overload_class.npy": np.concatenate(labels),
        "container_demand_series.npy": np.concatenate([
            demands_by_seed[s] for s in SEEDS
        ]),
    }
    for name, array in arrays.items():
        np.save(OUTPUT / name, array)

    manifest = {
        "name": OUTPUT.name,
        "block_size": BLOCK,
        "block_order": list(SEEDS),
        "split": {
            "training_blocks": list(TRAINING_SEEDS),
            "development_validation_block": 31,
            "locked_final_test_blocks": [37, 41, 47],
        },
        "label_policy": {
            "detection": "simulator overload dilated by one step within each block",
            "classification": "profile class at nearest exact overload",
            "threshold_fit_blocks": list(TRAINING_SEEDS),
            "percentile": 98,
            "ram_amplification": RAM_AMPLIFICATION,
            "disk_amplification": DISK_AMPLIFICATION,
        },
        "causality": (
            "Container demands and schedules are sampled before simulationStep; "
            "overload labels are sampled after placement.  Dilation never crosses "
            "a replay boundary."
        ),
        "sources": provenance,
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_sha256": {name: digest(OUTPUT / name) for name in arrays},
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    for index, seed in enumerate(SEEDS):
        block_labels = labels[index]
        counts = [int((block_labels == cls).sum()) for cls in range(1, 4)]
        print(f"seed={seed} positives={int((block_labels > 0).sum())} "
              f"classes={counts}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
