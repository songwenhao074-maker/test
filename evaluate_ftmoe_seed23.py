"""One-shot evaluation of the frozen ablation chain on replay seed 23."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, FTMoEAblation
from train_ftmoe_ablation_existing import (
    ExistingQoSDataset, metrics, set_resource_limits,
)


DATA = Path("recovery/PreGANSrc/data/ftmoe_replay_holdout_s23")
ARTIFACT_ROOT = Path("artifacts/ftmoe_ablation/replay_validation")
OUTPUT = Path("artifacts/ftmoe_ablation/formal_results_seed23")
SEEDS = (1, 2, 6)
RUNS = {
    "v0": ARTIFACT_ROOT / "s23_001_v0",
    "v1": ARTIFACT_ROOT / "s23_002_v1",
    "v2": ARTIFACT_ROOT / "s23_007_v2_budget1",
    "v3": ARTIFACT_ROOT / "s23_008_v3_graph",
    "v4": ARTIFACT_ROOT / "s23_009_v4",
}
SELECTION = {
    "v0": "F1",
    "v1": "F1 + 0.2*HR@100 + 0.1*NDCG@100",
    "v2": "one epoch; F1 + 0.2*HR@100 + 0.1*NDCG@100",
    "v3": "F1 + HR@100 + NDCG@100",
    "v4": "F1 + 5*HR@100 + 5*NDCG@100",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_loader() -> DataLoader:
    raw_time = np.load(DATA / "time_series.npy", mmap_mode="r")
    raw_graph = np.load(DATA / "container_demand_series.npy", mmap_mode="r")
    schedules = np.load(DATA / "schedule_series.npy", mmap_mode="r")
    labels = np.load(DATA / "labels_overload_class.npy", mmap_mode="r")
    train_rows = list(range(606))
    time_scale = np.asarray(raw_time[train_rows]).max(axis=0, keepdims=True)
    time = (np.asarray(raw_time) / np.maximum(time_scale, 1e-8)).astype(np.float32)
    graph_train = np.asarray(raw_graph[train_rows]).reshape(-1, 16, 7)
    graph_scale = graph_train.max(axis=(0, 1), keepdims=True)
    graph = (np.asarray(raw_graph).reshape(-1, 16, 7) /
             np.maximum(graph_scale, 1e-8)).reshape(-1, 112).astype(np.float32)
    dataset = ExistingQoSDataset(
        time, graph, np.asarray(schedules, dtype=np.float32),
        np.asarray(labels), list(range(808, 1010)),
    )
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite locked result: {OUTPUT}")
    set_resource_limits()
    OUTPUT.mkdir(parents=True)
    code_files = [
        Path("recovery/PreGANSrc/src/ftmoe_ablation.py"),
        Path("train_ftmoe_ablation_existing.py"), Path(__file__),
    ]
    data_files = [
        DATA / name for name in [
            "time_series.npy", "container_demand_series.npy",
            "schedule_series.npy", "labels_overload.npy",
            "labels_overload_class.npy", "manifest.json",
        ]
    ]
    checkpoints = {
        f"{variant}_seed{seed}": run / f"{variant}_seed{seed}" / "best.pt"
        for variant, run in RUNS.items() for seed in SEEDS
    }
    lock = {
        "status": "frozen-before-first-seed23-evaluation",
        "dataset": str(DATA),
        "training_rows": [0, 606],
        "development_validation_rows": [606, 808],
        "locked_test_rows": [808, 1010],
        "locked_test_replay_seed": 23,
        "model_seeds": list(SEEDS),
        "variant_runs": {key: str(value) for key, value in RUNS.items()},
        "checkpoint_selection": SELECTION,
        "model_config": {
            "experts": 4, "moe_residual_initial": 0.0,
            "eagate_residual_initial": 0.5,
            "graph_residual_initial": 0.0,
            "cmha_residual_initial": 0.0,
        },
        "decision_threshold": 0.5,
        "data_sha256": {str(path): sha256(path) for path in data_files},
        "code_sha256": {str(path): sha256(path) for path in code_files},
        "checkpoint_sha256": {
            key: sha256(path) for key, path in checkpoints.items()
        },
    }
    (OUTPUT / "config_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8",
    )

    loader = test_loader()
    config = AblationConfig(
        experts=4, moe_residual_initial=0.0,
        eagate_residual_initial=0.5,
        graph_residual_initial=0.0, cmha_residual_initial=0.0,
    )
    rows = []
    for variant in RUNS:
        for seed in SEEDS:
            checkpoint = checkpoints[f"{variant}_seed{seed}"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = FTMoEAblation(variant, config).float()
            incompatible = model.load_state_dict(state["model"], strict=False)
            if incompatible.unexpected_keys or incompatible.missing_keys:
                raise ValueError(
                    f"checkpoint mismatch for {variant}/{seed}: {incompatible}"
                )
            rows.append({
                "variant": variant, "seed": seed,
                "best_validation_epoch": int(state["epoch"]),
                **metrics(model, loader),
            })

    with (OUTPUT / "test_per_seed.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = ["f1", "hr_at_100", "ndcg_at_100"]
    summary = {
        "variants": {},
        "paper_reference": {
            "f1": 0.8766, "hr_at_100": 0.6496, "ndcg_at_100": 0.6021,
        },
    }
    for variant in RUNS:
        selected = [row for row in rows if row["variant"] == variant]
        summary["variants"][variant] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected], ddof=1)),
            } for metric in metric_names
        }
    summary["strict_mean_chain"] = {
        metric: all(
            summary["variants"][f"v{i}"][metric]["mean"] <
            summary["variants"][f"v{i + 1}"][metric]["mean"]
            for i in range(4)
        ) for metric in metric_names
    }
    summary["minimum_adjacent_f1_gain"] = min(
        summary["variants"][f"v{i + 1}"]["f1"]["mean"] -
        summary["variants"][f"v{i}"]["f1"]["mean"]
        for i in range(4)
    )
    (OUTPUT / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
