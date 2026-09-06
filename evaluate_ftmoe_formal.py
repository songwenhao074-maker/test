"""One-shot evaluation of the frozen FT-MoE ablation chain.

This script evaluates only replay block 6 (rows 404:606).  Normalization uses
the development training rows.  It writes per-seed metrics, aggregate means,
checkpoint hashes and a configuration lock before reporting the result.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, FTMoEAblation
from train_ftmoe_ablation_existing import ExistingQoSDataset, metrics


DATA = Path("recovery/PreGANSrc/data/ftmoe_tol1_with_demands")
ARTIFACT_ROOT = Path("artifacts/ftmoe_ablation/existing_data")
OUTPUT = Path("artifacts/ftmoe_ablation/formal_results")
SEEDS = (1, 2, 6)
RUNS = {
    "v0": ARTIFACT_ROOT / "iteration_002_default",
    "v1": ARTIFACT_ROOT / "iteration_003_v1_e4_gain025",
    "v2": ARTIFACT_ROOT / "iteration_014_v2_resource_heads",
    "v3": ARTIFACT_ROOT / "formal_009_v3_budget7",
    "v4": ARTIFACT_ROOT / "formal_012_v4_interaction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_test() -> DataLoader:
    raw_time = np.load(DATA / "time_series.npy", mmap_mode="r")
    raw_graph = np.load(DATA / "container_demand_series.npy", mmap_mode="r")
    schedules = np.load(DATA / "schedule_series.npy", mmap_mode="r")
    labels = np.load(DATA / "labels_overload_class.npy", mmap_mode="r")
    train_rows = list(range(0, 162)) + list(range(202, 364))
    time_scale = np.asarray(raw_time[train_rows]).max(axis=0, keepdims=True)
    time = (np.asarray(raw_time) / np.maximum(time_scale, 1e-8)).astype(np.float32)
    graph_train = np.asarray(raw_graph[train_rows]).reshape(-1, 16, 7)
    graph_scale = graph_train.max(axis=(0, 1), keepdims=True)
    graph = (np.asarray(raw_graph).reshape(-1, 16, 7) /
             np.maximum(graph_scale, 1e-8)).reshape(-1, 112).astype(np.float32)
    dataset = ExistingQoSDataset(
        time, graph, np.asarray(schedules, dtype=np.float32),
        np.asarray(labels), list(range(404, 606)),
    )
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
    lock = {
        "status": "frozen-before-test-evaluation",
        "dataset": str(DATA),
        "locked_rows": [404, 606],
        "model_seeds": list(SEEDS),
        "variant_runs": {key: str(value) for key, value in RUNS.items()},
        "model_config": {
            "experts": 4, "moe_residual_initial": 0.25,
            "eagate_residual_initial": 0.5,
            "graph_residual_initial": 0.0,
            "cmha_residual_initial": 0.0,
        },
        "selection": "best validation F1; fixed decision threshold 0.5",
        "data_sha256": {str(path): sha256(path) for path in data_files},
        "code_sha256": {str(path): sha256(path) for path in code_files},
    }
    (OUTPUT / "config_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )

    loader = load_test()
    rows = []
    config = AblationConfig(
        experts=4, moe_residual_initial=0.25,
        eagate_residual_initial=0.5,
        graph_residual_initial=0.0, cmha_residual_initial=0.0,
    )
    for variant, run_root in RUNS.items():
        for seed in SEEDS:
            checkpoint = run_root / f"{variant}_seed{seed}" / "best.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = FTMoEAblation(variant, config).float()
            incompatible = model.load_state_dict(state["model"], strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"unexpected checkpoint keys for {variant}/{seed}: "
                    f"{incompatible.unexpected_keys}"
                )
            result = metrics(model, loader)
            rows.append({
                "variant": variant, "seed": seed,
                "best_validation_epoch": int(state["epoch"]),
                "checkpoint_sha256": sha256(checkpoint),
                **result,
            })

    fields = list(rows[0].keys())
    with (OUTPUT / "test_per_seed.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = ["f1", "hr_at_100", "ndcg_at_100"]
    summary = {"variants": {}, "paper_reference": {
        "f1": 0.8766, "hr_at_100": 0.6496, "ndcg_at_100": 0.6021,
    }}
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
    (OUTPUT / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
