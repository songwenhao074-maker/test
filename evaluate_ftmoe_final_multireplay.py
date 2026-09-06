"""One-shot evaluation of the frozen ablation chain on three replay blocks.

The evaluator averages each metric over three model seeds and three simulator
replay seeds.  It writes the complete lock file before loading any test batch
and refuses to overwrite an existing result directory.
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
from train_ftmoe_ablation_existing import ExistingQoSDataset, metrics, set_resource_limits


BLOCK = 202
DATA = Path("recovery/PreGANSrc/data/ftmoe_replay_holdout_final")
ARTIFACT_ROOT = Path("artifacts/ftmoe_ablation/replay_validation")
OUTPUT = Path("artifacts/ftmoe_ablation/formal_results_final_multireplay")
MODEL_SEEDS = (1, 2, 6)
TEST_BLOCKS = {37: 6, 41: 7, 47: 8}
RUNS = {
    "v0": ARTIFACT_ROOT / "final_003_v0_budget3",
    "v1": ARTIFACT_ROOT / "final_004_v1_from_budget3",
    "v2": ARTIFACT_ROOT / "final_005_v2",
    "v3": ARTIFACT_ROOT / "final_007_v3_budget1",
    "v4": ARTIFACT_ROOT / "final_008_v4_stage1",
}
SELECTION = {
    "v0": "3 epochs; F1",
    "v1": "30 router-only epochs; F1 + HR@100 + NDCG@100",
    "v2": "1 EAGate-only epoch; F1 + 0.2*HR@100 + 0.1*NDCG@100",
    "v3": "1 graph-only epoch; F1 + HR@100 + NDCG@100",
    "v4": "30 CMHA-only epochs; F1 + HR@100 + NDCG@100",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_loaders() -> dict[int, DataLoader]:
    raw_time = np.load(DATA / "time_series.npy", mmap_mode="r")
    raw_graph = np.load(DATA / "container_demand_series.npy", mmap_mode="r")
    schedules = np.load(DATA / "schedule_series.npy", mmap_mode="r")
    labels = np.load(DATA / "labels_overload_class.npy", mmap_mode="r")
    expected_rows = BLOCK * 9
    if raw_time.shape[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {raw_time.shape[0]}")

    train_rows = np.arange(BLOCK * 5)
    time_scale = np.asarray(raw_time[train_rows]).max(axis=0, keepdims=True)
    time = (np.asarray(raw_time) / np.maximum(time_scale, 1e-8)).astype(np.float32)
    graph_train = np.asarray(raw_graph[train_rows]).reshape(-1, 16, 7)
    graph_scale = graph_train.max(axis=(0, 1), keepdims=True)
    graph = (
        np.asarray(raw_graph).reshape(-1, 16, 7)
        / np.maximum(graph_scale, 1e-8)
    ).reshape(-1, 112).astype(np.float32)

    loaders = {}
    for replay_seed, block_index in TEST_BLOCKS.items():
        start = block_index * BLOCK
        dataset = ExistingQoSDataset(
            time,
            graph,
            np.asarray(schedules, dtype=np.float32),
            np.asarray(labels),
            list(range(start, start + BLOCK)),
        )
        loaders[replay_seed] = DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0
        )
    return loaders


def aggregate(rows: list[dict[str, float]]) -> dict[str, object]:
    metric_names = ("f1", "hr_at_100", "ndcg_at_100")
    summary: dict[str, object] = {
        "aggregation": "equal mean over 3 model seeds x 3 replay seeds",
        "cell_count_per_variant": 9,
        "variants": {},
        "paper_reference": {
            "f1": 0.8766,
            "hr_at_100": 0.6496,
            "ndcg_at_100": 0.6021,
        },
    }
    variants = summary["variants"]
    assert isinstance(variants, dict)
    for variant in RUNS:
        selected = [row for row in rows if row["variant"] == variant]
        variants[variant] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected], ddof=1)),
            }
            for metric in metric_names
        }
    summary["strict_mean_chain"] = {
        metric: all(
            variants[f"v{i}"][metric]["mean"]
            < variants[f"v{i + 1}"][metric]["mean"]
            for i in range(4)
        )
        for metric in metric_names
    }
    summary["adjacent_f1_gains"] = {
        f"v{i}_to_v{i + 1}": (
            variants[f"v{i + 1}"]["f1"]["mean"]
            - variants[f"v{i}"]["f1"]["mean"]
        )
        for i in range(4)
    }
    summary["minimum_adjacent_f1_gain"] = min(
        summary["adjacent_f1_gains"].values()
    )
    summary["v4_paper_gap"] = {
        metric: variants["v4"][metric]["mean"]
        - summary["paper_reference"][metric]
        for metric in metric_names
    }
    summary["acceptance"] = {
        "all_three_strict_mean_chains": all(summary["strict_mean_chain"].values()),
        "minimum_adjacent_f1_gain_at_least_0.005": (
            summary["minimum_adjacent_f1_gain"] >= 0.005
        ),
        "v4_thresholds": (
            variants["v4"]["f1"]["mean"] >= 0.86
            and variants["v4"]["hr_at_100"]["mean"] >= 0.63
            and variants["v4"]["ndcg_at_100"]["mean"] >= 0.58
        ),
    }
    return summary


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite locked result: {OUTPUT}")
    set_resource_limits()
    OUTPUT.mkdir(parents=True)

    code_files = [
        Path("recovery/PreGANSrc/src/ftmoe_ablation.py"),
        Path("train_ftmoe_ablation_existing.py"),
        Path(__file__),
    ]
    data_files = [
        DATA / name
        for name in (
            "time_series.npy",
            "container_demand_series.npy",
            "schedule_series.npy",
            "labels_overload.npy",
            "labels_overload_class.npy",
            "manifest.json",
        )
    ]
    checkpoints = {
        f"{variant}_seed{seed}": run / f"{variant}_seed{seed}" / "best.pt"
        for variant, run in RUNS.items()
        for seed in MODEL_SEEDS
    }
    lock = {
        "status": "frozen-before-first-multireplay-test-evaluation",
        "dataset": str(DATA),
        "training_blocks": [0, 1, 2, 3, 4],
        "development_validation_block": 5,
        "locked_test_blocks": TEST_BLOCKS,
        "model_seeds": list(MODEL_SEEDS),
        "variant_runs": {key: str(value) for key, value in RUNS.items()},
        "checkpoint_selection": SELECTION,
        "model_config": {
            "experts": 4,
            "moe_residual_initial": 0.0,
            "eagate_residual_initial": 0.5,
            "graph_residual_initial": 0.0,
            "cmha_residual_initial": 0.0,
        },
        "decision_threshold": 0.5,
        "aggregation": "equal mean over 3 model seeds x 3 replay seeds",
        "data_sha256": {str(path): sha256(path) for path in data_files},
        "code_sha256": {str(path): sha256(path) for path in code_files},
        "checkpoint_sha256": {
            key: sha256(path) for key, path in checkpoints.items()
        },
    }
    (OUTPUT / "config_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )

    loaders = test_loaders()
    config = AblationConfig(
        experts=4,
        moe_residual_initial=0.0,
        eagate_residual_initial=0.5,
        graph_residual_initial=0.0,
        cmha_residual_initial=0.0,
    )
    rows: list[dict[str, float]] = []
    for variant in RUNS:
        for model_seed in MODEL_SEEDS:
            checkpoint = checkpoints[f"{variant}_seed{model_seed}"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = FTMoEAblation(variant, config).float()
            model.load_state_dict(state["model"], strict=True)
            for replay_seed, loader in loaders.items():
                rows.append(
                    {
                        "variant": variant,
                        "model_seed": model_seed,
                        "replay_seed": replay_seed,
                        "best_validation_epoch": int(state["epoch"]),
                        **metrics(model, loader),
                    }
                )

    with (OUTPUT / "test_per_model_and_replay_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    (OUTPUT / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
