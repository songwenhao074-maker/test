"""Train nested FT-MoE ablations on repository QoS replay datasets.

This runner is for validation-time ablation iteration.  It uses only the
first two 202-row replay blocks from ``qos_overload_ms_tol1``.  Their first
162 time steps form training data and their final 40 time steps form
validation data.  The third replay block remains outside this runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig, FTMoEAblation


BLOCK = 202
TRAIN_END = 162
WINDOW = 12


class ExistingQoSDataset(Dataset):
    def __init__(self, features: np.ndarray, graph_features: np.ndarray,
                 schedules: np.ndarray,
                 labels: np.ndarray,
                 rows: list[int]) -> None:
        self.features = features
        self.graph_features = graph_features
        self.schedules = schedules
        self.labels = labels
        self.rows = np.asarray(rows, dtype=np.int64)
        windows: list[list[int]] = []
        for row in self.rows:
            block_start = (int(row) // BLOCK) * BLOCK
            windows.append([
                max(block_start, int(row) - WINDOW + 1 + offset)
                for offset in range(WINDOW)
            ])
        self.windows = np.asarray(windows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Source layout [time, host*feature] becomes [host, time, feature].
        history = self.features[self.windows[item]].reshape(WINDOW, 16, 7)
        x = torch.from_numpy(history.transpose(1, 0, 2).copy())
        graph_history = self.graph_features[self.windows[item]].reshape(
            WINDOW, 16, 7
        )
        graph_x = torch.from_numpy(graph_history.transpose(1, 0, 2).copy())
        schedule = torch.from_numpy(self.schedules[self.windows[item]].copy())
        y = torch.from_numpy(self.labels[self.rows[item]].copy())
        return x, graph_x, schedule, y


def set_resource_limits() -> str:
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ["MKL_NUM_THREADS"] = "3"
    os.environ["OPENBLAS_NUM_THREADS"] = "3"
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "BelowNormal"
    except Exception as exc:
        return f"not-set:{type(exc).__name__}"


def resources() -> dict[str, float]:
    import psutil
    return {
        "available_ram_gib": psutil.virtual_memory().available / 1024 ** 3,
        "free_disk_gib": shutil.disk_usage("F:/").free / 1024 ** 3,
        "process_rss_gib": psutil.Process().memory_info().rss / 1024 ** 3,
    }


def loss_fn(model: FTMoEAblation, output: dict[str, torch.Tensor],
            labels: torch.Tensor, detection_weight: float,
            classification_weight: float, prototype_weight: float,
            balance_weight: float, positive_class_weight: float,
            ranking_weight: float) -> torch.Tensor:
    anomaly = (labels > 0).long()
    # Fixed weights reflect the existing training split's 11.3% positives.
    det_weight = torch.tensor(
        [0.60, positive_class_weight], dtype=torch.float32
    )
    detection = F.cross_entropy(
        output["detection_logits"].reshape(-1, 2), anomaly.reshape(-1),
        weight=det_weight,
    )
    positive = labels > 0
    classification = F.cross_entropy(
        output["class_logits"][positive], labels[positive] - 1,
    ) if positive.any() else detection.new_zeros(())
    prototype, balance = model.auxiliary_losses(output)
    ranking = detection.new_zeros(())
    if ranking_weight > 0 and positive.any() and (~positive).any():
        anomaly_probability = torch.softmax(
            output["detection_logits"], dim=-1,
        )[..., 1]
        class_probability = torch.softmax(output["class_logits"], dim=-1)
        true_index = (labels.clamp_min(1) - 1).unsqueeze(-1)
        true_class_probability = class_probability.gather(
            -1, true_index,
        ).squeeze(-1)
        positive_score = (
            anomaly_probability[positive] * true_class_probability[positive]
        )
        negative_score = (
            anomaly_probability[~positive] *
            class_probability[~positive].max(dim=-1).values
        )
        ranking = F.softplus(
            0.15 + negative_score.unsqueeze(0) - positive_score.unsqueeze(1)
        ).mean()
    return (detection_weight * detection +
            classification_weight * classification +
            prototype_weight * prototype + balance_weight * balance +
            ranking_weight * ranking)


@torch.no_grad()
def metrics(model: FTMoEAblation, loader: DataLoader) -> dict[str, float]:
    model.eval()
    probabilities, anomalies, class_probabilities, classes = [], [], [], []
    for x, graph_x, schedule, labels in loader:
        output = model(x.float(), schedule.float(), graph_x.float())
        probabilities.append(torch.softmax(output["detection_logits"], -1)[..., 1])
        anomalies.append(labels > 0)
        class_probabilities.append(torch.softmax(output["class_logits"], -1))
        classes.append(labels)
    probability = torch.cat(probabilities).flatten().numpy()
    anomaly = torch.cat(anomalies).flatten().numpy().astype(bool)
    class_probability = torch.cat(class_probabilities).reshape(-1, 3).numpy()
    target_class = torch.cat(classes).flatten().numpy()
    prediction = probability >= 0.5
    tp = int((prediction & anomaly).sum())
    fp = int((prediction & ~anomaly).sum())
    fn = int((~prediction & anomaly).sum())
    tn = int((~prediction & ~anomaly).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    score = probability * class_probability.max(axis=1)
    predicted_class = class_probability.argmax(axis=1) + 1
    relevance = (anomaly & (predicted_class == target_class)).astype(np.float32)
    order = np.argsort(-score)[:100]
    ideal = np.sort(relevance)[::-1][:100]
    hr = float(relevance[order].sum() / max(min(int(anomaly.sum()), 100), 1))
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((relevance[order] * discounts).sum())
    idcg = float((ideal * discounts[:len(ideal)]).sum())
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "hr_at_100": hr,
        "ndcg_at_100": dcg / max(idcg, 1e-12),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["v0", "v1", "v2", "v3", "v4"])
    parser.add_argument("--seed", type=int, choices=[1, 2, 6], required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--detection-loss-weight", type=float, default=0.55)
    parser.add_argument("--classification-loss-weight", type=float, default=0.35)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.10)
    parser.add_argument("--selection-hr-weight", type=float, default=0.05)
    parser.add_argument("--selection-ndcg-weight", type=float, default=0.05)
    parser.add_argument("--positive-class-weight", type=float, default=1.40)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--moe-residual-initial", type=float, default=0.25)
    parser.add_argument("--eagate-residual-initial", type=float, default=0.15)
    parser.add_argument("--graph-residual-initial", type=float, default=0.50)
    parser.add_argument("--cmha-residual-initial", type=float, default=0.05)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--train-new-module-only", action="store_true")
    parser.add_argument("--data", default=(
        "recovery/PreGANSrc/data/qos_overload_ms_tol1"
    ))
    parser.add_argument("--output", default="artifacts/ftmoe_ablation/existing_data")
    parser.add_argument("--normalization", choices=["per-column", "per-feature"],
                        default="per-column")
    parser.add_argument("--split-mode", choices=["temporal", "replay"],
                        default="temporal")
    parser.add_argument("--replay-training-blocks", type=int, default=2)
    args = parser.parse_args()
    if args.epochs > 60:
        raise ValueError("validation iteration is capped at 60 epochs")

    priority = set_resource_limits()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    data_path = Path(args.data)
    raw_features = np.load(data_path / "time_series.npy", mmap_mode="r")
    demand_path = data_path / "container_demand_series.npy"
    raw_graph_features = (np.load(demand_path, mmap_mode="r")
                          if demand_path.is_file() else raw_features)
    raw_schedules = np.load(data_path / "schedule_series.npy", mmap_mode="r")
    labels = np.load(data_path / "labels_overload_class.npy", mmap_mode="r")
    row_count = raw_features.shape[0]
    if (raw_features.shape != (row_count, 112) or row_count % BLOCK or
            row_count < 3 * BLOCK or
            raw_schedules.shape != (row_count, 16, 16) or
            labels.shape != (row_count, 16) or
            raw_graph_features.shape != (row_count, 112)):
        raise ValueError("unexpected QoS replay shapes")

    if args.split_mode == "temporal":
        train_rows = (list(range(0, TRAIN_END)) +
                      list(range(BLOCK, BLOCK + TRAIN_END)))
        validation_rows = (list(range(TRAIN_END, BLOCK)) +
                           list(range(BLOCK + TRAIN_END, 2 * BLOCK)))
        working_end = 2 * BLOCK
        split_description = {
            "mode": "within-development-block temporal holdout",
            "source_replay_blocks": ["42", "1"],
            "train_time_rows_per_block": [0, 162],
            "validation_time_rows_per_block": [162, 202],
            "locked_unused_replay_blocks": list(range(2, row_count // BLOCK)),
        }
    else:
        training_blocks = args.replay_training_blocks
        if training_blocks < 1 or row_count < (training_blocks + 2) * BLOCK:
            raise ValueError("replay split requires training, validation, and test blocks")
        train_rows = list(range(0, training_blocks * BLOCK))
        validation_rows = list(range(
            training_blocks * BLOCK, (training_blocks + 1) * BLOCK,
        ))
        working_end = (training_blocks + 1) * BLOCK
        split_description = {
            "mode": "whole-replay holdout",
            "training_block_indices": list(range(training_blocks)),
            "validation_block_index": training_blocks,
            "locked_unused_block_indices": list(range(
                training_blocks + 1, row_count // BLOCK,
            )),
        }
    # Normalize only with training rows.  Never copy locked block into a dataset.
    if args.normalization == "per-column":
        scale = np.asarray(raw_features[train_rows]).max(axis=0, keepdims=True)
        features = (np.asarray(raw_features[:working_end]) /
                    np.maximum(scale, 1e-8)).astype(np.float32)
    else:
        train_values = np.asarray(raw_features[train_rows]).reshape(-1, 16, 7)
        scale = train_values.max(axis=(0, 1), keepdims=True)
        working = np.asarray(raw_features[:working_end]).reshape(-1, 16, 7)
        features = (working / np.maximum(scale, 1e-8)).reshape(-1, 112).astype(np.float32)
    graph_train = np.asarray(raw_graph_features[train_rows]).reshape(-1, 16, 7)
    graph_scale = graph_train.max(axis=(0, 1), keepdims=True)
    graph_working = np.asarray(raw_graph_features[:working_end]).reshape(-1, 16, 7)
    graph_features = (graph_working / np.maximum(graph_scale, 1e-8)).reshape(
        -1, 112
    ).astype(np.float32)
    schedules = np.asarray(raw_schedules[:working_end], dtype=np.float32)
    labels_working = np.asarray(labels[:working_end])
    train_set = ExistingQoSDataset(
        features, graph_features, schedules, labels_working, train_rows,
    )
    validation_set = ExistingQoSDataset(
        features, graph_features, schedules, labels_working, validation_rows,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=64,
                                   shuffle=False, num_workers=0)

    run_dir = Path(args.output) / args.run_name / f"{args.variant}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model = FTMoEAblation(args.variant, AblationConfig(
        experts=args.experts,
        moe_residual_initial=args.moe_residual_initial,
        eagate_residual_initial=args.eagate_residual_initial,
        graph_residual_initial=args.graph_residual_initial,
        cmha_residual_initial=args.cmha_residual_initial,
    )).float()
    load_info = None
    if args.initialize_from is not None:
        state = torch.load(args.initialize_from, map_location="cpu",
                           weights_only=False)
        incompatible = model.load_state_dict(state["model"], strict=False)
        load_info = {
            "checkpoint": str(args.initialize_from),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        if incompatible.unexpected_keys:
            raise ValueError(f"unexpected warm-start keys: {incompatible.unexpected_keys}")
    if args.train_new_module_only:
        prefixes = {
            "v1": ("moe.", "moe_gain"),
            "v2": ("eagate.",),
            "v3": ("graph_encoder.", "graph_gain"),
            "v4": ("cmha.", "cmha_norm.", "cmha_gain",
                   "cmha_interaction_proj.",
                   "cmha_detection_adapter.", "cmha_class_adapter."),
        }
        if args.variant not in prefixes or args.initialize_from is None:
            raise ValueError("module-only training requires a warm-started v1-v4")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes[args.variant])
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=5,
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - 5, 1), eta_min=1e-5,
        ),
    ], milestones=[5])

    log_fields = [
        "epoch", "seconds", "learning_rate", "train_loss", "f1",
        "precision", "recall", "accuracy", "hr_at_100", "ndcg_at_100",
        "tp", "fp", "fn", "tn",
        "available_ram_gib", "free_disk_gib", "process_rss_gib",
    ]
    initial_validation = metrics(model, validation_loader)
    best_score = (initial_validation["f1"] +
                  args.selection_hr_weight * initial_validation["hr_at_100"] +
                  args.selection_ndcg_weight * initial_validation["ndcg_at_100"])
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    samples: list[dict[str, float]] = []
    torch.save({"model": model.state_dict(), "epoch": 0,
                "validation": initial_validation}, run_dir / "best.pt")
    with (run_dir / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=log_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            sample = resources()
            samples.append(sample)
            if sample["available_ram_gib"] < 4.5:
                raise RuntimeError(f"RAM guard stopped before epoch {epoch}: {sample}")
            if sample["free_disk_gib"] < 20.0:
                raise RuntimeError(f"disk guard stopped before epoch {epoch}: {sample}")
            model.train()
            temperature = 1.0 - 0.8 * (epoch - 1) / max(args.epochs - 1, 1)
            if args.variant == "v2":
                model.set_eagate_temperature(temperature)
            total = 0.0
            for x, graph_x, schedule, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(x.float(), schedule.float(), graph_x.float())
                loss = loss_fn(
                    model, output, y.long(), args.detection_loss_weight,
                    args.classification_loss_weight,
                    args.prototype_loss_weight, args.balance_weight,
                    args.positive_class_weight, args.ranking_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
            scheduler.step()
            validation = metrics(model, validation_loader)
            row = {
                "epoch": epoch, "seconds": time.perf_counter() - start,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": total / max(len(train_loader), 1),
                **validation, **sample,
            }
            writer.writerow(row)
            handle.flush()
            # Select with all three requested metrics, using F1 as primary.
            score = (validation["f1"] +
                     args.selection_hr_weight * validation["hr_at_100"] +
                     args.selection_ndcg_weight * validation["ndcg_at_100"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale = 0
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "validation": validation}, run_dir / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                break

    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    selected = metrics(model, validation_loader)
    summary = {
        "variant": args.variant, "seed": args.seed,
        "dataset": data_path.name,
        "split": split_description,
        "best_epoch": best_epoch, "validation": selected,
        "configuration": {
            "epochs": args.epochs, "patience": args.patience,
            "learning_rate": args.learning_rate,
            "balance_weight": args.balance_weight,
            "detection_loss_weight": args.detection_loss_weight,
            "classification_loss_weight": args.classification_loss_weight,
            "prototype_loss_weight": args.prototype_loss_weight,
            "selection_hr_weight": args.selection_hr_weight,
            "selection_ndcg_weight": args.selection_ndcg_weight,
            "positive_class_weight": args.positive_class_weight,
            "ranking_weight": args.ranking_weight,
            "normalization": args.normalization,
            "experts": args.experts,
            "moe_residual_initial": args.moe_residual_initial,
            "eagate_residual_initial": args.eagate_residual_initial,
            "graph_residual_initial": args.graph_residual_initial,
            "cmha_residual_initial": args.cmha_residual_initial,
            "initialize_from": load_info,
            "train_new_module_only": args.train_new_module_only,
            "trainable_parameters": sum(p.numel() for p in trainable),
        },
        "resources": {
            "threads": 3, "num_workers": 0, "priority": priority,
            "minimum_available_ram_gib": min(s["available_ram_gib"] for s in samples),
            "maximum_process_rss_gib": max(s["process_rss_gib"] for s in samples),
            "elapsed_seconds": time.perf_counter() - start,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
