"""Train exactly one Stage-A FT-MoE screening variant (v0 or v1).

The runner deliberately loads train/validation episodes only.  It has no
code path that opens the screening test episodes, so no test-set selection can
occur during this pilot.
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

from recovery.PreGANSrc.src.ftmoe_ablation import FTMoEAblation


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, episodes: np.ndarray,
                 wanted_episodes: list[int]) -> None:
        self.features = features
        self.labels = labels
        indices = np.flatnonzero(np.isin(episodes, wanted_episodes))
        windows = []
        for index in indices:
            first = index - (index % 40)
            history = [max(first, index - 11 + offset) for offset in range(12)]
            windows.append(history)
        self.indices = indices
        self.windows = np.asarray(windows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        # [window, host, feature] -> [host, window, feature]
        x = torch.from_numpy(self.features[self.windows[item]].transpose(1, 0, 2).copy())
        y = torch.from_numpy(self.labels[self.indices[item]].copy())
        return x, y


def _set_resource_limits() -> tuple[bool, str]:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    try:
        import psutil
        proc = psutil.Process()
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return True, "BelowNormal"
    except Exception as exc:  # Resource priority is best effort on Windows.
        return False, f"not-set: {type(exc).__name__}"


def _resources_ok() -> tuple[bool, dict[str, float]]:
    import psutil
    available_gib = psutil.virtual_memory().available / 1024 ** 3
    free_gib = shutil.disk_usage(Path.cwd().anchor).free / 1024 ** 3
    process_rss_gib = psutil.Process().memory_info().rss / 1024 ** 3
    # The plan says do not begin a next epoch below 5 GiB, and safety-stop
    # below 3.5 GiB.  This condition chooses the stricter start gate.
    return available_gib >= 5.0 and free_gib >= 20.0, {
        "available_ram_gib": available_gib, "free_disk_gib": free_gib,
        "process_rss_gib": process_rss_gib,
    }


def _loss(model: FTMoEAblation, output: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    anomaly = (labels > 0).long()
    det_weight = torch.tensor([0.65, 1.35], dtype=torch.float32, device=labels.device)
    detection = F.cross_entropy(output["detection_logits"].reshape(-1, 2), anomaly.reshape(-1), weight=det_weight)
    positive = labels > 0
    if positive.any():
        classification = F.cross_entropy(output["class_logits"][positive], labels[positive] - 1)
    else:
        classification = detection.new_zeros(())
    prototype, balance = model.auxiliary_losses(output)
    total = 0.55 * detection + 0.35 * classification + 0.10 * prototype + 0.01 * balance
    return total, {"detection_loss": float(detection.detach()), "classification_loss": float(classification.detach()),
                   "prototype_loss": float(prototype.detach()), "balance_loss": float(balance.detach())}


@torch.no_grad()
def _metrics(model: FTMoEAblation, loader: DataLoader) -> dict[str, float]:
    model.eval()
    all_probability, all_anomaly, all_class_prob, all_class = [], [], [], []
    for x, labels in loader:
        output = model(x.float())
        all_probability.append(torch.softmax(output["detection_logits"], -1)[..., 1].cpu())
        all_anomaly.append((labels > 0).cpu())
        all_class_prob.append(torch.softmax(output["class_logits"], -1).cpu())
        all_class.append(labels.cpu())
    probability = torch.cat(all_probability).flatten().numpy()
    anomaly = torch.cat(all_anomaly).flatten().numpy().astype(bool)
    class_prob = torch.cat(all_class_prob).reshape(-1, 3).numpy()
    target_class = torch.cat(all_class).flatten().numpy()
    prediction = probability >= 0.5
    tp = int((prediction & anomaly).sum()); fp = int((prediction & ~anomaly).sum())
    fn = int((~prediction & anomaly).sum()); tn = int((~prediction & ~anomaly).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    # Rank only diagnostic candidates.  A relevant item requires a correct
    # resource type and an anomaly, giving HR/NDCG a stated non-test metric.
    positive_score = probability * class_prob.max(axis=1)
    predicted_class = class_prob.argmax(axis=1) + 1
    relevance = (anomaly & (predicted_class == target_class)).astype(np.float32)
    order = np.argsort(-positive_score)[:100]
    ideal = np.sort(relevance)[::-1][:100]
    hr100 = float(relevance[order].sum() / max(min(int(anomaly.sum()), 100), 1))
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((relevance[order] * discounts).sum())
    idcg = float((ideal * discounts[:len(ideal)]).sum())
    ndcg100 = dcg / max(idcg, 1e-12)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
            "hr_at_100": hr100, "ndcg_at_100": ndcg100, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _save_checkpoint(path: Path, model: FTMoEAblation, optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, metrics: dict[str, float],
                     training_generator: torch.Generator) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "validation": metrics, "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "training_generator_rng": training_generator.get_state()}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["v0", "v1"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", default="artifacts/ftmoe_ablation/screening_001/data")
    parser.add_argument("--output", default="artifacts/ftmoe_ablation/screening_001")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="resume this isolated run from final.pt")
    args = parser.parse_args()
    if args.seed != 1 or args.epochs > 25:
        raise ValueError("Experiment 001 is locked to seed=1 and at most 25 epochs")
    priority_set, priority_detail = _set_resource_limits()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    data_path, output = Path(args.data), Path(args.output)
    manifest = json.loads((data_path / "manifest.json").read_text(encoding="utf-8"))
    features = np.load(data_path / "features.npy", mmap_mode="r")
    labels = np.load(data_path / "labels_next_capacity.npy", mmap_mode="r")
    episode_ids = np.load(data_path / "episode_ids.npy", mmap_mode="r")
    train_set = WindowDataset(features, labels, episode_ids, manifest["split"]["train_episodes"])
    validation_set = WindowDataset(features, labels, episode_ids, manifest["split"]["validation_episodes"])
    training_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, generator=training_generator)
    eval_train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=0)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"{args.variant}_seed{args.seed}"
    if args.resume:
        if not (run_dir / "final.pt").is_file():
            raise FileNotFoundError("--resume requires this pilot's final.pt")
    else:
        run_dir.mkdir(exist_ok=False)
    model = FTMoEAblation(args.variant).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - 5, 1), eta_min=1e-5),
    ], milestones=[5])
    log_path = run_dir / "epochs.csv"
    start = time.perf_counter(); best_f1, best_epoch, patience, start_epoch = -1.0, 0, 0, 1
    if args.resume:
        state = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        if "training_generator_rng" in state:
            training_generator.set_state(state["training_generator_rng"])
        else:
            # Checkpoints created by the first interrupted invocation predate
            # generator-state persistence.  This fallback is deterministic
            # but repeats the seed's shuffle sequence; all future resumes are
            # exact because generator state is now saved.
            print("resume_note=legacy_checkpoint_without_dataloader_rng", flush=True)
        start_epoch = int(state["epoch"]) + 1
        best_state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        best_f1 = float(best_state["validation"]["f1"]); best_epoch = int(best_state["epoch"])
        print(f"resuming variant={args.variant} from epoch={start_epoch}", flush=True)
    resource_samples = []
    completed_epoch = start_epoch - 1
    if args.resume and completed_epoch >= args.epochs:
        # A previous process completed training but was interrupted before
        # its compact summary was written.  Capture a recovery-time resource
        # sample without executing another epoch.
        _ok, recovered_resources = _resources_ok()
        resource_samples.append(recovered_resources)
    with log_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "seconds", "learning_rate", "train_loss", "train_f1", "validation_f1", "validation_precision", "validation_recall", "validation_hr_at_100", "validation_ndcg_at_100", "available_ram_gib", "free_disk_gib", "process_rss_gib"])
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            completed_epoch = epoch
            ok, resources = _resources_ok()
            resource_samples.append(resources)
            if not ok:
                raise RuntimeError(f"resource guard stopped before epoch {epoch}: {resources}")
            model.train(); total_loss = 0.0; batches = 0
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, _parts = _loss(model, model(x.float()), y.long())
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); total_loss += float(loss.detach()); batches += 1
            scheduler.step()
            train_metrics = _metrics(model, eval_train_loader)
            validation_metrics = _metrics(model, validation_loader)
            elapsed = time.perf_counter() - start
            row = {"epoch": epoch, "seconds": elapsed, "learning_rate": optimizer.param_groups[0]["lr"],
                   "train_loss": total_loss / max(batches, 1), "train_f1": train_metrics["f1"],
                   "validation_f1": validation_metrics["f1"], "validation_precision": validation_metrics["precision"],
                   "validation_recall": validation_metrics["recall"], "validation_hr_at_100": validation_metrics["hr_at_100"],
                   "validation_ndcg_at_100": validation_metrics["ndcg_at_100"], **resources}
            writer.writerow(row); handle.flush()
            print(json.dumps({"variant": args.variant, **row}), flush=True)
            _save_checkpoint(run_dir / "final.pt", model, optimizer, scheduler, epoch, validation_metrics, training_generator)
            if validation_metrics["f1"] > best_f1:
                best_f1, best_epoch, patience = validation_metrics["f1"], epoch, 0
                _save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, validation_metrics, training_generator)
            else:
                patience += 1
            if patience >= 8:
                print(f"early_stop epoch={epoch}", flush=True)
                break
    final_train = _metrics(model, eval_train_loader)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["validation"]
    run_summary = {"variant": args.variant, "seed": args.seed, "epochs_requested": args.epochs,
                   "epochs_completed": completed_epoch, "best_epoch": best_epoch, "best_validation": best,
                   "final_train": final_train, "elapsed_seconds": time.perf_counter() - start,
                   "resource_limits": {"threads": 4, "num_workers": 0, "dtype": "float32", "torch_compile": False,
                                       "priority_set": priority_set, "priority": priority_detail,
                                       "minimum_available_ram_gib": min(r["available_ram_gib"] for r in resource_samples),
                                       "minimum_free_disk_gib": min(r["free_disk_gib"] for r in resource_samples),
                                       "maximum_epoch_start_process_rss_gib": max(r["process_rss_gib"] for r in resource_samples)}}
    (run_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
