"""Protocol 019 S3 — same-domain common offline adaptation (warm-start).

Fine-tunes the registered 014 v4 checkpoint (per model seed) on the S3
adaptation dataset (adapted-BWGD2 contract, normalization v2, graph
semantics v2, creation ids passed as graph_context during training).

Dev budget (docs/FTMOE_ONLINE_PROTOCOL_019.md §8, The Plan §8.3):
    epochs 15, lr 1e-4, AdamW, batch 32, full-parameter warm start
    all model seeds use the same budget; dev metric = validation-block mean.

Outputs (per model seed):
    artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed{seed}/
        best.pt / last.pt / resume.pt / epochs.csv / progress.json /
        summary.json / checkpoints_by_epoch/epochNNN.pt

Checkpoint contents follow the 014 convention plus normalization v2:
    {variant, seed, epoch, model, validation, normalization}
    normalization = {'time_scale': v2 (flat 112), 'graph_scale': 7,
                     'graph_host_capacity': (16,3), 'normalization_version': 2}
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "3"
import numpy as np
import torch

from recovery.PreGANSrc.src.ftmoe_ablation import AblationConfig
from recovery.PreGANSrc.src.ftmoe_end_to_end import FTMoEEndToEnd
from train_ftmoe_ablation_existing import loss_fn
from train_ftmoe_end_to_end import METRICS, metric_arrays, load_data
from run_ftmoe_online import resolve_checkpoint, sha

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ftmoe_online/protocol_019"
DATA = ART / "adaptation_data/v1"
EPOCHS = 15
LR = 1e-4
BATCH = 32


def load_adaptation_blocks(directory=DATA):
    """Returns (training, validation, normalization, manifest, blocks).

    Follows train_ftmoe_end_to_end.load_data conventions: training windows are
    concatenated over train blocks; validation and blocks are keyed by seed
    string.  Every window set additionally carries creation ids (``ids``) for
    graph-context v2 training/evaluation.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    normalization = json.loads((directory / "normalization.json").read_text(encoding="utf8"))
    time_scale = np.asarray(normalization["time_scale_v2"], np.float32).reshape(-1)
    graph_scale = np.asarray(normalization["graph_scale"], np.float32)
    t = np.load(directory / "time_series.npy").astype(np.float32)
    g = np.load(directory / "container_demand_series.npy").astype(np.float32)
    schedules = np.load(directory / "schedule_series.npy").astype(np.float32)
    labels = np.load(directory / "labels.npy").astype(np.int64)
    creation = np.load(directory / "creation_ids.npy").astype(np.int64)
    t = (t / time_scale).reshape(*t.shape[:2], 16, 7)
    g = (g.reshape(*g.shape[:2], 16, 7) / graph_scale).astype(np.float32)
    length = t.shape[1]
    indices = np.maximum(np.arange(length)[:, None] - 11 + np.arange(12)[None], 0)
    blocks = {}
    for block in range(len(t)):
        seed_key = str(manifest["seeds"][block])
        blocks[seed_key] = {
            "x": torch.from_numpy(t[block, indices].transpose(0, 2, 1, 3).copy()),
            "graph_x": torch.from_numpy(g[block, indices].transpose(0, 2, 1, 3).copy()),
            "schedule": torch.from_numpy(schedules[block, indices].copy()),
            "labels": torch.from_numpy(labels[block].copy()),
            "ids": torch.from_numpy(creation[block, indices].copy()),
        }
    train_keys = [str(manifest["seeds"][b]) for b in manifest["train_blocks"]]
    validation_keys = [str(manifest["seeds"][b]) for b in manifest["validation_blocks"]]
    validation = {key: blocks[key] for key in validation_keys}
    train = {column: torch.cat([blocks[seed_key][column] for seed_key in train_keys])
             for column in ("x", "graph_x", "schedule", "labels", "ids")}
    return train, validation, normalization, manifest, blocks


@torch.no_grad()
def evaluate_v2(model, validation):
    """Block-wise evaluation with graph_context v2 (same metric schema as
    train_ftmoe_end_to_end.evaluate, plus per-block dicts)."""
    model.eval()
    per_block = {}
    for key, block in validation.items():
        probabilities, classes = [], []
        for start in range(0, len(block["x"]), 64):
            output = model(block["x"][start:start + 64], block["schedule"][start:start + 64],
                           block["graph_x"][start:start + 64],
                           graph_context={"creation_ids": block["ids"][start:start + 64]})
            probabilities.append(output["detection_logits"].softmax(-1)[..., 1].flatten())
            classes.append(output["class_logits"].softmax(-1).reshape(-1, 3))
        per_block[key] = metric_arrays(torch.cat(probabilities).numpy(),
                                       torch.cat(classes).numpy(), block["labels"].numpy().ravel())
    numeric = [key for key, value in next(iter(per_block.values())).items()
               if isinstance(value, (float, int))]
    mean = {key: float(np.mean([value[key] for value in per_block.values()])) for key in numeric}
    mean["score"] = float(np.mean([mean[key] for key in METRICS]))
    return {"mean": mean, "per_replay": per_block}


def save_pt(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run(args):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    seed = args.model_seed
    if seed not in (1, 2, 6):
        raise ValueError(f"Unregistered 014 v4 model seed for S3: {seed}")
    training, validation, normalization, manifest, _ = load_adaptation_blocks()
    checkpoint, source = resolve_checkpoint(seed)
    variant = checkpoint["variant"]
    if variant != "v4":
        raise ValueError(f"expected v4 warm start, got {variant}")
    torch.manual_seed(seed)
    model = FTMoEEndToEnd("v4", AblationConfig(
        experts=4, moe_residual_initial=0., eagate_residual_initial=.5,
        graph_residual_initial=0., cmha_residual_initial=0.)).float()
    model.load_state_dict(checkpoint["model"], strict=True)
    # capacity buffer = dataset host capacities normalized by graph_scale
    cap = torch.as_tensor(np.asarray(normalization["graph_host_capacity"], np.float32))
    model.graph_encoder.host_capacity.copy_(cap)
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 10000)
    out = ROOT / f"artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed{seed}"
    if (out / "summary.json").exists():
        raise FileExistsError(f"S3 run already completed: {out}")
    out.mkdir(parents=True, exist_ok=True)
    norm_record = {"time_scale": np.asarray(normalization["time_scale_v2"], np.float64).tolist(),
                   "graph_scale": np.asarray(normalization["graph_scale"], np.float64).tolist(),
                   "graph_host_capacity": np.asarray(normalization["graph_host_capacity"],
                                                     np.float64).tolist(),
                   "normalization_version": 2,
                   "time_scale_flat_len": 112}
    config = {"protocol": "019", "phase": "S3", "model_seed": seed,
              "warm_start": source, "epochs": EPOCHS, "learning_rate": LR,
              "batch_size": BATCH, "optimizer": "AdamW", "weight_decay": 1e-4,
              "data": str(DATA.resolve()), "normalization": norm_record,
              "graph_semantics_version": 2,
              "episode_steps": manifest["block_size"],
              "code_sha256": {"train_ftmoe_protocol019_s3.py": sha(__file__),
                              "build_ftmoe_protocol019_adaptation_dataset.py":
                                  sha(ROOT / "build_ftmoe_protocol019_adaptation_dataset.py")}}
    (out / "configuration.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf8")
    history = []
    best_score = -float("inf")
    best_epoch = 0
    elapsed_before = 0.
    first_epoch = 1
    if (out / "resume.pt").exists():
        resume = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["generator"])
        torch.set_rng_state(resume["torch_rng"])
        np.random.set_state(resume["numpy_rng"])
        random.setstate(resume["random_rng"])
        first_epoch = resume["epoch"] + 1
        best_epoch = resume["best_epoch"]
        best_score = resume["best_score"]
        history = resume["history"]
        elapsed_before = resume["elapsed_seconds"]
        print(f"resumed at epoch {first_epoch}")
    start_time = time.perf_counter()
    x, graph_x, schedule, labels, ids = (training[key] for key in
                                         ("x", "graph_x", "schedule", "labels", "ids"))
    for epoch in range(first_epoch, EPOCHS + 1):
        model.train()
        # EAGate temperature: constant (warm-start continuation; no anneal restart).
        model.set_eagate_temperature(1.0)
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.
        batches = 0
        for batch in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            output = model(x[batch], schedule[batch], graph_x[batch],
                           graph_context={"creation_ids": ids[batch]})
            loss = loss_fn(model, output, labels[batch], .7, .3, 0., .01, 2., .5)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        dev = evaluate_v2(model, validation)
        score = dev["mean"]["score"]
        elapsed = elapsed_before + time.perf_counter() - start_time
        row = {"epoch": epoch, "loss": total_loss / max(batches, 1), "seconds": elapsed,
               "lr": LR, **dev["mean"]}
        history.append(row)
        checkpoint_state = {"variant": "v4", "seed": seed, "epoch": epoch,
                            "model": model.state_dict(), "validation": dev,
                            "normalization": norm_record}
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_pt(out / "best.pt", checkpoint_state)
        save_pt(out / "last.pt", checkpoint_state)
        epoch_dir = out / "checkpoints_by_epoch"
        epoch_dir.mkdir(exist_ok=True)
        save_pt(epoch_dir / f"epoch{epoch:03d}.pt", checkpoint_state)
        save_pt(out / "resume.pt", dict(checkpoint_state,
            optimizer=optimizer.state_dict(), generator=generator.get_state(),
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            random_rng=random.getstate(), best_epoch=best_epoch, best_score=best_score,
            history=history, elapsed_seconds=elapsed))
        (out / "progress.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf8")
        with (out / "epochs.csv").open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"S3 v4 s{seed} ep{epoch}/{EPOCHS} F1={row['f1']:.4f} "
              f"PR={row['pr_auc']:.4f} score={score:.4f} best_ep={best_epoch} "
              f"seconds={elapsed:.1f}", flush=True)
    # Anchor probe on the original protocol-004 domain (legacy path, no ids).
    from train_ftmoe_end_to_end import evaluate
    _, anchor_validation, _, _ = load_data(ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical")
    anchor = evaluate(model, anchor_validation)
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    summary = {"protocol": "019", "phase": "S3", "variant": "v4", "seed": seed,
               "configuration": config, "best_epoch": best_epoch,
               "dev_best": best["validation"]["mean"],
               "dev_last": last["validation"]["mean"],
               "anchor_protocol004_legacy": anchor,
               "elapsed_seconds": history[-1]["seconds"],
               "warm_start_source": source,
               "normalization": norm_record}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf8")
    print("COMPLETE " + json.dumps({"best_epoch": best_epoch,
                                    "dev_best": best["validation"]["mean"],
                                    "anchor": anchor["mean"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, required=True)
    run(parser.parse_args())
