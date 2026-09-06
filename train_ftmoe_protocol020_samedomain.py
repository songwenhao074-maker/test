"""Protocol 020 S6 — same-domain common offline adaptation (warm start).

Fine-tunes the Protocol 019 adapted v4 checkpoint (per model seed) on the
protocol-020 adaptation bundle (train cohort episodes across the registered
capacity profiles: baseline / cpu_fault / ram_fault / disk_fault) with graph
semantics v3 (creation ids + before_placement + per-sample capacities) and
protocol-020 normalization v2.

Budget (registered): 15 epochs / lr 1e-4 / AdamW / batch 32 / full parameter
warm start / EAGate temperature constant 1.0 (as S3, plan §15).

Dev checkpoint selection (pre-registered, plan §16): per dev episode
    score_e = 0.6 * pr_auc_e + 0.4 * resource_macro_f1_e   (tolerance labels)
mean over dev episodes with positives; best epoch by that score.
Dev and online cohorts never take part in training.

Outputs (per model seed):
    artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed{seed}/
        best.pt / last.pt / resume.pt / epochs.csv / progress.json /
        summary.json / checkpoints_by_epoch/epochNNN.pt

Usage:
    python train_ftmoe_protocol020_samedomain.py --model-seed 1
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
from analyze_ftmoe_online import metrics as episode_metrics
from train_ftmoe_ablation_existing import loss_fn
from run_ftmoe_online import sha

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ftmoe_online/protocol_020"
DATA = ART / "adaptation_data/v1"
WARM = ROOT / "artifacts/ftmoe_online/protocol_019/s3/adapted_v4_seed1/best.pt"
EPOCHS = 15
LR = 1e-4
BATCH = 32
ALLOWED_SEEDS = {1}  # 019 produced adapted_v4_seed1 only (problem-log P08)


def load_bundle():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf8"))
    normalization = json.loads((DATA / "normalization.json").read_text(encoding="utf8"))
    time_scale = np.asarray(normalization["time_scale_v2"], np.float32).reshape(16, 7)
    graph_scale = np.asarray(normalization["graph_scale"], np.float32)
    blocks = {}
    for i in manifest["episode_indices"]:
        with np.load(DATA / "episodes" / f"{i:02d}.npz") as data:
            time_series = data["time"].astype(np.float32)
            demands = data["demands"].astype(np.float32)
            schedules = data["schedules"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            ids = data["creation_ids"].astype(np.int64)
            before = data["before_placement"].astype(np.int64)
            caps = data["capacities"].astype(np.float64)
        length = time_series.shape[0]
        idx = np.maximum(np.arange(length)[:, None] - 11 + np.arange(12)[None], 0)
        normalized_time = time_series / time_scale[None]
        normalized_graph = demands / graph_scale[None]
        x = torch.from_numpy(
            normalized_time[idx].transpose(0, 2, 1, 3).copy())
        g = torch.from_numpy(
            normalized_graph[idx].transpose(0, 2, 1, 3).copy())
        caps_n = (caps / graph_scale[[0, 1, 4]]).astype(np.float32)
        blocks[i] = {
            "x": x,
            "graph_x": g,
            "schedule": torch.from_numpy(schedules[idx].copy()),
            "labels": torch.from_numpy(labels.copy()),
            "ids": torch.from_numpy(ids[idx].copy()),
            "before": torch.from_numpy(before[idx].copy()),
            "caps": torch.from_numpy(caps_n[idx].copy()),
        }
    train = {name: torch.cat([blocks[i][name] for i in manifest["train_indices"]])
             for name in ("x", "graph_x", "schedule", "labels", "ids",
                          "before", "caps")}
    dev = {i: blocks[i] for i in manifest["dev_indices"]}
    return train, dev, manifest, normalization


@torch.no_grad()
def evaluate_v3(model, dev):
    """Per-dev-episode metrics with graph semantics v3 (tolerance labels).

    Selection score (registered): 0.6 * pr_auc + 0.4 * resource_macro_f1,
    averaged over dev episodes that contain positives.
    """
    model.eval()
    rows = []
    for index, block in dev.items():
        probabilities, classes = [], []
        for start in range(0, len(block["x"]), 64):
            output = model(block["x"][start:start + 64],
                           block["schedule"][start:start + 64],
                           block["graph_x"][start:start + 64],
                           graph_context={"creation_ids": block["ids"][start:start + 64],
                                          "before_placement": block["before"][start:start + 64],
                                          "capacities": block["caps"][start:start + 64]})
            probabilities.append(output["detection_logits"].softmax(-1)[..., 1].flatten())
            classes.append(output["class_logits"].softmax(-1).reshape(-1, 3))
        probability = torch.cat(probabilities).numpy()
        classes = torch.cat(classes).numpy()
        labels = block["labels"].numpy().ravel()
        result = episode_metrics(probability, classes, labels)
        result["episode"] = index
        result["score"] = None
        if result["pr_auc"] is not None and result["resource_macro_f1"] is not None:
            result["score"] = 0.6 * result["pr_auc"] + 0.4 * result["resource_macro_f1"]
        rows.append(result)
    scored = [r for r in rows if r["score"] is not None]
    mean = {key: float(np.mean([r[key] for r in scored]))
            for key in ("f1", "precision", "recall", "pr_auc",
                        "resource_macro_f1", "score")} if scored else {}
    mean["score"] = float(np.mean([r["score"] for r in scored])) if scored else None
    return {"per_episode": rows, "mean": mean}


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
    if seed not in ALLOWED_SEEDS:
        raise ValueError("Protocol 020 S6 warm start available for model seed(s): %s"
                         % sorted(ALLOWED_SEEDS))
    if not WARM.is_file():
        raise FileNotFoundError("Warm-start checkpoint missing: %s" % WARM)
    train, dev, manifest, normalization = load_bundle()
    warm = torch.load(WARM, map_location="cpu", weights_only=False)
    if warm["variant"] != "v4":
        raise ValueError("expected v4 warm start")
    torch.manual_seed(seed)
    model = FTMoEEndToEnd("v4", AblationConfig(
        experts=4, moe_residual_initial=0., eagate_residual_initial=.5,
        graph_residual_initial=0., cmha_residual_initial=0.)).float()
    model.load_state_dict(warm["model"], strict=True)
    # Static capacity buffer: only used by legacy paths (v3 always passes
    # per-sample capacities).  Value = baseline profile capacity, normalized.
    base_caps = train["caps"][0, 0].numpy()  # normalized already
    model.graph_encoder.host_capacity.copy_(torch.from_numpy(base_caps))
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 10000)
    out = ART / f"s6/adapted_v4_seed{seed}"
    if (out / "summary.json").exists():
        raise FileExistsError(f"S6 run already completed: {out}")
    out.mkdir(parents=True, exist_ok=True)
    norm_record = {
        "time_scale": np.asarray(normalization["time_scale_v2"],
                                 np.float64).reshape(-1).tolist(),
        "graph_scale": np.asarray(normalization["graph_scale"],
                                  np.float64).tolist(),
        "graph_host_capacity": base_caps.tolist(),
        "normalization_version": 2,
        "time_scale_flat_len": 112,
    }
    config = {"protocol": "020", "phase": "S6", "model_seed": seed,
              "warm_start": {"path": str(WARM), "sha256": sha(WARM)},
              "epochs": EPOCHS, "learning_rate": LR, "batch_size": BATCH,
              "optimizer": "AdamW", "weight_decay": 1e-4,
              "data": str(DATA.resolve()),
              "graph_semantics_version": 3,
              "selection_rule": "0.6*pr_auc + 0.4*resource_macro_f1 on dev "
                                "episodes with positives (pre-registered)",
              "normalization": norm_record,
              "code_sha256": {"train_ftmoe_protocol020_samedomain.py": sha(__file__),
                              "build_ftmoe_protocol020_adaptation_dataset.py":
                                  sha(ROOT / "build_ftmoe_protocol020_adaptation_dataset.py")}}
    (out / "configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
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
        best_epoch, best_score, history = (resume["best_epoch"],
                                           resume["best_score"], resume["history"])
        elapsed_before = resume["elapsed_seconds"]
    start_time = time.perf_counter()
    x, graph_x, schedule, labels = (train[name] for name in
                                    ("x", "graph_x", "schedule", "labels"))
    ids, before, caps = (train[name] for name in ("ids", "before", "caps"))
    for epoch in range(first_epoch, EPOCHS + 1):
        model.train()
        model.set_eagate_temperature(1.0)
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.
        batches = 0
        for batch in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            output = model(x[batch], schedule[batch], graph_x[batch],
                           graph_context={"creation_ids": ids[batch],
                                          "before_placement": before[batch],
                                          "capacities": caps[batch]})
            loss = loss_fn(model, output, labels[batch], .7, .3, 0., .01, 2., .5)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        dev_out = evaluate_v3(model, dev)
        score = dev_out["mean"].get("score")
        elapsed = elapsed_before + time.perf_counter() - start_time
        row = {"epoch": epoch, "loss": total_loss / max(batches, 1),
               "seconds": elapsed, "lr": LR, **dev_out["mean"]}
        history.append(row)
        checkpoint_state = {"variant": "v4", "seed": seed, "epoch": epoch,
                            "model": model.state_dict(), "validation": dev_out,
                            "normalization": norm_record,
                            "graph_semantics_version": 3}
        if score is not None and score > best_score:
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
            random_rng=random.getstate(), best_epoch=best_epoch,
            best_score=best_score, history=history,
            elapsed_seconds=elapsed))
        (out / "progress.json").write_text(json.dumps(row, indent=2) + "\n",
                                           encoding="utf8")
        with (out / "epochs.csv").open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({"epoch": epoch, "loss": row["loss"],
                          "score": score, "best_epoch": best_epoch,
                          "pr_auc": row.get("pr_auc"),
                          "resource_macro_f1": row.get("resource_macro_f1"),
                          "seconds": elapsed}), flush=True)
    # Anchor probe on the original protocol-004 domain (legacy path).
    from train_ftmoe_end_to_end import load_data, evaluate
    _, anchor_validation, _, _ = load_data(
        ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical")
    anchor = evaluate(model, anchor_validation)
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    summary = {"protocol": "020", "phase": "S6", "variant": "v4", "seed": seed,
               "configuration": config, "best_epoch": best_epoch,
               "dev_best": best["validation"]["mean"],
               "dev_last": last["validation"]["mean"],
               "anchor_protocol004_legacy": anchor,
               "elapsed_seconds": history[-1]["seconds"],
               "warm_start_source": config["warm_start"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 ensure_ascii=False) + "\n",
                                      encoding="utf8")
    print("COMPLETE " + json.dumps({"best_epoch": best_epoch,
                                    "dev_best": best["validation"]["mean"],
                                    "anchor": anchor["mean"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, required=True)
    run(parser.parse_args())
