"""Protocol 019 S3 — assemble the same-domain adaptation dataset (v1).

Reads the collected adapted-BWGD2 episodes (seeds 401-405 train / 406-408
dev) under artifacts/ftmoe_online/protocol_019/adaptation_data/raw and writes
a protocol_004-style dataset folder (adaptation_data/v1) that additionally
carries creation ids for graph-context v2 training:

    manifest.json            seeds/train_blocks/validation_blocks/... + hashes
    normalization.json       time_scale v2 (protocol_019 artifact), graph_scale,
                             graph_host_capacity
    time_series.npy          (B, L, 112)  raw host aggregates (rows 0..L-1)
    container_demand_series.npy (B, L, 112)
    schedule_series.npy      (B, L, 16, 16)
    creation_ids.npy         (B, L, 16)
    labels.npy               (B, L, 16)  physical labels, ±1 tolerance within
                             episode (guard row included for the last row)

The tolerance rule is the same as the online runner (nearest anomaly wins,
ties choose the earlier interval) computed over each episode's raw rows
0..steps, where row steps is the guard row.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "artifacts/ftmoe_online/protocol_019/adaptation_data/raw"
OUT = ROOT / "artifacts/ftmoe_online/protocol_019/adaptation_data/v1"
TRAIN_SEEDS = [401, 402, 403, 404, 405]
DEV_SEEDS = [406, 407, 408]
STEPS = 400


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tolerance_labels(raw, scored):
    """raw: (steps+1,16); scored rows 0..steps-1 get ±1 tolerance labels.

    Mirrors run_ftmoe_online.tolerance_label: start from the row's own raw
    label, then fill zeros from the previous row's raw label, then from the
    next row's raw label (nearest wins; equal distance chooses the earlier
    interval).  Row `steps` is the guard row that matures the last row.
    """
    labels = raw[:scored].copy()
    if scored >= 2:
        previous = raw[:scored - 1]          # previous[t] == raw[t-1]
        fill = (labels[1:] == 0) & (previous > 0)
        labels[1:][fill] = previous[fill]
    following = raw[1:scored + 1]            # following[t] == raw[t+1]
    fill = (labels == 0) & (following > 0)
    labels[fill] = following[fill]
    return labels


def build(out: Path = OUT):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    seeds = TRAIN_SEEDS + DEV_SEEDS
    manifests = {}
    arrays = {key: [] for key in ("time", "container", "schedule", "creation", "labels")}
    capacities = None
    for seed in seeds:
        folder = RAW / f"seed{seed}_steps{STEPS}"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
        manifests[seed] = manifest
        with np.load(folder / "stream.npz") as data:
            raw_labels = data["raw_labels"].astype(np.int64)  # (steps+1, 16)
            host = data["host_features"].astype(np.float32)[:STEPS]
            demands = data["demands"].astype(np.float32)[:STEPS]
            schedules = data["schedules"].astype(np.float32)[:STEPS]
            creation = data["creation_ids"].astype(np.int64)[:STEPS]
            episode_capacities = np.asarray(data["capacities"], dtype=np.float64)
            labels = tolerance_labels(raw_labels, STEPS)
        arrays["time"].append(host.reshape(STEPS, -1))
        arrays["container"].append(demands.reshape(STEPS, -1))
        arrays["schedule"].append(schedules)
        arrays["creation"].append(creation)
        arrays["labels"].append(labels)
        if capacities is None:
            capacities = episode_capacities
        assert np.array_equal(capacities, episode_capacities)
        print(json.dumps({"seed": seed, "class_counts_tol": np.bincount(labels.ravel(), minlength=4).tolist(),
                          "raw": manifest["raw_class_counts_scored"]}))
    capacities = np.asarray(capacities)
    normalization = json.loads((ROOT / "artifacts/ftmoe_online/protocol_019"
                                / "normalization_v2_time_scale.json").read_text(encoding="utf8"))
    time_scale_v2 = np.asarray(normalization["time_scale_v2_16x7"], dtype=np.float64).reshape(-1)
    graph_scale = np.asarray(normalization.get("graph_scale") if "graph_scale" in normalization
                             else None)
    np.save(out / "time_series.npy", np.stack(arrays["time"]))
    np.save(out / "container_demand_series.npy", np.stack(arrays["container"]))
    np.save(out / "schedule_series.npy", np.stack(arrays["schedule"]))
    np.save(out / "creation_ids.npy", np.stack(arrays["creation"]))
    np.save(out / "labels.npy", np.stack(arrays["labels"]))
    # graph_scale is taken from the 014 checkpoint normalization (registered in
    # normalization_v2_time_scale.json as graph_scale_014).
    graph_scale_from_artifact = normalization.get("graph_scale_014", None)
    if graph_scale_from_artifact is None:
        raise ValueError("normalization_v2_time_scale.json must carry graph_scale_014")
    graph_scale = np.asarray(graph_scale_from_artifact, dtype=np.float64)
    graph_host_capacity = capacities / graph_scale[[0, 1, 4]]
    normalization_record = {
        "normalization_version": 2,
        "time_scale_v2": time_scale_v2.tolist(),
        "graph_scale": graph_scale.tolist(),
        "graph_host_capacity": graph_host_capacity.tolist(),
        "source_artifact": "artifacts/ftmoe_online/protocol_019/normalization_v2_time_scale.json",
    }
    (out / "normalization.json").write_text(
        json.dumps(normalization_record, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    source_hashes = {}
    for seed in seeds:
        folder = RAW / f"seed{seed}_steps{STEPS}"
        source_hashes[str(seed)] = {
            "stream_sha256": manifests[seed]["stream_sha256"],
            "manifest_sha256": sha(folder / "manifest.json"),
        }
    manifest = {
        "schema_version": 2,
        "protocol": "019",
        "phase": "S3-data-v1",
        "seeds": seeds,
        "train_blocks": list(range(len(TRAIN_SEEDS))),
        "validation_blocks": list(range(len(TRAIN_SEEDS), len(seeds))),
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": DEV_SEEDS,
        "block_size": STEPS,
        "label_policy": "actual physical overload and largest overload ratio; "
                        "+/-1 within-episode tolerance (guard row included)",
        "host_capacities": capacities.tolist(),
        "environment": manifests[seeds[0]].get("demand_adapter"),
        "input_contract_version": 2,
        "normalization_version": 2,
        "graph_semantics_version": 2,
        "class_counts": {str(seed): np.bincount(
            np.load(out / "labels.npy")[i].ravel(), minlength=4).tolist()
            for i, seed in enumerate(seeds)},
        "source_episode_hashes": source_hashes,
        "array_sha256": {
            name: hashlib.sha256(np.load(out / f"{name}.npy").tobytes()).hexdigest()
            for name in ("time_series", "container_demand_series", "schedule_series",
                         "creation_ids", "labels")},
        "registration": {
            "status": "protocol019 S3 same-domain adaptation data; seeds disjoint from S2 dev stream (303)",
            "training_seeds": TRAIN_SEEDS,
            "validation_seeds": DEV_SEEDS,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf8")
    print(json.dumps({"written": str(out), "shapes": {k: list(v.shape) for k, v in
          [("time_series", np.load(out / "time_series.npy")),
           ("labels", np.load(out / "labels.npy")),
           ("creation_ids", np.load(out / "creation_ids.npy"))]},
          "normalization": normalization_record}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    build(args.output or OUT)
