"""Protocol 020 S6 — assemble the same-domain adaptation dataset (v1).

Reads the collected adaptation episodes
(adaptation_data/raw/{train,dev}/{profile}/seed{seed}_steps400) and writes a
bundle under adaptation_data/v1/ with one entry per episode:

    manifest.json      profiles/seeds/train/dev episode lists + hashes
    normalization.json time_scale v2 (protocol-020 registered), graph_scale,
                       coverage report, fallback columns
    episodes/<i>.npz   time_series (400,16,7) raw host aggregates (scored rows)
                       container_demand_series (400,16,7)
                       schedule_series (400,16,16)
                       creation_ids / before_placement (400,16)
                       capacities (400,16,3) per-interval (normalized at load)
                       labels (400,16)  tolerance ±1, within episode
    artifact: protocol_020/normalization_v2_time_scale.json  (registered,
        mirror of the 019 artifact key layout, for the 020 runner)

Label rule = runner tolerance_label semantics (nearest anomaly wins; ties to
the earlier interval), guard row 400 matures the last scored row.

Capacity/training statistics stay in the raw domain; normalization happens at
load time inside the trainer with the registered 020 time scale.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recovery.PreGANSrc.src.ftmoe_normalization import (
    compute_column_stats, apply_v2_time_scale, normalized_abs_max_report,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ftmoe_online/protocol_020"
RAW = ART / "adaptation_data/raw"
OUT = ART / "adaptation_data/v1"
PROFILES_PATH = ART / "adaptation/adaptation_profiles.json"
V2_019 = ROOT / "artifacts/ftmoe_online/protocol_019/normalization_v2_time_scale.json"
REGISTERED_OUT = ART / "normalization_v2_time_scale.json"
TRAIN_SEEDS = [401, 402, 403]
DEV_SEEDS = [404, 405]
STEPS = 400


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tolerance_labels(raw, scored):
    """raw: (steps+1,16); scored rows 0..steps-1 get ±1 tolerance labels
    (mirror of run_ftmoe_online.tolerance_label; ties choose the earlier row)."""
    labels = raw[:scored].copy()
    if scored >= 2:
        previous = raw[:scored - 1]
        fill = (labels[1:] == 0) & (previous > 0)
        labels[1:][fill] = previous[fill]
    following = raw[1:scored + 1]
    fill = (labels == 0) & (following > 0)
    labels[fill] = following[fill]
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    steps = args.steps
    if OUT.exists():
        raise FileExistsError(OUT)
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf8"))["profiles"]
    splits = {"train": TRAIN_SEEDS, "dev": DEV_SEEDS}
    episode_dirs = []  # (split, profile_key, seed, path)
    for split, seeds in splits.items():
        for key in profiles:
            for seed in seeds:
                folder = RAW / split / key / f"seed{seed}_steps{steps}"
                if not (folder / "manifest.json").is_file():
                    raise FileNotFoundError("missing episode: %s" % folder)
                episode_dirs.append((split, key, seed, folder))
    # Raw host aggregates for the coverage statistics (training split only).
    train_rows = []
    per_episode = {}
    for split, key, seed, folder in episode_dirs:
        with np.load(folder / "stream.npz") as data:
            raw_labels = data["raw_labels"].astype(np.int64)
            host = data["host_features"].astype(np.float32)[:steps]
            demands = data["demands"].astype(np.float32)[:steps]
            schedules = data["schedules"].astype(np.float32)[:steps]
            creation = data["creation_ids"].astype(np.int64)[:steps]
            before = data["before_placement"].astype(np.int64)[:steps]
            caps = data["capacities"].astype(np.float64)[:steps]
            labels = tolerance_labels(raw_labels, steps)
        per_episode[(split, key, seed)] = {
            "time": host, "demands": demands, "schedules": schedules,
            "creation_ids": creation, "before_placement": before,
            "capacities": caps, "labels": labels,
        }
        if split == "train":
            train_rows.append(host)
    # --- protocol-020 normalization v2 (registered) ---
    base = json.loads(V2_019.read_text(encoding="utf8"))
    base_scale = np.asarray(base["time_scale_v2_16x7"], dtype=np.float64)
    stats = compute_column_stats(np.concatenate(train_rows, axis=0))
    v2b, fallback_columns = apply_v2_time_scale(base_scale, stats)
    # normalized abs-max alarm report over train + dev scored rows
    reports = {}
    for split in ("train", "dev"):
        rows = [per_episode[(split, key, seed)]["time"]
                for key in profiles for seed in splits[split]]
        reports[split] = normalized_abs_max_report(np.concatenate(rows, axis=0), v2b)
    if reports["train"]["alarm"] or reports["dev"]["alarm"]:
        print(json.dumps({"normalization_alarm": True, "reports": reports}))
    graph_scale = np.asarray(base.get("graph_scale_014"), dtype=np.float64)
    if graph_scale.shape != (7,):
        raise ValueError("019 artifact must carry graph_scale_014 (7,)")
    registered = {
        "protocol": "020",
        "name": "protocol-020 normalization v2 (019 base + 020 train-cohort coverage)",
        "time_scale_v2_16x7": v2b.tolist(),
        "fallback_columns": fallback_columns,
        "graph_scale_014": graph_scale.tolist(),
        "coverage_stats_train": stats,
        "normalized_abs_max_report_train": reports["train"],
        "normalized_abs_max_report_dev": reports["dev"],
        "source_019_artifact": str(V2_019.relative_to(ROOT)),
        "source_sha256": sha(V2_019),
    }
    REGISTERED_OUT.parent.mkdir(parents=True, exist_ok=True)
    REGISTERED_OUT.write_text(json.dumps(registered, indent=1) + "\n", encoding="utf8")
    # --- write the bundle ---
    episode_folder = OUT / "episodes"
    OUT.mkdir(parents=True)
    episode_folder.mkdir()
    normalization_record = {
        "normalization_version": 2,
        "time_scale_v2": v2b.tolist(),
        "graph_scale": graph_scale.tolist(),
        "registered_artifact": "artifacts/ftmoe_online/protocol_020/normalization_v2_time_scale.json",
        "source_019_artifact": str(V2_019.relative_to(ROOT)),
    }
    (OUT / "normalization.json").write_text(
        json.dumps(normalization_record, indent=1, ensure_ascii=False) + "\n",
        encoding="utf8")
    hashes = {}
    for i, (split, key, seed, folder) in enumerate(episode_dirs):
        block = per_episode[(split, key, seed)]
        name = f"{split}_{key}_seed{seed}"
        with np.load(folder / "stream.npz") as data:
            raw_labels = data["raw_labels"].astype(np.int64)
        np.savez_compressed(episode_folder / f"{i:02d}.npz", **block)
        manifest_source = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
        hashes[name] = {
            "stream_sha256": manifest_source["stream_sha256"],
            "profile": manifest_source["profile"],
        }
    manifest = {
        "schema_version": 2, "protocol": "020", "phase": "S6-data-v1",
        "profiles": profiles,
        "train_episodes": [f"{split}_{key}_seed{seed}" for split, key, seed, _ in episode_dirs
                           if split == "train"],
        "dev_episodes": [f"{split}_{key}_seed{seed}" for split, key, seed, _ in episode_dirs
                         if split == "dev"],
        "episode_indices": [i for i, (split, key, seed, _) in enumerate(episode_dirs)],
        "train_indices": [i for i, (split, key, seed, _) in enumerate(episode_dirs)
                          if split == "train"],
        "dev_indices": [i for i, (split, key, seed, _) in enumerate(episode_dirs)
                        if split == "dev"],
        "block_size": steps,
        "label_policy": "actual physical overload, dominant ratio; +/-1 tolerance "
                        "(guard row matures the last row)",
        "episode_hashes": hashes,
        "normalization": normalization_record,
        "input_contract_version": 2, "normalization_version": 2,
        "graph_semantics_version": 3,
        "capacity_control_version": 1,
        "registration": "adaptation profiles from adaptation/adaptation_profiles.json; "
                        "seeds source-disjoint per cohort",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1,
                                                  ensure_ascii=False) + "\n",
                                       encoding="utf8")
    print(json.dumps({
        "written": str(OUT), "episodes": len(episode_dirs),
        "train_episodes": len(manifest["train_episodes"]),
        "dev_episodes": len(manifest["dev_episodes"]),
        "normalization_alarm_train": reports["train"]["alarm"],
        "normalized_abs_max_train": reports["train"]["max"],
        "normalized_abs_max_dev": reports["dev"]["max"],
        "fallback_columns": fallback_columns,
        "registered_artifact": str(REGISTERED_OUT.relative_to(ROOT)),
    }, indent=1))


if __name__ == "__main__":
    main()
