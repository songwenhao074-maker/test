"""Create the declared temporary, episode-disjoint Stage-A screening data.

It is a compact simulator-inspired generator, not the planned 10,000
interval shadow-simulator corpus.  Labels are next-step capacity violations;
they are never copied from migration edges or an event-type flag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HOSTS, FEATURES, WINDOW = 16, 7, 12
EPISODES, INTERVALS = 50, 40


def event_type_for_episode(episode: int) -> str:
    # Exact Stage-A allocation preserving the plan's 40/20/15/15/10 coverage.
    if episode < 20:
        return "time_trend"
    if episode < 30:
        return "heterogeneous_mode"
    if episode < 38:
        return "regime_shift"
    if episode < 46:
        return "schedule_overload"
    return "cross_path_interaction"


def _episode(seed: int, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed * 10_000 + episode)
    kind = event_type_for_episode(episode)
    feat = np.zeros((INTERVALS, HOSTS, FEATURES), dtype=np.float32)
    schedule = np.zeros((INTERVALS, 16, HOSTS), dtype=np.float32)
    # Eight workload/task profiles: the intended reason ordinary MoE can be
    # useful, while every profile is still visible in resource time series.
    profile = (np.arange(HOSTS) * 3 + episode) % 8
    base = rng.uniform(0.25, 0.44, size=(HOSTS, 3))
    state = rng.normal(0.0, 0.04, size=(HOSTS, 3))
    previous_place = (np.arange(16) + episode) % HOSTS
    next_main = np.zeros((INTERVALS, HOSTS, 3), dtype=np.float32)
    for t in range(INTERVALS):
        phase = 2.0 * np.pi * (t / INTERVALS)
        load = np.empty((HOSTS, 3), dtype=np.float32)
        for h in range(HOSTS):
            p = int(profile[h])
            emphasis = np.array([
                0.20 + 0.10 * (p % 3 == 0),
                0.19 + 0.10 * (p % 3 == 1),
                0.17 + 0.10 * (p % 3 == 2),
            ], dtype=np.float32)
            state[h] = 0.68 * state[h] + rng.normal(0, 0.042, 3)
            wave = np.sin(phase * (1.0 + (p % 4) * 0.35) + h * 0.61)
            stress = 0.0
            if kind == "time_trend":
                stress = 0.31 * max(0.0, (t - 19) / 20.0) * (p in {1, 4, 6})
            elif kind == "heterogeneous_mode":
                stress = 0.25 * (p in {2, 5, 7}) * (0.5 + 0.5 * wave)
            elif kind == "regime_shift":
                stress = 0.28 * (t >= 20) * (p in {0, 3, 6})
            elif kind == "schedule_overload":
                stress = 0.20 * (t >= 15) * (h in {3, 7, 11, 15})
            else:
                stress = 0.25 * (t >= 18) * (p in {1, 5}) * (0.35 + 0.65 * wave > 0.45)
            load[h] = base[h] + emphasis * (0.52 + 0.34 * wave + stress) + state[h]
        # Workload placement is recorded only for future graph variants; it
        # does not enter either labels or v0/v1 model inputs.
        place = previous_place.copy()
        if kind in {"schedule_overload", "cross_path_interaction"} and t in {13, 20, 27}:
            place[::4] = (place[::4] + 3 + episode) % HOSTS
        schedule[t, np.arange(16), place] = 1.0
        previous_place = place
        main = np.clip(load, 0.01, 1.25)
        feat[t, :, 0] = main[:, 0]
        feat[t, :, 1] = main[:, 1]
        feat[t, :, 4] = main[:, 2]
        feat[t, :, 2] = np.clip(main[:, 1] * (0.55 + 0.08 * (profile % 2)), 0, 1.25)
        feat[t, :, 3] = np.clip(main[:, 1] * (0.35 + 0.06 * ((profile + 1) % 2)), 0, 1.25)
        feat[t, :, 5] = np.clip(main[:, 2] * (0.48 + 0.07 * (profile % 2)), 0, 1.25)
        feat[t, :, 6] = np.clip(main[:, 2] * (0.30 + 0.05 * ((profile + 1) % 2)), 0, 1.25)
        next_main[t] = main
    # Capacity outcome at t+1: this is explicitly a next-interval outcome.
    labels = np.zeros((INTERVALS, HOSTS), dtype=np.int64)
    future = np.concatenate([next_main[1:], next_main[-1:]], axis=0)
    # Calibrated on the generator's train episodes to retain the plan's
    # 25--30% host-level positive-rate target without changing labels based
    # on event identities or placements.
    exceed = future > np.array([0.61, 0.60, 0.58], dtype=np.float32)
    resource = future.argmax(axis=-1)
    labels[exceed.any(axis=-1)] = resource[exceed.any(axis=-1)] + 1
    return feat, labels, schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/ftmoe_ablation/screening_001/data")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_feat, all_labels, all_schedule, episode_ids = [], [], [], []
    for episode in range(EPISODES):
        feat, labels, schedule = _episode(args.seed, episode)
        all_feat.append(feat); all_labels.append(labels); all_schedule.append(schedule)
        episode_ids.append(np.full(INTERVALS, episode, dtype=np.int64))
    feat = np.concatenate(all_feat); labels = np.concatenate(all_labels)
    schedule = np.concatenate(all_schedule); episode_ids = np.concatenate(episode_ids)
    # Normalization only uses train episodes.  The raw source stays separate
    # so its provenance is auditable.
    train_mask = episode_ids < 35
    scale = feat[train_mask].max(axis=(0, 1), keepdims=True).astype(np.float32)
    feat = feat / np.maximum(scale, 1e-6)
    np.save(output / "features.npy", feat)
    np.save(output / "labels_next_capacity.npy", labels)
    np.save(output / "schedule_for_future_graph_only.npy", schedule)
    np.save(output / "episode_ids.npy", episode_ids)
    split = {"train_episodes": list(range(35)), "validation_episodes": list(range(35, 40)),
             "test_episodes": list(range(40, 50))}
    manifest = {
        "stage": "A temporary screening data", "seed": args.seed,
        "intervals": int(feat.shape[0]), "episodes": EPISODES,
        "intervals_per_episode": INTERVALS, "input_shape": [HOSTS, WINDOW, FEATURES],
        "labels": "next-interval resource capacity violation; 1=cpu, 2=ram, 3=disk",
        "event_coverage": {kind: sum(event_type_for_episode(i) == kind for i in range(EPISODES))
                           for kind in ["time_trend", "heterogeneous_mode", "regime_shift", "schedule_overload", "cross_path_interaction"]},
        "split": split,
        "positive_rates": {name: float((labels[np.isin(episode_ids, ids)] > 0).mean())
                           for name, ids in [("train", split["train_episodes"]), ("validation", split["validation_episodes"]), ("test_unused", split["test_episodes"])]},
        "limitation": "Not the planned 10,000-interval shadow-simulator corpus; test episodes are generated but intentionally unused in experiment 001.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
