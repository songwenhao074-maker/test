"""Protocol 020 S7 — online replay runner (A/B/C; D joins at S8 gate v3).

Prequential replay identical in layout to run_ftmoe_protocol019_s4.py, but
on the protocol-020 stream contract:

- per-interval capacities (capacities_per_interval [T+1,16,3]) and
  before_placement enter every model window through graph semantics v3
  (graph_context = creation_ids + before_placement + capacities);
- the S7 session (recovery/PreGANSrc/src/ftmoe_online_s7.py) applies
  rare-event stratified sampling and class-balanced loss v3 (weights from
  the same-domain training bundle only);
- anchor memory = protocol-020 adaptation train episodes (tail rows).

The static 'capacities' key seen by the legacy session code is an alias of
row 0 (interval-0 capacity), so the inherited OnlineSessionV2/S4 machinery
(anchor reference probes, hashes, resume) stays byte-compatible.

Usage:
    python run_ftmoe_protocol020.py --method A --model-seed 1 \
        --checkpoint-path artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt \
        --stream artifacts/ftmoe_online/protocol_020/drift_streams/dev_seed500_steps2000 \
        --output artifacts/ftmoe_online/protocol_020/runs/A_model1_seed500
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "3"
import numpy as np
import psutil
import torch

from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_s4 import (
    RECENT_MAXLEN, ANCHOR_PER_UPDATE, MAX_EXPOSURE, LAMBDA_ANCHOR,
    LAMBDA_DISTILL, LR_SLOW_FACTOR,
)
from recovery.PreGANSrc.src.ftmoe_online_s7 import (
    S7Session, balanced_weights, RECENT_UNIFORM_PER_UPDATE, EVENT_PER_UPDATE,
    MAX_RARE_EXPOSURE,
)
from run_ftmoe_protocol019 import (
    extra_detection_metrics, resources, sha,
)
from run_ftmoe_online import write_json, save_state
from analyze_ftmoe_online import summarize_arrays

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ftmoe_online/protocol_020"
DATA = ART / "adaptation_data/v1"
ANCHOR_ROW_START = 272
ANCHOR_PER_EPISODE = 128


def load_v2_time_scale_p20(checkpoint_normalization):
    """Checkpoint v2 scale must equal the registered protocol-020 artifact."""
    artifact_path = ART / "normalization_v2_time_scale.json"
    if not artifact_path.exists():
        raise FileNotFoundError("Protocol 020 normalization artifact missing: %s"
                                % artifact_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf8"))
    registered = np.asarray(artifact["time_scale_v2_16x7"], dtype=np.float64)
    if checkpoint_normalization.get("normalization_version") != 2:
        raise ValueError("S6 checkpoints must carry normalization_version 2")
    v2 = np.asarray(checkpoint_normalization["time_scale"],
                    dtype=np.float64).reshape(16, 7)
    if not np.allclose(v2, registered, rtol=1e-12, atol=1e-12):
        raise AssertionError("S6 checkpoint v2 scale diverges from the "
                             "registered protocol-020 artifact")
    return v2, artifact.get("fallback_columns", [])


def tolerance_labels(raw, scored):
    """±1 within-episode tolerance (identical rule to the offline builder)."""
    labels = raw[:scored].copy()
    if scored >= 2:
        previous = raw[:scored - 1]
        fill = (labels[1:] == 0) & (previous > 0)
        labels[1:][fill] = previous[fill]
    following = raw[1:scored + 1]
    fill = (labels == 0) & (following > 0)
    labels[fill] = following[fill]
    return labels


class ReplayV3:
    """Protocol-020 causal replay: v2 window plus before_placement and
    per-interval capacities (graph semantics v3)."""

    def __init__(self, arrays, time_scale_v2, graph_scale, steps):
        self.arrays = arrays
        self.steps = steps
        self.time_scale = np.asarray(time_scale_v2, np.float32).reshape(16, 7)
        self.graph_scale = np.asarray(graph_scale, np.float32)
        if arrays["host_features"].shape != (steps + 1, 16, 7) or \
                arrays["raw_labels"].shape != (steps + 1, 16):
            raise ValueError("Stream must contain exact scored horizon plus one guard interval")
        for name in ("host_features", "demands", "schedules"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError("Nonfinite stream: " + name)
        if arrays["creation_ids"].shape != (steps + 1, 16):
            raise ValueError("Stream must carry creation_ids (steps+1,16)")
        if arrays["before_placement"].shape != (steps + 1, 16):
            raise ValueError("Stream must carry before_placement (steps+1,16)")
        caps = arrays["capacities_per_interval"]
        if caps.shape != (steps + 1, 16, 3) or not np.isfinite(caps).all():
            raise ValueError("Stream must carry capacities_per_interval (steps+1,16,3)")
        self.capacities = caps
        if (self.time_scale <= 0).any() or (self.graph_scale <= 0).any():
            raise ValueError("Invalid normalization")

    def window(self, index):
        """Legacy 4-tuple (used only by anchor-reference machinery if any)."""
        x, s, g, ids, _, _ = self.window_v3(index)
        return x, s, g, ids

    def window_v3(self, index):
        if not 0 <= index < self.steps:
            raise IndexError(index)
        positions = np.maximum(np.arange(index - 11, index + 1), 0)
        host = (self.arrays["host_features"][positions] /
                self.time_scale).transpose(1, 0, 2).astype(np.float32)
        graph = (self.arrays["demands"][positions] /
                 self.graph_scale).transpose(1, 0, 2).astype(np.float32)
        schedule = self.arrays["schedules"][positions].astype(np.float32)
        identity = self.arrays["creation_ids"][positions].astype(np.int64)
        before = self.arrays["before_placement"][positions].astype(np.int64)
        caps = (self.capacities[positions] /
                self.graph_scale[[0, 1, 4]]).astype(np.float32)
        return (torch.from_numpy(host), torch.from_numpy(schedule),
                torch.from_numpy(graph), torch.from_numpy(identity),
                torch.from_numpy(before), torch.from_numpy(caps))


def load_anchor_pool_v3():
    """Anchor pool from protocol-020 adaptation train episodes (tail rows).

    Returns (pool, class_balance): pool tensors carry graph v3 arrays
    (ids/before/caps windows); class_balance is computed once from the
    training tolerance labels (plan §19.1).
    """
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf8"))
    normalization = json.loads((DATA / "normalization.json").read_text(encoding="utf8"))
    time_scale = np.asarray(normalization["time_scale_v2"], np.float32).reshape(16, 7)
    graph_scale = np.asarray(normalization["graph_scale"], np.float32)
    parts = {key: [] for key in ("x", "graph_x", "schedule", "labels",
                                 "ids", "before", "caps")}
    weight_labels = []
    for index in manifest["train_indices"]:
        with np.load(DATA / "episodes" / f"{index:02d}.npz") as data:
            time_series = data["time"].astype(np.float32)
            demands = data["demands"].astype(np.float32)
            schedules = data["schedules"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            ids = data["creation_ids"].astype(np.int64)
            before = data["before_placement"].astype(np.int64)
            caps = data["capacities"].astype(np.float64)
        weight_labels.append(labels)
        rows = np.arange(ANCHOR_ROW_START, ANCHOR_ROW_START + ANCHOR_PER_EPISODE)
        idx = np.maximum(rows[:, None] - 11 + np.arange(12)[None], 0)
        parts["x"].append(torch.from_numpy(
            (time_series[idx] / time_scale[None]).transpose(0, 2, 1, 3).copy()))
        parts["graph_x"].append(torch.from_numpy(
            (demands[idx] / graph_scale[None]).transpose(0, 2, 1, 3).copy()))
        parts["schedule"].append(torch.from_numpy(schedules[idx].copy()))
        parts["labels"].append(torch.from_numpy(labels[rows].copy()))
        parts["ids"].append(torch.from_numpy(ids[idx].copy()))
        parts["before"].append(torch.from_numpy(before[idx].copy()))
        parts["caps"].append(torch.from_numpy(
            (caps[idx] / graph_scale[[0, 1, 4]]).astype(np.float32).copy()))
    pool = {key: torch.cat(parts[key]) for key in parts}
    balance = balanced_weights(np.concatenate(weight_labels, axis=0))
    return pool, balance


def run(args):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resources()
    if args.method == "D":
        raise ValueError("D requires the dynamic expert v3 gate (S8); A/B/C only here")
    if args.checkpoint_path is None:
        raise ValueError("S7 requires --checkpoint-path (S6-adapted checkpoint)")
    manifest = json.loads((args.stream / "manifest.json").read_text())
    if manifest["stream_sha256"] != sha(args.stream / "stream.npz"):
        raise AssertionError("Stream hash mismatch")
    if manifest.get("protocol") != "020":
        raise ValueError("Stream is not registered under Protocol 020")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "v4" or checkpoint["seed"] != args.model_seed:
        raise AssertionError("Checkpoint identity mismatch")
    if checkpoint.get("graph_semantics_version") != 3:
        raise ValueError("S6 checkpoint must carry graph_semantics_version 3")
    source = {"path": str(checkpoint_path.resolve()), "sha256": sha(checkpoint_path),
              "epoch": checkpoint["epoch"], "mode": "s6_adapted"}
    with np.load(args.stream / "stream.npz") as data:
        arrays = {key: data[key] for key in data.files}
    v2_scale, fallback_columns = load_v2_time_scale_p20(checkpoint["normalization"])
    graph_scale = np.asarray(checkpoint["normalization"]["graph_scale"], np.float64)
    # Static legacy alias = interval-0 capacity; full per-interval array stays
    # under 'capacities_per_interval' for graph semantics v3.
    caps_full = arrays.pop("capacities")
    arrays["capacities"] = np.asarray(caps_full[0], dtype=np.float64)
    arrays["capacities_per_interval"] = np.asarray(caps_full, dtype=np.float64)
    replay = ReplayV3(arrays, v2_scale, graph_scale, manifest["steps"])
    lr = 0. if args.method == "A" else (args.base_lr if args.base_lr is not None else 1e-5)
    if args.method != "A" and lr not in (1e-5, 3e-5, 1e-4):
        raise ValueError(f"Unregistered S7 base learning rate: {lr}")
    anchor_pool, class_balance = load_anchor_pool_v3()
    teacher = OnlineFTMoE(checkpoint, "A", args.model_seed)
    teacher.eval()
    code_files = ["run_ftmoe_protocol020.py", "run_ftmoe_protocol019_s4.py",
                  "run_ftmoe_protocol019.py", "run_ftmoe_online.py",
                  "analyze_ftmoe_online.py", "train_ftmoe_ablation_existing.py",
                  "train_ftmoe_end_to_end.py",
                  "recovery/PreGANSrc/src/ftmoe_online.py",
                  "recovery/PreGANSrc/src/ftmoe_online_s4.py",
                  "recovery/PreGANSrc/src/ftmoe_online_s7.py",
                  "recovery/PreGANSrc/src/ftmoe_ablation.py",
                  "recovery/PreGANSrc/src/ftmoe_end_to_end.py",
                  "recovery/PreGANSrc/src/ftmoe_normalization.py",
                  "recovery/PreGANSrc/src/ftmoe_input_contract.py"]
    config = {"schema_version": 1, "protocol": "020", "phase": args.phase,
              "scheme": "s7", "checkpoint_mode": "s6_adapted",
              "method": args.method, "model_seed": args.model_seed,
              "replay_seed": manifest["seed"], "steps": manifest["steps"],
              "learning_rate": lr, "source_checkpoint": source,
              "stream": str(args.stream.resolve()),
              "stream_sha256": manifest["stream_sha256"],
              "stream_manifest_sha256": sha(args.stream / "manifest.json"),
              "input_contract_version": 2, "normalization_version": 2,
              "graph_semantics_version": 3,
              "capacity_control_version": 1,
              "online_optimizer_version": 3,
              "normalization_v2": {
                  "scale": v2_scale.tolist(),
                  "fallback_columns": fallback_columns,
                  "artifact": "artifacts/ftmoe_online/protocol_020/normalization_v2_time_scale.json",
              },
              "s7_memory": {"recent_maxlen": RECENT_MAXLEN,
                            "uniform_per_update": RECENT_UNIFORM_PER_UPDATE,
                            "event_per_update": EVENT_PER_UPDATE,
                            "anchor_per_update": ANCHOR_PER_UPDATE,
                            "max_exposure": MAX_EXPOSURE,
                            "max_rare_exposure": MAX_RARE_EXPOSURE,
                            "anchor_source": "protocol-020 train episodes rows 272-399",
                            "lambda_anchor": LAMBDA_ANCHOR,
                            "lambda_distill": LAMBDA_DISTILL,
                            "teacher": "frozen starting checkpoint",
                            "lr_fast": lr, "lr_slow_factor": LR_SLOW_FACTOR,
                            "slow_groups": ["encoder.", "graph_encoder.", "cmha"]},
              "loss_v3": class_balance,
              "threshold": .5, "label_tolerance": 1, "label_delay_intervals": 1,
              "resource_guard_ram_gib": float(os.environ.get("FTMOE020_RAM_GUARD_GIB", "3.0")),
              "code_sha256": {name: sha(ROOT / name) for name in code_files}}
    out = args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError(out)
        if json.loads((out / "configuration.json").read_text()) != config:
            raise ValueError("Resume configuration changed")
        if (out / "summary.json").exists():
            raise ValueError("Run already completed")
    else:
        if args.resume:
            raise FileNotFoundError(out)
        out.mkdir(parents=True)
        write_json(out / "configuration.json", config)
    session = S7Session(checkpoint, args.method, args.model_seed, replay, lr,
                        manifest["seed"], v2_scale, anchor_pool, teacher,
                        class_balance, window_fn="window_v3")
    from train_ftmoe_end_to_end import load_data
    _, validation, normalization, _ = load_data(
        ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical")
    elapsed_before = 0.
    if args.resume:
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        session.restore(saved["session"])
        elapsed_before = saved["elapsed_seconds"]
    else:
        session.evaluate_reference(validation)
    began = time.perf_counter()
    end = args.stop_after or replay.steps
    if not session.cursor < end <= replay.steps:
        raise ValueError("Invalid stop-after cursor")
    try:
        with (out / "predictions.jsonl").open("w", encoding="utf8") as journal:
            for t in range(session.cursor):
                journal.write(json.dumps({"step": t + 1,
                    "model_version": int(session.predictions["model_version"][t]),
                    "probability": session.predictions["probability"][t].tolist(),
                    "class_probability": session.predictions["class_probability"][t].tolist(),
                    **{k: float(session.predictions[k][t]) for k in
                       ("expert_count", "mean_active", "unmatched_ratio")}}) + "\n")
            def sink(value):
                journal.write(json.dumps(value) + "\n")
                journal.flush()
            while session.cursor < end:
                if session.cursor % 10 == 0:
                    resources()
                session.step(sink)
                if session.cursor % 500 == 0:
                    session.evaluate_reference(validation)
                if session.cursor % 100 == 0 or session.cursor == end:
                    state = {"session": session.save(), "configuration": config,
                             "elapsed_seconds": elapsed_before +
                             time.perf_counter() - began}
                    save_state(out / f"step{session.cursor:05d}.pt", state)
                    save_state(out / "resume.pt", state)
                    print(json.dumps({"method": args.method, "step": session.cursor,
                                      "experts": len(session.model.eagate.ids),
                                      "elapsed_seconds": state["elapsed_seconds"]}),
                          flush=True)
        if end < replay.steps:
            return
        session.finish()
        if session.reference[-1]["step"] != session.cursor:
            session.evaluate_reference(validation)
        if session.model.frozen_hash() != session.initial_frozen_hash:
            raise AssertionError("Frozen parameters changed")
        metrics = summarize_arrays(**{k: session.predictions[k] for k in
            ("probability", "class_probability", "labels", "raw_labels")})
        n = session.replay.steps
        half, quarter = n // 2, n // 4
        bounds = {"full": (0, n), "first_half": (0, half), "second_half": (half, n),
                  "first_quarter": (0, quarter), "last_quarter": (n - quarter, n)}
        gate_metrics = {}
        for slice_name, (start, end_slice) in bounds.items():
            segment_labels = session.predictions["labels"][start:end_slice]
            segment_probability = session.predictions["probability"][start:end_slice]
            valid = segment_labels >= 0
            if valid.any():
                gate_metrics[slice_name] = extra_detection_metrics(
                    segment_probability[valid], segment_labels[valid])
            else:
                gate_metrics[slice_name] = None
        component_rows = [item.get("components") for item in session.updates
                          if item.get("components") is not None]
        exposure_values = sorted(session.exposure.values())
        rare_values = sorted(session.rare_exposure.values())

        def safe_mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None

        summary = {"configuration": config, "metrics": metrics,
                   "gate_metrics": gate_metrics,
                   "initial_state_hash": session.initial_state_hash,
                   "final_state_hash": session.model.state_hash(),
                   "frozen_parameters_unchanged": True,
                   "capacity_before": session.capacity_before,
                   "capacity_after": session.capacity_after,
                   "updates": session.update_number,
                   "final_experts": list(session.model.eagate.ids),
                   "elapsed_seconds": elapsed_before + time.perf_counter() - began,
                   "rss_gib": resources(),
                   "prediction_mean_seconds":
                       float(session.predictions["prediction_seconds"].mean()),
                   "prediction_p95_seconds": float(np.percentile(
                       session.predictions["prediction_seconds"], 95)),
                   "update_total_seconds":
                       sum(item["seconds"] for item in session.updates),
                   "s7_statistics": {
                       "updates_with_components": len(component_rows),
                       "mean_loss_online": safe_mean([r.get("loss_online")
                                                      for r in component_rows]),
                       "mean_loss_anchor": safe_mean([r.get("loss_anchor")
                                                      for r in component_rows]),
                       "mean_loss_distill": safe_mean([r.get("loss_distill")
                                                       for r in component_rows]),
                       "mean_recent_windows": safe_mean([r.get("recent_windows")
                                                         for r in component_rows]),
                       "mean_anchor_windows": safe_mean([r.get("anchor_windows")
                                                         for r in component_rows]),
                       "exposure_max": max(exposure_values) if exposure_values else None,
                       "exposure_violations": int(sum(1 for v in exposure_values
                                                      if v > MAX_EXPOSURE)),
                       "rare_exposure_max": max(rare_values) if rare_values else None,
                       "rare_exposure_violations": int(sum(
                           1 for v in rare_values if v > MAX_RARE_EXPOSURE)),
                   }}
        np.savez_compressed(out / "predictions.npz", **session.predictions)
        write_json(out / "updates.json", session.updates)
        write_json(out / "reference.json", session.reference)
        state = {"session": session.save(), "configuration": config,
                 "elapsed_seconds": summary["elapsed_seconds"]}
        save_state(out / "last.pt", state)
        save_state(out / "resume.pt", state)
        write_json(out / "summary.json", summary)
        print(json.dumps({"completed": str(out),
                          "second_half": summary["metrics"]["second_half"],
                          "s7": summary["s7_statistics"]}), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"error": str(exc), "step": session.cursor,
                                          "traceback": traceback.format_exc(),
                                          "next_action": "Report before changing the registered experiment"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list("ABCD"), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--base-lr", type=float, default=None,
                        help="fast-group LR for B/C (registered: 1e-5/3e-5/1e-4; default 1e-5)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", default="S7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    run(parser.parse_args())
