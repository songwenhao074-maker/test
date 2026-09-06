"""Protocol 019 S4 — online A/B/C comparison on the S4 update scheme.

Prequential replay identical to run_ftmoe_protocol019.py (predict -> reveal ->
mature -> update; tolerance ±1 labels maturing one interval late; frozen-hash
guards; reference probes; identical output layout), but the session and
update machinery come from recovery/PreGANSrc/src/ftmoe_online_s4.py
(Recent+Anchor memory, exposure cap, anchor + distill losses, grouped LR).

The common starting checkpoint is the S3-adapted v4 (user decision
2026-09-06: accept adapted_v4_seed1/best.pt and proceed to S4).
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
    S4Session, load_anchor_pool, RECENT_MAXLEN, RECENT_PER_UPDATE,
    ANCHOR_PER_UPDATE, MAX_EXPOSURE, LAMBDA_ANCHOR, LAMBDA_DISTILL,
    LR_FAST, LR_SLOW_FACTOR,
)
from recovery.PreGANSrc.src.ftmoe_online_s5 import OnlineFTMoEV2, S5Session
from run_ftmoe_protocol019 import (
    ART, ReplayV2, extra_detection_metrics, load_v2_time_scale, resources, sha,
)
from run_ftmoe_online import write_json, save_state
from analyze_ftmoe_online import summarize_arrays

ROOT = Path(__file__).resolve().parent


def run(args):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resources()
    if args.checkpoint_path is None:
        raise ValueError("S4 requires --checkpoint-path (S3-adapted checkpoint)")
    manifest = json.loads((args.stream / "manifest.json").read_text())
    if manifest["stream_sha256"] != sha(args.stream / "stream.npz"):
        raise AssertionError("Stream hash mismatch")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "v4" or checkpoint["seed"] != args.model_seed:
        raise AssertionError("Checkpoint identity mismatch")
    source = {"path": str(checkpoint_path.resolve()), "sha256": sha(checkpoint_path),
              "epoch": checkpoint["epoch"], "mode": "s3_adapted"}
    with np.load(args.stream / "stream.npz") as data:
        arrays = {key: data[key] for key in data.files}
    v2_scale, fallback_columns = load_v2_time_scale(checkpoint["normalization"])
    graph_scale = np.asarray(checkpoint["normalization"]["graph_scale"], np.float64)
    replay = ReplayV2(arrays, v2_scale, graph_scale, manifest["steps"])
    lr = 0. if args.method == "A" else (args.base_lr if args.base_lr is not None else LR_FAST)
    if args.method != "A" and lr not in (1e-5, 3e-5, 1e-4):
        raise ValueError(f"Unregistered S4 base learning rate: {lr}")
    if args.method == "D" and args.gate_version != "v2":
        raise ValueError("Protocol 019 runs D only with the v2 dynamic gate (plan §10)")
    if args.gate_version == "v2":
        session_class = S5Session
        teacher_class = OnlineFTMoEV2
        dynamic_expert_version = 2
    else:
        session_class = S4Session
        teacher_class = OnlineFTMoE
        dynamic_expert_version = 1
    anchor_pool = load_anchor_pool()
    teacher = teacher_class(checkpoint, "A", args.model_seed)
    teacher.eval()
    code_files = ["run_ftmoe_protocol019_s4.py", "run_ftmoe_protocol019.py",
                  "run_ftmoe_online.py", "analyze_ftmoe_online.py",
                  "train_ftmoe_ablation_existing.py", "train_ftmoe_end_to_end.py",
                  "train_ftmoe_protocol019_s3.py",
                  "recovery/PreGANSrc/src/ftmoe_online.py",
                  "recovery/PreGANSrc/src/ftmoe_online_s4.py",
                  "recovery/PreGANSrc/src/ftmoe_online_s5.py",
                  "recovery/PreGANSrc/src/ftmoe_ablation.py",
                  "recovery/PreGANSrc/src/ftmoe_end_to_end.py",
                  "recovery/PreGANSrc/src/ftmoe_normalization.py",
                  "recovery/PreGANSrc/src/ftmoe_input_contract.py"]
    config = {"schema_version": 3, "protocol": "019", "phase": args.phase,
              "scheme": "s4", "checkpoint_mode": "s3_adapted",
              "method": args.method, "model_seed": args.model_seed,
              "replay_seed": manifest["seed"], "steps": manifest["steps"],
              "learning_rate": lr, "source_checkpoint": source,
              "stream": str(args.stream.resolve()),
              "stream_sha256": manifest["stream_sha256"],
              "stream_manifest_sha256": sha(args.stream / "manifest.json"),
              "input_contract_version": 2, "normalization_version": 2,
              "graph_semantics_version": 2, "dynamic_expert_version": dynamic_expert_version,
              "gate_version": args.gate_version,
              "normalization_v2": {
                  "scale": v2_scale.tolist(),
                  "fallback_columns": fallback_columns,
                  "artifact": "artifacts/ftmoe_online/protocol_019/normalization_v2_time_scale.json",
                  "normalized_abs_max_report": replay.normalization_report,
              },
              "s4_memory": {"recent_maxlen": RECENT_MAXLEN,
                            "recent_per_update": RECENT_PER_UPDATE,
                            "anchor_per_update": ANCHOR_PER_UPDATE,
                            "max_exposure": MAX_EXPOSURE,
                            "anchor_seeds": [401, 402, 403],
                            "anchor_rows": "272-399 of each 400-row episode",
                            "lambda_anchor": LAMBDA_ANCHOR,
                            "lambda_distill": LAMBDA_DISTILL,
                            "teacher": "frozen starting checkpoint",
                            "lr_fast": lr, "lr_slow_factor": LR_SLOW_FACTOR,
                            "slow_groups": ["encoder.", "graph_encoder.", "cmha"],
                            "interval_stratification": "none (registered S4-dev choice)"},
              "threshold": .5, "label_tolerance": 1, "label_delay_intervals": 1,
              "resource_guard_ram_gib": float(os.environ.get("FTMOE019_RAM_GUARD_GIB", "3.0")),
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
    session = session_class(checkpoint, args.method, args.model_seed, replay, lr,
                            manifest["seed"], v2_scale, anchor_pool, teacher)
    from train_ftmoe_end_to_end import load_data
    _, validation, normalization, _ = load_data(
        ROOT / "artifacts/ftmoe_end_to_end/data/protocol_004_physical")
    # adapted checkpoints carry normalization v2; anchor reference stays legacy.
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
                             "elapsed_seconds": elapsed_before + time.perf_counter() - began}
                    save_state(out / f"step{session.cursor:05d}.pt", state)
                    save_state(out / "resume.pt", state)
                    print(json.dumps({"method": args.method, "step": session.cursor,
                                      "experts": len(session.model.eagate.ids),
                                      "elapsed_seconds": state["elapsed_seconds"]}), flush=True)
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
                   "prediction_mean_seconds": float(session.predictions["prediction_seconds"].mean()),
                   "prediction_p95_seconds": float(np.percentile(session.predictions["prediction_seconds"], 95)),
                   "update_total_seconds": sum(item["seconds"] for item in session.updates),
                   "s4_statistics": {
                       "updates_with_components": len(component_rows),
                       "mean_loss_online": safe_mean([r.get("loss_online") for r in component_rows]),
                       "mean_loss_anchor": safe_mean([r.get("loss_anchor") for r in component_rows]),
                       "mean_loss_distill": safe_mean([r.get("loss_distill") for r in component_rows]),
                       "mean_recent_windows": safe_mean([r.get("recent_windows") for r in component_rows]),
                       "mean_anchor_windows": safe_mean([r.get("anchor_windows") for r in component_rows]),
                       "exposure_max": max(exposure_values) if exposure_values else None,
                       "exposure_violations": int(sum(1 for v in exposure_values if v > MAX_EXPOSURE)),
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
                          "s4": summary["s4_statistics"]}), flush=True)
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
                        help="fast-group LR for B/C/D (registered: 1e-5/3e-5/1e-4; default 3e-5)")
    parser.add_argument("--gate-version", choices=("v1", "v2"), default="v1",
                        help="EAGate dynamics: v1 legacy topology (B/C only) or v2 plan-§10 gate (D requires v2)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", default="S4")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    run(parser.parse_args())
