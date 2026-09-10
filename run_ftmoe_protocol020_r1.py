"""Protocol-020 R1 runner: A/C-legacy/fixed frozen-base residual replay.

The runner reuses the registered Protocol-020 ``ReplayV3``, P20 v2 scale,
anchor pool and resource guard.  It writes a causal deployment journal,
R1 ledgers/events, 100-step resumable checkpoints, and the final metrics.
The only accepted methods are ``A``, ``C-legacy``, ``C-residual-off`` and
``C-residual-on``.  C methods use the fixed 1e-4 learning rate; A is 0.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "3"

import numpy as np
import psutil
import torch

from analyze_ftmoe_online import summarize_arrays
from run_ftmoe_online import save_state, write_json
from run_ftmoe_protocol019 import extra_detection_metrics
from run_ftmoe_protocol020 import (ART, ReplayV3, load_anchor_pool_v3,
                                   load_v2_time_scale_p20, resources, sha)
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_s7 import (
    ANCHOR_PER_UPDATE, EVENT_PER_UPDATE, MAX_EXPOSURE, MAX_RARE_EXPOSURE,
    RECENT_MAXLEN, RECENT_UNIFORM_PER_UPDATE, S7Session)

ROOT = Path(__file__).resolve().parent
METHODS = ("A", "C-legacy", "C-residual-off", "C-residual-on")
RESIDUAL = frozenset(METHODS[2:])
LR = 1e-4


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_jsonl(path, rows):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row), ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def save_predictions(path, predictions):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **predictions)
    temporary.replace(path)


LABEL_KEYS = frozenset(("label", "labels", "raw_label", "raw_labels", "target",
                        "targets", "y", "future_label", "future_labels",
                        "matured_label", "matured_labels"))
POST_PREDICTION_KEYS = frozenset((
    "matured_step", "maturity_raw_label_index", "label_matured",
    "assessment_excluded_reason", "rollback_event",
))


def deployment_view(value):
    """Keep prediction records free of observed/future target labels."""
    if isinstance(value, dict):
        return {k: deployment_view(v) for k, v in value.items()
                if str(k) not in LABEL_KEYS and str(k) not in POST_PREDICTION_KEYS}
    if isinstance(value, (list, tuple)):
        return [deployment_view(v) for v in value]
    return jsonable(value)


def _has_deployment_label(value):
    if isinstance(value, dict):
        if any(str(key) in LABEL_KEYS or str(key) in POST_PREDICTION_KEYS
               for key in value):
            return True
        return any(_has_deployment_label(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_deployment_label(child) for child in value)
    return False


def prepare_resume_journal(path, cursor):
    """Keep the original causal prefix and discard only an uncheckpointed tail."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Resume output has no predictions.jsonl")
    lines = [line for line in path.read_text(encoding="utf8").splitlines(True)
             if line.strip()]
    if len(lines) < int(cursor):
        raise ValueError("predictions.jsonl is shorter than resume cursor")
    prefix = lines[:int(cursor)]
    for line in prefix:
        row = json.loads(line)
        if not isinstance(row, dict) or not {"step", "probability", "class_probability"} <= set(row):
            raise ValueError("predictions.jsonl prefix is not the registered deployment schema")
        if _has_deployment_label(row):
            raise ValueError("predictions.jsonl prefix contains observed/future labels")
    if len(lines) != len(prefix):
        temporary = Path(str(path) + ".tmp")
        temporary.write_text("".join(prefix), encoding="utf8")
        temporary.replace(path)


def prediction_rows(session):
    for name in ("prediction_records", "deployment_records", "prediction_log",
                 "deployment_log"):
        records = getattr(session, name, None)
        if isinstance(records, dict):
            records = list(records.values())
        if isinstance(records, (list, tuple)):
            return [deployment_view(row) for row in records]
    p = session.predictions
    rows = []
    for index in range(session.cursor):
        row = {"step": index + 1,
               "model_version": int(p["model_version"][index]),
               "probability": p["probability"][index].tolist(),
               "class_probability": p["class_probability"][index].tolist()}
        for name in ("expert_count", "mean_active", "unmatched_ratio"):
            if name in p:
                row[name] = jsonable(p[name][index])
        rows.append(row)
    return rows


def session_records(session, names):
    for name in names:
        value = getattr(session, name, None)
        if value is None or callable(value):
            continue
        if isinstance(value, dict):
            if value and all(isinstance(v, dict) for v in value.values()):
                return list(value.values())
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def persist_snapshots(out, session):
    save_predictions(out / "predictions.npz", session.predictions)
    write_jsonl(out / "updates.jsonl", getattr(session, "updates", []))
    write_jsonl(out / "r1_ledger.jsonl", session_records(
        session, ("r1_ledger", "ledger", "protection_ledger")))
    write_jsonl(out / "r1_events.jsonl", session_records(
        session, ("r1_events", "events", "event_log", "protection_events")))
    write_json(out / "reference.json", getattr(session, "reference", []))


def ranges(manifest, steps, block_size=400):
    phases = manifest.get("phases") or []
    phase_len = int(manifest.get("phase_len") or steps)
    registered = []
    for index, phase in enumerate(phases):
        start, end = index * phase_len, min(steps, (index + 1) * phase_len)
        if start >= steps:
            break
        name = phase.get("name", f"phase_{index}") if isinstance(phase, dict) else str(phase)
        registered.append({"index": index, "name": name, "start": start,
                           "end": end, "kind": "registered_phase"})
    if not registered:
        registered = [{"index": 0, "name": "registered_stream", "start": 0,
                       "end": steps, "kind": "registered_phase"}]
    blocks = [{"index": i, "name": f"time_block_{i}", "start": start,
               "end": min(steps, start + block_size), "kind": "time_block_400"}
              for i, start in enumerate(range(0, steps, block_size))]
    return registered, blocks


def segment_metrics(predictions, start, end):
    from train_ftmoe_end_to_end import metric_arrays
    p = predictions["probability"][start:end]
    c = predictions["class_probability"][start:end]
    y = predictions["labels"][start:end]
    raw = predictions["raw_labels"][start:end]
    valid, raw_valid = y >= 0, raw >= 0
    if not valid.any() or not raw_valid.any():
        return {"status": "unfinalized", "tolerance": None, "raw": None}

    def one(probability, classes, labels):
        from sklearn.metrics import f1_score
        flat_p = probability.reshape(-1)
        flat_c = classes.reshape(-1, 3)
        flat_y = labels.reshape(-1)
        metrics = metric_arrays(flat_p, flat_c, flat_y)
        predicted_label = np.where(flat_p >= .5, flat_c.argmax(axis=1) + 1, 0)
        class_f1 = f1_score(flat_y, predicted_label, labels=[1, 2, 3],
                            average=None, zero_division=0)
        end_to_end = {
            "macro_f1": float(f1_score(
                flat_y, predicted_label, labels=[1, 2, 3],
                average="macro", zero_division=0)),
            "class_f1": {str(index): float(value)
                          for index, value in zip((1, 2, 3), class_f1)},
            "definition": "predicted_label=np.where(p>=0.5,argmax(classp)+1,0)",
        }
        return {"metrics": metrics,
                "extra_detection": extra_detection_metrics(flat_p, flat_y),
                "end_to_end": end_to_end}

    return {"status": "complete", "tolerance": one(p[valid], c[valid], y[valid]),
            "raw": one(p[raw_valid], c[raw_valid], raw[raw_valid])}


def summary_for(config, manifest, session, elapsed):
    p = session.predictions
    metrics = summarize_arrays(p["probability"], p["class_probability"],
                               p["labels"], p["raw_labels"])
    phase_ranges, block_ranges = ranges(manifest, session.replay.steps)
    def report_ranges(items):
        return [{**{key: item[key] for key in ("index", "name", "start", "end", "kind")},
                 "metrics": segment_metrics(p, item["start"], item["end"])}
                for item in items]
    r1_summary = {}
    for name in ("r1_summary", "r1_statistics", "protection_summary"):
        value = getattr(session, name, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        r1_summary = jsonable(value)
        break
    return jsonable({
        "schema_version": 1, "status": "complete", "configuration": config,
        "metrics": metrics, "gate_metrics": gate_metrics(p),
        "overall_diagnostics": segment_metrics(p, 0, session.replay.steps),
        "registered_phase_metrics": report_ranges(phase_ranges),
        "time_block_metrics_400": report_ranges(block_ranges),
        "phase_semantics": {"registered_phase_count": len(phase_ranges),
                             "dev501_single_phase_is_not_split_into_fault_phases": True,
                             "time_blocks_are_reporting_slices_only": True},
        "reference_source": "same_domain_train_anchor",
        "reference_records": len(getattr(session, "reference", [])),
        "r1_summary": r1_summary,
        "learner_state_hash": r1_summary.get("learner_state_hash") if isinstance(r1_summary, dict) else None,
        "initial_state_hash": session.initial_state_hash,
        "final_state_hash": session.model.state_hash(),
        "frozen_parameters_unchanged": session.model.frozen_hash() == session.initial_frozen_hash,
        "capacity_before": session.capacity_before, "capacity_after": session.capacity_after,
        "updates": int(session.update_number),
        "final_experts": list(getattr(session.model.eagate, "ids", [])),
        "elapsed_seconds": float(elapsed), "rss_gib": float(resources()),
        "prediction_mean_seconds": float(np.mean(p["prediction_seconds"])),
        "prediction_p95_seconds": float(np.percentile(p["prediction_seconds"], 95)),
        "update_total_seconds": float(sum(float(x.get("seconds", 0.0))
                                           for x in getattr(session, "updates", []))),
    })


def gate_metrics(predictions):
    n = len(predictions["labels"])
    result = {}
    for name, (start, end) in {"full": (0, n), "first_half": (0, n // 2),
                               "second_half": (n // 2, n),
                               "first_quarter": (0, n // 4),
                               "last_quarter": (n - n // 4, n)}.items():
        valid = predictions["labels"][start:end] >= 0
        result[name] = (extra_detection_metrics(
            predictions["probability"][start:end][valid],
            predictions["labels"][start:end][valid]) if valid.any() else None)
    return result


def load_inputs(args):
    stream = Path(args.stream)
    manifest = json.loads((stream / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("protocol") != "020":
        raise ValueError("Stream is not registered under Protocol 020")
    if manifest["stream_sha256"] != sha(stream / "stream.npz"):
        raise AssertionError("Stream hash mismatch")
    with np.load(stream / "stream.npz") as data:
        arrays = {key: data[key] for key in data.files}
    steps = int(manifest["steps"])
    capacities = np.asarray(arrays.get("capacities"), dtype=np.float64)
    if capacities.shape != (steps + 1, 16, 3):
        raise ValueError("Protocol-020 R1 requires capacities_per_interval")
    arrays["capacities_per_interval"], arrays["capacities"] = capacities, capacities[0]
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "v4" or checkpoint.get("seed") != args.model_seed:
        raise AssertionError("Checkpoint identity mismatch")
    if checkpoint.get("graph_semantics_version") != 3:
        raise ValueError("R1 requires graph_semantics_version 3")
    v2_scale, fallback = load_v2_time_scale_p20(checkpoint["normalization"])
    graph_scale = np.asarray(checkpoint["normalization"]["graph_scale"], dtype=np.float64)
    return (manifest, ReplayV3(arrays, v2_scale, graph_scale, steps), checkpoint,
            v2_scale, fallback)


def code_hashes(method):
    files = ["run_ftmoe_protocol020_r1.py", "run_ftmoe_protocol020.py",
             "run_ftmoe_protocol019.py", "run_ftmoe_online.py",
             "analyze_ftmoe_online.py", "train_ftmoe_end_to_end.py",
             "recovery/PreGANSrc/src/ftmoe_online.py",
             "recovery/PreGANSrc/src/ftmoe_ablation.py",
             "recovery/PreGANSrc/src/ftmoe_end_to_end.py",
             "recovery/PreGANSrc/src/ftmoe_online_s4.py",
             "recovery/PreGANSrc/src/ftmoe_online_s7.py",
             "recovery/PreGANSrc/src/ftmoe_input_contract.py"]
    if method in RESIDUAL:
        files.append("recovery/PreGANSrc/src/ftmoe_online_r1.py")
    missing = [name for name in files if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing provenance file(s): " + ", ".join(missing))
    return {name: sha(ROOT / name) for name in files}


def make_config(args, manifest, checkpoint, v2_scale, fallback, balance, source):
    method = args.method
    return {
        "schema_version": 1, "protocol": "020", "revision": "20260909-r1",
        "phase": "R1", "method": method,
        "method_family": "frozen_base_residual" if method in RESIDUAL else
        ("s7_legacy" if method == "C-legacy" else "frozen_base"),
        "protection_enabled": method == "C-residual-on",
        "model_class": "FrozenResidualFTMoE" if method in RESIDUAL else "OnlineFTMoE",
        "session_class": "R1Session" if method in RESIDUAL else "S7Session",
        "model_seed": int(args.model_seed), "replay_seed": int(manifest["seed"]),
        "start_rng_seed": int((args.model_seed * 7919 + manifest["seed"]) % (2 ** 32)),
        "steps": int(manifest["steps"]), "learning_rate": float(0.0 if method == "A" else LR),
        "source_checkpoint": source, "stream": str(Path(args.stream).resolve()),
        "stream_sha256": manifest["stream_sha256"],
        "stream_manifest_sha256": sha(Path(args.stream) / "manifest.json"),
        "input_contract_version": 2, "normalization_version": 2,
        "graph_semantics_version": 3, "capacity_control_version": 1,
        "online_optimizer_version": 3,
        "s7_memory": {"recent_maxlen": RECENT_MAXLEN,
                       "uniform_per_update": RECENT_UNIFORM_PER_UPDATE,
                       "event_per_update": EVENT_PER_UPDATE,
                       "anchor_per_update": ANCHOR_PER_UPDATE,
                       "max_exposure": MAX_EXPOSURE,
                       "max_rare_exposure": MAX_RARE_EXPOSURE,
                       "update_interval": 10, "anchor_source": "protocol-020 train episodes rows 272-399",
                       "lambda_anchor": 0.25, "lambda_distill": 0.10,
                       "loss": ".7 detection CE + .3 positive classification CE + .5 joint ranking + .01 balance"},
        "loss_v3": deepcopy(balance), "threshold": 0.5,
        "label_tolerance": 1, "label_delay_intervals": 1,
        "resource_guard_ram_gib": float(os.environ.get("FTMOE020_RAM_GUARD_GIB", "3.0")),
        "normalization_v2": {"scale": np.asarray(v2_scale).tolist(),
                             "fallback_columns": list(fallback),
                             "artifact": "artifacts/ftmoe_online/protocol_020/normalization_v2_time_scale.json"},
        "reference_source": "same_domain_train_anchor",
        "data_scope": "development_streams_only", "code_sha256": code_hashes(method),
    }


def make_session(args, checkpoint, replay, v2_scale, pool, teacher, balance):
    if args.method == "A":
        return S7Session(checkpoint, "A", args.model_seed, replay, 0.0,
                         args.replay_seed,
                         v2_scale, pool, teacher, balance, window_fn="window_v3")
    if args.method == "C-legacy":
        return S7Session(checkpoint, "C", args.model_seed, replay, LR,
                         args.replay_seed, v2_scale, pool, teacher, balance,
                         window_fn="window_v3")
    from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE, R1Session
    if getattr(R1Session, "model_class", None) is not FrozenResidualFTMoE:
        raise AssertionError("R1Session.model_class must be FrozenResidualFTMoE")
    return R1Session(checkpoint, "C", args.model_seed, replay, LR,
                     args.replay_seed, v2_scale, pool, teacher, balance,
                     window_fn="window_v3",
                     protection_enabled=args.method == "C-residual-on")


def run(args):
    if args.method not in METHODS:
        raise ValueError(args.method)
    if args.method == "A":
        if args.learning_rate not in (None, 0.0, LR):
            raise ValueError("A learning rate is fixed at 0")
    elif args.learning_rate not in (None, LR):
        raise ValueError("C learning rate is fixed at 1e-4")
    torch.set_num_threads(3)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError, OSError, psutil.Error):
        pass
    resources()
    if args.replay_seed is not None and args.replay_seed not in (500, 501):
        raise ValueError("R1 replay seed must be 500 or 501")
    manifest, replay, checkpoint, v2_scale, fallback = load_inputs(args)
    if int(manifest["seed"]) not in (500, 501):
        raise ValueError("R1 fixed comparison requires replay seed 500 or 501")
    if args.replay_seed is not None and int(manifest["seed"]) != args.replay_seed:
        raise ValueError("--replay-seed does not match stream manifest")
    args.replay_seed = int(manifest["seed"])
    seed = (args.model_seed * 7919 + args.replay_seed) % (2 ** 32)
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    pool, balance = load_anchor_pool_v3()
    teacher = OnlineFTMoE(checkpoint, "A", args.model_seed); teacher.eval()
    source = {"path": str(Path(args.checkpoint_path).resolve()),
              "sha256": sha(args.checkpoint_path), "epoch": checkpoint.get("epoch"),
              "mode": "s6_adapted"}
    config = make_config(args, manifest, checkpoint, v2_scale, fallback, balance, source)
    out = Path(args.output)
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use a new directory or --resume: {out}")
        if json.loads((out / "configuration.json").read_text(encoding="utf8")) != config:
            raise ValueError("Resume configuration changed")
        if (out / "summary.json").exists():
            raise ValueError("Run already completed")
        if not (out / "resume.pt").is_file():
            raise FileNotFoundError("Resume output has no resume.pt")
        if not (out / "predictions.jsonl").is_file():
            raise FileNotFoundError("Resume output has no predictions.jsonl")
    else:
        if args.resume:
            raise FileNotFoundError(out)
        out.mkdir(parents=True); write_json(out / "configuration.json", config)
    session = make_session(args, checkpoint, replay, v2_scale, pool, teacher, balance)
    elapsed_before = 0.0
    if args.resume:
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if state.get("configuration") != config:
            raise ValueError("Resume state configuration changed")
        session.restore(state["session"]); elapsed_before = float(state.get("elapsed_seconds", 0.0))
    else:
        session.evaluate_reference(None)
    end = replay.steps if args.stop_after is None else int(args.stop_after)
    if not session.cursor < end <= replay.steps:
        raise ValueError("Invalid stop-after cursor")
    began = time.perf_counter()

    def checkpoint(final=False):
        elapsed = elapsed_before + time.perf_counter() - began
        persist_snapshots(out, session)
        state = {"schema_version": 1, "status": "complete" if final else "partial",
                 "cursor": int(session.cursor), "configuration": config,
                 "elapsed_seconds": elapsed, "session": session.save()}
        save_state(out / f"step{session.cursor:05d}.pt", state)
        save_state(out / "resume.pt", state)
        return elapsed

    def sink(row):
        journal.write(json.dumps(deployment_view(row), ensure_ascii=False,
                                 allow_nan=False) + "\n")
        journal.flush()

    try:
        journal_path = out / "predictions.jsonl"
        if args.resume:
            prepare_resume_journal(journal_path, session.cursor)
        journal_mode = "a" if args.resume else "w"
        with journal_path.open(journal_mode, encoding="utf8") as journal:
            journal.flush()
            while session.cursor < end:
                if session.cursor % 10 == 0:
                    resources()
                session.step(sink)
                # The final tolerance label is not mature until ``finish``;
                # defer the endpoint reference evaluation so it describes the
                # completed deployment boundary rather than the pre-finish
                # state at cursor=steps.
                if session.cursor % 500 == 0 and session.cursor < replay.steps:
                    session.evaluate_reference(None)
                if session.cursor % 100 == 0 or session.cursor == end:
                    elapsed = checkpoint()
                    print(json.dumps({"method": args.method, "step": session.cursor,
                                      "elapsed_seconds": elapsed}), flush=True)
        if end < replay.steps:
            return
        session.finish()
        if not (session.predictions["labels"] >= 0).all():
            raise AssertionError("Final label maturation incomplete")
        session.evaluate_reference(None)
        if session.model.frozen_hash() != session.initial_frozen_hash:
            raise AssertionError("Frozen parameters changed")
        elapsed = elapsed_before + time.perf_counter() - began
        summary = summary_for(config, manifest, session, elapsed)
        checkpoint(final=True)
        write_json(out / "summary.json", summary)
        print(json.dumps({"completed": str(out), "method": args.method,
                          "f1": summary["metrics"]["full"]["f1"]}), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"error": str(exc), "step": session.cursor,
                                          "traceback": traceback.format_exc()})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model-seed", type=int, default=1)
    p.add_argument("--replay-seed", type=int)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("--learning-rate", "--base-lr", dest="learning_rate", type=float)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
