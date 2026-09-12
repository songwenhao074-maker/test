"""Protocol 022 P22-S5 — fixed-C capacity / trainability upper-bound diagnostic.

Purpose (plan §9)
-----------------
Separate two very different explanations before any online experiment:

    "the fixed correction's structure/features cannot learn this regime"
        from
    "the online optimisation strategy did not learn it"

So this is an *offline* capacity diagnostic on a strict temporal split of the
Protocol-022 development stream, not the final online claim:

    train      : early mature segment of Unseen-1   (labels matured first)
    validate   : mid segment of Unseen-1            (never trained on)
    test       : late segment of Unseen-1           (never trained on)
    recurrence : Unseen-recur                       (mechanism seen before)
    familiar-1 : before any unseen exposure         (anchor reference)
    familiar-2 : after the unseen block             (retention)

Rules taken literally from the plan: only the residual is trained; the base
logits/backbone stay frozen (verified by a frozen-parameter hash); no
phase/cascade identifier is ever an input; nothing is tuned on the test segment.

Variants (all share one base pass per step, so the comparison is paired)
    A              frozen base, no correction at all
    C-current      registered fixed residual bank (64 -> 32 -> 5, 4 experts)
    C-wide         wider fixed residual (64 -> 64 -> 5, 8 experts): the
                   "more capacity" control that must be beaten for a claim that
                   extra capacity is *needed*, not merely present
    C-budget       registered residual given a larger optimisation budget
                   (plan §14: extra gradient budget instead of extra parameters)
    C-causal       registered residual plus the causal capacity/temporal
                   features the P20 plan §1.4 item 4 asked for

Gate (plan §9.4)
    (future PR-AUC(A -> best fixed C) >= +0.03 absolute
     OR future onset AP >= +0.05 absolute)
    AND anchor/familiar F1 drop <= 0.03 absolute
    else STOP-CAP: fix the representation/feature path first; no online grid, no D.

Usage:
    python run_ftmoe_protocol022_s5.py --variants A C-current
    python run_ftmoe_protocol022_s5.py --all --seed 1
"""
import argparse
from contextlib import ExitStack
import datetime as _dt
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "3")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
P22 = ROOT / "artifacts/ftmoe_online/protocol_022"
STREAM_DIR = P22 / "development_streams"
CHECKPOINT = ROOT / "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt"


def registered_stream():
    """The collected development stream, found rather than hard-coded."""
    if not STREAM_DIR.is_dir():
        raise FileNotFoundError("no development stream directory: %s" % STREAM_DIR)
    candidates = []
    for directory in sorted(STREAM_DIR.glob("dev_seed*")):
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        payload = json.loads(manifest.read_text(encoding="utf8"))
        if payload.get("registered") and payload.get("protocol") == "022":
            candidates.append((int(payload.get("steps", 0)), directory))
    if not candidates:
        raise FileNotFoundError(
            "no REGISTERED protocol-022 development stream in %s; run "
            "prepare_ftmoe_protocol022_development.py first" % STREAM_DIR)
    return max(candidates)[1]


STREAM = STREAM_DIR / "dev_seed600_steps2380"      # replaced by registered_stream()
CHECKPOINT_SHA = "10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b"
ANCHOR_ROW_START = 272
ANCHOR_PER_EPISODE = 128
ANCHOR_EPISODES = 6
NEGATIVE_TOLERANCE = 1.0e-12

# Evaluation windows (registered here, before any result was seen), aligned to
# the realised development-stream phases: familiar_1 [0,350), unseen_1
# [350,1190), familiar_2 [1190,1540), unseen_recur [1540,2380).  The unseen
# block is split at the plan's own proportions (early 0-400 of 1200, mid
# 400-800, late 800-1200), scaled to its realised 840 intervals.
SPLITS = {
    "familiar_1": (0, 350),
    "unseen_early_train": (350, 630),
    "unseen_mid_validation": (630, 910),
    "unseen_late_test": (910, 1190),
    "familiar_2": (1190, 1540),
    "unseen_recurrence": (1540, 2380),
}
TRAIN_SPLIT = "unseen_early_train"
FUTURE_SPLITS = ("unseen_mid_validation", "unseen_late_test",
                 "unseen_recurrence")
GATE = {
    "pr_auc_gain_min": 0.03,
    "onset_ap_gain_min": 0.05,
    "familiar_f1_drop_max": 0.03,
    "anchor_f1_drop_max": 0.03,
}
# Class weights for the supervised terms.  Kept at the neutral setting the
# protocol registers for this diagnostic; the registered R1 detection positive
# weight is a separate (P20 P35) concern and is not re-tuned here.
BALANCE_WEIGHTS = {"detection": [1.0, 1.0], "resource": [1.0, 1.0, 1.0]}

SMOKE_SPLITS = {
    "familiar_1": (0, 200),
    "unseen_early_train": (200, 450),
    "unseen_mid_validation": (450, 700),
    "unseen_late_test": (700, 900),
    "familiar_2": (900, 1050),
    "unseen_recurrence": (1050, 1200),
}

VARIANTS = ("A", "C-current", "C-wide", "C-budget", "C-causal")
RESIDUAL_VARIANTS = tuple(v for v in VARIANTS if v != "A")


# --------------------------------------------------------------------------
# feature extensions
# --------------------------------------------------------------------------
def causal_features(ratios, capacities):
    """[T, H, F] causal capacity/temporal features (no phase/cascade id).

    Mirrors the P22-S4 probe families so the model-side and probe-side
    "causal capacity input" are the same object:
        cpu_headroom         1 - cpu_ratio
        ram_minus_cpu_lag4   ram_ratio - cpu_ratio shifted by the registered lag
        disk_minus_cpu_lag8  disk_ratio - cpu_ratio shifted by the registered lag
    """
    cpu, ram, disk = ratios[:, :, 0], ratios[:, :, 1], ratios[:, :, 2]
    out = np.zeros((ratios.shape[0], ratios.shape[1], 3), dtype=np.float32)
    out[:, :, 0] = 1.0 - cpu
    for j, (lag, src) in enumerate(((4, ram), (8, disk)), start=1):
        delta = np.zeros_like(src)
        delta[lag:] = src[lag:] - cpu[:-lag]
        out[:, :, j] = delta
    return out


# --------------------------------------------------------------------------
# residual banks
# --------------------------------------------------------------------------
class WideResidualExpert(nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width),
                                 nn.GELU(), nn.Linear(width, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(z)


class WideResidualBank(nn.Module):
    """Configurable fixed-topology residual bank (capacity control)."""

    def __init__(self, hidden=64, width=32, experts=4):
        super().__init__()
        self.hidden = hidden
        self.width = width
        self.expert_count = experts
        self.input_size = hidden
        self.router = nn.Linear(self.input_size, experts)
        self.experts = nn.ModuleList(
            [WideResidualExpert(self.input_size, width) for _ in range(experts)])

    def forward(self, z, extra=None):
        x = z if extra is None else torch.cat([z, extra], dim=-1)
        probabilities = torch.softmax(self.router(x), dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (probabilities.unsqueeze(-1) * outputs).sum(dim=-2), probabilities

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def average_precision(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    if y.size == 0 or y.sum() == 0:
        return None
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    distinct = np.nonzero(np.diff(score))[0]
    idx = np.r_[distinct, y.size - 1]
    return float(np.sum(np.diff(np.r_[0, tp[idx]]) * precision[idx]) / y.sum())


def detection_metrics(probability, labels):
    """Same metric set as the frozen P19/P20 runners, so numbers are comparable."""
    from run_ftmoe_protocol019 import extra_detection_metrics
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    anomaly = (labels > 0).astype(int)
    pred = probability >= 0.5
    tp = int((pred & (anomaly == 1)).sum())
    fp = int((pred & (anomaly == 0)).sum())
    fn = int((~pred & (anomaly == 1)).sum())
    tn = int((~pred & (anomaly == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out = {"n": int(labels.size), "positives": int(anomaly.sum()),
           "positive_rate": float(anomaly.mean()) if labels.size else None,
           "precision": precision, "recall": recall, "f1": f1,
           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "pr_auc": average_precision(anomaly, probability)}
    out.update(extra_detection_metrics(probability, labels))
    return out


def onset_metrics(probability, labels, horizon=1):
    """Onset AP: only rows whose host is NOT in the target fault at t."""
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0 or probability.size == 0:
        return {"n": 0, "positives": 0, "ap": None, "prevalence": None}
    current = labels > 0
    future = np.zeros_like(current)
    if horizon < current.size:
        future[:-horizon] = current[horizon:]
    target = future & (~current)
    keep = ~current
    y = target[keep].astype(int)
    if y.size == 0 or y.sum() == 0:
        return {"n": int(y.size), "positives": int(y.sum()), "ap": None,
                "prevalence": float(y.mean()) if y.size else None}
    return {"n": int(y.size), "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "ap": average_precision(y, probability[keep]),
            "roc_auc": _roc_auc(y, probability[keep])}


def _roc_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1)
    sorted_score = score[order]
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def resource_macro_f1(class_probability, labels):
    """Conditional-on-positive resource macro F1 (diagnosis quality)."""
    class_probability = np.asarray(class_probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels > 0
    if positive.sum() == 0:
        return {"macro_f1": None, "per_class": {}}
    truth = labels[positive] - 1
    pred = class_probability[positive].argmax(-1)
    per_class, scores = {}, []
    for klass in range(3):
        t = truth == klass
        p = pred == klass
        tp = int((t & p).sum())
        precision = tp / max(int(p.sum()), 1)
        recall = tp / max(int(t.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[klass + 1] = {"precision": precision, "recall": recall,
                                "f1": f1, "support": int(t.sum())}
        scores.append(f1)
    return {"macro_f1": float(np.mean(scores)), "per_class": per_class}


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=True) + "\n", encoding="utf8")
    temporary.replace(path)


def tolerance_labels(raw, scored):
    """±1 within-episode tolerance, applied in BOTH directions.

    The registered rule fills ``label[t] := label[t-1]`` when ``label[t]`` is 0
    and the previous interval was faulted, and ``label[t] := label[t+1]`` when
    ``label[t]`` is 0 and the next interval is faulted.  Both passes therefore
    run over the *same* interior rows; the first and last scored rows have only
    one neighbour each.

    Note on the frozen implementation: ``run_ftmoe_protocol020.tolerance_labels``
    applies the forward pass as ``labels[1:][fill] = previous[fill]``, where
    ``labels[1:]`` is a temporary view, so on an integer array that assignment
    never reaches ``labels`` and the forward fill is silently lost.  This copy
    assigns back through ``labels[1:] = block`` and also runs the backward pass
    with matching shapes.  Both differences are recorded in the problem log
    rather than left implicit.
    """
    raw = np.asarray(raw)
    labels = raw[:scored].copy()
    if labels.shape[0] != scored:
        raise ValueError("raw has %d rows but scored=%d"
                         % (labels.shape[0], scored))
    if scored >= 2:
        # backward fill: row t takes the previous row's state
        previous = raw[:scored - 1]
        block = labels[1:]
        fill = (block == 0) & (previous > 0)
        block[fill] = previous[fill]
        labels[1:] = block
        # forward fill: row t takes the next row's state
        following = raw[1:scored]
        block = labels[:-1]
        fill = (block == 0) & (following > 0)
        block[fill] = following[fill]
        labels[:-1] = block
    return labels


def load_anchor_pool(time_scale):
    """P20 adaptation train episodes, tail rows: the registered anchor source."""
    from run_ftmoe_protocol020 import ANCHOR_ROW_START as ROW_START
    data_dir = ROOT / "artifacts/ftmoe_online/protocol_020/adaptation_data/v1"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf8"))
    normalization = json.loads((data_dir / "normalization.json").read_text(encoding="utf8"))
    graph_scale = np.asarray(normalization["graph_scale"], np.float32)
    parts = {k: [] for k in ("x", "graph_x", "schedule", "labels", "ids",
                             "before", "caps", "extra")}
    for index in manifest["train_indices"][:ANCHOR_EPISODES]:
        with np.load(data_dir / "episodes" / ("%02d.npz" % index)) as data:
            time_series = data["time"].astype(np.float32)
            demands = data["demands"].astype(np.float32)
            schedules = data["schedules"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            ids = data["creation_ids"].astype(np.int64)
            before = data["before_placement"].astype(np.int64)
            caps = data["capacities"].astype(np.float64)
            host_features = np.asarray(data["time"], dtype=np.float64)
        rows = np.arange(ROW_START, ROW_START + ANCHOR_PER_EPISODE)
        idx = np.maximum(rows[:, None] - 11 + np.arange(12)[None], 0)
        parts["x"].append(torch.from_numpy(
            (time_series[idx] / time_scale[None]).transpose(0, 2, 1, 3).copy()))
        parts["graph_x"].append(torch.from_numpy(
            (demands[idx] / graph_scale[None]).transpose(0, 2, 1, 3).copy()))
        parts["schedule"].append(torch.from_numpy(schedules[idx].copy()))
        parts["labels"].append(torch.from_numpy(labels[rows].copy()))
        parts["ids"].append(torch.from_numpy(ids[idx].copy()))
        parts["before"].append(torch.from_numpy(before[idx].copy()))
        exp = caps[rows] / graph_scale[[0, 1, 4]][None]
        parts["caps"].append(torch.from_numpy(
            np.repeat(exp[:, None, :], 12, axis=1).copy()))
        anchor_caps = np.asarray(caps, dtype=np.float64)
        if anchor_caps.ndim == 3:                # per-interval capacities
            anchor_caps = anchor_caps[:, 0, :]
        elif anchor_caps.ndim == 2 and anchor_caps.shape[0] != host_features.shape[0]:
            anchor_caps = np.repeat(anchor_caps[None, :, :],
                                    host_features.shape[0], axis=0)
        anchor_ratios = np.stack(
            [host_features[:, :, 0] / np.maximum(anchor_caps[:, 0][:, None], 1e-9),
             host_features[:, :, 1] / np.maximum(anchor_caps[:, 1][:, None], 1e-9),
             host_features[:, :, 4] / np.maximum(anchor_caps[:, 2][:, None], 1e-9)],
            axis=-1).astype(np.float32)
        parts["extra"].append(torch.from_numpy(
            causal_features(anchor_ratios, anchor_caps)[rows]))
    return {k: torch.cat(v, dim=0) for k, v in parts.items()}


def graph_context(ids, before, caps):
    return {"creation_ids": ids, "before_placement": before, "capacities": caps}


def supervised_terms(detection, classification, target, balance, weights):
    """R1's loss decomposition, returned term by term for gradient health.

    Logits arrive as ``[..., 2]`` / ``[..., 3]`` over host rows while ``target``
    is a flat label vector, so every tensor is flattened to the host axis first;
    indexing a batched tensor with a flat mask is what R1's own supervised loss
    explicitly guards against.
    """
    detection = detection.reshape(-1, 2)
    classification = classification.reshape(-1, 3)
    target = target.reshape(-1)
    anomaly = (target > 0).long()
    det_weight = torch.as_tensor(weights.get("detection", [1.0, 1.0]),
                                 dtype=detection.dtype)
    res_weight = torch.as_tensor(weights.get("resource", [1.0, 1.0, 1.0]),
                                 dtype=classification.dtype)
    det_ce = F.cross_entropy(detection.reshape(-1, 2), anomaly.reshape(-1),
                             weight=det_weight)
    positive = target > 0
    cls_ce = (F.cross_entropy(classification[positive], target[positive] - 1,
                              weight=res_weight)
              if positive.any() else detection.new_zeros(()))
    ranking = detection.new_zeros(())
    if positive.any() and (~positive).any():
        anomaly_probability = detection.softmax(-1)[..., 1]
        class_probability = classification.softmax(-1)
        true_index = (target.clamp_min(1) - 1).unsqueeze(-1)
        true_probability = class_probability.gather(-1, true_index).squeeze(-1)
        positive_score = anomaly_probability[positive] * true_probability[positive]
        negative_score = (anomaly_probability[~positive]
                          * class_probability[~positive].max(-1).values)
        ranking = F.softplus(0.15 + negative_score.unsqueeze(0)
                             - positive_score.unsqueeze(1)).mean()
    return {"detection": det_ce, "classification": cls_ce, "ranking": ranking,
            "total": 0.7 * det_ce + 0.3 * cls_ce + 0.5 * ranking}


def run(args):
    torch.set_num_threads(3)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import psutil
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    from run_ftmoe_protocol019 import resources
    resources()

    global SPLITS
    if args.splits == "smoke":
        SPLITS = dict(SMOKE_SPLITS)
    stream_dir = Path(args.stream) if args.stream else registered_stream()
    if not stream_dir.is_absolute():
        stream_dir = (ROOT / stream_dir).resolve()

    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("protocol") != "022":
        raise ValueError("Development stream is not a Protocol 022 artifact")
    if manifest["stream_sha256"] != sha(stream_dir / "stream.npz"):
        raise AssertionError("Development stream hash mismatch")
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise AssertionError("Start checkpoint hash mismatch")

    torch.manual_seed((args.seed * 7919 + manifest["seed"]) % (2 ** 32))
    np.random.seed((args.seed * 7919 + manifest["seed"]) % (2 ** 32))
    random.seed((args.seed * 7919 + manifest["seed"]) % (2 ** 32))

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "v4" or checkpoint.get("seed") != 1:
        raise AssertionError("Checkpoint identity mismatch")

    from run_ftmoe_protocol020 import ReplayV3, load_v2_time_scale_p20
    from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE

    steps = int(manifest["steps"])
    with np.load(stream_dir / "stream.npz") as data:
        arrays = {k: data[k] for k in data.files}
    caps_per_interval = np.asarray(arrays["capacities"], np.float64)
    arrays["capacities_per_interval"] = caps_per_interval
    arrays["capacities"] = caps_per_interval[0]
    v2, _fallback = load_v2_time_scale_p20(checkpoint["normalization"])
    graph_scale = np.asarray(checkpoint["normalization"]["graph_scale"], np.float64)
    replay = ReplayV3(arrays, v2, graph_scale, steps)

    ratios = np.stack([arrays["host_features"][:, :, 0] / caps_per_interval[:, :, 0],
                       arrays["host_features"][:, :, 1] / caps_per_interval[:, :, 1],
                       arrays["host_features"][:, :, 4] / caps_per_interval[:, :, 2]],
                      axis=-1).astype(np.float32)
    causal = torch.from_numpy(causal_features(ratios, caps_per_interval))
    labels_tol = tolerance_labels(arrays["raw_labels"], steps)

    anchor = load_anchor_pool(np.asarray(v2, np.float32))

    out = Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANTS) if args.all else list(args.variants)
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError("Unknown variant %r" % variant)

    # A budget sweep answers the one question the plan makes decisive for D
    # (plan §11 case C): is the capacity gain explained by more training steps,
    # or does it plateau?  It re-uses the identical train/evaluate split and the
    # identical initialisation, so the only variable is the optimisation budget.
    if args.sweep:
        sweep = []
        for budget in [int(x) for x in args.sweep]:
            args.train_steps = budget
            results = _run_variants(args, ["A", args.sweep_variant], stream_dir,
                                    manifest, replay, causal, labels_tol, anchor,
                                    steps, checkpoint, FrozenResidualFTMoE)
            row = results[args.sweep_variant]
            sweep.append({
                "train_steps": budget,
                "updates": row["optimization_updates"],
                "late_pr_auc": row["post_training"]["unseen_late_test"]["detection"]["pr_auc"],
                "recurrence_pr_auc": row["post_training"]["unseen_recurrence"]["detection"]["pr_auc"],
                "mid_pr_auc": row["post_training"]["unseen_mid_validation"]["detection"]["pr_auc"],
                "late_onset_ap": row["post_training"]["unseen_late_test"]["onset"]["ap"],
                "familiar_2_f1": row["post_training"]["familiar_2"]["detection"]["f1"],
            })
            print(json.dumps({"sweep": args.sweep_variant, "train_steps": budget,
                              "late_pr_auc": row["post_training"]["unseen_late_test"]["detection"]["pr_auc"],
                              "recurrence_pr_auc": row["post_training"]["unseen_recurrence"]["detection"]["pr_auc"]},
                             ensure_ascii=False), flush=True)
        write_json(out / "budget_sweep.json", {
            "protocol": "022", "step": "P22-S5-budget-sweep",
            "variant": args.sweep_variant, "splits_mode": args.splits,
            "sweep": sweep,
            "interpretation": ("if the metric keeps improving with budget, the "
                               "plateau condition of plan 11 case C is NOT met "
                               "and a dynamic expert has no scientific basis"),
        })
        return {"sweep": sweep}

    results = _run_variants(args, variants, stream_dir, manifest, replay, causal,
                            labels_tol, anchor, steps, checkpoint,
                            FrozenResidualFTMoE)
    return results


def _run_variants(args, variants, stream_dir, manifest, replay, causal,
                  labels_tol, anchor, steps, checkpoint, model_class):
    """Run the requested variants on the shared split (extracted for the sweep)."""
    FrozenResidualFTMoE = model_class
    out = Path(args.output)

    def build_model(variant):
        model = FrozenResidualFTMoE(checkpoint, "A" if variant == "A" else "C",
                                    args.seed)
        model.eval()
        if variant == "A":
            model.set_deployment("live", 0.0)
            return model, None
        if variant == "C-current":
            model.set_deployment("learner", 1.0)
            return model, model.learner
        if variant == "C-wide":
            bank = WideResidualBank(64, 64, 8)
        elif variant == "C-causal":
            bank = WideResidualBank(64 + 3, 32, 4)
        elif variant == "C-budget":
            bank = WideResidualBank(64, 32, 4)
        else:
            raise ValueError(variant)
        model.learner = bank
        model.set_trainability()
        model.set_deployment("learner", 1.0)
        return model, bank

    def forward_window(model, index, extra=None):
        host, schedule, graph, ids, before, caps_w = replay.window_v3(index)
        context = graph_context(ids.unsqueeze(0), before.unsqueeze(0),
                                caps_w.unsqueeze(0))
        x = (host.unsqueeze(0), schedule.unsqueeze(0), graph.unsqueeze(0))
        return x, context

    results = {}
    diagnostics = []
    for variant in variants:
        began = time.perf_counter()
        model, bank = build_model(variant)
        frozen_before = model.frozen_hash()
        trainable = ([p for p in model.parameters() if p.requires_grad]
                     if variant != "A" else [])
        parameter_count = sum(p.numel() for p in trainable)
        budget = args.train_steps * (2 if variant == "C-budget" else 1)
        optimizer = (torch.optim.AdamW(trainable, lr=args.learning_rate,
                                       weight_decay=1e-4)
                     if trainable else None)

        # ---- evaluation at the pre-training boundary ---------------------
        def evaluate(tag):
            model.eval()
            metrics = {}
            for name, (start, end) in SPLITS.items():
                index = list(range(start, end))
                index_array = np.asarray(index)
                batch_host, batch_sched, batch_graph = [], [], []
                context_lists = {"creation_ids": [], "before_placement": [],
                                 "capacities": []}
                for i in index:
                    h, s, g, cid, bf, cp = replay.window_v3(int(i))
                    batch_host.append(h)
                    batch_sched.append(s)
                    batch_graph.append(g)
                    context_lists["creation_ids"].append(cid)
                    context_lists["before_placement"].append(bf)
                    context_lists["capacities"].append(cp)
                x = torch.stack(batch_host)
                s = torch.stack(batch_sched)
                g = torch.stack(batch_graph)
                context = {k: torch.stack(v) for k, v in context_lists.items()}
                extra = (causal[index_array] if variant == "C-causal" else None)
                with torch.no_grad():
                    if variant == "A":
                        raw = model.predict_deployment(x, s, g, graph_context=context)
                        probability = torch.softmax(
                            raw["base_final_detection_logits"], -1)[..., 1]
                        class_probability = torch.softmax(
                            raw["base_final_class_logits"], -1)
                        base_probability = probability
                        base_class = class_probability
                    else:
                        out_dict = (model(x, s, g, graph_context=context)
                                    if extra is None else
                                    _forward_with_extra(model, x, s, g, context,
                                                        extra))
                        probability = torch.softmax(
                            out_dict["detection_logits"], -1)[..., 1]
                        class_probability = torch.softmax(out_dict["class_logits"], -1)
                        base_probability = torch.softmax(
                            out_dict["base_final_detection_logits"], -1)[..., 1]
                        base_class = torch.softmax(
                            out_dict["base_final_class_logits"], -1)
                labs = labels_tol[start:end, :].reshape(-1)
                prob = probability.reshape(-1).detach().numpy().astype(np.float64)
                cls = class_probability.reshape(-1, 3).detach().numpy().astype(np.float64)
                base_prob = base_probability.reshape(-1).detach().numpy().astype(np.float64)
                base_cls = base_class.reshape(-1, 3).detach().numpy().astype(np.float64)
                detection = detection_metrics(prob, labs)
                detection_base = detection_metrics(base_prob, labs)
                metrics[name] = {
                    "detection": detection,
                    "detection_base": detection_base,
                    "diagnosis": resource_macro_f1(cls, labs),
                    "diagnosis_base": resource_macro_f1(base_cls, labs),
                    "onset": onset_metrics(prob, labs),
                    "onset_base": onset_metrics(base_prob, labs),
                }
            return {"tag": tag, "metrics": metrics}

        pre = evaluate("pre_training")

        # ---- training on the early mature segment ------------------------
        train_start, train_end = SPLITS[TRAIN_SPLIT]
        train_indices = list(range(train_start, train_end))
        anchor_n = anchor["x"].shape[0]
        rng = np.random.default_rng(args.seed * 7919 + 22)
        updates = 0
        if bank is not None:
            for step in range(budget):
                index = train_indices[step % len(train_indices)]
                host, schedule, graph, ids, before, caps_w = replay.window_v3(index)
                x = host.unsqueeze(0)
                s = schedule.unsqueeze(0)
                g = graph.unsqueeze(0)
                context = graph_context(ids.unsqueeze(0), before.unsqueeze(0),
                                        caps_w.unsqueeze(0))
                extra = (causal[index][None, :, :] if variant == "C-causal"
                         else None)
                target = torch.from_numpy(
                    labels_tol[index][None, :].astype(np.int64)).reshape(-1)
                out_dict = _forward_with_extra(model, x, s, g, context, extra)
                terms = supervised_terms(out_dict["detection_logits"],
                                         out_dict["class_logits"], target,
                                         BALANCE_WEIGHTS, BALANCE_WEIGHTS)
                # anchor protection term (same anchor source as the registered
                # R1 config); it keeps the residual from drifting off the
                # familiar task, and it is reported separately.
                anchor_index = int(rng.integers(0, anchor_n, 1)[0])
                anchor_extra = (anchor["extra"][anchor_index:anchor_index + 1]
                                if variant == "C-causal" else None)
                anchor_out = _forward_with_extra(
                    model, anchor["x"][anchor_index:anchor_index + 1],
                    anchor["schedule"][anchor_index:anchor_index + 1],
                    anchor["graph_x"][anchor_index:anchor_index + 1],
                    graph_context(anchor["ids"][anchor_index:anchor_index + 1],
                                  anchor["before"][anchor_index:anchor_index + 1],
                                  anchor["caps"][anchor_index:anchor_index + 1]),
                    anchor_extra)
                anchor_target = anchor["labels"][anchor_index].reshape(-1)
                anchor_terms = supervised_terms(
                    anchor_out["detection_logits"], anchor_out["class_logits"],
                    anchor_target, BALANCE_WEIGHTS, BALANCE_WEIGHTS)
                # distillation keeps the corrected logits near the frozen base
                distill = F.mse_loss(out_dict["detection_logits"],
                                     out_dict["base_final_detection_logits"].detach())
                total = (terms["total"] + 0.25 * anchor_terms["total"]
                         + 0.10 * distill)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                norms = {name: float(p.grad.norm()) for name, p in
                         model.named_parameters()
                         if p.grad is not None and p.requires_grad}
                base_leak = [name for name, p in model.named_parameters()
                             if p.grad is not None and not p.requires_grad]
                if base_leak:
                    raise AssertionError("frozen base received gradients: %s"
                                         % base_leak[:3])
                grad_norm = float(np.sqrt(sum(v * v for v in norms.values())))
                if step % args.diagnostic_interval == 0 or step == budget - 1:
                    with torch.no_grad():
                        response = (out_dict["detection_logits"]
                                    - out_dict["base_final_detection_logits"])
                        diagnostics.append({
                            "variant": variant, "step": int(step),
                            "index": int(index),
                            "grad_norm_detection": float(terms["detection"]),
                            "grad_norm_classification": float(terms["classification"]),
                            "grad_norm_ranking": float(terms["ranking"]),
                            "grad_norm_distill": float(distill),
                            "grad_norm_anchor": float(anchor_terms["total"]),
                            "grad_norm_total": grad_norm,
                            "loss_detection": float(terms["detection"]),
                            "loss_classification": float(terms["classification"]),
                            "loss_ranking": float(terms["ranking"]),
                            "loss_total": float(total),
                            "residual_logit_abs_mean": float(response.abs().mean()),
                            "residual_logit_abs_p90": float(
                                torch.quantile(response.abs().flatten(), 0.9)),
                            "base_logit_abs_mean": float(
                                out_dict["base_final_detection_logits"].abs().mean()),
                            "base_logit_abs_p90": float(torch.quantile(
                                out_dict["base_final_detection_logits"].abs().flatten(), 0.9)),
                            "correction_strength": float(
                                (response.abs().mean()
                                 / (out_dict["base_final_detection_logits"].abs().mean()
                                    + 1e-9))),
                            "zero_grad_fraction": float(np.mean(
                                [1.0 if v == 0.0 else 0.0 for v in norms.values()])),
                        })
                optimizer.step()
                updates += 1

        post = evaluate("post_training")
        frozen_after = model.frozen_hash()
        if frozen_after != frozen_before:
            raise AssertionError("Frozen base parameters changed during training")
        results[variant] = {
            "variant": variant,
            "trainable_parameters": int(parameter_count),
            "optimization_updates": int(updates),
            "learning_rate": args.learning_rate,
            "pre_training": pre["metrics"],
            "post_training": post["metrics"],
            "frozen_base_hash_unchanged": frozen_after == frozen_before,
            "elapsed_seconds": time.perf_counter() - began,
            "learner_state_sha256": (None if bank is None else
                                     model.learner_state_hash()),
        }
        print(json.dumps({"variant": variant,
                          "trainable_parameters": int(parameter_count),
                          "updates": int(updates),
                          "late_pr_auc": post["metrics"]["unseen_late_test"]["detection"]["pr_auc"],
                          "late_onset_ap": post["metrics"]["unseen_late_test"]["onset"]["ap"],
                          "elapsed_seconds": round(results[variant]["elapsed_seconds"], 1)},
                         ensure_ascii=False), flush=True)
        del model, optimizer
    write_json(out / "fixed_capacity_diagnostic.json", results)
    if diagnostics:
        with (out / "gradient_diagnostic.jsonl").open("w", encoding="utf8") as stream:
            for row in diagnostics:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "run_provenance.json", {
        "protocol": "022", "step": "P22-S5", "variant": variants,
        "command": [sys.executable] + sys.argv,
        "stream": str(stream_dir.relative_to(ROOT)).replace("\\", "/"),
        "stream_sha256": manifest["stream_sha256"],
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "splits": SPLITS, "splits_mode": args.splits,
        "train_split": TRAIN_SPLIT,
        "gate": GATE, "seed": args.seed,
        "learning_rate": args.learning_rate, "train_steps": args.train_steps,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                   stdout=subprocess.PIPE).stdout.decode().strip(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "started_at": args.started_at,
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "exit_code_source": "captured by the parent runner, not this process",
    })
    return results


def _forward_with_extra(model, x, s, g, context, extra=None):
    """Forward the model with an optionally extended residual input.

    Only the C-causal bank declares a wider input; every other variant takes the
    plain 64-dimensional frozen feature, so the extension is applied only when
    the bank actually expects it.
    """
    base_output, z = model._base_forward(x, s, g, graph_context=context,
                                         record=False)
    bank = model.learner
    if extra is not None and getattr(bank, "input_size", z.shape[-1]) == z.shape[-1]:
        extra = None
    if extra is None:
        correction, probabilities = bank(z)
    else:
        correction, probabilities = bank(z, extra)
    output = dict(base_output)
    output.update({
        "detection_logits": base_output["detection_logits"] + correction[..., :2],
        "class_logits": base_output["class_logits"] + correction[..., 2:],
        "base_final_detection_logits": base_output["detection_logits"],
        "base_final_class_logits": base_output["class_logits"],
        "correction_logits": correction,
        "correction_router_probabilities": probabilities,
        "router_probabilities": probabilities,
    })
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="*", default=["A", "C-current"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--diagnostic-interval", type=int, default=10)
    p.add_argument("--output", type=Path,
                   default=P22 / "fixed_c" / "s5_capacity_diagnostic")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--started-at", default=None)
    p.add_argument("--stream", type=Path, default=None,
                   help="development stream directory (default: the collected "
                        "registered stream)")
    p.add_argument("--splits", choices=("registered", "smoke"),
                   default="registered",
                   help="registered evaluation windows, or the smoke preset that "
                        "fits inside a 1200-step pilot stream")
    p.add_argument("--sweep", nargs="*", default=None,
                   help="optimisation-budget sweep, e.g. --sweep 100 200 400 800; "
                        "answers plan §11 case C (is the gain just more training?)")
    p.add_argument("--sweep-variant", default="C-current")
    return p


def main():
    args = parser().parse_args()
    args.started_at = args.started_at or _dt.datetime.now().isoformat(timespec="seconds")
    try:
        run(args)
    except Exception:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "failure.json", {
            "error": traceback.format_exc(),
            "variants": list(VARIANTS) if args.all else list(args.variants)})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
