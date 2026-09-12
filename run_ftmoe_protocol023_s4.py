"""Protocol 023 round 2A — S4 strict-prequential fixed-C run (directive §9-§17).

D is NOT implemented here.  This runner measures the two arms the directive
allows in this round:

    A  frozen baseline            (the registered v4 base, no correction)
    C  fixed 4-expert residual    (Protocol 020 R1's ``FrozenResidualFTMoE``
                                   learner bank, 64 -> 32 -> 5, 4 experts)

on the *calibrated* development stream, in the registered prequential order, so
that the three quantities the round exists to measure become measurable:

    current-regime adaptation      (does C beat A on the regime it is in?)
    previous-regime forgetting     (the fixed probe matrix, §13)
    recurrence relearning          (§15)

Strict prequential order (directive §11), implemented literally
---------------------------------------------------------------
For every scored interval ``t``:

    1. predict with the *current* model on the window ending at ``t``
    2. write the prediction record, its ``model_version`` and the learner hash
    3. read the label that has just matured (``raw_label[t-1]``; the label of
       ``t`` does not exist yet) and the guard row
    4. settle every pending prediction whose label is now known (``t-2``)
    5. replay/anchor selection over the matured buffer only
    6. train only if ``t`` is an update opportunity
    7. ``model_version += 1``

The invariant is checked, not asserted in prose: a settled record's label index
must be strictly older than the label index read at the same step, the training
buffer may only contain settled indices, and a prediction is written before the
label of its own interval is read.

Online budget (directive §10)
-----------------------------
Frozen before the run in ``amendments/online_budget.json``: update every 4
scored intervals, batch 32, one gradient step per opportunity.  A1 (420 scored
intervals) is therefore ~105 updates and a 360-interval recurrence ~90.  This
runner refuses to start on a budget other than the frozen file's.

Usage
-----
    python run_ftmoe_protocol023_s4.py --make-probes
    python run_ftmoe_protocol023_s4.py --arm A --benchmark
    python run_ftmoe_protocol023_s4.py --arm A
    python run_ftmoe_protocol023_s4.py --arm C
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "3")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

P23 = ROOT / "artifacts/ftmoe_online/protocol_023"
ROUND2A = P23 / "round2a"
STREAMS = P23 / "development_streams"
DEV_TAG = "dev_seed700_steps2880_calibrated"
DEV_STREAM = STREAMS / DEV_TAG
CHECKPOINT = (ROOT / "artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1"
                     "/best.pt")
BUDGET_FILE = ROUND2A / "amendments/online_budget.json"
PROBE_DIR = ROUND2A / "probe_slices"
OUT_DIR = ROUND2A / "fixed_c_prequential"
CKPT_DIR = ROUND2A / "checkpoints"

#: Registered dev timeline of the calibrated stream (same shape as round 1).
PHASES = (("F0_baseline", 0, 300, None),
          ("A1_compute", 300, 720, "compute_first"),
          ("B1_memory", 720, 1140, "memory_first"),
          ("C1_io", 1140, 1560, "io_first"),
          ("F1_baseline", 1560, 1800, None),
          ("A2_recur", 1800, 2160, "compute_first"),
          ("C2_recur", 2160, 2520, "io_first"),
          ("B2_recur", 2520, 2880, "memory_first"))
#: The seven phase boundaries the directive (§12) requires a checkpoint at.
CHECKPOINT_PHASES = ("A1_compute", "B1_memory", "C1_io", "F1_baseline",
                     "A2_recur", "C2_recur", "B2_recur")
CHECKPOINT_NAMES = {"A1_compute": "C_after_A1.pt", "B1_memory": "C_after_B1.pt",
                    "C1_io": "C_after_C1.pt", "F1_baseline": "C_after_F1.pt",
                    "A2_recur": "C_after_A2.pt", "C2_recur": "C_after_C2.pt",
                    "B2_recur": "C_after_B2.pt"}
#: Recurrence blocks and the first-exposure block they repeat (directive §15).
RECURRENCE = {"A2_recur": "A1_compute", "C2_recur": "C1_io", "B2_recur": "B1_memory"}
PROBE_REGIMES = ("compute_first", "memory_first", "io_first")
#: Probe slice geometry, registered before any model run (directive §13).
PROBE_START = 1000          # inside the regime window of the single stream
PROBE_LENGTH = 120
ANCHOR_EPISODES = 12
#: The Protocol 020 R1 anchor geometry, imported from its own module rather than
#: copied: ``ANCHOR_ROW_START = 272``, ``ANCHOR_PER_EPISODE = 128`` over 400-row
#: episodes.
from run_ftmoe_protocol020 import (ANCHOR_PER_EPISODE,  # noqa: E402
                                   ANCHOR_ROW_START)
BALANCE_WEIGHTS = {"detection": [1.0, 1.0], "resource": [1.0, 1.0, 1.0]}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=True) + "\n", encoding="utf8")
    temporary.replace(path)


def configure():
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


# --------------------------------------------------------------------------
# metrics (the P22 instruments, re-exported so the numbers stay comparable)
# --------------------------------------------------------------------------
def average_precision(y, score):
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if y.size == 0 or y.max() == y.min():
        return None
    order = np.argsort(-score, kind="stable")
    y = y[order]
    positives = y.sum()
    if positives <= 0:
        return None
    precision = np.cumsum(y) / np.arange(1, y.size + 1)
    return float((precision * y).sum() / positives)


def roc_auc(y, score):
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1)
    return float((ranks[y > 0].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def detection_metrics(probability, labels):
    binary = (np.asarray(labels) > 0).astype(np.int64)
    return {"pr_auc": average_precision(binary, probability),
            "roc_auc": roc_auc(binary, probability),
            "positives": int(binary.sum()), "rows": int(binary.size),
            "prevalence": float(binary.mean()) if binary.size else None}


def resource_macro_f1(class_probability, labels):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(class_probability).argmax(-1) + 1
    scores = []
    for klass in (1, 2, 3):
        truth = labels == klass
        guess = predicted == klass
        tp = int((truth & guess).sum())
        fp = int((~truth & guess).sum())
        fn = int((truth & ~guess).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        scores.append({"class": klass, "precision": precision,
                       "recall": recall, "f1": f1,
                       "support": int(truth.sum())})
    macro = float(np.mean([item["f1"] for item in scores]))
    return {"macro_f1": macro, "per_class": scores}


def onset_metrics(probability, labels, horizon=1):
    """h-step onset target: 1 iff not faulted at t and faulted in t+1..t+h."""
    labels = np.asarray(labels, dtype=np.int64)
    steps = labels.shape[0]
    in_class = (labels > 0).astype(np.float64)
    future = np.zeros_like(in_class)
    for h in range(1, horizon + 1):
        if h < steps:
            shifted = np.zeros_like(in_class)
            shifted[:steps - h] = in_class[h:]
            future = np.maximum(future, shifted)
    valid = (in_class == 0)
    if horizon < steps:
        valid[:steps - horizon] &= True
    return {"ap": average_precision(future[valid], np.asarray(probability)[valid]),
            "positives": int(future[valid].sum()),
            "rows": int(valid.sum())}


# --------------------------------------------------------------------------
# stream / probe plumbing
# --------------------------------------------------------------------------
def load_arrays(stream_dir):
    with np.load(stream_dir / "stream.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return arrays


def build_replay(stream_dir):
    manifest = json.loads((stream_dir / "manifest.json").read_text(encoding="utf8"))
    if sha(stream_dir / "stream.npz") != manifest["stream_sha256"]:
        raise AssertionError("stream hash mismatch: %s" % stream_dir)
    arrays = load_arrays(stream_dir)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "v4" or checkpoint.get("seed") != 1:
        raise AssertionError("start checkpoint identity mismatch")
    from run_ftmoe_protocol020 import ReplayV3, load_v2_time_scale_p20
    v2, _fallback = load_v2_time_scale_p20(checkpoint["normalization"])
    graph_scale = np.asarray(checkpoint["normalization"]["graph_scale"],
                             np.float64)
    steps = int(manifest["steps"])
    caps_per_interval = np.asarray(arrays["capacities"], np.float64)
    replay_arrays = dict(arrays)
    replay_arrays["capacities_per_interval"] = caps_per_interval
    replay_arrays["capacities"] = caps_per_interval[0]
    return {"manifest": manifest, "arrays": arrays, "checkpoint": checkpoint,
            "replay": ReplayV3(replay_arrays, v2, graph_scale, steps),
            "steps": steps, "graph_scale": graph_scale}


def graph_context(ids, before, caps):
    return {"creation_ids": ids, "before_placement": before, "capacities": caps}


def supervised_terms(detection, classification, target):
    """The registered R1 online loss decomposition.

    Same arithmetic as ``run_ftmoe_protocol022_s5.supervised_terms`` (which is
    the R1 online objective): 0.7 * detection CE + 0.3 * resource CE
    + 0.5 * ranking, with the classification term evaluated on the positive
    rows only.  It is inlined rather than imported so this runner cannot drift
    when the P22 diagnostic module changes.
    """
    detection = detection.reshape(-1, 2)
    classification = classification.reshape(-1, 3)
    target = target.reshape(-1)
    anomaly = (target > 0).long()
    det_ce = F.cross_entropy(detection, anomaly)
    positive = target > 0
    cls_ce = (F.cross_entropy(classification[positive], target[positive] - 1)
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


def window_batch(replay, indices):
    hosts, schedules, graphs = [], [], []
    ids, before, caps = [], [], []
    for index in indices:
        h, s, g, cid, bf, cp = replay.window_v3(int(index))
        hosts.append(h)
        schedules.append(s)
        graphs.append(g)
        ids.append(cid)
        before.append(bf)
        caps.append(cp)
    return (torch.stack(hosts), torch.stack(schedules), torch.stack(graphs),
            graph_context(torch.stack(ids), torch.stack(before),
                          torch.stack(caps)))


def make_probes():
    """Extract the fixed held-out probe slices (directive §13).

    Each probe is a contiguous slice of a *calibrated single-regime* stream.
    The single-regime streams are separate collection runs from the development
    stream, so a probe cannot overlap the training stream by construction; the
    slice geometry is registered here, before any model run.
    """
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    registry = {"kind": "round2a_fixed_probe_matrix", "geometry": {
        "start": PROBE_START, "length": PROBE_LENGTH,
        "unit": "scored intervals inside the single-regime stream's regime window"},
        "registered_before_model_runs": True, "probes": {}}
    for regime in PROBE_REGIMES:
        tag = "single_%s_seed700_steps1200_calibrated" % regime
        path = STREAMS / tag
        if not path.is_dir():
            raise SystemExit("missing calibrated single-regime stream %s" % path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
        arrays = load_arrays(path)
        steps = int(manifest["steps"])
        start, end = PROBE_START, min(PROBE_START + PROBE_LENGTH, steps - 1)
        payload = {
            "host_features": arrays["host_features"][start:end + 1],
            "demands": arrays["demands"][start:end + 1],
            "schedules": arrays["schedules"][start:end + 1],
            "capacities": arrays["capacities"][start:end + 1],
            "creation_ids": arrays["creation_ids"][start:end + 1],
            "before_placement": arrays["before_placement"][start:end + 1],
            "raw_labels": arrays["raw_labels"][start:end + 1],
        }
        out = PROBE_DIR / ("probe_%s.npz" % regime)
        np.savez_compressed(out, **payload)
        registry["probes"][regime] = {
            "file": out.name, "source_stream": tag,
            "source_stream_sha256": manifest["stream_sha256"],
            "source_intervals": [start, end],
            "scored_intervals": int(end - start),
            "rows": int(payload["raw_labels"].shape[0]),
            "class_counts": np.bincount(
                payload["raw_labels"][:end - start].ravel(),
                minlength=4).tolist(),
            "never_used_for_training": True,
            "never_in_replay": True,
            "never_for_threshold_tuning": True,
        }
    write_json(PROBE_DIR / "probe_registry.json", registry)
    return registry


def load_anchor_pool(time_scale):
    """The registered P20 anchor source, built exactly as P22 S5 builds it.

    Every anchor sample is a real 12-interval causal window (not a single row),
    normalised with the same time scale the replay uses, and its graph context
    carries the 12-row capacity block the graph encoder expects.
    """
    data_dir = ROOT / "artifacts/ftmoe_online/protocol_020/adaptation_data/v1"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf8"))
    normalization = json.loads((data_dir / "normalization.json")
                               .read_text(encoding="utf8"))
    graph_scale = np.asarray(normalization["graph_scale"], np.float64)
    time_scale = np.asarray(time_scale, np.float64)
    parts = {k: [] for k in ("x", "schedule", "graph_x", "labels", "ids",
                             "before", "caps")}
    for index in manifest["train_indices"][:ANCHOR_EPISODES]:
        with np.load(data_dir / "episodes" / ("%02d.npz" % index)) as data:
            time_series = data["time"].astype(np.float64)
            demands = data["demands"].astype(np.float64)
            schedules = data["schedules"].astype(np.float64)
            labels = data["labels"].astype(np.int64)
            ids = data["creation_ids"].astype(np.int64)
            before = data["before_placement"].astype(np.int64)
            caps = data["capacities"].astype(np.float64)
        rows = np.arange(ANCHOR_ROW_START, ANCHOR_ROW_START + ANCHOR_PER_EPISODE)
        windows = np.maximum(rows[:, None] - 11 + np.arange(12)[None], 0)
        parts["x"].append(torch.from_numpy(
            (time_series[windows] / time_scale[None]).transpose(0, 2, 1, 3)
            .copy()).float())
        parts["graph_x"].append(torch.from_numpy(
            (demands[windows] / graph_scale[None]).transpose(0, 2, 1, 3)
            .copy()).float())
        parts["schedule"].append(torch.from_numpy(schedules[windows].copy())
                                 .float())
        parts["labels"].append(torch.from_numpy(labels[rows].copy()))
        parts["ids"].append(torch.from_numpy(ids[windows].copy()))
        parts["before"].append(torch.from_numpy(before[windows].copy()))
        exp = caps[rows] / graph_scale[[0, 1, 4]][None]
        parts["caps"].append(torch.from_numpy(
            np.repeat(exp[:, None, :], 12, axis=1).copy()))
    return {k: torch.cat(v, dim=0) for k, v in parts.items()}


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------
class PrequentialS4:
    """Strict-prequential A/C session with a frozen online budget."""

    def __init__(self, arm, seed, replay_bundle, budget, out_dir,
                 probe_paths=None, anchor=None, learning_rate=1e-4):
        from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE

        self.arm = arm
        self.seed = int(seed)
        self.bundle = replay_bundle
        self.replay = replay_bundle["replay"]
        self.steps = replay_bundle["steps"]
        self.budget = budget
        self.learning_rate = float(learning_rate)
        self.anchor = anchor
        self.update_every = int(budget["update_every_scored_intervals"])
        self.batch_size = int(budget["batch_size"])
        self.grad_steps = int(budget["gradient_steps_per_opportunity"])
        self.buffer_limit = int(budget["replay_buffer_intervals"])
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.probe_paths = probe_paths or {}

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.model = FrozenResidualFTMoE(replay_bundle["checkpoint"],
                                         "A" if arm == "A" else "C", self.seed)
        self.model.eval()
        if arm == "A":
            self.model.set_deployment("live", 0.0)
            self.optimizer = None
        else:
            self.model.set_deployment("learner", 1.0)
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate,
                                               weight_decay=1e-4)
        self.frozen_hash = self.model.frozen_hash()
        self.model_version = 0
        self.updates = 0
        self.raw_seen = np.full((self.steps + 1, 16), -1, dtype=np.int64)
        self.predictions = {
            "probability": np.full((self.steps, 16), np.nan, np.float32),
            "class_probability": np.full((self.steps, 16, 3), np.nan, np.float32),
            "detection_logits": np.full((self.steps, 16, 2), np.nan, np.float32),
            "class_logits": np.full((self.steps, 16, 3), np.nan, np.float32),
            "labels": np.full((self.steps, 16), -1, np.int64),
            "raw_labels": np.full((self.steps, 16), -1, np.int64),
            "model_version": np.zeros(self.steps, np.int64),
            "learner_hash": np.empty(self.steps, dtype=object),
            "settled_at": np.full(self.steps, -1, np.int64),
            "prediction_seconds": np.zeros(self.steps, np.float64),
            "write_seconds": np.zeros(self.steps, np.float64),
        }
        self.learner_hash = self.model.learner_state_hash()
        self.buffer = []
        self.settled = np.zeros(self.steps, dtype=bool)
        self.update_log = []
        self.settlement_log = []
        self.phase_log = []
        self.probe_matrix = []
        self.checkpoints = {}
        self._probe_cache = None
        self.cursor = 0

    # -- integrity ---------------------------------------------------------
    def assert_frozen(self):
        if self.model.frozen_hash() != self.frozen_hash:
            raise AssertionError("the frozen base changed during the run")

    def learner_state_hash(self):
        return self.model.learner_state_hash()

    # -- one interval ------------------------------------------------------
    def step(self):
        t = self.cursor
        if t >= self.steps:
            raise StopIteration
        # 1./2. predict with the current model and WRITE the record before the
        # label of this interval is read anywhere.
        started = time.perf_counter()
        x, s, g, context = window_batch(self.replay, [t])
        out = self.model.predict_deployment(x, s, g, graph_context=context)
        probability = torch.softmax(out["detection_logits"], -1)[0, :, 1] \
            .detach().numpy().astype(np.float32)
        classes = torch.softmax(out["class_logits"], -1)[0] \
            .detach().numpy().astype(np.float32)
        if not (np.isfinite(probability).all() and np.isfinite(classes).all()):
            raise RuntimeError("non-finite prediction at %d" % t)
        self.predictions["probability"][t] = probability
        self.predictions["class_probability"][t] = classes
        self.predictions["detection_logits"][t] = out["detection_logits"][0] \
            .detach().numpy()
        self.predictions["class_logits"][t] = out["class_logits"][0] \
            .detach().numpy()
        self.predictions["model_version"][t] = self.model_version
        self.predictions["learner_hash"][t] = self.learner_hash
        self.predictions["prediction_seconds"][t] = time.perf_counter() - started
        assert np.all(self.raw_seen[t] == -1), \
            "label of %d was read before the prediction" % t

        # 3. the environment advances: the label that has *just* matured is the
        #    one of interval t-1 (a label is only usable one interval later).
        self.raw_seen[t] = np.asarray(
            self.bundle["arrays"]["raw_labels"][t], dtype=np.int64).copy()
        if t >= 1:
            self.predictions["raw_labels"][t - 1] = self.raw_seen[t - 1]
        # 4. settle every pending prediction whose label is now known (t-2)
        settled_now = t - 2
        if settled_now >= 0:
            label = tolerance_label(self.raw_seen, settled_now, t - 1)
            self.predictions["labels"][settled_now] = label
            self.predictions["settled_at"][settled_now] = t
            self.settled[settled_now] = True
            self.buffer.append(settled_now)
            if len(self.buffer) > self.buffer_limit:
                self.buffer = self.buffer[-self.buffer_limit:]
            # the settle step must not have used the label of its own interval
            assert self.predictions["model_version"][settled_now] \
                <= self.model_version, "settlement changed the model version"
        # 5./6. replay+anchor selection and, only on an update opportunity, train
        if self.arm == "C" and (t + 1) % self.update_every == 0:
            self.update(t)
        # 7. the model version advances with every scored interval
        self.model_version += 1
        self.cursor = t + 1
        return probability, classes

    def update(self, t):
        if self.optimizer is None:
            return
        usable = [i for i in self.buffer if self.settled[i]]
        if not usable:
            return
        rng = np.random.default_rng(self.seed * 7919 + self.updates)
        started = time.perf_counter()
        self.model.train()
        losses = []
        for _ in range(self.grad_steps):
            batch = list(rng.choice(usable,
                                    size=min(self.batch_size, len(usable)),
                                    replace=len(usable) < self.batch_size))
            if max(batch) > self.cursor - 2:
                raise AssertionError("an immature sample entered the batch")
            if not all(self.settled[int(i)] for i in batch):
                raise AssertionError("an unsettled sample entered the batch")
            x, s, g, context = window_batch(self.replay, batch)
            target = torch.from_numpy(
                self.predictions["labels"][batch].reshape(-1).astype(np.int64))
            out = self.model(x, s, g, graph_context=context)
            terms = supervised_terms(out["detection_logits"],
                                     out["class_logits"], target)
            distill = F.mse_loss(out["detection_logits"],
                                 out["base_final_detection_logits"].detach())
            total = terms["total"] + 0.10 * distill
            anchor_terms = None
            if self.anchor is not None:
                anchor_index = int(rng.integers(0, self.anchor["x"].shape[0]))
                anchor_out = self.model(
                    self.anchor["x"][anchor_index:anchor_index + 1],
                    self.anchor["schedule"][anchor_index:anchor_index + 1],
                    self.anchor["graph_x"][anchor_index:anchor_index + 1],
                    graph_context=graph_context(
                        self.anchor["ids"][anchor_index:anchor_index + 1],
                        self.anchor["before"][anchor_index:anchor_index + 1],
                        self.anchor["caps"][anchor_index:anchor_index + 1]))
                anchor_target = self.anchor["labels"][anchor_index].reshape(-1)
                anchor_terms = supervised_terms(anchor_out["detection_logits"],
                                                anchor_out["class_logits"],
                                                anchor_target)
                total = total + 0.25 * anchor_terms["total"]
            if not torch.isfinite(total):
                raise RuntimeError("non-finite online loss")
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            leaked = [name for name, p in self.model.named_parameters()
                      if p.grad is not None and not p.requires_grad]
            if leaked:
                raise AssertionError("the frozen base received gradients: %s"
                                     % leaked[:3])
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0))
            self.optimizer.step()
            losses.append({"total": float(total.detach()),
                           "detection": float(terms["detection"].detach()),
                           "classification": float(terms["classification"].detach()),
                           "ranking": float(terms["ranking"].detach()),
                           "distill": float(distill.detach()),
                           "anchor": (None if anchor_terms is None
                                      else float(anchor_terms["total"].detach())),
                           "grad_norm": grad_norm})
        self.model.eval()
        self.assert_frozen()
        self.updates += 1
        self.learner_hash = self.learner_state_hash()
        self.update_log.append({
            "opportunity": self.updates, "at_interval": int(t),
            "batch_size": int(min(self.batch_size, len(usable))),
            "gradient_steps": self.grad_steps,
            "buffer_size": len(usable),
            "buffer_indices": [int(i) for i in batch],
            "losses": losses,
            "learner_hash_after": self.learner_hash,
            "model_version_after": self.model_version,
            "seconds": time.perf_counter() - started,
            "settled_labels_only": True,
        })

    def finish(self):
        if self.cursor != self.steps:
            raise ValueError("cannot finalize a partial stream (%d/%d)"
                             % (self.cursor, self.steps))
        # The stream carries steps+1 rows: row ``steps`` is the guard interval
        # whose label the +/-1 tolerance of the LAST scored interval needs.  It
        # is read here, at the end, exactly as the collector's guard row is.
        self.raw_seen[self.steps] = np.asarray(
            self.bundle["arrays"]["raw_labels"][self.steps],
            dtype=np.int64).copy()
        self.predictions["raw_labels"][self.steps - 1] = \
            self.raw_seen[self.steps - 1]
        label = tolerance_label(self.raw_seen, self.steps - 1, self.steps)
        self.predictions["labels"][self.steps - 1] = label
        self.settled[self.steps - 1] = True
        self.predictions["settled_at"][self.steps - 1] = self.steps
        self.assert_frozen()

    # -- probes ------------------------------------------------------------
    def _probe_replays(self):
        """One ReplayV3 per fixed probe, built once (the slices never move)."""
        if self._probe_cache is not None:
            return self._probe_cache
        from run_ftmoe_protocol020 import ReplayV3
        cache = {}
        bundle = build_replay(DEV_STREAM)
        for regime, path in sorted(self.probe_paths.items()):
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
            steps = int(arrays["raw_labels"].shape[0]) - 1
            caps_per_interval = np.asarray(arrays["capacities"], np.float64)
            probe_arrays = dict(arrays)
            probe_arrays["capacities_per_interval"] = caps_per_interval
            probe_arrays["capacities"] = caps_per_interval[0]
            cache[regime] = {
                "replay": ReplayV3(probe_arrays, bundle["replay"].time_scale,
                                   bundle["replay"].graph_scale, steps),
                "steps": steps,
                "labels": np.asarray(arrays["raw_labels"][:steps])}
        self._probe_cache = cache
        return cache

    def probe_scores(self, checkpoint_label, index):
        """Score every fixed probe with the CURRENT model (directive §13)."""
        if not self.probe_paths:
            return
        cache = self._probe_replays()
        for regime, entry in sorted(cache.items()):
            replay, steps = entry["replay"], entry["steps"]
            indices = list(range(steps))
            probabilities, classes = [], []
            self.model.eval()
            with torch.no_grad():
                for offset in range(0, len(indices), 32):
                    chunk = indices[offset:offset + 32]
                    x, s, g, context = window_batch(replay, chunk)
                    out = self.model.predict_deployment(
                        x, s, g, graph_context=context)
                    probabilities.append(
                        torch.softmax(out["detection_logits"], -1)[..., 1]
                        .numpy().reshape(-1))
                    classes.append(torch.softmax(out["class_logits"], -1)
                                   .numpy().reshape(-1, 3))
            probability = np.concatenate(probabilities)
            class_probability = np.concatenate(classes)
            labels = entry["labels"].reshape(-1)
            self.probe_matrix.append({
                "checkpoint": checkpoint_label, "time_index": int(index),
                "probe": regime, "model_version": self.model_version,
                "learner_hash": self.learner_hash,
                "detection": detection_metrics(probability, labels),
                "diagnosis": resource_macro_f1(class_probability, labels),
                "onset": onset_metrics(probability, labels, horizon=1),
            })

    def save_checkpoint(self, phase_name, index):
        """A mature checkpoint (directive §12) with its provenance hashes."""
        name = CHECKPOINT_NAMES[phase_name]
        path = CKPT_DIR / name
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "023", "round": "2A", "arm": self.arm,
            "phase": phase_name, "time_index": int(index),
            "update_count": int(self.updates),
            "model_version": int(self.model_version),
            "model": self.model.state_dict(),
            "base_hash": self.model.frozen_hash(),
            "learner_hash": self.model.learner_state_hash(),
            "optimizer_hash": (None if self.optimizer is None else
                               _hash_state_dict(self.optimizer.state_dict())),
            "replay_hash": self.replay_hash(),
            "stream_sha256": self.bundle["manifest"]["stream_sha256"],
            "checkpoint_source_sha256": sha(CHECKPOINT),
            "learning_rate": self.learning_rate,
            "budget": self.budget,
            "stream": str(DEV_STREAM),
        }
        torch.save(payload, path)
        self.checkpoints[phase_name] = {
            "file": name, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha(path), "base_hash": payload["base_hash"],
            "learner_hash": payload["learner_hash"],
            "optimizer_hash": payload["optimizer_hash"],
            "replay_hash": payload["replay_hash"],
            "update_count": payload["update_count"],
            "phase": phase_name, "time_index": int(index),
        }
        return self.checkpoints[phase_name]

    def replay_hash(self):
        """Hash of the replay buffer state a checkpoint was taken with."""
        digest = hashlib.sha256()
        digest.update(json.dumps({"arm": self.arm, "seed": self.seed,
                                  "buffer_limit": self.buffer_limit,
                                  "update_every": self.update_every,
                                  "batch_size": self.batch_size,
                                  "steps": self.steps}).encode("utf8"))
        digest.update(np.asarray(sorted(self.buffer), dtype=np.int64).tobytes())
        digest.update(np.asarray(self.settled, dtype=np.int8).tobytes())
        return digest.hexdigest()

    def phase_metrics(self):
        """Per-phase, per-arm metrics from the settled predictions.

        Three readings per phase, because the H4-v2 gate is stated on the *late*
        phase and "late" has to be pinned down:

            whole        every scored interval of the phase
            first_100    the first 100 scored intervals (adaptation speed)
            late         the last 100 scored intervals (the gate's reading)
        """
        out = []
        for name, start, end, regime in PHASES:
            entry = {"phase": name, "regime": regime,
                     "intervals": [int(start), int(end)], "arm": self.arm,
                     "updates_by_phase_end": int(sum(
                         1 for item in self.update_log
                         if item["at_interval"] < end))}
            windows = {"whole": (start, end),
                       "first_100": (start, min(start + 100, end)),
                       "late": (max(end - 100, start), end)}
            for label, (low, high) in windows.items():
                labels = self.predictions["labels"][low:high].reshape(-1)
                raw = self.predictions["raw_labels"][low:high].reshape(-1)
                probability = self.predictions["probability"][low:high] \
                    .reshape(-1)
                classes = self.predictions["class_probability"][low:high] \
                    .reshape(-1, 3)
                mask = labels >= 0
                block = {"intervals": [int(low), int(high)],
                         "rows": int(mask.sum())}
                if mask.sum():
                    block["detection"] = detection_metrics(probability[mask],
                                                           labels[mask])
                    block["detection_raw"] = detection_metrics(
                        probability[mask], raw[mask])
                    block["diagnosis"] = resource_macro_f1(classes[mask],
                                                           labels[mask])
                    block["onset"] = onset_metrics(probability[mask],
                                                   labels[mask], horizon=1)
                entry[label] = block
            # the round-1-compatible flat view kept at the top level
            entry["rows"] = entry["whole"]["rows"]
            entry["detection"] = entry["whole"].get("detection")
            entry["detection_raw"] = entry["whole"].get("detection_raw")
            entry["diagnosis"] = entry["whole"].get("diagnosis")
            entry["onset"] = entry["whole"].get("onset")
            out.append(entry)
        return out

    def recurrence_metrics(self):
        """First-100 recurrence score minus the previous exposure's last score."""
        out = []
        by_phase = {item["phase"]: item for item in self.phase_metrics()}
        for recurrence, first in RECURRENCE.items():
            rec = by_phase.get(recurrence)
            prev = by_phase.get(first)
            if rec is None or prev is None:
                continue
            previous_end = ((prev.get("late") or {}).get("detection")
                            or prev.get("detection"))
            first100 = ((rec.get("first_100") or {}).get("detection")
                        or rec.get("detection"))
            late = ((rec.get("late") or {}).get("detection")
                    or rec.get("detection"))
            entry = {"recurrence_phase": recurrence,
                     "first_exposure_phase": first,
                     "previous_exposure_end_pr_auc": (
                         None if previous_end is None
                         else previous_end.get("pr_auc")),
                     "recurrence_first_100_pr_auc": (
                         None if first100 is None else first100.get("pr_auc")),
                     "recurrence_late_pr_auc": (
                         None if late is None else late.get("pr_auc")),
                     "windows": {"previous_exposure_end":
                                 (prev.get("late") or {}).get("intervals"),
                                 "recurrence_first_100":
                                 (rec.get("first_100") or {}).get("intervals"),
                                 "recurrence_late":
                                 (rec.get("late") or {}).get("intervals")}}
            if entry["previous_exposure_end_pr_auc"] is not None and \
                    entry["recurrence_first_100_pr_auc"] is not None:
                entry["relearning_gap"] = (
                    entry["previous_exposure_end_pr_auc"]
                    - entry["recurrence_first_100_pr_auc"])
                entry["relearning_gap_definition"] = (
                    "previous exposure end score - recurrence first-100 score")
                entry["recovery_gain_within_recurrence"] = (
                    None if entry["recurrence_late_pr_auc"] is None else
                    entry["recurrence_late_pr_auc"]
                    - entry["recurrence_first_100_pr_auc"])
            out.append(entry)
        return out

    def save(self):
        path = self.out_dir / "predictions.npz"
        np.savez_compressed(
            path,
            probability=self.predictions["probability"],
            class_probability=self.predictions["class_probability"],
            detection_logits=self.predictions["detection_logits"],
            class_logits=self.predictions["class_logits"],
            labels=self.predictions["labels"],
            raw_labels=self.predictions["raw_labels"],
            model_version=self.predictions["model_version"],
            settled_at=self.predictions["settled_at"],
            prediction_seconds=self.predictions["prediction_seconds"])
        return path


def _hash_state_dict(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(str(key).encode("utf8"))
        if isinstance(value, dict):
            for sub in sorted(value):
                item = value[sub]
                if torch.is_tensor(item):
                    digest.update(str(sub).encode("ascii"))
                    digest.update(item.detach().cpu().contiguous()
                                  .numpy().tobytes())
        elif torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tolerance_label(raw_seen, index, observed_until):
    """Mature label of ``index``, usable only once ``observed_until >= index+1``."""
    if index < 0 or index + 1 > observed_until:
        raise ValueError("tolerance label of %d has not matured" % index)
    labels = raw_seen[:index + 2].copy()
    result = labels[index].copy()
    if (labels[max(0, index - 1):index + 2] < 0).any():
        raise ValueError("unobserved raw labels around %d" % index)
    if index:
        fill = (result == 0) & (labels[index - 1] > 0)
        result[fill] = labels[index - 1][fill]
    fill = (result == 0) & (labels[index + 1] > 0)
    result[fill] = labels[index + 1][fill]
    return result


def run(args):
    configure()
    if not DEV_STREAM.is_dir():
        raise SystemExit("calibrated development stream missing: %s" % DEV_STREAM)
    budget = json.loads(BUDGET_FILE.read_text(encoding="utf8"))
    frozen = budget["frozen_configuration"]
    if (int(frozen["update_every_scored_intervals"]) != 4
            or int(frozen["batch_size"]) != 32
            or int(frozen["gradient_steps_per_opportunity"]) != 1):
        raise SystemExit("the frozen online budget is not the registered one; "
                         "S4 may not run with a changed budget")
    bundle = build_replay(DEV_STREAM)
    anchor = load_anchor_pool(bundle["replay"].time_scale)
    probes = {}
    probe_registry = PROBE_DIR / "probe_registry.json"
    if probe_registry.is_file():
        registry = json.loads(probe_registry.read_text(encoding="utf8"))
        for regime, entry in registry["probes"].items():
            probes[regime] = PROBE_DIR / entry["file"]
    else:
        raise SystemExit("no probe registry: run --make-probes first "
                         "(directive §13 requires the probe matrix to be fixed "
                         "before the model runs)")

    out_dir = OUT_DIR / ("arm_%s" % args.arm)
    session = PrequentialS4(args.arm, args.seed, bundle, frozen, out_dir,
                            probe_paths=probes, anchor=anchor,
                            learning_rate=args.learning_rate)
    phase_by_name = {p[0]: p for p in PHASES}
    session.probe_scores("start", 0)
    started = time.perf_counter()
    update_log_path = out_dir / "update_log.jsonl"
    settlement_path = out_dir / "settlements.jsonl"
    written_updates = 0
    with update_log_path.open("w", encoding="utf8") as update_log, \
            settlement_path.open("w", encoding="utf8") as settlements:
        while session.cursor < session.steps:
            session.step()
            t = session.cursor
            if t - 2 >= 0:
                settlements.write(json.dumps({
                    "settled_index": t - 2, "at_interval": t,
                    "model_version_of_prediction":
                        int(session.predictions["model_version"][t - 2]),
                    "current_model_version": session.model_version,
                    "label": session.predictions["labels"][t - 2].tolist(),
                }) + "\n")
                settlements.flush()
            while written_updates < len(session.update_log):
                update_log.write(json.dumps(
                    session.update_log[written_updates]) + "\n")
                written_updates += 1
                update_log.flush()
            for name in CHECKPOINT_PHASES:
                _, start, end, _ = phase_by_name[name]
                if t == end and name not in session.checkpoints:
                    label = "after_%s" % name.split("_")[0]
                    session.probe_scores(label, t)
                    session.save_checkpoint(name, t)
                    print(json.dumps({"checkpoint": label, "interval": t,
                                      "updates": session.updates,
                                      "learner_hash":
                                          session.learner_hash[:16]}),
                          flush=True)
            if t % 100 == 0:
                print(json.dumps({"arm": args.arm, "interval": t,
                                  "updates": session.updates,
                                  "elapsed_seconds": round(
                                      time.perf_counter() - started, 1)}),
                      flush=True)
            if args.benchmark and t >= args.benchmark_intervals:
                break
    if args.benchmark:
        elapsed = time.perf_counter() - started
        return {"benchmark": True, "arm": args.arm, "intervals": session.cursor,
                "updates": session.updates, "elapsed_seconds": elapsed,
                "prediction_mean_seconds": float(
                    session.predictions["prediction_seconds"][
                        :session.cursor].mean()),
                "update_mean_seconds": (float(np.mean(
                    [item["seconds"] for item in session.update_log]))
                    if session.update_log else None)}

    session.finish()
    session.save()
    write_json(out_dir / "phase_metrics.json", {
        "protocol": "023", "round": "2A", "arm": args.arm,
        "stream": str(DEV_STREAM.relative_to(ROOT)).replace("\\", "/"),
        "phases": session.phase_metrics(),
    })
    write_json(out_dir / "probe_matrix.json", {
        "protocol": "023", "round": "2A", "arm": args.arm,
        "probe_geometry": {"start": PROBE_START, "length": PROBE_LENGTH},
        "cells": session.probe_matrix,
    })
    write_json(out_dir / "recurrence_metrics.json", {
        "protocol": "023", "round": "2A", "arm": args.arm,
        "recurrence": session.recurrence_metrics(),
    })
    write_json(out_dir / "checkpoints.json", {
        "protocol": "023", "round": "2A", "arm": args.arm,
        "checkpoints": session.checkpoints,
        "required": list(CHECKPOINT_NAMES.values()),
    })
    summary = {
        "protocol": "023", "round": "2A", "arm": args.arm,
        "stream": str(DEV_STREAM.relative_to(ROOT)).replace("\\", "/"),
        "stream_sha256": bundle["manifest"]["stream_sha256"],
        "budget": frozen,
        "learning_rate": args.learning_rate,
        "updates": session.updates,
        "intervals": session.cursor,
        "model_version_final": int(session.model_version),
        "frozen_parameters_unchanged": session.model.frozen_hash()
        == session.frozen_hash,
        "initial_learner_hash": session.predictions["learner_hash"][0],
        "final_learner_hash": session.learner_hash,
        "prediction_mean_seconds": float(
            session.predictions["prediction_seconds"].mean()),
        "update_mean_seconds": (float(np.mean(
            [item["seconds"] for item in session.update_log]))
            if session.update_log else None),
        "update_total_seconds": float(sum(
            item["seconds"] for item in session.update_log)),
        "elapsed_seconds": time.perf_counter() - started,
        "prequential_integrity": prequential_integrity(session),
        "checkpoints": session.checkpoints,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def prequential_integrity(session):
    """Audit of the order the directive registers (directive §11, §21)."""
    predictions = session.predictions
    settled_at = predictions["settled_at"]
    versions = predictions["model_version"]
    settled = settled_at >= 0
    last = session.steps - 1
    # The final scored interval is the one interval whose label does not need a
    # later prediction: the stream carries that guard row itself.  It is settled
    # at ``steps`` and is checked separately, so the "strictly older" rule stays
    # exact for every other interval instead of being loosened globally.
    strict = settled.copy()
    strict[last] = False
    checks = {
        "every_prediction_written_before_its_label": bool(
            np.all(settled_at[settled] > np.arange(session.steps)[settled])),
        "settled_label_index_strictly_older_than_read_index": bool(
            np.all(settled_at[strict]
                   - np.arange(session.steps)[strict] >= 2)),
        "final_interval_settled_on_the_stream_guard_row": bool(
            settled_at[last] == session.steps),
        "prediction_version_never_after_settlement_version": True,
        "no_prediction_used_a_future_label": True,
        "update_count_matches_opportunities": bool(
            session.updates == len(session.update_log)),
    }
    # a prediction's model version must not exceed the version at which its
    # label was settled
    for index in np.nonzero(settled)[0]:
        if versions[index] > int(settled_at[index]):
            checks["prediction_version_never_after_settlement_version"] = False
            break
    for item in session.update_log:
        if max(item["buffer_indices"]) >= item["at_interval"] - 1:
            checks["no_prediction_used_a_future_label"] = False
            break
    checks["passed"] = all(checks.values())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "C"), default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--make-probes", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-intervals", type=int, default=120)
    parser.add_argument("--stream", type=Path, default=None,
                        help="development stream directory (default: the "
                             "calibrated registered stream)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: fixed_c_prequential/"
                             "arm_<arm>)")
    args = parser.parse_args()

    if args.stream is not None:
        global DEV_STREAM
        DEV_STREAM = (args.stream if args.stream.is_absolute()
                      else (ROOT / args.stream).resolve())
    if args.out is not None:
        global OUT_DIR
        OUT_DIR = args.out if args.out.is_absolute() else (ROOT / args.out)

    if args.make_probes:
        registry = make_probes()
        print(json.dumps({"probes": sorted(registry["probes"]),
                          "registry": str((PROBE_DIR / "probe_registry.json")
                                          .relative_to(ROOT))}, indent=2))
        return 0
    if args.arm is None:
        raise SystemExit("pass --arm A|--arm C, or --make-probes")
    if args.learning_rate is None:
        args.learning_rate = 1e-4
    try:
        summary = run(args)
    except Exception as exc:
        write_json(OUT_DIR / ("arm_%s" % args.arm) / "failure.json",
                   {"error": str(exc), "traceback": traceback.format_exc()})
        raise
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("checkpoints",)}, ensure_ascii=False,
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
