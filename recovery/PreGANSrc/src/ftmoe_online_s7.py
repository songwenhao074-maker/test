"""Protocol 020 S7 — online replay session (A/B/C) with rare-event sampling
and class-balanced loss v3 (plan §18/§19/§20).

Replaces the S4 memory design with:

- Recent memory (interval-level, last 128 matured intervals) stratified by
  matured interval event tags: every update draws
      12 uniform recent + 4 CPU-event + 4 RAM-event + 4 Disk-event
  (shortfalls are back-filled from the uniform pool), plus 8 anchors.
- Exposure caps: uniform intervals <= 3, rare-event intervals <= 5
  (registered, plan §18.4).
- Anchor memory: rows 272-399 of every protocol-020 adaptation train
  episode (12 episodes x 128 rows = 1536 windows), graph-semantics v3
  (creation ids + before_placement + per-sample capacities).
- Loss v3 (plan §19): class-balanced weights are computed ONCE from the
  same-domain training bundle (never from the online stream):
      detection  w1 = clip(sqrt(N0/N1), 2, 10),  w0 = 1
      resource   wc = clip(sqrt(Nmax/Nc), 1, 5) on positives
  L = L_online + 0.25 * L_anchor + 0.10 * L_distill (lambda 019 values,
  plan §19.3).  Anchor and online share the same weighted loss.
- Learning-rate groups and A/B/C trainability are inherited from S4.

Prequential core, label maturation, hashes and output layout follow
run_ftmoe_protocol019/OnlineSessionV2.  Graph semantics v3 needs the
extended ReplayV3.window (x, s, g, ids, before, capacities); the runner
passes it in.
"""
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from recovery.PreGANSrc.src.ftmoe_online_s4 import (
    S4Session, RECENT_MAXLEN, ANCHOR_PER_UPDATE, LAMBDA_ANCHOR,
    LAMBDA_DISTILL, SLOW_LR_GROUPS, LR_FAST, LR_SLOW_FACTOR,
)
from run_ftmoe_online import tolerance_label

RECENT_UNIFORM_PER_UPDATE = 12
EVENT_PER_UPDATE = 4          # per event pool (cpu / ram / disk)
MAX_EXPOSURE = 3
MAX_RARE_EXPOSURE = 5
W1_MIN, W1_MAX = 2.0, 10.0
WC_MIN, WC_MAX = 1.0, 5.0


def balanced_weights(labels):
    """labels: int64 host-step array (0 normal, 1/2/3 resources)."""
    counts = np.bincount(np.asarray(labels).ravel(), minlength=4).astype(np.float64)
    n0, n1, n2, n3 = counts
    anomaly = n1 + n2 + n3
    w1 = float(np.clip(np.sqrt(n0 / max(anomaly, 1)), W1_MIN, W1_MAX))
    n_res = np.asarray([n1, n2, n3])
    n_max = n_res.max()
    w_res = np.clip(np.sqrt(n_max / np.maximum(n_res, 1.0)), WC_MIN, WC_MAX)
    return {
        "counts": counts.tolist(),
        "detection_weight": [1.0, w1],
        "resource_weight": w_res.tolist(),
        "formula": "w1=clip(sqrt(N0/N1),2,10); wc=clip(sqrt(Nmax/Nc),1,5) "
                   "computed on the same-domain training bundle only",
    }


def s7_loss(model, output, labels, detection_weights, resource_weights,
            detection_weight, classification_weight, prototype_weight,
            balance_weight, ranking_weight):
    """Class-balanced v3 loss; all other terms equal the legacy loss_fn."""
    weights = torch.as_tensor(detection_weights, dtype=output["detection_logits"].dtype)
    anomaly = (labels > 0).long()
    detection = F.cross_entropy(
        output["detection_logits"].reshape(-1, 2), anomaly.reshape(-1),
        weight=weights)
    positive = labels > 0
    if positive.any():
        class_weights = torch.as_tensor(resource_weights,
                                        dtype=output["class_logits"].dtype)
        classification = F.cross_entropy(
            output["class_logits"][positive], labels[positive] - 1,
            weight=class_weights)
    else:
        classification = output["detection_logits"].new_zeros(())
    prototype, balance = model.auxiliary_losses(output)
    ranking = output["detection_logits"].new_zeros(())
    if ranking_weight > 0 and positive.any() and (~positive).any():
        anomaly_probability = torch.softmax(output["detection_logits"], -1)[..., 1]
        class_probability = torch.softmax(output["class_logits"], -1)
        true_index = (labels.clamp_min(1) - 1).unsqueeze(-1)
        true_class_probability = class_probability.gather(-1, true_index).squeeze(-1)
        positive_score = (anomaly_probability[positive] *
                          true_class_probability[positive])
        negative_score = (anomaly_probability[~positive] *
                          class_probability[~positive].max(dim=-1).values)
        ranking = F.softplus(
            0.15 + negative_score.unsqueeze(0) - positive_score.unsqueeze(1)).mean()
    return (detection_weight * detection +
            classification_weight * classification +
            prototype_weight * prototype + balance_weight * balance +
            ranking_weight * ranking)


def stratified_draw(buffer_indices, tags_of, rng, exposure, rare_exposure,
                    uniform_per_update=RECENT_UNIFORM_PER_UPDATE,
                    event_per_update=EVENT_PER_UPDATE,
                    max_exposure=MAX_EXPOSURE,
                    max_rare_exposure=MAX_RARE_EXPOSURE,
                    total_target=None):
    """Pure stratified draw over matured interval indices (plan §18.3).

    buffer_indices: iterable of interval indices (deduplicated inside).
    tags_of(i): set of event tags {'cpu','ram','disk'} for interval i.
    Draws uniform_per_update intervals with exposure < max_exposure, then
    event_per_update per event pool from intervals whose rare exposure is
    < max_rare_exposure; shortfalls are back-filled from the uniform pool
    (uniform cap only).  Mutates exposure/rare_exposure counters.
    Returns the ordered drawn list (never larger than total_target).
    """
    pool = list(dict.fromkeys(buffer_indices))
    uniform_pool = [i for i in pool if exposure.get(i, 0) < max_exposure]
    drawn = []
    uniform_drawn = rng.sample(uniform_pool,
                               min(uniform_per_update, len(uniform_pool)))
    for i in uniform_drawn:
        exposure[i] = exposure.get(i, 0) + 1
    drawn.extend(uniform_drawn)
    for tag in ("cpu", "ram", "disk"):
        event_pool = [i for i in pool
                      if i not in drawn and tag in tags_of(i)
                      and rare_exposure.get(i, 0) < max_rare_exposure]
        picked = rng.sample(event_pool, min(event_per_update, len(event_pool)))
        for i in picked:
            rare_exposure[i] = rare_exposure.get(i, 0) + 1
        drawn.extend(picked)
    if total_target is not None and len(drawn) < total_target:
        remainder = [i for i in pool if i not in drawn
                     and exposure.get(i, 0) < max_exposure]
        fill = rng.sample(remainder, min(total_target - len(drawn),
                                         len(remainder)))
        for i in fill:
            exposure[i] = exposure.get(i, 0) + 1
        drawn.extend(fill)
    return drawn


class S7Session(S4Session):
    """A/B/C session on the v3 input contract with stratified rare-event
    sampling and class-balanced loss v3."""

    def __init__(self, checkpoint, method, seed, replay, learning_rate,
                 replay_seed, normalization_v2, anchor_pool, anchor_teacher,
                 class_balance, window_fn="window_v3"):
        S4Session.__init__(self, checkpoint, method, seed, replay,
                           learning_rate, replay_seed, normalization_v2,
                           anchor_pool, anchor_teacher)
        self.class_balance = class_balance
        self.detection_weights = class_balance["detection_weight"]
        self.resource_weights = class_balance["resource_weight"]
        self.window_fn = window_fn
        self.rare_exposure = {}
        if method not in ("A", "B", "C"):
            raise ValueError("S7 A/B/C session only; D needs the v3 dynamic gate")

    @torch.no_grad()
    def _cache_teacher_probs(self):
        """Teacher probabilities on the anchor pool under graph semantics v3."""
        pool = self.anchor_pool
        teacher = self.anchor_teacher
        teacher.eval()
        dets, clss = [], []
        for start in range(0, pool["x"].shape[0], 64):
            out = teacher(pool["x"][start:start + 64],
                          pool["schedule"][start:start + 64],
                          pool["graph_x"][start:start + 64],
                          graph_context=self._context(
                              pool["ids"][start:start + 64],
                              pool["before"][start:start + 64],
                              pool["caps"][start:start + 64]))
            dets.append(out["detection_logits"].softmax(-1))
            clss.append(out["class_logits"].softmax(-1))
        return torch.cat(dets), torch.cat(clss)

    # ---- sampling -------------------------------------------------------
    def _interval_tags(self, index):
        """Event tags of one matured interval from its tolerance labels."""
        labels = self.predictions["labels"][index]
        tags = set()
        if (labels == 1).any():
            tags.add("cpu")
        if (labels == 2).any():
            tags.add("ram")
        if (labels == 3).any():
            tags.add("disk")
        return tags

    def _draw_recent(self, count):
        """Registered stratified draw (uniform + cpu/ram/disk event pools)."""
        return stratified_draw(self.buffer, self._interval_tags, self.sample_rng,
                               self.exposure, self.rare_exposure,
                               total_target=count)

    def _window(self, index):
        return getattr(self.replay, self.window_fn)(index)

    # ---- forward helpers ------------------------------------------------
    def _stack(self, indices):
        windows = [self._window(i) for i in indices]
        return (torch.stack([w[0] for w in windows]),
                torch.stack([w[1] for w in windows]),
                torch.stack([w[2] for w in windows]),
                torch.stack([w[3] for w in windows]),
                torch.stack([w[4] for w in windows]),
                torch.stack([w[5] for w in windows]))

    def _context(self, ids, before, caps):
        return {"creation_ids": ids, "before_placement": before,
                "capacities": caps}

    def step(self, prediction_sink=None):
        """Copy of OnlineSessionV2.step with the v3 window/context."""
        t = self.cursor
        if t >= self.replay.steps:
            raise StopIteration
        x, s, g, ids, before, caps = self._window(t)
        started = time.perf_counter()
        out = self.model.predict_online(
            x[None], s[None], g[None],
            graph_context=self._context(ids[None], before[None], caps[None]))
        probability = out["detection_logits"].softmax(-1)[0, :, 1].numpy()
        classes = out["class_logits"].softmax(-1)[0].numpy()
        if not np.isfinite(probability).all() or not np.isfinite(classes).all():
            raise RuntimeError("Nonfinite prediction")
        self.predictions["probability"][t] = probability
        self.predictions["class_probability"][t] = classes
        self.predictions["prediction_seconds"][t] = time.perf_counter() - started
        self.predictions["model_version"][t] = self.update_number
        for key, value in self.model.eagate.last_routing.items():
            self.predictions[key][t] = value
        if prediction_sink is not None:
            prediction_sink({"step": t + 1, "model_version": self.update_number,
                             "probability": probability.tolist(),
                             "class_probability": classes.tolist(),
                             **self.model.eagate.last_routing})
        self.raw_seen[t] = self.replay.arrays["raw_labels"][t]
        self.predictions["raw_labels"][t] = self.raw_seen[t]
        if t:
            self.predictions["labels"][t - 1] = tolerance_label(
                self.raw_seen, t - 1, t)
            self.buffer.append(t - 1)
        self.cursor = t + 1
        if self.cursor % 10 == 0 and self.optimizer is not None and self.buffer:
            self.update()
        if self.cursor % 100 == 0:
            if self.model.frozen_hash() != self.initial_frozen_hash:
                raise AssertionError("Frozen parameters changed")
            if self.method == "A":
                self.model.eagate.reset_statistics()
        return probability, classes

    def update(self):
        started = time.perf_counter()
        self.model.eval()
        recent = self._draw_recent(RECENT_UNIFORM_PER_UPDATE +
                                   3 * EVENT_PER_UPDATE)
        anchor_count = ANCHOR_PER_UPDATE
        anchor_draw = self.sample_rng.sample(
            self.anchor_indices, min(anchor_count, len(self.anchor_indices)))
        if not recent and not anchor_draw:
            self.update_number += 1
            self.updates.append({"step": self.cursor,
                                 "update_number": self.update_number,
                                 "loss": 0.0, "batches": 0,
                                 "buffer_indices": [], "anchor_indices": [],
                                 "seconds": time.perf_counter() - started,
                                 "topology_event": None,
                                 "components": {"recent_windows": 0,
                                                "anchor_windows": 0}})
            return
        x, s, g, ids, before, caps = (None,) * 6
        y = None
        if recent:
            x, s, g, ids, before, caps = self._stack(recent)
            y = torch.from_numpy(self.predictions["labels"][recent])
        ax = asch = ag = aids = abefore = acaps = ay = None
        if anchor_draw:
            pool = self.anchor_pool
            ax, asch, ag, aids = (pool["x"][anchor_draw],
                                  pool["schedule"][anchor_draw],
                                  pool["graph_x"][anchor_draw],
                                  pool["ids"][anchor_draw])
            abefore, acaps = pool["before"][anchor_draw], pool["caps"][anchor_draw]
            ay = pool["labels"][anchor_draw]
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=torch.float32)
        components = {}
        if recent:
            out = self.model(x, s, g, graph_context=self._context(ids, before, caps))
            loss_online = s7_loss(
                self.model, out, y, self.detection_weights, self.resource_weights,
                .7, .3, 0., .01, .5)
            components["loss_online"] = float(loss_online.detach())
            loss = loss + loss_online
        if anchor_draw:
            out_a = self.model(ax, asch, ag,
                               graph_context=self._context(aids, abefore, acaps))
            teacher_det, teacher_cls = (self.anchor_teacher_probs[0][anchor_draw],
                                        self.anchor_teacher_probs[1][anchor_draw])
            anomaly = (ay > 0).long()
            det_ce = F.cross_entropy(
                out_a["detection_logits"].reshape(-1, 2), anomaly.reshape(-1),
                weight=torch.as_tensor(self.detection_weights,
                                       dtype=out_a["detection_logits"].dtype))
            positive = ay > 0
            if positive.any():
                cls_ce = F.cross_entropy(
                    out_a["class_logits"][positive], ay[positive] - 1,
                    weight=torch.as_tensor(self.resource_weights,
                                           dtype=out_a["class_logits"].dtype))
            else:
                cls_ce = out_a["detection_logits"].new_zeros(())
            anchor_loss = det_ce + cls_ce
            kl_det = F.kl_div(F.log_softmax(out_a["detection_logits"], -1),
                              teacher_det, reduction="none").sum(-1).mean()
            kl_cls = F.kl_div(F.log_softmax(out_a["class_logits"], -1),
                              teacher_cls, reduction="none").sum(-1).mean()
            distill = (kl_det + kl_cls) / 2.0
            components["loss_anchor"] = float(anchor_loss.detach())
            components["loss_distill"] = float(distill.detach())
            loss = loss + LAMBDA_ANCHOR * anchor_loss + LAMBDA_DISTILL * distill
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite online loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        components["recent_windows"] = len(recent)
        components["anchor_windows"] = len(anchor_draw)
        self.update_number += 1
        event = None
        if self.update_number % 10 == 0:
            if self.method == "D":
                event = self.model.adapt(self.optimizer)
            else:
                self.model.eagate.reset_statistics()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise RuntimeError("Nonfinite online parameters")
        self.updates.append({"step": self.cursor, "update_number": self.update_number,
                             "loss": float(loss.detach()), "batches": 1,
                             "buffer_indices": list(recent),
                             "anchor_indices": anchor_draw,
                             "seconds": time.perf_counter() - started,
                             "topology_event": event, "components": components})

    def save(self):
        state = S4Session.save(self)
        state.update({"rare_exposure": dict(self.rare_exposure)})
        return state

    def restore(self, state):
        S4Session.restore(self)
        self.rare_exposure = dict(state.get("rare_exposure", {}))
