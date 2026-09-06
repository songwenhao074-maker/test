"""Protocol 019 S4 — online replay with the plan-§9 update redesign.

Subclasses the Protocol-019 online session (run_ftmoe_protocol019) and
replaces only the replay/update machinery:

- RecentReplayMemory: last 128 matured intervals, each with a per-interval
  ``times_sampled`` counter and a hard exposure cap (max 3); every update
  draws 24 recent + 8 anchor intervals (batch 32).
- AnchorReplayMemory: 128 same-domain intervals taken from the tail
  (rows 272-399) of adaptation train episodes 401-403 (registered, disjoint
  from the online dev stream seed 303).
- Loss:  L = L_online + 0.25 * L_anchor + 0.10 * L_distill
  where L_anchor uses the legacy weighted CE (normal 0.6 / fault 2.0,
  classification on positives only) on anchor labels, and L_distill is the
  mean KL(p_theta || p_theta0) over anchor detection and class softmaxes
  against the frozen starting (teacher) model.
- Learning-rate groups for B (plan §9.4): encoder./graph_encoder./cmha* at
  0.1 x LR; heads/MoE/EAGate/prototypes at 1.0 x LR.  C trains moe./eagate.
  only (single fast group).  A never updates.

The prequential core (predict -> reveal -> mature -> update), label
maturation, frozen-hash guards, references, hashes and output layout are
inherited unchanged from run_ftmoe_protocol019.
"""
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_ftmoe_protocol019 import (
    ROOT, ART, RAM_GUARD_GIB, OnlineSessionV2, ReplayV2,
    extra_detection_metrics, load_v2_time_scale, resources, sha, write_json,
)
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from train_ftmoe_ablation_existing import loss_fn

ANCHOR_SEEDS = [401, 402, 403]
ANCHOR_ROW_START = 272          # last 128 rows of each 400-row episode
ANCHOR_PER_EPISODE = 128
RECENT_MAXLEN = 128
RECENT_PER_UPDATE = 24
ANCHOR_PER_UPDATE = 8
MAX_EXPOSURE = 3
LAMBDA_ANCHOR = 0.25
LAMBDA_DISTILL = 0.10
SLOW_LR_GROUPS = ("encoder.", "graph_encoder.", "cmha")
LR_FAST = 3e-5
LR_SLOW_FACTOR = 0.1


def load_anchor_pool():
    """Returns anchor window tensors + labels + teacher probabilities.

    Pool = rows [ANCHOR_ROW_START, ANCHOR_ROW_START + ANCHOR_PER_EPISODE) of
    adaptation train episodes ANCHOR_SEEDS (S3-data-v1).  Anchor windows are
    fully causal (each window's lookback stays inside its own episode).
    """
    from train_ftmoe_protocol019_s3 import load_adaptation_blocks
    _, _, _, _, blocks = load_adaptation_blocks()
    parts = {key: [] for key in ("x", "graph_x", "schedule", "labels", "ids")}
    for seed in ANCHOR_SEEDS:
        block = blocks[str(seed)]
        for key in parts:
            parts[key].append(block[key][ANCHOR_ROW_START:ANCHOR_ROW_START + ANCHOR_PER_EPISODE])
    pool = {key: torch.cat(parts[key]) for key in parts}
    return pool


class S4Session(OnlineSessionV2):
    """OnlineSessionV2 with the S4 memory/loss/lr-group design."""

    def __init__(self, checkpoint, method, seed, replay, learning_rate, replay_seed,
                 normalization_v2, anchor_pool, anchor_teacher):
        super().__init__(checkpoint, method, seed, replay, learning_rate, replay_seed,
                         normalization_v2)
        if method not in ("A", "B", "C", "D"):
            raise ValueError(method)
        # S4 memory replaces the legacy 64-interval full-pass buffer.
        self.buffer = deque(maxlen=RECENT_MAXLEN)
        self.exposure = {}
        self.sample_rng = random.Random(seed * 7919 + replay_seed)
        self.anchor_pool = anchor_pool
        self.anchor_teacher = anchor_teacher
        self.anchor_indices = list(range(anchor_pool["x"].shape[0]))
        self.anchor_teacher_probs = self._cache_teacher_probs()
        self.optimizer = self.make_optimizer()

    @torch.no_grad()
    def _cache_teacher_probs(self):
        pool = self.anchor_pool
        teacher = self.anchor_teacher
        teacher.eval()
        dets, clss = [], []
        for start in range(0, pool["x"].shape[0], 64):
            out = teacher(pool["x"][start:start + 64], pool["schedule"][start:start + 64],
                          pool["graph_x"][start:start + 64],
                          graph_context={"creation_ids": pool["ids"][start:start + 64]})
            dets.append(out["detection_logits"].softmax(-1))
            clss.append(out["class_logits"].softmax(-1))
        return torch.cat(dets), torch.cat(clss)

    def make_optimizer(self):
        if self.method == "A":
            return None
        fast = []
        slow = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (slow if name.startswith(SLOW_LR_GROUPS) else fast).append(parameter)
        if not fast and not slow:
            return None
        base_lr = self.learning_rate if self.learning_rate else LR_FAST
        groups = []
        if fast:
            groups.append({"params": fast, "lr": base_lr})
        if slow:
            groups.append({"params": slow, "lr": base_lr * LR_SLOW_FACTOR})
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=1e-4)

    def _draw_recent(self, count):
        candidates = [i for i in self.buffer if self.exposure.get(i, 0) < MAX_EXPOSURE]
        if len(candidates) < count:
            count = len(candidates)
        drawn = self.sample_rng.sample(candidates, count) if count else []
        for i in drawn:
            self.exposure[i] = self.exposure.get(i, 0) + 1
        return drawn

    def _anchor_losses(self, out, teacher_det, teacher_cls, labels):
        anomaly = (labels > 0).long()
        det_ce = F.cross_entropy(out["detection_logits"].reshape(-1, 2), anomaly.reshape(-1),
                                 weight=torch.tensor([0.6, 2.0], dtype=out["detection_logits"].dtype))
        positive = labels > 0
        cls_ce = (F.cross_entropy(out["class_logits"][positive], labels[positive] - 1)
                  if positive.any() else out["detection_logits"].new_zeros(()))
        kl_det = F.kl_div(F.log_softmax(out["detection_logits"], -1), teacher_det,
                          reduction="none").sum(-1).mean()
        kl_cls = F.kl_div(F.log_softmax(out["class_logits"], -1), teacher_cls,
                          reduction="none").sum(-1).mean()
        return det_ce + cls_ce, (kl_det + kl_cls) / 2.0

    def update(self):
        started = time.perf_counter()
        self.model.eval()
        recent = self._draw_recent(RECENT_PER_UPDATE)
        anchor_count = ANCHOR_PER_UPDATE
        anchor_draw = self.sample_rng.sample(
            self.anchor_indices, min(anchor_count, len(self.anchor_indices)))
        total_loss = 0.
        components = {}
        if not recent and not anchor_draw:
            self.update_number += 1
            self.updates.append({"step": self.cursor, "update_number": self.update_number,
                                 "loss": 0.0, "batches": 0, "buffer_indices": [],
                                 "seconds": time.perf_counter() - started,
                                 "topology_event": None, "components": components})
            return
        if recent:
            windows = [self.replay.window(i) for i in recent]
            x = torch.stack([w[0] for w in windows])
            s = torch.stack([w[1] for w in windows])
            g = torch.stack([w[2] for w in windows])
            ids = torch.stack([w[3] for w in windows])
            y = torch.from_numpy(self.predictions["labels"][recent])
        if anchor_draw:
            pool = self.anchor_pool
            ax = pool["x"][anchor_draw]
            asch = pool["schedule"][anchor_draw]
            ag = pool["graph_x"][anchor_draw]
            aids = pool["ids"][anchor_draw]
            ay = pool["labels"][anchor_draw]
        self.optimizer.zero_grad(set_to_none=True)
        out_recent = None
        if recent:
            out_recent = self.model(x, s, g, graph_context={"creation_ids": ids})
            loss_online = loss_fn(self.model, out_recent, y, .7, .3, 0., .01, 2., .5)
        else:
            loss_online = None
        anchor_loss = None
        distill = None
        if anchor_draw:
            out_anchor = self.model(ax, asch, ag, graph_context={"creation_ids": aids})
            teacher_det, teacher_cls = (self.anchor_teacher_probs[0][anchor_draw],
                                        self.anchor_teacher_probs[1][anchor_draw])
            anchor_loss, distill = self._anchor_losses(out_anchor, teacher_det, teacher_cls, ay)
        loss = torch.zeros((), dtype=torch.float32)
        if loss_online is not None:
            loss = loss + loss_online
        if anchor_loss is not None:
            loss = loss + LAMBDA_ANCHOR * anchor_loss + LAMBDA_DISTILL * distill
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite online loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        self.optimizer.step()
        total_loss = float(loss.detach())
        components = {
            "recent_windows": len(recent), "anchor_windows": len(anchor_draw),
            "loss_online": float(loss_online.detach()) if loss_online is not None else None,
            "loss_anchor": float(anchor_loss.detach()) if anchor_loss is not None else None,
            "loss_distill": float(distill.detach()) if distill is not None else None,
        }
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
                             "loss": total_loss, "batches": 1,
                             "buffer_indices": list(recent), "anchor_indices": anchor_draw,
                             "seconds": time.perf_counter() - started,
                             "topology_event": event, "components": components})

    def save(self):
        state = super().save()
        state.update({"exposure": dict(self.exposure), "sample_rng": self.sample_rng.getstate()})
        return state

    def restore(self, state):
        super().restore(state)
        self.buffer = deque(state["buffer"], maxlen=RECENT_MAXLEN)
        self.exposure = dict(state.get("exposure", {}))
        self.sample_rng = random.Random()
        self.sample_rng.setstate(state["sample_rng"])
