"""Protocol 019 S5 — dynamic expert v2 (plan §10).

Replaces the legacy OnlineEAGate.adapt() heuristic ("any unmatched -> add a
zero-initialized expert; zero activation -> delete") with:

- EMA-gated addition (plan §10.1): only when routing_samples >= 128,
  unmatched_ratio EMA >= 0.05 and the last 3 routing windows were all
  non-trivial, with active experts < expert_max(8).
- Extreme-scale OOD guard (plan §10.2): normalization v2 already caps
  |normalized features| <= ~3 on the registered streams (alarm <= 50), so
  unmatched routing states cannot be normalization-explosion artifacts;
  unmatched_vectors are still kept for offline audit.
- Safe birth (plan §10.3): k=1 centroid of stored unmatched vectors; nearest
  existing expert by key cosine is cloned (deepcopy) — birth is output-
  continuous; the new key is the normalized centroid; the threshold is
  initialized to the 70% quantile of unmatched-to-centroid similarity
  (tanh domain); routing ramp 0 -> .25 -> .5 -> .75 -> 1.0 over the first
  four updates after birth.
- Retire/reactivate (plan §10.5): experts dormant (moved out of the forward,
  parameters retained, requires_grad off) instead of deleted when age >= 500
  intervals, activation EMA < 0.005 for 5 consecutive adaptation windows;
  a new unmatched centroid matching a dormant key at cosine > 0.90
  reactivates the old expert (same id) instead of adding one.

The forward pass, statistic recording and last_routing contract are
identical to OnlineEAGate so that OnlineFTMoE plumbing (hooks, adapt,
topology save/restore, state dicts) keeps working.
"""
from collections import deque
from copy import deepcopy
import math

import torch
from torch import nn
import torch.nn.functional as F

from .ftmoe_ablation import AblationConfig
from .ftmoe_end_to_end import FTMoEEndToEnd
from .ftmoe_online import OnlineFTMoE
from .ftmoe_online_s4 import S4Session

EXPERT_MAX = 8
CAP_ACTIVE = 4
ADD_MIN_SAMPLES = 128
ADD_EMA_MIN = 0.05
ADD_CONSECUTIVE_WINDOWS = 3
UNMATCHED_VECTOR_CAP = 256
RAMP_STEPS = 4                 # 0 -> .25 -> .5 -> .75 -> 1.0
DORMANT_MIN_AGE = 500          # intervals
DORMANT_ACTIVATION_EMA_MAX = 0.005
DORMANT_CONSECUTIVE_WINDOWS = 5
REACTIVATE_COSINE = 0.90
BIRTH_THRESHOLD_QUANTILE = 0.70


def ramp_value(steps_done):
    return min(1.0, 0.25 * max(steps_done, 0))


class OnlineEAGateV2(nn.Module):
    def __init__(self, source, seed):
        super().__init__()
        self.cfg = source.cfg
        for name in ("resource_proj", "output_norm", "detection_adapter", "class_adapter"):
            setattr(self, name, deepcopy(getattr(source, name)))
        self.residual_gain = nn.Parameter(source.residual_gain.detach().clone())
        self.register_buffer("temperature", source.temperature.detach().clone())
        self.experts = nn.ModuleDict()
        self.key_rows = nn.ParameterDict()
        self.threshold_rows = nn.ParameterDict()
        self.expert_detection_heads = nn.ModuleDict()
        self.expert_class_heads = nn.ModuleDict()
        self.dormant_experts = nn.ModuleDict()
        self.dormant_key_rows = nn.ParameterDict()
        self.dormant_threshold_rows = nn.ParameterDict()
        self.dormant_detection_heads = nn.ModuleDict()
        self.dormant_class_heads = nn.ModuleDict()
        self.ids = [str(i) for i in range(len(source.experts))]
        for i, key in enumerate(self.ids):
            self.experts[key] = deepcopy(source.experts[i])
            self.key_rows[key] = nn.Parameter(source.keys[i].detach().clone())
            self.threshold_rows[key] = nn.Parameter(source.thresholds[i].detach().clone())
            self.expert_detection_heads[key] = deepcopy(source.expert_detection_heads[i])
            self.expert_class_heads[key] = deepcopy(source.expert_class_heads[i])
        self.next_id = len(self.ids)
        self.generator = torch.Generator().manual_seed(seed + 40000)
        self.record_enabled = False
        self.last_routing = {}
        # v2 dynamics state
        self.ramp_steps_done = {key: RAMP_STEPS for key in self.ids}
        self.expert_age = {key: 0 for key in self.ids}          # intervals
        self.activation_ema = {key: 0.0 for key in self.ids}
        self.low_activation_windows = {key: 0 for key in self.ids}
        self.unmatched_ratio_ema = 0.0
        self.unmatched_ratio_history = deque(maxlen=8)
        self.reset_statistics()

    # ------------------------------------------------------------------ stats
    def reset_statistics(self):
        self.activation_counts = {key: 0 for key in self.ids}
        self.routing_samples = 0
        self.unmatched_count = 0
        self.unmatched_vectors = deque(maxlen=UNMATCHED_VECTOR_CAP)

    def set_temperature(self, value):
        self.temperature.fill_(max(float(value), .05))

    # ---------------------------------------------------------------- forward
    def forward(self, x, resources):
        state = x + self.resource_proj(resources)
        keys = torch.stack([self.key_rows[key] for key in self.ids])
        thresholds = torch.stack([self.threshold_rows[key] for key in self.ids]).tanh()
        score = F.normalize(state, dim=-1) @ F.normalize(keys, dim=-1).T
        soft_mask = torch.sigmoid((score - thresholds) / self.temperature.clamp_min(.05))
        eligible = score >= thresholds
        unmatched = ~eligible.any(dim=-1)
        hard_mask = eligible.to(score.dtype)
        # Cap slots are reserved for experts that are actually participating:
        # ramp-0 (just born) experts must not displace a mature expert from the
        # top-4 selection, which would break birth output continuity.
        ramp_vector = torch.tensor([ramp_value(self.ramp_steps_done[key]) for key in self.ids],
                                   dtype=score.dtype, device=score.device)
        cap_eligible = ramp_vector > 0
        cap_scores = score.masked_fill(~cap_eligible, float("-inf"))
        cap_k = min(CAP_ACTIVE, max(1, int(cap_eligible.sum())))
        top_indices = cap_scores.topk(cap_k, dim=-1).indices
        cap_mask = torch.zeros_like(hard_mask).scatter_(-1, top_indices, 1.)
        hard_mask = hard_mask * cap_mask
        fallback = torch.zeros_like(hard_mask).scatter_(
            -1, score.argmax(-1, keepdim=True), 1.)
        hard_mask = torch.where(unmatched.unsqueeze(-1), fallback, hard_mask)
        mask = hard_mask + soft_mask - soft_mask.detach()
        weight = torch.softmax(score / self.temperature.clamp_min(.05), dim=-1) * mask
        # Routing ramp (birth safety): young experts contribute progressively.
        weight = weight * ramp_vector
        row_sum = weight.sum(-1, keepdim=True)
        empty = row_sum < 1e-6
        if bool(empty.any()):
            onehot = torch.zeros_like(weight).scatter_(
                -1, score.argmax(-1, keepdim=True), 1.0)
            weight = torch.where(empty, onehot, weight / row_sum.clamp_min(1e-6))
        else:
            weight = weight / row_sum.clamp_min(1e-6)
        outputs = torch.stack([self.experts[key](state) for key in self.ids], dim=-2)
        mixed = self.output_norm((weight.unsqueeze(-1) * outputs).sum(-2))
        detection = torch.stack(
            [self.expert_detection_heads[key](resources) for key in self.ids], dim=-2)
        classes = torch.stack(
            [self.expert_class_heads[key](resources) for key in self.ids], dim=-2)
        self.last_routing = {
            "expert_count": len(self.ids), "mean_active": float(hard_mask.sum(-1).mean().detach()),
            "unmatched_ratio": float(unmatched.float().mean()),
        }
        if self.record_enabled:
            with torch.no_grad():
                counts = hard_mask.reshape(-1, len(self.ids)).sum(0).tolist()
                for key, count in zip(self.ids, counts):
                    self.activation_counts[key] += int(count)
                self.routing_samples += unmatched.numel()
                self.unmatched_count += int(unmatched.sum())
                for vector in state[unmatched].detach().cpu():
                    self.unmatched_vectors.append(vector.clone())
        return (self.residual_gain * mixed, weight, hard_mask.sum(-1),
                self.detection_adapter(mixed) + (weight.unsqueeze(-1) * detection).sum(-2),
                self.class_adapter(mixed) + (weight.unsqueeze(-1) * classes).sum(-2))

    # ---------------------------------------------------------------- topology
    def _blank_expert(self, key, key_vector=None):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            expert = nn.Sequential(nn.Linear(self.cfg.hidden, self.cfg.hidden * 2),
                                   nn.GELU(), nn.Linear(self.cfg.hidden * 2, self.cfg.hidden))
            det, cls = nn.Linear(3, 2), nn.Linear(3, self.cfg.classes)
            if key_vector is None or float(key_vector.norm()) == 0:
                key_vector = torch.randn(self.cfg.hidden)
            self.generator.set_state(torch.get_rng_state())
        for head in (expert[-1], det, cls):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        return (expert, det, cls,
                nn.Parameter(F.normalize(key_vector.float(), dim=0).clone()),
                nn.Parameter(torch.tensor(0.)))

    def ramp_step(self):
        """Advance the routing ramp by one update (young experts only)."""
        for key in self.ids:
            if self.ramp_steps_done.get(key, RAMP_STEPS) < RAMP_STEPS:
                self.ramp_steps_done[key] += 1

    def _insert_clone(self, key, parent_key, centroid):
        expert, det, cls = (deepcopy(self.experts[parent_key]),
                            deepcopy(self.expert_detection_heads[parent_key]),
                            deepcopy(self.expert_class_heads[parent_key]))
        self.experts[key] = expert
        self.expert_detection_heads[key] = det
        self.expert_class_heads[key] = cls
        self.key_rows[key] = nn.Parameter(F.normalize(centroid.float(), dim=0).clone())
        if len(self.unmatched_vectors) > 1:
            vectors = torch.stack(list(self.unmatched_vectors))
            similarities = F.normalize(vectors, dim=-1) @ F.normalize(centroid, dim=-1)
            quantile = float(torch.quantile(similarities, BIRTH_THRESHOLD_QUANTILE).clamp(-0.999, 0.999))
        else:
            quantile = 0.0
        raw = float(torch.atanh(torch.tensor(min(max(quantile, -0.999), 0.999)))) \
            if -0.999 < quantile < 0.999 else (3.0 if quantile >= 0.999 else -3.0)
        self.threshold_rows[key] = nn.Parameter(torch.tensor(raw))
        self.ramp_steps_done[key] = 0
        self.expert_age[key] = 0
        self.activation_ema[key] = 0.0
        self.low_activation_windows[key] = 0

    def _retire(self, key):
        for active, dormant in ((self.experts, self.dormant_experts),
                                (self.expert_detection_heads, self.dormant_detection_heads),
                                (self.expert_class_heads, self.dormant_class_heads)):
            dormant[key] = active.pop(key)
        for active, dormant in ((self.key_rows, self.dormant_key_rows),
                                (self.threshold_rows, self.dormant_threshold_rows)):
            dormant[key] = active.pop(key)
            dormant[key].requires_grad_(False)
        self.ids.remove(key)

    def _reactivate(self, key):
        for dormant, active in ((self.dormant_experts, self.experts),
                                (self.dormant_detection_heads, self.expert_detection_heads),
                                (self.dormant_class_heads, self.expert_class_heads)):
            active[key] = dormant.pop(key)
        for dormant, active in ((self.dormant_key_rows, self.key_rows),
                                (self.dormant_threshold_rows, self.threshold_rows)):
            active[key] = dormant.pop(key)
            active[key].requires_grad_(True)
        self.ids.append(key)

    def freeze_dormant(self):
        for name in ("dormant_experts", "dormant_detection_heads",
                     "dormant_class_heads", "dormant_key_rows", "dormant_threshold_rows"):
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(False)

    # ------------------------------------------------------------------- adapt
    def adapt(self):
        """One routing-window evaluation (called every 10th update)."""
        before = list(self.ids)
        event = {"before": before, "samples": self.routing_samples,
                 "activation_counts": dict(self.activation_counts),
                 "unmatched": self.unmatched_count, "removed": [], "added": [],
                 "reactivated": [], "dormant": [], "ema": self.unmatched_ratio_ema,
                 "decisions": []}
        samples = self.routing_samples
        if samples:
            ratio = self.unmatched_count / max(samples, 1)
            self.unmatched_ratio_ema = 0.9 * self.unmatched_ratio_ema + 0.1 * ratio
            self.unmatched_ratio_history.append(self.unmatched_ratio_ema)
            window_intervals = max(samples // 16, 1)   # 16 hosts per interval row
            for key in self.ids:
                self.expert_age[key] = self.expert_age.get(key, 0) + window_intervals
                activation = self.activation_counts.get(key, 0) / max(samples, 1)
                ema = self.activation_ema.get(key, 0.0)
                ema = 0.9 * ema + 0.1 * activation
                self.activation_ema[key] = ema
                if ema < DORMANT_ACTIVATION_EMA_MAX:
                    self.low_activation_windows[key] = self.low_activation_windows.get(key, 0) + 1
                else:
                    self.low_activation_windows[key] = 0
            # retire experts (never below expert_min = 2)
            for key in list(self.ids):
                if (len(self.ids) > 2 and self.expert_age[key] >= DORMANT_MIN_AGE and
                        self.activation_ema[key] < DORMANT_ACTIVATION_EMA_MAX and
                        self.low_activation_windows[key] >= DORMANT_CONSECUTIVE_WINDOWS):
                    self._retire(key)
                    event["dormant"].append(key)
            # reactivation or birth
            reactivated = False
            if self.unmatched_count and len(self.unmatched_vectors):
                centroid = torch.stack(list(self.unmatched_vectors)).mean(0)
                if self.dormant_key_rows:
                    dormant_keys = torch.stack(
                        [self.dormant_key_rows[key] for key in list(self.dormant_key_rows)])
                    cosine = (F.normalize(centroid, dim=0) @
                              F.normalize(dormant_keys, dim=-1).T)
                    best = int(cosine.argmax())
                    if float(cosine[best]) > REACTIVATE_COSINE:
                        key = list(self.dormant_key_rows)[best]
                        self._reactivate(key)
                        self.activation_ema[key] = 0.0
                        self.low_activation_windows[key] = 0
                        event["reactivated"].append(key)
                        event["decisions"].append("reactivate")
                        reactivated = True
            birth_ok = (not reactivated and len(self.ids) < EXPERT_MAX and
                        samples >= ADD_MIN_SAMPLES and
                        self.unmatched_ratio_ema >= ADD_EMA_MIN and
                        len(self.unmatched_ratio_history) >= ADD_CONSECUTIVE_WINDOWS and
                        all(window_value >= ADD_EMA_MIN for window_value in
                            list(self.unmatched_ratio_history)[-ADD_CONSECUTIVE_WINDOWS:]))
            if birth_ok and self.unmatched_count and len(self.unmatched_vectors):
                centroid = torch.stack(list(self.unmatched_vectors)).mean(0)
                parent = max(self.ids, key=lambda k: float(
                    F.cosine_similarity(F.normalize(centroid, dim=0),
                                        F.normalize(self.key_rows[k], dim=0), dim=0)))
                key = str(self.next_id)
                self.next_id += 1
                self._insert_clone(key, parent, centroid)
                self.ids.append(key)
                event["added"].append(key)
                event["parent"] = parent
                event["decisions"].append("birth_clone_parent")
            elif not birth_ok and self.unmatched_count:
                event["decisions"].append("birth_conditions_not_met")
        event["after"] = list(self.ids)
        self.reset_statistics()
        return event

    def topology_state(self):
        return {"ids": list(self.ids), "next_id": self.next_id,
                "generator": self.generator.get_state(),
                "activation_counts": dict(self.activation_counts),
                "routing_samples": self.routing_samples,
                "unmatched_count": self.unmatched_count,
                "unmatched_vectors": list(self.unmatched_vectors),
                "ramp_steps_done": dict(self.ramp_steps_done),
                "expert_age": dict(self.expert_age),
                "activation_ema": dict(self.activation_ema),
                "low_activation_windows": dict(self.low_activation_windows),
                "unmatched_ratio_ema": self.unmatched_ratio_ema,
                "unmatched_ratio_history": list(self.unmatched_ratio_history),
                "dormant_ids": [key for key in self.dormant_experts]}

    def restore_topology(self, state):
        for name in ("experts", "expert_detection_heads", "expert_class_heads",
                     "dormant_experts", "dormant_detection_heads", "dormant_class_heads"):
            setattr(self, name, nn.ModuleDict())
        for name in ("key_rows", "threshold_rows", "dormant_key_rows",
                     "dormant_threshold_rows"):
            setattr(self, name, nn.ParameterDict())
        self.ids = list(state["ids"])
        for key in self.ids:
            expert, det, cls, key_row, threshold = self._blank_expert(key)
            self.experts[key] = expert
            self.expert_detection_heads[key] = det
            self.expert_class_heads[key] = cls
            self.key_rows[key] = key_row
            self.threshold_rows[key] = threshold
        for key in state.get("dormant_ids", []):
            expert, det, cls, key_row, threshold = self._blank_expert(key)
            self.dormant_experts[key] = expert
            self.dormant_detection_heads[key] = det
            self.dormant_class_heads[key] = cls
            self.dormant_key_rows[key] = key_row
            self.dormant_threshold_rows[key] = threshold
        self.next_id = state["next_id"]
        self.generator.set_state(state["generator"])
        self.activation_counts = dict(state["activation_counts"])
        self.routing_samples, self.unmatched_count = state["routing_samples"], state["unmatched_count"]
        self.unmatched_vectors = deque(state["unmatched_vectors"], maxlen=UNMATCHED_VECTOR_CAP)
        self.ramp_steps_done = dict(state.get("ramp_steps_done",
                                              {key: RAMP_STEPS for key in self.ids}))
        self.expert_age = dict(state.get("expert_age", {key: 0 for key in self.ids}))
        self.activation_ema = dict(state.get("activation_ema", {key: 0.0 for key in self.ids}))
        self.low_activation_windows = dict(state.get("low_activation_windows",
                                                     {key: 0 for key in self.ids}))
        self.unmatched_ratio_ema = float(state.get("unmatched_ratio_ema", 0.0))
        self.unmatched_ratio_history = deque(
            state.get("unmatched_ratio_history", []), maxlen=8)
        self.freeze_dormant()


class OnlineFTMoEV2(OnlineFTMoE):
    """OnlineFTMoE with the v2 dynamic gate (identical freeze sets and hooks)."""

    def __init__(self, checkpoint, method, seed):
        if method not in "ABCD" or len(method) != 1:
            raise ValueError(method)
        torch.manual_seed(seed)
        FTMoEEndToEnd.__init__(self, "v4", AblationConfig(
            experts=4, moe_residual_initial=0., eagate_residual_initial=.5,
            graph_residual_initial=0., cmha_residual_initial=0.))
        self.load_state_dict(checkpoint["model"], strict=True)
        self.eagate = OnlineEAGateV2(self.eagate, seed)
        self.eagate.register_forward_hook(self._routing_hook("eagate"))
        self.method = method
        self.set_trainability()
        self.eval()

    def adapt(self, optimizer):
        event = self.eagate.adapt()
        self.set_trainability()
        self.eagate.freeze_dormant()
        live = [p for p in self.parameters() if p.requires_grad]
        live_set = set(live)
        for parameter in list(optimizer.state):
            if parameter not in live_set:
                del optimizer.state[parameter]
        optimizer.param_groups[0]["params"] = live
        self.eval()
        return event


class S5Session(S4Session):
    """S4Session over OnlineFTMoEV2 (dynamic expert v2); ramp advances once
    per real update for young experts of method D."""

    model_class = OnlineFTMoEV2

    def update(self):
        super().update()
        if self.method == "D" and self.updates and self.updates[-1].get("batches", 0) > 0:
            self.model.eagate.ramp_step()
