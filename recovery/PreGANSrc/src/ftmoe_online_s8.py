"""Protocol 020 development: causal, validated dynamic expert integration.

The source checkpoint remains untouched. Birth candidates are cloned and
trained in shadow; only future matured windows validate them. The main
optimizer never trains a shadow candidate. Activation starts at zero weight.
"""
from collections import deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ftmoe_ablation import AblationConfig
from .ftmoe_end_to_end import FTMoEEndToEnd
from .ftmoe_online import OnlineFTMoE, tensor_hash
from .ftmoe_online_s5 import OnlineEAGateV2
from .ftmoe_online_s7 import S7Session, s7_loss
from .ftmoe_dynamic_expert_v3 import V3TriggerState


class OnlineEAGateV3(OnlineEAGateV2):
    """V2 storage with smooth routing and an independent novelty observer."""

    def __init__(self, source, seed):
        super().__init__(source, seed)
        self.shadow_id = None
        self.preview_id = None
        self.calibration = None
        self.last_observation = None
        self.context_capacity = None
        self.capacity_aware = False
        self.capacity_projection = nn.Linear(3, self.cfg.hidden, bias=False)
        nn.init.zeros_(self.capacity_projection.weight)
        self.weight_mass = {key: 0.0 for key in self.ids}

    def reset_statistics(self):
        super().reset_statistics()
        self.weight_mass = {key: 0.0 for key in self.ids}

    def forward(self, x, resources):
        state = x + self.resource_proj(resources)
        if self.capacity_aware and self.context_capacity is not None:
            state = state + self.capacity_projection(
                torch.log(self.context_capacity.clamp_min(1e-6)))
        ids = list(self.ids)
        if self.preview_id is not None and self.preview_id not in ids:
            ids.append(self.preview_id)
        keys = torch.stack([self.key_rows[key] for key in ids])
        thresholds = torch.stack([self.threshold_rows[key] for key in ids]).tanh()
        score = F.normalize(state, dim=-1) @ F.normalize(keys, dim=-1).T
        eligible = score >= thresholds
        ramp = score.new_tensor([
            1.0 if key == self.preview_id else
            min(1.0, self.ramp_steps_done[key] / 10.0) for key in ids])
        # Original experts are fully ramped; inserted ones require 10 updates.
        participating = ramp > 0
        cap_score = score.masked_fill(~participating, float("-inf"))
        # Young experts enter smoothly without suddenly evicting a top-4 expert.
        mature = ramp >= 1
        mature_score = score.masked_fill(~mature, float("-inf"))
        count = min(4, int(mature.sum()))
        cap = torch.zeros_like(score)
        if count:
            cap.scatter_(-1, mature_score.topk(count, dim=-1).indices, 1.0)
        cap = torch.maximum(cap, ((ramp > 0) & (ramp < 1)).to(score.dtype))
        hard = eligible.to(score.dtype) * cap
        unmatched = hard.sum(-1) == 0
        fallback = torch.zeros_like(score).scatter_(
            -1, cap_score.argmax(-1, keepdim=True), 1.0)
        hard = torch.where(unmatched.unsqueeze(-1), fallback, hard)
        soft = torch.sigmoid((score - thresholds) / self.temperature.clamp_min(.05))
        mask = hard + soft - soft.detach()
        weights = torch.softmax(score / self.temperature.clamp_min(.05), -1) * mask * ramp
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        outputs = torch.stack([self.experts[key](state) for key in ids], -2)
        mixed = self.output_norm((weights.unsqueeze(-1) * outputs).sum(-2))
        detection = torch.stack([self.expert_detection_heads[key](resources) for key in ids], -2)
        classes = torch.stack([self.expert_class_heads[key](resources) for key in ids], -2)
        self.last_routing = {"expert_count": len(self.ids),
                             "mean_active": float(hard.sum(-1).mean().detach()),
                             "unmatched_ratio": float(unmatched.float().mean())}
        if self.record_enabled:
            with torch.no_grad():
                probability = torch.softmax(score / self.temperature.clamp_min(.05), -1)
                entropy = -(probability * probability.clamp_min(1e-9).log()).sum(-1) / math.log(len(ids))
                top = score.topk(2, dim=-1).values
                margin = top[..., 0] - top[..., 1]
                self.last_observation = {"entropy": entropy.detach().cpu(),
                                         "margin": margin.detach().cpu(),
                                         "vectors": state.detach().cpu()}
                self.routing_samples += unmatched.numel()
                self.unmatched_count += int(unmatched.sum())
                for j, key in enumerate(ids):
                    self.activation_counts[key] = self.activation_counts.get(key, 0) + int(hard[..., j].sum())
                    self.weight_mass[key] = self.weight_mass.get(key, 0.) + float(weights[..., j].sum())
        return (self.residual_gain * mixed, weights, hard.sum(-1),
                self.detection_adapter(mixed) + (weights.unsqueeze(-1) * detection).sum(-2),
                self.class_adapter(mixed) + (weights.unsqueeze(-1) * classes).sum(-2))

    def ramp_step(self):
        for key in self.ids:
            self.ramp_steps_done[key] = min(10, self.ramp_steps_done[key] + 1)

    def create_shadow(self, centroid):
        parent = max(self.ids, key=lambda k: float(F.cosine_similarity(
            centroid, self.key_rows[k].detach(), dim=0)))
        key = str(self.next_id)
        self.next_id += 1
        self._insert_clone(key, parent, centroid)
        # Match the parent's acceptance initially; novelty is not eligibility.
        self.threshold_rows[key].data.copy_(self.threshold_rows[parent].detach())
        self.shadow_id = key
        return key, parent

    def shadow_parameters(self):
        key = self.shadow_id
        if key is None:
            return []
        return [*self.experts[key].parameters(),
                *self.expert_detection_heads[key].parameters(),
                *self.expert_class_heads[key].parameters(),
                self.key_rows[key], self.threshold_rows[key]]

    @contextmanager
    def preview(self):
        previous = self.preview_id
        self.preview_id = self.shadow_id
        try:
            yield
        finally:
            self.preview_id = previous

    def activate_shadow(self):
        key = self.shadow_id
        self.ids.append(key)
        self.shadow_id = None
        self.ramp_steps_done[key] = 0
        self.reset_statistics()
        return key

    def discard_shadow(self):
        key = self.shadow_id
        for name in ("experts", "expert_detection_heads", "expert_class_heads",
                     "key_rows", "threshold_rows"):
            getattr(self, name).pop(key)
        for name in ("ramp_steps_done", "expert_age", "activation_ema", "low_activation_windows"):
            getattr(self, name).pop(key, None)
        self.shadow_id = None

    def topology_state(self):
        state = super().topology_state()
        state.update({"shadow_id": self.shadow_id,
                      "weight_mass": dict(self.weight_mass),
                      "capacity_aware": self.capacity_aware})
        return state

    def restore_topology(self, state):
        super().restore_topology(state)
        self.shadow_id = state.get("shadow_id")
        if self.shadow_id is not None:
            key = self.shadow_id
            expert, det, cls, key_row, threshold = self._blank_expert(key)
            self.experts[key], self.expert_detection_heads[key] = expert, det
            self.expert_class_heads[key] = cls
            self.key_rows[key], self.threshold_rows[key] = key_row, threshold
        self.weight_mass = dict(state.get("weight_mass", {}))
        self.capacity_aware = state.get("capacity_aware", False)
        self.generator.set_state(state["generator"])


class OnlineFTMoEV3(OnlineFTMoE):
    def __init__(self, checkpoint, method, seed):
        torch.manual_seed(seed)
        FTMoEEndToEnd.__init__(self, "v4", AblationConfig(
            experts=4, moe_residual_initial=0., eagate_residual_initial=.5,
            graph_residual_initial=0., cmha_residual_initial=0.))
        self.load_state_dict(checkpoint["model"], strict=True)
        self.eagate = OnlineEAGateV3(self.eagate, seed)
        self.eagate.ramp_steps_done = {key: 10 for key in self.eagate.ids}
        self.eagate.register_forward_hook(self._routing_hook("eagate"))
        self.method = method
        self.set_trainability()
        self.eval()

    def set_trainability(self):
        super().set_trainability()
        self.eagate.freeze_dormant()

    def forward(self, x, schedule=None, graph=None, graph_context=None):
        self.eagate.context_capacity = (graph_context["capacities"][:, -1]
            if graph_context is not None and "capacities" in graph_context else None)
        try:
            return super().forward(x, schedule, graph, graph_context=graph_context)
        finally:
            self.eagate.context_capacity = None

    def frozen_hash(self):
        # Dormant experts remain dynamic state, not part of the immutable backbone.
        return tensor_hash((name, p) for name, p in self.named_parameters()
                           if not name.startswith(("moe.", "eagate.")))


class S8Session(S7Session):
    model_class = OnlineFTMoEV3
    supports_dynamic = True

    def __init__(self, *args, dynamic_config=None, **kwargs):
        self.dynamic_config = dict(dynamic_config or {})
        self.validation_protocol = self.dynamic_config.get(
            "validation_protocol", "legacy_v3")
        if self.validation_protocol not in ("legacy_v3", "prequential_v1"):
            raise ValueError("Unknown S8 validation_protocol: %s" %
                             self.validation_protocol)
        self.shadow_optimizer = None
        super().__init__(*args, **kwargs)
        self.model.eagate.capacity_aware = self.dynamic_config.get("capacity_aware", False)
        self.trigger = V3TriggerState()
        calibration = self.dynamic_config["calibration"]
        self.trigger.register_baselines(calibration["novelty_threshold"],
                                        calibration["loss_baseline"])
        # Start EMAs at the calibrated stationary mean, not artificial zeros.
        self.trigger.novelty_ema = calibration["novelty_mean"]
        self.trigger.loss_ema = calibration["loss_baseline"]
        self.pending_observations = {}
        self.matured_observations = deque(maxlen=10)
        self.candidate_vectors = deque(maxlen=256)
        self.dynamic_events = []
        self.shadow_started = None
        self.shadow_training_steps = 0
        self.shadow_validation = []
        self.cooldown_until = 0
        # R0's causal ledger is intentionally session-local and step-boundary
        # serializable.  The legacy path leaves these fields empty and retains
        # the historical validation behavior below.
        self.topology_version = 0
        self.ramp_version = 0
        self.pending_predictions = {}
        self.qualification_records = []
        self.qualification_by_index = {}
        self.matured_indices = set()
        self.shadow_validation_indices = set()
        self.causal_audit = []
        self.audit_violation_count = 0
        self.diagnostic_cost = {
            "teacher_forward_count": 0,
            "teacher_forward_seconds": 0.0,
            "candidate_forward_count": 0,
            "candidate_forward_seconds": 0.0,
        }

    def make_optimizer(self):
        optimizer = super().make_optimizer()
        if optimizer is not None:
            excluded = set(self.model.eagate.shadow_parameters())
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p not in excluded]
        return optimizer

    def sync_optimizer(self):
        """Rebuild group membership while retaining surviving Adam moments."""
        previous = self.optimizer
        replacement = self.make_optimizer()
        if replacement is None:
            self.optimizer = None
            return
        for group in replacement.param_groups:
            for parameter in group["params"]:
                if previous is not None and parameter in previous.state:
                    replacement.state[parameter] = previous.state[parameter]
                elif self.shadow_optimizer is not None and parameter in self.shadow_optimizer.state:
                    replacement.state[parameter] = self.shadow_optimizer.state[parameter]
        self.optimizer = replacement

    def step(self, prediction_sink=None):
        # S7 invokes its sink after deployment prediction and before reading
        # the current raw label.  In the revised protocol this is the only
        # point at which the deployment score may be paired with the input;
        # teacher/candidate diagnostics are ordinary, observation-free
        # forwards over the same already-read window.
        index = self.cursor
        window = self._window(index) if self.validation_protocol == "prequential_v1" else None

        def sink(row):
            if self.validation_protocol == "prequential_v1":
                record = self._capture_prediction_record(index, row, window)
                row = dict(row)
                row["r0_prediction"] = self._audit_view(record)
            else:
                # Preserve the historical S8 ledger for legacy reproduction.
                self.pending_observations[self.cursor] = deepcopy(
                    self.model.eagate.last_observation)
            if prediction_sink is not None:
                prediction_sink(row)
        return super().step(sink)

    @staticmethod
    def _clone_state_value(value):
        """Clone observation values without retaining autograd references."""
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: S8Session._clone_state_value(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(S8Session._clone_state_value(item)
                               for item in value)
        return deepcopy(value)

    @contextmanager
    def _preserve_forward_state(self, model):
        """Run a diagnostic forward without changing observable model state.

        EAGate writes ``last_routing`` on every forward and the end-to-end
        hook writes ``routing_outputs``.  ``record_enabled=False`` prevents
        counter updates, while this context restores the remaining diagnostic
        fields, preview selection, capacity context and every module's
        training flag.  The same helper is used for the frozen teacher and
        the candidate preview.
        """
        modules = tuple(model.modules())
        training = [module.training for module in modules]
        gate = getattr(model, "eagate", None)
        saved = {
            "routing_outputs": self._clone_state_value(
                getattr(model, "routing_outputs", {})),
            "last_routing": self._clone_state_value(
                getattr(gate, "last_routing", None)) if gate is not None else None,
            "last_observation": self._clone_state_value(
                getattr(gate, "last_observation", None)) if gate is not None else None,
            "preview_id": getattr(gate, "preview_id", None) if gate is not None else None,
            "record_enabled": getattr(gate, "record_enabled", False) if gate is not None else None,
            "context_capacity": self._clone_state_value(
                getattr(gate, "context_capacity", None)) if gate is not None else None,
        }
        try:
            model.eval()
            if gate is not None and hasattr(gate, "record_enabled"):
                gate.record_enabled = False
            yield
        finally:
            for module, was_training in zip(modules, training):
                module.training = was_training
            if hasattr(model, "routing_outputs"):
                model.routing_outputs = saved["routing_outputs"]
            if gate is not None:
                if hasattr(gate, "last_routing"):
                    gate.last_routing = saved["last_routing"]
                if hasattr(gate, "last_observation"):
                    gate.last_observation = saved["last_observation"]
                if hasattr(gate, "preview_id"):
                    gate.preview_id = saved["preview_id"]
                if hasattr(gate, "record_enabled"):
                    gate.record_enabled = saved["record_enabled"]
                if hasattr(gate, "context_capacity"):
                    gate.context_capacity = saved["context_capacity"]

    @staticmethod
    def _hash_window(window):
        digest = hashlib.sha256()
        for value in window:
            array = (value.detach().cpu().contiguous().numpy()
                     if isinstance(value, torch.Tensor)
                     else np.ascontiguousarray(value))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _window_metadata(self, window):
        ids = window[3].detach().cpu().numpy()
        before = window[4].detach().cpu().numpy()
        return {
            "input_hash": self._hash_window(window),
            "creation_ids": ids[-1].astype(np.int64).tolist(),
            "creation_ids_hash": hashlib.sha256(
                np.ascontiguousarray(ids).tobytes()).hexdigest(),
            "before_placement_hash": hashlib.sha256(
                np.ascontiguousarray(before).tobytes()).hexdigest(),
        }

    def _diagnostic_output(self, model, window, kind, candidate=False):
        x, schedule, graph, ids, before, caps = window
        started = time.perf_counter()
        gate = getattr(model, "eagate", None)
        if candidate and (gate is None or gate.shadow_id is None):
            raise RuntimeError("Candidate diagnostic requested without a shadow")
        preview_context = (gate.preview() if candidate else nullcontext())
        with self._preserve_forward_state(model):
            with preview_context:
                with torch.no_grad():
                    output = model(
                        x[None], schedule[None], graph[None],
                        graph_context={"creation_ids": ids[None],
                                       "before_placement": before[None],
                                       "capacities": caps[None]})
                    probability = output["detection_logits"].softmax(-1)[0, :, 1]
                    classes = output["class_logits"].softmax(-1)[0]
                    probability = probability.detach().cpu().numpy().copy()
                    classes = classes.detach().cpu().numpy().copy()
                    if (not np.isfinite(probability).all() or
                            not np.isfinite(classes).all() or
                            (probability < 0.).any() or (probability > 1.).any() or
                            (classes < 0.).any() or (classes > 1.).any() or
                            not np.allclose(classes.sum(-1), 1., atol=1e-5,
                                             rtol=1e-5)):
                        raise RuntimeError(
                            "Nonfinite or invalid %s diagnostic prediction" % kind)
        elapsed = time.perf_counter() - started
        count_key = "%s_forward_count" % kind
        seconds_key = "%s_forward_seconds" % kind
        self.diagnostic_cost[count_key] += 1
        self.diagnostic_cost[seconds_key] += elapsed
        return {"probability": probability.tolist(),
                "class_probability": classes.tolist(),
                "seconds": elapsed}

    @staticmethod
    def _validate_score(score, kind):
        probability = np.asarray(score["probability"], dtype=np.float64)
        classes = np.asarray(score["class_probability"], dtype=np.float64)
        if (not np.isfinite(probability).all() or
                not np.isfinite(classes).all() or
                (probability < 0.).any() or (probability > 1.).any() or
                (classes < 0.).any() or (classes > 1.).any() or
                classes.ndim != 2 or not np.allclose(classes.sum(-1), 1.,
                                                      atol=1e-5, rtol=1e-5)):
            raise RuntimeError("Nonfinite or invalid %s prediction" % kind)

    def _capture_prediction_record(self, index, row, window):
        observation = deepcopy(self.model.eagate.last_observation)
        if observation is None:
            raise RuntimeError("S8 deployment prediction did not record routing observation")
        topology = self._topology_metadata()
        deployment = {
            "probability": list(row["probability"]),
            "class_probability": list(row["class_probability"]),
            "model_version": int(row.get("model_version", self.update_number)),
            **topology,
            "routing": {key: row[key] for key in
                         ("expert_count", "mean_active", "unmatched_ratio")
                         if key in row},
        }
        self._validate_score(deployment, "deployment")
        teacher = self._diagnostic_output(
            self.anchor_teacher, window, "teacher", candidate=False)
        teacher.update({"model_version": 0,
                        "topology_version": "frozen_A",
                        "ramp_version": "frozen_A"})
        candidate = None
        gate = self.model.eagate
        if gate.shadow_id is not None:
            candidate_id = str(gate.shadow_id)
            candidate = self._diagnostic_output(
                self.model, window, "candidate", candidate=True)
            candidate.update({
                "id": candidate_id,
                "birth_step": int(self.shadow_started if self.shadow_started is not None
                                   else self.cursor),
                "training_steps_at_prediction": int(self.shadow_training_steps),
                "model_version": int(self.update_number),
                "topology_version": int(self.topology_version),
                "ramp_version": int(self.ramp_version),
                "prediction_version": "%s@main%d@shadow%d" %
                    (candidate_id, self.update_number,
                     self.shadow_training_steps),
            })
        record = {
            "schema_version": 1,
            "validation_protocol": "prequential_v1",
            "window_index": int(index),
            "prediction_step": int(index + 1),
            "window": self._window_metadata(window),
            "deployment": deployment,
            "teacher": teacher,
            "candidate": candidate,
            "observation": observation,
            "label_matured": False,
        }
        # Keep a separate immutable view for resume/export.  The live record
        # below gains labels, losses and training-use audit entries later;
        # those fields must never rewrite the prediction made at this step.
        record["prediction_view"] = deepcopy({
            "schema_version": record["schema_version"],
            "validation_protocol": record["validation_protocol"],
            "window_index": record["window_index"],
            "prediction_step": record["prediction_step"],
            "window": record["window"],
            "deployment": record["deployment"],
            "teacher": record["teacher"],
            "candidate": record["candidate"],
            "label_matured": False,
        })
        self.pending_predictions[int(index)] = record
        return record

    @staticmethod
    def _audit_view(record):
        """JSON-safe prediction view; pending routing vectors stay in state."""
        view = deepcopy(record)
        view.pop("observation", None)
        view.pop("prediction_view", None)
        for key in ("teacher", "candidate"):
            if isinstance(view.get(key), dict):
                view[key].pop("seconds", None)
        return view

    @staticmethod
    def _prediction_view(record):
        """Return the immutable score/version view for candidate export."""
        if record.get("prediction_view") is not None:
            view = deepcopy(record["prediction_view"])
            for key in ("teacher", "candidate"):
                if isinstance(view.get(key), dict):
                    view[key].pop("seconds", None)
            return view
        # Compatibility for an early in-development state written before the
        # immutable field existed.  Such a state has no settled protocol
        # record that can safely be reconstructed as a future score.
        view = deepcopy(record)
        view.pop("observation", None)
        for key in ("labels", "matured_step", "maturity_raw_label_index",
                    "novel_hosts", "novel_fraction", "losses", "training_uses",
                    "qualification_blocks", "observation_summary", "label_matured"):
            view.pop(key, None)
        for key in ("teacher", "candidate"):
            if isinstance(view.get(key), dict):
                view[key].pop("seconds", None)
        view["label_matured"] = False
        return view

    def _topology_metadata(self):
        gate = self.model.eagate
        return {"topology_version": int(self.topology_version),
                "ramp_version": int(self.ramp_version),
                "active_expert_ids": list(gate.ids),
                "ramp_steps_done": dict(gate.ramp_steps_done)}

    def _cached_score_loss(self, score, labels):
        """Compute the S7 .7/.3/.5 loss from cached, clipped probabilities."""
        probability = np.clip(np.asarray(score["probability"], dtype=np.float64),
                              1e-7, 1. - 1e-7)
        classes = np.clip(np.asarray(score["class_probability"], dtype=np.float64),
                          1e-12, 1.)
        labels = np.asarray(labels, dtype=np.int64)
        positive = labels > 0
        anomaly_loss = np.where(positive, -np.log(probability),
                                -np.log1p(-probability))
        det_weights = np.where(positive, float(self.detection_weights[1]),
                               float(self.detection_weights[0]))
        detection_ce = float((anomaly_loss * det_weights).sum() /
                             max(det_weights.sum(), 1e-12))
        if positive.any():
            class_index = labels[positive] - 1
            class_loss = -np.log(classes[positive,
                                         class_index])
            class_weights = np.asarray(self.resource_weights,
                                       dtype=np.float64)[class_index]
            classification_ce = float((class_loss * class_weights).sum() /
                                      max(class_weights.sum(), 1e-12))
        else:
            classification_ce = 0.0
        ranking = 0.0
        if positive.any() and (~positive).any():
            positive_score = probability[positive] * classes[positive,
                                                               labels[positive] - 1]
            negative_score = probability[~positive] * classes[~positive].max(axis=-1)
            margins = .15 + negative_score[:, None] - positive_score[None, :]
            ranking = float(np.logaddexp(0., margins).mean())
        total = .7 * detection_ce + .3 * classification_ce + .5 * ranking
        return {"total": float(total),
                "detection_ce": detection_ce,
                "positiveclass_ce": classification_ce,
                "joint_ranking": ranking,
                "loss_formula": ".7*detectionCE+.3*positiveclassCE+.5*joint_ranking",
                "score_source": "cached_probabilities_clip_1e-7",
                "auxiliary_included": False}

    def observe_matured(self, index):
        if self.validation_protocol != "prequential_v1":
            observation = self.pending_observations.pop(index)
            calibration = self.dynamic_config["calibration"]
            novel = ((observation["entropy"] > calibration["entropy_p95"]) |
                     (observation["margin"] < calibration["margin_p05"]))
            labels = self.predictions["labels"][index]
            probability = self.predictions["probability"][index].clip(1e-7, 1 - 1e-7)
            target = labels > 0
            loss = float(np.mean(-np.where(target, np.log(probability),
                                           np.log1p(-probability))))
            self.matured_observations.append((float(novel.float().mean()), loss))
            vectors = observation["vectors"].reshape(-1, self.model.cfg.hidden)
            for vector in vectors[novel.reshape(-1)]:
                self.candidate_vectors.append(vector.clone())
            return None
        index = int(index)
        if index in self.matured_indices:
            return self.qualification_by_index.get(index)
        record = self.pending_predictions.pop(index, None)
        if record is None:
            raise RuntimeError("Missing prequential prediction record for matured window %d" % index)
        observation = record.pop("observation", None)
        if observation is None:
            raise RuntimeError("Matured prediction record has no routing observation")
        calibration = self.dynamic_config["calibration"]
        entropy = observation["entropy"].detach().cpu().numpy()
        margin = observation["margin"].detach().cpu().numpy()
        novel = ((entropy > calibration["entropy_p95"]) |
                 (margin < calibration["margin_p05"]))
        labels = self.predictions["labels"][index].copy()
        losses = {"deployment": self._cached_score_loss(
                      record["deployment"], labels),
                  "teacher": self._cached_score_loss(
                      record["teacher"], labels)}
        if record["candidate"] is not None:
            losses["candidate"] = self._cached_score_loss(
                record["candidate"], labels)
        else:
            losses["candidate"] = None
        candidate = record.get("candidate")
        record.update({
            "labels": labels.tolist(),
            "matured_step": int(index + 2),
            "maturity_raw_label_index": int(index + 1),
            "label_matured": True,
            "novel_hosts": novel.astype(bool).tolist(),
            "novel_fraction": float(novel.mean()),
            "losses": losses,
            "training_uses": [],
            "candidate_id": (None if candidate is None else
                             str(candidate.get("id"))),
            "candidate_prediction_version": (None if candidate is None else
                                              candidate.get("prediction_version")),
            "candidate_birth_step": (None if candidate is None else
                                      int(candidate.get("birth_step"))),
            "observation_summary": {
                "entropy_mean": float(entropy.mean()),
                "margin_mean": float(margin.mean()),
            },
        })
        self.matured_indices.add(index)
        self.qualification_by_index[index] = record
        self.qualification_records.append(record)
        # The trigger is a legacy S8 signal and keeps its unweighted binary
        # BCE calibration.  The composite .7/.3/.5 score above is reserved
        # for candidate qualification and is never fed back into the trigger.
        trigger_loss = self._legacy_trigger_loss(record["deployment"], labels)
        self.matured_observations.append((float(novel.mean()), trigger_loss))
        vectors = observation["vectors"].reshape(-1, self.model.cfg.hidden)
        mask = torch.from_numpy(novel.reshape(-1))
        for vector in vectors[mask]:
            self.candidate_vectors.append(vector.clone())
        return record

    @staticmethod
    def _legacy_trigger_loss(score, labels):
        probability = np.clip(np.asarray(score["probability"], dtype=np.float64),
                              1e-7, 1. - 1e-7)
        positive = np.asarray(labels, dtype=np.int64) > 0
        return float(np.mean(np.where(positive, -np.log(probability),
                                      -np.log1p(-probability))))

    def _bump_topology(self):
        self.topology_version += 1

    def _settle_shadow_validation(self):
        """Move cached, eligible candidate scores into validation blocks.

        This method only reads scores captured before the corresponding raw
        label.  It deliberately performs no model forward and marks each
        window once for the current candidate.
        """
        gate = self.model.eagate
        key = gate.shadow_id
        if key is None:
            return []
        shadow_cfg = self.dynamic_config.get("shadow", {})
        min_training_steps = int(shadow_cfg.get("validation_after_updates", 3))
        eligible = []
        for record in self.qualification_records:
            candidate = record.get("candidate")
            if not candidate or str(candidate.get("id")) != str(key):
                continue
            index = int(record["window_index"])
            if index in self.shadow_validation_indices:
                continue
            if int(candidate.get("training_steps_at_prediction", 0)) < min_training_steps:
                continue
            candidate_loss = record.get("losses", {}).get("candidate")
            if candidate_loss is None:
                continue
            eligible.append(record)
        if not eligible:
            return []
        # A qualification block is one causal comparison unit.  Concatenating
        # all hosts in the block preserves weighted-CE denominators and the
        # cross-host joint-ranking term used by s7_loss; averaging independent
        # per-window losses would silently change that objective.
        base_loss = self._cached_score_loss_block(eligible, "deployment")
        candidate_loss = self._cached_score_loss_block(eligible, "candidate")
        block = {
            "window_indices": [int(record["window_index"])
                               for record in eligible],
            "window_ids": [int(record["window_index"])
                           for record in eligible],
            "candidate_id": str(key),
            "candidate_prediction_versions": [
                record["candidate"].get("prediction_version")
                for record in eligible],
            "candidate_training_steps_at_prediction": [
                int(record["candidate"].get("training_steps_at_prediction", 0))
                for record in eligible],
            "prediction_steps": [int(record["prediction_step"])
                                  for record in eligible],
            "matured_steps": [int(record["matured_step"])
                              for record in eligible],
            "base_loss": float(base_loss["total"]),
            "candidate_loss": float(candidate_loss["total"]),
            "base_components": deepcopy(base_loss),
            "candidate_components": deepcopy(candidate_loss),
            "per_window": [{
                "window_index": int(record["window_index"]),
                "base_loss": float(record["losses"]["deployment"]["total"]),
                "candidate_loss": float(record["losses"]["candidate"]["total"]),
            } for record in eligible],
            "validated_before_training": True,
        }
        self.shadow_validation.extend([block])
        for record in eligible:
            index = int(record["window_index"])
            self.shadow_validation_indices.add(index)
            record.setdefault("qualification_blocks", []).append({
                "candidate_id": str(key),
                "candidate_prediction_version": record["candidate"].get(
                    "prediction_version"),
                "validated_before_training": True,
            })
        return [block]

    def _cached_score_loss_block(self, records, score_name):
        probabilities = np.concatenate([
            np.asarray(record[score_name]["probability"], dtype=np.float64)
            for record in records])
        classes = np.concatenate([
            np.asarray(record[score_name]["class_probability"], dtype=np.float64)
            for record in records], axis=0)
        labels = np.concatenate([
            np.asarray(record["labels"], dtype=np.int64)
            for record in records])
        return self._cached_score_loss(
            {"probability": probabilities, "class_probability": classes}, labels)

    def _record_training_audit(self, kind, indices, update_number,
                               settled_before=None, candidate_id=None):
        indices = [int(index) for index in indices]
        settled_before = (set(self.matured_indices)
                          if settled_before is None else set(settled_before))
        violations = [index for index in indices if index not in settled_before]
        entry = {
            "kind": kind,
            "audit_phase": ("before_training_preflight" if kind == "preflight"
                            else "after_training_call_snapshot"),
            "step": int(self.cursor),
            "causal_step": int(self.cursor),
            "update_number": int(update_number),
            "window_indices": indices,
            "settled_before_training": sorted(settled_before.intersection(indices)),
            "early_training_window_indices": violations,
            "candidate_id": candidate_id,
        }
        self.causal_audit.append(entry)
        self.audit_violation_count += len(violations)
        for index in indices:
            record = self.qualification_by_index.get(index)
            if record is not None:
                record.setdefault("training_uses", []).append({
                    "kind": kind,
                    "step": int(self.cursor),
                    "causal_step": int(self.cursor),
                    "update_number": int(update_number),
                    "candidate_id": candidate_id,
                    "before_qualification": index not in settled_before,
                })
        return entry

    def adapt_topology(self):
        # V3 decisions are made every update using only already-matured labels.
        return None

    def _forward_indices(self, indices):
        x, s, g, ids, before, caps = self._stack(indices)
        return self.model(x, s, g, graph_context=self._context(ids, before, caps))

    def _loss(self, out, indices):
        return s7_loss(self.model, out,
                       torch.from_numpy(self.predictions["labels"][indices]),
                       self.detection_weights, self.resource_weights,
                       .7, .3, 0., 0., .5)

    def _validate_shadow(self, indices):
        with torch.no_grad():
            base = float(self._loss(self._forward_indices(indices), indices))
            with self.model.eagate.preview():
                candidate = float(self._loss(self._forward_indices(indices), indices))
        self.shadow_validation.append((base, candidate))

    def _train_shadow(self, indices):
        gate = self.model.eagate
        self.shadow_optimizer.zero_grad(set_to_none=True)
        with gate.preview():
            loss = self._loss(self._forward_indices(indices), indices)
        parameters = gate.shadow_parameters()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        self.shadow_optimizer.step()
        self.shadow_optimizer.zero_grad(set_to_none=True)
        self.shadow_training_steps += 1

    def _update_legacy(self):
        started = time.perf_counter()
        super().update()
        gate = self.model.eagate
        gate.ramp_step()
        if not self.matured_observations:
            return
        novelty, loss = np.mean(list(self.matured_observations), axis=0)
        record = self.trigger.observe_window(float(novelty),
                    len(self.matured_observations) * 16, float(loss), loss_alpha=.2)
        # Samples eligible for birth must be recent novel states, not lifetime traffic.
        self.trigger.candidate_samples = len(self.candidate_vectors)
        record["candidate_samples"] = self.trigger.candidate_samples
        event = {"step": self.cursor, "before": list(gate.ids), "added": [],
                 "dormant": [], "reactivated": [], "decisions": [], **record}
        newest = list(self.buffer)[-10:]
        if gate.shadow_id is not None:
            # These windows arrived after the previous candidate update. Validate
            # before training on them; no replay/holdout overlap at evaluation time.
            if self.shadow_training_steps >= 3:
                self._validate_shadow(newest)
            self._train_shadow(newest)
            if self.shadow_training_steps >= 10:
                base, candidate = np.mean(self.shadow_validation, axis=0)
                event.update({"shadow_base_loss": float(base),
                              "shadow_candidate_loss": float(candidate)})
                if candidate < base * .995:
                    key = gate.activate_shadow()
                    self.sync_optimizer()
                    event["added"].append(key)
                    event["decisions"].append("shadow_validated_activate_ramp0")
                else:
                    gate.discard_shadow()
                    event["decisions"].append("shadow_rejected_no_future_loss_gain")
                self.shadow_optimizer = None
                self.cooldown_until = self.cursor + 100
                self.trigger.novelty_windows_above = 0
                self.candidate_vectors.clear()
        ready, checks = self.trigger.ready(len(gate.ids))
        event["checks"] = checks
        if ready and gate.shadow_id is None and self.cursor >= self.cooldown_until:
            centroid = torch.stack(list(self.candidate_vectors)).mean(0)
            dormant = list(gate.dormant_key_rows)
            match = None
            if dormant:
                similarity = [float(F.cosine_similarity(centroid, gate.dormant_key_rows[k], dim=0)) for k in dormant]
                if max(similarity) > .9:
                    match = dormant[int(np.argmax(similarity))]
            if match is not None:
                gate._reactivate(match)
                gate.ramp_steps_done[match] = 0
                gate.low_activation_windows[match] = 0
                self.model.set_trainability()
                self.sync_optimizer()
                event["reactivated"].append(match)
                self.cooldown_until = self.cursor + 100
            elif len(gate.ids) + len(gate.dormant_experts) < 8:
                key, parent = gate.create_shadow(centroid)
                self.shadow_optimizer = torch.optim.AdamW(gate.shadow_parameters(),
                    lr=self.dynamic_config.get("shadow_lr", self.learning_rate), weight_decay=1e-4)
                self.shadow_training_steps = 0
                self.shadow_validation = []
                self.shadow_started = self.cursor
                event.update({"shadow": key, "parent": parent})
                event["decisions"].append("shadow_created")
        # Weighted routing contribution can fall even when all experts are eligible.
        if self.cursor % 100 == 0:
            for key in list(gate.ids):
                gate.expert_age[key] = gate.expert_age.get(key, 0) + 100
                share = gate.weight_mass.get(key, 0.) / max(gate.routing_samples, 1)
                gate.activation_ema[key] = .8 * gate.activation_ema.get(key, share) + .2 * share
                low = gate.activation_ema[key] < .01
                gate.low_activation_windows[key] = gate.low_activation_windows.get(key, 0) + 1 if low else 0
                if len(gate.ids) > 4 and gate.expert_age[key] >= 500 and gate.low_activation_windows[key] >= 5:
                    gate._retire(key)
                    gate.freeze_dormant()
                    self.sync_optimizer()
                    event["dormant"].append(key)
            gate.reset_statistics()
        event["after"] = list(gate.ids)
        self.dynamic_events.append(event)
        self.updates[-1]["topology_event"] = event
        self.updates[-1]["seconds"] = time.perf_counter() - started

    def _update_prequential(self):
        """R0 update: settle and decide, then train on matured windows.

        ``S7Session.update`` keeps the registered memory and loss unchanged.
        All dynamic decisions that depend on the stream are made before that
        call, while shadow training is performed afterwards on already
        settled windows.  Candidate qualification consumes the ledger only;
        it never re-forwards historical inputs.
        """
        started = time.perf_counter()
        gate = self.model.eagate
        gate.ramp_step()
        self.ramp_version += 1
        settled_before = set(self.matured_indices)
        newest = list(self.buffer)[-10:]
        unsettled_buffer = [int(index) for index in self.buffer
                            if int(index) not in settled_before]
        if unsettled_buffer:
            # Fail before either optimizer can consume a target.  The audit
            # entry remains inspectable in a step-boundary failure state.
            self._record_training_audit(
                "preflight", unsettled_buffer, self.update_number,
                settled_before=settled_before)
            raise AssertionError(
                "Prequential update encountered unsettled training windows: %s"
                % unsettled_buffer)
        event = {"step": int(self.cursor),
                 "causal_step": int(self.cursor),
                 "validation_protocol": "prequential_v1",
                 "before": list(gate.ids), "added": [], "dormant": [],
                 "reactivated": [], "decisions": [],
                 "qualification_count": len(self.qualification_records),
                 "pending_prediction_count": len(self.pending_predictions)}

        if self.matured_observations:
            novelty, loss = np.mean(list(self.matured_observations), axis=0)
            trigger_record = self.trigger.observe_window(
                float(novelty), len(self.matured_observations) * 16,
                float(loss), loss_alpha=.2)
        else:
            trigger_record = {"novelty_ema": self.trigger.novelty_ema,
                              "loss_ema": self.trigger.loss_ema,
                              "candidate_samples": self.trigger.candidate_samples}
        # Samples eligible for birth are recent novel states, not lifetime
        # traffic.  This remains the legacy trigger gate and is independent
        # of the qualification loss used below.
        self.trigger.candidate_samples = len(self.candidate_vectors)
        trigger_record["candidate_samples"] = self.trigger.candidate_samples
        event.update(trigger_record)

        if gate.shadow_id is not None:
            validation_entries = self._settle_shadow_validation()
            current_validation = [entry for entry in self.shadow_validation
                                  if isinstance(entry, dict) and
                                  str(entry.get("candidate_id")) ==
                                  str(gate.shadow_id)]
            event["shadow_validation_records"] = len(current_validation)
            event["shadow_validation_new_records"] = len(validation_entries)
            shadow_cfg = self.dynamic_config.get("shadow", {})
            train_limit = int(shadow_cfg.get("training_updates", 10))
            minimum_records = max(1, int(shadow_cfg.get(
                "minimum_validation_records", 3)))
            event["shadow_training_steps"] = int(self.shadow_training_steps)
            event["shadow_validation_minimum_records"] = minimum_records
            if self.shadow_training_steps >= train_limit:
                if len(current_validation) < minimum_records:
                    event["decisions"].append(
                        "shadow_deferred_insufficient_future_validation")
                else:
                    base = float(np.mean([item["base_loss"]
                                          for item in current_validation]))
                    candidate = float(np.mean([item["candidate_loss"]
                                               for item in current_validation]))
                    event.update({"shadow_base_loss": base,
                                  "shadow_candidate_loss": candidate,
                                  "shadow_validation_records": len(current_validation)})
                    improvement = float(shadow_cfg.get(
                        "minimum_future_loss_improvement", .005))
                    event["shadow_minimum_future_loss_improvement"] = improvement
                    if candidate < base * (1. - improvement):
                        key = gate.activate_shadow()
                        self._bump_topology()
                        self.model.set_trainability()
                        self.sync_optimizer()
                        event["added"].append(key)
                        event["decisions"].append(
                            "shadow_validated_activate_ramp0")
                    else:
                        gate.discard_shadow()
                        self._bump_topology()
                        event["decisions"].append(
                            "shadow_rejected_no_future_loss_gain")
                    self.shadow_optimizer = None
                    self.cooldown_until = self.cursor + 100
                    self.trigger.novelty_windows_above = 0
                    self.candidate_vectors.clear()

        ready, checks = self.trigger.ready(len(gate.ids))
        event["checks"] = checks
        if (ready and gate.shadow_id is None and self.cursor >= self.cooldown_until
                and self.candidate_vectors):
            centroid = torch.stack(list(self.candidate_vectors)).mean(0)
            dormant = list(gate.dormant_key_rows)
            match = None
            if dormant:
                similarity = [float(F.cosine_similarity(
                    centroid, gate.dormant_key_rows[key], dim=0))
                              for key in dormant]
                if max(similarity) > .9:
                    match = dormant[int(np.argmax(similarity))]
            if match is not None:
                gate._reactivate(match)
                gate.ramp_steps_done[match] = 0
                gate.low_activation_windows[match] = 0
                self._bump_topology()
                self.model.set_trainability()
                self.sync_optimizer()
                event["reactivated"].append(match)
                self.cooldown_until = self.cursor + 100
                event["decisions"].append(
                    "reactivate_legacy_similarity_unvalidated")
            elif len(gate.ids) + len(gate.dormant_experts) < 8:
                key, parent = gate.create_shadow(centroid)
                self._bump_topology()
                self.shadow_optimizer = torch.optim.AdamW(
                    gate.shadow_parameters(),
                    lr=self.dynamic_config.get("shadow_lr", self.learning_rate),
                    weight_decay=1e-4)
                self.shadow_training_steps = 0
                self.shadow_validation = []
                self.shadow_validation_indices = set()
                self.shadow_started = self.cursor
                event.update({"shadow": key, "parent": parent})
                event["decisions"].append("shadow_created")

        # Retirement uses statistics accumulated by deployment predictions up
        # to this boundary.  It is a decision before this update's training.
        if self.cursor % 100 == 0:
            for key in list(gate.ids):
                gate.expert_age[key] = gate.expert_age.get(key, 0) + 100
                share = gate.weight_mass.get(key, 0.) / max(gate.routing_samples, 1)
                gate.activation_ema[key] = (.8 * gate.activation_ema.get(key, share)
                                             + .2 * share)
                low = gate.activation_ema[key] < .01
                gate.low_activation_windows[key] = (
                    gate.low_activation_windows.get(key, 0) + 1 if low else 0)
                if (len(gate.ids) > 4 and gate.expert_age[key] >= 500 and
                        gate.low_activation_windows[key] >= 5):
                    gate._retire(key)
                    self._bump_topology()
                    gate.freeze_dormant()
                    self.model.set_trainability()
                    self.sync_optimizer()
                    event["dormant"].append(key)
            gate.reset_statistics()
        event["after_decision"] = list(gate.ids)
        event["topology_version"] = int(self.topology_version)
        event["ramp_version"] = int(self.ramp_version)

        # Main S7 update happens only after all labels used by it have been
        # settled.  Its draw is preserved verbatim for the causal audit.
        super().update()
        main_update = self.updates[-1]
        main_indices = list(main_update.get("buffer_indices", []))
        main_audit = self._record_training_audit(
            "main", main_indices, main_update["update_number"],
            settled_before=settled_before)
        main_update["causal_audit"] = main_audit

        # Shadow training is intentionally after the main update.  Validation
        # for these same windows was read from cached scores above, before
        # either optimizer saw the labels.
        if gate.shadow_id is not None and self.shadow_optimizer is not None and newest:
            candidate_id = str(gate.shadow_id)
            candidate_audit = self._record_training_audit(
                "candidate", newest, main_update["update_number"],
                settled_before=settled_before, candidate_id=candidate_id)
            self._train_shadow(newest)
            candidate_audit["training_steps_after"] = int(self.shadow_training_steps)
            event["candidate_training_windows"] = list(newest)
            event["candidate_training_steps_after"] = int(self.shadow_training_steps)
        event["qualification_count_after"] = len(self.qualification_records)
        event["pending_prediction_count_after"] = len(self.pending_predictions)
        event["audit_violation_count"] = int(self.audit_violation_count)
        event["after"] = list(gate.ids)
        self.dynamic_events.append(event)
        main_update["topology_event"] = event
        main_update["seconds"] = time.perf_counter() - started

    def update(self):
        if self.validation_protocol == "prequential_v1":
            return self._update_prequential()
        return self._update_legacy()

    @torch.no_grad()
    def evaluate_reference(self, blocks=None):
        # Same-domain anchors use the same normalization and graph contract as replay.
        from train_ftmoe_end_to_end import metric_arrays
        pool = self.anchor_pool
        dets, clss = [], []
        for start in range(0, len(self.anchor_indices), 64):
            sl = slice(start, start + 64)
            out = self.model(pool["x"][sl], pool["schedule"][sl], pool["graph_x"][sl],
                graph_context=self._context(pool["ids"][sl], pool["before"][sl], pool["caps"][sl]))
            dets.append(out["detection_logits"].softmax(-1)[..., 1])
            clss.append(out["class_logits"].softmax(-1))
        score = metric_arrays(torch.cat(dets).numpy().reshape(-1),
                              torch.cat(clss).numpy().reshape(-1, 3),
                              pool["labels"].numpy().reshape(-1))
        self.reference.append({"step": self.cursor, "source": "same_domain_train_anchor", **score})

    def finish(self):
        """Finalize the guard label and settle the final prediction once."""
        super().finish()
        if self.validation_protocol == "prequential_v1":
            # OnlineSessionV2.finish writes the last tolerance label but has
            # no S7 maturity hook.  observe_matured is idempotent, so repeated
            # finish calls cannot append duplicate qualification records.
            self.observe_matured(self.cursor - 1)

    def save(self):
        state = super().save()
        state["dynamic"] = {"validation_protocol": self.validation_protocol,
            "trigger": self.trigger.state(),
            "pending_observations": self.pending_observations,
            "pending_predictions": self.pending_predictions,
            "qualification_records": self.qualification_records,
            "matured_indices": sorted(self.matured_indices),
            "matured_observations": list(self.matured_observations),
            "candidate_vectors": list(self.candidate_vectors),
            "dynamic_events": self.dynamic_events, "shadow_started": self.shadow_started,
            "shadow_training_steps": self.shadow_training_steps,
            "shadow_validation": self.shadow_validation,
            "shadow_validation_indices": sorted(self.shadow_validation_indices),
            "cooldown_until": self.cooldown_until,
            "topology_version": self.topology_version,
            "ramp_version": self.ramp_version,
            "causal_audit": self.causal_audit,
            "audit_violation_count": self.audit_violation_count,
            "diagnostic_cost": self.diagnostic_cost,
            "shadow_optimizer": None if self.shadow_optimizer is None else self.shadow_optimizer.state_dict()}
        return state

    def restore(self, state):
        dynamic = state.get("dynamic")
        if dynamic is None:
            raise ValueError("S8 checkpoint is missing dynamic validation state")
        saved_protocol = dynamic.get("validation_protocol", "legacy_v3")
        if saved_protocol != self.validation_protocol:
            raise ValueError(
                "Cannot resume S8 checkpoint with validation_protocol %s as %s"
                % (saved_protocol, self.validation_protocol))
        super().restore(state)
        self.trigger.restore(dynamic["trigger"])
        for name in ("pending_observations", "dynamic_events", "shadow_started",
                     "shadow_training_steps", "shadow_validation", "cooldown_until"):
            setattr(self, name, dynamic.get(name, getattr(self, name)))
        self.pending_predictions = {
            int(index): record for index, record in dynamic.get(
                "pending_predictions", {}).items()
        }
        self.qualification_records = list(dynamic.get("qualification_records", []))
        self.qualification_by_index = {
            int(record["window_index"]): record
            for record in self.qualification_records
        }
        self.matured_indices = set(int(index) for index in dynamic.get(
            "matured_indices", self.qualification_by_index))
        self.matured_observations = deque(dynamic["matured_observations"], maxlen=10)
        self.candidate_vectors = deque(dynamic["candidate_vectors"], maxlen=256)
        self.shadow_validation_indices = set(int(index) for index in dynamic.get(
            "shadow_validation_indices", []))
        self.topology_version = int(dynamic.get("topology_version", 0))
        self.ramp_version = int(dynamic.get("ramp_version", 0))
        self.causal_audit = list(dynamic.get("causal_audit", []))
        self.audit_violation_count = int(dynamic.get("audit_violation_count", 0))
        self.diagnostic_cost = dict(dynamic.get("diagnostic_cost", {
            "teacher_forward_count": 0,
            "teacher_forward_seconds": 0.0,
            "candidate_forward_count": 0,
            "candidate_forward_seconds": 0.0,
        }))
        # A prior session object may still hold an optimizer from its freshly
        # initialized topology.  Explicitly clear it even when the saved
        # candidate optimizer is empty.
        self.shadow_optimizer = None
        if dynamic.get("shadow_optimizer") is not None:
            if self.model.eagate.shadow_id is None:
                raise ValueError("Saved shadow optimizer has no shadow candidate")
            self.shadow_optimizer = torch.optim.AdamW(self.model.eagate.shadow_parameters(),
                lr=self.dynamic_config.get("shadow_lr", self.learning_rate), weight_decay=1e-4)
            self.shadow_optimizer.load_state_dict(dynamic["shadow_optimizer"])
