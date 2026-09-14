"""Causal expert lifecycle for Protocol-024.

The controller never receives phase/mode/event IDs.  Its only trigger signal is
matured supervised loss.  Frozen 64-D z is used only for expert representation
(parent/reuse bookkeeping); router entropy/margin are diagnostics.  Candidate
training consumes distinct matured intervals, while candidate validation uses
full-model predictions recorded *before* the corresponding label is revealed.

This module supports a debug-only forced trigger for mechanism tests.  Formal
runs must construct the controller with ``debug_force_trigger_after=None``.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_session import Protocol024Session, _copy_cpu


DEFAULT_LIFECYCLE_CONFIG = {
    "calibration_intervals": 256,
    "trigger_window": 32,
    "consecutive_abnormal_windows": 2,
    "loss_percentile": 99.0,
    "candidate_train_intervals": 64,
    "validation_intervals": 32,
    "accept_relative_loss_improvement": 0.01,
    "normal_probability_allowance": 0.01,
    "cooldown_intervals": 64,
    "retirement_enabled": False,
    "hard_delete_enabled": False,
    "debug_force_trigger_after": None,
}


def _loss_from_logits(detection_logits, class_logits, target):
    detection = torch.as_tensor(detection_logits, dtype=torch.float32)
    classes = torch.as_tensor(class_logits, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.long)
    with torch.no_grad():
        terms = s4.supervised_terms(detection, classes, target)
    return float(terms["total"].detach().cpu())


def _normal_anomaly_mean(detection_logits, target):
    detection = torch.as_tensor(detection_logits, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.long).reshape(-1)
    probability = detection.reshape(-1, 2).softmax(-1)[:, 1]
    normal = target == 0
    if not bool(normal.any()):
        return None, 0
    return float(probability[normal].mean().cpu()), int(normal.sum().item())


def _unit_rows(z):
    z = np.asarray(z, dtype=np.float64).reshape(-1, z.shape[-1])
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norm, 1e-12)


class CausalResidualLifecycle:
    """Loss-triggered birth controller with causal shadow qualification."""

    def __init__(self, config=None):
        cfg = dict(DEFAULT_LIFECYCLE_CONFIG)
        cfg.update(config or {})
        for key in ("calibration_intervals", "trigger_window",
                    "consecutive_abnormal_windows", "candidate_train_intervals",
                    "validation_intervals", "cooldown_intervals"):
            cfg[key] = int(cfg[key])
            if cfg[key] < 1:
                raise ValueError("%s must be >=1" % key)
        cfg["loss_percentile"] = float(cfg["loss_percentile"])
        cfg["accept_relative_loss_improvement"] = float(
            cfg["accept_relative_loss_improvement"])
        cfg["normal_probability_allowance"] = float(
            cfg["normal_probability_allowance"])
        if cfg["debug_force_trigger_after"] is not None:
            cfg["debug_force_trigger_after"] = int(cfg["debug_force_trigger_after"])
        self.config = cfg
        self.reset()

    def reset(self):
        self.phase = "calibrating"
        self.matured_count = 0
        self.loss_history = []
        self.calibration_threshold = None
        self.monitor_window = []
        self.abnormal_windows = 0
        self.cooldown_left = 0
        self.candidate_id = None
        self.candidate_parent_id = None
        self.candidate_training_indices = []
        self.candidate_validation_indices = []
        self.prelabel_candidate = {}
        self.validation_pairs = []
        self.route_mass = {}
        self.z_weighted_sum = {}
        self.z_weight = {}
        self.recent_z = []
        self.extra_compute = {"shadow_train_steps": 0,
                              "shadow_train_seconds": 0.0,
                              "shadow_validation_forwards": 0,
                              "shadow_validation_seconds": 0.0}
        self.events = []
        self.last_decision = None

    # ------------------------------------------------------------------
    # state / audit
    # ------------------------------------------------------------------
    def state_dict(self):
        return _copy_cpu({
            "config": self.config,
            "phase": self.phase,
            "matured_count": self.matured_count,
            "loss_history": self.loss_history,
            "calibration_threshold": self.calibration_threshold,
            "monitor_window": self.monitor_window,
            "abnormal_windows": self.abnormal_windows,
            "cooldown_left": self.cooldown_left,
            "candidate_id": self.candidate_id,
            "candidate_parent_id": self.candidate_parent_id,
            "candidate_training_indices": self.candidate_training_indices,
            "candidate_validation_indices": self.candidate_validation_indices,
            "prelabel_candidate": self.prelabel_candidate,
            "validation_pairs": self.validation_pairs,
            "route_mass": self.route_mass,
            "z_weighted_sum": self.z_weighted_sum,
            "z_weight": self.z_weight,
            "recent_z": self.recent_z,
            "extra_compute": self.extra_compute,
            "events": self.events,
            "last_decision": self.last_decision,
        })

    def load_state_dict(self, state):
        if dict(state["config"]) != self.config:
            raise ValueError("lifecycle config mismatch on resume")
        for key, value in state.items():
            if key == "config":
                continue
            setattr(self, key, deepcopy(value))

    def _sync_session(self, session):
        session.lifecycle_state = self.state_dict()
        session.lifecycle_state["enabled"] = True
        session.lifecycle_state["hard_delete_enabled"] = bool(
            self.config["hard_delete_enabled"])
        session.lifecycle_events = list(self.events)

    def _event(self, session, kind, **payload):
        record = {"kind": kind, "cursor": int(session.cursor),
                  "matured_count": int(self.matured_count),
                  "phase": self.phase}
        record.update(payload)
        self.events.append(record)
        self._sync_session(session)
        return record

    # ------------------------------------------------------------------
    # pre-label: z bookkeeping and prospective candidate prediction
    # ------------------------------------------------------------------
    def on_pre_label_prediction(self, session, index, output):
        if session.arm != "D":
            return
        bank = session.model.learner
        z_tensor = session.model._last_z.detach()
        z = z_tensor.cpu().numpy()
        routing = output.get("correction_router_probabilities")
        if routing is not None:
            routing_np = routing.detach().cpu().numpy().reshape(-1, routing.shape[-1])
            z_unit = _unit_rows(z)
            for column, key in enumerate(bank.ids):
                weight = routing_np[:, column]
                total = float(weight.sum())
                if total <= 0.0:
                    continue
                vector = (z_unit * weight[:, None]).sum(axis=0)
                self.route_mass[key] = float(self.route_mass.get(key, 0.0) + total)
                if key not in self.z_weighted_sum:
                    self.z_weighted_sum[key] = np.zeros(vector.shape, dtype=np.float64)
                    self.z_weight[key] = 0.0
                self.z_weighted_sum[key] = np.asarray(self.z_weighted_sum[key]) + vector
                self.z_weight[key] = float(self.z_weight[key] + total)
            mean_z = z_unit.mean(axis=0)
            mean_z /= max(np.linalg.norm(mean_z), 1e-12)
            self.recent_z.append(mean_z)
            if len(self.recent_z) > self.config["trigger_window"]:
                self.recent_z = self.recent_z[-self.config["trigger_window"]:]

        if self.phase != "candidate_validation" or bank.shadow_id is None:
            self._sync_session(session)
            return

        started = time.perf_counter()
        with torch.no_grad():
            correction, _ = bank.preview_with_shadow(z_tensor, shadow_ramp=1.0)
            det = output["base_final_detection_logits"] + correction[..., :2]
            cls = output["base_final_class_logits"] + correction[..., 2:]
        self.prelabel_candidate[int(index)] = {
            "candidate_detection_logits": det[0].detach().cpu().numpy(),
            "candidate_class_logits": cls[0].detach().cpu().numpy(),
            "live_detection_logits": output["detection_logits"][0].detach().cpu().numpy(),
            "live_class_logits": output["class_logits"][0].detach().cpu().numpy(),
            "recorded_before_label": True,
        }
        self.extra_compute["shadow_validation_forwards"] += 1
        self.extra_compute["shadow_validation_seconds"] += time.perf_counter() - started
        self._sync_session(session)

    # ------------------------------------------------------------------
    # matured-label path
    # ------------------------------------------------------------------
    def _current_live_loss(self, session, index, target):
        return _loss_from_logits(
            session.predictions["detection_logits"][index],
            session.predictions["class_logits"][index], target)

    def _finish_calibration_if_ready(self, session):
        count = self.config["calibration_intervals"]
        width = self.config["trigger_window"]
        if self.matured_count < count:
            return False
        baseline = np.asarray(self.loss_history[:count], dtype=np.float64)
        if baseline.size < width:
            raise RuntimeError("calibration prefix shorter than trigger window")
        means = np.asarray([baseline[i:i + width].mean()
                            for i in range(0, baseline.size - width + 1)],
                           dtype=np.float64)
        self.calibration_threshold = float(np.percentile(
            means, self.config["loss_percentile"]))
        self.phase = "monitoring"
        self.monitor_window = []
        self._event(session, "calibration_frozen",
                    loss_threshold=self.calibration_threshold,
                    calibration_intervals=count,
                    trigger_window=width,
                    percentile=self.config["loss_percentile"])
        return True

    def _select_parent(self, bank):
        if not bank.ids:
            raise RuntimeError("no active expert for birth parent")
        return max(bank.ids, key=lambda key: (float(self.route_mass.get(key, 0.0)),
                                              -int(key)))

    def _start_candidate(self, session, reason):
        bank = session.model.learner
        if bank.shadow_id is not None:
            return False
        if bank.resident_count() >= bank.max_experts:
            self._event(session, "birth_blocked_capacity",
                        resident_count=bank.resident_count(), max_experts=bank.max_experts)
            self.abnormal_windows = 0
            self.monitor_window = []
            return False
        parent = self._select_parent(bank)
        candidate = bank.create_shadow(parent)
        session.create_shadow_optimizer()
        self.candidate_id = candidate
        self.candidate_parent_id = parent
        self.candidate_training_indices = []
        self.candidate_validation_indices = []
        self.prelabel_candidate = {}
        self.validation_pairs = []
        self.phase = "candidate_training"
        self.abnormal_windows = 0
        self.monitor_window = []
        self._event(session, "candidate_created", candidate_id=candidate,
                    parent_id=parent, reason=reason,
                    resident_count=bank.resident_count())
        return True

    def _candidate_train_one(self, session, index, target):
        if int(index) in self.candidate_training_indices:
            raise AssertionError("candidate training interval reused")
        started = time.perf_counter()
        x, sched, graph, context = s4.window_batch(session.replay, [int(index)])
        base, z = session.model._base_forward(
            x, sched, graph, graph_context=context, record=False)
        session.model.zero_grad(set_to_none=True)
        session.shadow_optimizer.zero_grad(set_to_none=True)
        correction, _ = session.model.learner.preview_with_shadow(z, shadow_ramp=1.0)
        detection = base["detection_logits"] + correction[..., :2]
        classes = base["class_logits"] + correction[..., 2:]
        terms = s4.supervised_terms(detection, classes,
                                    torch.as_tensor(target).unsqueeze(0).long())
        terms["total"].backward()
        session.shadow_optimizer.step()
        session.model.zero_grad(set_to_none=True)
        self.candidate_training_indices.append(int(index))
        self.extra_compute["shadow_train_steps"] += 1
        self.extra_compute["shadow_train_seconds"] += time.perf_counter() - started

        if len(self.candidate_training_indices) >= self.config["candidate_train_intervals"]:
            for parameter in session.model.learner.shadow_parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            self.phase = "candidate_validation"
            self.prelabel_candidate = {}
            self.validation_pairs = []
            self._event(session, "candidate_training_complete",
                        candidate_id=self.candidate_id,
                        distinct_intervals=len(self.candidate_training_indices))

    def _score_candidate_validation(self, session, index, target):
        record = self.prelabel_candidate.pop(int(index), None)
        if record is None:
            return False
        if not record.get("recorded_before_label", False):
            raise AssertionError("candidate validation prediction was not pre-label")
        if int(index) in self.candidate_training_indices:
            raise AssertionError("validation interval leaked into candidate training")
        live_loss = _loss_from_logits(record["live_detection_logits"],
                                      record["live_class_logits"], target)
        candidate_loss = _loss_from_logits(record["candidate_detection_logits"],
                                           record["candidate_class_logits"], target)
        live_normal, normal_rows = _normal_anomaly_mean(
            record["live_detection_logits"], target)
        candidate_normal, _ = _normal_anomaly_mean(
            record["candidate_detection_logits"], target)
        pair = {"index": int(index), "live_loss": live_loss,
                "candidate_loss": candidate_loss,
                "normal_rows": int(normal_rows),
                "live_normal_probability": live_normal,
                "candidate_normal_probability": candidate_normal}
        self.validation_pairs.append(pair)
        self.candidate_validation_indices.append(int(index))
        if len(self.validation_pairs) >= self.config["validation_intervals"]:
            self._decide_candidate(session)
        return True

    def _decide_candidate(self, session):
        pairs = self.validation_pairs[:self.config["validation_intervals"]]
        live = float(np.mean([x["live_loss"] for x in pairs]))
        candidate = float(np.mean([x["candidate_loss"] for x in pairs]))
        required = float(self.config["accept_relative_loss_improvement"])
        improvement = ((live - candidate) / max(abs(live), 1e-12))
        normal_pairs = [x for x in pairs if x["normal_rows"] > 0]
        if normal_pairs:
            live_normal = float(np.average(
                [x["live_normal_probability"] for x in normal_pairs],
                weights=[x["normal_rows"] for x in normal_pairs]))
            candidate_normal = float(np.average(
                [x["candidate_normal_probability"] for x in normal_pairs],
                weights=[x["normal_rows"] for x in normal_pairs]))
            normal_ok = candidate_normal <= (live_normal +
                                              self.config["normal_probability_allowance"])
        else:
            live_normal = candidate_normal = None
            normal_ok = True
        accept = bool(improvement >= required and normal_ok)
        candidate_id = self.candidate_id
        decision = {"candidate_id": candidate_id,
                    "live_mean_loss": live,
                    "candidate_mean_loss": candidate,
                    "relative_improvement": improvement,
                    "required_relative_improvement": required,
                    "live_normal_probability": live_normal,
                    "candidate_normal_probability": candidate_normal,
                    "normal_safety_ok": bool(normal_ok),
                    "validation_intervals": len(pairs),
                    "accepted": accept}
        self.last_decision = decision
        if accept:
            activated = session.activate_shadow()
            if activated != candidate_id:
                raise AssertionError("activated candidate ID changed")
            self._event(session, "candidate_accepted", **decision)
        else:
            session.discard_shadow()
            self._event(session, "candidate_rejected", **decision)
        self.candidate_id = None
        self.candidate_parent_id = None
        self.phase = "cooldown"
        self.cooldown_left = self.config["cooldown_intervals"]
        self.monitor_window = []
        self.abnormal_windows = 0
        self.prelabel_candidate = {}
        self.validation_pairs = []

    def on_matured(self, session, index, target):
        if session.arm != "D":
            return
        index = int(index)
        loss = self._current_live_loss(session, index, target)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite matured supervised loss")
        self.loss_history.append(loss)
        self.matured_count += 1

        if self.phase == "calibrating":
            if self._finish_calibration_if_ready(session):
                self._sync_session(session)
                return
        elif self.phase == "candidate_training":
            self._candidate_train_one(session, index, target)
            self._sync_session(session)
            return
        elif self.phase == "candidate_validation":
            self._score_candidate_validation(session, index, target)
            self._sync_session(session)
            return
        elif self.phase == "cooldown":
            self.cooldown_left -= 1
            if self.cooldown_left <= 0:
                self.phase = "monitoring"
                self._event(session, "cooldown_complete")
            self._sync_session(session)
            return

        if self.phase != "monitoring":
            self._sync_session(session)
            return

        force_after = self.config.get("debug_force_trigger_after")
        if force_after is not None and self.matured_count >= force_after:
            self._start_candidate(session, "debug_forced_matured_count")
            self.config["debug_force_trigger_after"] = None
            self._sync_session(session)
            return

        self.monitor_window.append(loss)
        if len(self.monitor_window) < self.config["trigger_window"]:
            self._sync_session(session)
            return
        window_mean = float(np.mean(self.monitor_window))
        abnormal = window_mean > float(self.calibration_threshold)
        self.abnormal_windows = self.abnormal_windows + 1 if abnormal else 0
        self._event(session, "loss_window",
                    window_mean=window_mean,
                    threshold=float(self.calibration_threshold),
                    abnormal=bool(abnormal),
                    consecutive_abnormal=int(self.abnormal_windows))
        self.monitor_window = []
        if self.abnormal_windows >= self.config["consecutive_abnormal_windows"]:
            self._start_candidate(session, "matured_loss_threshold")
        self._sync_session(session)

    def on_after_live_update(self, session, index):
        if session.arm != "D":
            return
        bank = session.model.learner
        if any(float(bank.ramp[k]) < 1.0 for k in bank.ids):
            changed = bank.ramp_step()
            if changed:
                session.learner_hash = session.learner_state_hash()
                self._event(session, "ramp_step",
                            index=int(index), ramps=deepcopy(bank.ramp))
        self._sync_session(session)


class LifecycleProtocol024Session(Protocol024Session):
    """Protocol024Session with the controller wired into causal step hooks."""

    def __init__(self, *args, lifecycle_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.arm != "D":
            raise ValueError("LifecycleProtocol024Session is only for D")
        self.lifecycle_controller = CausalResidualLifecycle(lifecycle_config)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True

    def step(self):
        t = self.cursor
        if t >= self.steps:
            raise StopIteration
        started = time.perf_counter()
        x, sched, graph, context = s4.window_batch(self.replay, [t])
        out = self.model.predict_deployment(x, sched, graph, graph_context=context)
        probability = torch.softmax(out["detection_logits"], -1)[0, :, 1] \
            .detach().numpy().astype(np.float32)
        classes = torch.softmax(out["class_logits"], -1)[0] \
            .detach().numpy().astype(np.float32)
        if not (np.isfinite(probability).all() and np.isfinite(classes).all()):
            raise RuntimeError("non-finite prediction at %d" % t)
        self.predictions["probability"][t] = probability
        self.predictions["class_probability"][t] = classes
        self.predictions["detection_logits"][t] = out["detection_logits"][0].detach().cpu().numpy()
        self.predictions["class_logits"][t] = out["class_logits"][0].detach().cpu().numpy()
        self.predictions["model_version"][t] = self.model_version
        self.predictions["learner_hash"][t] = self.learner_hash
        self.predictions["prediction_seconds"][t] = time.perf_counter() - started
        if not np.all(self.raw_seen[t] == -1):
            raise AssertionError("label of %d was read before prediction" % t)

        # Crucial: prospective candidate prediction is recorded here, while raw[t]
        # and therefore any target involving t are still unread.
        self.lifecycle_controller.on_pre_label_prediction(self, t, out)

        self.raw_seen[t] = np.asarray(
            self.bundle["arrays"]["raw_labels"][t], dtype=np.int64).copy()
        self.predictions["raw_labels"][t] = self.raw_seen[t]

        settled_now = t - 2
        if settled_now >= 0:
            target = self._mature_label(settled_now, t)
            self.predictions["labels"][settled_now] = target
            self.predictions["settled_at"][settled_now] = t
            self.settled[settled_now] = True
            self.buffer.append(settled_now)
            if len(self.buffer) > self.buffer_limit:
                self.buffer = self.buffer[-self.buffer_limit:]
            self.lifecycle_controller.on_matured(self, settled_now, target)

        if (t + 1) % self.update_every == 0:
            self.update(t)
            self.lifecycle_controller.on_after_live_update(self, t)
        self.model_version += 1
        self.cursor = t + 1
        self.lifecycle_controller._sync_session(self)
        return probability, classes

    def _checkpoint_payload(self, label, index):
        self.lifecycle_controller._sync_session(self)
        return super()._checkpoint_payload(label, index)

    def restore_checkpoint(self, path):
        payload = super().restore_checkpoint(path)
        shadow = payload.get("shadow_optimizer")
        if shadow is not None:
            if self.model.learner.shadow_id is None:
                raise AssertionError("checkpoint has shadow optimizer without shadow expert")
            self.create_shadow_optimizer()
            self.shadow_optimizer.load_state_dict(shadow["state"])
            self.shadow_optimizer_state = deepcopy(shadow["meta"])
        self.lifecycle_controller.load_state_dict(self.lifecycle_state)
        self.lifecycle_controller._sync_session(self)
        return payload
