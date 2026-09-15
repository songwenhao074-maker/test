"""Protocol-024 next_round_v2a: budgeted periodic shadow proposals.

This module intentionally leaves the v1 loss trigger untouched.  v2a proposes
background candidates on a fixed matured-count schedule that does not read
phase/law/event IDs or switch times.  Candidate qualification remains causal:
all candidate/live validation predictions are recorded before labels mature.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_lifecycle import (
    CausalResidualLifecycle,
    LifecycleProtocol024Session,
    _loss_from_logits,
    _normal_anomaly_mean,
)


V2A_DEFAULT = {
    "proposal_start_matured": 600,
    "proposal_every_matured": 256,
    "candidate_train_intervals": 64,
    "validation_intervals": 32,
    "accept_relative_loss_improvement": 0.01,
    "normal_probability_allowance": 0.01,
    "validation_fpr_delta_allowance": 0.01,
    "guard_loss_relative_increase_allowance": 0.02,
    "guard_fpr_delta_allowance": 0.01,
    "cooldown_intervals": 1,
    "retirement_enabled": False,
    "hard_delete_enabled": False,
    "debug_force_trigger_after": None,
    "diagnostic_window": 32,
    "diagnostic_stride": 8,
    "shadow_budget_version": "distinct64_single",
    "buffer_train_horizon_intervals": 128,
    "buffer_size": 64,
    "buffer_minibatch": 32,
    "buffer_update_every_matured": 4,
    "buffer_updates_per_opportunity": 4,
}


def split_anchor_train_guard(anchor, modulo=5, guard_remainder=0):
    """Deterministic target-independent split used by A/C/D and candidate guard."""
    labels = anchor["labels"]
    n = int(labels.shape[0])
    pos = np.arange(n, dtype=np.int64)
    guard_mask = (pos % int(modulo)) == int(guard_remainder)
    train_idx = torch.as_tensor(pos[~guard_mask], dtype=torch.long)
    guard_idx = torch.as_tensor(pos[guard_mask], dtype=torch.long)
    if train_idx.numel() == 0 or guard_idx.numel() == 0:
        raise ValueError("anchor train/guard split is empty")

    def subset(indices, role):
        out = {}
        for key, value in anchor.items():
            if key == "meta":
                continue
            out[key] = value.index_select(0, indices)
        meta = dict(anchor.get("meta") or {})
        meta.update({
            "v2_anchor_role": role,
            "v2_split_modulo": int(modulo),
            "v2_guard_remainder": int(guard_remainder),
            "v2_samples": int(indices.numel()),
            "v2_target_dependent_split": False,
        })
        out["meta"] = meta
        return out

    return subset(train_idx, "online_train"), subset(guard_idx, "candidate_guard")


def proposal_due(matured_count, start=600, every=256):
    matured_count = int(matured_count)
    start, every = int(start), int(every)
    return matured_count >= start and (matured_count - start) % every == 0


def fixed_threshold_fpr(logits, target, threshold=0.5):
    logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1, 2)
    target = torch.as_tensor(target, dtype=torch.long).reshape(-1)
    normal = target == 0
    if not bool(normal.any()):
        return None, 0
    probability = torch.softmax(logits, -1)[:, 1]
    fp = int((probability[normal] >= float(threshold)).sum().item())
    count = int(normal.sum().item())
    return fp / float(count), count


def deployable_pressure(host_features, capacities):
    """Prediction-time pressure from pre-step observable host state only.

    Collector order is [CPU, RAM size/read/write, Disk size/read/write].  Only
    CPU, RAM-size and Disk-size are divided by their matching physical capacity.
    """
    host = np.asarray(host_features, dtype=np.float64)
    caps = np.asarray(capacities, dtype=np.float64)
    if host.shape[-1] != 7 or caps.shape[-1] != 3:
        raise ValueError("unexpected Protocol-024 host/capacity shape")
    values = np.stack([host[..., 0], host[..., 1], host[..., 4]], axis=-1)
    return np.max(values / np.maximum(caps, 1e-12), axis=-1)


def _terms_from_saved(session, index, target):
    detection = torch.as_tensor(
        session.predictions["detection_logits"][int(index)], dtype=torch.float32)
    classes = torch.as_tensor(
        session.predictions["class_logits"][int(index)], dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.long)
    with torch.no_grad():
        terms = s4.supervised_terms(detection, classes, target_t)
    normal_mean, normal_rows = _normal_anomaly_mean(detection, target_t)
    return {
        "detection_ce": float(terms["detection"].cpu()),
        "resource_ce": float(terms["classification"].cpu()),
        "ranking": float(terms["ranking"].cpu()),
        "total": float(terms["total"].cpu()),
        "positive_rows": int((target_t.reshape(-1) > 0).sum().item()),
        "normal_rows": int(normal_rows),
        "normal_probability": normal_mean,
    }


class PeriodicResidualLifecycle(CausalResidualLifecycle):
    """v2a controller: fixed periodic proposals + causal qualification."""

    def __init__(self, guard_anchor, config=None):
        cfg = dict(V2A_DEFAULT)
        cfg.update(config or {})
        base_cfg = {
            "calibration_intervals": int(cfg["proposal_start_matured"]),
            "trigger_window": int(cfg["diagnostic_window"]),
            "consecutive_abnormal_windows": 2,
            "loss_percentile": 99.0,
            "candidate_train_intervals": int(cfg["candidate_train_intervals"]),
            "validation_intervals": int(cfg["validation_intervals"]),
            "accept_relative_loss_improvement": float(cfg["accept_relative_loss_improvement"]),
            "normal_probability_allowance": float(cfg["normal_probability_allowance"]),
            "cooldown_intervals": int(cfg["cooldown_intervals"]),
            "retirement_enabled": bool(cfg["retirement_enabled"]),
            "hard_delete_enabled": bool(cfg["hard_delete_enabled"]),
            "debug_force_trigger_after": None,
        }
        super().__init__(base_cfg)
        self.v2_config = cfg
        self.guard_anchor = guard_anchor
        self.phase = "monitoring"
        self.diagnostic_rows = []
        self.proposal_opportunities = 0
        self.proposal_skipped = 0
        self.accepted_ids = []
        self.rejected_ids = []
        self.capacity_blocked = 0
        self.training_buffer = []
        self.training_targets = {}
        self.training_elapsed_distinct = 0
        self.shadow_parameter_start_norm = None
        self.accepted_cursor = {}
        self.first_influence_seen = set()
        self.full_ramp_seen = set()
        self.v1_diagnostic_threshold = None
        self.extra_compute.setdefault("shadow_train_examples", 0)
        self.extra_compute.setdefault("guard_forwards", 0)
        self.extra_compute.setdefault("guard_seconds", 0.0)
        self.extra_compute.setdefault("route_diagnostic_forwards", 0)

    def state_dict(self):
        state = super().state_dict()
        state["v2"] = deepcopy({
            "v2_config": self.v2_config,
            "diagnostic_rows": self.diagnostic_rows,
            "proposal_opportunities": self.proposal_opportunities,
            "proposal_skipped": self.proposal_skipped,
            "accepted_ids": self.accepted_ids,
            "rejected_ids": self.rejected_ids,
            "capacity_blocked": self.capacity_blocked,
            "training_buffer": self.training_buffer,
            "training_targets": self.training_targets,
            "training_elapsed_distinct": self.training_elapsed_distinct,
            "shadow_parameter_start_norm": self.shadow_parameter_start_norm,
            "accepted_cursor": self.accepted_cursor,
            "first_influence_seen": list(self.first_influence_seen),
            "full_ramp_seen": list(self.full_ramp_seen),
            "v1_diagnostic_threshold": self.v1_diagnostic_threshold,
        })
        return state

    def load_state_dict(self, state):
        payload = dict(state)
        v2 = payload.pop("v2", None)
        super().load_state_dict(payload)
        if v2 is None:
            raise ValueError("v2 lifecycle checkpoint lacks v2 state")
        if dict(v2["v2_config"]) != self.v2_config:
            raise ValueError("v2 config mismatch on resume")
        for key, value in v2.items():
            if key == "v2_config":
                continue
            setattr(self, key, deepcopy(value))
        self.first_influence_seen = set(self.first_influence_seen)
        self.full_ramp_seen = set(self.full_ramp_seen)

    def _shadow_parent_delta(self, session):
        bank = session.model.learner
        if bank.shadow_id is None or self.candidate_parent_id is None:
            return None
        sid, pid = str(bank.shadow_id), str(self.candidate_parent_id)
        total = 0.0
        with torch.no_grad():
            for shadow, parent in zip(bank.shadow_experts[sid].parameters(),
                                      bank.experts[pid].parameters()):
                total += float(((shadow - parent) ** 2).sum().cpu())
            total += float(((bank.shadow_router_weights[sid] -
                             bank.router_weights[pid]) ** 2).sum().cpu())
            total += float(((bank.shadow_router_biases[sid] -
                             bank.router_biases[pid]) ** 2).sum().cpu())
        return math.sqrt(max(total, 0.0))

    def _start_candidate(self, session, reason):
        before = session.model.learner.shadow_id
        started = super()._start_candidate(session, reason)
        if not started:
            if before is None and session.model.learner.resident_count() >= \
                    session.model.learner.max_experts:
                self.capacity_blocked += 1
            return False
        self.training_buffer = []
        self.training_targets = {}
        self.training_elapsed_distinct = 0
        self.shadow_parameter_start_norm = self._shadow_parent_delta(session)
        self._event(session, "candidate_training_budget_registered",
                    candidate_id=self.candidate_id,
                    shadow_budget_version=self.v2_config["shadow_budget_version"],
                    initial_parent_delta=self.shadow_parameter_start_norm)
        return True

    def on_pre_label_prediction(self, session, index, output):
        super().on_pre_label_prediction(session, index, output)
        bank = session.model.learner
        if self.phase == "candidate_validation" and bank.shadow_id is not None:
            record = self.prelabel_candidate.get(int(index))
            if record is not None:
                with torch.no_grad():
                    _, routed = bank.preview_with_shadow(
                        session.model._last_z.detach(), shadow_ramp=1.0)
                record["candidate_route_probability_mean"] = float(
                    routed[..., -1].mean().cpu())
                record["prediction_cursor"] = int(session.cursor)
                self.extra_compute["route_diagnostic_forwards"] += 1

        for key in list(self.accepted_ids):
            if key not in bank.ramp:
                continue
            ramp = float(bank.ramp[key])
            if ramp > 0.0 and key not in self.first_influence_seen:
                self.first_influence_seen.add(key)
                self._event(session, "candidate_first_influence_prediction",
                            expert_id=key, prediction_index=int(index), ramp=ramp,
                            accepted_cursor=self.accepted_cursor.get(key))
            if ramp >= 1.0 and key not in self.full_ramp_seen:
                self.full_ramp_seen.add(key)
                self._event(session, "candidate_full_ramp_prediction",
                            expert_id=key, prediction_index=int(index), ramp=ramp,
                            accepted_cursor=self.accepted_cursor.get(key))
        self._sync_session(session)

    def _diagnostic(self, session, index, target, loss):
        row = _terms_from_saved(session, index, target)
        row.update({"index": int(index), "matured_count": int(self.matured_count),
                    "total_loss": float(loss)})
        host = np.asarray(session.bundle["arrays"]["host_features"], dtype=np.float64)
        if int(index) > 0:
            delta = np.abs(host[int(index)] - host[int(index) - 1])
            row["visible_feature_mean_abs_change"] = float(delta.mean())
            row["visible_feature_max_abs_change"] = float(delta.max())
        else:
            row["visible_feature_mean_abs_change"] = 0.0
            row["visible_feature_max_abs_change"] = 0.0
        self.diagnostic_rows.append(row)
        width = int(self.v2_config["diagnostic_window"])
        stride = int(self.v2_config["diagnostic_stride"])
        if self.matured_count >= width and self.matured_count % stride == 0:
            recent = self.diagnostic_rows[-width:]
            self._event(
                session, "diagnostic_window",
                window=width,
                detection_ce=float(np.mean([x["detection_ce"] for x in recent])),
                resource_ce=float(np.mean([x["resource_ce"] for x in recent])),
                ranking=float(np.mean([x["ranking"] for x in recent])),
                total=float(np.mean([x["total"] for x in recent])),
                positive_rows=int(sum(x["positive_rows"] for x in recent)),
                normal_probability=float(np.mean([
                    x["normal_probability"] for x in recent
                    if x["normal_probability"] is not None])) if any(
                        x["normal_probability"] is not None for x in recent) else None,
                visible_feature_mean_abs_change=float(np.mean([
                    x["visible_feature_mean_abs_change"] for x in recent])))

        start = int(self.v2_config["proposal_start_matured"])
        if self.matured_count == start:
            baseline = np.asarray(self.loss_history[:start], dtype=np.float64)
            width = int(self.v2_config["diagnostic_window"])
            means = np.asarray([baseline[i:i + width].mean()
                                for i in range(0, baseline.size - width + 1)])
            self.v1_diagnostic_threshold = float(np.percentile(means, 99.0))
            self._event(session, "v1_p99_diagnostic_only",
                        threshold=self.v1_diagnostic_threshold,
                        used_for_proposal=False)

    def _buffer_train(self, session, index, target):
        idx = int(index)
        if idx in self.training_targets:
            raise AssertionError("buffered candidate saw duplicate matured interval")
        self.training_buffer.append(idx)
        self.training_targets[idx] = np.asarray(target, dtype=np.int64).copy()
        self.training_buffer = self.training_buffer[-int(self.v2_config["buffer_size"]):]
        self.training_elapsed_distinct += 1
        every = int(self.v2_config["buffer_update_every_matured"])
        batch = int(self.v2_config["buffer_minibatch"])
        updates = int(self.v2_config["buffer_updates_per_opportunity"])
        if self.training_elapsed_distinct % every == 0 and len(self.training_buffer) >= batch:
            indices = list(self.training_buffer[-batch:])
            targets = np.stack([self.training_targets[i] for i in indices], axis=0)
            x, sched, graph, context = s4.window_batch(session.replay, indices)
            for _ in range(updates):
                started = time.perf_counter()
                base, z = session.model._base_forward(
                    x, sched, graph, graph_context=context, record=False)
                session.model.zero_grad(set_to_none=True)
                session.shadow_optimizer.zero_grad(set_to_none=True)
                correction, _ = session.model.learner.preview_with_shadow(z, shadow_ramp=1.0)
                detection = base["detection_logits"] + correction[..., :2]
                classes = base["class_logits"] + correction[..., 2:]
                terms = s4.supervised_terms(
                    detection, classes, torch.as_tensor(targets).long())
                terms["total"].backward()
                session.shadow_optimizer.step()
                session.model.zero_grad(set_to_none=True)
                self.extra_compute["shadow_train_steps"] += 1
                self.extra_compute["shadow_train_examples"] += len(indices)
                self.extra_compute["shadow_train_seconds"] += time.perf_counter() - started
        if self.training_elapsed_distinct >= int(
                self.v2_config["buffer_train_horizon_intervals"]):
            self.candidate_training_indices = sorted(self.training_targets.keys())
            for parameter in session.model.learner.shadow_parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            self.phase = "candidate_validation"
            self.prelabel_candidate = {}
            self.validation_pairs = []
            self._event(session, "candidate_training_complete",
                        candidate_id=self.candidate_id,
                        distinct_intervals=self.training_elapsed_distinct,
                        shadow_train_steps=self.extra_compute["shadow_train_steps"],
                        shadow_train_examples=self.extra_compute["shadow_train_examples"],
                        parent_parameter_l2_change=self._shadow_parent_delta(session))

    def _guard_report(self, session):
        anchor = self.guard_anchor
        labels = anchor["labels"]
        started = time.perf_counter()
        live_det, live_cls, cand_det, cand_cls, targets = [], [], [], [], []
        with torch.no_grad():
            for left in range(0, int(labels.shape[0]), 32):
                right = min(int(labels.shape[0]), left + 32)
                x = anchor["x"][left:right]
                schedule = anchor["schedule"][left:right]
                graph = anchor["graph_x"][left:right]
                context = s4.graph_context(anchor["ids"][left:right],
                                           anchor["before"][left:right],
                                           anchor["caps"][left:right])
                out = session.model.predict_deployment(
                    x, schedule, graph, graph_context=context)
                correction, _ = session.model.learner.preview_with_shadow(
                    session.model._last_z.detach(), shadow_ramp=1.0)
                live_det.append(out["detection_logits"].detach().cpu())
                live_cls.append(out["class_logits"].detach().cpu())
                cand_det.append((out["base_final_detection_logits"] +
                                 correction[..., :2]).detach().cpu())
                cand_cls.append((out["base_final_class_logits"] +
                                 correction[..., 2:]).detach().cpu())
                targets.append(labels[left:right].detach().cpu())
                self.extra_compute["guard_forwards"] += 1
        live_det = torch.cat(live_det, 0)
        live_cls = torch.cat(live_cls, 0)
        cand_det = torch.cat(cand_det, 0)
        cand_cls = torch.cat(cand_cls, 0)
        target = torch.cat(targets, 0)
        live_loss = _loss_from_logits(live_det, live_cls, target)
        candidate_loss = _loss_from_logits(cand_det, cand_cls, target)
        live_fpr, normal_rows = fixed_threshold_fpr(live_det, target)
        candidate_fpr, _ = fixed_threshold_fpr(cand_det, target)
        positive_rows = int((target.reshape(-1) > 0).sum().item())
        available = normal_rows > 0 and positive_rows > 0
        rel_increase = ((candidate_loss - live_loss) / max(abs(live_loss), 1e-12))
        fpr_delta = (None if live_fpr is None or candidate_fpr is None
                     else float(candidate_fpr - live_fpr))
        self.extra_compute["guard_seconds"] += time.perf_counter() - started
        return {
            "available": bool(available),
            "rows": int(target.numel()),
            "normal_rows": int(normal_rows),
            "positive_rows": int(positive_rows),
            "live_loss": float(live_loss),
            "candidate_loss": float(candidate_loss),
            "relative_loss_increase": float(rel_increase),
            "live_fpr_0p5": live_fpr,
            "candidate_fpr_0p5": candidate_fpr,
            "fpr_delta_0p5": fpr_delta,
        }

    def _decide_candidate(self, session):
        pairs = self.validation_pairs[:int(self.config["validation_intervals"])]
        live = float(np.mean([x["live_loss"] for x in pairs]))
        candidate = float(np.mean([x["candidate_loss"] for x in pairs]))
        improvement = (live - candidate) / max(abs(live), 1e-12)
        required = float(self.config["accept_relative_loss_improvement"])
        normal_pairs = [x for x in pairs if x["normal_rows"] > 0]
        if normal_pairs:
            weights = [x["normal_rows"] for x in normal_pairs]
            live_normal = float(np.average(
                [x["live_normal_probability"] for x in normal_pairs], weights=weights))
            candidate_normal = float(np.average(
                [x["candidate_normal_probability"] for x in normal_pairs], weights=weights))
            normal_ok = candidate_normal <= live_normal + float(
                self.config["normal_probability_allowance"])
        else:
            live_normal = candidate_normal = None
            normal_ok = False

        # Fixed-threshold FPR on the same causal validation rows.
        val_live_logits = np.stack([
            self.prelabel_candidate.get(x["index"], {}).get("live_detection_logits",
                session.predictions["detection_logits"][x["index"]]) for x in pairs], axis=0)
        # Candidate logits were popped from prelabel_candidate when matured, so
        # reconstruct from the scored pair's stored fixed-FPR fields when present.
        live_fp = sum(int(x.get("live_fp_0p5", 0)) for x in pairs)
        cand_fp = sum(int(x.get("candidate_fp_0p5", 0)) for x in pairs)
        normal_rows = sum(int(x["normal_rows"]) for x in pairs)
        val_live_fpr = live_fp / float(normal_rows) if normal_rows else None
        val_candidate_fpr = cand_fp / float(normal_rows) if normal_rows else None
        val_fpr_delta = (None if val_live_fpr is None else
                         float(val_candidate_fpr - val_live_fpr))
        val_fpr_ok = (val_fpr_delta is not None and val_fpr_delta <= float(
            self.v2_config["validation_fpr_delta_allowance"]))

        guard = self._guard_report(session)
        guard_loss_ok = bool(guard["available"] and
            guard["relative_loss_increase"] <= float(
                self.v2_config["guard_loss_relative_increase_allowance"]))
        guard_fpr_ok = bool(guard["available"] and
            guard["fpr_delta_0p5"] is not None and
            guard["fpr_delta_0p5"] <= float(
                self.v2_config["guard_fpr_delta_allowance"]))
        loss_ok = improvement >= required
        accept = bool(loss_ok and normal_ok and val_fpr_ok and
                      guard_loss_ok and guard_fpr_ok)
        reasons = []
        if not loss_ok:
            reasons.append("relative_loss_improvement_below_1pct")
        if not normal_ok:
            reasons.append("mean_normal_probability_guard_failed_or_unavailable")
        if not val_fpr_ok:
            reasons.append("validation_fpr_delta_failed_or_unavailable")
        if not guard["available"]:
            reasons.append("old_knowledge_guard_unavailable")
        elif not guard_loss_ok:
            reasons.append("old_knowledge_guard_loss_regression")
        elif not guard_fpr_ok:
            reasons.append("old_knowledge_guard_fpr_regression")
        route_values = [x.get("candidate_route_probability_mean") for x in pairs
                        if x.get("candidate_route_probability_mean") is not None]
        decision = {
            "candidate_id": self.candidate_id,
            "live_mean_loss": live,
            "candidate_mean_loss": candidate,
            "relative_improvement": float(improvement),
            "required_relative_improvement": required,
            "live_normal_probability": live_normal,
            "candidate_normal_probability": candidate_normal,
            "normal_safety_ok": bool(normal_ok),
            "validation_live_fpr_0p5": val_live_fpr,
            "validation_candidate_fpr_0p5": val_candidate_fpr,
            "validation_fpr_delta_0p5": val_fpr_delta,
            "validation_fpr_ok": bool(val_fpr_ok),
            "guard": guard,
            "guard_loss_ok": bool(guard_loss_ok),
            "guard_fpr_ok": bool(guard_fpr_ok),
            "candidate_route_probability_mean": (float(np.mean(route_values))
                if route_values else None),
            "parent_parameter_l2_change": self._shadow_parent_delta(session),
            "validation_intervals": len(pairs),
            "accepted": accept,
            "reject_reasons": reasons,
            "shadow_budget_version": self.v2_config["shadow_budget_version"],
        }
        self.last_decision = decision
        candidate_id = self.candidate_id
        if accept:
            activated = session.activate_shadow()
            if activated != candidate_id:
                raise AssertionError("activated candidate ID changed")
            self.accepted_ids.append(str(candidate_id))
            self.accepted_cursor[str(candidate_id)] = int(session.cursor)
            self._event(session, "candidate_accepted", **decision)
        else:
            session.discard_shadow()
            self.rejected_ids.append(str(candidate_id))
            self._event(session, "candidate_rejected", **decision)
        self.candidate_id = None
        self.candidate_parent_id = None
        self.phase = "cooldown"
        self.cooldown_left = int(self.config["cooldown_intervals"])
        self.prelabel_candidate = {}
        self.validation_pairs = []
        self.candidate_training_indices = []

    def _score_candidate_validation(self, session, index, target):
        record = self.prelabel_candidate.get(int(index))
        if record is None:
            return False
        det_live = torch.as_tensor(record["live_detection_logits"], dtype=torch.float32)
        det_candidate = torch.as_tensor(record["candidate_detection_logits"], dtype=torch.float32)
        target_t = torch.as_tensor(target, dtype=torch.long).reshape(-1)
        normal = target_t == 0
        record["live_fp_0p5"] = int((torch.softmax(det_live, -1)[:, 1][normal] >= 0.5).sum().item()) if bool(normal.any()) else 0
        record["candidate_fp_0p5"] = int((torch.softmax(det_candidate, -1)[:, 1][normal] >= 0.5).sum().item()) if bool(normal.any()) else 0
        record["label_published_at_cursor"] = int(session.cursor)
        result = super()._score_candidate_validation(session, index, target)
        if self.validation_pairs:
            pair = self.validation_pairs[-1]
            pair["live_fp_0p5"] = int(record["live_fp_0p5"])
            pair["candidate_fp_0p5"] = int(record["candidate_fp_0p5"])
            pair["candidate_route_probability_mean"] = record.get(
                "candidate_route_probability_mean")
            pair["prediction_cursor"] = record.get("prediction_cursor")
            pair["label_published_at_cursor"] = int(session.cursor)
        return result

    def on_matured(self, session, index, target):
        if session.arm != "D":
            return
        index = int(index)
        loss = self._current_live_loss(session, index, target)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite matured supervised loss")
        self.loss_history.append(loss)
        self.matured_count += 1
        self._diagnostic(session, index, target, loss)

        due = proposal_due(self.matured_count,
                           self.v2_config["proposal_start_matured"],
                           self.v2_config["proposal_every_matured"])
        if due:
            self.proposal_opportunities += 1
            if self.phase != "monitoring" or session.model.learner.shadow_id is not None:
                self.proposal_skipped += 1
                self._event(session, "proposal_skipped_busy",
                            reason="candidate_or_cooldown_busy",
                            controller_phase=self.phase,
                            shadow_id=session.model.learner.shadow_id)
            elif session.model.learner.resident_count() >= session.model.learner.max_experts:
                self.capacity_blocked += 1
                self._event(session, "proposal_skipped_capacity",
                            reason="resident_capacity_reached",
                            resident_count=session.model.learner.resident_count(),
                            max_experts=session.model.learner.max_experts)
            else:
                self._start_candidate(session, "periodic_background_budget")
                self._sync_session(session)
                return

        if self.phase == "candidate_training":
            if self.v2_config["shadow_budget_version"] == "distinct64_single":
                before_phase = self.phase
                self._candidate_train_one(session, index, target)
                self.extra_compute["shadow_train_examples"] += 1
                if before_phase == "candidate_training" and self.phase == "candidate_validation":
                    self._event(session, "candidate_training_diagnostics",
                                candidate_id=self.candidate_id,
                                parent_parameter_l2_change=self._shadow_parent_delta(session),
                                shadow_train_steps=self.extra_compute["shadow_train_steps"],
                                shadow_train_examples=self.extra_compute["shadow_train_examples"])
            elif self.v2_config["shadow_budget_version"] == "buffer128_4x4":
                self._buffer_train(session, index, target)
            else:
                raise ValueError("unknown shadow budget version")
            self._sync_session(session)
            return
        if self.phase == "candidate_validation":
            self._score_candidate_validation(session, index, target)
            self._sync_session(session)
            return
        if self.phase == "cooldown":
            self.cooldown_left -= 1
            if self.cooldown_left <= 0:
                self.phase = "monitoring"
                self._event(session, "cooldown_complete")
            self._sync_session(session)
            return
        if self.phase != "monitoring":
            raise RuntimeError("unexpected v2a lifecycle phase %s" % self.phase)
        self._sync_session(session)

    def on_after_live_update(self, session, index):
        super().on_after_live_update(session, index)
        self._sync_session(session)


class V2AProtocol024Session(LifecycleProtocol024Session):
    """Lifecycle session with the v2a periodic controller."""

    def __init__(self, *args, guard_anchor=None, v2a_config=None, **kwargs):
        if guard_anchor is None:
            raise ValueError("v2a requires a frozen candidate guard anchor")
        # Parent constructor creates a v1 controller only for wiring; replace it
        # before any prediction/update has occurred.
        super().__init__(*args, lifecycle_config={
            "calibration_intervals": 600,
            "trigger_window": 32,
            "consecutive_abnormal_windows": 2,
            "loss_percentile": 99.0,
            "candidate_train_intervals": int((v2a_config or {}).get(
                "candidate_train_intervals", 64)),
            "validation_intervals": 32,
            "accept_relative_loss_improvement": 0.01,
            "normal_probability_allowance": 0.01,
            "cooldown_intervals": 1,
            "retirement_enabled": False,
            "hard_delete_enabled": False,
            "debug_force_trigger_after": None,
        }, **kwargs)
        self.lifecycle_controller = PeriodicResidualLifecycle(
            guard_anchor=guard_anchor, config=v2a_config)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self.lifecycle_controller._sync_session(self)
