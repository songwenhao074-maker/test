"""Protocol-029 passive dormant-memory utility diagnostic.

This module preserves the Protocol-028 online trajectory and adds only read-only
sidecar observations.  Passive predictions never enter lifecycle decisions,
training, routing state, optimizer state, or RNG state.
"""
from __future__ import annotations

from copy import deepcopy
import random
import time

import numpy as np
import torch

from ftmoe_protocol024_lifecycle import _loss_from_logits, _normal_anomaly_mean
from ftmoe_protocol024_v2c import V2C_DEFAULT
from ftmoe_protocol028_memory_protected import (
    Protocol028DynamicSession,
    Protocol028MemoryProtectedLifecycle,
)


def _unit(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-12)


def _tensor_versions(value):
    out = []
    if torch.is_tensor(value):
        out.append((id(value), int(value._version)))
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda x: str(x)):
            out.extend(_tensor_versions(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_tensor_versions(item))
    return tuple(out)


def _rng_snapshot():
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": (np_state[0], np_state[1].copy(), np_state[2], np_state[3], np_state[4]),
        "python": random.getstate(),
    }


def _rng_equal(a, b):
    if not torch.equal(a["torch"], b["torch"]):
        return False
    an, bn = a["numpy"], b["numpy"]
    if an[0] != bn[0] or an[2:] != bn[2:] or not np.array_equal(an[1], bn[1]):
        return False
    return a["python"] == b["python"]


def _normal_fp(logits, target):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.long).reshape(-1)
    normal = target == 0
    if not bool(normal.any()):
        return 0, 0
    prob = torch.softmax(logits, -1)[:, 1]
    return int((prob[normal] >= 0.5).sum().item()), int(normal.sum().item())


def evaluate_passive_candidate(pairs, guard, *, improvement_min, normal_allowance,
                               validation_fpr_allowance, guard_loss_allowance,
                               guard_fpr_allowance):
    """Apply the unchanged Protocol-028 reuse acceptance gates to passive pairs."""
    if not pairs:
        return {"available": False, "reason": "no_validation_pairs", "all_gates_pass": False}
    live = float(np.mean([p["live_loss"] for p in pairs]))
    cand = float(np.mean([p["candidate_loss"] for p in pairs]))
    improvement = float((live - cand) / max(abs(live), 1e-12))
    normal_pairs = [p for p in pairs if int(p["normal_rows"]) > 0]
    if normal_pairs:
        weights = [int(p["normal_rows"]) for p in normal_pairs]
        live_normal = float(np.average([p["live_normal_probability"] for p in normal_pairs], weights=weights))
        cand_normal = float(np.average([p["candidate_normal_probability"] for p in normal_pairs], weights=weights))
        normal_ok = bool(cand_normal <= live_normal + float(normal_allowance))
    else:
        live_normal = cand_normal = None
        normal_ok = False
    normal_rows = sum(int(p["normal_rows"]) for p in pairs)
    live_fp = sum(int(p["live_fp_0p5"]) for p in pairs)
    cand_fp = sum(int(p["candidate_fp_0p5"]) for p in pairs)
    live_fpr = live_fp / float(normal_rows) if normal_rows else None
    cand_fpr = cand_fp / float(normal_rows) if normal_rows else None
    fpr_delta = None if live_fpr is None else float(cand_fpr - live_fpr)
    validation_fpr_ok = bool(fpr_delta is not None and fpr_delta <= float(validation_fpr_allowance))
    loss_ok = bool(improvement >= float(improvement_min))
    guard_available = bool(guard.get("available"))
    guard_loss_ok = bool(guard_available and guard.get("relative_loss_increase") is not None
                         and float(guard["relative_loss_increase"]) <= float(guard_loss_allowance))
    guard_delta = guard.get("fpr_delta_0p5")
    guard_fpr_ok = bool(guard_available and guard_delta is not None
                        and float(guard_delta) <= float(guard_fpr_allowance))
    reasons = []
    if not loss_ok:
        reasons.append("relative_loss_improvement_below_1pct")
    if not normal_ok:
        reasons.append("mean_normal_probability_guard_failed_or_unavailable")
    if not validation_fpr_ok:
        reasons.append("validation_fpr_delta_failed_or_unavailable")
    if not guard_available:
        reasons.append("old_knowledge_guard_unavailable")
    elif not guard_loss_ok:
        reasons.append("old_knowledge_guard_loss_regression")
    elif not guard_fpr_ok:
        reasons.append("old_knowledge_guard_fpr_regression")
    return {
        "available": True,
        "validation_intervals": len(pairs),
        "live_mean_loss": live,
        "candidate_mean_loss": cand,
        "relative_improvement": improvement,
        "required_relative_improvement": float(improvement_min),
        "live_normal_probability": live_normal,
        "candidate_normal_probability": cand_normal,
        "normal_safety_ok": normal_ok,
        "validation_live_fpr_0p5": live_fpr,
        "validation_candidate_fpr_0p5": cand_fpr,
        "validation_fpr_delta_0p5": fpr_delta,
        "validation_fpr_ok": validation_fpr_ok,
        "guard": guard,
        "guard_loss_ok": guard_loss_ok,
        "guard_fpr_ok": guard_fpr_ok,
        "loss_ok": loss_ok,
        "all_gates_pass": bool(loss_ok and normal_ok and validation_fpr_ok and guard_loss_ok and guard_fpr_ok),
        "reject_reasons": reasons,
    }


class Protocol029PassiveAuditLifecycle(Protocol028MemoryProtectedLifecycle):
    """Protocol-028 lifecycle plus causally read-only dormant-memory observations."""

    def __init__(self, guard_anchor, config=None):
        super().__init__(guard_anchor=guard_anchor, config=config)
        self.passive_prediction_rows = []
        self.generalist_prediction_rows = []
        self.passive_by_index = {}
        self.generalist_by_index = {}
        self.opportunity_audit = []
        self.active_opportunities = []
        self.passive_overhead_seconds = 0.0
        self.passive_guard_seconds = 0.0
        self.passive_prediction_forwards = 0
        self.passive_guard_forwards = 0
        self.passive_state_checks = 0
        self.passive_state_check_failures = 0
        self._next_opportunity_id = 1

    def _integrity_snapshot(self, session):
        bank = session.model.learner
        optimizer_versions = _tensor_versions(getattr(session.optimizer, "state", {}))
        optimizer_versions += _tensor_versions(getattr(session, "optimizer_archive", {}))
        shadow = getattr(session, "shadow_optimizer", None)
        if shadow is not None:
            optimizer_versions += _tensor_versions(getattr(shadow, "state", {}))
        return {
            "parameter_versions": tuple((id(p), int(p._version)) for p in session.model.parameters()),
            "optimizer_versions": optimizer_versions,
            "rng": _rng_snapshot(),
            "topology": deepcopy(bank.topology_manifest()),
        }

    def _assert_integrity(self, session, before):
        after = self._integrity_snapshot(session)
        ok = (
            before["parameter_versions"] == after["parameter_versions"]
            and before["optimizer_versions"] == after["optimizer_versions"]
            and before["topology"] == after["topology"]
            and _rng_equal(before["rng"], after["rng"])
        )
        self.passive_state_checks += 1
        if not ok:
            self.passive_state_check_failures += 1
            raise AssertionError("Protocol029 passive observation mutated online state")

    def _generalists_only(self, session, z):
        bank = session.model.learner
        ids = list(self.generalist_ids)
        experts = {k: bank.experts[k] for k in ids}
        weights = {k: bank.router_weights[k] for k in ids}
        biases = {k: bank.router_biases[k] for k in ids}
        ramps = {k: float(bank.ramp.get(k, 1.0)) for k in ids}
        return bank._mixture(z, ids, experts, weights, biases, ramps)

    def _passive_prediction(self, session, index, output):
        started = time.perf_counter()
        before = self._integrity_snapshot(session)
        if not np.all(session.raw_seen[int(index)] == -1):
            raise AssertionError("Protocol029 passive prediction observed current label")
        bank = session.model.learner
        z = session.model._last_z.detach()
        dormant = sorted(
            [str(k) for k in bank.dormant_experts.keys() if str(k) in self.specialist_memory],
            key=int,
        )
        with torch.no_grad():
            general_correction, general_route = self._generalists_only(session, z)
            general_det = (output["base_final_detection_logits"] + general_correction[..., :2])[0].detach().cpu().numpy().astype(np.float32)
            general_cls = (output["base_final_class_logits"] + general_correction[..., 2:])[0].detach().cpu().numpy().astype(np.float32)
            general = {
                "prediction_index": int(index),
                "detection_logits": general_det,
                "class_logits": general_cls,
                "route_mean": float(general_route.mean().detach().cpu()),
                "recorded_before_label": True,
            }
            candidates = {}
            for key in dormant:
                correction, routed = self._preview_specialist(session, key, z)
                row = {
                    "prediction_index": int(index),
                    "expert_id": key,
                    "detection_logits": (output["base_final_detection_logits"] + correction[..., :2])[0].detach().cpu().numpy().astype(np.float32),
                    "class_logits": (output["base_final_class_logits"] + correction[..., 2:])[0].detach().cpu().numpy().astype(np.float32),
                    "route_mean": float(routed[..., -1].mean().detach().cpu()),
                    "recorded_before_label": True,
                }
                candidates[key] = row
                self.passive_prediction_rows.append(row)
                self.passive_prediction_forwards += 1
        self.generalist_prediction_rows.append(general)
        self.generalist_by_index[int(index)] = general
        self.passive_by_index[int(index)] = candidates
        self._assert_integrity(session, before)
        self.passive_overhead_seconds += time.perf_counter() - started
        return dormant

    def _similarity_snapshot(self, dormant_ids):
        width = int(self.v2b_config["reuse_recent_z_intervals"])
        dormant = list(dormant_ids)
        out = {"recent_z_available": len(self.recent_z), "recent_z_required": width, "candidates": {}}
        if len(self.recent_z) < width:
            for key in dormant:
                out["candidates"][key] = {
                    "similarity": None,
                    "required_similarity": float(self.specialist_memory[key]["similarity_threshold"]),
                    "similarity_ok": False,
                    "is_highest_similarity": False,
                }
            return out
        current = _unit(np.asarray(self.recent_z[-width:]).mean(axis=0))
        scored = []
        for key in dormant:
            mem = self.specialist_memory[key]
            sim = float(np.dot(current, np.asarray(mem["centroid"], dtype=np.float64)))
            threshold = float(mem["similarity_threshold"])
            scored.append((sim, -int(key), key, threshold))
        scored.sort(reverse=True)
        highest = scored[0][2] if scored else None
        for sim, _, key, threshold in scored:
            out["candidates"][key] = {
                "similarity": sim,
                "required_similarity": threshold,
                "similarity_ok": bool(sim >= threshold),
                "is_highest_similarity": key == highest,
            }
        return out

    @staticmethod
    def _classify_controller_events(events, phase_before):
        kinds = [e.get("kind") for e in events]
        selected = next((e for e in events if e.get("kind") == "reuse_candidate_selected"), None)
        if "reuse_due_skipped_busy" in kinds:
            outcome = "busy"
        elif "reuse_skipped_no_memory" in kinds:
            outcome = "no_memory"
        elif "reuse_skipped_insufficient_z" in kinds:
            outcome = "insufficient_z"
        elif "reuse_skipped_no_match" in kinds:
            outcome = "no_match"
        elif selected is not None:
            outcome = "selected"
        else:
            outcome = "no_explicit_reuse_event"
        return {
            "phase_before": phase_before,
            "busy": phase_before != "monitoring",
            "outcome": outcome,
            "selected_expert_id": None if selected is None else str(selected.get("expert_id")),
        }

    def _start_passive_opportunity(self, session, due_matured, phase_before, dormant_ids,
                                   similarity, event_delta):
        controller = self._classify_controller_events(event_delta, phase_before)
        opportunity = {
            "opportunity_id": self._next_opportunity_id,
            "due_cursor": int(session.cursor),
            "due_matured_count": int(due_matured),
            "validation_prediction_indices": list(range(int(session.cursor) + 1, int(session.cursor) + 17)),
            "frozen_dormant_ids": list(dormant_ids),
            "similarity": similarity,
            "controller": controller,
            "candidate_validation": {key: {"pairs": [], "censored": False, "censor_reason": None} for key in dormant_ids},
            "generalists_validation": {"pairs": [], "censored": False},
            "completed": False,
            "post_hoc_oracle_diagnostic": True,
        }
        self._next_opportunity_id += 1
        self.opportunity_audit.append(opportunity)
        if dormant_ids:
            self.active_opportunities.append(opportunity)
        else:
            opportunity["completed"] = True
            opportunity["completion_reason"] = "no_dormant_memory"

    def _passive_guard_report(self, session, key):
        started = time.perf_counter()
        before = self._integrity_snapshot(session)
        old_last_z = getattr(session.model, "_last_z", None)
        extra_before = deepcopy(self.extra_compute)
        try:
            report = super()._reuse_guard_report(session, key)
        finally:
            session.model._last_z = old_last_z
            self.extra_compute = extra_before
        self._assert_integrity(session, before)
        self.passive_guard_seconds += time.perf_counter() - started
        self.passive_guard_forwards += 1
        return report

    def _score_pair(self, live_det, live_cls, cand_det, cand_cls, target, index):
        live_loss = _loss_from_logits(live_det, live_cls, target)
        cand_loss = _loss_from_logits(cand_det, cand_cls, target)
        live_normal, normal_rows = _normal_anomaly_mean(live_det, target)
        cand_normal, _ = _normal_anomaly_mean(cand_det, target)
        live_fp, live_normal_rows = _normal_fp(live_det, target)
        cand_fp, cand_normal_rows = _normal_fp(cand_det, target)
        if live_normal_rows != cand_normal_rows or live_normal_rows != int(normal_rows):
            raise AssertionError("Protocol029 normal-row accounting mismatch")
        return {
            "index": int(index),
            "live_loss": float(live_loss),
            "candidate_loss": float(cand_loss),
            "normal_rows": int(normal_rows),
            "live_normal_probability": live_normal,
            "candidate_normal_probability": cand_normal,
            "live_fp_0p5": int(live_fp),
            "candidate_fp_0p5": int(cand_fp),
        }

    def _finalize_candidate(self, session, opportunity, key):
        rec = opportunity["candidate_validation"][key]
        if rec["censored"] or rec.get("decision") is not None:
            return
        needed = int(self.v2b_config["reuse_validation_intervals"])
        if len(rec["pairs"]) < needed:
            return
        bank = session.model.learner
        if key not in bank.dormant_experts:
            rec["censored"] = True
            rec["censor_reason"] = "expert_not_dormant_at_guard_time"
            return
        guard = self._passive_guard_report(session, key)
        rec["decision"] = evaluate_passive_candidate(
            rec["pairs"][:needed], guard,
            improvement_min=float(self.config["accept_relative_loss_improvement"]),
            normal_allowance=float(self.config["normal_probability_allowance"]),
            validation_fpr_allowance=float(self.v2_config["validation_fpr_delta_allowance"]),
            guard_loss_allowance=float(self.v2_config["guard_loss_relative_increase_allowance"]),
            guard_fpr_allowance=float(self.v2_config["guard_fpr_delta_allowance"]),
        )

    def _maybe_complete_opportunity(self, opportunity):
        if opportunity["completed"]:
            return
        needed = int(self.v2b_config["reuse_validation_intervals"])
        gen_done = len(opportunity["generalists_validation"]["pairs"]) >= needed
        candidates_done = all(
            row["censored"] or row.get("decision") is not None
            for row in opportunity["candidate_validation"].values()
        )
        if gen_done and candidates_done:
            opportunity["completed"] = True
            opportunity["completion_reason"] = "future_validation_matured"
            passing = [
                key for key, row in opportunity["candidate_validation"].items()
                if (row.get("decision") or {}).get("all_gates_pass")
            ]
            opportunity["passing_candidate_ids"] = passing
            opportunity["has_passing_candidate"] = bool(passing)
            controller = opportunity["controller"]
            similarity = opportunity["similarity"].get("candidates", {})
            opportunity["passing_blocked_by_busy"] = bool(passing and controller.get("busy"))
            opportunity["passing_blocked_by_similarity"] = bool(
                passing and not any(
                    similarity.get(k, {}).get("similarity_ok")
                    and similarity.get(k, {}).get("is_highest_similarity")
                    for k in passing
                )
            )
            selected = controller.get("selected_expert_id")
            opportunity["selected_candidate_passed"] = bool(selected in passing if selected else False)

    def _score_passive_matured(self, session, index, target):
        live_det = session.predictions["detection_logits"][int(index)]
        live_cls = session.predictions["class_logits"][int(index)]
        for opportunity in list(self.active_opportunities):
            if int(index) not in opportunity["validation_prediction_indices"]:
                continue
            general = self.generalist_by_index.get(int(index))
            if general is None:
                opportunity["generalists_validation"]["censored"] = True
            else:
                opportunity["generalists_validation"]["pairs"].append(
                    self._score_pair(live_det, live_cls, general["detection_logits"], general["class_logits"], target, index)
                )
            rows = self.passive_by_index.get(int(index), {})
            for key, rec in opportunity["candidate_validation"].items():
                if rec["censored"] or rec.get("decision") is not None:
                    continue
                row = rows.get(key)
                if row is None:
                    rec["censored"] = True
                    rec["censor_reason"] = "expert_not_dormant_during_future_validation"
                    rec["censor_prediction_index"] = int(index)
                    continue
                rec["pairs"].append(
                    self._score_pair(live_det, live_cls, row["detection_logits"], row["class_logits"], target, index)
                )
                self._finalize_candidate(session, opportunity, key)
            self._maybe_complete_opportunity(opportunity)
            if opportunity["completed"]:
                self.active_opportunities.remove(opportunity)

    def on_pre_label_prediction(self, session, index, output):
        self._passive_prediction(session, index, output)
        super().on_pre_label_prediction(session, index, output)

    def on_matured(self, session, index, target):
        self._score_passive_matured(session, index, target)
        next_matured = int(self.matured_count) + 1
        start = int(self.v2_config["proposal_start_matured"])
        reuse_every = int(self.v2b_config["reuse_every_matured"])
        reuse_due = next_matured >= start and next_matured % reuse_every == 0
        phase_before = self.phase
        dormant_before = sorted(
            [str(k) for k in session.model.learner.dormant_experts.keys() if str(k) in self.specialist_memory],
            key=int,
        )
        similarity = self._similarity_snapshot(dormant_before) if reuse_due else None
        event_start = len(self.events)
        super().on_matured(session, index, target)
        if reuse_due:
            self._start_passive_opportunity(
                session, next_matured, phase_before, dormant_before, similarity,
                deepcopy(self.events[event_start:]),
            )

    def finalize_passive_audits(self, steps):
        for opportunity in self.active_opportunities:
            opportunity["completed"] = True
            opportunity["completion_reason"] = "stream_end_censored"
            for row in opportunity["candidate_validation"].values():
                if row.get("decision") is None and not row["censored"]:
                    row["censored"] = True
                    row["censor_reason"] = "stream_end_before_16_future_predictions_matured"
            if len(opportunity["generalists_validation"]["pairs"]) < int(self.v2b_config["reuse_validation_intervals"]):
                opportunity["generalists_validation"]["censored"] = True
        self.active_opportunities = []


class Protocol029DiagnosticSession(Protocol028DynamicSession):
    """Unchanged Protocol-028 D session with a passive audit controller wrapper."""

    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg = dict(V2C_DEFAULT)
        cfg.update(v2c_config or {})
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        self.lifecycle_controller = Protocol029PassiveAuditLifecycle(
            guard_anchor=guard_anchor, config=cfg
        )
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator = "D_memory_protected_with_passive_audit"

    def comparator_manifest(self):
        base = deepcopy(super().comparator_manifest())
        base.update({
            "name": "D_memory_protected_with_passive_audit",
            "online_method": "Protocol028 D_memory_protected unchanged",
            "passive_audit": True,
            "passive_observation_changes_control": False,
        })
        return base
