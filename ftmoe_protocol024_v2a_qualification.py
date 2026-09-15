"""Qualification wiring for Protocol-024 v2a.

Kept separate from the periodic-policy module so validation-pair construction
is explicit: fixed-threshold FPR counts are stored before the decision routine
is allowed to run.
"""
from __future__ import annotations

import numpy as np
import torch

from ftmoe_protocol024_lifecycle import _loss_from_logits, _normal_anomaly_mean
from ftmoe_protocol024_v2a import (
    PeriodicResidualLifecycle,
    V2AProtocol024Session,
)


class QualifiedPeriodicResidualLifecycle(PeriodicResidualLifecycle):
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

        live_det = torch.as_tensor(record["live_detection_logits"], dtype=torch.float32)
        cand_det = torch.as_tensor(record["candidate_detection_logits"], dtype=torch.float32)
        target_t = torch.as_tensor(target, dtype=torch.long).reshape(-1)
        normal = target_t == 0
        if bool(normal.any()):
            live_fp = int((torch.softmax(live_det, -1)[:, 1][normal] >= 0.5).sum().item())
            cand_fp = int((torch.softmax(cand_det, -1)[:, 1][normal] >= 0.5).sum().item())
        else:
            live_fp = cand_fp = 0

        pair = {
            "index": int(index),
            "live_loss": live_loss,
            "candidate_loss": candidate_loss,
            "normal_rows": int(normal_rows),
            "live_normal_probability": live_normal,
            "candidate_normal_probability": candidate_normal,
            "live_fp_0p5": live_fp,
            "candidate_fp_0p5": cand_fp,
            "candidate_route_probability_mean": record.get(
                "candidate_route_probability_mean"),
            "prediction_cursor": record.get("prediction_cursor", int(index)),
            "label_published_at_cursor": int(session.cursor),
        }
        self.validation_pairs.append(pair)
        self.candidate_validation_indices.append(int(index))
        if len(self.validation_pairs) >= int(self.config["validation_intervals"]):
            self._decide_candidate(session)
        return True


class QualifiedV2AProtocol024Session(V2AProtocol024Session):
    def __init__(self, *args, guard_anchor=None, v2a_config=None, **kwargs):
        super().__init__(*args, guard_anchor=guard_anchor,
                         v2a_config=v2a_config, **kwargs)
        self.lifecycle_controller = QualifiedPeriodicResidualLifecycle(
            guard_anchor=guard_anchor, config=v2a_config)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self.lifecycle_controller._sync_session(self)
