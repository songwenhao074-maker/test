"""Protocol 020 R1: frozen-base fixed residual online correction.

This module is deliberately self-contained.  The archived v4/S7 modules are
left untouched; R1 wraps the v4 model with four fixed, small residual experts
that see only the frozen detection-head input.  Training always uses the
``learner`` residual, while deployment can use the protected ``live``
snapshot, the ``assessment`` snapshot, or the ``previous`` snapshot.

The session keeps the prequential ordering used by S7: a prediction is
recorded before the current raw label is read, a matured record is settled
from its cached logits, and only then may the S7 update consume the label.
"""

from __future__ import annotations

import hashlib
import math
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from recovery.PreGANSrc.src.ftmoe_online_s7 import S7Session


PROTOCOL = "r1_fixed_residual_v1"
EXPERT_COUNT = 4
HIDDEN_SIZE = 64
EXPERT_HIDDEN_SIZE = 32
CORRECTION_SIZE = 5
ASSESSMENT_UPDATES = 10
ASSESSMENT_SUPPORT = 100
LIVE_SUPPORT = 100
ADMISSION_MARGIN = 0.995
ROLLBACK_MARGIN = 1.02


def _hash_tensors(items):
    """Stable hash over named tensors, including buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(items):
        if not isinstance(value, torch.Tensor):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _detach_cpu(value):
    """Convert a tensor/array/tree to a checkpoint-safe CPU value."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _detach_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_cpu(item) for item in value)
    return deepcopy(value)


class FixedResidualExpert(nn.Sequential):
    """One fixed 64 -> 32 -> 5 residual expert."""

    def __init__(self):
        super().__init__(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, EXPERT_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(EXPERT_HIDDEN_SIZE, CORRECTION_SIZE),
        )
        # The residual is an exact no-op at construction, but its final layer
        # has a useful first gradient.  Earlier layers become trainable after
        # the first optimizer step.
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class FixedResidualBank(nn.Module):
    """Four experts and one dense softmax router with fixed topology."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(HIDDEN_SIZE, EXPERT_COUNT)
        self.experts = nn.ModuleList(
            [FixedResidualExpert() for _ in range(EXPERT_COUNT)]
        )

    def forward(self, z):
        probabilities = torch.softmax(self.router(z), dim=-1)
        outputs = torch.stack([expert(z) for expert in self.experts], dim=-2)
        correction = (probabilities.unsqueeze(-1) * outputs).sum(dim=-2)
        return correction, probabilities


class FrozenResidualFTMoE(OnlineFTMoE):
    """Frozen v4 base plus four fixed online residual correction experts.

    At runtime the base class is exactly ``OnlineFTMoE`` as required by the R1
    design.
    """

    expert_count = EXPERT_COUNT
    correction_dim = CORRECTION_SIZE
    base_module_names = frozenset(
        ("learner", "live", "assessment", "previous")
    )

    def __init__(self, checkpoint, method, seed):
        # The inherited model is intentionally constructed as A.  This loads
        # the registered v4 checkpoint, leaves every base parameter frozen and
        # keeps its graph/capacity contract intact.
        super().__init__(checkpoint, "A", seed)
        self.requested_method = method
        self.correction_input = HIDDEN_SIZE

        self.learner = FixedResidualBank()
        self.live = FixedResidualBank()
        self.assessment = FixedResidualBank()
        self.previous = FixedResidualBank()
        self._copy_bank(self.learner, self.live)
        self._copy_bank(self.learner, self.assessment)
        self._copy_bank(self.learner, self.previous)

        self._last_z = None
        self._z_capture_handle = self.detection_head.register_forward_pre_hook(
            self._capture_detection_input
        )
        self.deployment_module_name = "live"
        self.deployment_alpha = 0.0
        self.assessment_enabled = False
        self.record_enabled = True
        self.last_prediction_bundle = None
        self.base_forward_count = 0
        self.base_forward_seconds = 0.0
        self.extra_head_forward_count = 0
        self.extra_head_forward_seconds = 0.0
        self.prediction_forward_count = 0
        self.prediction_forward_seconds = 0.0
        self.set_trainability()
        self.eval()
        self._force_base_eval()

    @staticmethod
    def _copy_bank(source, target):
        target.load_state_dict(source.state_dict(), strict=True)

    def _capture_detection_input(self, _module, arguments):
        if not arguments:
            raise RuntimeError("detection_head pre-hook received no feature tensor")
        z = arguments[0]
        if not isinstance(z, torch.Tensor) or z.shape[-1] != HIDDEN_SIZE:
            raise RuntimeError("R1 detection-head feature must have final size 64")
        # The feature is part of the frozen base.  Detaching it is what keeps
        # the learner graph from reaching the v4 encoder or any base head.
        self._last_z = z.detach()

    def _force_base_eval(self):
        for name, module in self.named_children():
            if name not in self.base_module_names:
                module.eval()
        for name in self.base_module_names:
            module = getattr(self, name, None)
            if module is not None:
                module.eval()

    def train(self, mode: bool = True):
        # No base module may be switched back to train mode by a caller.  The
        # correction has no dropout, so evaluation mode is also deterministic
        # for learner training.
        super().train(False)
        self._force_base_eval()
        return self

    def set_trainability(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("learner."))
        self._force_base_eval()

    @property
    def experts(self):
        """Convenience view of the four learner experts."""
        return self.learner.experts

    @property
    def router(self):
        """Convenience view of the learner router."""
        return self.learner.router

    def correction_parameters(self, bank_name="learner"):
        bank = getattr(self, bank_name)
        return list(bank.parameters())

    def set_deployment(self, bank_name, alpha):
        if bank_name not in self.base_module_names:
            raise ValueError("Unknown R1 deployment bank: %s" % bank_name)
        if bank_name == "learner" and float(alpha) != 1.0:
            raise ValueError("learner deployment is always at alpha=1")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("R1 deployment alpha must be in [0, 1]")
        self.deployment_module_name = bank_name
        self.deployment_alpha = alpha
        self._force_base_eval()

    def _base_forward(self, time_windows, schedule_windows=None,
                      graph_time_windows=None, graph_context=None,
                      record=False):
        self._force_base_eval()
        self._last_z = None
        started = time.perf_counter()
        previous_record = getattr(self.eagate, "record_enabled", False)
        self.eagate.record_enabled = bool(record and self.record_enabled)
        try:
            with torch.no_grad():
                output = super().forward(
                    time_windows, schedule_windows, graph_time_windows,
                    graph_context=graph_context,
                )
        finally:
            self.eagate.record_enabled = previous_record
        elapsed = time.perf_counter() - started
        self.base_forward_count += 1
        self.base_forward_seconds += elapsed
        if self._last_z is None:
            raise RuntimeError("R1 detection-head pre-hook did not capture z")
        return output, self._last_z

    @staticmethod
    def _score_from_logits(detection_logits, class_logits):
        return {
            "detection_logits": detection_logits.detach().cpu().clone(),
            "class_logits": class_logits.detach().cpu().clone(),
            "probability": detection_logits.softmax(-1)[..., 1].detach().cpu().clone(),
            "class_probability": class_logits.softmax(-1).detach().cpu().clone(),
        }

    def _apply_bank(self, base_detection, base_class, z, bank_name, alpha):
        if bank_name is None or alpha == 0.0:
            correction = base_detection.new_zeros(
                (*base_detection.shape[:-1], CORRECTION_SIZE)
            )
            probabilities = None
            return base_detection, base_class, correction, probabilities
        bank = getattr(self, bank_name)
        started = time.perf_counter()
        with torch.no_grad():
            correction, probabilities = bank(z)
        elapsed = time.perf_counter() - started
        self.extra_head_forward_count += 1
        self.extra_head_forward_seconds += elapsed
        return (base_detection + float(alpha) * correction[..., :2],
                base_class + float(alpha) * correction[..., 2:],
                correction, probabilities)

    def _prediction_bundle(self, time_windows, schedule_windows=None,
                           graph_time_windows=None, graph_context=None,
                           include_assessment=None, record=True):
        started = time.perf_counter()
        base_output, z = self._base_forward(
            time_windows, schedule_windows, graph_time_windows,
            graph_context=graph_context, record=record,
        )
        base_detection = base_output["detection_logits"]
        base_class = base_output["class_logits"]

        deployment_detection, deployment_class, deployment_correction, deployment_router = \
            self._apply_bank(
                base_detection, base_class, z,
                self.deployment_module_name, self.deployment_alpha,
            )
        if include_assessment is None:
            include_assessment = self.assessment_enabled
        assessment = None
        if include_assessment:
            assessment_detection, assessment_class, assessment_correction, assessment_router = \
                self._apply_bank(base_detection, base_class, z, "assessment", 1.0)
            assessment = self._score_from_logits(
                assessment_detection, assessment_class
            )
            assessment["correction_logits"] = assessment_correction.detach().cpu().clone()
            assessment["router_probabilities"] = (
                None if assessment_router is None else assessment_router.detach().cpu().clone()
            )

        deployment = self._score_from_logits(
            deployment_detection, deployment_class
        )
        deployment["correction_logits"] = deployment_correction.detach().cpu().clone()
        deployment["router_probabilities"] = (
            None if deployment_router is None else deployment_router.detach().cpu().clone()
        )
        base = self._score_from_logits(base_detection, base_class)
        elapsed = time.perf_counter() - started
        self.prediction_forward_count += 1
        self.prediction_forward_seconds += elapsed
        bundle = {
            "base": base,
            "deployment": deployment,
            "assessment": assessment,
            "z": z.detach().cpu().clone(),
            "deployment_module": self.deployment_module_name,
            "deployment_alpha": float(self.deployment_alpha),
        }
        self.last_prediction_bundle = _detach_cpu(bundle)

        output = dict(base_output)
        output.update({
            "detection_logits": deployment_detection,
            "class_logits": deployment_class,
            # ``base_detection_logits`` is intentionally preserved from the
            # archived model and remains its pre-CMHA intermediate.  R1 uses
            # these explicit final fields for the complete A prediction.
            "base_final_detection_logits": base_detection,
            "base_final_class_logits": base_class,
            "correction_logits": deployment_correction,
            "correction_router_probabilities": deployment_router,
            "router_probabilities": deployment_router,
            "r1_assessment_detection_logits": (
                None if assessment is None else assessment["detection_logits"]
            ),
            "r1_assessment_class_logits": (
                None if assessment is None else assessment["class_logits"]
            ),
        })
        return output

    def forward(self, time_windows: torch.Tensor,
                schedule_windows: torch.Tensor | None = None,
                graph_time_windows: torch.Tensor | None = None,
                graph_context: dict | None = None):
        # S7's training call always enters this path.  The base forward is
        # detached/no-grad and the learner correction has unit strength so a
        # deployment alpha of zero can never suppress its gradients.
        base_output, z = self._base_forward(
            time_windows, schedule_windows, graph_time_windows,
            graph_context=graph_context, record=False,
        )
        base_detection = base_output["detection_logits"]
        base_class = base_output["class_logits"]
        correction, probabilities = self.learner(z)
        output = dict(base_output)
        output.update({
            "detection_logits": base_detection + correction[..., :2],
            "class_logits": base_class + correction[..., 2:],
            "base_final_detection_logits": base_detection,
            "base_final_class_logits": base_class,
            "correction_logits": correction,
            "correction_router_probabilities": probabilities,
            "router_probabilities": probabilities,
            "r1_assessment_detection_logits": None,
            "r1_assessment_class_logits": None,
        })
        return output

    @torch.no_grad()
    def predict_online(self, time_windows, schedule_windows=None,
                       graph_time_windows=None, graph_context=None):
        """Predict with the currently configured deployment path."""
        self.eval()
        return self._prediction_bundle(
            time_windows, schedule_windows, graph_time_windows,
            graph_context=graph_context, record=True,
        )

    @torch.no_grad()
    def predict_deployment(self, time_windows, schedule_windows=None,
                           graph_time_windows=None, graph_context=None):
        """Evaluate deployment without changing routing counters."""
        return self._prediction_bundle(
            time_windows, schedule_windows, graph_time_windows,
            graph_context=graph_context, include_assessment=False, record=False,
        )

    def auxiliary_losses(self, output):
        # Keep the archived prototype separation term exactly as the parent
        # defined it.  The balance term belongs only to the fixed learner
        # router; frozen base routing is not an R1 trainable objective.
        prototype, _ = super().auxiliary_losses(output)
        probabilities = output.get("correction_router_probabilities")
        if probabilities is None:
            balance = prototype.new_zeros(())
        else:
            balance = ((probabilities.mean(dim=(0, 1)) - 1.0 / EXPERT_COUNT) ** 2).mean()
        return prototype, balance

    def frozen_hash(self):
        # Include every immutable base parameter and buffer, including the
        # graph host_capacity buffer after OnlineSessionV2 installs stream
        # capacity.  All four residual banks and their deployment metadata are
        # intentionally excluded.
        items = (
            (name, value) for name, value in self.state_dict().items()
            if not any(name == bank or name.startswith(bank + ".")
                       for bank in self.base_module_names)
        )
        return _hash_tensors(items)

    def state_hash(self):
        return _hash_tensors(self.state_dict().items())

    def learner_state_hash(self):
        return _hash_tensors(self.learner.state_dict().items())

    def correction_parameter_counts(self):
        return {
            name: sum(parameter.numel() for parameter in getattr(self, name).parameters())
            for name in self.base_module_names
        }

    def correction_memory_bytes(self):
        result = {}
        for name in self.base_module_names:
            result[name] = sum(
                parameter.numel() * parameter.element_size()
                for parameter in getattr(self, name).parameters()
            )
        return result


def _jsonable_score(score):
    """A raw-logit/probability score with no input tensor retained."""
    if score is None:
        return None
    result = {}
    for key in ("detection_logits", "class_logits", "probability", "class_probability"):
        value = score.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result[key] = np.asarray(value).tolist()
    return result


class R1Session(S7Session):
    """S7 replay/update session with fixed residual correction protection."""

    model_class = FrozenResidualFTMoE
    supports_dynamic = False

    def __init__(self, checkpoint, method, seed, replay, learning_rate,
                 replay_seed, normalization_v2, anchor_pool, anchor_teacher,
                 class_balance, window_fn="window_v3",
                 protection_enabled: bool = True):
        # The registered R1 method is C-residual.  Keep the S7 method argument
        # in the public signature for runner compatibility, but do not allow a
        # legacy B/D implementation to silently masquerade as R1.
        if method != "C":
            raise ValueError("R1 fixed residual requires method C")
        self.protection_enabled = bool(protection_enabled)
        super().__init__(
            checkpoint, method, seed, replay, learning_rate, replay_seed,
            normalization_v2, anchor_pool, anchor_teacher, class_balance,
            window_fn=window_fn,
        )

        self.assessment_enabled = False
        self.learner_version = int(self.update_number)
        self.live_version = 0
        # Weight version and deployment version are separate.  A rollback
        # changes alpha and therefore starts a new deployment evidence stream
        # even though the retained live weights are unchanged.
        self.deployment_version = 0
        self.assessment_version = 0
        self.assessment_snapshot_id = None
        self.assessment_learner_version = None
        self.assessment_start_cursor = None
        self.assessment_records = []
        self.assessment_block_records = []
        self.assessment_record_count = 0
        self.pending_predictions = {}
        self.prediction_ledger = []
        self.matured_ledger = []
        self.matured_indices = set()
        self.deployment_evidence = {}
        self.deployment_evidence_history = []
        self.evidence_checked_versions = set()
        self.evidence_block_counts = {}
        self.r1_events = []
        self.r1_update_events = []
        self.last_admission = None
        self.alpha = 0.0 if self.protection_enabled else 1.0
        self.active_parameter = "live" if self.protection_enabled else "learner"
        self.model.set_deployment(self.active_parameter, self.alpha)
        self.model.assessment_enabled = False

    @property
    def live_alpha(self):
        return float(self.alpha)

    # Stable aliases used by the R1 runner and by small audit tools.  They are
    # properties so restore-time replacement of the canonical lists cannot
    # leave a stale alias behind.
    @property
    def prediction_records(self):
        return self.prediction_ledger

    @property
    def records(self):
        return self.prediction_ledger

    @property
    def r1_ledger(self):
        return self.prediction_ledger

    @property
    def ledger(self):
        return self.prediction_ledger

    @property
    def pending_records(self):
        return self.pending_predictions

    def _set_deployment(self, bank_name, alpha, reason=None,
                        force_new_policy=False):
        changed = (str(bank_name) != getattr(self, "active_parameter", None) or
                   float(alpha) != float(getattr(self, "alpha", 0.0)) or
                   bool(force_new_policy))
        self.active_parameter = str(bank_name)
        self.alpha = float(alpha)
        if changed and hasattr(self, "deployment_version"):
            self.deployment_version += 1
        self.model.set_deployment(bank_name, alpha)
        if reason is not None:
            self.r1_update_events.append({
                "step": int(self.cursor),
                "event": "deployment_state",
                "active_parameter": self.active_parameter,
                "alpha": float(self.alpha),
                "live_version": int(self.live_version),
                "deployment_version": int(self.deployment_version),
                "reason": str(reason),
            })

    def _snapshot_learner(self, target_name, snapshot_id):
        target = getattr(self.model, target_name)
        self.model._copy_bank(self.model.learner, target)
        target.eval()
        self.model.set_trainability()
        next_index = self._next_prediction_index()
        return {
            "snapshot_id": str(snapshot_id),
            "target": str(target_name),
            "learner_version": int(self.learner_version),
            "created_cursor": int(self.cursor),
            "created_prediction_index": int(next_index),
            "created_prediction_step": int(next_index + 1),
        }

    def _next_prediction_index(self):
        """Return the first index that can actually use a new snapshot.

        S7 calls ``observe_matured`` while the current prediction is still in
        ``pending_predictions``.  At an update boundary the cursor has already
        advanced to the next index and there is no pending current row.  The
        distinction keeps snapshot metadata aligned with the first prediction
        that can observe the copied assessment bank.
        """
        current = int(self.cursor)
        return current + 1 if current in self.pending_predictions else current

    def _start_assessment(self):
        self.assessment_version += 1
        self.assessment_snapshot_id = "assessment-v%d" % self.assessment_version
        self.assessment_learner_version = int(self.learner_version)
        next_index = self._next_prediction_index()
        self.assessment_start_cursor = int(next_index)
        self.assessment_records = []
        self.assessment_block_records = []
        self.model.assessment_enabled = True
        event = self._snapshot_learner("assessment", self.assessment_snapshot_id)
        event.update({
            "event": "assessment_created",
            "assessment_version": int(self.assessment_version),
            "assessment_learner_version": int(self.assessment_learner_version),
            "support_required": ASSESSMENT_SUPPORT,
            "assessment_start_cursor": int(next_index),
            "evidence_from_prediction_index": int(next_index),
            "evidence_from_prediction_step": int(next_index + 1),
        })
        self.r1_events.append(event)
        return event

    def _capture_prediction_record(self, index, row):
        bundle = self.model.last_prediction_bundle
        if bundle is None:
            raise RuntimeError("R1 prediction bundle missing at prediction sink")
        base = _jsonable_score(bundle.get("base"))
        deployment = _jsonable_score(bundle.get("deployment"))
        assessment = _jsonable_score(bundle.get("assessment"))
        if base is None or deployment is None:
            raise RuntimeError("R1 prediction bundle lacks base/deployment logits")
        record = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "window_index": int(index),
            "prediction_step": int(index + 1),
            "model_version": int(row.get("model_version", self.update_number)),
            "learner_version": int(self.learner_version),
            "deployment_parameter": str(self.active_parameter),
            "deployment_alpha": float(self.alpha),
            "live_version": int(self.live_version),
            "deployment_version": int(self.deployment_version),
            "assessment_snapshot_id": self.assessment_snapshot_id,
            "assessment_version": int(self.assessment_version),
            "assessment_learner_version": self.assessment_learner_version,
            "base": base,
            "deployment": deployment,
            # Evaluation calls this side "live": it is the output actually
            # deployed at the prediction time.  It is cached, never rerun.
            "live": deepcopy(deployment),
            "assessment": assessment,
            "routing": {
                key: row.get(key) for key in
                ("expert_count", "mean_active", "unmatched_ratio")
                if key in row
            },
            "label_matured": False,
        }
        if assessment is not None:
            record["assessment_birth_prediction_step"] = int(
                self.assessment_start_cursor + 1
            )
        self.pending_predictions[int(index)] = record
        self.prediction_ledger.append(record)
        return record

    @staticmethod
    def _record_view(record):
        view = deepcopy(record)
        # The raw logits are intentionally retained in the session ledger;
        # this view is only for the lightweight prediction sink.
        for name in ("base", "deployment", "live", "assessment"):
            score = view.get(name)
            if isinstance(score, dict):
                score.pop("detection_logits", None)
                score.pop("class_logits", None)
        return view

    def step(self, prediction_sink=None):
        # S7.step invokes the sink after the one deployment forward and before
        # reading the current raw label.  Capturing here preserves that causal
        # boundary while retaining S7's exact sampler/update schedule.
        def sink(row):
            index = self.cursor
            record = self._capture_prediction_record(index, row)
            payload = dict(row)
            payload["r1_prediction"] = self._record_view(record)
            if prediction_sink is not None:
                prediction_sink(payload)

        return super().step(sink)

    @staticmethod
    def _stable_cross_entropy(logits, target, weight=None):
        if logits.numel() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(logits, target, weight=weight)

    def _supervised_loss(self, records, kind):
        if not records:
            return float("nan")
        detections = []
        classes = []
        labels = []
        for record in records:
            score = record.get(kind)
            if score is None:
                continue
            # A recorded window keeps its leading batch dimension (normally
            # [1, hosts, classes]), while its labels are [hosts].  Flatten
            # each window to the same host axis before concatenating records;
            # concatenating on dim 0 first would leave [windows, hosts, ...]
            # logits against a [windows * hosts] target vector.
            detection = torch.as_tensor(
                score["detection_logits"], dtype=torch.float32
            ).reshape(-1, 2)
            classification = torch.as_tensor(
                score["class_logits"], dtype=torch.float32
            ).reshape(-1, 3)
            target = torch.as_tensor(record["labels"], dtype=torch.long).reshape(-1)
            if (detection.shape[0] != classification.shape[0] or
                    detection.shape[0] != target.numel()):
                raise ValueError(
                    "R1 score/label host count mismatch: "
                    "%d detection, %d class, %d labels" % (
                        detection.shape[0], classification.shape[0],
                        target.numel()))
            detections.append(detection)
            classes.append(classification)
            labels.append(target)
        if not labels:
            return float("nan")
        detection = torch.cat(detections, dim=0)
        classification = torch.cat(classes, dim=0)
        target = torch.cat(labels, dim=0)
        anomaly = (target > 0).long()
        det_weight = torch.as_tensor(
            getattr(self, "detection_weights", [1.0, 1.0]),
            dtype=detection.dtype,
        )
        resource_weight = torch.as_tensor(
            getattr(self, "resource_weights", [1.0, 1.0, 1.0]),
            dtype=classification.dtype,
        )
        det_ce = self._stable_cross_entropy(
            detection.reshape(-1, 2), anomaly.reshape(-1), det_weight
        )
        positive = target > 0
        if positive.any():
            cls_ce = self._stable_cross_entropy(
                classification[positive], target[positive] - 1, resource_weight
            )
        else:
            cls_ce = detection.new_zeros(())

        ranking = detection.new_zeros(())
        if positive.any() and (~positive).any():
            anomaly_probability = detection.softmax(-1)[..., 1]
            class_probability = classification.softmax(-1)
            true_index = (target.clamp_min(1) - 1).unsqueeze(-1)
            true_probability = class_probability.gather(-1, true_index).squeeze(-1)
            positive_score = anomaly_probability[positive] * true_probability[positive]
            negative_score = anomaly_probability[~positive] * class_probability[~positive].max(-1).values
            ranking = F.softplus(
                0.15 + negative_score.unsqueeze(0) - positive_score.unsqueeze(1)
            ).mean()
        return float((.7 * det_ce + .3 * cls_ce + .5 * ranking).detach())

    @staticmethod
    def _error_counts(records, kind):
        fp = fn = n_normal = n_positive = 0
        for record in records:
            score = record.get(kind)
            if score is None:
                continue
            probability = np.asarray(score["probability"], dtype=np.float64)
            labels = np.asarray(record["labels"], dtype=np.int64)
            anomaly = labels > 0
            predicted = probability >= .5
            fp += int((predicted & ~anomaly).sum())
            fn += int((~predicted & anomaly).sum())
            n_normal += int((~anomaly).sum())
            n_positive += int(anomaly.sum())
        return {
            "fp": int(fp), "fn": int(fn),
            "n_normal": int(n_normal), "n_positive": int(n_positive),
        }

    def _score_block(self, records, kind):
        score = self._error_counts(records, kind)
        score["loss"] = float(self._supervised_loss(records, kind))
        score["total_loss"] = score["loss"]
        return score

    def _block_event(self, records, assessment_id, reason_prefix):
        candidate = self._score_block(records, "assessment")
        live = self._score_block(records, "live")
        base = self._score_block(records, "base")
        n_normal = int(base["n_normal"])
        n_positive = int(base["n_positive"])
        fp_budget = max(2, int(math.ceil(.005 * n_normal)))
        fn_budget = max(1, int(math.ceil(.05 * n_positive)))
        reasons = []
        loss_ok = candidate["loss"] < live["loss"] * ADMISSION_MARGIN
        base_safety_ok = candidate["loss"] <= base["loss"] * ROLLBACK_MARGIN
        fp_ok = candidate["fp"] <= base["fp"] + fp_budget
        fn_ok = candidate["fn"] <= base["fn"] + fn_budget
        if not loss_ok:
            reasons.append("candidate_loss_not_below_live_0.995")
        if not base_safety_ok:
            reasons.append("candidate_loss_above_base_1.02")
        if not fp_ok:
            reasons.append("candidate_fp_budget_exceeded")
        if not fn_ok:
            reasons.append("candidate_fn_budget_exceeded")
        accepted = bool(loss_ok and base_safety_ok and fp_ok and fn_ok)
        if accepted:
            reasons.append("all_admission_rules_passed")
        event = {
            "event": "assessment_evaluated",
            "protocol": PROTOCOL,
            "step": int(self.cursor),
            "assessment_snapshot_id": str(assessment_id),
            "assessment_version": int(self.assessment_version),
            "learner_version": int(self.learner_version),
            "live_version_before": int(self.live_version),
            "candidate": candidate,
            "live": live,
            "base": base,
            "accepted_by_rules": accepted,
            "admission_thresholds": {
                "candidate_loss_multiplier": ADMISSION_MARGIN,
                "candidate_base_loss_multiplier": ROLLBACK_MARGIN,
                "candidate_base_loss_max": float(base["loss"] * ROLLBACK_MARGIN),
                "fp_budget": int(fp_budget),
                "fn_budget": int(fn_budget),
                "candidate_fp_max": int(base["fp"] + fp_budget),
                "candidate_fn_max": int(base["fn"] + fn_budget),
            },
            "reasons": [str(reason_prefix)] + reasons,
            "support": {
                "count": int(len(records)),
                "first_window_index": int(records[0]["window_index"]),
                "last_window_index": int(records[-1]["window_index"]),
                "first_prediction_step": int(records[0]["prediction_step"]),
                "last_prediction_step": int(records[-1]["prediction_step"]),
                "first_matured_step": int(records[0]["matured_step"]),
                "last_matured_step": int(records[-1]["matured_step"]),
            },
            "window_indices": [int(record["window_index"]) for record in records],
        }
        return event, fp_budget, fn_budget

    def _apply_assessment_decision(self, event):
        if not self.protection_enabled:
            event["decision"] = "protection_disabled_no_admission"
            event["reasons"].append("protection_disabled")
            self.last_admission = event
            self.r1_events.append(event)
            return
        if event["accepted_by_rules"]:
            # Preserve the deployed snapshot before replacing it.
            self.model._copy_bank(self.model.live, self.model.previous)
            self.model._copy_bank(self.model.assessment, self.model.live)
            self.live_version += 1
            self._set_deployment(
                "live", 1.0, reason="assessment_accepted",
                force_new_policy=True,
            )
            self.deployment_evidence[self._evidence_key()] = []
            event["decision"] = "accepted_live_replaced"
            event["live_version_after"] = int(self.live_version)
            event["deployment_version_after"] = int(self.deployment_version)
            event["active_parameter_after"] = self.active_parameter
            event["alpha_after"] = float(self.alpha)
        else:
            event["decision"] = "rejected_live_unchanged"
            event["live_version_after"] = int(self.live_version)
            event["deployment_version_after"] = int(self.deployment_version)
            event["active_parameter_after"] = self.active_parameter
            event["alpha_after"] = float(self.alpha)
        self.last_admission = event
        self.r1_events.append(event)

    def _start_next_assessment_after_block(self):
        # This is called before the current step's S7 update.  Consequently
        # the next assessment starts from the learner that existed when the
        # evidence was judged and cannot claim the current prediction as
        # evidence for its own snapshot.
        self._start_assessment()

    def _evidence_key(self, record=None):
        """Identify one deployment policy, including alpha."""
        if record is None:
            version = self.live_version
            deployment_version = self.deployment_version
            alpha = self.alpha
        else:
            version = int(record.get("live_version", self.live_version))
            deployment_version = int(record.get(
                "deployment_version", self.deployment_version
            ))
            alpha = float(record.get("deployment_alpha", self.alpha))
        return "live%d-deployment%d-alpha%.6g" % (
            int(version), int(deployment_version), float(alpha)
        )

    def _maybe_rollback(self, record):
        if not self.protection_enabled:
            return None
        # A record can reach maturity on the same boundary on which an
        # assessment is accepted.  That prediction was made under the old
        # deployment and must never be credited to the newly accepted live
        # policy.
        if (self.active_parameter != "live" or self.alpha != 1.0 or
                int(record.get("live_version", self.live_version)) !=
                int(self.live_version) or
                ("deployment_parameter" in record and
                 record.get("deployment_parameter") != "live") or
                ("deployment_alpha" in record and
                 float(record.get("deployment_alpha")) != 1.0) or
                ("deployment_version" in record and
                 int(record.get("deployment_version")) !=
                 int(self.deployment_version))):
            return None
        version = int(record["live_version"])
        # Accept the compact integer-key form used by early R1 fixtures while
        # writing the full key for all new records.
        if ("deployment_version" not in record and
                version in self.deployment_evidence):
            evidence_key = version
        else:
            evidence_key = self._evidence_key(record)
        evidence = self.deployment_evidence.setdefault(evidence_key, [])
        evidence.append(record)
        if len(evidence) < LIVE_SUPPORT:
            return None
        block = list(evidence[:LIVE_SUPPORT])
        self.deployment_evidence[evidence_key] = evidence[LIVE_SUPPORT:]
        block_index = int(self.evidence_block_counts.get(evidence_key, 0) + 1)
        self.evidence_block_counts[evidence_key] = block_index
        live = self._score_block(block, "live")
        base = self._score_block(block, "base")
        fp_budget = max(2, int(math.ceil(.005 * base["n_normal"])))
        fn_budget = max(1, int(math.ceil(.05 * base["n_positive"])))
        reasons = []
        if live["loss"] > base["loss"] * ROLLBACK_MARGIN:
            reasons.append("live_loss_above_base_1.02")
        if live["fp"] > base["fp"] + fp_budget:
            reasons.append("live_fp_budget_exceeded")
        if live["fn"] > base["fn"] + fn_budget:
            reasons.append("live_fn_budget_exceeded")
        self.evidence_checked_versions.add(
            "%s#block%d" % (evidence_key, block_index)
        )
        event = {
            "event": "live_evidence_evaluated",
            "protocol": PROTOCOL,
            "step": int(self.cursor),
            "live_version": version,
            "deployment_version": int(record.get(
                "deployment_version", self.deployment_version
            )),
            "evidence_key": evidence_key,
            "block_index": block_index,
            "candidate": None,
            "live": live,
            "base": base,
            "support": {
                "count": len(block),
                "cumulative_count": int(block_index * LIVE_SUPPORT),
                "first_window_index": int(block[0]["window_index"]),
                "last_window_index": int(block[-1]["window_index"]),
                "first_matured_step": int(block[0]["matured_step"]),
                "last_matured_step": int(block[-1]["matured_step"]),
            },
            "fp_budget": int(fp_budget),
            "fn_budget": int(fn_budget),
            "reasons": reasons or ["rollback_rules_not_triggered"],
            "rollback": bool(reasons),
        }
        if reasons:
            self._set_deployment("live", 0.0, reason="live_protection_rollback")
            event["decision"] = "rollback_to_frozen_A"
            event["alpha_after"] = float(self.alpha)
            # Do not count the same failed live snapshot again, while keeping
            # its complete evidence above for traceability.
            self.deployment_evidence[evidence_key] = []
        else:
            event["decision"] = "live_retained"
        self.deployment_evidence_history.append({
            "live_version": version,
            "evidence_key": evidence_key,
            "block_index": block_index,
            "records": deepcopy(block),
            "event": deepcopy(event),
        })
        self.r1_events.append(event)
        return event

    def observe_matured(self, index):
        index = int(index)
        if index in self.matured_indices:
            for old_record in self.matured_ledger:
                if int(old_record["window_index"]) == index:
                    return old_record
            return None
        record = self.pending_predictions.get(index)
        if record is None:
            # S7's buffer may contain an old index after a restored state.  A
            # missing prediction is a causal error, never an invitation to
            # re-forward the historical input.
            raise RuntimeError("Missing R1 prediction record for matured window %d" % index)
        labels = np.asarray(self.predictions["labels"][index], dtype=np.int64)
        if labels.ndim != 1 or (labels < 0).any():
            raise RuntimeError("R1 maturity requires a fully revealed label")
        record["labels"] = labels.tolist()
        record["matured_step"] = int(index + 2)
        record["maturity_raw_label_index"] = int(index + 1)
        record["label_matured"] = True
        self.matured_indices.add(index)
        self.matured_ledger.append(record)
        self.pending_predictions.pop(index, None)

        assessment_matches = (
            record.get("assessment") is not None and
            record.get("assessment_snapshot_id") == self.assessment_snapshot_id
        )
        if assessment_matches:
            self.assessment_records.append(record)
            self.assessment_block_records.append(record)
            self.assessment_record_count += 1
            if len(self.assessment_block_records) >= ASSESSMENT_SUPPORT:
                block = list(self.assessment_block_records[:ASSESSMENT_SUPPORT])
                self.assessment_block_records = self.assessment_block_records[ASSESSMENT_SUPPORT:]
                event, _fp_budget, _fn_budget = self._block_event(
                    block, self.assessment_snapshot_id, "assessment_rules"
                )
                self._apply_assessment_decision(event)
                self._start_next_assessment_after_block()
        elif record.get("assessment") is not None:
            # The record was made under an assessment that closed at this
            # boundary.  It remains in the immutable prediction/maturity
            # ledgers, but cannot seed the next snapshot's evidence block.
            record["assessment_excluded_reason"] = "snapshot_closed_before_maturity"

        rollback_event = self._maybe_rollback(record)
        if rollback_event is not None:
            record["rollback_event"] = deepcopy(rollback_event)
        return record

    def update(self):
        # Keep the registered S7 32-sample/update/loss/anchor-distill path
        # byte-for-byte in the parent and only add snapshot bookkeeping after
        # it has completed.
        result = super().update()
        self.learner_version = int(self.update_number)
        if self.update_number == ASSESSMENT_UPDATES and self.assessment_snapshot_id is None:
            self._start_assessment()
        return result

    @torch.no_grad()
    def evaluate_reference(self, blocks=None):
        """Evaluate the currently deployed path on same-domain train anchors."""
        from train_ftmoe_end_to_end import metric_arrays

        pool = self.anchor_pool
        dets, clss = [], []
        for start in range(0, len(self.anchor_indices), 64):
            sl = slice(start, start + 64)
            context = self._context(
                pool["ids"][sl], pool["before"][sl], pool["caps"][sl]
            )
            out = self.model.predict_deployment(
                pool["x"][sl], pool["schedule"][sl], pool["graph_x"][sl],
                graph_context=context,
            )
            dets.append(out["detection_logits"].softmax(-1)[..., 1])
            clss.append(out["class_logits"].softmax(-1))
        score = metric_arrays(
            torch.cat(dets).numpy().reshape(-1),
            torch.cat(clss).numpy().reshape(-1, 3),
            pool["labels"].numpy().reshape(-1),
        )
        self.reference.append({
            "step": int(self.cursor),
            "source": "same_domain_train_anchor",
            "deployment_parameter": self.active_parameter,
            "deployment_alpha": float(self.alpha),
            "live_version": int(self.live_version),
            **score,
        })

    def finish(self):
        # OnlineSessionV2 supplies the guard raw label and final tolerance
        # label.  Finish then settles that cached final prediction exactly
        # once; it never performs a new model forward or optimizer update.
        super().finish()
        self.predictions["raw_labels"][self.cursor - 1] = self.raw_seen[self.cursor - 1]
        last = self.cursor - 1
        if last >= 0 and last not in self.matured_indices:
            self.buffer.append(last)
            self.observe_matured(last)

    def save(self):
        state = super().save()
        state["protocol"] = PROTOCOL
        state["r1"] = {
            "protocol": PROTOCOL,
            "protection_enabled": bool(self.protection_enabled),
            "active_parameter": self.active_parameter,
            "alpha": float(self.alpha),
            "learner_version": int(self.learner_version),
            "live_version": int(self.live_version),
            "deployment_version": int(self.deployment_version),
            "assessment_version": int(self.assessment_version),
            "assessment_snapshot_id": self.assessment_snapshot_id,
            "assessment_learner_version": self.assessment_learner_version,
            "assessment_start_cursor": self.assessment_start_cursor,
            "assessment_records": self.assessment_records,
            "assessment_block_records": self.assessment_block_records,
            "assessment_record_count": int(self.assessment_record_count),
            "pending_predictions": self.pending_predictions,
            "prediction_ledger": self.prediction_ledger,
            "matured_ledger": self.matured_ledger,
            "matured_indices": sorted(self.matured_indices),
            "deployment_evidence": self.deployment_evidence,
            "deployment_evidence_history": self.deployment_evidence_history,
            "evidence_block_counts": self.evidence_block_counts,
            "evidence_checked_versions": sorted(self.evidence_checked_versions),
            "events": self.r1_events,
            "update_events": self.r1_update_events,
            "last_admission": self.last_admission,
            "model_cost": {
                "base_forward_count": int(self.model.base_forward_count),
                "base_forward_seconds": float(self.model.base_forward_seconds),
                "extra_head_forward_count": int(self.model.extra_head_forward_count),
                "extra_head_forward_seconds": float(self.model.extra_head_forward_seconds),
                "prediction_forward_count": int(self.model.prediction_forward_count),
                "prediction_forward_seconds": float(self.model.prediction_forward_seconds),
            },
        }
        return state

    def restore(self, state):
        saved_protocol = state.get("protocol")
        r1_state = state.get("r1")
        if saved_protocol != PROTOCOL or not isinstance(r1_state, dict):
            raise ValueError("Cannot restore checkpoint without protocol %s" % PROTOCOL)
        if r1_state.get("protocol") != PROTOCOL:
            raise ValueError("R1 checkpoint protocol mismatch")
        if bool(r1_state.get("protection_enabled")) != bool(self.protection_enabled):
            raise ValueError("R1 protection_enabled changed on restore")
        super().restore(state)
        for name in (
                "learner_version", "live_version", "deployment_version",
                "assessment_version",
                "assessment_snapshot_id", "assessment_learner_version",
                "assessment_start_cursor",
                "assessment_records", "assessment_block_records",
                "assessment_record_count",
                "pending_predictions", "prediction_ledger", "matured_ledger",
                "deployment_evidence", "deployment_evidence_history",
                "last_admission"):
            if name in r1_state:
                setattr(self, name, r1_state[name])
        self.r1_events = list(r1_state.get("events", []))
        self.r1_update_events = list(r1_state.get("update_events", []))
        self.learner_version = int(self.learner_version)
        self.live_version = int(self.live_version)
        self.deployment_version = int(self.deployment_version)
        self.assessment_version = int(self.assessment_version)
        if self.assessment_learner_version is not None:
            self.assessment_learner_version = int(self.assessment_learner_version)
        self.assessment_record_count = int(getattr(
            self, "assessment_record_count", r1_state.get("assessment_record_count", 0)
        ))
        self.matured_indices = set(int(i) for i in r1_state.get("matured_indices", []))
        self.evidence_checked_versions = set(
            str(i) for i in r1_state.get("evidence_checked_versions", [])
        )
        self.evidence_block_counts = {
            key: int(value) for key, value in
            r1_state.get("evidence_block_counts", {}).items()
        }
        self.active_parameter = str(r1_state["active_parameter"])
        self.alpha = float(r1_state["alpha"])
        self.model.set_deployment(self.active_parameter, self.alpha)
        self.model.assessment_enabled = self.assessment_snapshot_id is not None
        costs = r1_state.get("model_cost", {})
        for name in (
                "base_forward_count", "base_forward_seconds",
                "extra_head_forward_count", "extra_head_forward_seconds",
                "prediction_forward_count", "prediction_forward_seconds"):
            if name in costs:
                setattr(self.model, name, costs[name])
        self.model.set_trainability()
        if self.model.frozen_hash() != self.initial_frozen_hash:
            raise AssertionError("R1 frozen base hash mismatch on resume")

    def r1_summary(self):
        counts = {
            "total": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "base": sum(
                parameter.numel() for name, parameter in self.model.named_parameters()
                if not name.startswith(("learner.", "live.", "assessment.", "previous."))
            ),
            **self.model.correction_parameter_counts(),
        }
        memory = {
            "total": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ),
            "trainable": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "base": sum(
                parameter.numel() * parameter.element_size()
                for name, parameter in self.model.named_parameters()
                if not name.startswith(("learner.", "live.", "assessment.", "previous."))
            ),
            **self.model.correction_memory_bytes(),
        }
        return {
            "protocol": PROTOCOL,
            "protection_enabled": bool(self.protection_enabled),
            "active_parameter": self.active_parameter,
            "alpha": float(self.alpha),
            "learner_version": int(self.learner_version),
            "deployment_version": int(self.deployment_version),
            "live_version": int(self.live_version),
            "assessment_version": int(self.assessment_version),
            "assessment_snapshot_id": self.assessment_snapshot_id,
            "assessment_learner_version": self.assessment_learner_version,
            "assessment_count": int(len(self.assessment_records)),
            "assessment_record_count_total": int(self.assessment_record_count),
            "assessment_pending_count": int(len(self.assessment_block_records)),
            "snapshot_event_count": int(len(self.r1_events)),
            "base_hash": self.initial_frozen_hash,
            "base_hash_current": self.model.frozen_hash(),
            "base_hash_unchanged": bool(self.model.frozen_hash() == self.initial_frozen_hash),
            "learner_state_hash": self.model.learner_state_hash(),
            "parameter_counts": counts,
            "parameter_memory_bytes": memory,
            "optimizer_updates": int(self.update_number),
            "updates_recorded": int(len(self.updates)),
            "exposure": dict(self.exposure),
            "rare_exposure": dict(getattr(self, "rare_exposure", {})),
            "prediction_cost": {
                "base_forward_count": int(self.model.base_forward_count),
                "base_forward_seconds": float(self.model.base_forward_seconds),
                "extra_head_forward_count": int(self.model.extra_head_forward_count),
                "extra_head_forward_seconds": float(self.model.extra_head_forward_seconds),
                "prediction_forward_count": int(self.model.prediction_forward_count),
                "prediction_forward_seconds": float(self.model.prediction_forward_seconds),
            },
            "deployment_evidence_counts": {
                str(version): int(len(records))
                for version, records in self.deployment_evidence.items()
            },
            "deployment_evidence_block_counts": {
                str(version): int(count)
                for version, count in self.evidence_block_counts.items()
            },
            "event_count": int(len(self.r1_events)),
            "admission_events": [
                event for event in self.r1_events
                if event.get("event") in ("assessment_evaluated", "live_evidence_evaluated")
            ],
        }


__all__ = [
    "PROTOCOL", "FixedResidualExpert", "FixedResidualBank",
    "FrozenResidualFTMoE", "R1Session",
]
