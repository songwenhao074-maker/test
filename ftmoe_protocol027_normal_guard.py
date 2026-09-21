"""Protocol-027 normal-only historical regression guard.

Protocol-025/024 history is intentionally left unchanged.  This module only
changes the birth/reuse guard semantics registered for Protocol-027:
known-normal F0 rows protect binary detection NLL and FPR; positive rows are
reported but are not required for guard availability.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_v2c import ReplacementConsistentLifecycle, V2C_DEFAULT
from ftmoe_protocol025_session import Protocol025DynamicSession


NORMAL_NLL_RATIO_MAX = 1.02
NORMAL_NLL_ABS_TOL = 1e-6
NORMAL_FPR_DELTA_MAX = 0.01


def normal_detection_guard_report(live_detection_logits, candidate_detection_logits,
                                  target, *, absolute_tolerance=NORMAL_NLL_ABS_TOL):
    """Evaluate the registered historical-normal guard on binary logits only."""
    live = torch.as_tensor(live_detection_logits, dtype=torch.float32).reshape(-1, 2)
    cand = torch.as_tensor(candidate_detection_logits, dtype=torch.float32).reshape(-1, 2)
    target = torch.as_tensor(target, dtype=torch.long).reshape(-1)
    if live.shape != cand.shape or live.shape[0] != target.numel():
        raise ValueError("normal guard shape mismatch")

    normal = target == 0
    normal_rows = int(normal.sum().item())
    positive_rows = int((target > 0).sum().item())
    finite = bool(torch.isfinite(live).all() and torch.isfinite(cand).all())
    base = {
        "rows": int(target.numel()),
        "normal_rows": normal_rows,
        "positive_rows": positive_rows,
        "finite_logits": finite,
        "loss_definition": "binary_detection_cross_entropy_on_known_normal_rows",
        "normal_nll_ratio_max": NORMAL_NLL_RATIO_MAX,
        "normal_nll_absolute_tolerance": float(absolute_tolerance),
        "fpr_threshold": 0.5,
        "fpr_delta_max": NORMAL_FPR_DELTA_MAX,
        "positive_only_metrics": {
            "ap": None, "recall": None, "resource_macro_f1": None,
            "reason": "positive-only metrics are not part of the historical normal guard",
        },
    }
    if normal_rows == 0 or not finite:
        base.update({
            "available": False,
            "unavailable_reason": "no_normal_rows" if normal_rows == 0 else "nonfinite_logits",
            "live_loss": None,
            "candidate_loss": None,
            "relative_loss_increase": None,
            "relative_loss_increase_raw": None,
            "candidate_normal_nll_limit": None,
            "normal_nll_ok_direct": False,
            "live_fpr_0p5": None,
            "candidate_fpr_0p5": None,
            "fpr_delta_0p5": None,
        })
        return base

    zero = torch.zeros(normal_rows, dtype=torch.long)
    live_normal = live[normal]
    cand_normal = cand[normal]
    live_loss = float(F.cross_entropy(live_normal, zero).item())
    cand_loss = float(F.cross_entropy(cand_normal, zero).item())
    if not (math.isfinite(live_loss) and math.isfinite(cand_loss)):
        base.update({
            "available": False, "unavailable_reason": "nonfinite_normal_nll",
            "live_loss": None, "candidate_loss": None,
            "relative_loss_increase": None, "relative_loss_increase_raw": None,
            "candidate_normal_nll_limit": None, "normal_nll_ok_direct": False,
            "live_fpr_0p5": None, "candidate_fpr_0p5": None,
            "fpr_delta_0p5": None,
        })
        return base

    live_prob = torch.softmax(live_normal, -1)[:, 1]
    cand_prob = torch.softmax(cand_normal, -1)[:, 1]
    live_fpr = float((live_prob >= 0.5).float().mean().item())
    cand_fpr = float((cand_prob >= 0.5).float().mean().item())
    fpr_delta = float(cand_fpr - live_fpr)
    denom = max(abs(live_loss), 1e-12)
    raw_rel = float((cand_loss - live_loss) / denom)
    # Existing v2c decision code compares this field with the registered 0.02.
    # Subtract the separately registered absolute 1e-6 tolerance first so the
    # inherited comparison is exactly candidate <= live*1.02 + 1e-6.
    decision_rel = float((cand_loss - live_loss - float(absolute_tolerance)) / denom)
    limit = float(live_loss * NORMAL_NLL_RATIO_MAX + float(absolute_tolerance))
    base.update({
        "available": True,
        "unavailable_reason": None,
        "live_loss": live_loss,
        "candidate_loss": cand_loss,
        "relative_loss_increase": decision_rel,
        "relative_loss_increase_raw": raw_rel,
        "candidate_normal_nll_limit": limit,
        "normal_nll_ok_direct": bool(cand_loss <= limit),
        "live_fpr_0p5": live_fpr,
        "candidate_fpr_0p5": cand_fpr,
        "fpr_delta_0p5": fpr_delta,
    })
    return base


class Protocol027NormalGuardLifecycle(ReplacementConsistentLifecycle):
    """Protocol-024 v2c lifecycle with only the registered P027 guard change."""

    def _guard_report(self, session):
        anchor = self.guard_anchor
        labels = anchor["labels"]
        started = time.perf_counter()
        live_det, cand_det, targets = [], [], []
        with torch.no_grad():
            for left in range(0, int(labels.shape[0]), 32):
                right = min(int(labels.shape[0]), left + 32)
                context = s4.graph_context(anchor["ids"][left:right],
                                           anchor["before"][left:right],
                                           anchor["caps"][left:right])
                context["observed_history_length"] = anchor["observed_history_length"][left:right]
                out = session.model.predict_deployment(
                    anchor["x"][left:right], anchor["schedule"][left:right],
                    anchor["graph_x"][left:right], graph_context=context)
                correction, _ = self._birth_target_preview(
                    session, session.model._last_z.detach())
                live_det.append(out["detection_logits"].detach().cpu())
                cand_det.append((out["base_final_detection_logits"] +
                                 correction[..., :2]).detach().cpu())
                targets.append(labels[left:right].detach().cpu())
                self.extra_compute["guard_forwards"] += 1
        report = normal_detection_guard_report(
            torch.cat(live_det, 0), torch.cat(cand_det, 0), torch.cat(targets, 0))
        report["preview_topology"] = "four_generalists_plus_new_specialist"
        report["guard_role"] = "historical_normal_regression_guard"
        self.extra_compute["guard_seconds"] += time.perf_counter() - started
        return report

    def _reuse_guard_report(self, session, key):
        anchor = self.guard_anchor
        labels = anchor["labels"]
        started = time.perf_counter()
        live_det, cand_det, targets = [], [], []
        with torch.no_grad():
            for left in range(0, int(labels.shape[0]), 32):
                right = min(int(labels.shape[0]), left + 32)
                context = s4.graph_context(anchor["ids"][left:right],
                                           anchor["before"][left:right],
                                           anchor["caps"][left:right])
                context["observed_history_length"] = anchor["observed_history_length"][left:right]
                out = session.model.predict_deployment(
                    anchor["x"][left:right], anchor["schedule"][left:right],
                    anchor["graph_x"][left:right], graph_context=context)
                correction, _ = self._preview_specialist(
                    session, key, session.model._last_z.detach())
                live_det.append(out["detection_logits"].detach().cpu())
                cand_det.append((out["base_final_detection_logits"] +
                                 correction[..., :2]).detach().cpu())
                targets.append(labels[left:right].detach().cpu())
                self.extra_compute["reuse_guard_forwards"] += 1
        report = normal_detection_guard_report(
            torch.cat(live_det, 0), torch.cat(cand_det, 0), torch.cat(targets, 0))
        report["preview_topology"] = "four_generalists_plus_reused_specialist"
        report["guard_role"] = "historical_normal_regression_guard"
        self.extra_compute["reuse_guard_seconds"] += time.perf_counter() - started
        return report


class Protocol027DynamicSession(Protocol025DynamicSession):
    """P025 dynamic model/lifecycle with P027 normal-only guard semantics."""

    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg = dict(V2C_DEFAULT)
        cfg.update(v2c_config or {})
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        # Replace the historical two-class guard controller before any scored
        # prediction. No lifecycle event/candidate exists at this point.
        self.lifecycle_controller = Protocol027NormalGuardLifecycle(
            guard_anchor=guard_anchor, config=cfg)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
