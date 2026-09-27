"""Protocol-032: one-factor F0 normal-NLL guard-budget experiment.

Historical Protocol-027/028/031 modules are intentionally unchanged.  The only
scientific change here is the F0 known-normal NLL acceptance bound used by both
birth and nonblocking dormant-specialist reuse:

    L_candidate <= L_live + max(0.02 * L_live, 0.01) + 1e-6

All other Protocol-031 lifecycle, causal validation, FPR, normal-probability,
training-budget, capacity, crossfade and reuse rules are inherited unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import torch
import torch.nn.functional as F

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol031_nonblocking_reuse import (
    P031_CONFIG,
    Protocol031DynamicSession,
    Protocol031NonblockingReuseLifecycle,
)

RELATIVE_TOLERANCE = 0.02
ABSOLUTE_FLOOR_NATS = 0.01
NUMERICAL_TOLERANCE = 1e-6
NORMAL_FPR_DELTA_MAX = 0.01


def guard_budget_report(live_detection_logits, candidate_detection_logits, target):
    """Return an auditable Protocol-032 F0 normal-only guard report.

    ``relative_loss_increase`` is deliberately reparameterized so the inherited
    birth comparison ``<= 0.02`` is mathematically equivalent to the new bound.
    The true raw relative increase and the historical Protocol-031 decision are
    retained separately for mechanism evidence.
    """
    live = torch.as_tensor(live_detection_logits, dtype=torch.float32).reshape(-1, 2)
    cand = torch.as_tensor(candidate_detection_logits, dtype=torch.float32).reshape(-1, 2)
    y = torch.as_tensor(target, dtype=torch.long).reshape(-1)
    if live.shape != cand.shape or live.shape[0] != y.numel():
        raise ValueError("Protocol032 normal guard shape mismatch")

    normal = y == 0
    normal_rows = int(normal.sum().item())
    positive_rows = int((y > 0).sum().item())
    finite = bool(torch.isfinite(live).all() and torch.isfinite(cand).all())
    base = {
        "protocol": "032",
        "guard_policy": "F0_normal_nll_relative_2pct_with_absolute_floor_0p01_nats",
        "rows": int(y.numel()),
        "normal_rows": normal_rows,
        "positive_rows": positive_rows,
        "finite_logits": finite,
        "loss_definition": "binary_detection_cross_entropy_on_known_normal_rows",
        "normal_nll_relative_tolerance": RELATIVE_TOLERANCE,
        "normal_nll_absolute_floor_nats": ABSOLUTE_FLOOR_NATS,
        "normal_nll_numerical_tolerance": NUMERICAL_TOLERANCE,
        "fpr_threshold": 0.5,
        "fpr_delta_max": NORMAL_FPR_DELTA_MAX,
    }
    if normal_rows == 0 or not finite:
        base.update({
            "available": False,
            "unavailable_reason": "no_normal_rows" if normal_rows == 0 else "nonfinite_logits",
            "live_loss": None,
            "candidate_loss": None,
            "relative_loss_increase": None,
            "relative_loss_increase_raw": None,
            "old_candidate_normal_nll_limit": None,
            "candidate_normal_nll_limit": None,
            "old_normal_nll_ok_direct": False,
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
            "available": False,
            "unavailable_reason": "nonfinite_normal_nll",
            "live_loss": None,
            "candidate_loss": None,
            "relative_loss_increase": None,
            "relative_loss_increase_raw": None,
            "old_candidate_normal_nll_limit": None,
            "candidate_normal_nll_limit": None,
            "old_normal_nll_ok_direct": False,
            "normal_nll_ok_direct": False,
            "live_fpr_0p5": None,
            "candidate_fpr_0p5": None,
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
    old_limit = float(live_loss * (1.0 + RELATIVE_TOLERANCE) + NUMERICAL_TOLERANCE)
    allowance = max(RELATIVE_TOLERANCE * live_loss, ABSOLUTE_FLOOR_NATS)
    new_limit = float(live_loss + allowance + NUMERICAL_TOLERANCE)

    # Birth in the inherited v2a path still checks relative_loss_increase <= .02.
    # Remove only the extra absolute-floor allowance above the historical 2%
    # allowance (and the common numerical tolerance) before that comparison.
    extra_floor = max(ABSOLUTE_FLOOR_NATS - RELATIVE_TOLERANCE * live_loss, 0.0)
    decision_rel = float(
        (cand_loss - live_loss - extra_floor - NUMERICAL_TOLERANCE) / denom
    )
    old_ok = bool(cand_loss <= old_limit)
    new_ok = bool(cand_loss <= new_limit)
    inherited_birth_ok = bool(decision_rel <= RELATIVE_TOLERANCE)
    if inherited_birth_ok != new_ok:
        raise AssertionError("Protocol032 birth/reuse NLL decision equivalence failed")

    base.update({
        "available": True,
        "unavailable_reason": None,
        "live_loss": live_loss,
        "candidate_loss": cand_loss,
        "relative_loss_increase": decision_rel,
        "relative_loss_increase_raw": raw_rel,
        "historical_relative_loss_increase": raw_rel,
        "old_candidate_normal_nll_limit": old_limit,
        "candidate_normal_nll_limit": new_limit,
        "absolute_allowance_nats": float(allowance),
        "extra_absolute_floor_over_2pct_nats": float(extra_floor),
        "old_normal_nll_ok_direct": old_ok,
        "normal_nll_ok_direct": new_ok,
        "inherited_birth_relative_check_ok": inherited_birth_ok,
        "live_fpr_0p5": live_fpr,
        "candidate_fpr_0p5": cand_fpr,
        "fpr_delta_0p5": fpr_delta,
    })
    return base


class Protocol032GuardBudgetLifecycle(Protocol031NonblockingReuseLifecycle):
    """Protocol-031 lifecycle with only the Protocol-032 F0 NLL bound changed."""

    def state_dict(self):
        state = super().state_dict()
        state["protocol032_guard_budget"] = {
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_floor_nats": ABSOLUTE_FLOOR_NATS,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        }
        return state

    def load_state_dict(self, state):
        payload = dict(state)
        p032 = payload.pop("protocol032_guard_budget", None)
        if p032 is None:
            raise ValueError("Protocol032 lifecycle checkpoint lacks guard-budget state")
        expected = {
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_floor_nats": ABSOLUTE_FLOOR_NATS,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        }
        if dict(p032) != expected:
            raise ValueError("Protocol032 guard-budget checkpoint mismatch")
        super().load_state_dict(payload)

    def _guard_report(self, session):
        """Birth F0 guard; all preview mechanics are inherited unchanged."""
        anchor = self.guard_anchor
        labels = anchor["labels"]
        started = time.perf_counter()
        live_det, cand_det, targets = [], [], []
        with torch.no_grad():
            for left in range(0, int(labels.shape[0]), 32):
                right = min(int(labels.shape[0]), left + 32)
                context = s4.graph_context(
                    anchor["ids"][left:right], anchor["before"][left:right],
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
        report = guard_budget_report(
            torch.cat(live_det, 0), torch.cat(cand_det, 0), torch.cat(targets, 0))
        report["preview_topology"] = "four_generalists_plus_new_specialist"
        report["guard_role"] = "F0_known_normal_regression_guard"
        report["decision_path"] = "birth_inherited_relative_loss_check_reparameterized_exactly"
        self.extra_compute["guard_seconds"] += time.perf_counter() - started
        return report

    def _reuse_guard_report_isolated(self, session, key):
        """Reuse F0 guard; isolated preview mechanics remain Protocol-031's."""
        old_last_z = getattr(session.model, "_last_z", None)
        started = time.perf_counter()
        anchor = self.guard_anchor
        labels = anchor["labels"]
        live_det, cand_det, targets = [], [], []
        try:
            with torch.no_grad():
                for left in range(0, int(labels.shape[0]), 32):
                    right = min(int(labels.shape[0]), left + 32)
                    context = s4.graph_context(
                        anchor["ids"][left:right], anchor["before"][left:right],
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
                    self.extra_compute["p031_reuse_guard_forwards"] += 1
        finally:
            session.model._last_z = old_last_z
        report = guard_budget_report(
            torch.cat(live_det, 0), torch.cat(cand_det, 0), torch.cat(targets, 0))
        report["preview_topology"] = "four_generalists_plus_reused_specialist"
        report["guard_role"] = "F0_known_normal_regression_guard"
        report["decision_path"] = "reuse_normal_nll_ok_direct"
        self.extra_compute["p031_reuse_guard_seconds"] += time.perf_counter() - started
        return report


class Protocol032DynamicSession(Protocol031DynamicSession):
    """Protocol-031 D with the single registered Protocol-032 guard change."""

    def __init__(self, *args, guard_anchor, v2c_config=None, **kwargs):
        cfg = dict(P031_CONFIG)
        cfg.update(v2c_config or {})
        super().__init__(*args, guard_anchor=guard_anchor, v2c_config=cfg, **kwargs)
        self.lifecycle_controller = Protocol032GuardBudgetLifecycle(
            guard_anchor=guard_anchor, config=cfg)
        self.lifecycle_state = self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"] = True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator = "D_guard_budget"

    def comparator_manifest(self):
        base = deepcopy(super().comparator_manifest())
        base.update({
            "name": "D_guard_budget",
            "protocol032_only_change": "F0 normal NLL acceptance bound",
            "F0_normal_nll_bound": "L_candidate <= L_live + max(0.02*L_live,0.01) + 1e-6",
            "absolute_floor_nats": ABSOLUTE_FLOOR_NATS,
            "source_lifecycle": "Protocol031NonblockingReuseLifecycle",
        })
        return base
