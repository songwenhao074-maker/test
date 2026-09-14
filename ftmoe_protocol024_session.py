"""Protocol-024 fair A/C/D strict-prequential integration.

This module deliberately reuses Protocol-023's ``PrequentialS4`` machinery so
A, C and D share the same stream, replay/anchor selection, loss, optimizer
family, update opportunities and prediction-before-label ordering.  D changes
only the residual-bank container and (later) lifecycle policy.

It also contains the evaluator/integrity fixes required before the registered
Protocol-024 pilot: complete two-record tail settlement, same-host temporal
onset, positive-only resource macro-F1, and topology-aware dynamic-bank
checkpoint/restore helpers.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from recovery.PreGANSrc.src.ftmoe_online_r1 import FrozenResidualFTMoE


class DynamicResidualFTMoE(FrozenResidualFTMoE):
    """Standalone fixed-C model with only its learner residual bank dynamic."""

    def __init__(self, checkpoint, method, seed, max_experts=8, ramp_updates=10):
        if method != "D":
            raise ValueError("Protocol-024 dynamic residual model requires method D")
        super().__init__(checkpoint, "C", seed)
        source = self.learner
        self.learner = DynamicResidualBank(
            source, max_experts=max_experts, ramp_updates=ramp_updates)
        self.requested_method = "D"
        self.set_trainability()
        self.set_deployment("learner", 1.0)

    def auxiliary_losses(self, output):
        # Preserve the archived prototype term, but make router balancing use
        # the *current* active count instead of Protocol-023's fixed four.
        prototype, _ = super().auxiliary_losses(output)
        probabilities = output.get("correction_router_probabilities")
        if probabilities is None:
            balance = prototype.new_zeros(())
        else:
            count = probabilities.shape[-1]
            balance = ((probabilities.mean(dim=(0, 1)) - 1.0 / count) ** 2).mean()
        return prototype, balance


class Protocol024Session(s4.PrequentialS4):
    """Protocol-024 session with identical A/C mechanics and a fair D arm."""

    def __init__(self, arm, seed, replay_bundle, budget, out_dir,
                 probe_paths=None, anchor=None, learning_rate=1e-4,
                 max_experts=8, ramp_updates=10):
        if arm not in ("A", "C", "D"):
            raise ValueError("arm must be A, C, or D")
        # Build the registered A/C session first. For D, convert that exact
        # learner bank in place BEFORE any scored interval. This is stronger
        # than constructing a second nominally identical model: it guarantees
        # that D starts from the exact same C tensors and RNG realization.
        super().__init__("A" if arm == "A" else "C", seed, replay_bundle,
                         budget, out_dir, probe_paths=probe_paths, anchor=anchor,
                         learning_rate=learning_rate)
        self.arm = arm
        if arm != "D":
            return

        source = self.model.learner
        self.model.learner = DynamicResidualBank(
            source, max_experts=max_experts, ramp_updates=ramp_updates)
        self.model.requested_method = "D"
        self.model.set_trainability()
        self.model.set_deployment("learner", 1.0)
        self.model.eval()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=1e-4)
        self.frozen_hash = self.model.frozen_hash()
        self.learner_hash = self.model.learner_state_hash()
        self.protocol024_dynamic_container = True
        self.lifecycle_events = []

    def finish(self):
        """Settle BOTH still-pending tail records after reading the guard row.

        Protocol-023's historical runner settled only ``steps-1`` here, leaving
        ``steps-2`` unscored.  Protocol-024 does not mutate that historical
        artifact; it fixes the behavior in the new session.
        """
        if self.cursor != self.steps:
            raise ValueError("cannot finalize a partial stream (%d/%d)"
                             % (self.cursor, self.steps))
        self.raw_seen[self.steps] = np.asarray(
            self.bundle["arrays"]["raw_labels"][self.steps],
            dtype=np.int64).copy()
        self.predictions["raw_labels"][self.steps - 1] = \
            self.raw_seen[self.steps - 1]

        for index in range(max(0, self.steps - 2), self.steps):
            if self.settled[index]:
                continue
            label = s4.tolerance_label(self.raw_seen, index, self.steps)
            self.predictions["labels"][index] = label
            self.predictions["settled_at"][index] = self.steps
            self.settled[index] = True
            self.buffer.append(index)
        if len(self.buffer) > self.buffer_limit:
            self.buffer = self.buffer[-self.buffer_limit:]
        self.assert_frozen()


def same_host_onset_metrics(probability, labels, horizon=1):
    """Temporal onset AP without flattening hosts before shifting in time."""
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probability.shape != labels.shape or labels.ndim != 2:
        raise ValueError("probability and labels must both be [time, host]")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    steps, hosts = labels.shape
    if steps <= horizon:
        return {"ap": None, "positives": 0, "rows": 0, "horizon": horizon}

    current_fault = labels > 0
    future_fault = np.zeros((steps - horizon, hosts), dtype=bool)
    for offset in range(1, horizon + 1):
        future_fault |= labels[offset:steps - horizon + offset] > 0
    eligible = ~current_fault[:steps - horizon]
    target = future_fault[eligible].astype(np.int64)
    score = probability[:steps - horizon][eligible]
    return {
        "ap": s4.average_precision(target, score),
        "positives": int(target.sum()),
        "rows": int(target.size),
        "horizon": int(horizon),
    }


def positive_only_resource_macro_f1(class_probability, labels):
    """Resource diagnosis F1 on CPU/RAM/Disk fault rows only."""
    class_probability = np.asarray(class_probability)
    labels = np.asarray(labels, dtype=np.int64)
    flat_labels = labels.reshape(-1)
    flat_probability = class_probability.reshape(-1, 3)
    positive = flat_labels > 0
    if not positive.any():
        return {"macro_f1": None, "rows": 0, "per_class": []}
    result = s4.resource_macro_f1(flat_probability[positive], flat_labels[positive])
    result["rows"] = int(positive.sum())
    return result


def export_dynamic_bank(bank):
    """Return a torch-save-safe snapshot including non-Parameter topology."""
    if not isinstance(bank, DynamicResidualBank):
        raise TypeError("bank must be DynamicResidualBank")
    topology = deepcopy(bank.topology_manifest())
    groups = {}
    for group_name, experts, weights, biases in (
        ("active", bank.experts, bank.router_weights, bank.router_biases),
        ("dormant", bank.dormant_experts, bank.dormant_router_weights,
         bank.dormant_router_biases),
        ("shadow", bank.shadow_experts, bank.shadow_router_weights,
         bank.shadow_router_biases),
    ):
        payload = {}
        for key in experts.keys():
            payload[str(key)] = {
                "expert": {name: tensor.detach().cpu().clone()
                           for name, tensor in experts[key].state_dict().items()},
                "router_weight": weights[key].detach().cpu().clone(),
                "router_bias": biases[key].detach().cpu().clone(),
            }
        groups[group_name] = payload
    return {"topology": topology, "groups": groups}


def restore_dynamic_bank(source_bank, snapshot):
    """Restore dynamic topology + tensors from :func:`export_dynamic_bank`."""
    topology = deepcopy(snapshot["topology"])
    target = DynamicResidualBank(
        source_bank, max_experts=int(topology["max_experts"]),
        ramp_updates=int(topology["ramp_updates"]))
    active = [str(x) for x in topology["active_ids"]]
    dormant = [str(x) for x in topology["dormant_ids"]]
    shadow = topology.get("shadow_id")
    shadow = None if shadow is None else str(shadow)
    next_id = int(topology["next_id"])

    # Dynamic ids are allocated monotonically. Recreate every allocated id in
    # order; a live shadow, if present, must be the last allocated id because
    # the bank supports at most one candidate at a time.
    if shadow is not None and int(shadow) != next_id - 1:
        raise ValueError("snapshot shadow id is not the most recent allocation")
    while target.next_id < next_id:
        parent = target.ids[0]
        key = target.create_shadow(parent)
        if shadow is not None and key == shadow:
            break
        target.activate_shadow()
        target.ramp[key] = 1.0

    for key in list(target.ids):
        if key in dormant:
            target.retire(key)
    if set(target.ids) != set(active):
        raise ValueError("snapshot active topology cannot be reconstructed")
    if set(target.dormant_experts.keys()) != set(dormant):
        raise ValueError("snapshot dormant topology cannot be reconstructed")
    if target.shadow_id != shadow:
        raise ValueError("snapshot shadow topology cannot be reconstructed")

    target.ids = list(active)  # restore routing order, not merely membership
    target.ramp = {str(k): float(v) for k, v in topology["ramp"].items()}
    target.next_id = next_id

    for group_name, experts, weights, biases in (
        ("active", target.experts, target.router_weights, target.router_biases),
        ("dormant", target.dormant_experts, target.dormant_router_weights,
         target.dormant_router_biases),
        ("shadow", target.shadow_experts, target.shadow_router_weights,
         target.shadow_router_biases),
    ):
        for key, payload in snapshot["groups"][group_name].items():
            experts[key].load_state_dict(payload["expert"], strict=True)
            weights[key].data.copy_(payload["router_weight"].to(weights[key]))
            biases[key].data.copy_(payload["router_bias"].to(biases[key]))

    for key in target.dormant_experts.keys():
        for parameter in target.dormant_experts[key].parameters():
            parameter.requires_grad_(False)
        target.dormant_router_weights[key].requires_grad_(False)
        target.dormant_router_biases[key].requires_grad_(False)
    return target
