"""Protocol-024 dynamic residual bank prototype.

Provides a fair dynamic counterpart to Protocol-023's FixedResidualBank:
- same residual experts and router rows at t=0;
- same initial active expert count;
- lifecycle management is the added capability.

Lifecycle policy (trigger, causal shadow validation, retirement decision) remains
outside this bank. Protocol-020's D-v3 trigger/validation logic is reference
material, but its OnlineEAGateV3 container must not be used directly as the
Protocol-024 D arm because that would change the model family.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import torch
from torch import nn


class DynamicResidualBank(nn.Module):
    """Expandable/retirable bank initialized exactly from a fixed residual bank."""

    def __init__(self, source_bank, max_experts=8, ramp_updates=10):
        super().__init__()
        if not hasattr(source_bank, "router") or not hasattr(source_bank, "experts"):
            raise TypeError("source_bank must expose .router and .experts")
        if source_bank.router.weight.ndim != 2:
            raise ValueError("source router must be a Linear-like [N,H] weight")
        n_experts, hidden = source_bank.router.weight.shape
        if n_experts != len(source_bank.experts):
            raise ValueError("router rows and expert count disagree")
        if n_experts < 1:
            raise ValueError("at least one initial expert is required")
        if max_experts < n_experts:
            raise ValueError("max_experts cannot be below initial expert count")
        if ramp_updates < 1:
            raise ValueError("ramp_updates must be >= 1")

        self.hidden = int(hidden)
        self.max_experts = int(max_experts)
        self.ramp_updates = int(ramp_updates)

        self.experts = nn.ModuleDict()
        self.router_weights = nn.ParameterDict()
        self.router_biases = nn.ParameterDict()
        self.dormant_experts = nn.ModuleDict()
        self.dormant_router_weights = nn.ParameterDict()
        self.dormant_router_biases = nn.ParameterDict()
        self.shadow_experts = nn.ModuleDict()
        self.shadow_router_weights = nn.ParameterDict()
        self.shadow_router_biases = nn.ParameterDict()

        self.ids = []
        for i in range(n_experts):
            key = str(i)
            self.ids.append(key)
            self.experts[key] = deepcopy(source_bank.experts[i])
            self.router_weights[key] = nn.Parameter(
                source_bank.router.weight[i].detach().clone())
            self.router_biases[key] = nn.Parameter(
                source_bank.router.bias[i].detach().clone())

        self.next_id = n_experts
        self.ramp = {key: 1.0 for key in self.ids}
        self.shadow_id = None

    @staticmethod
    def _stack_rows(parameter_dict, ids):
        return torch.stack([parameter_dict[key] for key in ids], dim=0)

    def forward(self, z, return_routing=False):
        if not self.ids:
            raise RuntimeError("dynamic bank has no active experts")
        if z.shape[-1] != self.hidden:
            raise ValueError("last feature dimension does not match router")
        weight = self._stack_rows(self.router_weights, self.ids)
        bias = self._stack_rows(self.router_biases, self.ids)
        logits = z @ weight.t() + bias
        base_probability = torch.softmax(logits, dim=-1)
        ramp = z.new_tensor([float(self.ramp[key]) for key in self.ids])
        routed = base_probability * ramp
        denom = routed.sum(dim=-1, keepdim=True)
        if bool((denom <= 1e-12).any()):
            raise RuntimeError("all active routing ramps are zero")
        routed = routed / denom
        outputs = torch.stack([self.experts[key](z) for key in self.ids], dim=-2)
        correction = (routed.unsqueeze(-1) * outputs).sum(dim=-2)
        if return_routing:
            return correction, routed, {
                "active_ids": list(self.ids),
                "ramps": {key: float(self.ramp[key]) for key in self.ids},
                "dormant_ids": list(self.dormant_experts.keys()),
                "shadow_id": self.shadow_id,
            }
        return correction, routed

    def create_shadow(self, parent_id):
        """Clone one active expert into a registered but non-deployed shadow."""
        parent_id = str(parent_id)
        if self.shadow_id is not None:
            raise RuntimeError("only one shadow candidate is supported at a time")
        if parent_id not in self.ids:
            raise KeyError("parent expert is not active")
        if len(self.ids) + len(self.dormant_experts) >= self.max_experts:
            raise RuntimeError("expert capacity reached")
        key = str(self.next_id)
        self.next_id += 1
        self.shadow_experts[key] = deepcopy(self.experts[parent_id])
        self.shadow_router_weights[key] = nn.Parameter(
            self.router_weights[parent_id].detach().clone())
        self.shadow_router_biases[key] = nn.Parameter(
            self.router_biases[parent_id].detach().clone())
        self.shadow_id = key
        return key

    def shadow_parameters(self):
        if self.shadow_id is None:
            return []
        key = self.shadow_id
        return [*self.shadow_experts[key].parameters(),
                self.shadow_router_weights[key], self.shadow_router_biases[key]]

    def activate_shadow(self):
        """Move a validated shadow into the active bank with exact ramp-0 safety."""
        if self.shadow_id is None:
            raise RuntimeError("no shadow candidate to activate")
        key = self.shadow_id
        self.experts[key] = self.shadow_experts.pop(key)
        self.router_weights[key] = self.shadow_router_weights.pop(key)
        self.router_biases[key] = self.shadow_router_biases.pop(key)
        self.ids.append(key)
        self.ramp[key] = 0.0
        self.shadow_id = None
        return key

    def discard_shadow(self):
        if self.shadow_id is None:
            return None
        key = self.shadow_id
        self.shadow_experts.pop(key)
        self.shadow_router_weights.pop(key)
        self.shadow_router_biases.pop(key)
        self.shadow_id = None
        return key

    def ramp_step(self):
        increment = 1.0 / float(self.ramp_updates)
        for key in self.ids:
            if self.ramp[key] < 1.0:
                self.ramp[key] = min(1.0, self.ramp[key] + increment)

    def retire(self, key):
        """Remove from forward but preserve parameters and expert id."""
        key = str(key)
        if key not in self.ids:
            raise KeyError("expert is not active")
        if len(self.ids) <= 1:
            raise RuntimeError("cannot retire the last active expert")
        expert = self.experts.pop(key)
        weight = self.router_weights.pop(key)
        bias = self.router_biases.pop(key)
        self.dormant_experts[key] = expert
        self.dormant_router_weights[key] = weight
        self.dormant_router_biases[key] = bias
        for parameter in self.dormant_experts[key].parameters():
            parameter.requires_grad_(False)
        self.dormant_router_weights[key].requires_grad_(False)
        self.dormant_router_biases[key].requires_grad_(False)
        self.ids.remove(key)
        self.ramp.pop(key, None)
        return key

    def reactivate(self, key):
        """Restore a dormant expert with the SAME id and ramp it in from zero."""
        key = str(key)
        if key not in self.dormant_experts:
            raise KeyError("expert is not dormant")
        self.experts[key] = self.dormant_experts.pop(key)
        self.router_weights[key] = self.dormant_router_weights.pop(key)
        self.router_biases[key] = self.dormant_router_biases.pop(key)
        for parameter in self.experts[key].parameters():
            parameter.requires_grad_(True)
        self.router_weights[key].requires_grad_(True)
        self.router_biases[key].requires_grad_(True)
        self.ids.append(key)
        self.ramp[key] = 0.0
        return key

    def active_parameter_count(self):
        total = 0
        for key in self.ids:
            total += sum(p.numel() for p in self.experts[key].parameters())
            total += self.router_weights[key].numel()
            total += self.router_biases[key].numel()
        return int(total)

    def topology_manifest(self):
        return {"active_ids": list(self.ids),
                "dormant_ids": list(self.dormant_experts.keys()),
                "shadow_id": self.shadow_id,
                "next_id": int(self.next_id),
                "ramp": {k: float(v) for k, v in self.ramp.items()},
                "max_experts": self.max_experts,
                "ramp_updates": self.ramp_updates}

    def active_state_hash(self):
        digest = hashlib.sha256()
        for key in self.ids:
            for name, tensor in sorted(self.experts[key].state_dict().items()):
                digest.update(("expert:%s:%s" % (key, name)).encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            for kind, tensor in (("router_weight", self.router_weights[key]),
                                 ("router_bias", self.router_biases[key])):
                digest.update(("%s:%s" % (kind, key)).encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
