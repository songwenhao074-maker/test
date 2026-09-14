"""Protocol-024 dynamic residual bank for the causal lifecycle pilot.

The bank starts exactly from Protocol-023 fixed C.  It deliberately contains no
future/audit-ID policy.  It only provides safe topology operations, stable
routing, deterministic identity, and role-aware parameter access for the
Protocol-024 lifecycle controller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import torch
from torch import nn


class DynamicResidualBank(nn.Module):
    """Expandable/retirable residual bank initialized exactly from fixed C."""

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
        self.topology_version = 0
        self.behavior_version = 0
        self.set_role_trainability()

    @staticmethod
    def _stack_rows(parameter_dict, ids):
        return torch.stack([parameter_dict[key] for key in ids], dim=0)

    @staticmethod
    def _stable_mixture(logits, ramps):
        """Softmax(logit + log(ramp)), masking exact-zero ramps pre-softmax."""
        if logits.shape[-1] != ramps.numel():
            raise ValueError("router/ramp width mismatch")
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite router logits")
        positive = ramps > 0
        if not bool(positive.any()):
            raise RuntimeError("at least one active expert must have positive ramp")
        adjusted = torch.full_like(logits, float("-inf"))
        adjusted[..., positive] = (
            logits[..., positive] + torch.log(ramps[positive]).to(logits))
        routed = torch.softmax(adjusted, dim=-1)
        if not torch.isfinite(routed).all():
            raise RuntimeError("non-finite routed probabilities")
        return routed

    def _mixture(self, z, ids, experts, weights, biases, ramps):
        weight = self._stack_rows(weights, ids)
        bias = self._stack_rows(biases, ids)
        logits = z @ weight.t() + bias
        ramp_tensor = z.new_tensor([float(ramps[key]) for key in ids])
        routed = self._stable_mixture(logits, ramp_tensor)
        outputs = torch.stack([experts[key](z) for key in ids], dim=-2)
        correction = (routed.unsqueeze(-1) * outputs).sum(dim=-2)
        return correction, routed

    def forward(self, z, return_routing=False):
        if not self.ids:
            raise RuntimeError("dynamic bank has no active experts")
        if z.shape[-1] != self.hidden:
            raise ValueError("last feature dimension does not match router")
        correction, routed = self._mixture(
            z, self.ids, self.experts, self.router_weights,
            self.router_biases, self.ramp)
        if return_routing:
            return correction, routed, {
                "active_ids": list(self.ids),
                "ramps": {key: float(self.ramp[key]) for key in self.ids},
                "dormant_ids": list(self.dormant_experts.keys()),
                "shadow_id": self.shadow_id,
                "topology_version": int(self.topology_version),
                "behavior_version": int(self.behavior_version),
            }
        return correction, routed

    def preview_with_shadow(self, z, shadow_ramp=1.0):
        """Counterfactual full-bank correction without mutating live routing.

        Used for causal candidate training/qualification.  The shadow remains
        absent from :meth:`forward`; this method is an explicit preview path.
        """
        if self.shadow_id is None:
            raise RuntimeError("no shadow candidate to preview")
        value = float(shadow_ramp)
        if not (0.0 < value <= 1.0):
            raise ValueError("shadow_ramp must be in (0,1]")
        key = self.shadow_id
        ids = list(self.ids) + [key]
        experts = {k: self.experts[k] for k in self.ids}
        weights = {k: self.router_weights[k] for k in self.ids}
        biases = {k: self.router_biases[k] for k in self.ids}
        experts[key] = self.shadow_experts[key]
        weights[key] = self.shadow_router_weights[key]
        biases[key] = self.shadow_router_biases[key]
        ramps = {k: float(self.ramp[k]) for k in self.ids}
        ramps[key] = value
        return self._mixture(z, ids, experts, weights, biases, ramps)

    def resident_count(self):
        return (len(self.ids) + len(self.dormant_experts)
                + len(self.shadow_experts))

    def create_shadow(self, parent_id):
        parent_id = str(parent_id)
        if self.shadow_id is not None:
            raise RuntimeError("only one shadow candidate is supported at a time")
        if parent_id not in self.ids:
            raise KeyError("parent expert is not active")
        if self.resident_count() >= self.max_experts:
            raise RuntimeError("expert capacity reached")
        key = str(self.next_id)
        self.next_id += 1
        self.shadow_experts[key] = deepcopy(self.experts[parent_id])
        self.shadow_router_weights[key] = nn.Parameter(
            self.router_weights[parent_id].detach().clone())
        self.shadow_router_biases[key] = nn.Parameter(
            self.router_biases[parent_id].detach().clone())
        self.shadow_id = key
        self.topology_version += 1
        self.behavior_version += 1
        self.set_role_trainability()
        return key

    def shadow_parameters(self):
        if self.shadow_id is None:
            return []
        key = self.shadow_id
        return [*self.shadow_experts[key].parameters(),
                self.shadow_router_weights[key], self.shadow_router_biases[key]]

    def activate_shadow(self):
        """Move validated shadow into live bank at exact ramp zero."""
        if self.shadow_id is None:
            raise RuntimeError("no shadow candidate to activate")
        key = self.shadow_id
        self.experts[key] = self.shadow_experts.pop(key)
        self.router_weights[key] = self.shadow_router_weights.pop(key)
        self.router_biases[key] = self.shadow_router_biases.pop(key)
        self.ids.append(key)
        self.ramp[key] = 0.0
        self.shadow_id = None
        self.topology_version += 1
        self.behavior_version += 1
        self.set_role_trainability()
        return key

    def discard_shadow(self):
        if self.shadow_id is None:
            return None
        key = self.shadow_id
        self.shadow_experts.pop(key)
        self.shadow_router_weights.pop(key)
        self.shadow_router_biases.pop(key)
        self.shadow_id = None
        self.topology_version += 1
        self.behavior_version += 1
        return key

    def set_ramp(self, key, value):
        key = str(key)
        if key not in self.ids:
            raise KeyError("expert is not active")
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError("ramp must be in [0,1]")
        if len(self.ids) == 1 and value <= 0.0:
            raise RuntimeError("cannot set the last active expert ramp to zero")
        if self.ramp[key] != value:
            self.ramp[key] = value
            self.behavior_version += 1
        return value

    def ramp_step(self):
        increment = 1.0 / float(self.ramp_updates)
        changed = False
        for key in self.ids:
            if self.ramp[key] < 1.0:
                self.ramp[key] = min(1.0, self.ramp[key] + increment)
                changed = True
        if changed:
            self.behavior_version += 1
        return changed

    def ramp_down_step(self, key):
        key = str(key)
        if key not in self.ids:
            raise KeyError("expert is not active")
        if len(self.ids) <= 1:
            raise RuntimeError("cannot ramp down the last active expert")
        decrement = 1.0 / float(self.ramp_updates)
        before = self.ramp[key]
        self.ramp[key] = max(0.0, before - decrement)
        if self.ramp[key] != before:
            self.behavior_version += 1
        return self.ramp[key]

    def retire(self, key):
        """Move an active expert to dormant storage; caller controls ramp-down."""
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
        self.ids.remove(key)
        self.ramp.pop(key, None)
        self.topology_version += 1
        self.behavior_version += 1
        self.set_role_trainability()
        return key

    def reactivate(self, key):
        key = str(key)
        if key not in self.dormant_experts:
            raise KeyError("expert is not dormant")
        self.experts[key] = self.dormant_experts.pop(key)
        self.router_weights[key] = self.dormant_router_weights.pop(key)
        self.router_biases[key] = self.dormant_router_biases.pop(key)
        self.ids.append(key)
        self.ramp[key] = 0.0
        self.topology_version += 1
        self.behavior_version += 1
        self.set_role_trainability()
        return key

    def purge(self, key):
        """Hard-delete a dormant expert and free one resident-capacity slot."""
        key = str(key)
        if key not in self.dormant_experts:
            raise KeyError("only a dormant expert can be purged")
        self.dormant_experts.pop(key)
        self.dormant_router_weights.pop(key)
        self.dormant_router_biases.pop(key)
        self.topology_version += 1
        self.behavior_version += 1
        return key

    def set_role_trainability(self):
        """Active+shadow trainable, dormant frozen; never use broad learner.*."""
        for key in self.experts.keys():
            for p in self.experts[key].parameters():
                p.requires_grad_(True)
            self.router_weights[key].requires_grad_(True)
            self.router_biases[key].requires_grad_(True)
        for key in self.dormant_experts.keys():
            for p in self.dormant_experts[key].parameters():
                p.requires_grad_(False)
                p.grad = None
            self.dormant_router_weights[key].requires_grad_(False)
            self.dormant_router_biases[key].requires_grad_(False)
            self.dormant_router_weights[key].grad = None
            self.dormant_router_biases[key].grad = None
        for key in self.shadow_experts.keys():
            for p in self.shadow_experts[key].parameters():
                p.requires_grad_(True)
            self.shadow_router_weights[key].requires_grad_(True)
            self.shadow_router_biases[key].requires_grad_(True)

    def active_named_parameters(self):
        """Stable (expert_id,tensor_name) keys for optimizer-state migration."""
        out = []
        for key in self.ids:
            for name, p in self.experts[key].named_parameters():
                out.append(((key, "expert." + name), p))
            out.append(((key, "router_weight"), self.router_weights[key]))
            out.append(((key, "router_bias"), self.router_biases[key]))
        return out

    def shadow_named_parameters(self):
        if self.shadow_id is None:
            return []
        key = self.shadow_id
        out = [((key, "expert." + name), p)
               for name, p in self.shadow_experts[key].named_parameters()]
        out.append(((key, "router_weight"), self.shadow_router_weights[key]))
        out.append(((key, "router_bias"), self.shadow_router_biases[key]))
        return out

    def active_parameter_count(self):
        return int(sum(p.numel() for _, p in self.active_named_parameters()))

    def topology_manifest(self):
        return {"active_ids": list(self.ids),
                "dormant_ids": list(self.dormant_experts.keys()),
                "shadow_id": self.shadow_id,
                "next_id": int(self.next_id),
                "ramp": {k: float(v) for k, v in self.ramp.items()},
                "max_experts": self.max_experts,
                "ramp_updates": self.ramp_updates,
                "topology_version": int(self.topology_version),
                "behavior_version": int(self.behavior_version)}

    def behavior_state_hash(self):
        """Hash every live-forward determinant: tensor state, order and ramps."""
        digest = hashlib.sha256()
        meta = {"active_ids": list(self.ids),
                "ramp": {k: float(self.ramp[k]) for k in self.ids},
                "topology_version": int(self.topology_version),
                "behavior_version": int(self.behavior_version)}
        digest.update(json.dumps(meta, sort_keys=True,
                                 separators=(",", ":")).encode("utf8"))
        for key in self.ids:
            for name, tensor in sorted(self.experts[key].state_dict().items()):
                digest.update(("expert:%s:%s" % (key, name)).encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            for kind, tensor in (("router_weight", self.router_weights[key]),
                                 ("router_bias", self.router_biases[key])):
                digest.update(("%s:%s" % (kind, key)).encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def active_state_hash(self):
        """Backward-compatible name; now correctly includes ramp/topology."""
        return self.behavior_state_hash()
