"""Protocol-025 registered residual models.

All learned comparators receive the same 73-D residual input: the frozen base
64-D detection-head feature concatenated with nine causal pressure/change
features reconstructed from the current 12-step window and capacity context.
No service/phase/event ID is consumed.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from recovery.PreGANSrc.src.ftmoe_online_r1 import (
    FrozenResidualFTMoE, CORRECTION_SIZE, EXPERT_HIDDEN_SIZE,
)

BASE_Z_DIM = 64
COMMON_DIM = 9
RESIDUAL_INPUT_DIM = BASE_Z_DIM + COMMON_DIM
EXPERT_SEED_BASE = 250100
ROUTER_SEED_BASE = 250200


class RegisteredResidualExpert(nn.Sequential):
    def __init__(self, expert_id: int):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(EXPERT_SEED_BASE + int(expert_id))
            super().__init__(
                nn.LayerNorm(RESIDUAL_INPUT_DIM),
                nn.Linear(RESIDUAL_INPUT_DIM, EXPERT_HIDDEN_SIZE),
                nn.GELU(),
                nn.Linear(EXPERT_HIDDEN_SIZE, CORRECTION_SIZE),
            )
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class RegisteredFixedResidualBank(nn.Module):
    """Deterministic N-expert bank with shared-prefix initialization.

    When ``topk`` is set, only the selected expert MLPs are executed for each
    token. The full router is still evaluated, as is standard sparse-MoE
    accounting, but unselected expert forwards are not performed.
    """
    def __init__(self, expert_count: int, topk: int | None = None):
        super().__init__()
        self.expert_count = int(expert_count)
        if self.expert_count < 1:
            raise ValueError("expert_count must be positive")
        self.topk = None if topk is None else int(topk)
        if self.topk is not None and not 1 <= self.topk <= self.expert_count:
            raise ValueError("topk outside expert count")
        self.router = nn.Linear(RESIDUAL_INPUT_DIM, self.expert_count)
        self.experts = nn.ModuleList([RegisteredResidualExpert(i) for i in range(self.expert_count)])
        bound = 1.0 / math.sqrt(float(RESIDUAL_INPUT_DIM))
        with torch.no_grad():
            for i in range(self.expert_count):
                gen = torch.Generator(device="cpu"); gen.manual_seed(ROUTER_SEED_BASE + i)
                self.router.weight[i].copy_(torch.empty(RESIDUAL_INPUT_DIM).uniform_(-bound, bound, generator=gen))
                self.router.bias[i].copy_(torch.empty(()).uniform_(-bound, bound, generator=gen))

    def forward(self, z):
        if z.shape[-1] != RESIDUAL_INPUT_DIM:
            raise ValueError("Protocol025 residual input must be 73-D")
        logits = self.router(z)
        if self.topk is None or self.topk >= self.expert_count:
            probabilities = torch.softmax(logits, dim=-1)
            outputs = torch.stack([expert(z) for expert in self.experts], dim=-2)
            correction = (probabilities.unsqueeze(-1) * outputs).sum(dim=-2)
            return correction, probabilities

        indices = torch.topk(logits, self.topk, dim=-1).indices
        keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
        routed_logits = logits.masked_fill(~keep, float("-inf"))
        probabilities = torch.softmax(routed_logits, dim=-1)

        flat_z = z.reshape(-1, z.shape[-1])
        flat_prob = probabilities.reshape(-1, self.expert_count)
        flat_idx = indices.reshape(-1, self.topk)
        flat_correction = z.new_zeros((flat_z.shape[0], CORRECTION_SIZE))
        for expert_id, expert in enumerate(self.experts):
            selected = (flat_idx == int(expert_id)).any(dim=-1)
            if not bool(selected.any()):
                continue
            expert_out = expert(flat_z[selected])
            flat_correction[selected] = (
                flat_correction[selected]
                + flat_prob[selected, expert_id].unsqueeze(-1) * expert_out
            )
        correction = flat_correction.reshape(*z.shape[:-1], CORRECTION_SIZE)
        return correction, probabilities


def causal_common_features(time_windows, graph_context, time_scale, graph_scale):
    """Return [B,16,9] using only current/past host state and capacities.

    time_windows: [B,16,12,7], normalized by ReplayV3.time_scale.
    context capacities: [B,12,16,3], normalized by graph_scale[[0,1,4]].
    """
    if time_windows.ndim != 4 or time_windows.shape[1:] != (16, 12, 7):
        raise ValueError("unexpected Protocol025 time-window shape")
    if graph_context is None or "capacities" not in graph_context:
        raise ValueError("Protocol025 common features require causal capacities")
    caps = graph_context["capacities"]
    if caps.ndim != 4 or caps.shape[1:] != (12, 16, 3):
        raise ValueError("unexpected Protocol025 capacity-window shape")
    ts = torch.as_tensor(time_scale, dtype=time_windows.dtype, device=time_windows.device).reshape(1,16,1,7)
    gs = torch.as_tensor(graph_scale, dtype=time_windows.dtype, device=time_windows.device)
    raw = time_windows * ts
    cap_scale = gs[[0,1,4]].reshape(1,1,1,3)
    raw_caps = caps.to(time_windows) * cap_scale
    raw_caps = raw_caps.transpose(1,2)  # [B,16,12,3]
    values = torch.stack([raw[...,0], raw[...,1], raw[...,4]], dim=-1)
    pressure = values / raw_caps.clamp_min(1e-12)
    current = pressure[:,:,-1,:]
    delta = current - pressure[:,:,-2,:]
    slope4 = (current - pressure[:,:,-5,:]) / 4.0
    # Match the registered stored feature: slope4 is zero until four *real*
    # lag intervals exist. Repeated padding must not create an early slope.
    history = graph_context.get("observed_history_length")
    if history is not None:
        history = torch.as_tensor(history, device=time_windows.device).reshape(-1)
        if history.numel() != time_windows.shape[0] or bool(((history < 1) | (history > 12)).any()):
            raise ValueError("invalid observed history length")
        delta = torch.where((history >= 2)[:, None, None], delta, torch.zeros_like(delta))
        slope4 = torch.where((history >= 5)[:, None, None], slope4, torch.zeros_like(slope4))
    out = torch.cat([current, delta, slope4], dim=-1)
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite Protocol025 common observable features")
    return out


class Protocol025ResidualFTMoE(FrozenResidualFTMoE):
    """Frozen base with registered N-expert 73-D residual bank."""
    def __init__(self, checkpoint, seed, expert_count, topk, time_scale, graph_scale):
        super().__init__(checkpoint, "C", seed)
        self.learner = RegisteredFixedResidualBank(expert_count, topk=topk)
        self.requested_method = "C"
        self.correction_input = RESIDUAL_INPUT_DIM
        self.register_buffer("protocol025_time_scale", torch.as_tensor(time_scale, dtype=torch.float32).reshape(16,7).clone())
        self.register_buffer("protocol025_graph_scale", torch.as_tensor(graph_scale, dtype=torch.float32).clone())
        self.set_trainability()
        self.set_deployment("learner", 1.0)
        self.eval()

    def _base_forward(self, time_windows, schedule_windows=None,
                      graph_time_windows=None, graph_context=None, record=False):
        output, z64 = super()._base_forward(
            time_windows, schedule_windows, graph_time_windows,
            graph_context=graph_context, record=record)
        common = causal_common_features(
            time_windows, graph_context,
            self.protocol025_time_scale, self.protocol025_graph_scale)
        z73 = torch.cat([z64, common.to(z64)], dim=-1)
        if z73.shape[-1] != RESIDUAL_INPUT_DIM:
            raise AssertionError("Protocol025 z concatenation failed")
        self._last_z = z73.detach()
        return output, z73

    def set_trainability(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("learner."))
        self._force_base_eval()

    def protocol025_config(self):
        return {
            "residual_input_dim": RESIDUAL_INPUT_DIM,
            "common_dim": COMMON_DIM,
            "expert_count": int(getattr(self.learner, "expert_count", len(getattr(self.learner, "ids", [])))),
            "topk": getattr(self.learner, "topk", None),
            "service_id_input": False,
        }
