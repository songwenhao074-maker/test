"""Inactive, parameter/input-matched routing ablation family.

Unlike the historical stacked chain, v2 replaces the ordinary router in the
same expert pool. Scheduling enters only through the graph in v3/v4.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .ftmoe_end_to_end import FTMoEEndToEnd


class MatchedRoutingMoE(nn.Module):
    def __init__(self, existing, adaptive):
        super().__init__()
        assert not hasattr(existing, 'context_projection')
        self.adaptive = bool(adaptive)
        # Retain the freshly initialized ordinary experts/heads byte-for-byte.
        for name, child in existing.named_children():
            self.add_module(name, child)
        self.residual_gain = existing.residual_gain
        with torch.random.fork_rng(devices=[]):
            self.resource_proj = nn.Linear(3, self.router.in_features, bias=False).to(self.router.weight)
        nn.init.zeros_(self.resource_proj.weight)
        self.register_buffer('temperature', torch.tensor(1.).to(self.router.weight))
        self.active_experts = None

    def probabilities(self, routing_state):
        if not self.adaptive:
            self.active_experts = torch.full(routing_state.shape[:-1],self.router.out_features,
                                            device=routing_state.device,dtype=routing_state.dtype)
            return self.router(routing_state).softmax(-1)
        scores = F.normalize(routing_state,dim=-1) @ F.normalize(self.router.weight,dim=-1).T
        thresholds = self.router.bias.tanh()
        temperature = self.temperature.clamp_min(.05)
        soft = torch.sigmoid((scores-thresholds)/temperature)
        hard = (scores>=thresholds).to(scores.dtype)
        cap = min(4,self.router.out_features)
        cap_mask = torch.zeros_like(hard).scatter_(-1,scores.topk(cap,dim=-1).indices,1.)
        hard = hard*cap_mask
        fallback = torch.zeros_like(hard).scatter_(-1,scores.argmax(dim=-1,keepdim=True),1.)
        hard = torch.where(hard.sum(-1,keepdim=True)==0,fallback,hard)
        mask = hard+soft-soft.detach()
        weights = (scores/temperature).softmax(-1)*mask
        self.active_experts = hard.sum(-1)
        return weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)

    def forward(self, tokens, raw_resources):
        routing_state = tokens+self.resource_proj(raw_resources)
        probabilities = self.probabilities(routing_state)
        outputs = torch.stack([expert(routing_state) for expert in self.experts],dim=-2)
        mixed = self.output_norm((probabilities.unsqueeze(-1)*outputs).sum(-2))
        detection = torch.stack([head(raw_resources) for head in self.expert_detection_heads],dim=-2)
        classification = torch.stack([head(raw_resources) for head in self.expert_class_heads],dim=-2)
        return (self.residual_gain*mixed,probabilities,
                self.detection_adapter(mixed)+(probabilities.unsqueeze(-1)*detection).sum(-2),
                self.class_adapter(mixed)+(probabilities.unsqueeze(-1)*classification).sum(-2))


class FTMoEMatchedRouting(FTMoEEndToEnd):
    def __init__(self, variant, cfg=None):
        super().__init__(variant,cfg)
        self.eagate = None
        if self.moe is not None:
            self.moe = MatchedRoutingMoE(self.moe,adaptive=int(variant[1])>=2)
            self.moe.register_forward_hook(self._routing_hook('moe'))

    def set_eagate_temperature(self, temperature):
        if self.moe is not None:
            self.moe.temperature.fill_(max(float(temperature),.05))

    def forward(self, time_windows, schedule_windows=None, graph_time_windows=None):
        result = super().forward(time_windows,schedule_windows,graph_time_windows)
        if self.moe is not None:
            result['active_experts'] = self.moe.active_experts
        return result
