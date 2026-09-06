"""Small, batch-first models for the FT-MoE ablation protocol.

This module is deliberately independent from the legacy ``models_v2.py``
chain.  The two models implemented here are only the Stage-A variants v0 and
v1; later ablations must be added as separate, reviewed changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AblationConfig:
    hosts: int = 16
    window: int = 12
    features: int = 7
    hidden: int = 64
    classes: int = 3
    prototype_dim: int = 8
    experts: int = 8
    dropout: float = 0.1
    moe_residual_initial: float = 0.25
    eagate_residual_initial: float = 0.15
    graph_residual_initial: float = 0.50
    cmha_residual_initial: float = 0.05


class TimeEncoder(nn.Module):
    """Shared per-host [B,H,12,7] -> [B,H,64] time encoder."""

    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.features, cfg.hidden)
        self.position = nn.Parameter(torch.zeros(1, cfg.window, cfg.hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden, nhead=4, dim_feedforward=cfg.hidden * 2,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(cfg.hidden)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, time_windows: torch.Tensor) -> torch.Tensor:
        if time_windows.ndim != 4:
            raise ValueError("time_windows must have shape [B,H,W,F]")
        b, h, w, f = time_windows.shape
        if (h, w, f) != (self.cfg.hosts, self.cfg.window, self.cfg.features):
            raise ValueError("unexpected FT-MoE screening input shape")
        x = time_windows.reshape(b * h, w, f)
        x = self.input_proj(x) + self.position
        x = self.transformer(x)
        x = (x[:, -1] + x.mean(dim=1)) * 0.5
        return self.norm(x).reshape(b, h, self.cfg.hidden)


class SoftmaxMoE(nn.Module):
    """The ordinary all-expert learnable softmax router specified for v1."""

    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.router = nn.Linear(cfg.hidden, cfg.experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden, cfg.hidden * 2), nn.GELU(),
                nn.Linear(cfg.hidden * 2, cfg.hidden),
            ) for _ in range(cfg.experts)
        ])
        self.output_norm = nn.LayerNorm(cfg.hidden)
        self.residual_gain = nn.Parameter(torch.tensor(cfg.moe_residual_initial))
        self.detection_adapter = nn.Linear(cfg.hidden, 2)
        self.class_adapter = nn.Linear(cfg.hidden, cfg.classes)
        self.expert_detection_heads = nn.ModuleList([
            nn.Linear(3, 2) for _ in range(cfg.experts)
        ])
        self.expert_class_heads = nn.ModuleList([
            nn.Linear(3, cfg.classes) for _ in range(cfg.experts)
        ])
        for head in [self.detection_adapter, self.class_adapter,
                     *self.expert_detection_heads, *self.expert_class_heads]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor, raw_resources: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # All configured expert outputs are materialized and weighted.  There is
        # no threshold, top-k, adaptive expert count, or schedule input.
        probabilities = torch.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = (probabilities.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        mixed = self.output_norm(mixed)
        expert_detection = torch.stack([
            head(raw_resources) for head in self.expert_detection_heads
        ], dim=-2)
        expert_class = torch.stack([
            head(raw_resources) for head in self.expert_class_heads
        ], dim=-2)
        detection = (probabilities.unsqueeze(-1) * expert_detection).sum(dim=-2)
        classification = (probabilities.unsqueeze(-1) * expert_class).sum(dim=-2)
        return (self.residual_gain * mixed, probabilities,
                self.detection_adapter(mixed) + detection,
                self.class_adapter(mixed) + classification)


class EAGateMoE(nn.Module):
    """Trainable Top-any expert correction with straight-through masks."""

    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden, cfg.hidden * 2), nn.GELU(),
                nn.Linear(cfg.hidden * 2, cfg.hidden),
            ) for _ in range(cfg.experts)
        ])
        self.keys = nn.Parameter(torch.empty(cfg.experts, cfg.hidden))
        self.thresholds = nn.Parameter(torch.full((cfg.experts,), -0.10))
        self.resource_proj = nn.Linear(3, cfg.hidden)
        self.output_norm = nn.LayerNorm(cfg.hidden)
        self.residual_gain = nn.Parameter(
            torch.tensor(cfg.eagate_residual_initial)
        )
        self.detection_adapter = nn.Linear(cfg.hidden, 2)
        self.class_adapter = nn.Linear(cfg.hidden, cfg.classes)
        self.expert_detection_heads = nn.ModuleList([
            nn.Linear(3, 2) for _ in range(cfg.experts)
        ])
        self.expert_class_heads = nn.ModuleList([
            nn.Linear(3, cfg.classes) for _ in range(cfg.experts)
        ])
        self.register_buffer("temperature", torch.tensor(1.0))
        nn.init.normal_(self.keys, std=cfg.hidden ** -0.5)
        # Zero output adapters make a warm-started v2 exactly equal to v1,
        # while their gradients remain non-zero on the first update.
        nn.init.zeros_(self.detection_adapter.weight)
        nn.init.zeros_(self.detection_adapter.bias)
        nn.init.zeros_(self.class_adapter.weight)
        nn.init.zeros_(self.class_adapter.bias)
        for head in [*self.expert_detection_heads, *self.expert_class_heads]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def set_temperature(self, temperature: float) -> None:
        self.temperature.fill_(max(float(temperature), 0.05))

    def forward(self, x: torch.Tensor, raw_resources: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor]:
        routing_state = x + self.resource_proj(raw_resources)
        score = (F.normalize(routing_state, dim=-1) @
                 F.normalize(self.keys, dim=-1).T)
        threshold = torch.tanh(self.thresholds)
        soft_mask = torch.sigmoid(
            (score - threshold) / self.temperature.clamp_min(0.05)
        )
        hard_mask = (score >= threshold).to(score.dtype)

        # Keep at most four experts and guarantee at least one expert.
        cap = min(4, self.cfg.experts)
        top_indices = score.topk(cap, dim=-1).indices
        cap_mask = torch.zeros_like(hard_mask).scatter_(-1, top_indices, 1.0)
        hard_mask = hard_mask * cap_mask
        empty = hard_mask.sum(dim=-1, keepdim=True) == 0
        fallback = torch.zeros_like(hard_mask).scatter_(
            -1, score.argmax(dim=-1, keepdim=True), 1.0
        )
        hard_mask = torch.where(empty, fallback, hard_mask)

        # Hard sparse forward, soft threshold gradient backward.
        mask = hard_mask + soft_mask - soft_mask.detach()
        base_weight = torch.softmax(
            score / self.temperature.clamp_min(0.05), dim=-1
        )
        weight = base_weight * mask
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_outputs = torch.stack(
            [expert(routing_state) for expert in self.experts], dim=-2
        )
        mixed = (weight.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        mixed = self.output_norm(mixed)
        expert_detection = torch.stack([
            head(raw_resources) for head in self.expert_detection_heads
        ], dim=-2)
        expert_class = torch.stack([
            head(raw_resources) for head in self.expert_class_heads
        ], dim=-2)
        detection = (weight.unsqueeze(-1) * expert_detection).sum(dim=-2)
        classification = (weight.unsqueeze(-1) * expert_class).sum(dim=-2)
        return (self.residual_gain * mixed, weight,
                hard_mask.sum(dim=-1),
                self.detection_adapter(mixed) + detection,
                self.class_adapter(mixed) + classification)


class ScheduleGraphEncoder(nn.Module):
    """Encode scheduled workload aggregates and directed migration edges."""

    def __init__(self, cfg: AblationConfig):
        super().__init__()
        self.cfg = cfg
        # Seven scheduled resource values, occupancy, inbound and outbound
        # migration counts form one host token at each interval.
        self.input_proj = nn.Linear(cfg.features + 6, cfg.hidden)
        capacity = torch.ones(cfg.hosts, 3)
        capacity[:8, 1] = 4295.0 / 8192.0
        self.register_buffer("host_capacity", capacity)
        self.query = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.key = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.value = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.output = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.detection_adapter = nn.Linear(cfg.hidden, 2)
        self.class_adapter = nn.Linear(cfg.hidden, cfg.classes)
        nn.init.zeros_(self.detection_adapter.weight)
        nn.init.zeros_(self.detection_adapter.bias)
        nn.init.zeros_(self.class_adapter.weight)
        nn.init.zeros_(self.class_adapter.bias)

    def graph_migration_and_occupancy(
            self, schedule_windows: torch.Tensor,
            graph_context: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (occupancy [B,W,H,1], migrations [B,W-1,H,H] | [B,W,H,H]).

        Occupancy is the number of *valid* containers scheduled onto each host
        at each window position.  Migrations count container transitions
        h -> k; with graph_context v2 only transitions of the same creation
        identity count (consecutive proposed rows).

        Graph semantics v3 (Protocol 020): when graph_context additionally
        carries ``before_placement`` [B,W,C] int64, each window position is
        decoded independently from information known *before* the placement
        decision of that step (plan §6): src = before_placement[t,c] (the
        actual current host), dst = argmax(proposed_schedule[t,c]).  The
        result has shape [B,W,H,H].  A rejected previous proposal therefore
        never pollutes the current source host (019 problem §1.5).
        """
        eye = torch.eye(
            self.cfg.hosts, dtype=schedule_windows.dtype,
            device=schedule_windows.device,
        ).view(1, 1, self.cfg.hosts, self.cfg.hosts)
        if graph_context is not None and \
                graph_context.get("before_placement") is not None:
            identity = graph_context["creation_ids"]
            before = graph_context["before_placement"]
            if identity.ndim != 3 or before.ndim != 3 or \
                    identity.shape != before.shape or \
                    identity.shape[:2] != schedule_windows.shape[:2]:
                raise ValueError(
                    "graph_context before_placement/creation_ids must have "
                    "shape [B,W,C] matching schedule_windows"
                )
            valid = (identity >= 0).to(schedule_windows.dtype)
            occupancy = torch.einsum(
                "btch,btc->bth", schedule_windows, valid
            ).unsqueeze(-1)
            host_ids = torch.arange(
                self.cfg.hosts, device=schedule_windows.device
            )
            deployed = ((before >= 0) & (identity >= 0)).to(
                schedule_windows.dtype
            )
            # [B,W,C,H]: one-hot actual source host before the step decision.
            source = (before.unsqueeze(-1) == host_ids.view(
                1, 1, 1, self.cfg.hosts
            )).to(schedule_windows.dtype) * deployed.unsqueeze(-1)
            # dst one-hot rows are the proposed schedule itself ([B,W,C,H]).
            migrations = torch.einsum("btch,btck->bthk", source, schedule_windows)
        elif graph_context is not None:
            identity = graph_context["creation_ids"]
            if identity.ndim != 3 or identity.shape[:2] != (
                    schedule_windows.shape[0], schedule_windows.shape[1]):
                raise ValueError(
                    "graph_context creation_ids must have shape [B,W,C] matching "
                    "schedule_windows"
                )
            valid = (identity >= 0).to(schedule_windows.dtype)
            occupancy = torch.einsum(
                "btch,btc->bth", schedule_windows, valid
            ).unsqueeze(-1)
            previous_identity = identity[:, :-1]
            current_identity = identity[:, 1:]
            same_identity = (
                (previous_identity >= 0) & (previous_identity == current_identity)
            ).to(schedule_windows.dtype)
            previous = schedule_windows[:, :-1]
            current = schedule_windows[:, 1:]
            previous_weighted = previous * same_identity.unsqueeze(-1)
            migrations = torch.einsum(
                "btch,btck->bthk", previous_weighted, current
            )
        else:
            # Legacy v1 semantics: every slot row of the schedule contributes,
            # so occupancy counts scheduled rows and slot replacements between
            # consecutive intervals read as migrations.
            previous = schedule_windows[:, :-1]
            current = schedule_windows[:, 1:]
            occupancy = schedule_windows.sum(dim=2).unsqueeze(-1)
            migrations = torch.einsum("btch,btck->bthk", previous, current)
        migrations = migrations * (1.0 - eye)
        return occupancy, migrations

    def forward(self, time_windows: torch.Tensor,
                schedule_windows: torch.Tensor,
                graph_context: dict | None = None) -> torch.Tensor:
        if schedule_windows.ndim != 4:
            raise ValueError("schedule_windows must have shape [B,W,C,H]")
        # The 16 resource groups are workload slots in the replay.  Aggregate
        # each slot's resource vector onto the host selected by the schedule.
        slot_features = time_windows.permute(0, 2, 1, 3)
        scheduled = torch.einsum(
            "btch,btcf->bthf", schedule_windows, slot_features
        )
        occupancy, migrations = self.graph_migration_and_occupancy(
            schedule_windows, graph_context
        )
        incoming = migrations.sum(dim=(-3, -2)).unsqueeze(1).unsqueeze(-1)
        outgoing = migrations.sum(dim=(-3, -1)).unsqueeze(1).unsqueeze(-1)
        incoming = incoming.expand(-1, self.cfg.window, -1, -1)
        outgoing = outgoing.expand(-1, self.cfg.window, -1, -1)

        if graph_context is not None and \
                graph_context.get("capacities") is not None:
            capacity = graph_context["capacities"]
            if capacity.ndim != 4 or capacity.shape[:3] != scheduled.shape[:3] \
                    or capacity.shape[3] != 3:
                raise ValueError(
                    "graph_context capacities must have shape "
                    "[B,W,H,3] matching schedule_windows"
                )
            capacity = capacity.to(scheduled.dtype)
        else:
            # Legacy: a static per-stream capacity buffer ([1,1,H,3]).
            capacity = self.host_capacity.view(
                1, 1, self.cfg.hosts, 3
            ).expand(scheduled.shape[0], self.cfg.window, -1, -1)
        graph_input = torch.cat(
            [scheduled, occupancy, incoming, outgoing, capacity], dim=-1
        )
        tokens = F.gelu(self.input_proj(graph_input))
        host_tokens = 0.70 * tokens[:, -1] + 0.30 * tokens.mean(dim=1)

        adjacency = migrations.sum(dim=1) > 0
        adjacency = adjacency | torch.eye(
            self.cfg.hosts, dtype=torch.bool, device=adjacency.device,
        ).unsqueeze(0)
        scores = (self.query(host_tokens) @ self.key(host_tokens).transpose(-2, -1))
        scores = scores / (self.cfg.hidden ** 0.5)
        scores = scores.masked_fill(~adjacency, -1e4)
        attention = torch.softmax(scores, dim=-1)
        graph = attention @ self.value(host_tokens)
        return self.norm(host_tokens + self.output(graph))


class FTMoEAblation(nn.Module):
    """Nested v0--v4 implementation for the execution-plan ablation chain."""

    def __init__(self, variant: str, cfg: AblationConfig | None = None):
        super().__init__()
        if variant not in {"v0", "v1", "v2", "v3", "v4"}:
            raise ValueError("variant must be one of v0, v1, v2, v3, v4")
        self.variant = variant
        self.cfg = cfg or AblationConfig()
        self.encoder = TimeEncoder(self.cfg)
        self.detection_head = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden), nn.Linear(self.cfg.hidden, 2)
        )
        self.class_embedding = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden), nn.Linear(self.cfg.hidden, self.cfg.prototype_dim)
        )
        self.prototypes = nn.Parameter(torch.empty(self.cfg.classes, self.cfg.prototype_dim))
        nn.init.orthogonal_(self.prototypes)
        # Construct the optional branch last.  With a shared RNG seed, every
        # parameter above (the common v0/v1 path) therefore has identical
        # initialization in the two variants.
        level = int(variant[1])
        self.moe = SoftmaxMoE(self.cfg) if level >= 1 else None
        self.eagate = EAGateMoE(self.cfg) if level >= 2 else None
        self.graph_encoder = ScheduleGraphEncoder(self.cfg) if level >= 3 else None
        self.graph_gain = nn.Parameter(
            torch.tensor(self.cfg.graph_residual_initial)
        ) if level >= 3 else None
        if level >= 4:
            self.cmha = nn.MultiheadAttention(
                self.cfg.hidden, num_heads=4, dropout=self.cfg.dropout,
                batch_first=True,
            )
            self.cmha_norm = nn.LayerNorm(self.cfg.hidden)
            self.cmha_interaction_proj = nn.Linear(
                self.cfg.hidden, self.cfg.hidden
            )
            self.cmha_gain = nn.Parameter(
                torch.tensor(self.cfg.cmha_residual_initial)
            )
            self.cmha_detection_adapter = nn.Linear(self.cfg.hidden, 2)
            self.cmha_class_adapter = nn.Linear(
                self.cfg.hidden, self.cfg.classes
            )
            nn.init.zeros_(self.cmha_detection_adapter.weight)
            nn.init.zeros_(self.cmha_detection_adapter.bias)
            nn.init.zeros_(self.cmha_class_adapter.weight)
            nn.init.zeros_(self.cmha_class_adapter.bias)
        else:
            self.cmha = None
            self.cmha_norm = None
            self.cmha_interaction_proj = None
            self.cmha_gain = None
            self.cmha_detection_adapter = None
            self.cmha_class_adapter = None

    def forward(self, time_windows: torch.Tensor,
                schedule_windows: torch.Tensor | None = None,
                graph_time_windows: torch.Tensor | None = None,
                graph_context: dict | None = None) -> dict[str, torch.Tensor]:
        z = self.encoder(time_windows)
        active_experts = None
        softmax_detection = None
        softmax_class = None
        eagate_detection = None
        eagate_class = None
        if self.moe is None:
            router_probabilities = None
        else:
            raw_resources = time_windows[:, :, -1, [0, 1, 4]]
            (moe_output, router_probabilities,
             softmax_detection, softmax_class) = self.moe(z, raw_resources)
            z = z + moe_output
        if self.eagate is not None:
            if schedule_windows is not None and graph_time_windows is not None:
                # EAGate routes each host from its currently scheduled workload.
                # This is causal: both arrays are sampled before simulationStep.
                container_resources = graph_time_windows[:, :, -1, :]
                current_schedule = schedule_windows[:, -1, :, :]
                host_resources = torch.einsum(
                    "bch,bcf->bhf", current_schedule, container_resources,
                )
                raw_resources = host_resources[:, :, [0, 1, 4]]
            else:
                raw_resources = time_windows[:, :, -1, [0, 1, 4]]
            (eagate_output, router_probabilities, active_experts,
             eagate_detection, eagate_class) = self.eagate(z, raw_resources)
            z = z + eagate_output
        graph_features = None
        graph_detection = None
        graph_class = None
        pre_graph = z
        if self.graph_encoder is not None:
            if schedule_windows is None:
                raise ValueError(f"{self.variant} requires schedule windows")
            graph_input = (time_windows if graph_time_windows is None
                           else graph_time_windows)
            graph_features = self.graph_encoder(
                graph_input, schedule_windows, graph_context=graph_context
            )
            z = z + self.graph_gain * graph_features
            graph_detection = self.graph_encoder.detection_adapter(graph_features)
            graph_class = self.graph_encoder.class_adapter(graph_features)
        cmha_features = None
        if self.cmha is not None:
            cross, _ = self.cmha(pre_graph, graph_features, graph_features,
                                 need_weights=False)
            interaction = self.cmha_interaction_proj(
                pre_graph * graph_features
            )
            cmha_features = self.cmha_norm(cross + interaction)
        embeddings = F.normalize(self.class_embedding(z), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        class_logits = -((embeddings.unsqueeze(-2) - prototypes) ** 2).sum(dim=-1) / 0.2
        detection_logits = self.detection_head(z)
        if softmax_detection is not None:
            detection_logits = detection_logits + softmax_detection
            class_logits = class_logits + softmax_class
        if eagate_detection is not None:
            detection_logits = detection_logits + eagate_detection
            class_logits = class_logits + eagate_class
        if graph_detection is not None:
            detection_logits = detection_logits + graph_detection
            class_logits = class_logits + graph_class
        base_detection_logits = detection_logits
        base_class_logits = class_logits
        if cmha_features is not None:
            scaled_cmha = (1.0 + self.cmha_gain) * cmha_features
            detection_logits = detection_logits + self.cmha_detection_adapter(scaled_cmha)
            class_logits = class_logits + self.cmha_class_adapter(scaled_cmha)
        return {
            "detection_logits": detection_logits,
            "class_logits": class_logits,
            "base_detection_logits": base_detection_logits,
            "base_class_logits": base_class_logits,
            "router_probabilities": router_probabilities,
            "active_experts": active_experts,
            "graph_features": graph_features,
        }

    def set_eagate_temperature(self, temperature: float) -> None:
        if self.eagate is not None:
            self.eagate.set_temperature(temperature)

    def auxiliary_losses(self, output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = prototypes @ prototypes.T
        off_diagonal = similarity[~torch.eye(self.cfg.classes, dtype=torch.bool,
                                             device=similarity.device)]
        prototype_separation = F.relu(off_diagonal + 0.20).mean()
        probabilities = output["router_probabilities"]
        if probabilities is None:
            balance = prototype_separation.new_zeros(())
        else:
            mean_probability = probabilities.mean(dim=(0, 1))
            balance = ((mean_probability - 1.0 / self.cfg.experts) ** 2).mean()
        return prototype_separation, balance
