"""Online-only adapter for the archived v4; legacy model files stay unchanged."""
from collections import deque
from copy import deepcopy
import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from .ftmoe_ablation import AblationConfig
from .ftmoe_end_to_end import FTMoEEndToEnd


def tensor_hash(items):
    result = hashlib.sha256()
    for name, tensor in sorted(items):
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class OnlineEAGate(nn.Module):
    """Stable per-expert parameters retain Adam state across topology changes."""
    def __init__(self, source, seed):
        super().__init__()
        self.cfg = source.cfg
        for name in ('resource_proj', 'output_norm', 'detection_adapter', 'class_adapter'):
            setattr(self, name, deepcopy(getattr(source, name)))
        self.residual_gain = nn.Parameter(source.residual_gain.detach().clone())
        self.register_buffer('temperature', source.temperature.detach().clone())
        self.experts = nn.ModuleDict()
        self.key_rows = nn.ParameterDict()
        self.threshold_rows = nn.ParameterDict()
        self.expert_detection_heads = nn.ModuleDict()
        self.expert_class_heads = nn.ModuleDict()
        self.ids = [str(i) for i in range(len(source.experts))]
        for i, key in enumerate(self.ids):
            self.experts[key] = deepcopy(source.experts[i])
            self.key_rows[key] = nn.Parameter(source.keys[i].detach().clone())
            self.threshold_rows[key] = nn.Parameter(source.thresholds[i].detach().clone())
            self.expert_detection_heads[key] = deepcopy(source.expert_detection_heads[i])
            self.expert_class_heads[key] = deepcopy(source.expert_class_heads[i])
        self.next_id = len(self.ids)
        self.generator = torch.Generator().manual_seed(seed + 40000)
        self.record_enabled = False
        self.last_routing = {}
        self.reset_statistics()

    def reset_statistics(self):
        self.activation_counts = {key: 0 for key in self.ids}
        self.routing_samples = 0
        self.unmatched_count = 0
        self.unmatched_vectors = deque(maxlen=256)

    def set_temperature(self, value):
        self.temperature.fill_(max(float(value), .05))

    def forward(self, x, resources):
        state = x + self.resource_proj(resources)
        keys = torch.stack([self.key_rows[key] for key in self.ids])
        thresholds = torch.stack([self.threshold_rows[key] for key in self.ids]).tanh()
        score = F.normalize(state, dim=-1) @ F.normalize(keys, dim=-1).T
        soft_mask = torch.sigmoid((score - thresholds) / self.temperature.clamp_min(.05))
        eligible = score >= thresholds
        unmatched = ~eligible.any(dim=-1)
        hard_mask = eligible.to(score.dtype)
        top_indices = score.topk(min(4, len(self.ids)), dim=-1).indices
        cap_mask = torch.zeros_like(hard_mask).scatter_(-1, top_indices, 1.)
        hard_mask = hard_mask * cap_mask
        fallback = torch.zeros_like(hard_mask).scatter_(-1, score.argmax(-1, keepdim=True), 1.)
        hard_mask = torch.where(unmatched.unsqueeze(-1), fallback, hard_mask)
        mask = hard_mask + soft_mask - soft_mask.detach()
        weight = torch.softmax(score / self.temperature.clamp_min(.05), dim=-1) * mask
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-6)
        outputs = torch.stack([self.experts[key](state) for key in self.ids], dim=-2)
        mixed = self.output_norm((weight.unsqueeze(-1) * outputs).sum(-2))
        detection = torch.stack([self.expert_detection_heads[key](resources) for key in self.ids], dim=-2)
        classes = torch.stack([self.expert_class_heads[key](resources) for key in self.ids], dim=-2)
        self.last_routing = {
            'expert_count': len(self.ids), 'mean_active': float(hard_mask.sum(-1).mean().detach()),
            'unmatched_ratio': float(unmatched.float().mean()),
        }
        if self.record_enabled:
            with torch.no_grad():
                counts = hard_mask.reshape(-1, len(self.ids)).sum(0).tolist()
                for key, count in zip(self.ids, counts):
                    self.activation_counts[key] += int(count)
                self.routing_samples += unmatched.numel()
                self.unmatched_count += int(unmatched.sum())
                for vector in state[unmatched].detach().cpu():
                    self.unmatched_vectors.append(vector.clone())
        return (self.residual_gain * mixed, weight, hard_mask.sum(-1),
                self.detection_adapter(mixed) + (weight.unsqueeze(-1) * detection).sum(-2),
                self.class_adapter(mixed) + (weight.unsqueeze(-1) * classes).sum(-2))

    def _insert(self, key, key_vector=None):
        # Topology randomness is independent from batch order and all other methods.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            expert = nn.Sequential(nn.Linear(self.cfg.hidden, self.cfg.hidden * 2), nn.GELU(),
                                   nn.Linear(self.cfg.hidden * 2, self.cfg.hidden))
            det, cls = nn.Linear(3, 2), nn.Linear(3, self.cfg.classes)
            if key_vector is None or float(key_vector.norm()) == 0:
                key_vector = torch.randn(self.cfg.hidden)
            self.generator.set_state(torch.get_rng_state())
        for head in (expert[-1], det, cls):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.experts[key] = expert
        self.expert_detection_heads[key] = det
        self.expert_class_heads[key] = cls
        self.key_rows[key] = nn.Parameter(F.normalize(key_vector.float(), dim=0).clone())
        self.threshold_rows[key] = nn.Parameter(torch.tensor(0.))

    def adapt(self):
        event = {'before': list(self.ids), 'samples': self.routing_samples,
                 'activation_counts': dict(self.activation_counts), 'unmatched': self.unmatched_count,
                 'removed': [], 'added': []}
        if self.routing_samples:
            removable = [key for key in self.ids if self.activation_counts[key] == 0]
            for key in removable[:max(0, len(self.ids) - 2)]:
                for name in ('experts', 'key_rows', 'threshold_rows', 'expert_detection_heads', 'expert_class_heads'):
                    del getattr(self, name)[key]
                self.ids.remove(key)
                event['removed'].append(key)
            if self.unmatched_count and len(self.ids) < 8:
                vector = torch.stack(list(self.unmatched_vectors)).mean(0)
                key = str(self.next_id)
                self.next_id += 1
                self._insert(key, vector)
                self.ids.append(key)
                event['added'].append(key)
        event['after'] = list(self.ids)
        self.reset_statistics()
        return event

    def topology_state(self):
        return {'ids': list(self.ids), 'next_id': self.next_id,
                'generator': self.generator.get_state(), 'activation_counts': dict(self.activation_counts),
                'routing_samples': self.routing_samples, 'unmatched_count': self.unmatched_count,
                'unmatched_vectors': list(self.unmatched_vectors)}

    def restore_topology(self, state):
        for name in ('experts', 'expert_detection_heads', 'expert_class_heads'):
            setattr(self, name, nn.ModuleDict())
        self.key_rows, self.threshold_rows = nn.ParameterDict(), nn.ParameterDict()
        self.ids = list(state['ids'])
        for key in self.ids:
            self._insert(key)
        self.next_id = state['next_id']
        self.generator.set_state(state['generator'])
        self.activation_counts = dict(state['activation_counts'])
        self.routing_samples, self.unmatched_count = state['routing_samples'], state['unmatched_count']
        self.unmatched_vectors = deque(state['unmatched_vectors'], maxlen=256)


class OnlineFTMoE(FTMoEEndToEnd):
    def __init__(self, checkpoint, method, seed):
        if method not in 'ABCD' or len(method) != 1:
            raise ValueError(method)
        torch.manual_seed(seed)
        super().__init__('v4', AblationConfig(experts=4, moe_residual_initial=0.,
            eagate_residual_initial=.5, graph_residual_initial=0., cmha_residual_initial=0.))
        self.load_state_dict(checkpoint['model'], strict=True)
        self.eagate = OnlineEAGate(self.eagate, seed)
        self.eagate.register_forward_hook(self._routing_hook('eagate'))
        self.method = method
        self.set_trainability()
        self.eval()

    def set_trainability(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(self.method == 'B' or
                (self.method in ('C', 'D') and name.startswith(('moe.', 'eagate.'))))

    def auxiliary_losses(self, output):
        prototype, _ = super().auxiliary_losses(output)
        terms = [((p.mean((0, 1)) - 1. / p.shape[-1]) ** 2).mean()
                 for p in self.routing_outputs.values()]
        return prototype, torch.stack(terms).mean()

    def frozen_hash(self):
        return tensor_hash((name, p) for name, p in self.named_parameters() if not p.requires_grad)

    def state_hash(self):
        return tensor_hash(self.state_dict().items())

    def predict_online(self, x, schedule, graph):
        self.eval()
        self.eagate.record_enabled = True
        try:
            with torch.no_grad():
                return self(x, schedule, graph)
        finally:
            self.eagate.record_enabled = False

    def adapt(self, optimizer):
        event = self.eagate.adapt()
        self.set_trainability()
        live = [p for p in self.parameters() if p.requires_grad]
        live_set = set(live)
        for parameter in list(optimizer.state):
            if parameter not in live_set:
                del optimizer.state[parameter]
        optimizer.param_groups[0]['params'] = live
        # Surviving Parameter identities (including each key row) never changed.
        self.eval()
        return event
