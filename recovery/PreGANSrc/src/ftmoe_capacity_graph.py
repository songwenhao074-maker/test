"""Optional graph feature candidate; not enabled by the current experiment trainer."""
from __future__ import annotations
import torch
from torch import nn
from .ftmoe_ablation import ScheduleGraphEncoder


class CapacityRatioGraphEncoder(ScheduleGraphEncoder):
    """Expose unit-invariant demand/capacity features using existing graph inputs.

    The original graph already receives the seven aggregated resource features
    and the three host capacities. This adds their log1p ratios before the
    existing graph activation. It introduces 3 * hidden trainable weights and
    no labels, new sensors, threshold rules or extra scheduling information.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ratio_projection = nn.Linear(3, cfg.hidden, bias=False)
        nn.init.zeros_(self.ratio_projection.weight)
        self.input_proj.register_forward_hook(self._add_ratios)

    @staticmethod
    def resource_ratios(graph_input):
        resources = graph_input[..., [0, 1, 4]]
        capacities = graph_input[..., -3:]
        return (resources / capacities.clamp_min(1e-8)).clamp_min(0).log1p()

    def _add_ratios(self, module, arguments, output):
        return output + self.ratio_projection(self.resource_ratios(arguments[0]))

    @classmethod
    def from_existing(cls, original):
        # Preserve both all existing parameter values and the caller's RNG.
        # Zero initialization makes paired v3/v4 additions exactly identical.
        with torch.random.fork_rng(devices=[]):
            result = cls(original.cfg)
        missing, unexpected = result.load_state_dict(original.state_dict(), strict=False)
        assert missing == ['ratio_projection.weight'] and not unexpected
        result.train(original.training)
        return result
