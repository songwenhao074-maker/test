"""End-to-end experiment adapter; the archived ablation implementation is untouched."""
import torch
from .ftmoe_ablation import AblationConfig, FTMoEAblation


class FTMoEEndToEnd(FTMoEAblation):
    def __init__(self, variant, cfg=None):
        super().__init__(variant, cfg)
        self.routing_outputs = {}
        for name in ('moe', 'eagate'):
            module = getattr(self, name)
            if module is not None:
                module.register_forward_hook(self._routing_hook(name))

    def _routing_hook(self, name):
        def record(module, arguments, result):
            self.routing_outputs[name] = result[1]
        return record

    def auxiliary_losses(self, output):
        prototype, _ = super().auxiliary_losses(output)
        terms = [((p.mean(dim=(0, 1)) - 1 / self.cfg.experts) ** 2).mean()
                 for p in self.routing_outputs.values()]
        balance = torch.stack(terms).mean() if terms else prototype.new_zeros(())
        return prototype, balance
