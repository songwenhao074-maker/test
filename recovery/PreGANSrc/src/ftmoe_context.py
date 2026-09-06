"""Optional, parameter-matched local/global context controls for ordinary MoE."""
import torch
from torch import nn


def enable_moe_context(moe, mode):
    """Install on a freshly initialized MoE; no checkpoint or frozen parameters.

    Global context pools host tokens within each sample. It never pools across
    batch examples or reads schedules, labels or future steps. The local
    control has the same trainable projection and operates on each host alone.
    """
    if mode not in ('local','global'):
        raise ValueError(mode)
    if hasattr(moe,'context_projection'):
        raise ValueError('Context was already installed')
    hidden = moe.router.in_features
    reference = next(moe.parameters())
    with torch.random.fork_rng(devices=[]):
        projection = nn.Linear(hidden,hidden,bias=False).to(reference)
    nn.init.zeros_(projection.weight)
    moe.add_module('context_projection',projection)
    moe.context_mode = mode

    def add_context(module, arguments):
        tokens, raw_resources = arguments
        context = tokens.mean(dim=1,keepdim=True) if mode=='global' else tokens
        return tokens + module.context_projection(context), raw_resources

    moe.register_forward_pre_hook(add_context)
    return moe
