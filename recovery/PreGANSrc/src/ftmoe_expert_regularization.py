"""Optional dropout on expert hidden activations, with unchanged parameters."""
import torch.nn.functional as F


def enable_expert_dropout(moe, probability):
    if not 0 < probability < 1:
        raise ValueError(probability)
    if hasattr(moe, 'expert_dropout'):
        raise ValueError('Expert dropout already installed')
    moe.expert_dropout = float(probability)

    def regularize(activation, arguments, output):
        return F.dropout(output, p=probability, training=activation.training)

    for expert in moe.experts:
        # The existing expert is Linear -> GELU -> Linear. Preserve all keys.
        assert len(expert) == 3 and expert[1].__class__.__name__ == 'GELU'
        expert[1].register_forward_hook(regularize)
    return moe
