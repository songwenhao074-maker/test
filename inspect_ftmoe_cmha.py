"""Development perturbations to distinguish cross-attention and interaction pathways."""
import json
import psutil
import torch
from train_ftmoe_end_to_end import ART, create_model, evaluate, load_data, write_json


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if psutil.virtual_memory().available / 2**30 < 4.5:
        raise RuntimeError('RAM guard')
    run = ART / 'runs/initial_lr0003'
    _, validation, _, _ = load_data(ART / 'data/protocol_001')
    state = torch.load(run / 'v4_seed1/best.pt', map_location='cpu', weights_only=False)
    result = {}
    for setting in ('original', 'no_cross_attention', 'no_interaction_projection', 'diagonal_attention_only'):
        model = create_model('v4', 1)
        model.load_state_dict(state['model'])
        original = model.cmha.forward
        masses = []
        def forward(query, key, value, **kwargs):
            kwargs['need_weights'] = True
            if setting == 'diagonal_attention_only':
                kwargs['attn_mask'] = ~torch.eye(query.shape[1], dtype=torch.bool)
            cross, weights = original(query, key, value, **kwargs)
            masses.append(float(weights.diagonal(dim1=-2, dim2=-1).mean()))
            return (torch.zeros_like(cross) if setting == 'no_cross_attention' else cross), weights
        model.cmha.forward = forward
        if setting == 'no_interaction_projection':
            with torch.no_grad():
                model.cmha_interaction_proj.weight.zero_(); model.cmha_interaction_proj.bias.zero_()
        values = evaluate(model, validation)['mean']
        values['mean_same_host_attention_mass'] = sum(masses) / len(masses)
        result[setting] = values
    write_json(run / 'cmha_perturbation_seed1.json', {'note': 'Development inference perturbations, not retrained ablations',
                                                    'epoch': state['epoch'], 'settings': result})
    print(json.dumps({key: {m: v[m] for m in ('f1', 'hr_at_100', 'ndcg_at_100', 'mean_same_host_attention_mass')}
                      for key, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
