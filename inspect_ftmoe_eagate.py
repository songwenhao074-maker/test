"""Development-only perturbation diagnostics; these are not retrained ablations."""
import copy
import json
import psutil
import torch
from train_ftmoe_end_to_end import ART, ROOT, create_model, evaluate, load_data, write_json


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if psutil.virtual_memory().available / 2**30 < 4.5:
        raise RuntimeError('RAM guard')
    run = ART / 'runs/initial_lr0003'
    _, validation, _, _ = load_data(ART / 'data/protocol_001')
    state = torch.load(run / 'v2_seed1/best.pt', map_location='cpu', weights_only=False)
    results = {}
    for setting in ('original', 'no_direct_resource_heads', 'no_eagate_branch', 'zero_eagate_resource_projection'):
        model = create_model('v2', 1)
        model.load_state_dict(state['model'])
        if setting == 'no_direct_resource_heads':
            with torch.no_grad():
                for head in [*model.eagate.expert_detection_heads, *model.eagate.expert_class_heads]:
                    head.weight.zero_(); head.bias.zero_()
        elif setting == 'no_eagate_branch':
            model.eagate = None
        elif setting == 'zero_eagate_resource_projection':
            with torch.no_grad():
                model.eagate.resource_proj.weight.zero_(); model.eagate.resource_proj.bias.zero_()
        results[setting] = evaluate(model, validation)['mean']
    write_json(run / 'eagate_perturbation_seed1.json', {'note': 'Post-training perturbations on development data; not causal retrained architecture ablations',
                                                     'epoch': state['epoch'], 'settings': results})
    print(json.dumps({key: {m: value[m] for m in ('f1', 'hr_at_100', 'ndcg_at_100', 'positive_class_macro_f1')}
                      for key, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
