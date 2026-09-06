"""Development-only fit and inference perturbations for the ordinary MoE branch."""
import argparse
import json
import psutil
import torch
from train_ftmoe_end_to_end import ART, ROOT, create_model, evaluate, guard, load_data, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    root = ART/'runs'/args.run
    configuration = json.loads((root/'configuration.json').read_text())['arguments']
    training, validation, norm, _ = load_data(ROOT/configuration['data'])
    result = {}
    for variant in ['v0','v1']:
        for seed in [1,2,6]:
            guard()
            state = torch.load(root/f'{variant}_seed{seed}/last.pt',map_location='cpu',weights_only=False)
            assert state['epoch'] == configuration['epochs']
            model = create_model(variant,seed,norm,**state.get('model_options', {}))
            cases = ['original'] if variant=='v0' else ['original','no_raw_class_heads','no_feature_class_adapter','no_moe_hidden_residual','no_class_correction_heads']
            model_result = {}
            for case in cases:
                model.load_state_dict(state['model'])
                with torch.no_grad():
                    if case in ['no_raw_class_heads','no_class_correction_heads']:
                        for head in model.moe.expert_class_heads:
                            head.weight.zero_(); head.bias.zero_()
                    if case in ['no_feature_class_adapter','no_class_correction_heads']:
                        model.moe.class_adapter.weight.zero_(); model.moe.class_adapter.bias.zero_()
                    if case=='no_moe_hidden_residual':
                        model.moe.residual_gain.zero_()
                model_result[case] = {'development':evaluate(model,validation)}
                if case=='original':
                    model_result[case]['training'] = evaluate(model,{'train':training})
                print(variant,seed,case,{k:round(model_result[case]['development']['mean'][k],5)
                      for k in ['f1','diagnosis_hr_at_100pct','positive_class_macro_f1']},flush=True)
            result[f'{variant}_seed{seed}'] = model_result
    write_json(ART/f'{args.run}_moe_generalization.json',result)


if __name__=='__main__':
    main()
