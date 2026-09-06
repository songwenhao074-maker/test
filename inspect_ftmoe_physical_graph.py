"""Inference perturbations diagnose graph paths; these are not retrained ablations."""
import json
import torch
import psutil
from train_ftmoe_end_to_end import ART, create_model, evaluate, guard, load_data, write_json


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    _, validation, norm, _ = load_data(ART / 'data/protocol_004_physical')
    base = ART / 'diagnosis_reassessment/physical_lr0003_e30'
    reports = {}
    for variant in ['v3', 'v4']:
        source = json.loads((base / f'{variant}_seed1.json').read_text())['best_checkpoint']
        state = torch.load(source, map_location='cpu', weights_only=False)
        model = create_model(variant, 1, norm, **state.get('model_options', {}))
        reports[variant] = {}
        for intervention in ['original', 'zero_shared_graph_gain', 'zero_graph_detection_adapter', 'both_detection_paths_off']:
            guard()
            model.load_state_dict(state['model'])
            with torch.no_grad():
                if intervention in ['zero_shared_graph_gain', 'both_detection_paths_off']:
                    model.graph_gain.zero_()
                if intervention in ['zero_graph_detection_adapter', 'both_detection_paths_off']:
                    model.graph_encoder.detection_adapter.weight.zero_()
                    model.graph_encoder.detection_adapter.bias.zero_()
            reports[variant][intervention] = evaluate(model, validation)
            print(variant, intervention, {k:round(reports[variant][intervention]['mean'][k],4)
                  for k in ['f1','precision','recall','diagnosis_hr_at_100pct']}, flush=True)
    write_json(base / 'graph_path_perturbations_seed1.json', reports)


if __name__ == '__main__':
    main()
