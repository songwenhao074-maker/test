"""Inspect train/development gaps of completed checkpoints; no optimization."""
import argparse
import json
from pathlib import Path
import torch
import psutil
from train_ftmoe_end_to_end import ART, ROOT, create_model, evaluate, load_data, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--variants', nargs='+', default=['v0', 'v1'])
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if psutil.virtual_memory().available / 2**30 < 4.5:
        raise RuntimeError('RAM guard')
    run = ART / 'runs' / args.run
    config = json.loads((run / 'configuration.json').read_text())
    training, validation, _, _ = load_data(ROOT / config['arguments']['data'])
    result = {}
    for variant in args.variants:
        result[variant] = {}
        for name in ('best', 'last'):
            state = torch.load(run / f'{variant}_seed{args.seed}' / f'{name}.pt', map_location='cpu', weights_only=False)
            model = create_model(variant, args.seed)
            model.load_state_dict(state['model'])
            train = evaluate(model, {'train': training})['mean']
            dev = evaluate(model, validation)['mean']
            result[variant][name] = {'epoch': state['epoch'], 'train': train, 'development': dev}
    write_json(run / f'fit_diagnostics_seed{args.seed}.json', result)
    print(json.dumps({v: {n: {'epoch': r['epoch'], 'train_f1': r['train']['f1'], 'dev_f1': r['development']['f1'],
                             'dev_hr': r['development']['hr_at_100']} for n, r in protocols.items()}
                      for v, protocols in result.items()}, indent=2))


if __name__ == '__main__':
    main()
