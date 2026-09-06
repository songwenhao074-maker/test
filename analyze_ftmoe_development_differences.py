"""Conditional development uncertainty; not an independent significance test."""
import argparse
import json
import numpy as np
from train_ftmoe_end_to_end import ART, METRICS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    base = ART / 'runs' / args.run
    reassessment = ART / 'diagnosis_reassessment' / args.run
    rows = {}
    for v in range(5):
        for seed in [1,2,6]:
            path = (reassessment / f'v{v}_seed{seed}.json' if (reassessment / 'aggregate.json').exists()
                    else base / f'v{v}_seed{seed}/summary.json')
            rows[v,seed] = json.loads(path.read_text())
    report = {'disclosure': 'Development checkpoints and hyperparameters were selected on these same replays. Intervals describe conditional variability only; they are not confirmatory tests.', 'run':args.run}
    for protocol in ['A_last','B_best']:
        report[protocol] = {}
        for metric in METRICS:
            cube = np.array([[[rows[v,s][protocol]['per_replay'][str(r)][metric]
                               for r in [31,101,102]] for s in [1,2,6]] for v in range(5)])
            delta = np.diff(cube, axis=0)
            rng = np.random.default_rng(819)
            boot = np.array([delta[:,rng.integers(3,size=3)][:,:,rng.integers(3,size=3)].mean((1,2)) for _ in range(5000)])
            report[protocol][metric] = {f'v{i}_to_v{i+1}': {
                'mean':float(delta[i].mean()), 'per_model_seed_mean':dict(zip([1,2,6],delta[i].mean(1).tolist())),
                'ci95_conditional':np.quantile(boot[:,i],[.025,.975]).tolist()} for i in range(4)}
    write_json(ART / f'{args.run}_development_differences.json', report)
    print(json.dumps(report['B_best']['f1'], indent=2))


if __name__ == '__main__':
    main()
