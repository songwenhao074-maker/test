"""Apply a common learning-rate selection rule to complete development runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from train_ftmoe_end_to_end import ART, METRICS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = []
    for name in args.runs:
        root = ART / 'runs' / name
        configuration = json.loads((root / 'configuration.json').read_text())['arguments']
        reassessment = ART / 'diagnosis_reassessment' / name
        summaries = []
        for variant in ['v0','v1','v2','v3','v4']:
            for seed in [1,2,6]:
                path = (reassessment / f'{variant}_seed{seed}.json' if (reassessment / 'aggregate.json').exists()
                        else root / f'{variant}_seed{seed}/summary.json')
                summary = json.loads(path.read_text())
                if 'last10_best_score_gain' not in summary:
                    history = [r['metrics']['mean']['score'] for r in summary['epochs']]
                    summary['last10_best_score_gain'] = max(history[-10:]) - max(history[:-10])
                summaries.append(summary)
        runs.append({'name': name, 'configuration': configuration, 'summaries': summaries})
    for key in ['data', 'epochs', 'stop_after']:
        assert len({r['configuration'].get(key) for r in runs}) == 1, key
    result = {}
    for protocol in ['A_last', 'B_best']:
        candidates = []
        for run in runs:
            variants = {f'v{i}': {metric: float(np.mean([s[protocol]['mean'][metric] for s in run['summaries'] if s['variant']==f'v{i}']))
                        for metric in METRICS} for i in range(5)}
            score = float(np.mean([value for values in variants.values() for value in values.values()]))
            candidates.append({'run': run['name'], 'learning_rate': run['configuration']['learning_rate'],
                               'mean_score': score, 'variants': variants})
        maximum = max(c['mean_score'] for c in candidates)
        selected = min([c for c in candidates if c['mean_score'] >= maximum - .002], key=lambda c:c['learning_rate'])
        v = selected['variants']
        chain = {m: all(v[f'v{i+1}'][m] > v[f'v{i}'][m] for i in range(4)) for m in METRICS}
        margin = min(v[f'v{i+1}']['f1'] - v[f'v{i}']['f1'] for i in range(4))
        target = v['v4']['f1'] >= .86 and v['v4'][METRICS[1]] >= .63 and v['v4'][METRICS[2]] >= .58
        result[protocol] = {'candidates': candidates, 'selected_run': selected['run'],
                            'chain': chain, 'minimum_f1_gain': margin,
                            'passed': all(chain.values()) and margin >= .005 and target}
        selected_rows = next(r['summaries'] for r in runs if r['name'] == selected['run'])
        late_counts = {f'v{i}': sum(s['last10_best_score_gain'] > .002 for s in selected_rows if s['variant']==f'v{i}') for i in range(5)}
        result[protocol]['late_improvement_seed_counts'] = late_counts
        result[protocol]['budget_extension_triggered'] = any(count >= 2 for count in late_counts.values())
    result['primary_protocol'] = 'B_best' if result['B_best']['passed'] else 'A_last' if result['A_last']['passed'] else None
    write_json(ART / args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
