"""Apply the registered shared-epoch rule after the complete development run."""
import csv
import json
from pathlib import Path
import numpy as np

ART = Path('artifacts/ftmoe_end_to_end')
RUN = ART / 'runs/initial_lr0003'


def main():
    rule = json.loads((ART / 'protocol_002_shared_epoch.json').read_text())
    output = ART / 'selection_002.json'
    if output.exists():
        raise FileExistsError('Shared epoch has already been selected')
    records = {}
    for variant in ('v0', 'v1', 'v2', 'v3', 'v4'):
        records[variant] = []
        for seed in (1, 2, 6):
            directory = RUN / f'{variant}_seed{seed}'
            assert (directory / 'summary.json').exists(), 'Wait for full development run'
            with (directory / 'epochs.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 60
            records[variant].append(rows)
    candidates = []
    for epoch in range(10, 61):
        values = {variant: {metric: float(np.mean([float(rows[epoch-1][metric]) for rows in runs]))
                    for metric in ('f1', 'hr_at_100', 'ndcg_at_100')}
                    for variant, runs in records.items()}
        margin = min(values[f'v{i+1}']['f1'] - values[f'v{i}']['f1'] for i in range(4))
        chain = all(values[f'v{i+1}'][m] > values[f'v{i}'][m] for i in range(4) for m in ('f1', 'hr_at_100', 'ndcg_at_100'))
        target = values['v4']['f1'] >= .86 and values['v4']['hr_at_100'] >= .63 and values['v4']['ndcg_at_100'] >= .58
        candidates.append({'epoch': epoch, 'mean': values, 'overall_score': float(np.mean([x for v in values.values() for x in v.values()])),
                           'minimum_f1_gain': margin, 'passed': chain and margin >= .005 and target})
    highest = max(c['overall_score'] for c in candidates)
    passing = [c for c in candidates if c['passed'] and highest - c['overall_score'] <= .005]
    selected = min(passing, key=lambda c: (-c['overall_score'], c['epoch'])) if passing else None
    result = {'registered_rule': rule, 'highest_overall_score': highest, 'selected': selected,
              'all_candidates': candidates, 'status': 'selected_for_independent_reproduction' if selected else 'failed'}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'selected': selected, 'passing_epochs': [c['epoch'] for c in passing]}, indent=2))


if __name__ == '__main__':
    main()
