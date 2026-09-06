"""Compare shared epochs on development logs; never mix epochs across variants."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='initial_lr0003')
    args = parser.parse_args()
    root = Path('artifacts/ftmoe_end_to_end/runs') / args.run
    records = {}
    for variant in ('v0', 'v1', 'v2', 'v3', 'v4'):
        paths = [root / f'{variant}_seed{seed}/epochs.csv' for seed in (1, 2, 6)]
        if not all(path.exists() for path in paths):
            break
        records[variant] = [list(csv.DictReader(path.open())) for path in paths]
    if not records:
        return
    maximum = min(len(rows) for runs in records.values() for rows in runs)
    results = []
    for epoch in range(1, maximum + 1):
        variants = {variant: {m: float(np.mean([float(rows[epoch-1][m]) for rows in runs]))
                              for m in ('f1', 'hr_at_100', 'ndcg_at_100')}
                    for variant, runs in records.items()}
        chains = {m: all(variants[f'v{i+1}'][m] > variants[f'v{i}'][m] for i in range(len(variants)-1))
                  for m in ('f1', 'hr_at_100', 'ndcg_at_100')}
        margin = min((variants[f'v{i+1}']['f1'] - variants[f'v{i}']['f1'] for i in range(len(variants)-1)), default=0.)
        results.append({'epoch': epoch, 'variants': variants, 'chains': chains, 'min_f1_margin': margin,
                        'overall_score': float(np.mean([value for values in variants.values() for value in values.values()]))})
    output = {'complete_variant_count': len(records), 'maximum_shared_epoch': maximum,
              'top_overall_score': sorted(results, key=lambda row: -row['overall_score'])[:5],
              'passing_epochs': [row for row in results if all(row['chains'].values()) and row['min_f1_margin'] >= .005]}
    (root / 'shared_epoch_diagnostics.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({'complete_variants': len(records), 'shared_epochs': maximum,
                      'top_score_epochs': [(r['epoch'], round(r['overall_score'], 5), r['chains'], round(r['min_f1_margin'], 5)) for r in output['top_overall_score']],
                      'passing_epochs': [r['epoch'] for r in output['passing_epochs']]}, indent=2))


if __name__ == '__main__':
    main()
