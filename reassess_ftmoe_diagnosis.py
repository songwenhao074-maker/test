"""Reassess saved development epochs without changing legacy results or weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_ftmoe_end_to_end import ART, METRICS, aggregate, guard, sha, write_json


def convert(validation):
    per = validation['per_replay']
    for metrics in per.values():
        confusion = np.asarray(metrics['class_confusion'])
        accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
        metrics['diagnosis_hr_at_100pct'] = accuracy
        metrics['diagnosis_ndcg_at_100pct'] = accuracy
    for metric in METRICS:
        validation['mean'][metric] = float(np.mean([m[metric] for m in per.values()]))
    validation['mean']['score'] = float(np.mean([validation['mean'][m] for m in METRICS]))
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ART / 'runs' / args.run
    destination = ART / 'diagnosis_reassessment' / args.run
    destination.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for variant in ['v0', 'v1', 'v2', 'v3', 'v4']:
        for seed in [1, 2, 6]:
            source = run / f'{variant}_seed{seed}'
            assert (source / 'summary.json').exists(), f'Incomplete training: {source}'
            rows = []
            for path in sorted((source / 'checkpoints_by_epoch').glob('epoch*.pt')):
                guard()
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                rows.append({'epoch': checkpoint['epoch'], 'checkpoint': str(path),
                             'metrics': convert(checkpoint['validation'])})
            assert len(rows) == 30
            best = max(rows, key=lambda r: r['metrics']['mean']['score'])
            summary = {'variant': variant, 'seed': seed, 'best_epoch': best['epoch'],
                       'best_checkpoint': best['checkpoint'], 'best_sha256': sha(best['checkpoint']),
                       'A_last': rows[-1]['metrics'], 'B_best': best['metrics'], 'epochs': rows}
            write_json(destination / f'{variant}_seed{seed}.json', summary)
            all_summaries.append(summary)
    result = aggregate(all_summaries)
    write_json(destination / 'aggregate.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
