"""Read-only information and learning-curve diagnostics on development data."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts/ftmoe_end_to_end'


def dataset_diagnostics():
    data = ART / 'data/protocol_001'
    y = np.load(data / 'labels.npy')
    t = np.load(data / 'time_series.npy')
    train = y[:5] > 0
    positive_counts = train.sum(axis=0)
    # Optimal empirical F1 when equal (time,host) inputs must share a prediction.
    ordered = np.sort(positive_counts.ravel())[::-1]
    cumulative_tp = ordered.cumsum()
    n_pred = 5 * np.arange(1, len(ordered) + 1)
    f1 = 2 * cumulative_tp / np.maximum(n_pred + train.sum(), 1)
    return {'all_replays_share_time_trace': all(np.array_equal(t[0], block) for block in t),
            'train_positive_fraction': float(train.mean()),
            'conflicting_time_host_fraction': float(((positive_counts > 0) & (positive_counts < 5)).mean()),
            'empirical_train_F1_upper_bound_for_trace_only': float(f1.max()),
            'note': 'Optimistic in-sample information bound, not a trained baseline or unseen-data bound.',
            'train_class_counts': np.bincount(y[:5].ravel(), minlength=4).tolist(),
            'validation_class_counts': np.bincount(y[5:].ravel(), minlength=4).tolist()}


def main():
    report = {'dataset': dataset_diagnostics(), 'runs': {}}
    for run in sorted((ART / 'runs').glob('*')):
        models = {}
        for path in sorted(run.glob('*/summary.json')):
            s = json.loads(path.read_text(encoding='utf8'))
            models[path.parent.name] = {'epoch': s['best_epoch'],
                'A': {k: round(s['A_last']['mean'][k], 5) for k in ('f1', 'hr_at_100', 'ndcg_at_100')},
                'B': {k: round(s['B_best']['mean'][k], 5) for k in ('f1', 'hr_at_100', 'ndcg_at_100')},
                'recent_gain': round(s['last10_best_score_gain'], 5)}
        if models:
            report['runs'][run.name] = models
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
