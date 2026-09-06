"""Verify and summarize the single-seed scenario experiments without bootstrap pseudo-CIs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_ftmoe_online import summarize_arrays, plot_rows


def main(folder):
    rows = []
    first = None
    result = json.loads((folder / 'result.json').read_text(encoding='utf8'))
    for method in 'ABCD':
        run = folder / 'runs' / method
        summary = json.loads((run / 'summary.json').read_text(encoding='utf8'))
        config = summary['configuration']
        with np.load(run / 'predictions.npz') as saved:
            pred = dict(saved)
        with np.load(Path(config['stream']) / 'stream.npz') as saved:
            raw = saved['raw_labels']
            ratio = saved['post_totals'] / saved['capacities']
            np.testing.assert_array_equal(raw, np.where((ratio > 1).any(-1), ratio.argmax(-1) + 1, 0))
        y = raw[:-1].copy()
        for t in range(len(y)):
            for delta in (-1, 1):
                if 0 <= t + delta < len(raw):
                    use = (y[t] == 0) & (raw[t + delta] > 0)
                    y[t, use] = raw[t + delta, use]
        np.testing.assert_array_equal(pred['labels'], y)
        np.testing.assert_array_equal(pred['raw_labels'], raw[:-1])
        metrics = summarize_arrays(**{k: pred[k] for k in ('probability', 'class_probability', 'labels', 'raw_labels')})
        assert metrics == summary['metrics']
        if first is None:
            first = (pred['probability'][0], summary['initial_state_hash'], config['stream_sha256'])
        np.testing.assert_array_equal(pred['probability'][0], first[0])
        assert (summary['initial_state_hash'], config['stream_sha256']) == first[1:]
        assert summary['frozen_parameters_unchanged']
        if method == 'A':
            assert summary['initial_state_hash'] == summary['final_state_hash']
        reference = json.loads((run / 'reference.json').read_text(encoding='utf8'))
        resource_recall = {}
        for cls in (1, 2, 3):
            mask = y == cls
            resource_recall[str(cls)] = float((pred['probability'][mask] >= .5).mean()) if mask.any() else None
        rows.append(dict(configuration=config, metrics=metrics, reference=reference,
                         topology={k: pred[k] for k in ('expert_count', 'unmatched_ratio')},
                         per_resource_detection_recall=resource_recall,
                         expert_count_range=[int(pred['expert_count'].min()), int(pred['expert_count'].max())],
                         reference_f1_initial=reference[0]['mean']['f1'], reference_f1_final=reference[-1]['mean']['f1'],
                         timing={k: summary[k] for k in ('elapsed_seconds', 'rss_gib', 'prediction_mean_seconds',
                                                       'prediction_p95_seconds', 'update_total_seconds')}))
    out = folder / 'analysis'
    out.mkdir(exist_ok=True)
    second = result['second_half_f1']
    report = dict(verification='saved metrics, physical and mature labels, paired initial state/output, frozen parameters all verified',
                  distribution=result['distribution'],
                  paired_differences={f'D_minus_{m}': second['D'] - second[m]
                                      if second['D'] is not None and second[m] is not None else None for m in 'BC'},
                  ci95=None, inference='one model and one replay; no independent-repeat confidence interval',
                  runs=[{k: v for k, v in row.items() if k != 'topology'} for row in rows])
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    with (out / 'comparison.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'full_f1', 'first_half_f1', 'second_half_f1', 'raw_second_half_f1',
                         'second_precision', 'second_recall', 'reference_f1_final', 'elapsed_seconds'])
        for row in rows:
            m = row['metrics']
            writer.writerow([row['configuration']['method'], *[m[k]['f1'] for k in ('full', 'first_half', 'second_half')],
                             m['raw']['second_half']['f1'], m['second_half']['precision'], m['second_half']['recall'],
                             row['reference_f1_final'], row['timing']['elapsed_seconds']])
    plot_rows(rows, out)
    print(json.dumps({'verified': str(folder), 'paired_differences': report['paired_differences']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
