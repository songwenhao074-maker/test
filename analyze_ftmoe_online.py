"""Recompute prequential metrics and paired online comparisons from predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def metrics(probability, classes, labels):
    p, c, y = probability.reshape(-1), classes.reshape(-1, 3), labels.reshape(-1)
    anomaly, predicted = y > 0, p >= .5
    tp = int((predicted & anomaly).sum()); fp = int((predicted & ~anomaly).sum())
    fn = int((~predicted & anomaly).sum()); tn = int((~predicted & ~anomaly).sum())
    resource = c.argmax(-1) + 1
    return {'f1': 2 * tp / max(2 * tp + fp + fn, 1) if anomaly.any() else None,
        'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1) if anomaly.any() else None,
        'accuracy': (tp + tn) / max(len(y), 1),
        'pr_auc': float(average_precision_score(anomaly, p)) if anomaly.any() else None,
        'diagnosis_accuracy_hr_equals_ndcg': float((resource[anomaly] == y[anomaly]).mean()) if anomaly.any() else None,
        'resource_macro_f1': float(f1_score(y[anomaly], resource[anomaly], labels=[1, 2, 3],
            average='macro', zero_division=0)) if anomaly.any() else None,
        'class_support': np.bincount(y, minlength=4).tolist(),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'positives': int(anomaly.sum()), 'samples': len(y)}


def summarize_arrays(probability, class_probability, labels, raw_labels):
    n = len(labels)
    if probability.shape != labels.shape or class_probability.shape != (*labels.shape, 3):
        raise ValueError('Prediction shapes do not match labels')
    if n < 4 or not np.isfinite(probability).all() or not np.isfinite(class_probability).all():
        raise ValueError('Insufficient or nonfinite predictions')
    if (labels < 0).any() or (labels > 3).any():
        raise ValueError('Unfinalized or invalid labels')
    slices = {'full': (0, n), 'first_half': (0, n // 2), 'second_half': (n // 2, n),
              'first_quarter': (0, n // 4), 'last_quarter': (n - n // 4, n)}
    result = {key: metrics(probability[a:b], class_probability[a:b], labels[a:b])
              for key, (a, b) in slices.items()}
    result['raw'] = {key: metrics(probability[a:b], class_probability[a:b], raw_labels[a:b])
                     for key, (a, b) in slices.items()}
    result['rolling'] = [dict(end=t, start=t-99, **metrics(probability[t-100:t], class_probability[t-100:t], labels[t-100:t]))
                         for t in range(100, n + 1, 10)]
    result['segments'] = [dict(end=t, start=t-99, **metrics(probability[t-100:t], class_probability[t-100:t], labels[t-100:t]))
                          for t in range(100, n + 1, 100)]
    later = [r for r in result['rolling'] if r['start'] > n // 2]
    valid = [r['f1'] for r in later if r['f1'] is not None]
    segments = [(r['end'], r['f1']) for r in result['segments'] if r['f1'] is not None]
    first, last = result['first_quarter']['f1'], result['last_quarter']['f1']
    result['stability'] = {'later_rolling_minimum': min(valid) if valid else None,
        'later_rolling_below065_fraction': float(np.mean(np.asarray(valid) < .65)) if valid else None,
        'later_rolling_valid': len(valid), 'later_rolling_total': len(later),
        'last_minus_first_quarter': last-first if last is not None and first is not None else None,
        'nonoverlap_f1_slope_per_interval': float(np.polyfit(*zip(*segments), 1)[0]) if len(segments) >= 2 else None}
    return result


def bootstrap_difference(rows, first, second):
    seeds = sorted({r['configuration']['model_seed'] for r in rows})
    replays = sorted({r['configuration']['replay_seed'] for r in rows})
    lookup = {(r['configuration']['method'], r['configuration']['model_seed'], r['configuration']['replay_seed']):
              r['metrics']['second_half']['f1'] for r in rows}
    differences = np.asarray([[lookup[first, s, r]-lookup[second, s, r] for r in replays] for s in seeds])
    rng = np.random.RandomState(15001)
    means = []
    for _ in range(5000):
        i = rng.randint(len(seeds), size=len(seeds)); j = rng.randint(len(replays), size=len(replays))
        means.append(float(differences[np.ix_(i, j)].mean()))
    return {'mean': float(differences.mean()), 'paired_differences': differences.tolist(),
        'model_seeds': seeds, 'replay_seeds': replays,
        'ci95_paired_crossed_bootstrap': np.percentile(means, [2.5, 97.5]).tolist(),
        'resamples': 5000, 'bootstrap_seed': 15001}


def analyze(runs, output, stage):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for path in sorted(Path(runs).glob('*/configuration.json')):
        folder = path.parent
        if not (folder / 'summary.json').exists():
            raise ValueError('Incomplete run: ' + str(folder))
        config = json.loads(path.read_text(encoding='utf8'))
        with np.load(folder / 'predictions.npz') as pred:
            summary = summarize_arrays(pred['probability'], pred['class_probability'], pred['labels'], pred['raw_labels'])
            topology = {k: pred[k].copy() for k in ('expert_count', 'unmatched_ratio')}
        recorded = json.loads((folder / 'summary.json').read_text(encoding='utf8'))
        if recorded['metrics'] != summary:
            raise AssertionError('Independent metric recomputation differs: ' + str(folder))
        rows.append({'configuration': config, 'metrics': summary, 'folder': str(folder), 'topology': topology,
                     'reference': json.loads((folder / 'reference.json').read_text()),
                     'initial_state_hash':recorded['initial_state_hash'],
                     'timing':{key:recorded.get(key) for key in ('elapsed_seconds','rss_gib','prediction_mean_seconds',
                         'prediction_p95_seconds','update_total_seconds')},
                     'frozen_parameters_unchanged':recorded['frozen_parameters_unchanged']})
    seeds, replays, steps = ({'pilot': ([1], [301], 300),
        'development': ([1, 2], [301, 302], 1000),
        'confirmation': ([1, 2, 6, 17, 42], [401, 402, 403], 2000)})[stage]
    keys = [(r['configuration']['method'], r['configuration']['model_seed'], r['configuration']['replay_seed'],
             r['configuration']['learning_rate'] if stage == 'development' else None) for r in rows]
    expected = {(m, s, r, lr if stage == 'development' else None) for m in 'ABCD' for s in seeds for r in replays
                for lr in ([0.] if m == 'A' else [1e-5, 3e-5, 1e-4])} if stage == 'development' else {
                    (m,s,r,None) for m in 'ABCD' for s in seeds for r in replays}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Run grid incomplete or duplicated')
    if any(r['configuration']['steps'] != steps for r in rows):
        raise ValueError('Unequal run horizon')
    if stage == 'pilot' and any(r['configuration']['learning_rate'] != (0. if r['configuration']['method']=='A' else 3e-5) for r in rows):
        raise ValueError('Unexpected pilot learning rate')
    for seed in seeds:
        for replay in replays:
            group=[r for r in rows if r['configuration']['model_seed']==seed and r['configuration']['replay_seed']==replay]
            if len({r['initial_state_hash'] for r in group})!=1 or len({r['configuration']['stream_sha256'] for r in group})!=1:
                raise ValueError('Paired methods did not start from identical state/data')
    if not all(r['frozen_parameters_unchanged'] for r in rows):
        raise ValueError('Frozen parameter integrity failed')
    problems = []
    if any(r['metrics']['second_half']['f1'] is None for r in rows):
        problems.append('No positive labels in at least one second half; F1 comparison undefined')
    report = {'stage': stage, 'runs': [{k:v for k,v in r.items() if k not in ('topology','reference')} for r in rows],
              'problems': problems, 'ready_for_next_stage': False, 'accepted': False}
    if stage == 'development' and not problems:
        selected = {}
        for method in 'BCD':
            scores = {lr: float(np.mean([r['metrics']['second_half']['f1'] for r in rows
                if r['configuration']['method']==method and r['configuration']['learning_rate']==lr]))
                for lr in (1e-5, 3e-5, 1e-4)}
            selected[method] = {'learning_rate': max(sorted(scores), key=lambda lr:scores[lr]), 'scores': scores}
        report['selection'] = selected
        d = max(selected['D']['scores'].values()); b = max(selected['B']['scores'].values())
        report['ready_for_next_stage'] = d >= .65 and d > b
        if not report['ready_for_next_stage']:
            problems.append('Development did not show the approved advantage/floor; report before confirmation')
    elif not problems:
        means = {m: float(np.mean([r['metrics']['second_half']['f1'] for r in rows if r['configuration']['method']==m])) for m in 'ABCD'}
        report['second_half_means'] = means
        report['second_half_std_across_pairs'] = {m:float(np.std([r['metrics']['second_half']['f1'] for r in rows
            if r['configuration']['method']==m],ddof=1)) if len(seeds)*len(replays)>1 else None for m in 'ABCD'}
        report['paired'] = {f'D_minus_{m}': bootstrap_difference(rows, 'D', m) for m in 'BC'}
        report['accepted'] = means['D'] >= .65 and means['D'] > means['B']
        report['dynamic_contribution_positive_mean'] = means['D'] > means['C']
        report['ready_for_next_stage'] = report['accepted'] if stage == 'pilot' else False
        if stage == 'pilot':
            for row in rows:
                counts = row['metrics']['full']['class_support']
                if counts[0] < 50 or sum(counts[1:]) < 50:
                    problems.append('Pilot requires at least 50 normal and 50 anomalous host-intervals')
                    break
        if not report['accepted']:
            problems.append('D second-half F1 is below 0.65 or does not exceed B')
        if problems:
            report['ready_for_next_stage'] = False
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    with (output / 'comparison.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method','model_seed','replay_seed','learning_rate','full_f1','first_half_f1','second_half_f1','raw_second_half_f1'])
        for row in rows:
            c,m = row['configuration'],row['metrics']
            writer.writerow([c[k] for k in ('method','model_seed','replay_seed','learning_rate')]+
                            [m[k]['f1'] for k in ('full','first_half','second_half')]+[m['raw']['second_half']['f1']])
    plot_rows(rows, output)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'}, ensure_ascii=False, indent=2))
    return report


def plot_rows(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {'A':'#777777','B':'#d55e00','C':'#0072b2','D':'#009e73'}
    for method in 'ABCD':
        group = [r for r in rows if r['configuration']['method']==method]
        if not group:
            continue
        # Development configurations remain individual traces, never silently blended.
        for row in group:
            roll = row['metrics']['rolling']
            axes[0,0].plot([x['end'] for x in roll], [x['f1'] if x['f1'] is not None else np.nan for x in roll],
                           color=colors[method], alpha=.35, linewidth=.8)
            axes[1,0].plot(np.arange(len(row['topology']['expert_count']))+1, row['topology']['expert_count'], color=colors[method],alpha=.3)
            axes[1,1].plot([x['step'] for x in row['reference']], [x['mean']['f1'] for x in row['reference']], color=colors[method],alpha=.4)
        if len({r['configuration']['learning_rate'] for r in group}) == 1:
            roll = group[0]['metrics']['rolling']
            values=np.array([[x['f1'] if x['f1'] is not None else np.nan for x in r['metrics']['rolling']] for r in group])
            valid=np.isfinite(values).sum(0)
            mean=np.divide(np.nansum(values,0),valid,out=np.full(len(valid),np.nan),where=valid>0)
            axes[0,0].plot([x['end'] for x in roll],mean,color=colors[method],label=method,linewidth=2)
        else:
            axes[0,0].plot([],[],color=colors[method],label=method+' (all learning rates)')
        vals=[r['metrics']['second_half']['f1'] if r['metrics']['second_half']['f1'] is not None else np.nan for r in group]
        axes[0,1].scatter([ord(method)-65]*len(vals),vals,color=colors[method],s=22)
    if len({r['configuration']['learning_rate'] for r in rows if r['configuration']['method']=='B'})==1:
        pairs=sorted({(r['configuration']['model_seed'],r['configuration']['replay_seed']) for r in rows})
        for pair in pairs:
            values=[]
            for method in 'ABCD':
                match=[r for r in rows if (r['configuration']['model_seed'],r['configuration']['replay_seed'])==pair and r['configuration']['method']==method]
                value=match[0]['metrics']['second_half']['f1'] if match else None
                values.append(value if value is not None else np.nan)
            axes[0,1].plot(range(4),values,color='#bbbbbb',alpha=.5,zorder=0)
    secondary=axes[1,0].twinx()
    for row in rows:
        if row['configuration']['method']=='D':
            secondary.plot(np.arange(len(row['topology']['unmatched_ratio']))+1,row['topology']['unmatched_ratio'],color='#cc79a7',alpha=.25,linewidth=.7)
    secondary.set_ylabel('Unmatched fraction (D)')
    axes[0,0].axhline(.65,color='black',linestyle='--',linewidth=1)
    axes[0,1].axhline(.65,color='black',linestyle='--',linewidth=1)
    axes[0,0].legend(); axes[0,1].set_xticks(range(4));axes[0,1].set_xticklabels(list('ABCD'))
    for ax,title in zip(axes.flat,['Prequential rolling F1 (100 intervals)','Paired second-half F1','Expert count and unmatched inputs','Fixed reference retention']):
        ax.set_title(title);ax.grid(alpha=.2)
    for ax in (axes[0,0],axes[1,0],axes[1,1]):ax.set_xlabel('Simulation interval')
    for ax in (axes[0,0],axes[0,1],axes[1,1]):ax.set_ylim(0,1.02);ax.set_ylabel('F1')
    axes[1,0].set_ylabel('EAGate experts')
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(output/('online_comparison.'+suffix),dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['pilot','development','confirmation'],required=True)
    args=parser.parse_args();analyze(args.runs,args.output,args.stage)
