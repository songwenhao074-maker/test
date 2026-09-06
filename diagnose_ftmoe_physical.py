"""Development-only error analysis after the physical full-model comparison."""
import argparse
import json
import numpy as np
import psutil
import torch
from train_ftmoe_end_to_end import ART, create_model, guard, load_data, metric_arrays, write_json


@torch.no_grad()
def predict(model, block):
    x, graph, schedule, labels = block
    pp, cc = [], []
    model.eval()
    for start in range(0, len(x), 64):
        result = model(x[start:start+64], schedule[start:start+64], graph[start:start+64])
        pp.append(result['detection_logits'].softmax(-1)[..., 1].flatten())
        cc.append(result['class_logits'].softmax(-1).reshape(-1, 3))
    return torch.cat(pp).numpy(), torch.cat(cc).numpy(), labels.numpy().ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    torch.set_num_threads(3); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    train, dev, norm, _ = load_data(ART / 'data/protocol_004_physical')
    destination = ART / 'diagnosis_reassessment' / args.run
    reports = []
    for variant in ['v0','v1','v2','v3','v4']:
        guard()
        summary = json.loads((destination / f'{variant}_seed1.json').read_text())
        checkpoint = torch.load(summary['best_checkpoint'], map_location='cpu', weights_only=False)
        model = create_model(variant, 1, norm, **checkpoint.get('model_options', {}))
        model.load_state_dict(checkpoint['model'])
        results = {}
        for key, block in {'train': train, **dev}.items():
            guard()
            prob, classes, labels = predict(model, block)
            metrics = metric_arrays(prob, classes, labels)
            results[key] = {'metrics': metrics}
            if key != 'train':
                np.savez_compressed(destination / f'{variant}_seed1_replay{key}_predictions.npz',
                                    probability=prob, class_probability=classes, labels=labels)
                raw = np.load(ART / f'physical_replays/seed{key}/labels_overload.npy').ravel() > 0
                positive = labels > 0
                results[key]['error_counts'] = {
                    'false_positive': int(((prob >= .5) & ~positive).sum()),
                    'missed_actual_overload': int(((prob < .5) & raw).sum()),
                    'missed_tolerance_only': int(((prob < .5) & positive & ~raw).sum())}
                results[key]['threshold_f1_curve'] = {str(round(float(threshold),2)): float(
                    2 * ((prob >= threshold) & positive).sum() / max((prob >= threshold).sum() + positive.sum(), 1))
                    for threshold in np.arange(.1, .901, .05)}
        reports.append({'variant': variant, 'seed': 1, 'epoch': checkpoint['epoch'], 'results': results})
    write_json(destination / 'error_analysis_seed1.json', reports)
    for report in reports:
        values = list(report['results'].values())
        print(report['variant'], 'train', {k:round(values[0]['metrics'][k],4) for k in ('f1','diagnosis_hr_at_100pct')},
              'dev', {k:round(float(np.mean([v['metrics'][k] for v in values[1:]])),4) for k in ('f1','diagnosis_hr_at_100pct')}, flush=True)


if __name__ == '__main__':
    main()
