"""Calibrate fixed development checkpoints under a registered common threshold grid."""
import argparse
import json
import numpy as np
import psutil
import torch
from diagnose_ftmoe_physical import predict
from train_ftmoe_end_to_end import ART, ROOT, METRICS, aggregate, create_model, guard, load_data, metric_arrays, sha, write_json


def physical_rule_predictions(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    demands = np.load(directory/'container_demand_series.npy').reshape(-1,202,16,7)
    schedule = np.load(directory/'schedule_series.npy')
    labels = np.load(directory/'labels.npy')
    ratio = np.einsum('btch,btcf->bthf',schedule,demands)[...,[0,1,4]] / np.asarray(manifest['host_capacities'])
    result = {}
    for block in manifest['validation_blocks']:
        values = ratio[block].reshape(-1,3)
        maximum = values.max(-1)
        probability = maximum/(1+maximum)
        probability[maximum == 1] = np.nextafter(.5,0.)
        classes = values/np.maximum(values.sum(-1,keepdims=True),1e-12)
        result[str(manifest['seeds'][block])] = (probability,classes,labels[block].ravel())
    return result


def metrics_at_threshold(probability, classes, labels, threshold):
    result = metric_arrays(probability, classes, labels)
    prediction = probability >= threshold
    positive = labels > 0
    tp = int((prediction & positive).sum()); fp = int((prediction & ~positive).sum())
    fn = int((~prediction & positive).sum()); tn = int((~prediction & ~positive).sum())
    result.update(tp=tp, fp=fp, fn=fn, tn=tn, f1=2*tp/max(2*tp+fp+fn,1),
                  precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1))
    return result


def select_threshold(predictions, grid):
    curve = []
    for threshold in grid:
        scores = []
        for probability, _, labels in predictions.values():
            positive = labels > 0
            predicted = probability >= threshold
            scores.append(float(2*(predicted & positive).sum()/max(predicted.sum()+positive.sum(),1)))
        curve.append({'threshold':threshold, 'f1':float(np.mean(scores)), 'per_replay_f1':dict(zip(predictions,scores))})
    best = max(curve, key=lambda row:(row['f1'], -abs(row['threshold']-.5), -row['threshold']))
    threshold = best['threshold']
    per_replay = {key:metrics_at_threshold(*values, threshold) for key,values in predictions.items()}
    numeric = [key for key,value in next(iter(per_replay.values())).items() if isinstance(value,(int,float))]
    mean = {key:float(np.mean([row[key] for row in per_replay.values()])) for key in numeric}
    mean['score'] = float(np.mean([mean[key] for key in METRICS]))
    return threshold, {'mean':mean,'per_replay':per_replay}, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,6])
    parser.add_argument('--variants', nargs='+', choices=['v0','v1','v2','v3','v4'], default=['v0','v1','v2','v3','v4'])
    args = parser.parse_args()
    torch.set_num_threads(3); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    registration = ART / 'protocol_008_validation_calibration.json'
    protocol = json.loads(registration.read_text())
    root = ART / 'runs' / args.run
    original_configuration = json.loads((root / 'configuration.json').read_text())
    assert original_configuration['arguments']['epochs'] == 60
    directory = ROOT / original_configuration['arguments']['data']
    _, validation, normalization, manifest = load_data(directory)
    assert set(validation) == {'31','101','102'}
    assert not set(manifest['seeds']) & {201,202,203,204,205}
    for name,digest in manifest['array_sha256'].items():
        assert sha(directory/f'{name}.npy')==digest, f'Changed development array: {name}'
    sources = {}
    for variant in args.variants:
        for seed in args.seeds:
            source = root / f'{variant}_seed{seed}'
            assert (source/'summary.json').exists(), f'Incomplete training: {source}'
            for name in ['last.pt','best.pt']:
                sources[str(source/name)] = sha(source/name)
    output = ART / 'calibrated' / args.run
    configuration = {'run':args.run, 'seeds':args.seeds, 'variants':args.variants, 'registration_sha256':sha(registration),
                     'script_sha256':sha(__file__), 'source_checkpoints':sources,
                     'evaluation_code_sha256':{name:sha(ROOT/name) for name in [
                         'train_ftmoe_end_to_end.py','diagnose_ftmoe_physical.py',
                         'recovery/PreGANSrc/src/ftmoe_end_to_end.py',
                         'recovery/PreGANSrc/src/ftmoe_context.py',
                         'recovery/PreGANSrc/src/ftmoe_expert_regularization.py',
                         'recovery/PreGANSrc/src/ftmoe_fusion_controls.py',
                         'recovery/PreGANSrc/src/ftmoe_source_fusion.py',
                         'recovery/PreGANSrc/src/ftmoe_ablation.py']},
                     'training_configuration':original_configuration,
                     'development_manifest_sha256':sha(directory/'manifest.json')}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'configuration.json').exists():
        assert json.loads((output/'configuration.json').read_text()) == configuration
    else:
        write_json(output/'configuration.json',configuration)
    rule_threshold, rule_metrics, rule_curve = select_threshold(physical_rule_predictions(directory),protocol['grid'])
    write_json(output/'physical_rule_calibration.json',{'threshold':rule_threshold,'metrics':rule_metrics,
        'curve':rule_curve,'description':'One common development-selected threshold for the physical rule; no model-seed replication'})
    summaries = []
    for variant in args.variants:
        for seed in args.seeds:
            result_path = output/f'{variant}_seed{seed}.json'
            if result_path.exists():
                summaries.append(json.loads(result_path.read_text())); continue
            original = json.loads((root/f'{variant}_seed{seed}/summary.json').read_text())
            result = {'variant':variant,'seed':seed,'configuration':original['configuration'],
                      'original_fixed_threshold':{key:original[key] for key in ['A_last','B_best']},
                      'calibration':{}}
            for key, filename in [('A_last','last.pt'),('B_best','best.pt')]:
                guard()
                checkpoint_path = root/f'{variant}_seed{seed}'/filename
                checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                assert checkpoint['normalization'] == normalization
                model = create_model(variant,seed,normalization,**checkpoint.get('model_options', {}))
                model.load_state_dict(checkpoint['model'])
                predictions = {replay:predict(model,block) for replay,block in validation.items()}
                threshold, metrics, curve = select_threshold(predictions,protocol['grid'])
                result[key] = metrics
                result['calibration'][key] = {'threshold':threshold, 'epoch':checkpoint['epoch'],
                    'checkpoint':str(checkpoint_path),'checkpoint_sha256':sources[str(checkpoint_path)],'curve':curve}
                for replay, (probability,classes,labels) in predictions.items():
                    np.savez_compressed(output/f'{variant}_seed{seed}_{key}_replay{replay}.npz',
                        probability=probability,class_probability=classes,labels=labels)
            write_json(result_path,result)
            summaries.append(result)
            print(variant,seed,{key:{'threshold':result['calibration'][key]['threshold'],
                                   'f1':round(result[key]['mean']['f1'],5)} for key in ['A_last','B_best']},flush=True)
    write_json(output/'aggregate.json',aggregate(summaries))
    print(json.dumps(aggregate(summaries),indent=2))


if __name__=='__main__':
    main()
