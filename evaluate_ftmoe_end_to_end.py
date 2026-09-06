"""Freeze a successful development protocol, then evaluate new registered replays once."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '3'
import numpy as np
import psutil
import torch

from train_ftmoe_end_to_end import ART, ROOT, METRICS, create_model, evaluate, guard, load_data, metric_arrays, sha, write_json
from build_ftmoe_replay_holdout_dataset import profile_classes, dilate_one_block
from calibrate_ftmoe_validation import physical_rule_predictions

SEEDS = (1, 2, 6, 17, 42)


def evaluate_physical_rule(directory):
    """A causal, untrained resource-capacity rule; one result per replay."""
    return [{'replay_seed':int(seed),**metric_arrays(*values)}
            for seed,values in physical_rule_predictions(directory).items()]


def build_test_dataset(development, test_sources, destination):
    manifest = json.loads((development / 'manifest.json').read_text())
    train_indices = manifest['train_blocks']
    names = ('time_series', 'container_demand_series', 'schedule_series', 'labels')
    arrays = {name: list(np.load(development / f'{name}.npy')[train_indices]) for name in names}
    physical = 'host_capacities' in manifest
    thresholds = None if physical else np.percentile(np.concatenate(arrays['time_series']), 98, axis=0).reshape(16, 7)
    for seed, source in test_sources.items():
        features = np.load(source / 'time_series.npy').astype('float32')
        arrays['time_series'].append(features)
        arrays['schedule_series'].append(np.load(source / 'schedule_series.npy').astype('float32'))
        overload = np.load(source / 'labels_overload.npy')
        classes = (np.load(source / 'labels_overload_class.npy') if physical
                   else profile_classes(features, overload, thresholds))
        arrays['labels'].append(dilate_one_block(overload, classes)[1])
        with np.load(source / 'replay_log.npz') as replay:
            arrays['container_demand_series'].append(replay['container_demands'].reshape(202, 112).astype('float32'))
            if physical:
                capacity = np.stack([replay['host_cpu_capacity'], replay['host_ram_capacity'], replay['host_disk_capacity']], -1)
                np.testing.assert_allclose(capacity, manifest['host_capacities'])
    destination.mkdir()
    for name, values in arrays.items():
        np.save(destination / f'{name}.npy', np.stack(values))
    result = {'seeds': [manifest['seeds'][i] for i in train_indices] + list(test_sources),
              'train_blocks': list(range(len(train_indices))),
              'validation_blocks': list(range(len(train_indices), len(train_indices) + len(test_sources))),
              'block_size': 202, 'development_manifest_sha256': sha(development / 'manifest.json'),
              'array_sha256': {name: sha(destination / f'{name}.npy') for name in names},
              'test_source_sha256': {str(path / 'replay_log.npz'): sha(path / 'replay_log.npz') for path in test_sources.values()}}
    if physical:
        result['host_capacities'] = manifest['host_capacities']
    write_json(destination / 'manifest.json', result)


def paired_statistics(rows, metric, bootstrap_seed=819, replicates=5000):
    replay_seeds = sorted({row['replay_seed'] for row in rows})
    cubes = np.empty((5, len(SEEDS), len(replay_seeds)))
    for i in range(5):
        for j, seed in enumerate(SEEDS):
            for k, replay in enumerate(replay_seeds):
                matches = [row for row in rows if row['variant'] == f'v{i}' and row['model_seed'] == seed and row['replay_seed'] == replay]
                assert len(matches) == 1
                cubes[i, j, k] = matches[0][metric]
    rng = np.random.default_rng(bootstrap_seed)
    differences = np.diff(cubes, axis=0)
    samples = np.empty((replicates, 4))
    for sample in range(replicates):
        model_indices = rng.integers(len(SEEDS), size=len(SEEDS))
        replay_indices = rng.integers(len(replay_seeds), size=len(replay_seeds))
        samples[sample] = differences[:, model_indices][:, :, replay_indices].mean(axis=(1, 2))
    return {f'v{i}_to_v{i+1}': {'mean': float(differences[i].mean()),
        'ci95': np.quantile(samples[:, i], [.025, .975]).tolist()} for i in range(4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--protocol', choices=['A_last', 'B_best'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(3); torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    output = ART / args.output
    if output.exists():
        raise FileExistsError('One-shot output exists; refusing to repeat test')
    summaries = {}
    for run in args.runs:
        for path in (ART / 'runs' / run).glob('*/summary.json'):
            summary = json.loads(path.read_text(encoding='utf8'))
            key = (summary['variant'], summary['seed'])
            if key in summaries:
                raise ValueError(f'Duplicate model: {key}')
            summaries[key] = (path, summary)
    expected = {(f'v{i}', seed) for i in range(5) for seed in SEEDS}
    if set(summaries) != expected:
        raise ValueError('Require exactly five models x five registered model seeds')
    configs = [summary['configuration'] for _, summary in summaries.values()]
    for key in ('data', 'epochs', 'learning_rate', 'stop_after'):
        assert len({config.get(key) for config in configs}) == 1, key
    assert len({summary['configuration'].get('moe_context', 'none')
                for (variant, _), (_, summary) in summaries.items() if variant != 'v0'}) == 1, 'Mixed MoE contexts'
    assert len({summary['configuration'].get('expert_dropout', 0.)
                for (variant, _), (_, summary) in summaries.items() if variant != 'v0'}) == 1, 'Mixed expert dropout'
    dev = {f'v{i}': {metric: float(np.mean([summaries[(f'v{i}', seed)][1][args.protocol]['mean'][metric] for seed in SEEDS]))
        for metric in METRICS} for i in range(5)}
    assert all(dev[f'v{i+1}'][metric] > dev[f'v{i}'][metric] for i in range(4) for metric in METRICS), 'Development chain has not passed'
    assert min(dev[f'v{i+1}']['f1'] - dev[f'v{i}']['f1'] for i in range(4)) >= .005, 'Development F1 margin has not passed'
    assert dev['v4']['f1'] >= .86 and dev['v4'][METRICS[1]] >= .63 and dev['v4'][METRICS[2]] >= .58, 'Development target has not passed'
    development = ROOT / configs[0]['data']
    development_manifest = json.loads((development / 'manifest.json').read_text())
    for name, digest in development_manifest['array_sha256'].items():
        assert sha(development / f'{name}.npy') == digest, f'Development array changed: {name}'
    registration = json.loads((ART / 'protocol_001.json').read_text())
    test_seeds = registration['reserved_test_replay_seeds']
    physical = 'host_capacities' in development_manifest
    simulator_environment = development_manifest.get('environment',
        development_manifest.get('registration', {}).get('environment', registration['simulator_environment']))
    replay_root = ART / ('physical_replays' if physical else 'corrected_replays' if 'corrected' in str(development) else 'replays')
    for seed in test_seeds:
        assert not (replay_root / f'seed{seed}').exists(), 'Reserved test already generated; investigate provenance before testing'
    guard()
    output.mkdir(parents=True)
    paths = {}
    for (variant, seed), (summary_path, _) in summaries.items():
        for protocol, filename in [('A_last', 'last.pt'), ('B_best', 'best.pt')]:
            paths[f'{protocol}/{variant}/seed{seed}'] = summary_path.parent / filename
    code = [Path(__file__).resolve(), ROOT / 'train_ftmoe_end_to_end.py', ROOT / 'prepare_ftmoe_end_to_end.py',
            ROOT / 'dump_replay.py', ROOT / 'recovery/PreGANSrc/src/ftmoe_end_to_end.py',
            ROOT / 'recovery/PreGANSrc/src/ftmoe_ablation.py', ROOT / 'train_ftmoe_ablation_existing.py',
            ROOT / 'build_ftmoe_replay_holdout_dataset.py']
    code += list((ROOT / 'simulator').rglob('*.py')) + list((ROOT / 'scheduler').rglob('*.py'))
    code += [ROOT / 'stats/Stats.py', ROOT / 'recovery/Recovery.py']
    code += [ROOT / 'calibrate_ftmoe_validation.py', ROOT / 'diagnose_ftmoe_physical.py']
    code += [ROOT / 'recovery/PreGANSrc/src/ftmoe_context.py']
    code += [ROOT / 'recovery/PreGANSrc/src/ftmoe_expert_regularization.py']
    replay_inputs = [ROOT / 'recovery/PreGANSrc/data/qos/time_series.npy']
    for relative in ['checkpoints/energy_latency_16_Trained.ckpt', 'datasets/energy_latency_16_scheduling.csv']:
        path = ROOT / relative
        if not path.exists():
            path = ROOT / 'scheduler/BaGTI' / relative
        assert path.exists(), f'Missing replay dependency: {path}'
        replay_inputs.append(path)
    lock = {'status': 'frozen_before_test_replay_generation', 'primary_protocol': args.protocol,
            'test_seeds': test_seeds, 'model_seeds': SEEDS, 'development_means': dev,
            'checkpoints': {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()},
            'code_sha256': {str(path.relative_to(ROOT)): sha(path) for path in code},
            'replay_input_sha256': {str(path.relative_to(ROOT)): sha(path) for path in replay_inputs},
            'metric_names': METRICS,
            'training_runs': args.runs, 'simulator_environment': simulator_environment,
            'development_manifest_sha256': sha(development / 'manifest.json'), 'physical_resource_labels': physical}
    write_json(output / 'configuration_lock.json', lock)
    for seed in test_seeds:
        guard()
        destination = replay_root / f'seed{seed}'
        destination.mkdir(parents=True)
        environment = dict(os.environ, **simulator_environment, QOS_OVERLOAD_OUT=str(destination))
        environment.pop('FIXED_SCHEDULE_PATH', None)
        with (destination / 'generation.log').open('w') as log:
            subprocess.run([sys.executable, '-u', 'dump_replay.py', str(seed)], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    dataset = output / 'test_data'
    build_test_dataset(development, {seed: replay_root / f'seed{seed}' for seed in test_seeds}, dataset)
    lock['test_data_manifest_sha256'] = sha(dataset / 'manifest.json')
    for relative, digest in {**lock['code_sha256'], **lock['replay_input_sha256']}.items():
        assert sha(ROOT / relative) == digest, f'Frozen dependency changed: {relative}'
    lock['status'] = 'frozen_before_first_test_model_evaluation'
    write_json(output / 'evaluation_lock.json', lock)
    if physical:
        rule = evaluate_physical_rule(dataset)
        write_json(output / 'physical_rule_cells.json', rule)
        write_json(output / 'physical_rule_summary.json', {
            'description': 'Untrained proposed-demand/capacity rule; one observation per replay, no artificial model-seed replication',
            'metrics': {m: {'mean': float(np.mean([row[m] for row in rule])),
                            'std_across_replays': float(np.std([row[m] for row in rule], ddof=1))} for m in METRICS}})
    _, loaders, normalization, _ = load_data(dataset)
    rows = []
    for key, checkpoint_path in paths.items():
        protocol, variant, seed_string = key.split('/')
        seed = int(seed_string[4:])
        guard()
        assert sha(checkpoint_path) == lock['checkpoints'][key]['sha256']
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert state['normalization'] == normalization, 'Preprocessing changed'
        model = create_model(variant, seed, normalization, **state.get('model_options', {}))
        model.load_state_dict(state['model'], strict=True)
        result = evaluate(model, loaders)
        for replay, values in result['per_replay'].items():
            rows.append({'protocol': protocol, 'variant': variant, 'model_seed': seed, 'replay_seed': int(replay),
                         'epoch': state['epoch'], **values})
        print(key, result['mean'], flush=True)
    write_json(output / 'test_cells.json', rows)
    with (output / 'test_cells.csv').open('w', newline='', encoding='utf8') as handle:
        fields = [key for key in rows[0] if key != 'class_confusion']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    result = {}
    for protocol in ('A_last', 'B_best'):
        selected = [row for row in rows if row['protocol'] == protocol]
        variants = {f'v{i}': {metric: {'mean': float(np.mean([r[metric] for r in selected if r['variant'] == f'v{i}'])),
                  'std': float(np.std([r[metric] for r in selected if r['variant'] == f'v{i}'], ddof=1))}
                  for metric in METRICS} for i in range(5)}
        result[protocol] = {'variants': variants,
            'strict_mean_chain': {m: all(variants[f'v{i+1}'][m]['mean'] > variants[f'v{i}'][m]['mean'] for i in range(4)) for m in METRICS},
            'paired_two_way_cluster_bootstrap': {m: paired_statistics(selected, m) for m in METRICS}}
    write_json(output / 'test_summary.json', {'primary_protocol': args.protocol, **result})


if __name__ == '__main__':
    main()
