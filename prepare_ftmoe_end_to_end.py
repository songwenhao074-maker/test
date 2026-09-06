"""Generate registered development replays and build a train-only normalized corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '3'
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts/ftmoe_end_to_end'
DATA = ROOT / 'recovery/PreGANSrc/data'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard():
    while psutil.virtual_memory().available / 2**30 < 4.5:
        print('WAIT: available RAM below 4.5 GiB', flush=True)
        time.sleep(15)
    if shutil.disk_usage(ROOT).free / 2**30 < 20:
        raise RuntimeError('Free disk below 20 GiB')


def build(seeds, output):
    from build_ftmoe_replay_holdout_dataset import profile_classes, dilate_one_block
    sources = {}
    for seed in seeds:
        if seed in (42, 1, 6):
            sources[seed] = DATA / f'qos_overload_fixed_rs35_s{seed}'
        elif seed in (17, 23, 31):
            sources[seed] = DATA / f'qos_overload_formal_rs35_s{seed}'
        else:
            sources[seed] = ART / 'replays' / f'seed{seed}'
    time_blocks, demand_blocks, schedule_blocks, raw_labels = [], [], [], []
    for seed in seeds:
        source = sources[seed]
        time_blocks.append(np.load(source / 'time_series.npy').astype('float32'))
        schedule_blocks.append(np.load(source / 'schedule_series.npy').astype('float32'))
        raw_labels.append(np.load(source / 'labels_overload.npy'))
        with np.load(source / 'replay_log.npz') as replay:
            demand_blocks.append(replay['container_demands'].reshape(202, 112).astype('float32'))
    thresholds = np.percentile(np.concatenate(time_blocks[:5]), 98, axis=0).reshape(16, 7)
    labels = [dilate_one_block(overload, profile_classes(features, overload, thresholds))[1]
              for features, overload in zip(time_blocks, raw_labels)]
    arrays = {'time_series': np.stack(time_blocks), 'container_demand_series': np.stack(demand_blocks),
              'schedule_series': np.stack(schedule_blocks), 'labels': np.stack(labels)}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in arrays.items():
        np.save(output / f'{name}.npy', value)
    manifest = {'seeds': seeds, 'train_blocks': list(range(5)), 'validation_blocks': list(range(5, len(seeds))),
                'block_size': 202, 'label_policy': 'original train-p98 profile classes; +/-1 dilation within block',
                'source_sha256': {str(sources[s] / 'replay_log.npz'): sha(sources[s] / 'replay_log.npz') for s in seeds},
                'array_sha256': {name: sha(output / f'{name}.npy') for name in arrays},
                'class_counts': {str(s): np.bincount(y.ravel(), minlength=4).tolist() for s, y in zip(seeds, labels)}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    protocol = json.loads((ART / 'protocol_001.json').read_text())
    for seed in (101, 102):
        dest = ART / 'replays' / f'seed{seed}'
        if (dest / 'replay_log.npz').exists():
            continue
        if args.build_only:
            raise FileNotFoundError(dest)
        guard()
        dest.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, **protocol['simulator_environment'], QOS_OVERLOAD_OUT=str(dest))
        environment.pop('FIXED_SCHEDULE_PATH', None)
        print(f'Generating development replay {seed}', flush=True)
        with (dest / 'generation.log').open('w') as log:
            subprocess.run([sys.executable, '-u', 'dump_replay.py', str(seed)], cwd=ROOT,
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    output = ART / 'data/protocol_001'
    if not output.exists():
        build(protocol['training_replay_seeds'] + protocol['validation_replay_seeds'], output)


if __name__ == '__main__':
    main()
