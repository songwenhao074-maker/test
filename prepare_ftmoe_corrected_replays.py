"""Rebuild every development block under the corrected live-container trace semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import psutil

from prepare_ftmoe_end_to_end import ART, ROOT, guard, sha
from build_ftmoe_replay_holdout_dataset import profile_classes, dilate_one_block

SEEDS = [42, 1, 6, 17, 23, 31, 101, 102]
BASE = ART / 'corrected_replays'
OUTPUT = ART / 'data/protocol_003_corrected'


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration = {'name': 'protocol_003_corrected_live_trace', 'training_seeds': SEEDS[:5],
        'validation_seeds': SEEDS[5:], 'reserved_test_seeds': [201, 202, 203, 204, 205],
        'change': 'Fix stale container references, sparse slot indexing, and trace identity after migration; otherwise keep data/label policy unchanged.',
        'environment': {'RAM_SCALE': '3.5', 'DISK_SCALE': '1.0', 'DISK_CAP_SCALE': '1.0', 'TRACE_BINDING_AUDIT': '1'},
        'code_sha256': {p: sha(ROOT / p) for p in ['dump_replay.py', 'simulator/workload/OfflineTraceWorkloadV2.py']}}
    registry = ART / 'protocol_003_corrected_replays.json'
    if registry.exists():
        assert json.loads(registry.read_text()) == registration
    else:
        registry.write_text(json.dumps(registration, indent=2))
    for seed in SEEDS:
        destination = BASE / f'seed{seed}'
        if (destination / 'replay_log.npz').exists():
            continue
        guard()
        destination.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, **registration['environment'], QOS_OVERLOAD_OUT=str(destination))
        environment.pop('FIXED_SCHEDULE_PATH', None)
        print(f'REBUILD seed={seed}', flush=True)
        with (destination / 'generation.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', 'dump_replay.py', str(seed)], cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Replay {seed} failed; inspect {destination / "generation.log"}')
    times, demands, schedules, overloads = [], [], [], []
    for seed in SEEDS:
        source = BASE / f'seed{seed}'
        times.append(np.load(source / 'time_series.npy').astype('float32'))
        schedules.append(np.load(source / 'schedule_series.npy').astype('float32'))
        overloads.append(np.load(source / 'labels_overload.npy'))
        with np.load(source / 'replay_log.npz') as replay:
            demands.append(replay['container_demands'].reshape(202, 112).astype('float32'))
    threshold = np.percentile(np.concatenate(times[:5]), 98, axis=0).reshape(16, 7)
    labels = [dilate_one_block(overload, profile_classes(features, overload, threshold))[1]
              for features, overload in zip(times, overloads)]
    OUTPUT.mkdir(parents=True)
    arrays = {'time_series': np.stack(times), 'container_demand_series': np.stack(demands),
              'schedule_series': np.stack(schedules), 'labels': np.stack(labels)}
    for name, values in arrays.items():
        np.save(OUTPUT / f'{name}.npy', values)
    manifest = {'seeds': SEEDS, 'train_blocks': list(range(5)), 'validation_blocks': [5, 6, 7], 'block_size': 202,
                'registration': registration, 'array_sha256': {name: sha(OUTPUT / f'{name}.npy') for name in arrays},
                'source_sha256': {str(BASE / f'seed{s}/replay_log.npz'): sha(BASE / f'seed{s}/replay_log.npz') for s in SEEDS},
                'class_counts': {str(s): np.bincount(y.ravel(), minlength=4).tolist() for s,y in zip(SEEDS,labels)}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
