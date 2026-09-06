"""Build causal physical-host data using a model-free calibrated simulator configuration."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import psutil
from prepare_ftmoe_end_to_end import ART, ROOT, guard, sha
from build_ftmoe_replay_holdout_dataset import dilate_one_block


def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    registration = json.loads((ART / 'protocol_004_physical.json').read_text())
    calibration = ART / 'physical_calibration/candidate_001_seed42'
    cal = json.loads((calibration / 'summary.json').read_text())
    assert .05 <= cal['positive_fraction'] <= .35
    assert min(cal['positive_class_fractions']) >= .05
    seeds = registration['training_seeds'] + registration['validation_seeds']
    base = ART / 'physical_replays'
    output = ART / 'data/protocol_004_physical'
    if output.exists():
        raise FileExistsError(output)
    for seed in seeds:
        destination = base / f'seed{seed}'
        if (destination / 'replay_log.npz').exists():
            continue
        guard()
        destination.mkdir(parents=True, exist_ok=True)
        if seed == 42:
            for path in calibration.iterdir():
                if path.suffix in ('.npy', '.npz', '.log'):
                    shutil.copy2(path, destination / path.name)
            continue
        environment = dict(os.environ, **cal['environment'], QOS_OVERLOAD_OUT=str(destination))
        environment.pop('FIXED_SCHEDULE_PATH', None)
        print(f'PHYSICAL_REPLAY seed={seed}', flush=True)
        with (destination / 'generation.log').open('w') as log:
            subprocess.run([sys.executable, '-u', 'dump_replay.py', str(seed)], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    time_blocks, demand_blocks, schedule_blocks, label_blocks = [], [], [], []
    capacities = None
    for seed in seeds:
        source = base / f'seed{seed}'
        time_blocks.append(np.load(source / 'time_series.npy').astype('float32'))
        schedule_blocks.append(np.load(source / 'schedule_series.npy').astype('float32'))
        overload = np.load(source / 'labels_overload.npy')
        classes = np.load(source / 'labels_overload_class.npy')
        label_blocks.append(dilate_one_block(overload, classes)[1])
        with np.load(source / 'replay_log.npz') as replay:
            demand_blocks.append(replay['container_demands'].reshape(202, 112).astype('float32'))
            cap = np.stack([replay['host_cpu_capacity'], replay['host_ram_capacity'], replay['host_disk_capacity']], axis=1)
            if capacities is not None:
                np.testing.assert_array_equal(cap, capacities)
            capacities = cap
            np.testing.assert_allclose(time_blocks[-1], replay['causal_host_features'].reshape(202, 112), rtol=1e-6)
    output.mkdir(parents=True)
    arrays = {'time_series': np.stack(time_blocks), 'container_demand_series': np.stack(demand_blocks),
              'schedule_series': np.stack(schedule_blocks), 'labels': np.stack(label_blocks)}
    for name, value in arrays.items():
        np.save(output / f'{name}.npy', value)
    manifest = {'seeds': seeds, 'train_blocks': list(range(5)), 'validation_blocks': [5, 6, 7], 'block_size': 202,
                'label_policy': 'actual physical overload and largest overload ratio; +/-1 within-block tolerance',
                'host_capacities': capacities.tolist(), 'registration': registration, 'environment': cal['environment'],
                'calibration': cal, 'array_sha256': {name: sha(output / f'{name}.npy') for name in arrays},
                'source_sha256': {str(base / f'seed{s}/replay_log.npz'): sha(base / f'seed{s}/replay_log.npz') for s in seeds},
                'class_counts': {str(s): np.bincount(y.ravel(), minlength=4).tolist() for s,y in zip(seeds,label_blocks)}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'class_counts': manifest['class_counts'], 'host_capacities': capacities.tolist()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
