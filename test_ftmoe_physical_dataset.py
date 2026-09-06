"""Verify conservation, physical labels, and normalized capacities without model training."""
import json
from pathlib import Path
import numpy as np
import torch
from train_ftmoe_end_to_end import ART, load_data, create_model
from build_ftmoe_replay_holdout_dataset import dilate_one_block


def main():
    directory = ART / 'data/protocol_004_physical'
    t = np.load(directory / 'time_series.npy').reshape(8, 202, 16, 7)
    g = np.load(directory / 'container_demand_series.npy').reshape(8, 202, 16, 7)
    np.testing.assert_allclose(t.sum(2), g.sum(2), rtol=1e-6, atol=.02)
    y = np.load(directory / 'labels.npy')
    _, _, normalization, manifest = load_data(directory)
    assert set(manifest['seeds']).isdisjoint({37, 41, 47, 201, 202, 203, 204, 205})
    for b, seed in enumerate(manifest['seeds']):
        source = ART / 'physical_replays' / f'seed{seed}'
        with np.load(source / 'replay_log.npz') as r:
            ratio = np.stack([r['total_base_ips']/r['host_cpu_capacity'],
                              r['total_ram']/r['host_ram_capacity'], r['total_disk']/r['host_disk_capacity']], -1)
        overload = (ratio > 1).any(-1).astype(int)
        classes = np.where(overload, ratio.argmax(-1) + 1, 0)
        np.testing.assert_array_equal(classes, np.load(source / 'labels_overload_class.npy'))
        np.testing.assert_array_equal(y[b], dilate_one_block(overload, classes)[1])
    model = create_model('v3', 1, normalization)
    restored = model.graph_encoder.host_capacity.numpy() * np.asarray(normalization['graph_scale'])[[0, 1, 4]]
    np.testing.assert_allclose(restored, manifest['host_capacities'], rtol=1e-6)
    print(json.dumps({'preaction_resource_conservation': True, 'physical_label_definition': True,
                      'within_block_tolerance': True, 'capacity_units': True, 'test_seeds_excluded': True}))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
