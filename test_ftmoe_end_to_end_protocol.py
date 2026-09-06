"""Protocol regression checks that do not train or read any test replay."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_ftmoe_end_to_end import load_data, metric_arrays
from train_ftmoe_ablation_existing import metrics as legacy_metrics


class FixedPredictions(torch.nn.Module):
    def forward(self, x, schedule, graph):
        return {'detection_logits': x, 'class_logits': graph}


def main():
    rng = np.random.default_rng(981)
    logits = torch.from_numpy(rng.normal(size=(15, 16, 2)).astype('float32'))
    class_logits = torch.from_numpy(rng.normal(size=(15, 16, 3)).astype('float32'))
    labels = torch.from_numpy(rng.choice(4, size=(15, 16), p=[.6, .2, .1, .1]))
    loader = DataLoader(TensorDataset(logits, class_logits, logits, labels), batch_size=7)
    legacy = legacy_metrics(FixedPredictions(), loader)
    current = metric_arrays(logits.softmax(-1)[..., 1].numpy().ravel(),
                            class_logits.softmax(-1).numpy().reshape(-1, 3), labels.numpy().ravel())
    for key in ('f1', 'precision', 'recall', 'hr_at_100', 'ndcg_at_100', 'tp', 'fp', 'fn', 'tn'):
        assert abs(legacy[key] - current[key]) < 1e-7, (key, legacy[key], current[key])
    directory = Path('artifacts/ftmoe_end_to_end/data/protocol_001')
    train, validation, normalization, manifest = load_data(directory)
    raw = np.load(directory / 'time_series.npy')
    np.testing.assert_allclose(normalization['time_scale'], np.maximum(raw[:5].reshape(-1, 112).max(0), 1e-8))
    assert set(validation) == {'31', '101', '102'}
    assert not set(manifest['seeds']) & {37, 41, 47, 201, 202, 203, 204, 205}
    x = train[0]
    for block in range(5):
        start = block * 202
        assert torch.equal(x[start, :, 0], x[start, :, -1])
        assert torch.equal(x[start + 11, :, 0], x[start, :, -1])
        assert torch.equal(x[start + 12, :, 0], x[start + 1, :, -1])
    perfect_y = np.array([0, 1, 2, 3])
    perfect_p = np.array([.01, .99, .99, .99])
    perfect_c = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    perfect = metric_arrays(perfect_p, perfect_c, perfect_y)
    assert perfect['f1'] == perfect['hr_at_100'] == perfect['ndcg_at_100'] == 1
    assert abs(perfect['diagnostic_pair_ndcg_at_100'] - 1) < 1e-12
    assert perfect['diagnosis_hr_at_100pct'] == perfect['diagnosis_ndcg_at_100pct'] == 1
    # Diagnosis includes every truly anomalous host-step, including false negatives.
    # Global top-100 retrieval, detection confidence and normal-host classes cannot alter it.
    partial = metric_arrays(np.array([.99, .1, .9, .2]),
                            np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]), perfect_y)
    assert abs(partial['diagnosis_hr_at_100pct'] - 2 / 3) < 1e-12
    assert partial['diagnosis_hr_at_100pct'] == partial['diagnosis_ndcg_at_100pct']
    from sklearn.metrics import ndcg_score
    reference = np.mean([ndcg_score(np.eye(3)[[label - 1]], scores.reshape(1, -1), k=1)
                         for label, scores in zip(labels.numpy().ravel(), class_logits.softmax(-1).numpy().reshape(-1, 3)) if label])
    assert abs(reference - current['diagnosis_ndcg_at_100pct']) < 1e-12
    print(json.dumps({'legacy_metrics_match': True, 'window_boundaries_valid': True,
                      'train_only_normalization': True, 'test_seeds_excluded': True, 'perfect_prediction_check': True}))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
