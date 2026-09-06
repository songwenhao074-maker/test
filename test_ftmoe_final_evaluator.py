"""Verify final-evaluation assembly and paired statistics using development data only."""
import json
from pathlib import Path
import numpy as np
from evaluate_ftmoe_end_to_end import SEEDS, build_test_dataset, evaluate_physical_rule, paired_statistics
from train_ftmoe_end_to_end import ART, load_data


def main():
    development = ART / 'data/protocol_004_physical'
    fixture = ART / 'evaluator_development_fixture'
    if not fixture.exists():
        build_test_dataset(development, {31: ART / 'physical_replays/seed31'}, fixture)
    _, validation, actual, manifest = load_data(fixture)
    _, expected_validation, expected, _ = load_data(development)
    assert actual == expected
    assert set(validation) == {'31'}
    assert not set(manifest['seeds']) & {201,202,203,204,205}
    for a, b in zip(validation['31'], expected_validation['31']):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    rule = evaluate_physical_rule(fixture)
    recorded = json.loads((ART / 'physical_rule_diagnostic.json').read_text())['rows'][0]
    assert len(rule) == 1 and rule[0]['replay_seed'] == 31
    np.testing.assert_allclose(rule[0]['f1'], recorded['dilated_f1'], atol=1e-12)
    np.testing.assert_allclose(rule[0]['diagnosis_hr_at_100pct'], recorded['dominant_class_accuracy'], atol=1e-12)
    rows = [{'variant': f'v{i}', 'model_seed': seed, 'replay_seed': replay,
             'constant': .1 + i * .02} for i in range(5) for seed in SEEDS for replay in [31,101,102]]
    differences = paired_statistics(rows, 'constant', replicates=100)
    for result in differences.values():
        np.testing.assert_allclose([result['mean'], *result['ci95']], [.02,.02,.02], atol=1e-14)
    try:
        paired_statistics(rows[:-1], 'constant', replicates=1)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing evaluation cell was accepted')
    print(json.dumps({'training_normalization_preserved': True, 'physical_labels_preserved': True,
                      'paired_resampling_constant_difference': True, 'missing_cell_rejected': True,
                      'reserved_test_replays_untouched': True}))


if __name__ == '__main__':
    main()
