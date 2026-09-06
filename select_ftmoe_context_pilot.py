"""Apply protocol 009 after all parameter-matched development pilots finish."""
import json
import numpy as np
from train_ftmoe_end_to_end import ART, METRICS, aggregate, sha, write_json


def select_mode(scores):
    assert set(scores) == {'none', 'local', 'global'}
    maximum = max(scores.values())
    return next(mode for mode in ['none', 'local', 'global'] if scores[mode] >= maximum - .002)


def main():
    registration = ART / 'protocol_009_context_pilot.json'
    protocol = json.loads(registration.read_text())
    runs = {'none': 'physical_lr003_e30', 'local': 'context_local_lr003_e30',
            'global': 'context_global_lr003_e30'}
    control = ART / 'runs' / runs['none'] / 'v0_seed1'
    candidates = {}; provenance = {}
    for mode, run in runs.items():
        rows = []
        for variant in ['v0', 'v1', 'v2', 'v3', 'v4']:
            root = control if variant == 'v0' else ART / 'runs' / run / f'{variant}_seed1'
            path = root / 'summary.json'
            summary = json.loads(path.read_text())
            configuration = summary['configuration']
            assert summary['variant'] == variant and summary['seed'] == 1
            assert summary['all_parameters_trainable']
            assert configuration['data'] == protocol['data']
            assert configuration['epochs'] == 30 and configuration['stop_after'] is None
            assert configuration['learning_rate'] == .003
            assert variant == 'v0' or configuration.get('moe_context', 'none') == mode
            for file in [path, root / 'last.pt', root / 'best.pt']:
                provenance[str(file)] = sha(file)
            rows.append(summary)
        candidates[mode] = {'run': run, 'v0_control': str(control), 'aggregate': aggregate(rows),
                            'mean_B_score': float(np.mean([s['B_best']['mean']['score'] for s in rows])),
                            'parameters': {s['variant']: s['parameters'] for s in rows}}
    for variant in ['v1', 'v2', 'v3', 'v4']:
        assert candidates['local']['parameters'][variant] == candidates['global']['parameters'][variant]
        assert candidates['local']['parameters'][variant] - candidates['none']['parameters'][variant] == 4096
    scores = {mode: row['mean_B_score'] for mode, row in candidates.items()}
    selected = select_mode(scores)
    result = {'registration_sha256': sha(registration), 'candidates': candidates,
              'selected_mode': selected, 'rule': protocol['selection'], 'source_sha256': provenance,
              'followup_required': selected != 'none', 'independent_confirmation': False,
              'reserved_tests_used': False}
    write_json(ART / 'selection_009_context_pilot.json', result)
    print(json.dumps({'scores': scores, 'selected_mode': selected,
                      'B_means': {mode: {v: {m: data[m]['mean'] for m in METRICS}
                         for v, data in c['aggregate']['B_best']['variants'].items()}
                         for mode, c in candidates.items()}}, indent=2))


if __name__ == '__main__':
    main()
