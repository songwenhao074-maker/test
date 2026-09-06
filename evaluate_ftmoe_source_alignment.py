"""Complete protocol012 comparison, retaining rejected and prior fusion arms."""
import json
import numpy as np
from audit_ftmoe_history import read_row, SEEDS
from train_ftmoe_end_to_end import ART, METRICS, sha, write_json
from evaluate_ftmoe_branch_development import paired_difference


def main():
    registration = ART/'protocol_012_source_alignment.json'
    config = json.loads(registration.read_text())
    arms = {v: ('expert_dropout01_lr003_e60_chain', v) for v in ['v0','v1','v2','v3']}
    arms.update(original_attention=('expert_dropout01_lr003_e60_chain','v4'),
                original_gated=('gated_fusion_lr003_e60','v4'),
                source_attention=('source_attention_lr003_e60','v4'),
                source_gated=('source_gated_lr003_e60','v4'))
    for seed in SEEDS:
        parameters = [json.loads((ART/'runs'/run/f'v4_seed{seed}/summary.json').read_text())['parameters']
                      for run,variant in arms.values() if variant=='v4']
        assert len(set(parameters))==1
    output = {'registration_sha256':sha(registration),'source_sha256':{},
        'disclosure':'Development results after repeated exploratory selection. Conditional intervals are not independent significance. Both registered new arms and prior alternatives are retained; reserved tests are not read.'}
    for calibrated in [False,True]:
        rows = {}
        for arm,(run,variant) in arms.items():
            rows[arm] = {}
            for seed in SEEDS:
                row,training_config,hashes = read_row(run,variant,seed,calibrated)
                for key in ['data','epochs','learning_rate']:
                    assert training_config[key]==config[key]
                assert training_config.get('moe_context','none')=='none'
                assert variant=='v0' or training_config.get('expert_dropout',0)==.1
                expected = {'original_gated':'gated','source_attention':'source_attention','source_gated':'source_gated'}.get(arm,'attention')
                assert training_config.get('fusion','attention')==expected
                rows[arm][seed] = row; output['source_sha256'].update(hashes)
        mode = 'calibrated' if calibrated else 'raw'; output[mode] = {}
        for protocol in ['A_last','B_best']:
            means = {arm:{m:float(np.mean([r[protocol]['mean'][m] for r in values.values()])) for m in METRICS} for arm,values in rows.items()}
            comparisons = [('v1','v0'),('v2','v1'),('original_attention','v2'),('original_attention','original_gated'),
                ('source_attention','v2'),('source_attention','source_gated'),('source_attention','v3'),('source_attention','original_attention')]
            paired = {f'{hi}_minus_{lo}':{m:paired_difference({s:rows[hi][s][protocol] for s in SEEDS},
                {s:rows[lo][s][protocol] for s in SEEDS},m,SEEDS) for m in METRICS} for hi,lo in comparisons}
            output[mode][protocol] = {'means':means,'paired':paired}
    scores = output['raw']['B_best']['means']
    gain = float(np.mean(list(scores['source_attention'].values()))-np.mean(list(scores['original_attention'].values())))
    selected = gain>.002
    output['selection'] = {'raw_B_mean_S_gain':gain,'required_gain_strictly_greater_than':.002,
        'selected_full':'source_attention' if selected else 'original_attention',
        'alignment_refinement_adopted':selected}
    for mode in ['raw','calibrated']:
        for protocol in ['A_last','B_best']:
            data = output[mode][protocol]
            full,gate = ('source_attention','source_gated') if selected else ('original_attention','original_gated')
            pairs = [('v1','v0'),('v2','v1'),(full,'v2'),(full,gate)]
            passes = {f'{hi}_minus_{lo}': data['paired'][f'{hi}_minus_{lo}']['f1']['mean']>=.005 and
                all(data['paired'][f'{hi}_minus_{lo}'][m]['mean']>0 for m in METRICS) for hi,lo in pairs}
            target = data['means'][full]['f1']>=.86 and data['means'][full][METRICS[1]]>=.63 and data['means'][full][METRICS[2]]>=.58
            data['selected_primary_passes'] = passes
            data['selected_primary_passed'] = all(passes.values()) and target
    write_json(ART/'comparison_012_complete.json',output)
    print(json.dumps({'selection':output['selection'], **{mode:{p:output[mode][p]['means'] for p in ['A_last','B_best']} for mode in ['raw','calibrated']}},indent=2))


if __name__=='__main__': main()
