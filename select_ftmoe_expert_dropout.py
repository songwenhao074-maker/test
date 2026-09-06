"""Select a common expert dropout probability after protocol 010 pilots finish."""
import json
import numpy as np
from train_ftmoe_end_to_end import ART, aggregate, sha, write_json


def main():
    protocol_path = ART/'protocol_010_expert_dropout.json'
    protocol = json.loads(protocol_path.read_text())
    assert json.loads((ART/'selection_009_context_pilot.json').read_text())['selected_mode']=='none'
    candidates = {}; provenance = {}
    runs = {0.: 'physical_lr003_e30', .1: 'expert_dropout01_lr003_e30', .3: 'expert_dropout03_lr003_e30'}
    for probability,run in runs.items():
        rows = []
        for variant in ['v0','v1','v2','v3','v4']:
            source = ART/'runs'/(runs[0.] if variant=='v0' else run)/f'{variant}_seed1'
            summary = json.loads((source/'summary.json').read_text())
            config = summary['configuration']
            assert summary['variant']==variant and summary['seed']==1 and summary['all_parameters_trainable']
            assert config['epochs']==30 and config['stop_after'] is None and config['learning_rate']==.003
            assert config['data']==protocol['data'] and config.get('moe_context','none')=='none'
            assert variant=='v0' or config.get('expert_dropout',0.)==probability
            for filename in ['summary.json','best.pt','last.pt']:
                path = source/filename; provenance[str(path)] = sha(path)
            rows.append(summary)
        candidates[probability] = {'run':run,'aggregate':aggregate(rows),
             'mean_B_score':float(np.mean([r['B_best']['mean']['score'] for r in rows])),
             'parameters':{r['variant']:r['parameters'] for r in rows}}
    assert candidates[0.]['parameters']==candidates[.1]['parameters']==candidates[.3]['parameters']
    maximum = max(c['mean_B_score'] for c in candidates.values())
    selected = min(p for p,c in candidates.items() if c['mean_B_score']>=maximum-.002)
    result = {'registration_sha256':sha(protocol_path),'selected_probability':selected,'candidates':candidates,
              'source_sha256':provenance,'rule':protocol['selection'],'independent_confirmation':False,
              'reserved_tests_used':False,'followup_required':selected>0}
    write_json(ART/'selection_010_expert_dropout.json',result)
    print(json.dumps({'scores':{p:c['mean_B_score'] for p,c in candidates.items()},'selected_probability':selected,
          'B_means':{p:{v:{m:d['mean'] for m,d in values.items()} for v,values in c['aggregate']['B_best']['variants'].items()}
                     for p,c in candidates.items()}},indent=2))


if __name__=='__main__': main()
