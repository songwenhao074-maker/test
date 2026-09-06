"""Protocol 011's development comparison graph; never opens reserved tests."""
import argparse
import json
import numpy as np
from train_ftmoe_end_to_end import ART, METRICS, sha, write_json


def assess(arms, comparisons):
    differences = {f'{higher}_minus_{lower}':{m:arms[higher][m]-arms[lower][m] for m in METRICS}
                   for higher,lower in comparisons}
    gain = all(d['f1']>=.005 and all(d[m]>0 for m in METRICS) for d in differences.values())
    full = arms['v4_full']
    target = full['f1']>=.86 and full[METRICS[1]]>=.63 and full[METRICS[2]]>=.58
    return {'differences':differences,'passed':bool(gain and target),'target_passed':bool(target)}


def paired_difference(higher,lower,metric,seeds,replicates=5000):
    replays = ['31','101','102']
    delta = np.array([[higher[s]['per_replay'][r][metric]-lower[s]['per_replay'][r][metric]
                      for r in replays] for s in seeds])
    rng = np.random.default_rng(819)
    samples = np.array([delta[rng.integers(len(seeds),size=len(seeds))][:,rng.integers(3,size=3)].mean()
                        for _ in range(replicates)])
    return {'mean':float(delta.mean()),'per_model_seed':dict(zip(seeds,delta.mean(1).tolist())),
            'ci95_conditional':np.quantile(samples,[.025,.975]).tolist()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base-run',required=True)
    parser.add_argument('--gated-run',required=True); parser.add_argument('--calibrated',action='store_true')
    parser.add_argument('--output',required=True); args = parser.parse_args()
    protocol_path = ART/'protocol_011_branch_ablations.json'; registration = json.loads(protocol_path.read_text())
    seeds = registration['model_seeds']; rows = {}; source_hashes = {}
    arms = {'v0':('v0',args.base_run),'v1':('v1',args.base_run),'v2':('v2',args.base_run),
            'v3_simple_fusion':('v3',args.base_run),'v4_full':('v4',args.base_run),'v4_gated':('v4',args.gated_run)}
    for arm,(variant,run) in arms.items():
        rows[arm] = {}
        for seed in seeds:
            directory = ART/'runs'/run/f'{variant}_seed{seed}'
            path = ART/'calibrated'/run/f'{variant}_seed{seed}.json' if args.calibrated else directory/'summary.json'
            summary = json.loads(path.read_text()); original = json.loads((directory/'summary.json').read_text())
            assert original['all_parameters_trainable']
            assert summary['configuration']==original['configuration']
            configuration = summary['configuration']
            assert configuration['data']==registration['data'] and configuration['epochs']==60
            assert configuration['stop_after'] is None and configuration['learning_rate']==.003
            assert variant=='v0' or configuration.get('expert_dropout',0.)==.1
            assert configuration.get('moe_context','none')=='none'
            assert configuration.get('fusion','attention')==('gated' if arm=='v4_gated' else 'attention')
            for filename in ['best.pt','last.pt']:
                source_hashes[str(directory/filename)] = sha(directory/filename)
            source_hashes[str(path)] = sha(path)
            if args.calibrated:
                for p,filename in [('A_last','last.pt'),('B_best','best.pt')]:
                    assert summary['calibration'][p]['checkpoint_sha256']==sha(directory/filename)
            rows[arm][seed] = summary
    for seed in seeds:
        a = json.loads((ART/'runs'/args.base_run/f'v4_seed{seed}/summary.json').read_text())
        b = json.loads((ART/'runs'/args.gated_run/f'v4_seed{seed}/summary.json').read_text())
        assert a['parameters']==b['parameters']
    report = {'registration_sha256':sha(protocol_path),'calibrated':args.calibrated,'source_sha256':source_hashes,
              'disclosure':'All checkpoints/hyperparameters were selected using these development replays. Bootstrap intervals describe conditional variability, not independent confirmatory significance. Reserved test replays remain unused.'}
    for p in ['A_last','B_best']:
        means = {arm:{m:float(np.mean([s[p]['mean'][m] for s in values.values()])) for m in METRICS}
                 for arm,values in rows.items()}
        report[p] = dict(assess(means,registration['primary_comparisons']),arms=means)
        report[p]['paired'] = {f'{higher}_minus_{lower}':{m:paired_difference(
             {s:rows[higher][s][p] for s in seeds},{s:rows[lower][s][p] for s in seeds},m,seeds) for m in METRICS}
             for higher,lower in registration['primary_comparisons']}
    report['primary_protocol'] = 'B_best' if report['B_best']['passed'] else 'A_last' if report['A_last']['passed'] else None
    write_json(ART/args.output,report)
    print(json.dumps({p:{k:v for k,v in report[p].items() if k!='paired'} for p in ['A_last','B_best']},indent=2))


if __name__=='__main__': main()
