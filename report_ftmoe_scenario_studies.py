"""Verify saved predictions and summarize independent capacity/Google scenario pilots."""
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent


def summarize(name):
    folder=ROOT/'artifacts/ftmoe_online'/name
    if not (folder/'result.json').exists():return None
    result=json.loads((folder/'result.json').read_text(encoding='utf8'))
    rows={};first=None;labels=None
    for method in 'ABCD':
        run=folder/'runs'/method
        s=json.loads((run/'summary.json').read_text());p=dict(np.load(run/'predictions.npz'))
        if first is None:first=p['probability'][:10].copy();labels=p['labels'].copy()
        else:
            np.testing.assert_array_equal(first,p['probability'][:10])
            np.testing.assert_array_equal(labels,p['labels'])
        n=len(p['probability'])
        verified={}
        for part,cut in [('full',slice(None)),('first_half',slice(0,n//2)),('second_half',slice(n//2,None))]:
            y=p['labels'][cut]>0;pred=p['probability'][cut]>=.5
            actual=float(f1_score(y.ravel(),pred.ravel(),zero_division=0)) if y.any() else None
            assert actual==s['metrics'][part]['f1'],(method,part,actual)
            verified[part]=actual
        ref=json.loads((run/'reference.json').read_text())
        rows[method]={'f1':verified,'raw_second_half_f1':s['metrics']['raw']['second_half']['f1'],
                      'precision':s['metrics']['second_half']['precision'],'recall':s['metrics']['second_half']['recall'],
                      'updates':s['updates'],'final_experts':s['final_experts'],
                      'reference_initial_f1':ref[0]['mean']['f1'],'reference_final_f1':ref[-1]['mean']['f1'],
                      'elapsed_seconds':s['elapsed_seconds'],'rss_gib':s['rss_gib']}
    stream=Path(s['configuration']['stream'])
    a=np.load(stream/'stream.npz');cap=a['capacities'];raw=a['raw_labels']
    ratios=a['post_totals']/cap
    np.testing.assert_array_equal(np.where((ratios>1).any(-1),ratios.argmax(-1)+1,0),raw)
    d=rows['D']['f1']['second_half'];b=rows['B']['f1']['second_half'];c=rows['C']['f1']['second_half']
    report={'scenario':name,'distribution':result['distribution'],'methods':rows,
            'initial_first_10_probabilities_identical':True,'saved_prediction_f1_verified':True,'raw_physical_labels_verified':True,
            'D_minus_B':d-b if d is not None and b is not None else None,
            'D_minus_C':d-c if d is not None and c is not None else None,
            'pilot_performance_target_met':bool(d is not None and b is not None and d>=.65 and d>b),
            'statistical_limitation':'one paired run; no meaningful population CI or five-seed confirmation'}
    (folder/'verified_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


def plot(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(reports),2,figsize=(12,4*len(reports)),squeeze=False)
    colors={'A':'#777777','B':'#d55e00','C':'#0072b2','D':'#009e73'}
    for pair,report in zip(axes,reports):
        folder=ROOT/'artifacts/ftmoe_online'/report['scenario']
        for method in 'ABCD':
            s=json.loads((folder/'runs'/method/'summary.json').read_text())
            rolling=s['metrics']['rolling'];pair[0].plot([r['end'] for r in rolling],[r['f1'] if r['f1'] is not None else np.nan for r in rolling],label=method,color=colors[method])
            ref=json.loads((folder/'runs'/method/'reference.json').read_text())
            pair[1].plot([r['step'] for r in ref],[r['mean']['f1'] for r in ref],label=method,color=colors[method])
        for axis in pair:
            axis.set_ylim(0,1);axis.set_xlabel('Simulation interval');axis.set_ylabel('F1');axis.grid(alpha=.2);axis.legend()
        pair[0].axhline(.65,color='black',linestyle='--',linewidth=1)
        pair[0].set_title(report['scenario']+' / rolling 100-interval F1')
        pair[1].set_title('Fixed offline reference')
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(ROOT/f'artifacts/ftmoe_online/scenarios_017_018.{suffix}',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    reports=[r for name in ('capacity_017','google2011_018') if (r:=summarize(name)) is not None]
    if reports:plot(reports)
    print(json.dumps(reports,ensure_ascii=False,indent=2))
