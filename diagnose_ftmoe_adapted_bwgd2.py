"""Audit saved protocol016 data and, when present, all four saved prediction runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016'


def score(pred,y):
    return float(f1_score((y>0).ravel(),np.asarray(pred).ravel(),zero_division=0)) if (y>0).any() else None


def main():
    arrays=dict(np.load(OUT/'streams/seed301_steps300/stream.npz'))
    raw=arrays['raw_labels'];y=raw[:300].copy()
    for t in range(300):
        for delta in (-1,1):
            if 0<=t+delta<301:
                use=(y[t]==0)&(raw[t+delta]>0);y[t,use]=raw[t+delta,use]
    rawcounts=np.bincount(raw[:300].ravel(),minlength=4)
    counts=np.bincount(y.ravel(),minlength=4)
    positive=float((y>0).mean())
    data=ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical'
    train_h=np.load(data/'time_series.npy')[:5].reshape(5,202,16,7)
    scale=np.maximum(train_h.max(axis=(0,1)),1e-8)
    g=arrays['demands'][:300];host=arrays['host_features'][:300]
    before=arrays['before_placement'];after=arrays['after_placement']
    reconstructed=np.zeros_like(arrays['host_features'])
    for t in range(301):
        for c in range(16):
            if before[t,c]>=0:reconstructed[t,before[t,c]]+=arrays['demands'][t,c]
    ratio=arrays['post_totals']/arrays['capacities']
    calc=np.where((ratio>1).any(-1),ratio.argmax(-1)+1,0)
    proposed=np.einsum('tch,tcf->thf',arrays['schedules'][:300],g)[...,[0,1,4]]
    ids=arrays['creation_ids'][:300];dest=arrays['schedules'][:300].argmax(-1)
    changed=dest[1:]!=dest[:-1];same=(ids[1:]==ids[:-1])&(ids[1:]>=0)
    fractions=(counts[1:]/max(1,counts[1:].sum())).tolist()
    passed=bool(.05<=positive<=.25 and fractions[0]>.5 and (rawcounts[1:]>0).all() and counts[0]>=50 and counts[1:].sum()>=50)
    report={'distribution_passed':passed,'main_class_counts':counts.tolist(),'raw_class_counts':rawcounts.tolist(),
        'main_positive_fraction':positive,'main_positive_class_fraction':fractions,
        'demand_mean_cpu_ram_disk':g[...,[0,1,4]].mean(axis=(0,1)).tolist(),
        'demand_p95_cpu_ram_disk':np.quantile(g[...,[0,1,4]],.95,axis=(0,1)).tolist(),
        'host_normalized_max':(host/scale).max(axis=(0,1)).tolist(),
        'host_normalized_abs_above_1e6_fraction':float((np.abs(host/scale)>1e6).mean()),
        'host13_normalized_max':(host[:,13]/scale[13]).max(0).tolist(),
        'max_post_resource_ratios':ratio[:300].max(axis=(0,1)).tolist(),
        'raw_label_mismatches':int((raw!=calc).sum()),
        'host_reconstruction_max_abs_error':float(np.abs(reconstructed-arrays['host_features']).max()),
        'identity_mismatches':int(((arrays['creation_ids']!=arrays['after_creation_ids'])&(after>=0)).sum()),
        'proposal_changed_slots':int(changed.sum()),'changes_with_new_or_missing_identity':int((changed&~same).sum()),
        'causal_pre_host_rule_f1':{'raw':score((host[...,[0,1,4]]/arrays['capacities']>1).any(-1),raw[:300]),
                                 'tolerance':score((host[...,[0,1,4]]/arrays['capacities']>1).any(-1),y)},
        'causal_proposed_rule_f1':{'raw':score((proposed/arrays['capacities']>1).any(-1),raw[:300]),
                                 'tolerance':score((proposed/arrays['capacities']>1).any(-1),y)},
        'runs':{}}
    for method in 'ABCD':
        folder=OUT/'runs'
        matches=list(folder.glob(method+'_*/predictions.npz'))
        if not matches:continue
        p=dict(np.load(matches[0]));summary=json.loads((matches[0].parent/'summary.json').read_text())
        assert np.array_equal(p['labels'],y)
        slices={}
        for cls in (1,2,3):
            mask=y==cls
            slices[str(cls)]={'support':int(mask.sum()),'detection_recall':float((p['probability'][mask]>=.5).mean()) if mask.any() else None}
        report['runs'][method]={'summary':summary['metrics'],'resource_detection_recall':slices,
            'false_positives_by_host':((p['probability']>=.5)&(y==0)).sum(0).tolist(),
            'initial_reference_and_final':json.loads((matches[0].parent/'reference.json').read_text()),
            'final_experts':summary['final_experts'],'elapsed_seconds':summary['elapsed_seconds']}
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [OUT/'protocol.json',OUT/'disk_law.json',OUT/'streams/seed301_steps300/stream.npz']}
    target=OUT/('diagnostic.json' if report['runs'] else 'data_check.json')
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','source_sha256')},indent=2))
    if report['runs']:print(json.dumps({m:{k:v['summary'][k]['f1'] for k in ('full','first_half','second_half')} for m,v in report['runs'].items()}))


if __name__=='__main__':main()
