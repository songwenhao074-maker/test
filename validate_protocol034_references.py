"""Verify C_ref and D_frozen_ref reproduce the frozen Protocol-033 references."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REC=("U_rec1","V_rec1","U_rec2","V_rec2","U_rec3","V_rec3")

def J(p): return json.loads(Path(p).read_text(encoding='utf8'))
def jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x.strip()]

def ap(y,score):
    y=(np.asarray(y).reshape(-1)>0).astype(np.float64); s=np.asarray(score).reshape(-1)
    if y.sum()<=0 or y.sum()>=y.size: return None
    order=np.argsort(-s,kind='stable'); yy=y[order]; prec=np.cumsum(yy)/np.arange(1,yy.size+1)
    return float((prec*yy).sum()/yy.sum())

def phase_map(summary): return {x['phase']:x for x in summary['phases']}

def canonical(events):
    keys=('kind','cursor','matured_count','phase','candidate_id','expert_id','parent_id','old_id','new_id',
          'step','total','transition_kind','replacement_id','accepted','reason','reuse_event_id')
    return [tuple(e.get(k) for k in keys) for e in events]

def compare_arm(newdir,refdir,name,atol,rtol):
    with np.load(Path(newdir)/'predictions.npz') as n, np.load(Path(refdir)/'predictions.npz') as r:
        report={'arm':name,'exact_labels':bool(np.array_equal(n['labels'],r['labels'])),
                'exact_raw_labels':bool(np.array_equal(n['raw_labels'],r['raw_labels']))}
        if not report['exact_labels'] or not report['exact_raw_labels']:
            raise AssertionError(name+' label mismatch')
        for key in ('probability','class_probability','detection_logits','class_logits'):
            delta=float(np.max(np.abs(n[key].astype(np.float64)-r[key].astype(np.float64))))
            ok=bool(np.allclose(n[key],r[key],atol=atol,rtol=rtol,equal_nan=True))
            report[key+'_max_abs_delta']=delta; report[key+'_within_tolerance']=ok
            if not ok: raise AssertionError('%s %s reference mismatch max=%g'%(name,key,delta))
    ns=J(Path(newdir)/'summary.json'); rs=J(Path(refdir)/'summary.json')
    report['shared_initialization_exact']=J(Path(newdir)/'initialization.json')['shared_first4_expert_and_router_rows_sha256']==J(Path(refdir)/'initialization.json')['shared_first4_expert_and_router_rows_sha256']
    if not report['shared_initialization_exact']: raise AssertionError(name+' shared initialization mismatch')
    nph=phase_map(ns); rph=phase_map(rs); diffs={}
    for ph in REC:
        na=nph[ph]['detection']['ap']; ra=rph[ph]['detection']['ap']; d=abs(float(na)-float(ra))
        diffs[ph]=d
        if d>1e-4: raise AssertionError('%s %s AP reference mismatch %g'%(name,ph,d))
    report['recurrence_ap_abs_deltas']=diffs
    if name=='D_frozen_ref':
        nc=canonical(jsonl(Path(newdir)/'lifecycle.jsonl')); rc=canonical(jsonl(Path(refdir)/'lifecycle.jsonl'))
        report['exact_discrete_lifecycle']=nc==rc
        report['discrete_lifecycle_events_new']=len(nc); report['discrete_lifecycle_events_ref']=len(rc)
        if nc!=rc:
            mismatch=next((i for i,(x,y) in enumerate(zip(nc,rc)) if x!=y),min(len(nc),len(rc)))
            report['first_discrete_lifecycle_mismatch_index']=mismatch
            raise AssertionError('D_frozen_ref discrete lifecycle mismatch at %d'%mismatch)
    return report

def main():
    apg=argparse.ArgumentParser(); apg.add_argument('--run-root',required=True); apg.add_argument('--reference-root',required=True); apg.add_argument('--output',required=True)
    a=apg.parse_args(); run=Path(a.run_root); ref=Path(a.reference_root)
    out={'protocol':'034','reference_source_run_id':36417604442,'probability_atol':2e-6,'probability_rtol':2e-6,'recurrence_AP_atol':1e-4}
    out['C_ref']=compare_arm(run/'C_ref',ref/'C_fixed5','C_ref',2e-6,2e-6)
    out['D_frozen_ref']=compare_arm(run/'D_frozen_ref',ref/'D_guard_budget','D_frozen_ref',2e-6,2e-6)
    out['all_reference_checks_passed']=True
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
