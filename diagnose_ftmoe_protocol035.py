"""Protocol-035 fixed, no-training capability-coverage diagnosis over Protocol-034 caches."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from protocol035_common import dump_json, load_npz, binary_metrics, phase_bounds, WINDOWS, W_BLOCKS


def metric_mask(prob,labels,mask):
    p=np.asarray(prob); y=np.asarray(labels); m=np.asarray(mask,dtype=bool)
    return binary_metrics(p[m],y[m])

def support_flag(m): return bool(m['positives']>=8 and m['negatives']>=8)

def online_summary(prob,labels,bounds):
    wins=[]
    for name,a,b in WINDOWS:
        wins.append({'window':name,'whole':binary_metrics(prob[a:b],labels[a:b]),
                     'prefix32':binary_metrics(prob[a:a+32],labels[a:a+32]),'prefix64':binary_metrics(prob[a:a+64],labels[a:a+64])})
    late=[x['whole']['ap'] for x in wins if x['window'] in {'U_rec2','V_rec2','U_rec3','V_rec3'}]
    w=[]
    for name in W_BLOCKS:
        a,b=bounds[name]; w.append({'phase':name,**binary_metrics(prob[a:b],labels[a:b])})
    return {'full':binary_metrics(prob,labels),'recurrences':wins,'six_window_equal_weight_AP':float(np.mean([x['whole']['ap'] for x in wins])),
            'late4_equal_weight_AP':float(np.mean(late)),'W_blocks':w,'W_equal_weight_AP':float(np.mean([x['ap'] for x in w]))}

def eligible_snapshots(meta,caches,start):
    before=np.arange(0,max(0,int(start)-2),dtype=np.int64)
    rows=[]
    for m in meta:
        d=caches[m['snapshot_id']]
        valid=np.isfinite(d['full_probability'][before]).all(axis=1)&np.isfinite(d['fragment_probability'][before]).all(axis=1)
        count=int(valid.sum())
        if int(m['creation_prediction_index'])<int(start) and count>=32:
            rows.append((m,d,count))
    return rows

def snapshot_cell_rows(eligible,labels,selector):
    rows=[]
    for m,d,support in eligible:
        fp=d['full_probability']; gp=d['fragment_probability']
        fm=selector(fp,labels); gm=selector(gp,labels)
        rows.append({'snapshot_id':m['snapshot_id'],'lineage_id':m['lineage_id'],'source_phase':m.get('source_phase_id_audit_only'),
                     'creation_prediction_index':int(m['creation_prediction_index']),'prewindow_matured_support':support,
                     'full':fm,'fragment':gm,'full_minus_fragment_AP':None if fm['ap'] is None or gm['ap'] is None else float(fm['ap']-gm['ap'])})
    return rows

def best(rows,key):
    valid=[r for r in rows if r[key]['ap'] is not None]
    if not valid: return None
    return max(valid,key=lambda r:(r[key]['ap'],-int(r['lineage_id']),-int(r['snapshot_id'].split('_')[-1])))

def cell(name,cmet,rows):
    bf=best(rows,'full'); bg=best(rows,'fragment'); sufficient=support_flag(cmet)
    return {'cell':name,'support_sufficient':sufficient,'support_rule':'positives>=8 and negatives>=8','C_ref':cmet,
            'snapshot_metrics':rows,'best_full':None if bf is None else {'snapshot_id':bf['snapshot_id'],'metric':bf['full'],'AP_delta_vs_C':None if bf['full']['ap'] is None or cmet['ap'] is None else float(bf['full']['ap']-cmet['ap'])},
            'best_fragment':None if bg is None else {'snapshot_id':bg['snapshot_id'],'metric':bg['fragment'],'AP_delta_vs_C':None if bg['fragment']['ap'] is None or cmet['ap'] is None else float(bg['fragment']['ap']-cmet['ap'])},
            'hindsight_diagnostic':True,'online_result':False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol034-root',required=True); ap.add_argument('--stream-manifest',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); root=Path(a.protocol034_root); bounds=phase_bounds(a.stream_manifest)
    arms={'C_ref':'C_ref','D_frozen_ref':'D_frozen_ref','D_live':'D_live','D_route_fragment':'D_route_fragment','D_route_full':'D_route_full'}
    pred={k:load_npz(root/v/'predictions.npz') for k,v in arms.items()}; labels=pred['C_ref']['labels']
    if not all(np.array_equal(labels,x['labels']) for x in pred.values()): raise AssertionError('historical arm labels differ')
    online={k:online_summary(v['probability'],labels,bounds) for k,v in pred.items()}
    meta=json.loads((root/'D_live'/'snapshot_metadata.json').read_text())
    cache_root=root/'D_live'/'snapshot_cache'; caches={}
    for m in meta:
        path=cache_root/(m['snapshot_id']+'.npz')
        with np.load(path,allow_pickle=False) as d: caches[m['snapshot_id']]={k:d[k].copy() for k in d.files}
    pair_table=[]; host_cells=[]; block_cells=[]; risk_cells=[]
    cprob=pred['C_ref']['probability']
    for wname,s,e in WINDOWS:
        eligible=eligible_snapshots(meta,caches,s)
        for m,d,support in eligible:
            fm=binary_metrics(d['full_probability'][s:e],labels[s:e]); gm=binary_metrics(d['fragment_probability'][s:e],labels[s:e])
            pair_table.append({'window':wname,'snapshot_id':m['snapshot_id'],'lineage_id':m['lineage_id'],'source_phase':m.get('source_phase_id_audit_only'),
                               'creation_prediction_index':int(m['creation_prediction_index']),'prewindow_matured_support':support,'full':fm,'fragment':gm,
                               'full_minus_fragment_AP':None if fm['ap'] is None or gm['ap'] is None else float(fm['ap']-gm['ap'])})
        for h in range(16):
            cmet=binary_metrics(cprob[s:e,h],labels[s:e,h])
            rows=snapshot_cell_rows(eligible,labels,lambda p,y,h=h,s=s,e=e:binary_metrics(p[s:e,h],y[s:e,h]))
            host_cells.append(cell('%s/host_%02d'%(wname,h),cmet,rows))
        for q in range(4):
            a0=s+32*q; b0=a0+32; cmet=binary_metrics(cprob[a0:b0],labels[a0:b0])
            rows=snapshot_cell_rows(eligible,labels,lambda p,y,a0=a0,b0=b0:binary_metrics(p[a0:b0],y[a0:b0]))
            block_cells.append(cell('%s/block_%d'%(wname,q+1),cmet,rows))
        bins=((0.0,0.1,False),(0.1,0.5,False),(0.5,1.0,True))
        for lo,hi,last in bins:
            mask=(cprob[s:e]>=lo)&((cprob[s:e]<=hi) if last else (cprob[s:e]<hi))
            cmet=metric_mask(cprob[s:e],labels[s:e],mask)
            rows=snapshot_cell_rows(eligible,labels,lambda p,y,s=s,e=e,mask=mask:metric_mask(p[s:e],y[s:e],mask))
            risk_cells.append(cell('%s/C_risk_[%g,%g%s'%(wname,lo,hi,']' if last else ')'),cmet,rows))
    initial=[]
    for m in meta:
        phase=m.get('source_phase_id_audit_only')
        if phase not in ('U_first','V_first'): continue
        d=caches[m['snapshot_id']]; start=int(m['creation_prediction_index'])+1; pend=bounds[phase][1]; end=min(start+128,pend)
        row={'snapshot_id':m['snapshot_id'],'lineage_id':m['lineage_id'],'source_phase':phase,'creation_prediction_index':int(m['creation_prediction_index']),
             'initial_followup_intervals':[start,end],'initial_followup_actual_length':max(0,end-start)}
        if end>start:
            row['initial_followup']={'C_ref':binary_metrics(cprob[start:end],labels[start:end]),'full':binary_metrics(d['full_probability'][start:end],labels[start:end]),
                                     'fragment':binary_metrics(d['fragment_probability'][start:end],labels[start:end])}
        row['recurrences']=[]
        for wn,s,e in WINDOWS:
            if not wn.startswith(phase[0]+'_'): continue
            row['recurrences'].append({'window':wn,'C_ref':binary_metrics(cprob[s:e],labels[s:e]),'full':binary_metrics(d['full_probability'][s:e],labels[s:e]),
                                       'fragment':binary_metrics(d['fragment_probability'][s:e],labels[s:e])})
        initial.append(row)
    local_positive=0; sufficient_cells=0
    for group in (host_cells,block_cells,risk_cells):
        for c in group:
            if c['support_sufficient']:
                sufficient_cells+=1; vals=[]
                for key in ('best_full','best_fragment'):
                    if c[key] and c[key]['AP_delta_vs_C'] is not None: vals.append(c[key]['AP_delta_vs_C'])
                if vals and max(vals)>0: local_positive+=1
    full_online=online['D_route_full']['six_window_equal_weight_AP']-online['C_ref']['six_window_equal_weight_AP']
    out={'protocol':'035','diagnostic_suite_count':1,'parameter_fitting':False,'uses_only_existing_034_predictions_and_caches':True,
         'snapshot_count':len(meta),'online_recomputed_tie_grouped_AP':online,'full_fragment_window_pair_table':pair_table,
         'initial_UV_snapshot_followup':initial,'predefined_cells':{'host_x_recurrence':host_cells,'four_32step_blocks':block_cells,'C_probability_risk_bins':risk_cells},
         'summary':{'sufficient_support_cells':sufficient_cells,'sufficient_cells_with_hindsight_local_positive_AP':local_positive,
                    'existing_online_full_route_six_window_AP_delta_vs_C':float(full_online),
                    'local_capability_evidence_present':bool(local_positive>0),'local_capability_already_converted_to_online_recurrence_gain':bool(full_online>0),
                    'interpretation_note':'Local hindsight capability and late insufficiency may coexist; this diagnostic does not itself infer causality, and cells are not independent seeds.'},
         'hindsight_diagnostic':True,'online_result':False,'statistical_confirmation':False}
    dump_json(a.output,out); print(json.dumps({'snapshot_count':len(meta),'sufficient_cells':sufficient_cells,'local_positive_cells':local_positive,'online_route_delta':full_online},indent=2))
if __name__=='__main__': main()
