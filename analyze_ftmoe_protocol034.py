"""Analyze the three preregistered Protocol-034 hypotheses."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

WINDOWS=(("U_rec1",3300,3428),("V_rec1",3808,3936),("U_rec2",4316,4444),
         ("V_rec2",4824,4952),("U_rec3",5332,5460),("V_rec3",5840,5968))
LATE={"U_rec2","V_rec2","U_rec3","V_rec3"}
W_BLOCKS=("W_long","W_gap1","W_gap2","W_gap3","W_gap4","W_gap5")

def J(p): return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')

def ap(y,score):
    y=(np.asarray(y).reshape(-1)>0).astype(np.float64); s=np.asarray(score).reshape(-1).astype(np.float64)
    if y.size==0 or y.sum()<=0 or y.sum()>=y.size: return None
    o=np.argsort(-s,kind='stable'); yy=y[o]; prec=np.cumsum(yy)/np.arange(1,yy.size+1)
    return float((prec*yy).sum()/yy.sum())

def metric(prob,labels):
    y=(np.asarray(labels).reshape(-1)>0); p=np.asarray(prob).reshape(-1); pred=p>=0.5
    pos=int(y.sum()); neg=int((~y).sum()); tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=pos-tp; tn=neg-fp
    return {'ap':ap(y,p),'rows':int(y.size),'positives':pos,'negatives':neg,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'recall':(tp/pos if pos else None),'fpr':(fp/neg if neg else None)}

def load_pred(path):
    with np.load(path) as d: return {k:d[k].copy() for k in d.files}

def phase_bounds(stream_manifest):
    m=J(stream_manifest); return {x['name']:(int(x['start']),int(x['end'])) for x in m['timeline']}

def pooled_fpr(prob,labels):
    blocks=[(prob[a:b],labels[a:b]) for _,a,b in WINDOWS]
    return metric(np.concatenate([x[0] for x in blocks],0),np.concatenate([x[1] for x in blocks],0))['fpr']

def dev_reference(rows_delta,pooled_fpr_delta,w_mean):
    mean=float(np.mean([r['delta_ap'] for r in rows_delta])); positive=sum(r['delta_ap']>0 for r in rows_delta)
    return {'mean_ap_delta':mean,'mean_ap_delta_ge_0p005':mean>=0.005,'prior_033_strong_0p03_reference':mean>=0.03,
            'positive_window_count':int(positive),'positive_window_count_ge4':positive>=4,
            'pooled_normal_fpr_delta':float(pooled_fpr_delta),'pooled_normal_fpr_delta_le_0p01':pooled_fpr_delta<=0.01,
            'W_equal_weight_mean_ap_delta':float(w_mean),'W_mean_ap_delta_ge_minus0p02':w_mean>=-0.02,
            'development_signal_all_pass':bool(mean>=0.005 and positive>=4 and pooled_fpr_delta<=0.01 and w_mean>=-0.02)}

def contrast(name,p_a,p_b,labels,bounds):
    rows=[]
    for ph,a,b in WINDOWS:
        ma=metric(p_a[a:b],labels[a:b]); mb=metric(p_b[a:b],labels[a:b])
        rows.append({'phase':ph,'A':ma,'B':mb,'delta_ap':float(ma['ap']-mb['ap']),
                     'delta_fpr':float(ma['fpr']-mb['fpr'])})
    late=float(np.mean([r['delta_ap'] for r in rows if r['phase'] in LATE]))
    fa=pooled_fpr(p_a,labels); fb=pooled_fpr(p_b,labels); w=[]
    for ph in W_BLOCKS:
        a,b=bounds[ph]; w.append(metric(p_a[a:b],labels[a:b])['ap']-metric(p_b[a:b],labels[a:b])['ap'])
    wmean=float(np.mean(w)); ref=dev_reference(rows,fa-fb,wmean)
    return {'contrast':name,'windows':rows,'equal_weight_mean_ap_delta':float(np.mean([r['delta_ap'] for r in rows])),
            'late_four_equal_weight_mean_ap_delta':late,'pooled_recurrence_normal_fpr_delta':float(fa-fb),
            'W_block_ap_deltas':dict(zip(W_BLOCKS,[float(x) for x in w])),'W_equal_weight_mean_ap_delta':wmean,
            'development_reference':ref}

def h2_analysis(live_dir,c_ref,labels):
    meta=J(Path(live_dir)/'snapshot_metadata.json'); cache_root=Path(live_dir)/'snapshot_cache'; caches={}
    for m in meta:
        with np.load(cache_root/(m['snapshot_id']+'.npz')) as d: caches[m['snapshot_id']]={k:d[k].copy() for k in d.files}
    windows=[]
    for ph,a,b in WINDOWS:
        support=np.arange(0,max(0,a-2),dtype=np.int64); eligible=[]
        for m in meta:
            d=caches[m['snapshot_id']]
            paired=np.isfinite(d['full_probability'][support]).all(axis=1)&np.isfinite(d['fragment_probability'][support]).all(axis=1)
            support_count=int(paired.sum())
            if m['creation_prediction_index']<a and support_count>=32:
                if not (np.isfinite(d['full_probability'][a:b]).all() and np.isfinite(d['fragment_probability'][a:b]).all()): continue
                fm=metric(d['full_probability'][a:b],labels[a:b]); gm=metric(d['fragment_probability'][a:b],labels[a:b])
                eligible.append({'snapshot_id':m['snapshot_id'],'lineage_id':m['lineage_id'],'trigger':m['trigger'],
                    'creation_prediction_index':m['creation_prediction_index'],'pre_window_matured_paired_support':support_count,
                    'full':fm,'fragment':gm,'full_minus_fragment_ap':float(fm['ap']-gm['ap'])})
        cmet=metric(c_ref[a:b],labels[a:b])
        if eligible:
            best_full=max(eligible,key=lambda x:(x['full']['ap'],-int(x['lineage_id']),-int(x['snapshot_id'].split('_')[-1])))
            best_frag=max(eligible,key=lambda x:(x['fragment']['ap'],-int(x['lineage_id']),-int(x['snapshot_id'].split('_')[-1])))
            block={'phase':ph,'eligible_snapshot_count':len(eligible),'paired_snapshots':eligible,
                   'mean_paired_full_minus_fragment_ap':float(np.mean([x['full_minus_fragment_ap'] for x in eligible])),
                   'best_full_snapshot_id':best_full['snapshot_id'],'best_full_ap':best_full['full']['ap'],
                   'best_fragment_snapshot_id':best_frag['snapshot_id'],'best_fragment_ap':best_frag['fragment']['ap'],
                   'C_ref_ap':cmet['ap'],'best_full_minus_C_ref_ap':float(best_full['full']['ap']-cmet['ap']),
                   'best_fragment_minus_C_ref_ap':float(best_frag['fragment']['ap']-cmet['ap']),
                   'best_full_minus_best_fragment_ap':float(best_full['full']['ap']-best_frag['fragment']['ap']),
                   'hindsight_diagnostic':True,'online_result':False}
        else:
            block={'phase':ph,'eligible_snapshot_count':0,'paired_snapshots':[],
                   'mean_paired_full_minus_fragment_ap':None,'reason':'no_snapshot_with_32_matured_pre_window_paired_predictions',
                   'hindsight_diagnostic':True,'online_result':False}
        windows.append(block)
    valid=[x['mean_paired_full_minus_fragment_ap'] for x in windows if x['mean_paired_full_minus_fragment_ap'] is not None]
    return {'protocol':'034','hypothesis':'H2_complete_prediction_memory','selection_uses_future_labels':True,
            'online_claim_allowed':False,'snapshot_count':len(meta),'windows':windows,
            'six_window_equal_weight_mean_paired_full_minus_fragment_ap':(float(np.mean(valid)) if len(valid)==6 else None),
            'all_six_windows_have_eligible_paired_snapshots':len(valid)==6}

def route_usage(route,meta,window):
    ph,a,b=window; sel=route['selected_snapshot_seq'][a:b]; total=sel.size; chosen=sel>=0
    ages=[]; pre=0; within=0; byseq={i+1:m for i,m in enumerate(meta)}
    for rel,h in np.argwhere(chosen):
        t=a+int(rel); m=byseq[int(sel[rel,h])]; ages.append(t-int(m['creation_prediction_index']))
        if int(m['creation_prediction_index'])<a: pre+=1
        else: within+=1
    return {'phase':ph,'selected_host_predictions':int(chosen.sum()),'fallback_host_predictions':int(total-chosen.sum()),
            'selection_fraction':float(chosen.mean()),'mean_selected_snapshot_age_predictions':(float(np.mean(ages)) if ages else None),
            'pre_window_snapshot_selections':int(pre),'within_window_snapshot_selections':int(within)}

def main():
    apg=argparse.ArgumentParser(); apg.add_argument('--run-root',required=True); apg.add_argument('--stream-manifest',required=True)
    apg.add_argument('--output-dir',required=True); apg.add_argument('--docs-output',required=True); apg.add_argument('--run-id',required=True)
    a=apg.parse_args(); root=Path(a.run_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    C=load_pred(root/'C_ref'/'predictions.npz'); F=load_pred(root/'D_frozen_ref'/'predictions.npz'); L=load_pred(root/'D_live'/'predictions.npz')
    RF=load_pred(root/'D_route_full'/'predictions.npz'); RG=load_pred(root/'D_route_fragment'/'predictions.npz'); labels=C['labels']
    for other in (F,L,RF,RG):
        if not np.array_equal(labels,other['labels']): raise AssertionError('Protocol034 labels differ across arms')
    bounds=phase_bounds(a.stream_manifest)
    h1={'protocol':'034','hypothesis':'H1_active_specialist_continuous_learning',
        'D_live_minus_D_frozen_ref':contrast('D_live-D_frozen_ref',L['probability'],F['probability'],labels,bounds),
        'D_live_minus_C_ref':contrast('D_live-C_ref',L['probability'],C['probability'],labels,bounds),'resource_equivalence_claim':False}
    h2=h2_analysis(root/'D_live',C['probability'],labels)
    h3={'protocol':'034','hypothesis':'H3_fixed_causal_routing','trained_parameters':False,'current_labels_allowed':False,
        'D_route_full_minus_C_ref':contrast('D_route_full-C_ref',RF['probability'],C['probability'],labels,bounds),
        'D_route_full_minus_D_live':contrast('D_route_full-D_live',RF['probability'],L['probability'],labels,bounds),
        'D_route_full_minus_D_route_fragment':contrast('D_route_full-D_route_fragment',RF['probability'],RG['probability'],labels,bounds),
        'D_route_fragment_minus_C_ref':contrast('D_route_fragment-C_ref',RG['probability'],C['probability'],labels,bounds)}
    meta=J(root/'D_live'/'snapshot_metadata.json'); h3['full_routing_usage']=[route_usage(RF,meta,w) for w in WINDOWS]
    h3['fragment_routing_usage']=[route_usage(RG,meta,w) for w in WINDOWS]
    h3['full_causal_cutoff_verified']=J(root/'D_route_full'/'routing_report.json')['causal_cutoff_verified']
    h3['fragment_causal_cutoff_verified']=J(root/'D_route_fragment'/'routing_report.json')['causal_cutoff_verified']
    dump(out/'hypothesis1.json',h1); dump(out/'hypothesis2.json',h2); dump(out/'hypothesis3.json',h3)
    refs=J(root/'reference_checks.json'); live=J(root/'D_live'/'summary.json')
    status={'protocol':'034','run_id':str(a.run_id),'completed':True,'full_training_replays_started':3,'full_training_replay_budget':3,
            'completed_training_arms':['C_ref','D_frozen_ref','D_live'],'prediction_cache_routing_passes':2,
            'completed_routing_arms':['D_route_fragment','D_route_full'],'new_simulator_streams':0,'extra_seeds':0,'threshold_sweeps':0,
            'reference_checks_passed':bool(refs['all_reference_checks_passed']),'snapshot_count':int(live['snapshot_count']),
            'H1_development_signal':h1['D_live_minus_D_frozen_ref']['development_reference']['development_signal_all_pass'],
            'H2_complete_memory_paired_delta':h2['six_window_equal_weight_mean_paired_full_minus_fragment_ap'],
            'H3_full_vs_C_development_signal':h3['D_route_full_minus_C_ref']['development_reference']['development_signal_all_pass'],
            'automatic_scientific_followups_started':[],'development_only':True,'statistical_confirmation':False,'stop_after_registered_budget':True}
    dump(out/'status.json',status); dump(out/'comparison.json',{'protocol':'034','run_id':str(a.run_id),'H1':h1,'H2':h2,'H3':h3,'status':status})
    md=['# Protocol-034 Results','', 'Run: `%s`. Development-only diagnostics selected after Protocol-033; not independent confirmation.'%a.run_id,'']
    x=h1['D_live_minus_D_frozen_ref']; md += ['## H1 — active specialist continuous learning','',
        '- Six-window mean AP delta D_live − D_frozen_ref: **%.6f**.'%x['equal_weight_mean_ap_delta'],
        '- Late-four mean AP delta: **%.6f**.'%x['late_four_equal_weight_mean_ap_delta'],
        '- Positive recurrence windows: **%d/6**.'%x['development_reference']['positive_window_count'],
        '- Registered development signal pass: **%s**.'%str(x['development_reference']['development_signal_all_pass']),'']
    md += ['## H2 — complete prediction memory','', '- Retained snapshot versions: **%d**.'%h2['snapshot_count'],
        '- Six-window equal-weight paired Full − Fragment AP delta: **%s**.'%(('%.6f'%h2['six_window_equal_weight_mean_paired_full_minus_fragment_ap']) if h2['six_window_equal_weight_mean_paired_full_minus_fragment_ap'] is not None else 'null'),
        '- Hindsight best-snapshot results are diagnostic only and are not online routing results.','']
    z=h3['D_route_full_minus_C_ref']; md += ['## H3 — fixed causal routing','',
        '- Six-window mean AP delta D_route_full − C_ref: **%.6f**.'%z['equal_weight_mean_ap_delta'],
        '- Six-window mean AP delta D_route_full − D_live: **%.6f**.'%h3['D_route_full_minus_D_live']['equal_weight_mean_ap_delta'],
        '- Six-window mean AP delta D_route_full − D_route_fragment: **%.6f**.'%h3['D_route_full_minus_D_route_fragment']['equal_weight_mean_ap_delta'],
        '- Registered development signal vs C_ref pass: **%s**.'%str(z['development_reference']['development_signal_all_pass']),
        '- Both cache passes used only labels satisfying `i + 2 < t`; no gradients or learning feedback were used.','']
    md += ['## Budget / validity','', '- Frozen Protocol-033 reference reproduction passed: **%s**.'%str(status['reference_checks_passed']),
        '- Exactly **3** full training replays and **2** fixed no-training cache routing passes were executed.',
        '- No new simulator stream, extra seed, threshold sweep, or automatic scientific follow-up was started.',
        '- Resource-equivalence and statistical-confirmation claims are not made.','']
    Path(a.docs_output).write_text('\n'.join(md)+'\n',encoding='utf8'); print(json.dumps(status,indent=2))
if __name__=='__main__': main()
