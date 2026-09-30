"""Protocol-036 frozen analysis: correction controls and three new streams."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from protocol035_common import dump_json, load_npz, binary_metrics, phase_bounds, WINDOWS, W_BLOCKS, pooled_ranges

METHODS=('D_cal','D_lin','D_corr')
SEEDS=(3601,3602,3603)
LATE={'U_rec2','V_rec2','U_rec3','V_rec3'}

def J(p): return json.loads(Path(p).read_text(encoding='utf8'))
def diff(d,c):
    return {k+'_delta':None if d.get(k) is None or c.get(k) is None else float(d[k]-c[k]) for k in ('ap','bce','recall','fpr')}
def block(p,y,a,b): return binary_metrics(p[a:b],y[a:b])
def mean_defined(values):
    return None if any(v is None for v in values) else float(np.mean(values))
def ge(v,x): return v is not None and v>=x
def gt(v,x): return v is not None and v>x

def analyze_pair(C,D,bounds):
    y=C['labels']
    if not np.array_equal(y,D['labels']): raise AssertionError('label mismatch')
    full_c=binary_metrics(C['probability'],y); full_d=binary_metrics(D['probability'],y)
    windows=[]
    for n,s,e in WINDOWS:
        cm=block(C['probability'],y,s,e); dm=block(D['probability'],y,s,e)
        c32=block(C['probability'],y,s,s+32); d32=block(D['probability'],y,s,s+32)
        c64=block(C['probability'],y,s,s+64); d64=block(D['probability'],y,s,s+64)
        windows.append({'window':n,'intervals':[s,e],'C':cm,'D':dm,**diff(dm,cm),
                        'prefix32':{'C':c32,'D':d32,**diff(d32,c32)},
                        'prefix64':{'C':c64,'D':d64,**diff(d64,c64)}})
    six=mean_defined([w['ap_delta'] for w in windows]); late=mean_defined([w['ap_delta'] for w in windows if w['window'] in LATE])
    p32=mean_defined([w['prefix32']['ap_delta'] for w in windows]); p64=mean_defined([w['prefix64']['ap_delta'] for w in windows])
    pc=pooled_ranges(C['probability'],y,WINDOWS); pd=pooled_ranges(D['probability'],y,WINDOWS)
    wrows=[]
    for n in W_BLOCKS:
        s,e=bounds[n]; cm=block(C['probability'],y,s,e); dm=block(D['probability'],y,s,e); wrows.append({'phase':n,'intervals':[s,e],'C':cm,'D':dm,**diff(dm,cm)})
    wmean=mean_defined([r['ap_delta'] for r in wrows])
    local=[]
    for w in windows:
        rd=w['recall_delta']; r32=w['prefix32']['recall_delta']; local.append({'window':w['window'],'recall_delta':rd,'prefix32_recall_delta':r32,
            'recall_pass':ge(rd,-.02),'prefix32_recall_pass':ge(r32,-.03),'C_support':{k:w['C'][k] for k in ('positives','negatives','tp','fp','fn','tn')},'D_support':{k:w['D'][k] for k in ('positives','negatives','tp','fp','fn','tn')}})
    guards={'full_fpr_delta':diff(full_d,full_c)['fpr_delta'],'full_recall_delta':diff(full_d,full_c)['recall_delta'],
            'recurrence_pooled_fpr_delta':diff(pd,pc)['fpr_delta'],'recurrence_pooled_recall_delta':diff(pd,pc)['recall_delta'],
            'W_equal_weight_AP_delta':wmean,'local_recurrence':local}
    common=bool(guards['full_fpr_delta'] is not None and guards['full_fpr_delta']<=.01 and guards['recurrence_pooled_fpr_delta'] is not None and guards['recurrence_pooled_fpr_delta']<=.01 and guards['full_recall_delta'] is not None and guards['full_recall_delta']>=-.01 and guards['recurrence_pooled_recall_delta'] is not None and guards['recurrence_pooled_recall_delta']>=-.01 and ge(wmean,-.02))
    localpass=all(x['recall_pass'] and x['prefix32_recall_pass'] for x in local); guards['common_pass']=common; guards['local_pass']=localpass; guards['all_pass']=common and localpass
    hosts=[]
    for h in range(16):
        cm=binary_metrics(C['probability'][:,h],y[:,h]); dm=binary_metrics(D['probability'][:,h],y[:,h]); hosts.append({'host':h,'C':cm,'D':dm,**diff(dm,cm)})
    b32=[]
    for n,s,e in WINDOWS:
        for q in range(4):
            aa=s+q*32; bb=aa+32; cm=block(C['probability'],y,aa,bb); dm=block(D['probability'],y,aa,bb); b32.append({'window':n,'block':q+1,'intervals':[aa,bb],'C':cm,'D':dm,**diff(dm,cm)})
    return {'C_full':full_c,'D_full':full_d,**diff(full_d,full_c),'windows':windows,'six_window_equal_weight_AP_delta':six,
            'late4_equal_weight_AP_delta':late,'prefix32_equal_weight_AP_delta':p32,'prefix64_equal_weight_AP_delta':p64,
            'recurrence_pooled':{'C':pc,'D':pd,**diff(pd,pc)},'W_blocks':wrows,'guardrails':guards,'per_host':hosts,'four_nonoverlap32_blocks':b32}

def load_stage(root,seed):
    base=Path(root)/'stage_B'/f'seed{seed}'; C=load_npz(base/'C_ref/predictions.npz'); bounds=phase_bounds(base/'stream_manifest.json') if (base/'stream_manifest.json').exists() else None
    return base,C,bounds

def validity_for_seed(base,C,input_audit):
    d_off=load_npz(base/'D_off_exact_C_copy/predictions.npz')
    d_off_exact=all(k in d_off and np.array_equal(C[k],d_off[k]) for k in ('probability','class_probability','detection_logits','class_logits','labels','raw_labels'))
    csum=J(base/'C_ref/summary.json'); ca=J(base/'C_ref/causal_access_audit.json')
    arms={}
    for arm in METHODS:
        sm=J(base/arm/'summary.json'); iso=J(base/arm/'branch_isolation_audit.json'); caus=J(base/arm/'causality_audit.json')
        arms[arm]={'summary_pass':bool(sm['completed'] and sm['initial_zero_output_exact'] and sm['parameter_updated']),
                   'causality_pass':bool(sm['causality_all_pass'] and caus['all_update_batches_mature'] and caus['future_rows_read_count']==0 and caus['terminal_drain_training_steps']==0),
                   'isolation_pass':bool(sm['isolation_all_pass'] and iso['unchanged'] and iso['optimizer_parameters_exactly_branch_parameters'] and iso['classification_exact_copy'])}
        arms[arm]['all_pass']=all(arms[arm].values())
    out={'input_audit_pass':bool(input_audit['audit_pass']),'C_causal_access_pass':bool(csum['causal_access_all_pass'] and ca['prediction_visible_input_rule_ok'] and ca['prediction_label_max_rule_ok'] and ca['all_actual_batches_mature'] and ca['terminal_settlement_optimizer_steps']==0),
         'D_off_exact_C':d_off_exact,'arms':arms}; out['all_pass']=out['input_audit_pass'] and out['C_causal_access_pass'] and d_off_exact and all(x['all_pass'] for x in arms.values()); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',required=True); ap.add_argument('--stage-a-manifest',required=True); ap.add_argument('--fixture',required=True); ap.add_argument('--implementation-manifest',required=True); ap.add_argument('--budget-ledger',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--docs-output',required=True); ap.add_argument('--run-id',required=True)
    for seed in SEEDS:
        ap.add_argument(f'--seed{seed}-manifest',required=True); ap.add_argument(f'--seed{seed}-input-audit',required=True)
    a=ap.parse_args(); root=Path(a.run_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    fixture=J(a.fixture); impl=J(a.implementation_manifest); ledger=J(a.budget_ledger)
    # Stage A is attribution-only on the observed development stream.
    A=root/'stage_A_seed700'; AC=load_npz(A/'C_ref_existing/predictions.npz'); abounds=phase_bounds(a.stage_a_manifest); stageA={}
    for arm,path in [('D_cal',A/'D_cal/predictions.npz'),('D_lin',A/'D_lin/predictions.npz'),('D_corr',A/'D_corr_existing/predictions.npz')]:
        stageA[arm]=analyze_pair(AC,load_npz(path),abounds)
    # Stage B primary, exactly three registered streams.
    per_stream={}; all_valid=True
    for seed in SEEDS:
        base=root/'stage_B'/f'seed{seed}'; C=load_npz(base/'C_ref/predictions.npz'); bounds=phase_bounds(getattr(a,f'seed{seed}_manifest'))
        methods={arm:analyze_pair(C,load_npz(base/arm/'predictions.npz'),bounds) for arm in METHODS}
        valid=validity_for_seed(base,C,J(getattr(a,f'seed{seed}_input_audit'))); all_valid &= valid['all_pass']
        per_stream[str(seed)]={'methods':methods,'validity':valid,'stream_sha256':J(base/'C_ref/summary.json')['stream_sha256']}
    corr=[per_stream[str(s)]['methods']['D_corr'] for s in SEEDS]
    mf=mean_defined([x['ap_delta'] for x in corr]); m6=mean_defined([x['six_window_equal_weight_AP_delta'] for x in corr]); ml=mean_defined([x['late4_equal_weight_AP_delta'] for x in corr]); mp32=mean_defined([x['prefix32_equal_weight_AP_delta'] for x in corr])
    both=sum(gt(x['ap_delta'],0) and gt(x['six_window_equal_weight_AP_delta'],0) for x in corr)
    floor_ok=all(x['ap_delta'] is not None and x['ap_delta']>=-.002 and x['six_window_equal_weight_AP_delta'] is not None and x['six_window_equal_weight_AP_delta']>=-.002 for x in corr)
    guards_all=all(x['guardrails']['all_pass'] for x in corr)
    seqs=ledger.get('sequences',{}); budget_pass=(len(seqs)==14 and all(v.get('started') and v.get('completed') for v in seqs.values()) and int(ledger.get('hyperparameter_sweeps',-1))==0 and int(ledger.get('extra_model_seeds',-1))==0 and int(ledger.get('old_D_retraining',-1))==0 and int(ledger.get('cache_router_trials',-1))==0)
    fixture_pass=bool(fixture.get('passed') and fixture.get('real_stream_prefix_executions')<=2 and fixture.get('max_real_prefix_steps')<=256)
    impl_pass=bool(impl.get('protocol')=='036' and impl.get('checkout_sha') not in (None,'unknown') and impl.get('workflow_ref_sha') is not None and impl.get('plan_sha256')=='339980818a4d0d7518b6a33e696bcfa02ea3ec27d69467d4242264277c051a47')
    audit_all=bool(all_valid and budget_pass and fixture_pass and impl_pass)
    cross=bool(ge(mf,.005) and ge(m6,.005) and ge(ml,0) and ge(mp32,-.005) and both>=2 and floor_ok and guards_all and audit_all)
    control_signals={}
    for ctrl in ('D_cal','D_lin'):
        six=[]; full=[]; rows=[]
        for s in SEEDS:
            c=per_stream[str(s)]['methods']; ds=c['D_corr']; cs=c[ctrl]
            sd=None if ds['six_window_equal_weight_AP_delta'] is None or cs['six_window_equal_weight_AP_delta'] is None else float(ds['six_window_equal_weight_AP_delta']-cs['six_window_equal_weight_AP_delta'])
            fd=None if ds['ap_delta'] is None or cs['ap_delta'] is None else float(ds['ap_delta']-cs['ap_delta'])
            six.append(sd); full.append(fd); rows.append({'seed':s,'six_AP_Dcorr_minus_control':sd,'full_AP_Dcorr_minus_control':fd,'D_corr_guardrails_pass':ds['guardrails']['all_pass'],'control_guardrails_pass':cs['guardrails']['all_pass']})
        ms=mean_defined(six); mfull=mean_defined(full); signal=bool(ge(ms,.002) and ge(mfull,0)); control_signals[ctrl]={'mean_six_AP_Dcorr_minus_control':ms,'mean_full_AP_Dcorr_minus_control':mfull,'incremental_signal':signal,'per_stream':rows}
    ap_core=bool(ge(mf,.005) and ge(m6,.005) and ge(ml,0) and ge(mp32,-.005) and both>=2 and floor_ok)
    local_ok=all(all(w['recall_pass'] and w['prefix32_recall_pass'] for w in x['guardrails']['local_recurrence']) for x in corr)
    operating_risk=bool(ap_core and not local_ok)
    if cross and all(v['incremental_signal'] for v in control_signals.values()): label='cross_stream_gain_with_incremental_mlp_evidence'
    elif cross: label='cross_stream_gain_controls_not_both_outperformed'
    elif operating_risk: label='operating_point_risk'
    else: label='cross_stream_gain_not_established'
    aggregate={'D_corr_mean_full_AP_delta':mf,'D_corr_mean_six_AP_delta':m6,'D_corr_mean_late4_AP_delta':ml,'D_corr_mean_prefix32_AP_delta':mp32,'streams_both_full_and_six_positive':int(both),'stream_floor_pass':floor_ok,'all_D_corr_guardrails_pass':guards_all,'all_validity_audits_pass':audit_all,'cross_stream_gain_signal':cross,'operating_point_risk':operating_risk}
    comp={'protocol':'036','run_id':str(a.run_id),'development_only':True,'statistical_confirmation':False,'stage_A_attribution_only':stageA,'stage_B_per_stream':per_stream,'stage_B_aggregate':aggregate,'control_incremental_signals':control_signals,'registered_result_label':label,'validity':{'fixture_pass':fixture_pass,'implementation_manifest_pass':impl_pass,'budget_pass':budget_pass,'stream_and_model_audits_pass':all_valid,'all_pass':audit_all}}
    dump_json(out/'comparison.json',comp)
    # Cost profile: explicit C + branch costs, never present branch-only time as total deployment cost.
    costs={'protocol':'036','streams':{},'stage_A':{}}
    for arm in ('D_cal','D_lin'): costs['stage_A'][arm]=J(A/arm/'summary.json')['cost']
    for s in SEEDS:
        base=root/'stage_B'/f'seed{s}'; costs['streams'][str(s)]={'C_ref':J(base/'C_ref/summary.json')['cost'],**{arm:J(base/arm/'summary.json')['cost'] for arm in METHODS}}
    costs['accounting']='Each corrected deployment includes the full C path plus its branch; branch-only wall time is not total deployment cost.'; dump_json(out/'cost_profile.json',costs)
    status={'protocol':'036','run_id':str(a.run_id),'scientific_status':'completed','publication_status':'pending_until_main_sync','training_sequences_registered':14,'training_sequences_completed':14,'new_streams_registered':3,'new_streams_completed':3,'hyperparameter_sweeps':0,'automatic_followup_training':False,'cross_stream_gain_signal':cross,'D_corr_over_D_cal_incremental_signal':control_signals['D_cal']['incremental_signal'],'D_corr_over_D_lin_incremental_signal':control_signals['D_lin']['incremental_signal'],'operating_point_risk':operating_risk,'registered_result_label':label,'validity_all_pass':audit_all,'statistical_confirmation':False,'stop_after_registered_budget':True}; dump_json(out/'status.json',status)
    def fmt(v): return 'null' if v is None else f'{float(v):+.6f}'
    md=['# Protocol-036 结果','','本轮严格按冻结计划执行：开发流700仅训练 D_cal/D_lin；三条新流 3601/3602/3603 各执行 C_ref、D_cal、D_lin、D_corr。共14条新训练序列、3条新原始流，无扫参、无额外模型种子、无旧动态D重训。','',f'登记结果标签：**{label}**。本轮不构成统计学确认。','', '## B阶段：三条新流主结果','', '| Seed | D_corr 全程 ΔAP | 六窗 ΔAP | late4 ΔAP | prefix32 ΔAP | 护栏 |', '|---:|---:|---:|---:|---:|:---:|']
    for s in SEEDS:
        x=per_stream[str(s)]['methods']['D_corr']; md.append(f"| {s} | {fmt(x['ap_delta'])} | {fmt(x['six_window_equal_weight_AP_delta'])} | {fmt(x['late4_equal_weight_AP_delta'])} | {fmt(x['prefix32_equal_weight_AP_delta'])} | {x['guardrails']['all_pass']} |")
    md += ['',f"三流等权平均：全程 ΔAP **{fmt(mf)}**；六窗 ΔAP **{fmt(m6)}**；late4 **{fmt(ml)}**；prefix32 **{fmt(mp32)}**。",f"同时全程和六窗为正的流：**{both}/3**；跨流收益信号：**{cross}**；operating-point risk：**{operating_risk}**。",'', '## 简化控制对照','']
    for ctrl in ('D_cal','D_lin'):
        x=control_signals[ctrl]; md.append(f"- D_corr 相对 {ctrl}：三流平均六窗增量 **{fmt(x['mean_six_AP_Dcorr_minus_control'])}**，平均全程增量 **{fmt(x['mean_full_AP_Dcorr_minus_control'])}**，登记增量证据：**{x['incremental_signal']}**。")
    md += ['', '## A阶段：开发流700归因（不进入跨流主结论）','']
    for arm in METHODS:
        x=stageA[arm]; md.append(f"- {arm}: 全程 ΔAP {fmt(x['ap_delta'])}；六窗 ΔAP {fmt(x['six_window_equal_weight_AP_delta'])}。")
    md += ['', '## 有效性与边界','',f'- 工程 fixture（真实连续/断点恢复、真实未来扰动前缀不变、AP ties）：**{fixture_pass}**。',f'- 三流输入审计、C因果访问、D_off精确复制、分支隔离/成熟批次：**{all_valid}**。',f'- 14条训练预算账本完整：**{budget_pass}**。','- 3条流是3个独立流随机性重复；host/窗口不是独立重复。固定模型种子1，因此不能声称跨模型初始化确认。','- D_cal/D_lin/D_corr 容量和优化几何不同；控制比较不能解释为严格因果分解或同容量优势。','- 本轮不新增动态记忆、路由或硬删除，因此不提出抗遗忘、记忆恢复或动态专家机制优势结论。','']
    Path(a.docs_output).write_text('\n'.join(md)+'\n',encoding='utf8')
    print(json.dumps(status,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
