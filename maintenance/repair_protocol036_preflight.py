from pathlib import Path

RUNNER = Path('run_ftmoe_protocol036.py')
ANALYZER = Path('analyze_ftmoe_protocol036.py')


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f'missing expected fragment: {label}')
    return text.replace(old, new, 1)


def repair_runner():
    t = RUNNER.read_text(encoding='utf8')
    t = replace_once(
        t,
        "'causal_access_all_pass':all(access.values())",
        "'causal_access_all_pass':bool(access['prediction_visible_input_rule_ok'] and access['prediction_label_max_rule_ok'] and access['all_actual_batches_mature'] and access['terminal_settlement_optimizer_steps']==0)",
        'C causal access boolean')
    t = replace_once(t,
        "if perturb_from is not None:\n                pf=int(perturb_from); TT['z'][pf:]+=1234.0; TT['c_detection_logits'][pf:,0]-=777.0; TT['c_detection_logits'][pf:,1]+=777.0; TT['labels'][pf:]=np.where(TT['labels'][pf:]>0,0,1)",
        "if perturb_from is not None:\n                pf=int(perturb_from)\n                if pf>=TT['z'].shape[0]: raise AssertionError('future perturbation has no actual future rows')\n                TT['z'][pf:]+=1234.0; TT['c_detection_logits'][pf:,0]-=777.0; TT['c_detection_logits'][pf:,1]+=777.0; TT['labels'][pf:]=np.where(TT['labels'][pf:]>0,0,1)",
        'branch future perturbation nonempty')
    t = replace_once(t,
        "raw={'access_events':accessor.events,'updates':logs,'checkpoint_files':[str(x) for x in (out_dir/'checkpoints').glob('*')],'max_accessed_index':max((max(e['indices']) for e in accessor.events if e['indices']),default=-1)}",
        "raw={'access_events':accessor.events,'updates':logs,'checkpoint_files':[str(x) for x in (out_dir/'checkpoints').glob('*')],'max_accessed_index':max((max(e['indices']) for e in accessor.events if e['indices']),default=-1),'future_perturbation_start':None if perturb_from is None else int(perturb_from),'future_perturbation_rows':0 if perturb_from is None else int(TT['z'].shape[0]-int(perturb_from))}",
        'branch perturbation evidence')
    t = replace_once(t,
        "r={'protocol':'036','synthetic':{},'real_stream_prefix_executions':2,'max_real_prefix_steps':80,'passed':False}",
        "r={'protocol':'036','synthetic':{},'real_stream_prefix_executions':2,'max_real_prefix_steps':96,'passed':False}",
        'fixture max prefix')
    old = "c1,_,sha=new_c700(a.stream,a.registration,a.input_lock,root/'continuous');\n    for _ in range(80): c1.step()\n    c1_hash=c1.learner_state_hash(); T={'z':c1.z_tape[:80].copy(),'c_detection_logits':c1.predictions['detection_logits'][:80].copy(),'c_class_probability':c1.predictions['class_probability'][:80].copy(),'labels':c1.predictions['labels'][:80].copy(),'raw_labels':c1.predictions['raw_labels'][:80].copy()}; batches={int(x['at_interval']):[int(i) for i in x['buffer_indices']] for x in c1.update_log if int(x['at_interval'])<80}; cont={arm:branch_prefix(T,batches,arm,root/'continuous'/arm) for arm in ('D_cal','D_lin','D_corr')}"
    new = "c1,_,sha=new_c700(a.stream,a.registration,a.input_lock,root/'continuous');\n    for _ in range(80): c1.step()\n    c1_prob80=c1.predictions['probability'][:80].copy(); c1_z80=c1.z_tape[:80].copy(); c1_hash80=c1.learner_state_hash()\n    for _ in range(16): c1.step()\n    T={'z':c1.z_tape[:96].copy(),'c_detection_logits':c1.predictions['detection_logits'][:96].copy(),'c_class_probability':c1.predictions['class_probability'][:96].copy(),'labels':c1.predictions['labels'][:96].copy(),'raw_labels':c1.predictions['raw_labels'][:96].copy()}; batches={int(x['at_interval']):[int(i) for i in x['buffer_indices']] for x in c1.update_log if int(x['at_interval'])<80}; cont={arm:branch_prefix(T,batches,arm,root/'continuous'/arm) for arm in ('D_cal','D_lin','D_corr')}"
    t = replace_once(t, old, new, 'continuous fixture 96 rows')
    t = replace_once(t,
        "'C_checkpoint_resume_probability_exact':bool(np.array_equal(c1.predictions['probability'][:80],c3.predictions['probability'][:80])),'C_checkpoint_resume_z_exact':bool(np.array_equal(c1.z_tape[:80],c3.z_tape[:80])),'C_checkpoint_resume_state_hash_exact':bool(c1_hash==c3.learner_state_hash())",
        "'C_checkpoint_resume_probability_exact':bool(np.array_equal(c1_prob80,c3.predictions['probability'][:80])),'C_checkpoint_resume_z_exact':bool(np.array_equal(c1_z80,c3.z_tape[:80])),'C_checkpoint_resume_state_hash_exact':bool(c1_hash80==c3.learner_state_hash())",
        'C fixture compare cursor80')
    t = replace_once(t,
        "'max_accessed_index_before80':int(split[3]['max_accessed_index'])<=79,'actual_batch_indices_exact'",
        "'future_perturbation_rows_positive':int(split[3]['future_perturbation_rows'])>0,'max_accessed_index_before80':int(split[3]['max_accessed_index'])<=79,'actual_batch_indices_exact'",
        'branch fixture evidence')
    RUNNER.write_text(t, encoding='utf8')


def repair_analyzer():
    t = ANALYZER.read_text(encoding='utf8')
    t = replace_once(
        t,
        "'C_causal_access_pass':bool(csum['causal_access_all_pass'] and all(ca.values())),",
        "'C_causal_access_pass':bool(csum['causal_access_all_pass'] and ca['prediction_visible_input_rule_ok'] and ca['prediction_label_max_rule_ok'] and ca['all_actual_batches_mature'] and ca['terminal_settlement_optimizer_steps']==0),",
        'analyzer C causality boolean')
    marker = "    md=['# Protocol-036 结果'"
    pos = t.find(marker)
    if pos < 0:
        raise RuntimeError('missing analyzer markdown marker')
    tail = '''    def fmt(v): return 'null' if v is None else f'{float(v):+.6f}'
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
    Path(a.docs_output).write_text('\\n'.join(md)+'\\n',encoding='utf8')
    print(json.dumps(status,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
'''
    ANALYZER.write_text(t[:pos] + tail, encoding='utf8')


if __name__ == '__main__':
    repair_runner()
    repair_analyzer()
    print('Protocol036 deterministic preflight repairs applied')
