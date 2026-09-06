"""Protocol014: never select or discard confirmation seeds based on their results."""
import json
import numpy as np
from audit_ftmoe_history import read_row
from evaluate_ftmoe_branch_development import paired_difference
from train_ftmoe_end_to_end import ART, METRICS, sha, write_json


def main():
    registration_path = ART/'protocol_014_historical_confirmation.json'
    registration = json.loads(registration_path.read_text())
    old_seeds = registration['original_model_seeds']; new_seeds = registration['new_model_seeds']
    seeds = old_seeds + new_seeds
    arms = ['v0','v1','v2','v3','v4','v4_gated']
    rows = {arm:{} for arm in arms}; hashes = {}; parameters = {}
    for arm in arms:
        for seed in seeds:
            variant = 'v4' if arm=='v4_gated' else arm
            if seed in old_seeds:
                run = 'gated_fusion_lr0003_e30' if arm=='v4_gated' else 'physical_lr0003_e30'
            else:
                run = registration['gated_run'] if arm=='v4_gated' else registration['attention_run']
            row,config,source = read_row(run,variant,seed)
            for key in ['data','epochs','learning_rate']:
                assert config[key] == registration[key]
            assert config.get('expert_dropout',0)==0 and config.get('moe_context','none')=='none'
            assert config.get('fusion','attention') == ('gated' if arm=='v4_gated' else 'attention')
            original = json.loads((ART/'runs'/run/f'{variant}_seed{seed}/summary.json').read_text())
            parameters[arm,seed] = original['parameters']
            rows[arm][seed] = row; hashes.update(source)
    for seed in seeds:
        assert parameters['v4',seed] == parameters['v4_gated',seed]
    comparisons = registration['primary_comparisons'] + registration['supplemental_comparisons']
    output = {'registration_sha256':sha(registration_path),'source_sha256':hashes,
              'disclosure':registration['limits'],'primary_protocol':'B_best'}
    for name,subset in [('original_three',old_seeds),('new_two',new_seeds),('pooled_five',seeds)]:
        output[name] = {'seeds':subset}
        for protocol in ['A_last','B_best']:
            means = {arm:{m:float(np.mean([rows[arm][s][protocol]['mean'][m] for s in subset])) for m in METRICS} for arm in arms}
            paired = {f'{hi}_minus_{lo}':{m:paired_difference({s:rows[hi][s][protocol] for s in subset},
                {s:rows[lo][s][protocol] for s in subset},m,subset) for m in METRICS} for hi,lo in comparisons}
            directions = {f'{hi}_minus_{lo}':all(paired[f'{hi}_minus_{lo}'][m]['mean']>0 for m in METRICS)
                          for hi,lo in registration['primary_comparisons']}
            gates = {key:passed and paired[key]['f1']['mean']>=.005 for key,passed in directions.items()}
            targets = means['v4']['f1']>=.86 and means['v4'][METRICS[1]]>=.63 and means['v4'][METRICS[2]]>=.58
            output[name][protocol] = {'means':means,'paired':paired,'positive_primary_directions':directions,
                'primary_effect_gates':gates,'target_passed':targets,'passed':all(gates.values()) and targets,
                'best_epochs':{arm:{s:rows[arm][s]['best_epoch'] for s in subset} for arm in arms}}
    output['initialization_check_supportive'] = all(output['new_two']['B_best']['positive_primary_directions'].values())
    output['eligible_for_frozen_test'] = bool(output['pooled_five']['B_best']['passed'] and output['initialization_check_supportive'])
    write_json(ART/'comparison_014_complete.json',output)
    lines = ['# 历史候选的新增模型种子确认', '',
        '配置固定为学习率 0.0003、30 轮、全参数训练。新增种子 17/42 与原种子 1/2/6 分别报告；所有种子均保留。',
        '这些实验只检验同一开发数据上的初始化稳定性，尚不是独立测试。B_best 为预先固定的主要选择方式。', '',
        '| 模型种子 | 选择 | v0 F1 | v1 F1 | v2 F1 | v3 F1 | 完整 v4 F1 | 门控 v4 F1 |', '|---|---|---|---|---|---|---|---|']
    for group in ['original_three','new_two','pooled_five']:
        for protocol in ['A_last','B_best']:
            values = ' | '.join(f"{output[group][protocol]['means'][arm]['f1']:.6f}" for arm in arms)
            lines.append(f'| {group} | {protocol} | {values} |')
    lines += ['', f"新种子全部主比较方向为正：{output['initialization_check_supportive']}。",
              f"达到预先登记的冻结测试资格：{output['eligible_for_frozen_test']}。", '',
              '五种子合并后，B 的主链及完整模型相对门控替换仍保持正向均值；但完整 v4 相对 v2 的 F1 增量仅约 0.00380，相对门控仅约 0.00155，均低于先前登记的 0.005。两项检测差值的条件区间均跨零。',
              '新增种子 17/42 中，门控模型的检测 F1 均略高于完整注意力模型；v1 相对 v0 的诊断均值也没有改善。因此不能把旧三种子的注意力优势写成已稳定复现。',
              '这些仍是可报告的有效实验；未通过更严格的开发筛选标准，不等于实验作废。可以作为探索性消融结果使用，但须同时报告方差、A/B 选择差异及新种子反例。', '',
              '逐比较差值、诊断准确率、逐种子方向、条件置信区间和权重来源见 `comparison_014_complete.json`。',
              'HR 与 NDCG 在本数据的单资源标签下相等。开发集选模的条件区间不能替代独立测试；预留测试 201–205 未由本程序访问。']
    (ART.parent.parent/'FTMOE_HISTORICAL_CONFIRMATION.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'eligible_for_frozen_test':output['eligible_for_frozen_test'],
        'initialization_check_supportive':output['initialization_check_supportive'],
        'pooled_B':output['pooled_five']['B_best']['means'],
        'pooled_gates':output['pooled_five']['B_best']['primary_effect_gates']},indent=2))


if __name__=='__main__': main()
