"""Retrospective development audit using the user's revised comparison graph."""
import json
from pathlib import Path
import numpy as np
from train_ftmoe_end_to_end import ART, ROOT, METRICS, sha, write_json
from evaluate_ftmoe_branch_development import paired_difference

SEEDS = [1, 2, 6]
COMPARISONS = [('v1', 'v0'), ('v2', 'v1'), ('v4', 'v2'), ('v4', 'v3')]


def read_row(run, variant, seed, calibrated=False):
    directory = ART/'runs'/run/f'{variant}_seed{seed}'
    original = json.loads((directory/'summary.json').read_text())
    reassessed = ART/'diagnosis_reassessment'/run/f'{variant}_seed{seed}.json'
    path = ART/'calibrated'/run/f'{variant}_seed{seed}.json' if calibrated else reassessed if reassessed.exists() else directory/'summary.json'
    row = json.loads(path.read_text())
    assert original['all_parameters_trainable']
    config = original['configuration']
    assert config['data'] == 'artifacts/ftmoe_end_to_end/data/protocol_004_physical'
    assert config['stop_after'] is None
    run_configuration = json.loads((directory.parent/'configuration.json').read_text())
    assert run_configuration['manifest_sha256'] == sha(ROOT/config['data']/'manifest.json')
    best = Path(row['best_checkpoint']) if 'best_checkpoint' in row else directory/'best.pt'
    hashes = {str(p): sha(p) for p in (path, directory/'summary.json', best, directory/'last.pt')}
    if 'best_sha256' in row:
        assert hashes[str(best)] == row['best_sha256']
    if calibrated:
        for protocol, checkpoint in [('A_last', directory/'last.pt'), ('B_best', best)]:
            assert row['calibration'][protocol]['checkpoint_sha256'] == hashes[str(checkpoint)]
    return row, config, hashes


def audit(run, calibrated=False, gated=None):
    rows = {}; hashes = {}; configs = {}
    for v in ['v0', 'v1', 'v2', 'v3', 'v4'] + (['v4_gated'] if gated else []):
        rows[v] = {}
        for seed in SEEDS:
            row, config, source = read_row(gated if v=='v4_gated' else run, 'v4' if v=='v4_gated' else v, seed, calibrated)
            rows[v][seed] = row; hashes.update(source); configs[f'{v}_seed{seed}'] = config
    reference = configs['v4_seed1']
    if gated:
        for seed in SEEDS:
            original_full = json.loads((ART/'runs'/run/f'v4_seed{seed}/summary.json').read_text())
            original_gate = json.loads((ART/'runs'/gated/f'v4_seed{seed}/summary.json').read_text())
            assert original_full['parameters'] == original_gate['parameters']
    for key, config in configs.items():
        for field in ('data', 'epochs', 'learning_rate'):
            assert config[field] == reference[field], (key, field)
        assert config.get('moe_context', 'none') == reference.get('moe_context', 'none')
        if not key.startswith('v0_'):
            assert config.get('expert_dropout', 0) == reference.get('expert_dropout', 0)
        assert config.get('fusion', 'attention') == ('gated' if key.startswith('v4_gated_') else 'attention')
    result = {'run': run, 'calibrated': calibrated, 'configuration': reference,
              'source_sha256': hashes, 'missing_matched_fusion_control': not bool(gated)}
    for protocol in ['A_last', 'B_best']:
        means = {v: {m: float(np.mean([r[protocol]['mean'][m] for r in values.values()])) for m in METRICS} for v, values in rows.items()}
        pairs = COMPARISONS + ([('v4', 'v4_gated')] if gated else [])
        differences = {f'{hi}_minus_{lo}': {m: means[hi][m]-means[lo][m] for m in METRICS} for hi,lo in pairs}
        passes = {key: d['f1']>=.005 and all(d[m]>0 for m in METRICS) for key,d in differences.items()}
        result[protocol] = {'means': means, 'differences': differences, 'passes': passes,
            'main_chain_passed': all(passes[f'{hi}_minus_{lo}'] for hi,lo in COMPARISONS[:3]),
            'main_and_simple_fusion_passed': all(passes[f'{hi}_minus_{lo}'] for hi,lo in COMPARISONS),
            'paired': {f'{hi}_minus_{lo}': {m: paired_difference({s: rows[hi][s][protocol] for s in SEEDS},
                {s: rows[lo][s][protocol] for s in SEEDS}, m, SEEDS) for m in METRICS} for hi,lo in pairs},
            'best_epochs': {v: [r['best_epoch'] for r in values.values()] for v,values in rows.items()} if not calibrated else {}}
        result[protocol]['target_passed'] = means['v4']['f1']>=.86 and means['v4'][METRICS[1]]>=.63 and means['v4'][METRICS[2]]>=.58
        result[protocol]['complete_primary_passed'] = bool(gated and result[protocol]['main_chain_passed'] and
            passes['v4_minus_v4_gated'] and result[protocol]['target_passed'])
    return result


def main():
    directory = ART/'data/protocol_004_physical'
    manifest = json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['array_sha256'].items():
        assert sha(directory/f'{name}.npy') == digest, name
    assert not set(manifest['seeds']) & {201,202,203,204,205}
    inventory = []
    for directory in sorted((ART/'runs').iterdir()):
        if not directory.is_dir():
            continue
        summaries = [json.loads(p.read_text()) for p in directory.glob('v*/summary.json')]
        configuration = directory/'configuration.json'
        inventory.append({'run': directory.name, 'completed_models': len(summaries),
            'models': [f"{r['variant']}_seed{r['seed']}" for r in summaries],
            'configuration': json.loads(configuration.read_text()) if configuration.exists() else None,
            'status': 'preflight only' if (directory/'preflight.json').exists() else
                      'assembly with provenance' if (directory/'assembly_provenance.json').exists() else 'training run'})
    groups = [audit(run) for run in ['physical_lr0003_e30', 'physical_lr001_e30', 'physical_lr003_e30', 'physical_lr003_e60']]
    for calibrated in [False, True]:
        groups.append(audit('expert_dropout01_lr003_e60_chain', calibrated, 'gated_fusion_lr003_e60'))
    historic_gate = ART/'runs/gated_fusion_lr0003_e30'
    historic_gate_complete = all((historic_gate/f'v4_seed{s}/summary.json').exists() for s in SEEDS)
    if historic_gate_complete:
        groups.append(audit('physical_lr0003_e30', gated='gated_fusion_lr0003_e30'))
    source = [json.loads(p.read_text()) for p in (ART/'runs/source_attention_lr003_e60').glob('v*/summary.json')]
    source_means = {p: {m: float(np.mean([r[p]['mean'][m] for r in source])) for m in METRICS} for p in ['A_last','B_best']}
    report = {'status': 'retrospective exploratory audit; not preregistered selection or confirmatory evidence',
        'model_seeds': SEEDS, 'development_replays': [31,101,102], 'inventory': inventory, 'groups': groups,
        'source_attention_completed': len(source), 'source_attention_means': source_means,
        'source_gated_completed': len(list((ART/'runs/source_gated_lr003_e60').glob('v*/summary.json'))),
        'excluded': {'initial_lr0003': 'superseded simulator lifecycle/input data; not comparable with corrected physical benchmark',
                     'initial_lr001': 'same data issue and incomplete training',
                     'staged_frozen_runs': 'different budgets/losses and frozen prefixes; retain as historical study',
                     'context_and_dropout_pilots': 'one model seed; useful hypothesis screening, not three-seed confirmation'},
        'limitations': ['Development data were repeatedly used for configuration selection.',
                       'Bootstrap intervals are conditional on selected checkpoints and these development replays.',
                       'HR and NDCG equal top-1 accuracy on these single-resource labels; not two independent achievements.',
                       'All replays share one workload trace; paper data and label definitions differ.',
                       'No reserved test replays are accessed by this audit.']}
    write_json(ART/'history_reuse_audit.json', report)
    lines = ['# 历史实验复用审计', '', '这是依据用户更新消融关系进行的回顾性开发集审计，不是预注册筛选或独立验证。', '',
        '主链为 v0、v1、v2、v4；v3 是简单融合对照。下表按 v0/v1/v2/v3/v4 列出 F1。通过要求每个主链比较 F1 增加至少 0.005、诊断指标均值增加。', '',
        '| 运行 | 选择 | F1 | 主链通过 | 完整模型优于简单融合 |', '|---|---|---|---|---|']
    for group in groups:
        name = group['run'] + ('（阈值校准）' if group['calibrated'] else '')
        if not group['missing_matched_fusion_control']:
            name += '（含门控对照）'
        for protocol in ['A_last','B_best']:
            data = group[protocol]
            values = '/'.join(f"{data['means'][f'v{i}']['f1']:.6f}" for i in range(5))
            lines.append(f"| {name} | {protocol} | {values} | {data['main_chain_passed']} | {data['passes']['v4_minus_v3']} |")
    lines += ['', '## 复用决策', '',
        ('physical_lr0003_e30 的同配置门控融合对照已完成。原 v4 的 B F1=0.903615，门控为 0.899456；原 v4 诊断均值也更高。融合 F1 增量 0.004160 未达到原登记门槛 0.005，条件区间跨零，完整验收仍失败。协议 014 固定配置检查模型种子 17/42，未降低门槛。' if historic_gate_complete else
         '优先补齐 physical_lr0003_e30 的同配置门控融合对照。该组 B_best 在三种子均值上通过新主链与简单融合比较。A_last 没有通过，不能隐藏。'),
        '旧组必须使用 diagnosis_reassessment 中重新选择的每轮 checkpoint，不能使用原 best.pt 冒充当前诊断指标下的最佳模型。审计核验了 15 个重新选择的权重 SHA-256。',
        '复用是节省重复训练，不是确认显著性：v4 相对 v2 的 F1 均值增加约 0.00559，三个模型种子并非均为正，开发集条件置信区间跨零。',
        '旧数据运行、冻结逐级训练和单种子 pilot 仅保留为历史或筛选证据，不能拼接成当前全模型消融表。',
        '贡献解释也需受控：当前 v2 相比 v1 同时增加第二专家池和调度后资源输入，并非只替换路由器；v4 相比 v3 同时增加注意力与逐元素交互。整套图扩展移除对应 v2，不能据此单独证明图消息传递的贡献。',
        '同一开发集的无训练物理规则基线 F1 约 0.93057、诊断准确率约 0.98194，高于旧候选的完整模型。满足内部消融方向不能替代与此基线的比较。',
        '60 轮 dropout 组及门控组的原始、校准结果全部保留。校准后简单融合 v3 的检测更好，不能选择性只展示某些版本的校准结果。',
        f"协议 012 的 source_attention 已完成 {len(source)} 个模型；B_best F1={source_means['B_best']['f1']:.6f}，诊断准确率={source_means['B_best'][METRICS[1]]:.6f}。配对 source_gated 尚需按协议完成，不能宣称该实验已结束。", '',
        '新增对照仍使用物理数据协议 004、lr=0.0003、30 轮、dropout=0、模型种子 1/2/6、相同损失和独立全参数训练。开发后还需要未用模型种子确认；测试 201–205 保持封闭。', '',
        '完整来源哈希、逐比较差值与配对区间见 `artifacts/ftmoe_end_to_end/history_reuse_audit.json`。']
    confirmation = ART/'comparison_014_complete.json'
    if confirmation.exists():
        confirmed = json.loads(confirmation.read_text())
        lines += ['', '## 后续确认已完成', '',
            '协议 014 完成新增种子 17/42 的五版本及门控控制，共 12 个独立全模型训练。完整分组结果见 `FTMOE_HISTORICAL_CONFIRMATION.md`。',
            f"合并五种子的 B 主链 F1：" + '/'.join(f"{confirmed['pooled_five']['B_best']['means'][v]['f1']:.6f}" for v in ['v0','v1','v2','v4']) + '。',
            f"达到原登记冻结测试资格：{confirmed['eligible_for_frozen_test']}。失败的筛选门槛不使结果无效，但必须保留小效应、波动及新增种子反例，不能宣称已得到稳健的模块贡献证明。"]
    (ROOT/'FTMOE_HISTORY_REUSE_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps({'groups': len(groups), 'source_attention': source_means, 'report': str(ROOT/'FTMOE_HISTORY_REUSE_AUDIT.md')}, indent=2))


if __name__ == '__main__':
    main()
