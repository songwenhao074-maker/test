"""Explain fixed composite checkpoint selection without changing any experiment."""
import json
import numpy as np
from audit_ftmoe_history import read_row
from train_ftmoe_end_to_end import ART, ROOT, METRICS, write_json


def main():
    rows = []
    for variant in ['v0','v1','v2','v3','v4']:
        for seed in [1,2,6]:
            row,_,_ = read_row('physical_lr0003_e30',variant,seed)
            result = {'variant':variant,'seed':seed,'best_epoch':row['best_epoch']}
            for protocol in ['A_last','B_best']:
                values = row[protocol]
                cpu_fraction = []; cpu_baseline = []
                for replay in ['31','101','102']:
                    confusion = np.asarray(values['per_replay'][replay]['class_confusion'])
                    cpu_fraction.append(float(confusion[:,0].sum()/confusion.sum()))
                    cpu_baseline.append(float(confusion[0].sum()/confusion.sum()))
                result[protocol] = {k:values['mean'][k] for k in [*METRICS,'positive_class_macro_f1','score']}
                result[protocol]['predicted_cpu_fraction_on_true_anomalies'] = float(np.mean(cpu_fraction))
                result[protocol]['always_cpu_diagnosis_accuracy'] = float(np.mean(cpu_baseline))
            rows.append(result)
    report = {'purpose':'Read-only explanation of the previously fixed S selection, not new checkpoint selection.',
        'formula':'S=(F1+HR+NDCG)/3=(F1+2*resource_top1_accuracy)/3 on this single-label dataset.',
        'rows':rows,
        'interpretation':'A common composite rule is procedurally fair but emphasizes diagnosis twice. High CPU prevalence lets nearly constant resource predictions receive a high diagnosis score, so early checkpoints can beat later checkpoints despite worse detection and macro resource F1. This explains a selection tradeoff, not proof of data leakage or an architecture bug.'}
    write_json(ART/'historical_selection_diagnostic.json',report)
    lines = ['# 历史候选选模诊断', '',
        '以下只解释已固定的综合选模规则，不重新选 checkpoint。单资源标签使 HR=NDCG=资源 top-1 准确率，因此 S=(F1+2×资源准确率)/3。',
        '当前开发集异常样本以 CPU 类为主。近乎恒定地预测 CPU 也可能获得较高诊断分数，因此早期 checkpoint 可以在检测 F1 较差时胜过末轮。统一规则保证流程一致，但不能消除类别不均衡或分数权重的影响。', '',
        '| 版本/种子 | 最佳轮次 | B 检测 F1 | B 资源 macro F1 | B 预测 CPU 比例 | A 检测 F1 | A 资源 macro F1 |', '|---|---|---|---|---|---|---|']
    for r in rows:
        a,b=r['A_last'],r['B_best']
        lines.append(f"| {r['variant']}/{r['seed']} | {r['best_epoch']} | {b['f1']:.5f} | {b['positive_class_macro_f1']:.5f} | {b['predicted_cpu_fraction_on_true_anomalies']:.5f} | {a['f1']:.5f} | {a['positive_class_macro_f1']:.5f} |")
    lines += ['', '上述 CPU 比例只在真实异常样本上统计，以三个开发重放等权平均。资源 macro F1 是三类资源分类 F1 的平均，与异常检测 F1 不同。',
        '当前确认实验保留原 S。后续若研究类别平衡损失或 macro 指标选模，应另行登记，并对所有模型使用同一方案。不能仅替换不利的基线 checkpoint。']
    (ROOT/'FTMOE_HISTORICAL_SELECTION_DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'always_cpu_accuracy':rows[0]['B_best']['always_cpu_diagnosis_accuracy'],
        'early_baseline_rows':[r for r in rows if r['variant'] in ['v0','v1'] and r['best_epoch']<=3]},indent=2))


if __name__=='__main__': main()
