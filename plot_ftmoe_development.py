"""Plot complete, audited development comparisons with model-seed variability."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_ftmoe_end_to_end import ART


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'path'})
    fig, axes = plt.subplots(2,2,figsize=(9.2,6.5),sharex=True)
    colors = ['#3565CF','#E38727','#138F75','#A658B3']
    epochs = []
    for run, color in zip(args.runs,colors):
        root = ART / 'runs' / run
        conf = json.loads((root / 'configuration.json').read_text())['arguments']
        epochs.append(conf['epochs'])
        reassessed = ART / 'diagnosis_reassessment' / run
        summaries = []
        for v in range(5):
            group = []
            for seed in [1,2,6]:
                path = (reassessed / f'v{v}_seed{seed}.json' if (reassessed / 'aggregate.json').exists()
                        else root / f'v{v}_seed{seed}/summary.json')
                group.append(json.loads(path.read_text()))
            summaries.append(group)
        for i, protocol in enumerate(['A_last','B_best']):
            for j, metric in enumerate(['f1','diagnosis_hr_at_100pct']):
                points = np.array([[row[protocol]['mean'][metric] for row in group] for group in summaries])
                axes[i,j].errorbar(np.arange(5), points.mean(1), yerr=points.std(1,ddof=1),
                                  color=color, marker='o',markersize=4,capsize=3,linewidth=1.6,
                                  label=f"LR {conf['learning_rate']:g}")
    for i in range(2):
        for j in range(2):
            axis = axes[i,j]
            axis.set_xticks(np.arange(5))
            axis.set_xticklabels(['v0','v1','v2','v3','v4'])
            axis.set_ylim(.4 if j==0 else .6,1.015)
            axis.grid(axis='y',color='#DFE3E9',linewidth=.7)
            for spine in ['top','right']:
                axis.spines[spine].set_visible(False)
            axis.set_title(('A: common final epoch' if i==0 else 'B: validation-best checkpoint') +
                           (' | detection' if j==0 else ' | diagnosis'),fontsize=11,pad=10)
            axis.set_ylabel('Detection F1' if j==0 else 'Resource diagnosis accuracy')
    assert len(set(epochs)) == 1, 'Do not label different budgets as one common-budget comparison'
    fig.suptitle(f'Development comparison: full-model training, {epochs[0]} epochs',fontsize=14,y=.985)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.052),ncol=len(args.runs),frameon=False)
    fig.text(.5,.014,'Error bars: SD across 3 model seeds. Diagnosis HR@100% = NDCG@100% for single-label targets.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.11,1,.95])
    output = ART / 'figures'; output.mkdir(exist_ok=True)
    for suffix in ['png','svg']:
        fig.savefig(output / f'{args.name}.{suffix}',dpi=180,facecolor='white')
    print(output / f'{args.name}.png')


if __name__=='__main__':
    main()
