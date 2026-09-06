"""Standalone development figure retaining every model seed and both selections."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_ftmoe_history import read_row
from train_ftmoe_end_to_end import ART


def main():
    report = json.loads((ART/'comparison_014_complete.json').read_text())
    arms = ['v0','v1','v2','v3','v4','v4_gated']
    labels = ['v0','v1','v2','v3\nsimple fusion','v4\nfull attention','v4\ngated fusion']
    seeds = [1,2,6,17,42]
    rows = {}
    for arm in arms:
        for seed in seeds:
            if seed in [1,2,6]:
                run = 'gated_fusion_lr0003_e30' if arm=='v4_gated' else 'physical_lr0003_e30'
            else:
                run = 'historical_gated_confirm_lr0003_e30' if arm=='v4_gated' else 'historical_confirm_lr0003_e30'
            rows[arm,seed] = read_row(run,'v4' if arm=='v4_gated' else arm,seed)[0]
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    colors = ['#4477aa','#66ccee','#228833','#cc6677','#aa3377']
    for row,protocol in enumerate(['B_best','A_last']):
        for column,metric in enumerate(['f1','diagnosis_hr_at_100pct']):
            ax = axes[row,column]
            values = np.array([[rows[arm,s][protocol]['mean'][metric] for s in seeds] for arm in arms])
            for j,seed in enumerate(seeds):
                ax.scatter(np.arange(6)+(j-2)*.06,values[:,j],s=25,color=colors[j],alpha=.8,
                           label=f'Seed {seed}' + (' (new)' if seed in [17,42] else ''))
            ax.plot(np.arange(6),values.mean(1),'k_',markersize=18,markeredgewidth=2,label='Five-seed mean')
            ax.set_xticks(np.arange(6)); ax.set_xticklabels(labels,fontsize=9)
            ax.set_ylim((.35,1.0) if metric=='f1' else (.65,1.015))
            ax.set_title(f"{'B: validation-selected' if row==0 else 'A: epoch 30'} | {'Detection F1' if column==0 else 'Resource top-1 accuracy'}")
            ax.grid(axis='y',alpha=.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,.025))
    fig.suptitle('Historical configuration: lr=0.0003, 30 epochs, full-model training',fontsize=14)
    fig.text(.5,.012,'Each dot averages the same three development replays. These are initialization checks, not independent test results.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95))
    output=ART/'figures'; output.mkdir(exist_ok=True)
    for extension in ['png','svg']:
        fig.savefig(output/f'historical_confirmation_014.{extension}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    print(str(output/'historical_confirmation_014.png'))


if __name__=='__main__': main()
