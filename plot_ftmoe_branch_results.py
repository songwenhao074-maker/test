"""Plot the registered paired comparisons without implying v2<v3<v4."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_ftmoe_end_to_end import ART


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--raw',required=True)
    parser.add_argument('--calibrated',required=True); parser.add_argument('--name',required=True)
    args = parser.parse_args()
    reports = [json.loads((ART/args.raw).read_text()),json.loads((ART/args.calibrated).read_text())]
    keys = ['v1_minus_v0','v2_minus_v1','v4_full_minus_v2','v4_full_minus_v4_gated']
    labels = ['MoE - temporal','EAGate - MoE','Full - no graph extension','Full - gated fusion']
    fig,axes = plt.subplots(2,2,figsize=(13,7.5))
    for row,(report,title) in enumerate(zip(reports,['Fixed threshold 0.5','Validation-calibrated thresholds'])):
        for column,(metric,heading) in enumerate([('f1','Detection F1'),('diagnosis_hr_at_100pct','Diagnosis HR = NDCG')]):
            ax = axes[row,column]
            for index,key in enumerate(keys):
                value = report['B_best']['paired'][key][metric]
                lo,hi = value['ci95_conditional']; mean = value['mean']
                ax.hlines(index,lo,hi,color='#3c6c94',linewidth=2)
                ax.scatter([mean],[index],s=42,color='#24557b',zorder=3)
                ax.annotate(f'{mean:+.4f}',(hi,index),xytext=(6,0),textcoords='offset points',fontsize=9,va='center')
            ax.axvline(0,color='#555555',linewidth=1)
            if metric=='f1': ax.axvline(.005,color='#bb782b',linestyle='--',linewidth=1,label='Registered F1 margin')
            ax.set_yticks(range(4)); ax.set_yticklabels(labels,fontsize=9)
            ax.set_ylim(3.6,-.6); ax.margins(x=.24)
            ax.set_title(f'{title}\n{heading}',fontsize=11)
            ax.set_xlabel('Paired difference on development replays')
            ax.grid(axis='x',alpha=.2)
            for spine in ['top','right']: ax.spines[spine].set_visible(False)
    fig.suptitle('Protocol 011: branch ablations | validation-selected checkpoints',fontsize=14,y=.985)
    fig.text(.5,.015,'Intervals: conditional 95% two-axis bootstrap over 3 model seeds and 3 development replays.\n'
             'These data were used for model selection; intervals are not independent confirmation. No v3 > v2 requirement.',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.07,1,.95])
    destination = ART/'figures'; destination.mkdir(exist_ok=True)
    for suffix in ['png','svg']: fig.savefig(destination/f'{args.name}.{suffix}',dpi=170)
    print(destination/f'{args.name}.png')


if __name__=='__main__': main()
