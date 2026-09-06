"""Verify calibration changes only detection decisions and uses a fixed tie rule."""
import json
import numpy as np
from calibrate_ftmoe_validation import metrics_at_threshold, select_threshold
from train_ftmoe_end_to_end import metric_arrays


def main():
    probability = np.array([.05,.6,.65,.7])
    labels = np.array([0,0,1,2])
    classes = np.array([[1,0,0],[0,1,0],[1,0,0],[0,1,0]],dtype=float)
    original = metric_arrays(probability,classes,labels)
    assert metrics_at_threshold(probability,classes,labels,.5) == original
    predictions = {'31':(probability,classes,labels),'101':(probability,classes,labels)}
    grid = (np.arange(5,100,5)/100).tolist()
    threshold, result, _ = select_threshold(predictions,grid)
    assert threshold == .65 and result['mean']['f1'] == 1
    for metric in ['diagnosis_hr_at_100pct','diagnosis_ndcg_at_100pct','hr_at_100','ndcg_at_100','pr_auc']:
        assert result['per_replay']['31'][metric] == original[metric]
    perfect = {'31':(np.array([.01,.01,.99,.99]),classes,labels)}
    threshold, _, _ = select_threshold(perfect,grid)
    assert threshold == .5
    print(json.dumps({'threshold_half_preserves_all_metrics':True,'grid_maximizes_f1':True,
                      'diagnosis_and_ranking_unchanged':True,'tie_prefers_half':True}))


if __name__=='__main__':
    main()
