import copy
import json
from evaluate_ftmoe_branch_development import assess, paired_difference
from train_ftmoe_end_to_end import METRICS


def main():
    comparisons = [('v1','v0'),('v2','v1'),('v4_full','v2'),('v4_full','v4_gated')]
    arms = {name:{m:score for m in METRICS} for name,score in
            [('v0',.70),('v1',.76),('v2',.84),('v4_full',.90),('v4_gated',.87),('v3_simple_fusion',.60)]}
    assert assess(arms,comparisons)['passed']  # v3 below v2 must not invalidate the new comparison.
    failed = copy.deepcopy(arms); failed['v4_gated']['f1'] = .91
    assert not assess(failed,comparisons)['passed']
    failed = copy.deepcopy(arms); failed['v2']['f1'] = .898
    assert not assess(failed,comparisons)['passed']
    low = {s:{'per_replay':{r:{'f1':.8} for r in ['31','101','102']}} for s in [1,2,6]}
    high = {s:{'per_replay':{r:{'f1':.9} for r in ['31','101','102']}} for s in [1,2,6]}
    result = paired_difference(high,low,'f1',[1,2,6],100)
    assert all(abs(x-.1)<1e-12 for x in [result['mean'],*result['ci95_conditional']])
    print(json.dumps({'v3_not_required_to_exceed_v2':True,'full_vs_replacement_required':True,
                      'f1_margin_required':True,'paired_constant_difference':True}))


if __name__=='__main__': main()
