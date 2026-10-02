"""Validate frozen Protocol039 registration only; does not certify scientific execution."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA='2182aa9c08a2067ea8abb89b70555a8174fc9a2809d0655c59effa4174cd9811'
DOC_SHA='87ca213eaa64b01a05da57319aad31e2e70bc7116e56d2b0d57b0d3ea68b1a28'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pp=ROOT/'artifacts/ftmoe_online/protocol_039/plan.json';p=json.loads(pp.read_text(encoding='utf-8'))
    assert sha(pp)==PLAN_SHA==(pp.parent/'plan.sha256').read_text().split()[0]
    assert sha(ROOT/p['instructions'])==DOC_SHA==p['instructions_sha256']
    assert p['protocol']=='039' and p['revision']==1
    b=p['budget'];assert b['sequence_names']==['D_sleepwake'] and b['new_training_sequence_slots']==1
    assert b['scientific_optimizer_steps_max']==352
    for k in ('new_streams','baseline_retraining','other_real_streams','hyperparameter_sweeps','donor_updates','real_engineering_gradient_steps','permanent_deletions'):assert b[k]==0
    assert b['max_births']==b['max_sleeps']==b['max_wakes']==1
    assert not b['restart_from_zero_after_start'] and not b['automatic_next_protocol']
    f=p['F_research'];assert f['status']=='DEFERRED' and f['training']==f['new_comparisons']==f['donor_trials']==0 and not f['pass_gate']
    assert p['analysis']['primary_baseline']=='C_ref'
    assert p['expert']['parameters']==74 and p['expert']['initialization']=='w=b=0'
    assert p['sleep']['recent_intervals']==128 and p['sleep']['min_matured_active_intervals']==256 and p['sleep']['consecutive_checks']==3
    assert p['wake']['min_matured_sleep_intervals']==64 and p['wake']['restore_optimizer_exactly'] and not p['wake']['reset_optimizer']
    assert not p['freeze']['this_turn_launch_authorized'] and not p['runtime']['helper_training_allowed']
    print(json.dumps({'protocol':'039','registration_files_valid':True,'plan_sha256':PLAN_SHA,'instructions_sha256':DOC_SHA,'scientific_execution_validated':False},indent=2))
if __name__=='__main__':main()
