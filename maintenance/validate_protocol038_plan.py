"""Validate Protocol038 frozen registration only; no training or execution certification."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PLAN_SHA='7fb212d591009865ebbf0b5e3e77f5e8f0b096e124f2fa4e1885025a89b93ef3'
EXPECTED_DOC_SHA='6ef7b5cde53e49a52c43763ee9e429a6aad9c1da0b8e23efba5b7955d1c60e0b'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    path=ROOT/'artifacts/ftmoe_online/protocol_038/plan.json'
    p=json.loads(path.read_text(encoding='utf-8'))
    assert sha(path)==EXPECTED_PLAN_SHA==(path.parent/'plan.sha256').read_text().split()[0]
    assert sha(ROOT/p['instructions'])==EXPECTED_DOC_SHA==p['instructions_sha256']
    assert p['protocol']=='038' and p['revision']==1
    b=p['budget']; assert b['new_full_training_sequences']==['D_bias','D_warm']
    assert b['total_training_slots']==b['new_full_training_sequence_slots']+b['donor_prefix_slots']==3
    assert b['max_new_scientific_optimizer_steps']==b['donor_steps']+2*b['steps_per_new_arm']==725
    for k in ('new_streams','baseline_retraining','other_real_streams','hyperparameter_sweeps','real_engineering_gradient_steps'): assert b[k]==0
    assert not b['restart_from_zero_after_start'] and not b['automatic_next_protocol']
    assert p['donor']['updates_at']==list(range(15,336,16)) and p['donor']['last_update']==335
    assert p['source036']['replay_seed']==3601 and p['source036']['already_observed_development_data']
    assert p['birth']['expected_t_on_locked_source']==351 and p['birth']['hardcoded_birth_forbidden']
    assert p['expert']['trainable_parameters']==74 and p['expert']['architecture']=='Linear(73,1)'
    for a in ('D_bias','D_warm'): assert p['arms'][a]['optimizer_state']=='fresh_empty' and p['arms'][a]['trainable_after_birth']=='all_74'
    assert p['runtime']['formal_trigger']=='workflow_dispatch_only' and not p['runtime']['helper_training_allowed']
    assert not p['freeze']['this_turn_launch_authorized']
    print(json.dumps({'protocol':'038','registration_files_valid':True,'plan_sha256':EXPECTED_PLAN_SHA,'instructions_sha256':EXPECTED_DOC_SHA,'scientific_execution_validated':False},indent=2))
if __name__=='__main__': main()
