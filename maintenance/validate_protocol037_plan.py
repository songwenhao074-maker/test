"""Validate frozen Protocol037 registration files; does not certify execution."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PLAN_SHA='01240a944bb654d529577b397602cf67c24c2245c6fbe909564b47da2726012a'
EXPECTED_DOC_SHA='01bb7cad5abb517afe52d01843735cd719c67ad6cc4e80aa633aca20b1f3caf3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pp=ROOT/'artifacts/ftmoe_online/protocol_037/plan.json'; p=json.loads(pp.read_text(encoding='utf8'))
 assert sha(pp)==EXPECTED_PLAN_SHA==(pp.parent/'plan.sha256').read_text().split()[0]
 assert p['instructions_sha256']==EXPECTED_DOC_SHA==sha(ROOT/p['instructions'])
 assert p['protocol']=='037' and p['revision']==1
 assert p['code_base_commit']=='876172e1a16ebcaa321f3ca713bb06ee6ba31b87'
 assert p['source']['replay_seed']==3601 and p['source']['already_observed_development_data']
 b=p['budget']; assert b['new_training_sequence_slots']==2 and b['sequence_names']==['F_extra','D_birth']
 for k in ('new_raw_streams','baseline_retraining','other_real_stream_evaluations','hyperparameter_or_trigger_sweeps','sleep_wake_delete_trials'):assert b[k]==0
 assert b['max_births']==b['max_extra_experts']==1
 assert b['real_joint_prefix_executions_max']==2 and b['real_prefix_steps_max']==256
 assert b['automatic_next_protocol'] is False and b['restart_from_zero_after_start'] is False
 assert p['expert']['trainable_parameters']==74 and p['expert']['architecture']=='Linear(73,1)'
 assert p['baseline']['logits_source']=='stage_B/seed3601/D_lin/predictions.npz'
 t=p['birth'];assert t['min_matured_intervals']==t['recent_interval_count']+t['previous_interval_count']==320
 assert t['loss_ratio_min']==1.25 and t['absolute_loss_increase_min']==0.02 and t['consecutive_eligible_checks']==2
 assert t['max_events']==1 and p['expert']['update_every']==16
 assert p['causality']['event_order']==['predict','settle_label','check_birth_if_due','scheduled_update']
 assert p['freeze']['this_turn_launch_authorized'] is False and p['freeze']['future_roadmap_is_not_execution_authorization'] is True
 print(json.dumps({'protocol':'037','registration_files_valid':True,'plan_sha256':EXPECTED_PLAN_SHA,'instructions_sha256':EXPECTED_DOC_SHA,'scientific_execution_validated':False},indent=2))
if __name__=='__main__':main()
