from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/ftmoe_online/protocol_036/plan.json'
PLAN_SHA = ROOT / 'artifacts/ftmoe_online/protocol_036/plan.sha256'
INSTRUCTIONS = ROOT / 'docs/PROTOCOL036_CORRECTION_CONTROLS_AND_NEW_STREAMS_20260930.md'
WORKFLOW = ROOT / '.github/workflows/protocol036-correction-controls.yml'
EXPECTED_PLAN_SHA = '339980818a4d0d7518b6a33e696bcfa02ea3ec27d69467d4242264277c051a47'
EXPECTED_INSTRUCTIONS_SHA = 'b68ca17324c37d52718c9b894594cec01aa7b225c39d86ec8293e51de02bbc3f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    got = sha(PLAN)
    file_sha = PLAN_SHA.read_text(encoding='utf8').split()[0]
    assert got == file_sha == EXPECTED_PLAN_SHA, (got, file_sha)
    p = json.loads(PLAN.read_text(encoding='utf8'))
    assert p['protocol'] == '036' and p['revision'] == 1
    assert p['code_base_commit'] == '84282143b849dd3b5684e7f879964a2ab485fab0'
    assert p['execution_branch'] == 'codex/protocol-036-correction-controls-20260930'
    b = p['budget']
    assert b['new_full_training_sequences'] == 14
    assert b['stage_A_cached_700_sequences'] == ['D_cal', 'D_lin']
    assert b['stage_B_new_stream_seeds'] == [3601, 3602, 3603]
    assert b['stage_B_sequences_per_stream'] == ['C_ref', 'D_cal', 'D_lin', 'D_corr']
    assert b['new_raw_streams'] == 3
    assert b['engineering_real_prefix_executions_max'] == 2
    assert b['engineering_prefix_steps_max'] == 256
    assert b['hyperparameter_sweeps'] == 0 and b['extra_model_seeds'] == 0
    assert b['old_D_retraining'] == 0 and b['cache_router_trials'] == 0
    assert b['automatic_followup'] is False and b['restart_from_zero_after_scientific_start'] is False
    d = p['data']
    assert d['scored_intervals'] == 5968 and d['raw_intervals'] == 5969 and d['hosts'] == 16
    assert d['feature_dimension'] == 73 and d['event_probability'] == 0.3
    assert d['model_seed'] == 1 and d['branch_seed'] == 3501
    assert d['prediction_label_rule'] == 'i+2<t' and d['update_label_rule'] == 'i+2<=t'
    assert d['future_statistic_preprocessing_prohibited'] is True
    assert p['arms']['D_cal']['parameters'] == 2
    assert p['arms']['D_lin']['parameters'] == 74
    assert p['arms']['D_corr']['parameters'] == 2401
    tr = p['branch_training']
    assert tr['lr'] == 1e-4 and tr['weight_decay'] == 1e-4
    assert tr['update_every'] == 16 and tr['replay_recent_matured_intervals'] == 64 and tr['max_batch_intervals'] == 32
    assert tr['checkpoint_every_predictions'] == 512 and tr['real_resume_required'] is True
    rt = p['runtime']
    assert rt['formal_workflow'] == 'protocol036-correction-controls.yml'
    assert rt['formal_trigger'] == 'workflow_dispatch_only'
    assert rt['connector_only_one_shot_push_dispatch_helper_allowed'] is True
    assert rt['helper_training_allowed'] is False and rt['helper_automatic_dispatch_retry'] is False
    assert rt['exact_resume_only'] is True and rt['workflow_and_checkout_SHA_separate'] is True
    assert p['freeze']['both_stages_and_analysis_frozen_before_any_036_results'] is True
    assert p['freeze']['stage_A_driven_changes_to_stage_B'] is False
    assert p['instructions_sha256'] == EXPECTED_INSTRUCTIONS_SHA == sha(INSTRUCTIONS)
    if WORKFLOW.is_file():
        text = WORKFLOW.read_text(encoding='utf8')
        on_block = text.split('permissions:', 1)[0]
        assert 'workflow_dispatch:' in on_block
        assert '\n  push:' not in on_block and '\n  workflow_run:' not in on_block and '\n  schedule:' not in on_block
    print(json.dumps({'protocol':'036','plan_sha256':got,'instructions_sha256':sha(INSTRUCTIONS),'budget_sequences':14,'new_streams':3,'validated':True}, indent=2))


if __name__ == '__main__':
    main()
