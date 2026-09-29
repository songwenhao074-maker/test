"""Validate the frozen Protocol035 plan only. Does not run or validate an experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re

EXPECTED_DIGEST = '410f2d07992f7cb71e04701fd2036bbf753f11aecb302257a817c73a72ae4767'
EXPECTED_WINDOWS = [
    ['U_rec1', 3300, 3428], ['V_rec1', 3808, 3936],
    ['U_rec2', 4316, 4444], ['V_rec2', 4824, 4952],
    ['U_rec3', 5332, 5460], ['V_rec3', 5840, 5968],
]

def validate(plan):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(plan['protocol'] == '035' and plan['revision'] == 1, 'Wrong protocol/revision')
    check(plan['status'] == 'planned_not_implemented_not_run', 'Registration is immutable; write execution status in a run directory')
    for name in ('input_033', 'evidence_034'):
        check(re.fullmatch(r'[0-9a-f]{64}', plan['source'][name]['zip_sha256']) is not None, 'Invalid artifact hash: ' + name)
    check(re.fullmatch(r'[0-9a-f]{64}', plan['source']['input_033']['stream_sha256']) is not None, 'Invalid stream hash')
    check(plan['source']['execution_branch'] == 'codex/protocol-035-c-preserving-correction-20260929', 'Unexpected execution branch')
    check(plan['input']['total_rows'] == plan['input']['scored_rows'] + 1 == 5969, 'Invalid row support')
    check(plan['input']['hosts'] == 16 and plan['input']['feature_dimensions'] == 73, 'Unexpected input dimensions')
    b = plan['budget']
    check(b['full_training_sequences'] == b['C_reference_replays'] + b['correction_training_replays'] == 2, 'Training budget mismatch')
    check(b['training_sequence_names'] == ['C_ref_035', 'D_corr_branch'], 'Unexpected training arms')
    for key in ('new_simulator_streams', 'extra_seeds', 'hyperparameter_sweeps', 'old_D_training_replays', 'additional_cache_router_trials'):
        check(b[key] == 0, 'Unregistered budget: ' + key)
    check(b['cache_diagnostic_suites'] == 1 and b['cache_diagnostic_parameter_fitting'] is False, 'Diagnostic scope changed')
    check(b['restart_from_zero_after_scientific_start_allowed'] is False and b['automatic_followup_training'] is False, 'Forbidden restart/followup')
    check(b['engineering_fixture_max_real_stream_prefix'] == 256 and b['engineering_fixture_max_real_stream_executions'] == 2, 'Fixture budget changed')
    c = plan['correction']
    check(c['architecture'] == [{'type':'Linear','in_features':73,'out_features':32,'bias':True}, {'type':'ReLU'}, {'type':'Linear','in_features':32,'out_features':1,'bias':True}], 'Unregistered architecture')
    check(c['trainable_parameters'] == (73 + 1)*32 + (32 + 1) == 2401, 'Parameter count mismatch')
    check(c['initialization_seed'] == 3501 and c['output_initialization'] == 'weight_and_bias_zero', 'Initialization changed')
    check(c['delta'] == '2*tanh(raw_branch_output/2)' and c['delta_penalty'] == 0.001, 'Correction scale/loss changed')
    for key in ('loss_gradient_to_C_or_z', 'C_optimizer_shared_with_branch', 'future_C_weights_for_replay', 'dynamic_birth', 'memory_expert_bank', 'hard_deletion', 'extra_router', 'normalization_updates', 'dropout', 'AMP'):
        check(c[key] is False, 'Forbidden component/feedback: ' + key)
    for arm in (plan['reference'], c):
        check(arm['update_every'] == 16 and arm['gradient_steps_per_opportunity'] == 1, 'Update cadence mismatch')
        check(arm['replay_intervals'] == 64 and arm['batch_intervals'] == 32, 'Replay/batch mismatch')
        check(arm['learning_rate'] == 1e-4 and arm['weight_decay'] == 1e-4 and arm['gradient_clip_norm'] == 1.0, 'Optimizer scope changed')
    e = plan['evaluation']
    check(e['windows'] == EXPECTED_WINDOWS and all(z-y == 128 for _, y, z in e['windows']), 'Recurrence windows changed')
    check(e['threshold'] == 0.5 and e['secondary_cells_cannot_determine_success'] is True, 'Evaluation selection changed')
    check(len(e['two_primary_axes']) == 2 and e['independent_confirmation'] is False, 'Evidence scope changed')
    check(plan['diagnosis']['hindsight_diagnostic'] is True and plan['diagnosis']['online_result'] is False, 'Hindsight mislabelled')
    check(plan['runtime']['workflow_trigger'] == 'workflow_dispatch_only' and plan['runtime']['dispatch_in_plan_publication'] is False, 'Dispatch scope changed')
    canonical = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
    digest = hashlib.sha256(canonical).hexdigest()
    check(digest == EXPECTED_DIGEST, 'Frozen revision1 differs from the published registration; do not silently edit the scientific design')
    return errors, digest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/ftmoe_online/protocol_035/plan.json')
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding='utf8'))
        errors, digest = validate(plan)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors, digest = ['Invalid plan: ' + str(exc)], None
    print(json.dumps({'protocol':'035','plan_valid':not errors,'canonical_plan_sha256':digest,
                      'algorithm_implementation_checked':False,'experiment_execution_checked':False,
                      'results_validated':False,'errors':errors}, indent=2))
    raise SystemExit(1 if errors else 0)

if __name__ == '__main__':
    main()
