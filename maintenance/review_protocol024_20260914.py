"""Bounded review diagnostics; preserve all original Protocol-024 outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_dynamic_residual import DynamicResidualBank
from ftmoe_protocol024_eval import temporal_onset_metrics
from ftmoe_protocol024_session import (
    Protocol024Session, export_dynamic_bank, restore_dynamic_bank,
    same_host_onset_metrics,
)
from test_ftmoe_protocol024_dynamic_residual import ReferenceFixedBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-intervals', type=int, default=1140)
    args = parser.parse_args()
    output = ROOT / 'artifacts/ftmoe_online/protocol_024/review_20260914/review_evidence.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite review evidence')
    s4.configure()
    torch.manual_seed(24024)
    source = ReferenceFixedBank()
    result = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip(),
              'formal_D_performance_result': False, 'diagnostics': {}}
    findings = result['diagnostics']
    session = Protocol024Session.__new__(Protocol024Session)
    session.arm = 'D'
    session.update_log = []
    labels = np.zeros((2880, 16), dtype=np.int64)
    labels[301, 0] = 1
    p = np.full(labels.shape, 0.1)
    p[300, 0] = 0.9
    session.predictions = {'labels': labels, 'raw_labels': labels.copy(),
                          'probability': p, 'class_probability': np.ones((2880, 16, 3)) / 3}
    phase = session.phase_metrics()[1]
    findings['evaluator_wiring'] = {
        'phase_metrics_inherited_unchanged': Protocol024Session.phase_metrics is s4.PrequentialS4.phase_metrics,
        'probe_scores_inherited_unchanged': Protocol024Session.probe_scores is s4.PrequentialS4.probe_scores,
        'actual_phase_output_onset': phase['whole']['onset'],
        'correct_same_host_onset': temporal_onset_metrics(p[300:720], labels[300:720]),
        'fixture': 'One onset at t=301, host=0; highest prediction at t=300, host=0'}
    unknown = np.array([[0], [-1], [1]])
    score = np.array([[0.9], [0.8], [0.1]])
    findings['duplicate_onset_unknown_handling'] = {
        'session_helper': same_host_onset_metrics(score, unknown),
        'eval_helper': temporal_onset_metrics(score, unknown)}
    bank = DynamicResidualBank(source)
    rejected = bank.create_shadow('0')
    bank.discard_shadow()
    child = bank.create_shadow('0')
    bank.activate_shadow()
    try:
        restore_dynamic_bank(source, export_dynamic_bank(bank))
        restore_result = {'restored': True}
    except (ValueError, RuntimeError) as exc:
        restore_result = {'restored': False, 'error_type': type(exc).__name__, 'error': str(exc)}
    findings['restore_after_rejection'] = dict(restore_result, rejected_id=rejected,
                                              activated_id=child, topology=bank.topology_manifest())
    bank = DynamicResidualBank(source)
    for _ in range(4):
        child = bank.create_shadow('0')
        bank.activate_shadow()
        bank.ramp[child] = 1.0
    bank.retire(child)
    try:
        bank.create_shadow('0')
        capacity = {'birth_after_retirement_allowed': True}
    except RuntimeError as exc:
        capacity = {'birth_after_retirement_allowed': False, 'error': str(exc)}
    findings['retirement_capacity'] = dict(capacity, active=len(bank.ids),
                                          dormant=len(bank.dormant_experts), max_experts=bank.max_experts)
    bank = DynamicResidualBank(source)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=1e-4)
    child = bank.create_shadow('0')
    with torch.no_grad():
        bank.shadow_experts[child][-1].bias.add_(1.0)
    bank.activate_shadow()
    z = torch.randn(2, 16, 64)
    before = bank(z)[0].detach()
    hash_before = bank.active_state_hash()
    bank.ramp_step()
    after = bank(z)[0].detach()
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    findings['topology_optimizer_and_hash'] = {
        'ramp_changes_output_max_abs': float((after - before).abs().max()),
        'active_hash_unchanged_when_ramp_changes': bank.active_state_hash() == hash_before,
        'new_child_tensors_missing_from_preexisting_optimizer': sum(id(p) not in optimizer_ids for p in bank.experts[child].parameters()),
        'note': 'Lifecycle integration is not implemented; optimizer registration must be added before enabling it.'}
    bank = DynamicResidualBank(source)
    child = bank.create_shadow('0')
    with torch.no_grad():
        bank.shadow_router_weights[child].zero_()
        bank.shadow_router_biases[child].fill_(1000)
    bank.activate_shadow()
    try:
        bank(z)
        extreme = {'forward_ok_at_ramp_zero': True}
    except RuntimeError as exc:
        extreme = {'forward_ok_at_ramp_zero': False, 'error': str(exc)}
    findings['extreme_logit_ramp_zero'] = extreme
    print(json.dumps({'diagnostics': findings}, indent=2), flush=True)
    if args.smoke_intervals:
        from run_ftmoe_protocol024_session_smoke import run
        result['extended_lifecycle_off_smoke'] = run(intervals=args.smoke_intervals)
        with np.load(s4.DEV_STREAM / 'stream.npz') as data:
            raw = data['raw_labels'][:args.smoke_intervals]
            result['smoke_exposure'] = {'raw_fault_host_steps': int((raw > 0).sum()),
                                        'raw_fault_classes': [int(x) for x in np.unique(raw) if x > 0],
                                        'phases': [name for name, start, end, _ in s4.PHASES if start < args.smoke_intervals]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(result.get('extended_lifecycle_off_smoke', {}), indent=2), flush=True)
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
