"""Pre-science engineering fixture for Protocol-034 optimizer/freeze/observer rules."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
from run_ftmoe_protocol033 import validate_input
from ftmoe_protocol033_guard_budget import P033_CONFIG
from run_ftmoe_protocol034_live import InstrumentedProtocol034Session, PredictionMemoryObserver
from ftmoe_protocol034_live_memory import GENERALISTS, active_specialist_hash, dormant_state_hash, optimizer_names


def build(stream,registration,input_lock,out,run_id,observer=True):
    Path(out).mkdir(parents=True,exist_ok=True)
    stream_sha,manifest,lock,reg=validate_input(stream,registration,input_lock)
    bundle=s4.build_replay(Path(stream)); pdefs=p31.phases(manifest); guard=p31.build_guard(bundle)
    guard['meta'].update({'protocol':'034','normal_nll_rule':'candidate <= live + max(0.02*live,0.01) + 1e-6'})
    common=dict(seed=1,replay_bundle=bundle,budget=p31.budget(),out_dir=Path(out),run_id=run_id,
                stream_dir=Path(stream),phase_defs=pdefs,stream_sha=stream_sha,
                registration={'protocol':'034','fixture':True,'stream_sha256':stream_sha},learning_rate=1e-4)
    s=InstrumentedProtocol034Session(guard_anchor=guard,v2c_config=P033_CONFIG,**common)
    if observer:
        obs=PredictionMemoryObserver(s.steps,int(s.predictions['probability'].shape[1]),pdefs,s.model.frozen_hash())
        s.attach_snapshot_observer(obs)
    return s

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stream',required=True); ap.add_argument('--registration',required=True)
    ap.add_argument('--input-lock',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    p31.deterministic_runtime(); root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    s=build(a.stream,a.registration,a.input_lock,root/'session','protocol034_fixture',observer=True)
    while s.cursor<1100 and s.lifecycle_controller.active_specialist_id is None: s.step()
    while s.cursor<1100 and s.lifecycle_controller.transition is not None: s.step()
    active=s.lifecycle_controller.active_specialist_id
    if active is None: raise AssertionError('fixture did not obtain first stable active specialist')
    active=str(active); names=optimizer_names(s)
    if not any(x.startswith(active+'|') for x in names): raise AssertionError('stable active specialist absent from live optimizer')
    if not all(any(x.startswith(g+'|') for x in names) for g in GENERALISTS): raise AssertionError('generalist missing from live optimizer')
    before=active_specialist_hash(s); updates=s.updates
    while s.updates==updates and s.cursor<1150: s.step()
    after=active_specialist_hash(s)
    if before==after: raise AssertionError('stable active specialist did not learn on common update')
    if not s.snapshot_observer.isolation_checks: raise AssertionError('observer isolation check did not execute')
    s.retire_expert(active); d0=dormant_state_hash(s); s.update(s.cursor); d1=dormant_state_hash(s)
    if d0!=d1: raise AssertionError('dormant specialist changed under a common update')
    s.reactivate_expert(active); s.lifecycle_controller.active_specialist_id=active; s._apply_specialist_freeze()
    bank=s.model.learner; new=bank.create_shadow(active); s.create_shadow_optimizer(); loss=None
    for p in bank.shadow_parameters():
        term=(p*p).mean(); loss=term if loss is None else loss+term
    s.shadow_optimizer.zero_grad(set_to_none=True); loss.backward(); s.shadow_optimizer.step(); s.model.zero_grad(set_to_none=True)
    new2=s.activate_shadow()
    if str(new2)!=str(new): raise AssertionError('fixture shadow identity changed')
    transfer=s.optimizer_transfer_events[-1]
    if transfer['transferred_state_count']<=0: raise AssertionError('shadow AdamW state was not transferred')
    s.lifecycle_controller._begin_transition(s,active,str(new),'fixture_crossfade')
    trans_names=optimizer_names(s)
    if any(x.startswith(active+'|') or x.startswith(str(new)+'|') for x in trans_names):
        raise AssertionError('crossfade specialist entered common optimizer')
    for _ in range(int(P033_CONFIG["crossfade_prediction_intervals"])): s.step()
    if s.lifecycle_controller.transition is not None: raise AssertionError('fixture crossfade did not finish')
    if str(s.lifecycle_controller.active_specialist_id)!=str(new): raise AssertionError('new specialist not active after crossfade')
    if not any(x.startswith(str(new)+'|') for x in optimizer_names(s)): raise AssertionError('new stable specialist did not resume training')
    if active not in s.model.learner.dormant_experts: raise AssertionError('old specialist not dormant after crossfade')
    ck=s.save_checkpoint('p034_fixture_roundtrip',s.cursor-1); hash0=s.learner_state_hash(); dorm0=dormant_state_hash(s); names0=optimizer_names(s)
    r=build(a.stream,a.registration,a.input_lock,root/'restore','protocol034_fixture',observer=False); r.restore_checkpoint(ck['path'])
    if r.learner_state_hash()!=hash0 or dormant_state_hash(r)!=dorm0 or optimizer_names(r)!=names0:
        raise AssertionError('Protocol034 checkpoint roundtrip mismatch')
    p0,_=s.step(); p1,_=r.step()
    if not np.allclose(p0,p1,atol=2e-6,rtol=2e-6): raise AssertionError('checkpoint-restored next prediction mismatch')
    report={'protocol':'034','fixture_passed':True,'natural_first_specialist_id':active,
            'active_specialist_changed_on_common_update':True,'dormant_hash_unchanged_on_common_update':True,
            'crossfade_specialists_excluded_from_live_optimizer':True,
            'crossfade_intervals':int(P033_CONFIG["crossfade_prediction_intervals"]),
            'shadow_optimizer_state_transferred':True,'shadow_transfer_state_count':transfer['transferred_state_count'],
            'observer_isolation_check_passed':True,'checkpoint_roundtrip_passed':True,
            'next_prediction_after_restore_within_reference_tolerance':True,'scientific_full_replay_consumed':False}
    (root/'fixture_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
