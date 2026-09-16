"""Strengthened model-free audit for Protocol-025 data_revision_001.

This script is deliberately run after stream generation but before any model
execution.  It validates the one registered data revision semantically rather
than accepting mere key/value metadata equality.  It never imports or evaluates
C/D models and never reads confirmation/test seeds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from simulator.workload.BitbrainWorkloadProtocol025 import (
    DATA_REVISION_ID, SERVICE_IDS, SERVICE_LAWS,
    _periodic_growth_release, s4_cpu_target_fraction,
)

ROOT = Path(__file__).resolve().parent
REG_PATH = ROOT / "artifacts/ftmoe_online/protocol_025/registration.json"
REV_PATH = ROOT / "artifacts/ftmoe_online/protocol_025/data_revision_001.json"


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def _s4_semantics():
    v16=s4_cpu_target_fraction(16)
    v28=s4_cpu_target_fraction(28)
    v40=s4_cpu_target_fraction(40)
    v51=s4_cpu_target_fraction(51)
    expected28=0.55+(1.16-0.55)*0.5
    expected40=0.55+(1.16-0.55)*0.25
    passed=(abs(v16-1.16)<1e-12 and abs(v28-expected28)<1e-12 and
            abs(v40-expected40)<1e-12 and v16>v28>v40>v51>0.55 and
            SERVICE_LAWS['S4'].get('semantic_revision')==DATA_REVISION_ID)
    return {"passed":bool(passed),"fraction_age16":v16,"fraction_age28":v28,
            "fraction_age40":v40,"fraction_age51":v51,"floor":0.55,
            "half_life":12}


def _s5_semantics():
    shape=np.asarray(_periodic_growth_release(108,36,22,6,0.42),dtype=np.float64)
    no_release=np.asarray(_periodic_growth_release(36,36,22,6,0.0),dtype=np.float64)
    # End of every 36-step cycle retains exactly 58% induced excess; a second
    # and third cycle then grow again instead of collapsing to familiar demand.
    passed=(shape.shape==(108,) and shape[0]==0.0 and
            abs(shape[35]-0.58)<1e-12 and abs(shape[36]-0.58)<1e-12 and
            abs(shape[71]-0.58)<1e-12 and abs(shape[72]-0.58)<1e-12 and
            shape[50]>shape[36] and shape[86]>shape[72] and
            abs(no_release[35]-1.0)<1e-12 and
            SERVICE_LAWS['S5'].get('semantic_revision')==DATA_REVISION_ID)
    return {"passed":bool(passed),"period":36,"growth":22,"hold":8,
            "release":6,"release_fraction":0.42,"retained_excess":0.58,
            "cycle1_end":float(shape[35]),"cycle2_start":float(shape[36]),
            "cycle2_end":float(shape[71]),"cycle3_start":float(shape[72]),
            "no_release_cycle1_end":float(no_release[35])}


def audit(data_root):
    data_root=Path(data_root)
    reg=json.loads(REG_PATH.read_text(encoding='utf8'))
    rev=json.loads(REV_PATH.read_text(encoding='utf8'))
    manifest=json.loads((data_root/'manifest.json').read_text(encoding='utf8'))
    old=json.loads((data_root/'data_audit.json').read_text(encoding='utf8'))
    events=json.loads((data_root/'events.json').read_text(encoding='utf8'))
    revision_sha=_sha(REV_PATH); registration_sha=_sha(REG_PATH)

    revision_registered=(reg.get('data_revision',{}).get('used_count')==1 and
        reg['data_revision'].get('active_revision_id')==DATA_REVISION_ID and
        reg['data_revision'].get('pre_revision_stream_eligible_for_model_results') is False and
        reg['data_revision'].get('no_further_data_revision_allowed') is True and
        rev.get('revision_number')==1 and
        rev.get('registered_before_any_protocol025_model_result') is True and
        rev.get('uses_only_allowed_data_revision') is True)

    with np.load(data_root/'stream.npz') as d:
        raw=np.asarray(d['raw_labels'],dtype=np.int64)
        ratio=np.asarray(d['overload_ratio'],dtype=np.float64)
        recompute=np.where((ratio>1.0).any(-1),ratio.argmax(-1)+1,0)
        label_rule=bool(np.array_equal(raw,recompute))
        stream_shape=tuple(raw.shape)
    common=np.load(data_root/'common_observable_features.npz')['features']
    common_finite=bool(common.shape==(5521,16,9) and np.isfinite(common).all())

    # Exact guard geometry used later by the frozen runner: F0 positions
    # 0,5,...,295; target is raw fault at t+1 and remains wholly inside F0.
    guard_indices=np.arange(0,299,dtype=np.int64)
    guard_indices=guard_indices[guard_indices%5==0]
    guard_y=raw[guard_indices+1].reshape(-1)
    guard_normal=int((guard_y==0).sum()); guard_positive=int((guard_y>0).sum())
    f0_guard_ok=guard_normal>0 and guard_positive>0

    recurrence=old.get('recurrence_first100_class_coverage',{})
    rec_names=['S1_rec1','S3_rec1','S2_rec1','S4_rec1','S2_rec2',
               'S6_rec1','S1_rec2','S5_rec1','S3_rec2']
    recurrence_ok=(set(recurrence)==set(rec_names) and
                   all(bool(recurrence[x].get('ap_defined')) for x in rec_names))
    phases=old.get('per_phase',{})
    service_phase_ok=all(
        int(row.get('positive_host_steps',0))>0 and int(row.get('negative_host_steps',0))>0
        for name,row in phases.items() if name!='F0')

    event_counts={s:sum(1 for e in events if e.get('service_id')==s) for s in SERVICE_IDS}
    events_every_service=all(event_counts[s]>0 for s in SERVICE_IDS)
    event_revision_ok=all(
        e.get('family')=='protocol025_service_turnover_v1r1' and
        e.get('data_revision_id')==DATA_REVISION_ID for e in events)
    s5_windows=[int(e['ram_window'][1])+1 for e in events if e.get('service_id')=='S5']
    s5_dynamic_event_windows=bool(s5_windows and min(s5_windows)>=36)

    s4=_s4_semantics(); s5=_s5_semantics()
    old_gates=dict(old.get('gates',{}))
    # The old audit is retained as evidence, but semantic revision gates below
    # are authoritative for revision-1.
    gates={
        'revision_registered_before_model_results':bool(revision_registered),
        'pre_revision_stream_ineligible':reg.get('data_revision',{}).get('pre_revision_stream_eligible_for_model_results') is False,
        'stream_shape_5521x16':stream_shape==(5521,16),
        'label_rule_physical_capacity_recompute':label_rule,
        'common_observable_features_finite_5521x16x9':common_finite,
        'events_every_service':events_every_service,
        'event_revision_provenance':event_revision_ok,
        'S4_registered_floor_half_life_effective':bool(s4['passed']),
        'S5_periodic_release_fraction_effective':bool(s5['passed']),
        'S5_event_windows_respect_minimum_trace':s5_dynamic_event_windows,
        'all_service_phases_have_positive_and_negative':service_phase_ok,
        'all_nine_recurrence_first100_ap_defined':recurrence_ok,
        'F0_guard_has_positive_and_negative':bool(f0_guard_ok),
    }
    audit_pass=bool(all(gates.values()))
    augmented=dict(old)
    augmented.update({
        'protocol':'025','kind':'pre_model_data_audit_revision1',
        'data_revision_id':DATA_REVISION_ID,
        'data_revision_sha256':revision_sha,
        'registration_sha256':registration_sha,
        'pre_revision_generation_run':int(reg['data_revision']['pre_revision_generation_run']),
        'pre_revision_stream_eligible_for_model_results':False,
        'model_results_seen':False,
        'semantic_checks':{'S4':s4,'S5':s5},
        'F0_guard':{'indices':guard_indices.tolist(),'normal_rows':guard_normal,
                    'positive_rows':guard_positive,'target':'raw_next_fault'},
        'service_event_counts':event_counts,
        'S5_event_ram_window_lengths':s5_windows,
        'legacy_pre_revision_style_gates':old_gates,
        'gates':gates,
        'audit_pass':audit_pass,
        'revision_allowed_if_failed':0,
        'no_further_data_revision_allowed':True,
    })
    (data_root/'data_audit.json').write_text(
        json.dumps(augmented,indent=2,allow_nan=False)+'\n',encoding='utf8')
    manifest.update({
        'data_revision_id':DATA_REVISION_ID,
        'data_revision_sha256':revision_sha,
        'registration_sha256':registration_sha,
        'pre_revision_generation_run':int(reg['data_revision']['pre_revision_generation_run']),
        'pre_revision_stream_eligible_for_model_results':False,
        'audit_pass':audit_pass,
        'audit_kind':'pre_model_data_audit_revision1',
    })
    (data_root/'manifest.json').write_text(
        json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (data_root/'data_revision_snapshot.json').write_bytes(REV_PATH.read_bytes())
    result={'audit_pass':audit_pass,'gates':gates,'stream_sha256':manifest['stream_sha256'],
            'data_revision_id':DATA_REVISION_ID,'F0_guard':augmented['F0_guard'],
            'semantic_checks':augmented['semantic_checks']}
    print(json.dumps(result,indent=2,allow_nan=False))
    if not audit_pass:
        raise SystemExit(2)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True)
    args=p.parse_args(); audit(args.data_root)
