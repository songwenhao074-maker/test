"""Protocol-025 engineering-only fixed4 vs dynamic-container equivalence smoke.

Uses the existing Protocol-023 development replay only as an engineering input.
It is NOT a Protocol-025 model result and never reads Protocol-025 generated
labels. Twenty-four intervals are far before the registered first proposal at
matured_count=600, so lifecycle topology must remain untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol025_session import Protocol025FixedSession, Protocol025DynamicSession


def _budget():
    out=dict(json.loads(s4.BUDGET_FILE.read_text(encoding='utf8'))['frozen_configuration'])
    out.update({'update_every_scored_intervals':4,'batch_size':32,'gradient_steps_per_opportunity':1,'replay_buffer_intervals':64,'learning_rate':1e-4})
    return out


def _engineering_phases(steps):
    """One metadata-only phase for the legacy P23 replay used by this smoke."""
    return [{'name':'engineering_only','start':0,'end':int(steps),
             'regime':'legacy_p23_replay'}]


def _engineering_guard(bundle):
    """Structurally valid guard that this <600-step smoke must never consult.

    v2c requires a non-null guard at construction.  The smoke deliberately runs
    only 24 intervals, while the frozen first proposal is at matured_count=600,
    so no qualification path may read this object.  Using one real legacy P23
    row keeps the tensor contract valid without touching Protocol-025 data.
    """
    index=0
    x,s,g,context=s4.window_batch(bundle['replay'],[index])
    labels=torch.as_tensor(
        np.asarray(bundle['arrays']['raw_labels'][index+1:index+2],dtype=np.int64),
        dtype=torch.long)
    return {
        'x':x,'schedule':s,'graph_x':g,'labels':labels,
        'ids':context['creation_ids'],'before':context['before_placement'],
        'caps':context['capacities'],
        'meta':{'engineering_only':True,'formal_protocol025_guard':False,
                'must_never_be_consulted_before_matured_600':True},
    }


def run(intervals=24):
    if int(intervals)>=600:
        raise ValueError('engineering smoke must stay below the first lifecycle proposal')
    bundle=s4.build_replay(s4.DEV_STREAM); sha=bundle['manifest']['stream_sha256']
    registration={'protocol':'025','kind':'engineering_equivalence_smoke','formal_protocol025_result':False,'service_ids_used':False}
    with tempfile.TemporaryDirectory(prefix='p25_equiv_') as td:
        root=Path(td)
        common=dict(seed=1,replay_bundle=bundle,budget=_budget(),run_id='p25_engineering_smoke',stream_dir=s4.DEV_STREAM,phase_defs=_engineering_phases(bundle['steps']),stream_sha=sha,registration=registration,learning_rate=1e-4)
        fixed=Protocol025FixedSession('C_fixed4',out_dir=root/'fixed',**common)
        dynamic=Protocol025DynamicSession(out_dir=root/'dynamic',guard_anchor=_engineering_guard(bundle),**common)
        pmax=cmax=0.0
        for _ in range(int(intervals)):
            pf,cf=fixed.step(); pd,cd=dynamic.step()
            pmax=max(pmax,float(np.max(np.abs(pf-pd)))); cmax=max(cmax,float(np.max(np.abs(cf-cd))))
            if not np.allclose(pf,pd,atol=3e-6,rtol=3e-6): raise AssertionError('P25 fixed4/dynamic detection diverged before lifecycle')
            if not np.allclose(cf,cd,atol=3e-6,rtol=3e-6): raise AssertionError('P25 fixed4/dynamic class diverged before lifecycle')
        topo=dynamic.model.learner.topology_manifest()
        checks={
            'prediction_equivalent':True,
            'updates_equal':fixed.updates==dynamic.updates,
            'topology_untouched':topo['active_ids']==['0','1','2','3'] and not topo['dormant_ids'] and topo['shadow_id'] is None,
            'frozen_base_unchanged':fixed.model.frozen_hash()==fixed.frozen_hash and dynamic.model.frozen_hash()==dynamic.frozen_hash,
            'no_lifecycle_event':len(dynamic.lifecycle_controller.events)==0,
            'below_first_lifecycle_proposal':int(intervals)<600,
        }
        result={'protocol':'025','kind':'engineering_equivalence_smoke','formal_protocol025_result':False,'intervals':int(intervals),'updates_fixed':fixed.updates,'updates_dynamic':dynamic.updates,'probability_max_abs':pmax,'class_probability_max_abs':cmax,'topology':topo,'checks':checks,'passed':all(checks.values())}
        if not result['passed']: raise AssertionError(result)
        return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
