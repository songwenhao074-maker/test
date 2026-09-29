"""Enforce Protocol-035 C_ref_035 reproduction against frozen Protocol-034 C_ref."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from protocol035_common import dump_json, load_npz, grouped_average_precision, WINDOWS


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--c-ref-dir',required=True); ap.add_argument('--reference-c-dir',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); c=load_npz(Path(a.c_ref_dir)/'predictions.npz'); r=load_npz(Path(a.reference_c_dir)/'predictions.npz')
    exact_labels=bool(np.array_equal(c['labels'],r['labels'])); exact_raw=bool(np.array_equal(c['raw_labels'],r['raw_labels']))
    prob=np.allclose(c['probability'],r['probability'],atol=2e-6,rtol=2e-6,equal_nan=False)
    cls=np.allclose(c['class_probability'],r['class_probability'],atol=2e-6,rtol=2e-6,equal_nan=False)
    maxp=float(np.max(np.abs(c['probability'].astype(np.float64)-r['probability'].astype(np.float64))))
    maxc=float(np.max(np.abs(c['class_probability'].astype(np.float64)-r['class_probability'].astype(np.float64))))
    maxlog=float(np.max(np.abs(c['detection_logits'].astype(np.float64)-r['detection_logits'].astype(np.float64))))
    aps=[]; ap_ok=True
    for name,s,e in WINDOWS:
        ac=grouped_average_precision(c['labels'][s:e],c['probability'][s:e]); ar=grouped_average_precision(r['labels'][s:e],r['probability'][s:e])
        d=abs(ac-ar) if ac is not None and ar is not None else None
        ok=(d is not None and d<=1e-4); ap_ok=ap_ok and ok
        aps.append({'window':name,'C_ref_035_AP':ac,'Protocol034_C_ref_AP':ar,'abs_delta':d,'pass':ok})
    init_new=json.loads((Path(a.c_ref_dir)/'initialization.json').read_text())
    init_old=json.loads((Path(a.reference_c_dir)/'initialization.json').read_text())
    old_first4=init_old.get('shared_first4_expert_and_router_rows_sha256') or init_old.get('shared_first4_expert_and_router_rows_sha256'.replace('_and_','_'))
    first4=init_new['shared_first4_expert_and_router_rows_sha256']==old_first4
    all5_historical=init_old.get('all5_learner_state_sha256')
    out={'protocol':'035','reference':'Protocol034 C_ref','exact_labels':exact_labels,'exact_raw_labels':exact_raw,
         'probability_allclose_atol_2e-6_rtol_2e-6':bool(prob),'class_probability_allclose_atol_2e-6_rtol_2e-6':bool(cls),
         'probability_max_abs':maxp,'class_probability_max_abs':maxc,'detection_logit_max_abs_diagnostic_only':maxlog,
         'recurrence_AP_checks':aps,'all_recurrence_AP_within_1e-4':bool(ap_ok),'first4_initialization_exact':bool(first4),
         'new_all5_initialization_sha256':init_new['all5_learner_state_sha256'],'historical_all5_hash_present':all5_historical is not None,
         'historical_direct_initialization_comparison_scope':'first4 exact; all5 newly recorded and constructor fixture checked',
         'all_reference_checks_passed':bool(exact_labels and exact_raw and prob and cls and ap_ok and first4),
         'logits_are_diagnostic_not_gate':True}
    dump_json(a.output,out); print(json.dumps(out,indent=2))
    if not out['all_reference_checks_passed']: raise SystemExit(3)
if __name__=='__main__': main()
