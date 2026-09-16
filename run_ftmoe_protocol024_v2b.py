"""Run Protocol-024 next_round_v2b memory replay on immutable seed700."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_anchor import anchor_metadata, load_protocol024_raw_next_anchor
from ftmoe_protocol024_eval import binary_detection_metrics, positive_resource_macro_f1, temporal_onset_metrics
from ftmoe_protocol024_session import NEXT_TARGET_MODE
from ftmoe_protocol024_v2a import split_anchor_train_guard
from ftmoe_protocol024_v2b import V2BProtocol024Session, V2B_DEFAULT

EXPECTED_STREAM_SHA="468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946"
FIRST_NAMES=("R1_first","R2_first","R3_first")
RECURRENCE_NAMES=("R1_rec1","R3_rec1","R2_rec1","R1_rec2","R2_rec2","R3_rec2")


def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf8")

def write_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf8") as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+"\n")

def phase_defs(manifest):
    return [{"name":p["name"],"start":int(p["start"]),"end":int(p["end"]),"response_law":p.get("response_law")} for p in manifest["timeline"]]

def budget():
    out=dict(json.loads(s4.BUDGET_FILE.read_text(encoding="utf8"))["frozen_configuration"])
    out.update({"update_every_scored_intervals":4,"batch_size":32,"gradient_steps_per_opportunity":1,"replay_buffer_intervals":64,"learning_rate":1e-4})
    return out

def npz_dict(path):
    with np.load(path) as d: return {k:d[k] for k in d.files}

def model_metrics(prob,cls,labels,raw):
    return {"detection":binary_detection_metrics(prob,labels),"onset":temporal_onset_metrics(prob,raw,1),"resource":positive_resource_macro_f1(cls,labels)}

def run(stream_dir,v2a_root,out_root):
    stream_dir=Path(stream_dir); v2a_root=Path(v2a_root); out_root=Path(out_root)
    if out_root.exists(): raise FileExistsError("refusing to overwrite v2b run %s"%out_root)
    out_root.mkdir(parents=True)
    bundle=s4.build_replay(stream_dir)
    if bundle["steps"]!=4980 or bundle["manifest"].get("stream_sha256")!=EXPECTED_STREAM_SHA: raise AssertionError("immutable stream mismatch")
    v2a_status=json.loads((v2a_root/"status.json").read_text())
    if not v2a_status.get("advance_to_v2b") or not v2a_status.get("v2a_any_candidate_accepted"): raise RuntimeError("v2a artifact does not authorize v2b")
    v2a_run=v2a_root/"runs/v2a_buffer128_seed700_model1"
    c=npz_dict(v2a_run/"arm_C/predictions.npz")
    v2a_cmp=json.loads((v2a_run/"comparison.json").read_text())
    anchor=load_protocol024_raw_next_anchor(); train_anchor,guard_anchor=split_anchor_train_guard(anchor,modulo=5,guard_remainder=0)
    cfg=dict(V2B_DEFAULT)
    run_id="seed700_model1_v2b_memory_buffer128_4x4"
    registration={
        "protocol":"024","round":"next_round_v2","stage":"v2b","run_id":run_id,
        "replay_seed":700,"model_seed":1,"target":NEXT_TARGET_MODE,"update_every":4,
        "birth_shadow_budget":"buffer128_4x4","v2a_source_run":34989082187,
        "v2a_source_artifact":10405894781,"confirmation_run":False,
        "generalist_ids":["0","1","2","3"],"specialist_frozen":True,
        "max_live_specialists":1,"resident_limit":8,"reuse_every_matured":32,
        "reuse_recent_z_intervals":32,"reuse_validation_intervals":16,
        "reuse_match":"max cosine to frozen centroid; candidate requires >= its causal q10 within-memory similarity",
        "crossfade_prediction_intervals":8,"purge_rule":"oldest last causal acceptance then expert ID",
        "anchor":anchor_metadata(train_anchor),"guard":anchor_metadata(guard_anchor),
    }
    session=V2BProtocol024Session(
        "D",1,bundle,budget(),out_root/"arm_D_v2b",anchor=train_anchor,
        guard_anchor=guard_anchor,v2b_config=cfg,learning_rate=1e-4,max_experts=8,
        run_id=run_id,stream_dir=stream_dir,phase_defs=phase_defs(bundle["manifest"]),
        target_mode=NEXT_TARGET_MODE,stream_sha=EXPECTED_STREAM_SHA,registration=registration)
    started=time.perf_counter()
    for _ in range(session.steps): session.step()
    session.finish(); elapsed=time.perf_counter()-started; session.save()
    ctrl=session.lifecycle_controller; events=list(ctrl.events); counts=dict(Counter(x["kind"] for x in events))
    write_jsonl(out_root/"arm_D_v2b/lifecycle.jsonl",events)
    dprob=session.predictions["probability"]; dcls=session.predictions["class_probability"]
    labels=session.predictions["labels"]; raw=session.predictions["raw_labels"]
    if not np.array_equal(labels,c["labels"]): raise AssertionError("C and v2b target rows differ")
    phases={p["name"]:p for p in phase_defs(bundle["manifest"])}
    switches=[]
    for name in FIRST_NAMES+RECURRENCE_NAMES:
        p=phases[name]; start,end=p["start"],min(p["start"]+100,p["end"])
        cm=binary_detection_metrics(c["probability"][start:end],labels[start:end])
        dm=binary_detection_metrics(dprob[start:end],labels[start:end])
        delta=None if cm["ap"] is None or dm["ap"] is None else float(dm["ap"]-cm["ap"])
        switches.append({"phase":name,"response_law":p.get("response_law"),"intervals":[start,end],"C":cm,"D_v2b":dm,"D_v2b_minus_C_ap":delta})
    recurrence=[x for x in switches if x["phase"] in RECURRENCE_NAMES]
    deltas=[x["D_v2b_minus_C_ap"] for x in recurrence if x["D_v2b_minus_C_ap"] is not None]
    idx=np.concatenate([np.arange(x["intervals"][0],x["intervals"][1]) for x in recurrence])
    cm=binary_detection_metrics(c["probability"][idx],labels[idx]); dm=binary_detection_metrics(dprob[idx],labels[idx])
    fpr_delta=None if cm["fpr"] is None or dm["fpr"] is None else float(dm["fpr"]-cm["fpr"])
    mean_delta=float(np.mean(deltas)) if deltas else None; positive=int(sum(x>0 for x in deltas))
    signal=bool(len(deltas)==6 and mean_delta>=0.03 and positive>=4 and fpr_delta is not None and fpr_delta<=0.01)
    lifecycle={
        "event_counts":counts,"births":int(counts.get("candidate_accepted",0)),
        "retirements":int(ctrl.retirements),"reactivations":int(ctrl.reactivations),
        "purges":int(ctrl.purges),"purged_bytes":int(ctrl.purged_bytes),
        "reuse_opportunities":int(ctrl.reuse_opportunities),"reuse_accepted":int(ctrl.reuse_accepted),
        "reuse_rejected":int(ctrl.reuse_rejected),"reuse_skipped_no_memory":int(ctrl.reuse_skipped_no_memory),
        "reuse_skipped_no_match":int(ctrl.reuse_skipped_no_match),"reuse_skipped_busy":int(ctrl.reuse_skipped_busy),
        "accepted_birth_ids":list(ctrl.accepted_ids),"active_specialist_id":ctrl.active_specialist_id,
        "resident_memory_ids":sorted(ctrl.specialist_memory.keys(),key=int),
        "final_topology":session.model.learner.topology_manifest(),"extra_compute":ctrl.extra_compute,
        "reactivation_checkpoint_records":ctrl.reactivation_checkpoint_records,
    }
    summary={"protocol":"024","round":"next_round_v2","stage":"v2b","run_id":run_id,
             "elapsed_seconds":float(elapsed),"full":model_metrics(dprob,dcls,labels,raw),
             "lifecycle":lifecycle,"confirmation_run":False,"stream_sha256":EXPECTED_STREAM_SHA}
    write_json(out_root/"arm_D_v2b/summary.json",summary)
    comparison={
        "protocol":"024","round":"next_round_v2","stage":"v2b","development_only":True,
        "confirmation_run":False,"stream_sha256":EXPECTED_STREAM_SHA,"replay_seed":700,"model_seed":1,
        "controls":{"C_u4_source":"v2a buffer128 artifact","D_v2a_growth_only":v2a_cmp},
        "full":{"C":model_metrics(c["probability"],c["class_probability"],labels,raw),"D_v2b":summary["full"]},
        "switch_first100":switches,
        "recurrence":{"mean_D_v2b_minus_C_ap":mean_delta,"positive_windows":positive,"valid_windows":len(deltas),
                      "C_threshold_metrics":cm,"D_v2b_threshold_metrics":dm,"normal_fpr_D_v2b_minus_C":fpr_delta},
        "lifecycle":lifecycle,"development_signal":signal,
        "development_signal_rule":"six recurrence mean D-C>=0.03, >=4/6 positive, normal FPR delta<=0.01",
    }
    write_json(out_root/"comparison.json",comparison)

    recovery={"available":False,"reason":"no completed reactivation checkpoint"}
    if ctrl.reactivation_checkpoint_records:
        ck=ctrl.reactivation_checkpoint_records[0]
        path=Path(ck["path"])
        fresh=V2BProtocol024Session(
            "D",1,s4.build_replay(stream_dir),budget(),out_root/"recovery_audit_session",
            anchor=train_anchor,guard_anchor=guard_anchor,v2b_config=cfg,learning_rate=1e-4,max_experts=8,
            run_id=run_id,stream_dir=stream_dir,phase_defs=phase_defs(bundle["manifest"]),
            target_mode=NEXT_TARGET_MODE,stream_sha=EXPECTED_STREAM_SHA,registration=registration)
        fresh.restore_checkpoint(path); start=fresh.cursor
        event_before=len(fresh.lifecycle_controller.events); updates_before=fresh.updates
        checked=0
        while fresh.cursor<fresh.steps and checked<4:
            idx0=fresh.cursor; p0,c0=fresh.step()
            if not np.allclose(p0,dprob[idx0],atol=1e-7,rtol=1e-6): raise AssertionError("checkpoint next prediction mismatch")
            if not np.allclose(c0,dcls[idx0],atol=1e-7,rtol=1e-6): raise AssertionError("checkpoint next class prediction mismatch")
            checked+=1
        recovery={"available":True,"checkpoint":ck,"restored_cursor":int(start),"prediction_steps_checked":checked,
                  "next_predictions_match":True,"updates_advanced":int(fresh.updates-updates_before),
                  "events_advanced":int(len(fresh.lifecycle_controller.events)-event_before),
                  "memory_ids_after":sorted(fresh.lifecycle_controller.specialist_memory.keys(),key=int),
                  "active_specialist_after":fresh.lifecycle_controller.active_specialist_id}
    write_json(out_root/"checkpoint_recovery_audit.json",recovery)
    status={"protocol":"024","round":"next_round_v2","stage":"v2b","completed":True,
            "development_signal":signal,"births":lifecycle["births"],"retirements":lifecycle["retirements"],
            "reactivations":lifecycle["reactivations"],"purges":lifecycle["purges"],
            "checkpoint_recovery_verified":bool(recovery.get("next_predictions_match",False)),
            "confirmation_run":False,"confirmation_seeds_used":[],
            "run_additional_controls_only_if_development_signal":bool(signal)}
    write_json(out_root/"status.json",status)
    print(json.dumps({"status":status,"recurrence":comparison["recurrence"],"full":comparison["full"],"lifecycle":lifecycle},indent=2,allow_nan=False))
    return status

def main():
    p=argparse.ArgumentParser(); p.add_argument("--stream",required=True); p.add_argument("--v2a-root",required=True); p.add_argument("--out-root",required=True)
    a=p.parse_args(); run(a.stream,a.v2a_root,a.out_root)
if __name__=="__main__": main()
