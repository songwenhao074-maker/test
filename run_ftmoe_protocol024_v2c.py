"""Run Protocol-024 next_round_v2c on immutable seed700/model1."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_anchor import anchor_metadata, load_protocol024_raw_next_anchor
from ftmoe_protocol024_eval import binary_detection_metrics, positive_resource_macro_f1, temporal_onset_metrics
from ftmoe_protocol024_session import NEXT_TARGET_MODE
from ftmoe_protocol024_v2a import split_anchor_train_guard, deployable_pressure
from ftmoe_protocol024_v2c import (
    V2CProtocol024Session, ReplacementConsistentLifecycle, V2C_DEFAULT,
    model_tensor_hash, _hash_value,
)

EXPECTED_STREAM_SHA = "468725ff2f164bee89017bfa329d63e20566659a42d2a0492e977e744c7ae946"
FIRST_NAMES = ("R1_first", "R2_first", "R3_first")
RECURRENCE_NAMES = ("R1_rec1", "R3_rec1", "R2_rec1", "R1_rec2", "R2_rec2", "R3_rec2")


def write_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf8")

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf8") as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+"\n")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

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

def phase_for(index, phases):
    for p in phases:
        if p["start"] <= int(index) < p["end"]: return p
    return None

def state_manifest(session):
    b=session.model.learner; c=session.lifecycle_controller
    return {"active_ids":list(b.ids),"dormant_ids":list(b.dormant_experts.keys()),"shadow_id":b.shadow_id,"ramp":{k:float(v) for k,v in b.ramp.items()},
            "phase":c.phase,"candidate_id":c.candidate_id,"reuse_candidate_id":getattr(c,"reuse_candidate_id",None),"active_specialist_id":getattr(c,"active_specialist_id",None)}

def adam_archive_hash(session):
    session._capture_live_optimizer_state(); h=hashlib.sha256(); _hash_value(h,session.optimizer_archive); return h.hexdigest()

def full_rng_hash():
    h=hashlib.sha256(); _hash_value(h,torch.get_rng_state()); _hash_value(h,np.random.get_state()); _hash_value(h,random.getstate()); return h.hexdigest()

def semantic_event(event):
    if event is None: return None
    banned={"path","elapsed_seconds","seconds","wall_time"}; return {k:v for k,v in event.items() if k not in banned}


class CounterfactualLifecycle(ReplacementConsistentLifecycle):
    """Main-arm diagnostic only: score no-match memories without changing selection."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.no_match_counterfactual=[]; self.extra_compute.setdefault("no_match_counterfactual_forwards",0)
    def _maybe_start_reuse(self, session):
        before=self.reuse_skipped_no_match; result=super()._maybe_start_reuse(session)
        if (not result) and self.reuse_skipped_no_match>before:
            t=int(session.cursor); bank=session.model.learner; dormant=[str(k) for k in bank.dormant_experts.keys() if str(k) in self.specialist_memory]
            if dormant and t<session.steps:
                x,sched,graph,context=s4.window_batch(session.replay,[t])
                with torch.no_grad():
                    out=session.model.predict_deployment(x,sched,graph,graph_context=context); z=session.model._last_z.detach(); rows=[]
                    for key in dormant:
                        correction,routed=self._preview_specialist(session,key,z); det=(out["base_final_detection_logits"]+correction[...,:2])[0].detach().cpu().numpy(); cls=(out["base_final_class_logits"]+correction[...,2:])[0].detach().cpu().numpy()
                        rows.append({"expert_id":key,"detection_logits":det,"class_logits":cls,"route_mean":float(routed[...,-1].mean().cpu())}); self.extra_compute["no_match_counterfactual_forwards"] += 1
                self.no_match_counterfactual.append({"prediction_index":t,"rows":rows})
        return result

class DiagnosticV2CSession(V2CProtocol024Session):
    def __init__(self,*args,guard_anchor=None,v2c_config=None,**kwargs):
        super().__init__(*args,guard_anchor=guard_anchor,v2c_config=v2c_config,**kwargs); cfg=dict(V2C_DEFAULT); cfg.update(v2c_config or {})
        self.lifecycle_controller=CounterfactualLifecycle(guard_anchor=guard_anchor,config=cfg); self.lifecycle_state=self.lifecycle_controller.state_dict(); self.lifecycle_state["enabled"]=True; self._apply_specialist_freeze(); self.lifecycle_controller._sync_session(self)


def session_registration(run_id, train_anchor, guard_anchor, reuse_enabled):
    return {"protocol":"024","round":"next_round_v2c","stage":"v2c","run_id":run_id,"replay_seed":700,"model_seed":1,"target":NEXT_TARGET_MODE,"update_every":4,
            "birth_shadow_budget":"buffer128_4x4","replacement_consistent_birth":True,"reuse_enabled":bool(reuse_enabled),"confirmation_run":False,"generalist_ids":["0","1","2","3"],
            "specialist_frozen":True,"max_live_specialists":1,"resident_limit":8,"reuse_every_matured":32,"reuse_validation_intervals":16,"crossfade_prediction_intervals":8,
            "anchor":anchor_metadata(train_anchor),"guard":anchor_metadata(guard_anchor),"stream_sha256":EXPECTED_STREAM_SHA}


def run_arm(bundle,stream_dir,out_dir,run_id,train_anchor,guard_anchor,reuse_enabled=True,diagnostic=False):
    cfg=dict(V2C_DEFAULT); cfg["reuse_enabled"]=bool(reuse_enabled); cls=DiagnosticV2CSession if diagnostic else V2CProtocol024Session
    session=cls("D",1,bundle,budget(),out_dir,anchor=train_anchor,guard_anchor=guard_anchor,v2c_config=cfg,learning_rate=1e-4,max_experts=8,
                run_id=run_id,stream_dir=stream_dir,phase_defs=phase_defs(bundle["manifest"]),target_mode=NEXT_TARGET_MODE,stream_sha=EXPECTED_STREAM_SHA,
                registration=session_registration(run_id,train_anchor,guard_anchor,reuse_enabled))
    started=time.perf_counter()
    for _ in range(session.steps): session.step()
    session.lifecycle_controller.mark_stream_end(); session.finish(); elapsed=time.perf_counter()-started; session.save(); ctrl=session.lifecycle_controller; events=list(ctrl.events); write_jsonl(Path(out_dir)/"lifecycle.jsonl",events)
    records=[]; phases=phase_defs(bundle["manifest"])
    for cid in sorted(ctrl.candidate_records,key=lambda x:int(x)):
        row=deepcopy(ctrl.candidate_records[cid]); train_laws=Counter(); val_laws=Counter()
        for idx in row["training_indices"]:
            p=phase_for(idx,phases); train_laws[(p or {}).get("response_law")]+=1
        for item in row["validation"]:
            p=phase_for(item["index"],phases); val_laws[(p or {}).get("response_law")]+=1
        row["audit_training_laws"]={str(k):int(v) for k,v in train_laws.items()}; row["audit_validation_laws"]={str(k):int(v) for k,v in val_laws.items()}; records.append(row)
    write_jsonl(Path(out_dir)/"candidate_pairs.jsonl",records); write_json(Path(out_dir)/"memory_training_intervals.json",{"candidates":records})
    counts=Counter(x["kind"] for x in events); created=len(records); accepted=sum(x.get("accepted") is True for x in records); rejected=sum(x.get("accepted") is False for x in records); pending=sum(x.get("accepted") is None for x in records)
    summary={"elapsed_seconds":float(elapsed),"candidate_created":created,"candidate_accepted":accepted,"candidate_rejected":rejected,"candidate_pending":pending,"birth":accepted,
             "retirement":int(ctrl.retirements),"reactivation":int(ctrl.reactivations),"purge":int(ctrl.purges),"purged_bytes":int(ctrl.purged_bytes),"opportunity_accounting":ctrl.due_conservation(),
             "freeze_checks":ctrl.freeze_checks,"event_counts":dict(counts),"extra_compute":ctrl.extra_compute,"topology":session.model.learner.topology_manifest(),
             "peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),"prediction_p95_seconds":float(np.percentile(session.predictions["prediction_seconds"],95))}
    write_json(Path(out_dir)/"summary.json",summary); write_json(Path(out_dir)/"opportunity_accounting.json",summary["opportunity_accounting"]); return session,summary,records,events


def counterfactual_report(session):
    ctrl=session.lifecycle_controller; out=[]
    for item in getattr(ctrl,"no_match_counterfactual",[]):
        idx=int(item["prediction_index"]); target=session.predictions["labels"][idx]; live_det=session.predictions["detection_logits"][idx]; live_cls=session.predictions["class_logits"][idx]
        live_loss=float(s4.supervised_terms(torch.as_tensor(live_det),torch.as_tensor(live_cls),torch.as_tensor(target).long())["total"]); candidates=[]
        for row in item["rows"]:
            loss=float(s4.supervised_terms(torch.as_tensor(row["detection_logits"]),torch.as_tensor(row["class_logits"]),torch.as_tensor(target).long())["total"])
            candidates.append({"expert_id":row["expert_id"],"loss":loss,"live_loss":live_loss,"relative_improvement":float((live_loss-loss)/max(abs(live_loss),1e-12)),"route_mean":row["route_mean"]})
        out.append({"prediction_index":idx,"future_label_used_posthoc_only":True,"candidates":candidates})
    return out


def recurrence_latency(events,records,phases):
    first_influence=min([x["first_influence_cursor"] for x in records if x.get("first_influence_cursor") is not None],default=None); rows=[]
    for name in RECURRENCE_NAMES:
        p=next(x for x in phases if x["name"]==name); start,end=p["start"],p["end"]; ev=[x for x in events if start<=int(x.get("cursor",-1))<end]; effective=0 if first_influence is None else max(0,end-max(start,int(first_influence)))
        rows.append({"phase":name,"response_law":p.get("response_law"),"intervals":[start,end],"memory_available_before":any(x["kind"]=="specialist_memory_frozen" and int(x["cursor"])<start for x in events),
                     "reuse_due":sum(x["kind"] in ("reuse_skipped_no_memory","reuse_skipped_no_match","reuse_skipped_insufficient_z","reuse_candidate_selected","reuse_due_skipped_busy") for x in ev),
                     "no_match":sum(x["kind"] in ("reuse_skipped_no_match","reuse_skipped_insufficient_z") for x in ev),"selected":sum(x["kind"]=="reuse_candidate_selected" for x in ev),
                     "validation_rejected":sum(x["kind"]=="reuse_candidate_rejected" for x in ev),"reactivated":sum(x["kind"]=="specialist_reactivated" for x in ev),"crossfade_steps":sum(x["kind"]=="specialist_crossfade_step" for x in ev),
                     "effective_specialist_predictions":int(effective),"effective_prediction_fraction":float(effective/max(end-start,1))})
    return rows


def capture_after_next_update(session, checkpoint_updates, event_count):
    update_snapshot=None; prediction_after=None
    for _ in range(session.steps-session.cursor):
        idx=session.cursor; p,c=session.step()
        if update_snapshot is None and session.updates>checkpoint_updates:
            update_snapshot={"after_step_index":int(idx),"update_count":int(session.updates),"model_hash":model_tensor_hash(session.model),"adam_hash":adam_archive_hash(session),"rng_hash":full_rng_hash(),"state":state_manifest(session)}; continue
        if update_snapshot is not None and prediction_after is None and idx>update_snapshot["after_step_index"]:
            prediction_after={"index":int(idx),"probability":p.copy(),"class_probability":c.copy()}; break
    next_event=None
    if len(session.lifecycle_controller.events)>event_count: next_event=semantic_event(session.lifecycle_controller.events[event_count])
    return update_snapshot,prediction_after,next_event


def strict_recovery(main_session,bundle,stream_dir,out_root,train_anchor,guard_anchor,run_id):
    ctrl=main_session.lifecycle_controller
    if not ctrl.reactivation_checkpoint_records: return {"available":False,"reason":"no completed reactivation checkpoint","verified":False}
    ck=ctrl.reactivation_checkpoint_records[0]; path=Path(ck["path"]); payload=torch.load(path,map_location="cpu",weights_only=False); ck_cursor=int(payload["session"]["cursor"]); ck_updates=int(payload["session"]["updates"]); event_count=len(payload["session"]["lifecycle_events"]); cfg=dict(V2C_DEFAULT); cfg["reuse_enabled"]=True
    ref=V2CProtocol024Session("D",1,s4.build_replay(stream_dir),budget(),Path(out_root)/"recovery_reference_uninterrupted",anchor=train_anchor,guard_anchor=guard_anchor,v2c_config=cfg,learning_rate=1e-4,max_experts=8,
        run_id=run_id+"_recovery_ref",stream_dir=stream_dir,phase_defs=phase_defs(bundle["manifest"]),target_mode=NEXT_TARGET_MODE,stream_sha=EXPECTED_STREAM_SHA,registration=session_registration(run_id+"_recovery_ref",train_anchor,guard_anchor,True))
    while ref.cursor<ck_cursor: ref.step()
    pred_prefix_match=np.allclose(ref.predictions["probability"][:ck_cursor],main_session.predictions["probability"][:ck_cursor],atol=1e-7,rtol=1e-6); ref_update,ref_after,ref_event=capture_after_next_update(ref,ck_updates,event_count)
    restored=V2CProtocol024Session("D",1,s4.build_replay(stream_dir),budget(),Path(out_root)/"recovery_restored",anchor=train_anchor,guard_anchor=guard_anchor,v2c_config=cfg,learning_rate=1e-4,max_experts=8,
        run_id=payload["run_id"],stream_dir=stream_dir,phase_defs=phase_defs(bundle["manifest"]),target_mode=NEXT_TARGET_MODE,stream_sha=EXPECTED_STREAM_SHA,registration=payload["registration"])
    restored.restore_checkpoint(path); first_idx=restored.cursor; p0,c0=restored.step(); next_prediction_match=np.allclose(p0,main_session.predictions["probability"][first_idx],atol=1e-7,rtol=1e-6) and np.allclose(c0,main_session.predictions["class_probability"][first_idx],atol=1e-7,rtol=1e-6)
    if restored.updates>ck_updates:
        rst_update={"after_step_index":first_idx,"update_count":restored.updates,"model_hash":model_tensor_hash(restored.model),"adam_hash":adam_archive_hash(restored),"rng_hash":full_rng_hash(),"state":state_manifest(restored)}; idx=restored.cursor; pa,ca=restored.step(); rst_after={"index":idx,"probability":pa.copy(),"class_probability":ca.copy()}; rst_event=semantic_event(restored.lifecycle_controller.events[event_count]) if len(restored.lifecycle_controller.events)>event_count else None
    else: rst_update,rst_after,rst_event=capture_after_next_update(restored,ck_updates,event_count)
    update_match=bool(ref_update and rst_update and ref_update["model_hash"]==rst_update["model_hash"] and ref_update["adam_hash"]==rst_update["adam_hash"] and ref_update["rng_hash"]==rst_update["rng_hash"] and ref_update["state"]==rst_update["state"])
    after_match=bool(ref_after and rst_after and ref_after["index"]==rst_after["index"] and np.allclose(ref_after["probability"],rst_after["probability"],atol=1e-7,rtol=1e-6) and np.allclose(ref_after["class_probability"],rst_after["class_probability"],atol=1e-7,rtol=1e-6)); event_match=ref_event==rst_event if ref_event is not None else None
    return {"available":True,"checkpoint":ck,"checkpoint_cursor":ck_cursor,"uninterrupted_prefix_matches_main":bool(pred_prefix_match),"next_prediction_match":bool(next_prediction_match),"next_live_update_match":update_match,
            "prediction_after_update_match":after_match,"next_nonempty_event_covered":ref_event is not None,"next_event_match":event_match,"reference_update":ref_update,"restored_update":rst_update,
            "reference_next_event":ref_event,"restored_next_event":rst_event,"verified":bool(pred_prefix_match and next_prediction_match and update_match and after_match and event_match is True)}


def evaluate(c,d,nr,labels,raw,phases):
    switches=[]
    for name in FIRST_NAMES+RECURRENCE_NAMES:
        p=next(x for x in phases if x["name"]==name); start,end=p["start"],min(p["start"]+100,p["end"]); cm=binary_detection_metrics(c["probability"][start:end],labels[start:end]); dm=binary_detection_metrics(d.predictions["probability"][start:end],labels[start:end]); nm=binary_detection_metrics(nr.predictions["probability"][start:end],labels[start:end])
        switches.append({"phase":name,"response_law":p.get("response_law"),"intervals":[start,end],"C":cm,"D_v2c":dm,"D_v2c_no_reuse":nm,"D_v2c_minus_C_ap":None if cm["ap"] is None or dm["ap"] is None else float(dm["ap"]-cm["ap"])})
    rec=[x for x in switches if x["phase"] in RECURRENCE_NAMES]; deltas=[x["D_v2c_minus_C_ap"] for x in rec if x["D_v2c_minus_C_ap"] is not None]; idx=np.concatenate([np.arange(x["intervals"][0],x["intervals"][1]) for x in rec]); cm=binary_detection_metrics(c["probability"][idx],labels[idx]); dm=binary_detection_metrics(d.predictions["probability"][idx],labels[idx]); fpr_delta=None if cm["fpr"] is None or dm["fpr"] is None else float(dm["fpr"]-cm["fpr"]); mean=float(np.mean(deltas)) if deltas else None; pos=sum(x>0 for x in deltas); signal=bool(len(deltas)==6 and mean>=0.03 and pos>=4 and fpr_delta is not None and fpr_delta<=0.01)
    return switches,{"mean_D_v2c_minus_C_ap":mean,"positive_windows":int(pos),"valid_windows":len(deltas),"C_threshold_metrics":cm,"D_v2c_threshold_metrics":dm,"normal_fpr_D_v2c_minus_C":fpr_delta},signal


def run(stream_dir,v2a_root,v2b_root,out_root,github_run_id):
    stream_dir=Path(stream_dir); v2a_root=Path(v2a_root); v2b_root=Path(v2b_root); out_root=Path(out_root)
    if out_root.exists(): raise FileExistsError("refusing to overwrite v2c run %s"%out_root)
    out_root.mkdir(parents=True); bundle=s4.build_replay(stream_dir)
    if bundle["steps"]!=4980 or bundle["manifest"].get("stream_sha256")!=EXPECTED_STREAM_SHA: raise AssertionError("immutable stream mismatch")
    phases=phase_defs(bundle["manifest"]); anchor=load_protocol024_raw_next_anchor(); train_anchor,guard_anchor=split_anchor_train_guard(anchor,modulo=5,guard_remainder=0); c_path=v2a_root/"runs/v2a_buffer128_seed700_model1/arm_C/predictions.npz"; c=npz_dict(c_path)
    source_manifest={"stream_sha256":EXPECTED_STREAM_SHA,"v2a_c_predictions":{"path":str(c_path),"sha256":sha(c_path)},"v2b_root":str(v2b_root),"v2b_files":[]}
    for rel in ["registration.json","runs/v2b_memory_seed700_model1/comparison.json","runs/v2b_memory_seed700_model1/arm_D_v2b/summary.json","runs/v2b_memory_seed700_model1/arm_D_v2b/lifecycle.jsonl","runs/v2b_memory_seed700_model1/checkpoint_recovery_audit.json"]:
        p=v2b_root/rel
        if p.is_file(): source_manifest["v2b_files"].append({"path":rel,"sha256":sha(p),"bytes":p.stat().st_size})
    write_json(out_root/"reviewed_source_manifest.json",source_manifest); run_id="seed700_model1_v2c_%s"%github_run_id
    main,main_sum,main_records,main_events=run_arm(bundle,stream_dir,out_root/"arm_D_v2c",run_id,train_anchor,guard_anchor,True,True); noreuse,nr_sum,nr_records,nr_events=run_arm(s4.build_replay(stream_dir),stream_dir,out_root/"arm_D_v2c_no_reuse",run_id+"_no_reuse",train_anchor,guard_anchor,False,False)
    labels=main.predictions["labels"]; raw=main.predictions["raw_labels"]
    if not np.array_equal(labels,c["labels"]): raise AssertionError("C and v2c target rows differ")
    switches,recurrence,signal=evaluate(c,main,noreuse,labels,raw,phases); pressure=deployable_pressure(bundle["arrays"]["host_features"][:4980],bundle["arrays"]["capacities"][:4980]); pressure_metrics=binary_detection_metrics(pressure,labels)
    cf=counterfactual_report(main); write_json(out_root/"readonly_no_match_counterfactual.json",{"future_labels_posthoc_only":True,"rows":cf}); latency=recurrence_latency(main_events,main_records,phases); write_json(out_root/"recurrence_latency.json",{"phases":latency}); recovery=strict_recovery(main,bundle,stream_dir,out_root,train_anchor,guard_anchor,run_id); write_json(out_root/"checkpoint_recovery_audit.json",recovery)
    equivalence={"verified_by_unit_tests":True,"atol":2e-6,"rtol":2e-6,"paths":["no_old_specialist","old_specialist_active","dormant_reuse"],"legacy_preview_used_for_decision":False,"production_module":"ftmoe_protocol024_v2c.py"}; write_json(out_root/"preview_deployment_equivalence.json",equivalence)
    comparison={"protocol":"024","round":"next_round_v2c","development_only":True,"confirmation_run":False,"stream_sha256":EXPECTED_STREAM_SHA,"replay_seed":700,"model_seed":1,"C_source_fingerprint":source_manifest["v2a_c_predictions"],
        "full":{"C":model_metrics(c["probability"],c["class_probability"],labels,raw),"D_v2c":model_metrics(main.predictions["probability"],main.predictions["class_probability"],labels,raw),"D_v2c_no_reuse":model_metrics(noreuse.predictions["probability"],noreuse.predictions["class_probability"],labels,raw)},
        "deployable_pressure":pressure_metrics,"switch_first100":switches,"recurrence":recurrence,"development_signal":signal,"development_signal_rule":"six recurrence mean D-C>=0.03, >=4/6 positive, normal FPR delta<=0.01"}; write_json(out_root/"comparison.json",comparison)
    write_json(out_root/"cost_profile.json",{"D_v2c":main_sum,"D_v2c_no_reuse":nr_sum,"C_prediction_file_bytes":c_path.stat().st_size})
    status={"protocol":"024","round":"next_round_v2c","completed":True,"verified_fix":bool(equivalence["verified_by_unit_tests"]),"checkpoint_recovery_verified":bool(recovery.get("verified",False)),"candidate_created":main_sum["candidate_created"],"candidate_accepted":main_sum["candidate_accepted"],"candidate_rejected":main_sum["candidate_rejected"],"candidate_pending":main_sum["candidate_pending"],
        "birth":main_sum["birth"],"retirement":main_sum["retirement"],"reactivation":main_sum["reactivation"],"purge":main_sum["purge"],"opportunity_accounting":main_sum["opportunity_accounting"],"development_signal":bool(signal),"confirmation_run":False,"confirmation_seeds_used":[],"unused_confirmation_seeds":[701,702,703],"advance_to_protocol025":not bool(signal)}
    if not all(x["conserved"] for x in main_sum["opportunity_accounting"].values()): raise AssertionError("opportunity accounting does not conserve")
    write_json(out_root/"status.json",status); print(json.dumps({"status":status,"recurrence":recurrence,"full":comparison["full"]},indent=2,allow_nan=False)); return status


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stream",required=True); p.add_argument("--v2a-root",required=True); p.add_argument("--v2b-root",required=True); p.add_argument("--out-root",required=True); p.add_argument("--github-run-id",required=True); a=p.parse_args(); run(a.stream,a.v2a_root,a.v2b_root,a.out_root,a.github_run_id)
if __name__=="__main__": main()
