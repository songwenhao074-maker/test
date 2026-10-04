"""Protocol-044 revision1 transactional bounded-lifecycle production engine."""
from __future__ import annotations
import copy, hashlib, json, os, random, time
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import run_ftmoe_protocol037 as p37
from protocol035_common import sha256_file, sha256_state_dict
import protocol044_common as c

def _mh(x): return sha256_state_dict(x["model"].state_dict())
def _oh(x): return c.opt_digest(x["optimizer"])

class EngineeringCheckpointStop(RuntimeError):
    pass

class Machine044:
    SUBSTEPS=("predict","settle","control","live_update","shadow_update","finish")
    def __init__(self,src,work_dir,arm,allow_gradient=True,real_science=False,resume=False):
        if arm not in c.ARMS: raise ValueError(arm)
        self.src=src; self.arm=arm; self.work_dir=Path(work_dir); self.work_dir.mkdir(parents=True,exist_ok=True)
        self.allow_gradient=bool(allow_gradient); self.real_science=bool(real_science)
        self.cursor=0; self.next_substep="predict"; self.terminal_progress=0
        self.deployment_epoch=0; self.epoch_start_prediction=0
        self.experts={}; self.active_ids=[]; self.shadow=None; self.next_id=0; self.deleted_ids=set(); self.tombstones=[]
        self.attempts_after_e0=0; self.last_attempt_end_m=None; self.first_birth_t=None
        self.birth_streak=0; self.pressure_streak=0
        self.reuse_slot=None; self.reuse_started=0; self.reuse_history=deque(maxlen=2); self.last_reuse_quality_failure=None
        self.pending_train_context=None
        self.tracker=c.UtilityTracker(); self.tracker.reset(0,0,[],"initial")
        n=int(src["n"]); self.audit_array_dir=self.work_dir/"audit_arrays"; self.audit_array_dir.mkdir(parents=True,exist_ok=True)
        self.out=self._open_output_arrays(n,resume)
        self.settled=self._open_array("settled",np.int8,(n,),0,resume)
        self.b_loss=self._open_array("b_loss",np.float64,(n,),np.nan,resume)
        self.d_loss=self._open_array("d_loss",np.float64,(n,),np.nan,resume)
        self.birth_checks=deque(maxlen=64); self.pressure_checks=deque(maxlen=64); self.utility_checks=deque(maxlen=64); self.sleep_tables=deque(maxlen=64)
        self.lifecycle_events=deque(maxlen=64); self.candidate_decisions=deque(maxlen=64); self.reuse_decisions=deque(maxlen=64); self.opportunity_log=deque(maxlen=64)
        self.update_log=deque(maxlen=64); self.settlement_log=deque(maxlen=64); self.qualification_records=deque(maxlen=64); self.reclamation_events=deque(maxlen=64)
        self.live_optimizer_steps=0; self.shadow_optimizer_steps=0; self.deployed_forwards=0
        self.reuse_preview_forwards=0; self.shadow_preview_forwards=0; self.live_training_forwards=0; self.shadow_training_forwards=0
        self.controller_cpu_seconds=0.; self.prediction_seconds=0.; self.update_seconds=0.; self.io_seconds=0.
        self.max_active_seen=0; self.max_resident_seen=0; self.peak_resident_tensor_bytes=0
        self.permanent_deletions=0; self.actual_optimizer_calls=0; self.near_threshold_recompute_count=0
        self.action_seq=0; self.completed_action_ids=deque(maxlen=64); self.completed_action_count=0; self.action_journal=self.work_dir/"action_journal.json"; self.checkpoint_dir=None
        self.crash_probe=None; self.engineering_stop_prefix=None; self.terminal_counter_delta=[]
        (self.work_dir/"streams").mkdir(parents=True,exist_ok=True); self.audit_state={}

    def _open_array(self,name,dtype,shape,fill,resume):
        p=self.audit_array_dir/(name+".npy")
        if resume:
            if not p.exists(): raise FileNotFoundError("audit array missing "+str(p))
            return np.lib.format.open_memmap(p,mode="r+",dtype=dtype,shape=shape)
        a=np.lib.format.open_memmap(p,mode="w+",dtype=dtype,shape=shape); a[...] = fill; a.flush(); return a

    def _open_output_arrays(self,n,resume):
        return {
          "probability":self._open_array("probability",np.float32,(n,c.HOSTS),np.nan,resume),
          "detection_logits":self._open_array("detection_logits",np.float32,(n,c.HOSTS,2),np.nan,resume),
          "total_delta":self._open_array("total_delta",np.float32,(n,c.HOSTS),np.nan,resume),
          "live_margin":self._open_array("live_margin",np.float32,(n,c.HOSTS),np.nan,resume),
          "B_margin":self._open_array("B_margin",np.float32,(n,c.HOSTS),np.nan,resume),
          "deployment_epoch":self._open_array("deployment_epoch",np.int32,(n,),0,resume),
          "active_ids":self._open_array("active_ids",np.int16,(n,2),-1,resume),
          "active_count":self._open_array("active_count",np.int8,(n,),0,resume),
          "shadow_present":self._open_array("shadow_present",np.int8,(n,),0,resume),
          "accepted_count":self._open_array("accepted_count",np.int8,(n,),0,resume),
          "expert_versions":self._open_array("expert_versions",np.int32,(n,2),-1,resume),
          "expert_hashes":self._open_array("expert_hashes",np.dtype("<U64"),(n,2),"",resume),
          "contribution":self._open_array("contribution",np.float32,(5,n,c.HOSTS),np.nan,resume),
        }

    def _flush_arrays(self):
        for a in list(self.out.values())+[self.settled,self.b_loss,self.d_loss]:
            if hasattr(a,"flush"): a.flush()

    def _stream(self,name,row):
        t0=time.perf_counter()
        line=json.dumps(row,ensure_ascii=False,allow_nan=False,default=str,separators=(",",":"))+"\n"
        p=self.work_dir/"streams"/(name+".jsonl"); st=self.audit_state.get(name,{"count":0,"chain":"0"*64,"offset":0})
        with open(p,"a",encoding="utf8") as f:
            f.write(line); off=f.tell()
        chain=hashlib.sha256((st["chain"]+line).encode("utf8")).hexdigest()
        self.audit_state[name]={"count":int(st["count"])+1,"chain":chain,"offset":int(off)}
        self.io_seconds+=time.perf_counter()-t0

    def _record(self,name,buf,row):
        buf.append(copy.deepcopy(row)); self._stream(name,row)

    def _verify_stream_prefixes(self):
        for name,st in self.audit_state.items():
            p=self.work_dir/"streams"/(name+".jsonl")
            if not p.exists() or p.stat().st_size!=int(st["offset"]):
                raise RuntimeError("ambiguous_state: audit offset mismatch "+name)
            chain="0"*64; count=0
            with open(p,"r",encoding="utf8") as f:
                for line in f:
                    chain=hashlib.sha256((chain+line).encode("utf8")).hexdigest(); count+=1
            if count!=int(st["count"]) or chain!=st["chain"]:
                raise RuntimeError("ambiguous_state: audit hash mismatch "+name)

    def _model_hash(self,x): return _mh(x)
    def _opt_hash(self,x): return _oh(x)
    def _due16(self,t): return (int(t)+1)%16==0
    def _due64(self,t): return (int(t)+1)%64==0
    def resident_count(self): return len(self.experts)+(1 if self.shadow is not None else 0)
    def dormant_ids(self): return sorted(e for e,x in self.experts.items() if x["role"]=="dormant")
    def current_tensor_bytes(self):
        return int(sum(c.tensor_bytes(x) for x in self.experts.values())+(0 if self.shadow is None else c.tensor_bytes(self.shadow)))
    def _resource_sample(self):
        self.max_active_seen=max(self.max_active_seen,len(self.active_ids)); self.max_resident_seen=max(self.max_resident_seen,self.resident_count())
        self.peak_resident_tensor_bytes=max(self.peak_resident_tensor_bytes,self.current_tensor_bytes())

    def _begin_action(self,kind,t,meta=None):
        self.action_seq+=1
        row={"protocol":"044","revision":1,"arm":self.arm,"action_seq":self.action_seq,"status":"pending","kind":kind,
          "cursor":int(t),"next_substep":self.next_substep,"meta":meta or {}}
        c.W(self.action_journal,row); return row
    def _commit_action(self,row,extra=None):
        z=dict(row); z["status"]="committed"
        if extra: z.update(extra)
        c.W(self.action_journal,z); self.completed_action_ids.append(int(row["action_seq"])); self.completed_action_count+=1

    def _cancel_shadow(self,t,reason):
        if self.shadow is None: return
        row={"event":"shadow_cancel","at_interval":int(t),"candidate_id":int(self.shadow["id"]),"updates":int(self.shadow.get("updates",0)),
          "status":self.shadow.get("status"),"reason":reason}
        self._record("candidate_decisions",self.candidate_decisions,row)
        if int(self.shadow["id"])>0: self.last_attempt_end_m=int(t)-2
        self.shadow=None

    def _cancel_reuse(self,t,reason):
        if self.reuse_slot is None: return
        s=self.reuse_slot
        row={"event":"reuse_cancel","at_interval":int(t),"reason":reason,"start_t":s["start_t"],
          "candidate_ids":sorted(int(x) for x in s["candidate_ids"]),"issued_count":len(s.get("issued",{})),"settled_count":len(s.get("settled_rows",{}))}
        self._record("reuse_decisions",self.reuse_decisions,row); self.reuse_slot=None

    def _topology_epoch(self,t,reason,active_before=None):
        old=self.deployment_epoch; self.deployment_epoch+=1; self.epoch_start_prediction=int(t)+1
        self.tracker=c.UtilityTracker(); self.tracker.reset(self.deployment_epoch,self.epoch_start_prediction,self.active_ids,reason)
        self.pressure_streak=0
        row={"event":"epoch_transition","at_interval":int(t),"old_epoch":old,"new_epoch":self.deployment_epoch,
          "active_ids_before":list(active_before if active_before is not None else self.active_ids),"active_ids_after":list(self.active_ids),
          "first_affected_prediction":int(t)+1,"reason":reason}
        self._record("lifecycle",self.lifecycle_events,row)

    def _topology_cleanup(self,t,reason,active_before,keep_shadow=False,keep_reuse=False):
        if not keep_shadow: self._cancel_shadow(t,"topology_"+reason)
        if not keep_reuse: self._cancel_reuse(t,"topology_"+reason)
        self._topology_epoch(t,reason,active_before)

    def _make_active(self,eid,t,reason):
        eid=int(eid); x=self.experts[eid]; before=list(self.active_ids)
        dormant_predictions=0
        if x["role"]=="dormant" and x.get("dormant_since_prediction") is not None:
            if x.get("dormant_model_hash")!=self._model_hash(x) or x.get("dormant_optimizer_hash")!=self._opt_hash(x) or x.get("dormant_optimizer_step")!=c.opt_step(x["optimizer"]):
                raise AssertionError("dormant state changed")
            ds=x.get("dormant_since_prediction"); dormant_predictions=max(0,int(t)+1-int(ds))
            x["reactivations"]=int(x.get("reactivations",0))+1
        x["role"]="active"; x["dormant_since_prediction"]=None
        if x.get("first_active_t") is None: x["first_active_t"]=int(t)+1
        if eid not in self.active_ids: self.active_ids.append(eid); self.active_ids.sort()
        row={"event":"activate","expert_id":eid,"at_interval":int(t),"reason":reason,"active_ids_before":before,"active_ids_after":list(self.active_ids),
          "first_affected_prediction":int(t)+1,"dormant_predictions_before_reactivation":dormant_predictions,
          "model_hash":self._model_hash(x),"optimizer_hash":self._opt_hash(x),"optimizer_step":c.opt_step(x["optimizer"])}
        self._record("lifecycle",self.lifecycle_events,row)

    def _make_dormant(self,eid,t,reason,score):
        eid=int(eid); x=self.experts[eid]; before=list(self.active_ids)
        if eid not in before: raise AssertionError("sleep nonactive")
        self.active_ids.remove(eid); x["role"]="dormant"; x["dormant_since_prediction"]=int(t)+1; x["last_active_prediction"]=int(t)
        x["dormant_model_hash"]=self._model_hash(x); x["dormant_optimizer_hash"]=self._opt_hash(x); x["dormant_optimizer_step"]=c.opt_step(x["optimizer"])
        x["historical_utility"].append({"epoch":self.deployment_epoch,"at_interval":int(t),"reason":reason,"score":copy.deepcopy(score)})
        row={"event":"sleep","expert_id":eid,"at_interval":int(t),"reason":reason,"active_ids_before":before,"active_ids_after":list(self.active_ids),
          "resident_ids_before":sorted(self.experts),"resident_ids_after":sorted(self.experts),"dormant_from_prediction":int(t)+1,
          "last_active_prediction":int(t),"utility":copy.deepcopy(score),"model_hash":x["dormant_model_hash"],
          "optimizer_hash":x["dormant_optimizer_hash"],"optimizer_step":x["dormant_optimizer_step"],"first_affected_prediction":int(t)+1}
        self._record("lifecycle",self.lifecycle_events,row)

    def _epoch_matured(self,m):
        if m<self.epoch_start_prediction: return 0
        idx=np.arange(self.epoch_start_prediction,m+1,dtype=np.int64)
        if len(idx)==0: return 0
        return int(np.sum(self.settled[idx].astype(bool)&(self.out["deployment_epoch"][idx]==self.deployment_epoch)))

    def _first_birth_check(self,t):
        m=int(t)-2
        if m<319: return None
        recent=np.arange(m-63,m+1); prev=np.arange(m-319,m-63)
        if not np.isfinite(self.b_loss[recent]).all() or not np.isfinite(self.b_loss[prev]).all(): return None
        R=float(self.b_loss[recent].mean()); P=float(self.b_loss[prev].mean()); y=self.src["B"]["labels"][recent]
        pos=int((y>0).sum()); neg=int((y<=0).sum()); cand=bool(R>=1.25*max(P,1e-6) and R-P>=.02 and pos>=16 and neg>=16)
        before=self.birth_streak; self.birth_streak=before+1 if cand else 0
        row={"at_interval":int(t),"m":m,"R":R,"P":P,"positive_support":pos,"negative_support":neg,"candidate":cand,
          "streak_before":before,"streak_after":self.birth_streak}
        self._record("birth_checks",self.birth_checks,row); return row

    def _first_birth(self,t):
        if self.experts or self.next_id!=0: raise AssertionError("bad E0 state")
        row=self._begin_action("first_birth",t)
        x=c.new_expert(0,t,"active"); x["accepted"]=True; x["first_active_t"]=int(t)+1
        self.experts[0]=x; self.active_ids=[0]; self.next_id=1; self.first_birth_t=int(t)
        ev={"event":"first_birth","expert_id":0,"at_interval":int(t),"active_ids_before":[],"active_ids_after":[0],
          "resident_ids_before":[],"resident_ids_after":[0],"first_affected_prediction":int(t)+1,
          "model_hash_before_update":self._model_hash(x),"optimizer_hash_before_update":self._opt_hash(x),"optimizer_step_before_update":c.opt_step(x["optimizer"])}
        self._record("lifecycle",self.lifecycle_events,ev)
        self._topology_epoch(t,"first_birth",[]); self._commit_action(row); self.save_checkpoint("post_first_birth_t%d"%t,True)

    def _predict_expert(self,x,t):
        t0=time.perf_counter(); d=c.expert_delta_np(x["model"],self.src["core"]["T"]["z"][t]); self.prediction_seconds+=time.perf_counter()-t0; return d

    def _predict(self,t):
        B=self.src["B"]; bm=c.bmargin(B,t); deltas={}; total=np.zeros(c.HOSTS,np.float32)
        for slot,eid in enumerate(sorted(self.active_ids)):
            x=self.experts[eid]; d=self._predict_expert(x,t); self.deployed_forwards+=1; deltas[eid]=d; total+=d
            x["last_active_prediction"]=int(t)
            self.out["active_ids"][t,slot]=eid; self.out["expert_versions"][t,slot]=int(x["version"]); self.out["expert_hashes"][t,slot]=self._model_hash(x)
            self.out["contribution"][eid,t]=d
        margin=(bm+total).astype(np.float32); self.out["probability"][t]=c.sigmoid_margin(margin)
        self.out["detection_logits"][t]=c.logits_from_margin(B["detection_logits"][t],total); self.out["total_delta"][t]=total
        self.out["live_margin"][t]=margin; self.out["B_margin"][t]=bm; self.out["deployment_epoch"][t]=self.deployment_epoch
        self.out["active_count"][t]=len(self.active_ids); self.out["shadow_present"][t]=int(self.shadow is not None); self.out["accepted_count"][t]=len(self.experts)
        if self.shadow is not None and self.shadow.get("status")=="validating" and t>=int(self.shadow["validation_start"]) and len(self.shadow["issued"])<32:
            if self.shadow["proposal_epoch"]!=self.deployment_epoch: raise AssertionError("stale shadow")
            cd=self._predict_expert(self.shadow,t); self.shadow_preview_forwards+=1; cm=(margin+cd).astype(np.float32)
            self.shadow["issued"][int(t)]={"candidate_margin":cm,"live_margin":margin.copy(),"B_margin":bm.copy(),"candidate_delta":cd}
        if self.reuse_slot is not None and t>=int(self.reuse_slot["prediction_start"]) and len(self.reuse_slot["issued"])<32:
            if self.reuse_slot["proposal_epoch"]!=self.deployment_epoch: raise AssertionError("stale reuse")
            rr={"live_margin":margin.copy(),"B_margin":bm.copy(),"candidates":{}}
            for eid in self.reuse_slot["candidate_ids"]:
                x=self.experts[int(eid)]
                if x["role"]!="dormant" or self._model_hash(x)!=self.reuse_slot["candidate_hashes"][int(eid)]: raise AssertionError("reuse candidate changed")
                dd=self._predict_expert(x,t); self.reuse_preview_forwards+=1
                rr["candidates"][int(eid)]={"margin":(margin+dd).astype(np.float32),"delta":dd}
            self.reuse_slot["issued"][int(t)]=rr
        if self.deployed_forwards>11904 or self.reuse_preview_forwards>3072 or self.shadow_preview_forwards>128: raise AssertionError("forward budget")
        self._resource_sample()

    def _settle(self,t):
        i=int(t)-2
        if i<0 or i>=self.src["n"]: return
        if self.settled[i]: raise AssertionError("duplicate settlement")
        y=self.src["B"]["labels"][i]; self.b_loss[i]=c.stable_bce_interval(self.out["B_margin"][i],y); self.d_loss[i]=c.stable_bce_interval(self.out["live_margin"][i],y); self.settled[i]=1
        ep=int(self.out["deployment_epoch"][i]); deltas={}
        for eid in range(5):
            d=self.out["contribution"][eid,i]
            if np.isfinite(d).all(): deltas[eid]=d.copy()
        if ep==self.deployment_epoch: self.tracker.settle(i,ep,self.out["live_margin"][i],deltas,y)
        if self.engineering_stop_prefix=="window_expiry_after" and ep==self.deployment_epoch and i>=128:
            active=list(self.active_ids)
            if active:
                ss=self.tracker.score(active[0],i)
                if ss.get("valid") and ss.get("first_i")==i-127 and i>=129:
                    self.save_checkpoint("window_expiry_after_t%d"%t,True)
        if self.engineering_stop_prefix=="late_old_epoch_settlement" and ep!=self.deployment_epoch:
            self.save_checkpoint("late_old_epoch_settlement_t%d"%t,True)
        row={"at_interval":int(t),"settled_interval":i,"issued_epoch":ep,"current_epoch":self.deployment_epoch,
          "control_eligible":bool(ep==self.deployment_epoch),"B_bce":float(self.b_loss[i]),"D_bce":float(self.d_loss[i]),"active_ids":sorted(deltas)}
        self._record("settlements",self.settlement_log,row)
        if self.shadow is not None and self.shadow.get("status")=="validating" and i in self.shadow.get("issued",{}):
            r=self.shadow["issued"][i]; self.shadow["settled_rows"][i]={**r,"y":np.asarray(y).copy()}
            if len(self.shadow["settled_rows"])==32:
                self.shadow["ready"]=True
                if self.engineering_stop_prefix=="qualification_ready_before_decision":
                    self.save_checkpoint("qualification_ready_before_decision_t%d"%t,True)
        if self.reuse_slot is not None and i in self.reuse_slot.get("issued",{}):
            r=self.reuse_slot["issued"][i]; self.reuse_slot["settled_rows"][i]={"y":np.asarray(y).copy(),**r}
            if len(self.reuse_slot["settled_rows"])==32: self.reuse_slot["ready"]=True

    def _pressure_check(self,t):
        if not self.experts: self.pressure_streak=0; return None
        m=int(t)-2
        if m<319: self.pressure_streak=0; return None
        recent=np.arange(m-63,m+1); prev=np.arange(m-319,m-63)
        same=bool(np.all(self.out["deployment_epoch"][recent]==self.deployment_epoch))
        matured=self._epoch_matured(m); minm=256 if self.active_ids else 64
        eligible=bool(same and matured>=minm and np.isfinite(self.d_loss[recent]).all() and np.isfinite(self.d_loss[prev]).all())
        R=P=None; pos=neg=0; cand=False
        if eligible:
            R=float(self.d_loss[recent].mean()); P=float(self.d_loss[prev].mean()); y=self.src["B"]["labels"][recent]
            pos=int((y>0).sum()); neg=int((y<=0).sum()); cand=bool(R>=1.25*max(P,1e-6) and R-P>=.02 and pos>=16 and neg>=16)
        before=self.pressure_streak; self.pressure_streak=before+1 if cand else 0; current=bool(cand and self.pressure_streak>=2)
        row={"at_interval":int(t),"m":m,"epoch":self.deployment_epoch,"active_ids":list(self.active_ids),"eligible":eligible,
          "matured_current_epoch":matured,"R":R,"P":P,"positive_support":pos,"negative_support":neg,"candidate":cand,
          "streak_before":before,"streak_after":self.pressure_streak,"current_pressure":current}
        self._record("pressure_checks",self.pressure_checks,row); return row

    def _update_sleep(self,t):
        m=int(t)-2; rows=[]; eligible=[]
        for eid in sorted(self.active_ids):
            r=self.tracker.check(eid,m); row={"at_interval":int(t),"epoch":self.deployment_epoch,"expert_id":eid,**r}
            self._record("utility_checks",self.utility_checks,row); rows.append(row)
            if r.get("eligible_three"): eligible.append(row)
        if not eligible: return None,rows
        eligible.sort(key=lambda r:(float(r["score"]),int(r["expert_id"])))
        return int(eligible[0]["expert_id"]),rows

    def _sleep(self,eid,t,score):
        before=list(self.active_ids); row=self._begin_action("sleep",t,{"expert_id":int(eid),"active_ids_before":before})
        self._make_dormant(eid,t,"nonuseful_three_checks",score); self._topology_cleanup(t,"sleep",before)
        self._commit_action(row,{"active_ids_after":list(self.active_ids)}); self.save_checkpoint("post_sleep_t%d"%t,True)

    def _start_reuse(self,t):
        if self.reuse_slot is not None or self.reuse_started>=32 or len(self.active_ids)>=2: return False
        if self._epoch_matured(int(t)-2)<128: return False
        dormant=self.dormant_ids()
        if not dormant: return False
        hashes={eid:self._model_hash(self.experts[eid]) for eid in dormant}
        self.reuse_slot={"start_t":int(t),"prediction_start":int(t)+1,"proposal_epoch":self.deployment_epoch,
          "candidate_ids":dormant,"candidate_hashes":hashes,"issued":{},"settled_rows":{},"ready":False}
        self.reuse_started+=1
        row={"event":"reuse_start","at_interval":int(t),"epoch":self.deployment_epoch,"candidate_ids":dormant,
          "candidate_hashes":hashes,"active_ids":list(self.active_ids),"slot_number":self.reuse_started,"prediction_start":int(t)+1}
        self._record("reuse_decisions",self.reuse_decisions,row); self._defer_control_checkpoint("reuse_start_t%d"%t,False); return True

    def _evaluate_reuse(self,t):
        s=self.reuse_slot
        if s is None or not s.get("ready"): return False
        if s["proposal_epoch"]!=self.deployment_epoch: self._cancel_reuse(t,"epoch_changed"); return False
        keys=sorted(s["settled_rows"])
        if len(keys)!=32: raise AssertionError("reuse ready count")
        live=np.stack([s["settled_rows"][i]["live_margin"] for i in keys]); bm=np.stack([s["settled_rows"][i]["B_margin"] for i in keys]); y=np.stack([s["settled_rows"][i]["y"] for i in keys])
        rows=[]
        for eid in s["candidate_ids"]:
            cm=np.stack([s["settled_rows"][i]["candidates"][int(eid)]["margin"] for i in keys])
            q=c.reuse_evidence(cm,live,bm,y); q.update({"candidate_id":int(eid),"candidate_hash":s["candidate_hashes"][int(eid)]})
            rows.append(q)
        passing=[r for r in rows if r["pass"]]; winner=None if not passing else sorted(passing,key=lambda r:(float(r["candidate"]["bce"]),int(r["candidate_id"])))[0]
        ev={"event":"reuse_evaluate","at_interval":int(t),"decision_t":int(t),"start_t":s["start_t"],"prediction_start":s["prediction_start"],
          "epoch":self.deployment_epoch,"intervals":[keys[0],keys[-1]+1],"candidate_ids":list(s["candidate_ids"]),
          "candidate_hashes":copy.deepcopy(s["candidate_hashes"]),"candidates":rows,"winner":None if winner is None else int(winner["candidate_id"])}
        self.reuse_history.append(copy.deepcopy(ev)); self._record("reuse_decisions",self.reuse_decisions,ev)
        all_quality_rejected=bool(rows and all(r.get("support") and r.get("finite") and not r.get("pass") for r in rows))
        if all_quality_rejected:
            self.last_reuse_quality_failure={"eval_t":int(t),"epoch":self.deployment_epoch,"candidate_ids":list(s["candidate_ids"]),
              "candidate_hashes":copy.deepcopy(s["candidate_hashes"]),"intervals":[keys[0],keys[-1]+1]}
        if winner is None:
            self.reuse_slot=None; self._defer_control_checkpoint("reuse_reject_t%d"%t,False); return False
        eid=int(winner["candidate_id"]); before=list(self.active_ids); self.reuse_slot=None
        self._cancel_shadow(t,"cancelled_by_reuse_accept"); self._make_active(eid,t,"reuse_accept"); self._topology_cleanup(t,"reuse_accept",before,keep_reuse=True)
        self.save_checkpoint("reuse_accept_t%d"%t,True); return True

    def _evaluate_shadow(self,t):
        s=self.shadow
        if s is None or s.get("status")!="validating" or not s.get("ready"): return False
        if s["proposal_epoch"]!=self.deployment_epoch: self._cancel_shadow(t,"epoch_changed_before_decision"); return False
        keys=sorted(s["settled_rows"])
        if len(keys)!=32: raise AssertionError("shadow ready count")
        cm=np.stack([s["settled_rows"][i]["candidate_margin"] for i in keys]); live=np.stack([s["settled_rows"][i]["live_margin"] for i in keys])
        bm=np.stack([s["settled_rows"][i]["B_margin"] for i in keys]); y=np.stack([s["settled_rows"][i]["y"] for i in keys])
        q=c.qualify_margin(cm,live,bm,y); q.update({"event":"shadow_evaluate","at_interval":int(t),"candidate_id":int(s["id"]),
          "intervals":[keys[0],keys[-1]+1],"updates":int(s["updates"]),"retained_ids":list(s["retained_ids"]),"proposal_epoch":s["proposal_epoch"],
          "created_by_reclamation":bool(s.get("created_by_reclamation")),"reclaimed_id":s.get("reclaimed_id")})
        self._record("candidate_decisions",self.candidate_decisions,q); self._record("qualification_records",self.qualification_records,copy.deepcopy(q))
        eid=int(s["id"]); self.last_attempt_end_m=int(t)-2
        if q["pass"]:
            before=list(self.active_ids); s["accepted"]=True; s["role"]="active"
            for k in ("status","updates","validation_start","issued","settled_rows","ready","proposal_epoch","retained_ids","candidate_id"):
                s.pop(k,None)
            self.experts[eid]=s; self.shadow=None; self._make_active(eid,t,"shadow_accept"); self._topology_cleanup(t,"shadow_accept",before,keep_shadow=True)
            self.save_checkpoint("shadow_accept_t%d"%t,True); return True
        self.shadow=None; self._defer_control_checkpoint("shadow_reject_t%d"%t,True); return False

    def _cooldown_ok(self,t):
        if self.attempts_after_e0==0 or self.last_attempt_end_m is None: return True
        return int(t)-2-int(self.last_attempt_end_m)>=256

    def _prior_reuse_allows_birth(self,t):
        dormant=self.dormant_ids()
        if not dormant: return True
        f=self.last_reuse_quality_failure
        if not f or int(t)-int(f["eval_t"])>64 or int(f["epoch"])!=self.deployment_epoch: return False
        hashes={eid:self._model_hash(self.experts[eid]) for eid in dormant}
        return f["candidate_ids"]==dormant and f["candidate_hashes"]==hashes

    def _tail_feasible(self,t):
        return int(t)+287 < int(self.src["n"])

    def _birth_noncapacity(self,t,pressure):
        return {"pressure":bool(pressure and pressure.get("current_pressure")),"active_vacancy":len(self.active_ids)<2,
          "shadow_free":self.shadow is None,"reuse_free":self.reuse_slot is None,"attempt_budget":self.attempts_after_e0<4,
          "id_budget":self.next_id<5,"cooldown":self._cooldown_ok(t),"reuse_evidence":self._prior_reuse_allows_birth(t),"tail_feasible":self._tail_feasible(t)}

    def _reclamation_table(self,t):
        dormant=self.dormant_ids(); current_hashes={eid:self._model_hash(self.experts[eid]) for eid in dormant}
        rows=[]; eligible=[]
        for eid,x in sorted(self.experts.items()):
            row={"expert_id":int(eid),"state":x["role"],"eligible":False,"reasons":[],"last_active_prediction":x.get("last_active_prediction")}
            if x["role"]!="dormant": row["reasons"].append("state_not_dormant"); rows.append(row); continue
            ds=x.get("dormant_since_prediction"); dp=0 if ds is None else max(0,int(t)+1-int(ds)); row["dormant_predictions"]=dp
            if dp<256: row["reasons"].append("protection_period")
            if self.reuse_slot is not None and eid in self.reuse_slot.get("candidate_ids",[]): row["reasons"].append("pending_reuse_reference")
            hist=[h for h in self.reuse_history if int(h["epoch"])==self.deployment_epoch and h["candidate_ids"]==dormant and h["candidate_hashes"]==current_hashes and int(h["decision_t"])<=int(t)]
            hist=hist[-2:]; row["evidence_count"]=len(hist); row["evidence"]=[]
            if len(hist)<2: row["reasons"].append("need_two_windows")
            else:
                a,b=hist
                if int(a["intervals"][1])>int(b["intervals"][0]): row["reasons"].append("overlap")
                if int(a["intervals"][0])<int(t)-255 or int(b["intervals"][0])<int(t)-255: row["reasons"].append("stale_window")
                if not (0<=int(t)-int(b["decision_t"])<=64): row["reasons"].append("latest_decision_age")
                for h in hist:
                    q=next((q for q in h["candidates"] if int(q["candidate_id"])==eid),None)
                    row["evidence"].append(q)
                    if q is None or not q.get("support") or not q.get("finite"): row["reasons"].append("support_unknown")
                    elif q.get("pass"): row["reasons"].append("reuse_passed")
                    elif q.get("positive_intervals",0)<4: row["reasons"].append("positive_intervals")
                    elif q.get("Q") is None or q.get("Q_pos") is None or q.get("Q_neg") is None or q["Q"]>0 or q["Q_pos"]>0 or q["Q_neg"]>0:
                        row["reasons"].append("positive_contribution")
            row["eligible"]=len(row["reasons"])==0
            rows.append(row)
            if row["eligible"]: eligible.append(row)
        if self.shadow is not None:
            rows.append({"expert_id":int(self.shadow["id"]),"state":"shadow","eligible":False,"reasons":["state_not_dormant"]})
        eligible.sort(key=lambda r:(10**18 if r["last_active_prediction"] is None else int(r["last_active_prediction"]),int(r["expert_id"])))
        return (None if not eligible else int(eligible[0]["expert_id"])),rows

    def _new_shadow_object(self,eid,t,reclaimed_id=None):
        s=c.new_expert(eid,t,"shadow")
        if any(torch.count_nonzero(p).item() for p in s["model"].parameters()): raise AssertionError("shadow not zero")
        s.update({"status":"training","updates":0,"validation_start":None,"issued":{},"settled_rows":{},"ready":False,
          "candidate_id":eid,"proposal_epoch":self.deployment_epoch,"retained_ids":list(self.active_ids),
          "created_by_reclamation":reclaimed_id is not None,"reclaimed_id":reclaimed_id})
        return s

    def _start_shadow_vacancy(self,t):
        if self.resident_count()>=3: return False
        eid=self.next_id; self.next_id+=1; self.attempts_after_e0+=1; self.shadow=self._new_shadow_object(eid,t)
        ev={"event":"shadow_start","at_interval":int(t),"candidate_id":eid,"attempt":self.attempts_after_e0,"proposal_epoch":self.deployment_epoch,
          "retained_ids":list(self.active_ids),"created_by_reclamation":False,"resident_ids_before":sorted(self.experts),
          "resident_ids_after":sorted(list(self.experts)+[eid]),"model_hash":self._model_hash(self.shadow),"optimizer_state_empty":len(self.shadow["optimizer"].state)==0}
        self._record("candidate_decisions",self.candidate_decisions,ev); self._defer_control_checkpoint("shadow_start_t%d"%t,True); return True

    def _reclaim_and_start_shadow(self,t,victim,table):
        if self.arm!="D_bounded" or victim is None or self.resident_count()!=3: return False
        if victim not in self.experts or self.experts[victim]["role"]!="dormant": raise AssertionError("illegal reclaim")
        before_ids=sorted(self.experts); x=self.experts[victim]; released=c.tensor_bytes(x)
        row=self._begin_action("reclaim_and_create_shadow",t,{"delete_id":int(victim),"resident_ids_before":before_ids,"next_id":self.next_id,
          "attempt_before":self.attempts_after_e0})
        tomb={"expert_id":int(victim),"deleted_at_interval":int(t),"role_before":"dormant","last_active_prediction":x.get("last_active_prediction"),
          "dormant_since_prediction":x.get("dormant_since_prediction"),"model_hash":self._model_hash(x),"optimizer_hash":self._opt_hash(x),
          "optimizer_step":c.opt_step(x["optimizer"]),"released_tensor_bytes":released,"evidence_table":copy.deepcopy(table)}
        del self.experts[victim]; self.deleted_ids.add(int(victim)); self.tombstones.append(tomb); self.permanent_deletions+=1
        if self.crash_probe=="after_reclaim_before_shadow": raise RuntimeError("injected_crash_after_reclaim_before_shadow")
        eid=self.next_id; self.next_id+=1; self.attempts_after_e0+=1; self.shadow=self._new_shadow_object(eid,t,reclaimed_id=victim)
        ev={"event":"permanent_reclaim_and_shadow_start","at_interval":int(t),"deleted_id":int(victim),"candidate_id":eid,
          "active_ids_before":list(self.active_ids),"active_ids_after":list(self.active_ids),"resident_ids_before":before_ids,
          "resident_ids_after":sorted(list(self.experts)+[eid]),"released_tensor_bytes":released,"attempt":self.attempts_after_e0,
          "candidate_model_hash":self._model_hash(self.shadow),"candidate_optimizer_empty":len(self.shadow["optimizer"].state)==0,
          "first_candidate_prediction":None,"table":copy.deepcopy(table)}
        self._record("reclamation",self.reclamation_events,ev)
        ce={"event":"shadow_start","at_interval":int(t),"candidate_id":eid,
          "attempt":self.attempts_after_e0,"proposal_epoch":self.deployment_epoch,"retained_ids":list(self.active_ids),
          "created_by_reclamation":True,"reclaimed_id":int(victim),"model_hash":self._model_hash(self.shadow),"optimizer_state_empty":True}
        self._record("candidate_decisions",self.candidate_decisions,ce)
        if self.crash_probe=="after_shadow_before_reclaim_commit": raise RuntimeError("injected_crash_after_shadow_before_reclaim_commit")
        self._commit_action(row,{"deleted_id":int(victim),"candidate_id":eid,"attempt_after":self.attempts_after_e0,"resident_after":self.resident_count()})
        self._defer_control_checkpoint("reclaim_create_t%d"%t,True); return True

    def _attempt_birth(self,t,pressure):
        gates=self._birth_noncapacity(t,pressure); ok=all(gates.values())
        if not ok:
            self._record("control",self.opportunity_log,{"event":"birth_gate_blocked","at_interval":int(t),"gates":gates,"resident":self.resident_count(),"active_ids":list(self.active_ids)}); return False
        if self.resident_count()<3: return self._start_shadow_vacancy(t)
        victim,table=self._reclamation_table(t)
        row={"event":"capacity_full_birth_opportunity","at_interval":int(t),"arm":self.arm,"would_delete_id":victim,"table":table,
          "active_ids":list(self.active_ids),"resident_ids":sorted(self.experts),"attempts_after_e0":self.attempts_after_e0}
        self._record("reclamation",self.reclamation_events,copy.deepcopy(row))
        if self.arm=="D_no_gc":
            row2={"event":"capacity_blocked_no_gc","at_interval":int(t),"would_delete_id":victim,"attempt_consumed":False}
            self._record("reclamation",self.reclamation_events,row2); return False
        if victim is None:
            row2={"event":"capacity_blocked_no_safe_eviction","at_interval":int(t),"attempt_consumed":False,"table":table}
            self._record("reclamation",self.reclamation_events,row2); return False
        return self._reclaim_and_start_shadow(t,victim,table)

    def _defer_control_checkpoint(self,reason,named=False):
        self.deferred_control_checkpoints.append((str(reason),bool(named)))

    def _flush_deferred_control_checkpoints(self):
        pending=list(self.deferred_control_checkpoints); self.deferred_control_checkpoints=[]
        for reason,named in pending:
            self.save_checkpoint(reason,named)

    def _control(self,t):
        if not self._due16(t): return False
        t0=time.process_time(); transitioned=False
        if not self.experts and self.next_id==0:
            r=self._first_birth_check(t)
            if r and r["streak_after"]>=2: self._first_birth(t); transitioned=True
            
        if not transitioned and self.reuse_slot is not None and self.reuse_slot.get("ready"): transitioned=self._evaluate_reuse(t)
        if not transitioned and self.shadow is not None and self.shadow.get("status")=="validating" and self.shadow.get("ready"): transitioned=self._evaluate_shadow(t)
        selected=None; table=[]
        if not transitioned and self.active_ids:
            selected,table=self._update_sleep(t)
            self._record("sleep_tables",self.sleep_tables,{"at_interval":int(t),"epoch":self.deployment_epoch,"active_ids_before":list(self.active_ids),"rows":copy.deepcopy(table),"selected":selected})
            if selected is not None:
                score=next(r for r in table if int(r["expert_id"])==selected); self._sleep(selected,t,score); transitioned=True
        if transitioned:
            self.controller_cpu_seconds+=time.process_time()-t0; self._flush_deferred_control_checkpoints(); return True
        pressure=self._pressure_check(t)
        birth_started=self._attempt_birth(t,pressure) if pressure and pressure.get("current_pressure") else False
        reuse_started=False
        if not birth_started and self._due64(t) and self.reuse_slot is None: reuse_started=self._start_reuse(t)
        row={"event":"due16","at_interval":int(t),"epoch":self.deployment_epoch,"pressure":None if pressure is None else bool(pressure.get("current_pressure")),
          "birth_started":birth_started,"reuse_started":reuse_started,"active_ids":list(self.active_ids),"resident_ids":sorted(self.experts),
          "resident":self.resident_count(),"attempts_after_e0":self.attempts_after_e0,"cooldown_ok":self._cooldown_ok(t)}
        self._record("control",self.opportunity_log,row); self.controller_cpu_seconds+=time.process_time()-t0; self._flush_deferred_control_checkpoints(); return False

    def _batch(self,t):
        if t not in self.src["core"]["by_t"]: return None
        b=[int(i) for i in self.src["core"]["by_t"][t]["batch_indices"]]
        if b and max(b)+2>t: raise AssertionError("immature batch")
        return b

    def _live_update(self,t):
        batch=self._batch(t)
        if batch is None: self.pending_train_context=None; return False
        z=torch.from_numpy(self.src["core"]["T"]["z"][batch].astype(np.float32)); bm=torch.from_numpy(c.bmargin(self.src["B"],batch).astype(np.float32))
        y=torch.from_numpy((self.src["B"]["labels"][batch]>0).astype(np.float32)); active=list(self.active_ids); deltas={}; total=torch.zeros_like(bm)
        for eid in active:
            d=self.experts[eid]["model"].delta(z); deltas[eid]=d; total=total+d; self.live_training_forwards+=1
        retained=[] if self.shadow is None or self.shadow.get("status")!="training" else list(self.shadow["retained_ids"])
        bg=torch.zeros_like(bm)
        for eid in retained:
            if eid not in deltas: raise AssertionError("shadow retained active mismatch")
            bg=bg+deltas[eid]
        self.pending_train_context={"at_interval":int(t),"batch_indices":batch,"retained_ids":retained,
          "B_margin":bm.detach().cpu().numpy().astype(np.float32),"background_delta":bg.detach().cpu().numpy().astype(np.float32)}
        if not active: return False
        if not self.allow_gradient: raise AssertionError("real engineering prefix gradient forbidden")
        if self.live_optimizer_steps+len(active)>744: raise AssertionError("live optimizer budget")
        row=self._begin_action("joint_live_optimizer_steps",t,{"expert_ids":active,"batch_indices":batch})
        before={eid:{"model":self._model_hash(self.experts[eid]),"opt":self._opt_hash(self.experts[eid]),"step":c.opt_step(self.experts[eid]["optimizer"])} for eid in active}
        for eid in active: self.experts[eid]["optimizer"].zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(bm+total,y); reg=sum((d*d).mean() for d in deltas.values())*.001; objective=loss+reg
        t0=time.perf_counter(); objective.backward(); gns={}
        for eid in active: gns[eid]=float(torch.nn.utils.clip_grad_norm_(self.experts[eid]["model"].parameters(),1.0))
        for j,eid in enumerate(active):
            self.experts[eid]["optimizer"].step(); self.actual_optimizer_calls+=1; self.experts[eid]["version"]+=1; self.live_optimizer_steps+=1
            if self.crash_probe=="after_first_joint_step" and j==0: raise RuntimeError("injected_crash_after_first_joint_step")
        self.update_seconds+=time.perf_counter()-t0
        for eid in active:
            x=self.experts[eid]; ur={"at_interval":int(t),"kind":"live","expert_id":eid,"joint_active_ids":active,"batch_indices":batch,
              "loss":float(loss.detach()),"regularizer":float(reg.detach()),"objective":float(objective.detach()),"grad_norm":gns[eid],
              "model_hash_before":before[eid]["model"],"model_hash_after":self._model_hash(x),"optimizer_hash_before":before[eid]["opt"],
              "optimizer_hash_after":self._opt_hash(x),"optimizer_step_before":before[eid]["step"],"optimizer_step_after":c.opt_step(x["optimizer"]),"version_after":x["version"]}
            self._record("updates",self.update_log,ur)
        if self.crash_probe=="after_audit_before_action_commit":
            raise RuntimeError("injected_crash_after_audit_before_action_commit")
        self._commit_action(row,{"optimizer_calls":len(active),"live_steps_after":self.live_optimizer_steps})
        self.save_checkpoint("live_update_t%d"%t,False)
        return True

    def _shadow_update(self,t):
        s=self.shadow; batch=self._batch(t)
        if s is None or s.get("status")!="training" or batch is None: self.pending_train_context=None; return False
        if not self.allow_gradient: raise AssertionError("real engineering prefix shadow gradient forbidden")
        if self.shadow_optimizer_steps>=64 or s["updates"]>=16: raise AssertionError("shadow budget")
        ctx=self.pending_train_context
        if ctx is None or ctx["at_interval"]!=int(t) or ctx["batch_indices"]!=batch or ctx["retained_ids"]!=list(s["retained_ids"]): raise AssertionError("missing detached shadow context")
        z=torch.from_numpy(self.src["core"]["T"]["z"][batch].astype(np.float32)); bm=torch.from_numpy(ctx["B_margin"]); bg=torch.from_numpy(ctx["background_delta"]).detach()
        y=torch.from_numpy((self.src["B"]["labels"][batch]>0).astype(np.float32)); d=s["model"].delta(z); self.shadow_training_forwards+=1
        s["optimizer"].zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(bm+bg+d,y); reg=.001*(d*d).mean(); obj=loss+reg
        row=self._begin_action("shadow_optimizer_step",t,{"expert_id":int(s["id"]),"batch_indices":batch,"update_number":int(s["updates"])+1})
        before={"model":self._model_hash(s),"opt":self._opt_hash(s),"step":c.opt_step(s["optimizer"])}
        t0=time.perf_counter(); obj.backward(); gn=float(torch.nn.utils.clip_grad_norm_(s["model"].parameters(),1.0)); s["optimizer"].step()
        self.actual_optimizer_calls+=1; self.update_seconds+=time.perf_counter()-t0; s["version"]+=1; s["updates"]+=1; self.shadow_optimizer_steps+=1
        ev={"at_interval":int(t),"kind":"shadow","expert_id":int(s["id"]),"batch_indices":batch,"retained_ids":list(s["retained_ids"]),
          "loss":float(loss.detach()),"regularizer":float(reg.detach()),"objective":float(obj.detach()),"grad_norm":gn,
          "model_hash_before":before["model"],"model_hash_after":self._model_hash(s),"optimizer_hash_before":before["opt"],
          "optimizer_hash_after":self._opt_hash(s),"optimizer_step_before":before["step"],"optimizer_step_after":c.opt_step(s["optimizer"]),
          "version_after":s["version"],"shadow_update_after":s["updates"]}
        self._record("updates",self.update_log,ev)
        if s["updates"]==16:
            s["status"]="validating"; s["validation_start"]=int(t)+1; s["issued"]={}; s["settled_rows"]={}; s["ready"]=False
            q={"event":"shadow_training_complete","at_interval":int(t),"candidate_id":int(s["id"]),"updates":16,"validation_start":int(t)+1,
              "model_hash":self._model_hash(s),"optimizer_step":c.opt_step(s["optimizer"])}
            self._record("candidate_decisions",self.candidate_decisions,q)
        self._commit_action(row,{"shadow_updates_after":int(s["updates"]),"status_after":s["status"],"optimizer_step_after":c.opt_step(s["optimizer"])})
        self.pending_train_context=None
        named=s["updates"] in (1,15,16); self.save_checkpoint("shadow_%02d_t%d"%(s["updates"],t),named); return True

    def advance(self,stop=None):
        stop=self.src["n"] if stop is None else min(int(stop),int(self.src["n"]))
        while self.cursor<stop:
            t=int(self.cursor)
            if self.next_substep=="predict": self.next_substep="settle"; self._predict(t)
            elif self.next_substep=="settle": self.next_substep="control"; self._settle(t)
            elif self.next_substep=="control": self.next_substep="live_update"; self._control(t)
            elif self.next_substep=="live_update": self.next_substep="shadow_update"; self._live_update(t)
            elif self.next_substep=="shadow_update": self.next_substep="finish"; self._shadow_update(t)
            elif self.next_substep=="finish":
                if len(self.active_ids)>2 or self.resident_count()>3 or self.next_id>5 or self.permanent_deletions>4: raise AssertionError("capacity/id invariant")
                if self.shadow is not None and self.shadow["id"] in self.experts: raise AssertionError("shadow resident duplicate")
                if any(e in self.deleted_ids for e in self.experts): raise AssertionError("deleted id reloaded")
                if any(self.experts[e]["role"]!="active" for e in self.active_ids): raise AssertionError("active role mismatch")
                self.cursor+=1; self.next_substep="predict"; self.pending_train_context=None; self._resource_sample()
                if self.cursor%512==0: self.save_checkpoint("cursor_%04d"%self.cursor,True)
            else: raise AssertionError("bad substep")
        return self

    def terminal_settle(self):
        counters=(self.live_optimizer_steps,self.shadow_optimizer_steps,self.deployed_forwards,self.reuse_preview_forwards,self.shadow_preview_forwards,self.actual_optimizer_calls,self.permanent_deletions)
        if self.terminal_progress==0:
            self._settle(self.src["n"]); self.terminal_progress=1; self.next_substep="terminal_second_settle"; self.save_checkpoint("terminal_first_settle",True)
        if self.terminal_progress==1:
            self._settle(self.src["n"]+1); self.terminal_progress=2; self.next_substep="terminal_censor"; self.save_checkpoint("terminal_second_settle",True)
        if self.terminal_progress==2:
            if self.reuse_slot is not None:
                ev={"event":"reuse_censored","at_interval":self.src["n"]+1,"start_t":self.reuse_slot["start_t"],"issued_count":len(self.reuse_slot.get("issued",{})),"settled_count":len(self.reuse_slot.get("settled_rows",{}))}
                self._record("reuse_decisions",self.reuse_decisions,ev); self.reuse_slot=None
            if self.shadow is not None:
                ev={"event":"shadow_censored","at_interval":self.src["n"]+1,"candidate_id":int(self.shadow["id"]),"status":self.shadow.get("status"),"updates":int(self.shadow.get("updates",0)),"issued_count":len(self.shadow.get("issued",{}))}
                self._record("candidate_decisions",self.candidate_decisions,ev)
            self.terminal_progress=3; self.next_substep="done"; self.save_checkpoint("terminal_done",True)
        after=(self.live_optimizer_steps,self.shadow_optimizer_steps,self.deployed_forwards,self.reuse_preview_forwards,self.shadow_preview_forwards,self.actual_optimizer_calls,self.permanent_deletions)
        self.terminal_counter_delta=[after[i]-counters[i] for i in range(len(after))]
        if any(self.terminal_counter_delta): raise AssertionError("terminal changed counters")
        return self

    def snapshot(self):
        self._flush_arrays()
        return {"protocol":"044","revision":1,"arm":self.arm,"cursor":self.cursor,"next_substep":self.next_substep,"terminal_progress":self.terminal_progress,
          "deployment_epoch":self.deployment_epoch,"epoch_start_prediction":self.epoch_start_prediction,
          "experts":{int(k):c.pack_expert(v) for k,v in self.experts.items()},"active_ids":list(self.active_ids),
          "shadow":None if self.shadow is None else c.pack_expert(self.shadow),"next_id":self.next_id,"deleted_ids":sorted(self.deleted_ids),"tombstones":copy.deepcopy(self.tombstones),
          "attempts_after_e0":self.attempts_after_e0,"last_attempt_end_m":self.last_attempt_end_m,"first_birth_t":self.first_birth_t,
          "birth_streak":self.birth_streak,"pressure_streak":self.pressure_streak,"reuse_slot":copy.deepcopy(self.reuse_slot),"reuse_started":self.reuse_started,
          "reuse_history":list(copy.deepcopy(self.reuse_history)),"last_reuse_quality_failure":copy.deepcopy(self.last_reuse_quality_failure),
          "pending_train_context":copy.deepcopy(self.pending_train_context),"tracker":copy.deepcopy(self.tracker),
          "tails":{"birth_checks":list(self.birth_checks),"pressure_checks":list(self.pressure_checks),"utility_checks":list(self.utility_checks),
            "sleep_tables":list(self.sleep_tables),"lifecycle_events":list(self.lifecycle_events),"candidate_decisions":list(self.candidate_decisions),
            "reuse_decisions":list(self.reuse_decisions),"opportunity_log":list(self.opportunity_log),"update_log":list(self.update_log),
            "settlement_log":list(self.settlement_log),"qualification_records":list(self.qualification_records),"reclamation_events":list(self.reclamation_events)},
          "audit_state":copy.deepcopy(self.audit_state),
          "live_optimizer_steps":self.live_optimizer_steps,"shadow_optimizer_steps":self.shadow_optimizer_steps,"deployed_forwards":self.deployed_forwards,
          "reuse_preview_forwards":self.reuse_preview_forwards,"shadow_preview_forwards":self.shadow_preview_forwards,"live_training_forwards":self.live_training_forwards,
          "shadow_training_forwards":self.shadow_training_forwards,"controller_cpu_seconds":self.controller_cpu_seconds,"prediction_seconds":self.prediction_seconds,
          "update_seconds":self.update_seconds,"io_seconds":self.io_seconds,"max_active_seen":self.max_active_seen,"max_resident_seen":self.max_resident_seen,
          "peak_resident_tensor_bytes":self.peak_resident_tensor_bytes,"permanent_deletions":self.permanent_deletions,"actual_optimizer_calls":self.actual_optimizer_calls,
          "near_threshold_recompute_count":self.near_threshold_recompute_count,"action_seq":self.action_seq,
          "completed_action_ids":list(self.completed_action_ids),"completed_action_count":self.completed_action_count,
          "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate()}

    def online_payload_bytes(self):
        import pickle
        x=self.snapshot()
        return len(pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL))

    def save_checkpoint(self,reason,named=False):
        if self.checkpoint_dir is None: return
        d=Path(self.checkpoint_dir); d.mkdir(parents=True,exist_ok=True); snap=self.snapshot(); tmp=d/"latest.pt.tmp"; final=d/"latest.pt"
        torch.save(snap,tmp)
        if self.crash_probe=="checkpoint_before_atomic_replace": raise RuntimeError("injected_crash_checkpoint_before_atomic_replace")
        os.replace(tmp,final)
        meta={"protocol":"044","revision":1,"arm":self.arm,"cursor":self.cursor,"next_substep":self.next_substep,"terminal_progress":self.terminal_progress,
          "reason":reason,"live_steps":self.live_optimizer_steps,"shadow_steps":self.shadow_optimizer_steps,"actual_optimizer_calls":self.actual_optimizer_calls,
          "attempts_after_e0":self.attempts_after_e0,"permanent_deletions":self.permanent_deletions,"action_seq":self.action_seq,
          "completed_action_count":self.completed_action_count,"online_payload_bytes":self.online_payload_bytes(),
          "audit_state":copy.deepcopy(self.audit_state),"sha256":sha256_file(final)}
        c.W(d/"latest.json",meta)
        if self.crash_probe=="checkpoint_after_atomic_replace": raise RuntimeError("injected_crash_checkpoint_after_atomic_replace")
        if named:
            safe="".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in reason); q=d/(safe+".pt"); qtmp=d/(safe+".pt.tmp")
            torch.save(snap,qtmp); os.replace(qtmp,q); c.W(str(q)+".json",{**meta,"sha256":sha256_file(q)})
        if self.engineering_stop_prefix and str(reason).startswith(str(self.engineering_stop_prefix)):
            raise EngineeringCheckpointStop(str(reason))

    @classmethod
    def restore(cls,src,work_dir,path,arm=None,allow_gradient=True,real_science=False,strict_journal=True):
        x=torch.load(path,map_location="cpu",weights_only=False)
        if (x.get("protocol"),x.get("revision"))!=("044",1): raise AssertionError("bad checkpoint")
        if arm is not None and x.get("arm")!=arm: raise AssertionError("arm mismatch")
        m=cls(src,work_dir,x["arm"],allow_gradient=allow_gradient,real_science=real_science,resume=True)
        simple=["cursor","next_substep","terminal_progress","deployment_epoch","epoch_start_prediction","active_ids","next_id","deleted_ids","tombstones",
          "attempts_after_e0","last_attempt_end_m","first_birth_t","birth_streak","pressure_streak","reuse_slot","reuse_started",
          "last_reuse_quality_failure","pending_train_context","tracker",
          "live_optimizer_steps","shadow_optimizer_steps","deployed_forwards","reuse_preview_forwards","shadow_preview_forwards","live_training_forwards",
          "shadow_training_forwards","controller_cpu_seconds","prediction_seconds","update_seconds","io_seconds","max_active_seen","max_resident_seen",
          "peak_resident_tensor_bytes","permanent_deletions","actual_optimizer_calls","near_threshold_recompute_count","action_seq","completed_action_count"]
        for k in simple: setattr(m,k,copy.deepcopy(x[k]))
        m.deleted_ids=set(m.deleted_ids); m.reuse_history=deque(copy.deepcopy(x["reuse_history"]),maxlen=2)
        m.completed_action_ids=deque(copy.deepcopy(x["completed_action_ids"]),maxlen=64)
        tails=x["tails"]
        for k in ("birth_checks","pressure_checks","utility_checks","sleep_tables","lifecycle_events","candidate_decisions","reuse_decisions",
                  "opportunity_log","update_log","settlement_log","qualification_records","reclamation_events"):
            setattr(m,k,deque(copy.deepcopy(tails[k]),maxlen=64))
        m.experts={int(k):c.unpack_expert(v) for k,v in x["experts"].items()}; m.shadow=None if x["shadow"] is None else c.unpack_expert(x["shadow"])
        m.audit_state=copy.deepcopy(x["audit_state"]); m._verify_stream_prefixes()
        torch.set_rng_state(x["torch_rng"]); np.random.set_state(x["numpy_rng"]); random.setstate(x["python_rng"])
        if strict_journal and m.action_journal.exists():
            j=c.J(m.action_journal)
            if j.get("status")=="pending": raise RuntimeError("ambiguous_state: pending action journal")
            if int(j.get("action_seq",0))>m.action_seq: raise RuntimeError("ambiguous_state: journal newer than checkpoint")
        return m

def _arr_prefix_hash(a,cursor,axis0=True):
    x=np.asarray(a[:int(cursor)] if axis0 else a[:, :int(cursor)])
    return hashlib.sha256(x.tobytes()).hexdigest()

def semantic_state(m):
    npstate=np.random.get_state(); pyst=random.getstate()
    tracker=m.tracker.online_payload() if hasattr(m.tracker,"online_payload") else copy.deepcopy(m.tracker.__dict__)
    # Timing is resource evidence, not logical recovery state; elapsed time may differ after fresh-process restore.
    if isinstance(tracker,dict): tracker.pop("rebuild_seconds",None)
    return {"arm":m.arm,"cursor":m.cursor,"next_substep":m.next_substep,"terminal_progress":m.terminal_progress,"epoch":m.deployment_epoch,
      "epoch_start":m.epoch_start_prediction,"active_ids":list(m.active_ids),"next_id":m.next_id,"deleted_ids":sorted(m.deleted_ids),
      "tombstones":m.tombstones,"attempts":m.attempts_after_e0,"first_birth_t":m.first_birth_t,
      "birth_streak":m.birth_streak,"pressure_streak":m.pressure_streak,"reuse_started":m.reuse_started,
      "reuse_history":list(m.reuse_history),"reuse_slot":m.reuse_slot,"last_reuse_quality_failure":m.last_reuse_quality_failure,
      "pending_train_context":m.pending_train_context,"tracker":tracker,
      "experts":{str(k):{"role":v["role"],"version":v["version"],"model":m._model_hash(v),"opt":m._opt_hash(v),
        "step":c.opt_step(v["optimizer"]),"first_active_t":v.get("first_active_t"),"last_active_prediction":v.get("last_active_prediction"),
        "dormant_since_prediction":v.get("dormant_since_prediction"),"reactivations":v.get("reactivations")} for k,v in sorted(m.experts.items())},
      "shadow":None if m.shadow is None else {"id":m.shadow["id"],"status":m.shadow.get("status"),"updates":m.shadow.get("updates"),
        "model":m._model_hash(m.shadow),"opt":m._opt_hash(m.shadow),"step":c.opt_step(m.shadow["optimizer"]),
        "proposal_epoch":m.shadow.get("proposal_epoch"),"retained_ids":m.shadow.get("retained_ids"),"issued":m.shadow.get("issued"),
        "settled_rows":m.shadow.get("settled_rows"),"ready":m.shadow.get("ready")},
      "counters":[m.live_optimizer_steps,m.shadow_optimizer_steps,m.deployed_forwards,m.reuse_preview_forwards,
        m.shadow_preview_forwards,m.live_training_forwards,m.shadow_training_forwards,m.actual_optimizer_calls,m.permanent_deletions],
      "audit_state":copy.deepcopy(m.audit_state),"completed_action_count":m.completed_action_count,"action_seq":m.action_seq,
      "out_hashes":{
        "probability":_arr_prefix_hash(m.out["probability"],m.cursor),
        "detection_logits":_arr_prefix_hash(m.out["detection_logits"],m.cursor),
        "total_delta":_arr_prefix_hash(m.out["total_delta"],m.cursor),
        "live_margin":_arr_prefix_hash(m.out["live_margin"],m.cursor),
        "B_margin":_arr_prefix_hash(m.out["B_margin"],m.cursor),
        "deployment_epoch":_arr_prefix_hash(m.out["deployment_epoch"],m.cursor),
        "active_ids":_arr_prefix_hash(m.out["active_ids"],m.cursor),
        "expert_versions":_arr_prefix_hash(m.out["expert_versions"],m.cursor),
        "expert_hashes":_arr_prefix_hash(m.out["expert_hashes"],m.cursor),
        "contribution":_arr_prefix_hash(m.out["contribution"],m.cursor,False),
        "settled":_arr_prefix_hash(m.settled,min(m.cursor+2,m.src["n"])),
        "b_loss":_arr_prefix_hash(m.b_loss,min(m.cursor+2,m.src["n"])),
        "d_loss":_arr_prefix_hash(m.d_loss,min(m.cursor+2,m.src["n"]))},
      "rng":{"torch":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "numpy":hashlib.sha256(repr(npstate).encode()).hexdigest(),"python":hashlib.sha256(repr(pyst).encode()).hexdigest()}}
def semantic_digest(m):
    return hashlib.sha256(json.dumps(semantic_state(m),sort_keys=True,default=str).encode()).hexdigest()
