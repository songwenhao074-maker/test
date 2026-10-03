"""Protocol-042 rev2 transactional online engines."""
from __future__ import annotations
import copy, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import run_ftmoe_protocol037 as p37
import run_ftmoe_protocol040 as p40
import run_ftmoe_protocol041 as p41
from protocol035_common import sha256_file, sha256_state_dict
import protocol042_common as c

def _hash_model(x): return sha256_state_dict(x["model"].state_dict())
def _hash_opt(x): return c.opt_digest(x["optimizer"])

class UMachine(p41.Machine041):
    """041 W_parent semantics under the 042 arm name; numerical path is unchanged."""

    def __init__(self,src,work_dir,allow_gradient=True,real_science=False):
        super().__init__(src,work_dir,"W_parent",allow_gradient=allow_gradient,real_science=real_science)
        self.arm042="U_parent"

    def _begin_action(self,kind,t,meta=None):
        self.action_seq+=1
        row={"protocol":"042","revision":2,"arm":"U_parent","action_seq":self.action_seq,"status":"pending",
             "kind":kind,"cursor":int(t),"next_substep":self.next_substep,"meta":meta or {}}
        c.W(self.action_journal,row); return row

    def _commit_action(self,row,extra=None):
        z=dict(row); z["status"]="committed"
        if extra: z.update(extra)
        c.W(self.action_journal,z); self.completed_action_ids.append(int(row["action_seq"]))

    def snapshot(self):
        x=super().snapshot()
        # retain 041-compatible model state while explicitly identifying the 042 wrapper.
        x["protocol"]="042U"; x["revision"]=2; x["arm042"]="U_parent"
        return x

    def save_checkpoint(self,reason,named=False):
        if self.checkpoint_dir is None: return
        d=Path(self.checkpoint_dir); d.mkdir(parents=True,exist_ok=True)
        snap=self.snapshot(); tmp=d/"latest.pt.tmp"; final=d/"latest.pt"
        torch.save(snap,tmp); os.replace(tmp,final)
        meta={"protocol":"042","revision":2,"arm":"U_parent","cursor":self.cursor,"next_substep":self.next_substep,
              "terminal_progress":self.terminal_progress,"reason":reason,"live_steps":self.live_optimizer_steps,
              "shadow_steps":self.shadow_optimizer_steps,"action_seq":self.action_seq,"sha256":sha256_file(final)}
        c.W(d/"latest.json",meta)
        if named:
            safe="".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in reason)
            q=d/(safe+".pt"); torch.save(snap,q); c.W(str(q)+".json",{**meta,"sha256":sha256_file(q)})

    @classmethod
    def restore(cls,src,work_dir,path,allow_gradient=True,real_science=False,strict_journal=True):
        x=torch.load(path,map_location="cpu")
        if x.get("protocol")!="042U" or x.get("arm042")!="U_parent": raise AssertionError("bad U_parent checkpoint")
        # Temporarily translate to the exact 041 snapshot schema and reuse its deterministic restore.
        y=copy.deepcopy(x); y["protocol"]="041"; y["arm"]="W_parent"
        tmp=Path(work_dir)/"_translate041.pt"; Path(work_dir).mkdir(parents=True,exist_ok=True); torch.save(y,tmp)
        base=p41.Machine041.restore(src,work_dir,tmp,arm="W_parent",allow_gradient=allow_gradient,real_science=real_science,strict_journal=False)
        tmp.unlink()
        m=cls(src,work_dir,allow_gradient=allow_gradient,real_science=real_science)
        m.__dict__.update(base.__dict__); m.arm042="U_parent"; m.action_journal=Path(work_dir)/"action_journal.json"
        if strict_journal and m.action_journal.exists():
            j=c.J(m.action_journal)
            if j.get("status")=="pending": raise RuntimeError("ambiguous_step: U pending action")
            if int(j.get("action_seq",0))>int(m.action_seq): raise RuntimeError("ambiguous_step: U journal newer than checkpoint")
        return m

    def _shadow_update_step(self,t):
        s=self.shadow
        if s is None or s.get("status")!="training" or t not in self.src["core"]["by_t"]: return False
        if not self.allow_gradient: raise AssertionError("engineering prefix attempted shadow gradient")
        if self.shadow_optimizer_steps>=16: raise AssertionError("U shadow step budget")
        batch=[int(i) for i in self.src["core"]["by_t"][t]["batch_indices"]]
        if batch and max(batch)+2>t: raise AssertionError("immature batch")
        row=self._begin_action("shadow_optimizer_step",t,{"expert_id":int(s["id"]),"batch_indices":batch,"update_number":int(s["updates"])+1})
        before=self.model_hash(s); ob=self.optimizer_hash(s); t0=time.perf_counter()
        loss,gn,_=p37.do_update(s["model"],s["optimizer"],self.src["core"],batch)
        self.update_seconds+=time.perf_counter()-t0; s["version"]+=1
        self.shadow_optimizer_steps+=1; s["updates"]+=1
        ev={"at_interval":int(t),"kind":"shadow","expert_id":int(s["id"]),"batch_indices":batch,"loss":float(loss),"grad_norm":float(gn),
            "model_hash_before":before,"model_hash_after":self.model_hash(s),"optimizer_hash_before":ob,"optimizer_hash_after":self.optimizer_hash(s),
            "optimizer_step_after":p40.opt_step(s["optimizer"]),"version_after":int(s["version"])}
        self.update_log.append(ev)
        if s["updates"]==1:
            if p40.opt_step(s["optimizer"])!=1: raise AssertionError("U child first Adam step")
            if self.initialization_events:
                self.initialization_events[-1]["candidate_first_optimizer_step_after"]=1
                self.initialization_events[-1]["candidate_hash_after_first_update"]=self.model_hash(s)
        if s["updates"]==16:
            s["status"]="validating"; s["validation_start"]=int(t)+1; s["issued"]={}; s["settled_rows"]={}; s["ready"]=False
            self.candidate_decisions.append({"event":"shadow_training_complete","at_interval":int(t),"candidate_id":int(s["id"]),"updates":16,
              "validation_start":int(t)+1,"model_hash":self.model_hash(s),"optimizer_step":p40.opt_step(s["optimizer"])})
        self._commit_action(row,{"model_hash_after":ev["model_hash_after"],"optimizer_step_after":ev["optimizer_step_after"],
                                 "shadow_updates_after":int(s["updates"]),"status_after":s["status"]})
        self.save_checkpoint(("shadow_training_complete_t%d" if s["updates"]==16 else "shadow_update_t%d")%t,named=s["updates"]==16)
        return True


class NewMachine:
    SUBSTEPS=("predict","settle","control","live_update","shadow_update","finish")

    def __init__(self,src,work_dir,arm,allow_gradient=True,real_science=False):
        if arm not in c.NEW_ARMS: raise ValueError(arm)
        self.src=src; self.arm=arm; self.work_dir=Path(work_dir); self.work_dir.mkdir(parents=True,exist_ok=True)
        self.allow_gradient=bool(allow_gradient); self.real_science=bool(real_science)
        self.cursor=0; self.next_substep="predict"; self.terminal_progress=0
        self.deployment_epoch=0; self.epoch_start_prediction=0
        self.experts={}; self.active_ids=[]; self.shadow=None; self.next_id=0
        self.attempts_after_e0=0; self.last_attempt_end_m=None; self.first_birth_t=None
        self.birth_streak=0; self.pressure_streak=0; self.reuse_slot=None; self.reuse_started=0
        self.last_reuse_quality_failure=None; self.pending_train_context=None
        self.trackers={0:c.UtilityTracker("hist" if arm=="A_hist" else "win128")}
        self.trackers[0].reset(0,0,[],"initial")
        n=int(src["n"])
        self.out={
          "probability":np.full((n,c.HOSTS),np.nan,np.float32),
          "detection_logits":np.full((n,c.HOSTS,2),np.nan,np.float32),
          "total_delta":np.full((n,c.HOSTS),np.nan,np.float32),
          "live_margin":np.full((n,c.HOSTS),np.nan,np.float32),
          "B_margin":np.full((n,c.HOSTS),np.nan,np.float32),
          "deployment_epoch":np.zeros(n,np.int32),
          "active_ids":np.full((n,2),-1,np.int16),
          "active_count":np.zeros(n,np.int8),
          "shadow_present":np.zeros(n,np.int8),
          "accepted_count":np.zeros(n,np.int8),
          "expert_versions":np.full((n,2),-1,np.int32),
          "expert_hashes":np.full((n,2),"",dtype="<U64"),
          "contribution":np.full((3,n,c.HOSTS),np.nan,np.float32),
        }
        self.settled=np.zeros(n,np.int8); self.b_loss=np.full(n,np.nan,np.float64); self.d_loss=np.full(n,np.nan,np.float64)
        self.birth_checks=[]; self.pressure_checks=[]; self.utility_checks=[]; self.victim_tables=[]
        self.lifecycle_events=[]; self.candidate_decisions=[]; self.reuse_decisions=[]; self.opportunity_log=[]
        self.update_log=[]; self.settlement_log=[]; self.qualification_records=[]; self.archive_events=[]
        self.live_optimizer_steps=0; self.shadow_optimizer_steps=0; self.deployed_forwards=0
        self.reuse_preview_forwards=0; self.shadow_preview_forwards=0
        self.live_training_forwards=0; self.shadow_training_forwards=0
        self.controller_cpu_seconds=0.0; self.prediction_seconds=0.0; self.update_seconds=0.0
        self.max_active_seen=0; self.max_resident_seen=0; self.permanent_deletions=0
        self.action_seq=0; self.completed_action_ids=[]; self.checkpoint_dir=None
        self.action_journal=self.work_dir/"action_journal.json"
        self.actual_optimizer_calls=0; self.crash_probe=None

    def tracker(self): return self.trackers[self.deployment_epoch]
    def _due16(self,t): return (int(t)+1)%16==0
    def _due64(self,t): return (int(t)+1)%64==0
    def resident_count(self): return len(self.experts)+(1 if self.shadow is not None else 0)
    def _model_hash(self,x): return _hash_model(x)
    def _opt_hash(self,x): return _hash_opt(x)

    def _begin_action(self,kind,t,meta=None):
        self.action_seq+=1
        row={"protocol":"042","revision":2,"arm":self.arm,"action_seq":self.action_seq,"status":"pending",
             "kind":kind,"cursor":int(t),"next_substep":self.next_substep,"meta":meta or {}}
        c.W(self.action_journal,row); return row

    def _commit_action(self,row,extra=None):
        z=dict(row); z["status"]="committed"
        if extra: z.update(extra)
        c.W(self.action_journal,z); self.completed_action_ids.append(int(row["action_seq"]))

    def _topology_epoch(self,t,reason):
        old=self.deployment_epoch; self.deployment_epoch+=1; self.epoch_start_prediction=int(t)+1
        mode="hist" if self.arm=="A_hist" else "win128"; tr=c.UtilityTracker(mode)
        tr.reset(self.deployment_epoch,self.epoch_start_prediction,self.active_ids,reason); self.trackers[self.deployment_epoch]=tr
        self.pressure_streak=0
        self.lifecycle_events.append({"event":"epoch_transition","at_interval":int(t),"old_epoch":old,"new_epoch":self.deployment_epoch,
                                      "active_ids":list(self.active_ids),"first_affected_prediction":int(t)+1,"reason":reason})

    def _epoch_matured_indices(self,m):
        if m<0: return np.array([],dtype=np.int64)
        ep=self.out["deployment_epoch"][:m+1]
        return np.flatnonzero((ep==self.deployment_epoch)&self.settled[:m+1].astype(bool))

    def _cancel_shadow(self,t,reason):
        if self.shadow is None: return
        self.candidate_decisions.append({"event":"shadow_cancel","at_interval":int(t),"candidate_id":int(self.shadow["id"]),
          "updates":int(self.shadow.get("updates",0)),"status":self.shadow.get("status"),"reason":reason})
        if int(self.shadow["id"])>0: self.last_attempt_end_m=int(t)-2
        self.shadow=None

    def _cancel_reuse(self,t,reason):
        if self.reuse_slot is None: return
        s=self.reuse_slot
        self.reuse_decisions.append({"event":"reuse_cancel","at_interval":int(t),"reason":reason,"start_t":s["start_t"],
          "issued_count":len(s.get("issued",{})),"settled_count":len(s.get("settled_rows",{}))})
        self.reuse_slot=None

    def _topology_cleanup(self,t,reason,keep_shadow=False,keep_reuse=False):
        if not keep_shadow: self._cancel_shadow(t,"topology_"+reason)
        if not keep_reuse: self._cancel_reuse(t,"topology_"+reason)
        self._topology_epoch(t,reason)

    def _make_active(self,eid,t,reason):
        x=self.experts[int(eid)]
        if x["role"]=="dormant":
            if x.get("dormant_model_hash") and self._model_hash(x)!=x["dormant_model_hash"]: raise AssertionError("dormant model changed")
            if x.get("dormant_optimizer_hash") and self._opt_hash(x)!=x["dormant_optimizer_hash"]: raise AssertionError("dormant optimizer changed")
            if x.get("dormant_optimizer_step") is not None and c.opt_step(x["optimizer"])!=x["dormant_optimizer_step"]: raise AssertionError("dormant Adam step changed")
            ds=x.get("dormant_since_prediction"); x["reactivations"]=int(x.get("reactivations",0))+1
            dormant_predictions=0 if ds is None else max(0,int(t)+1-int(ds))
        else: dormant_predictions=0
        x["role"]="active"; x["dormant_since_prediction"]=None
        if x.get("first_active_t") is None: x["first_active_t"]=int(t)+1
        if int(eid) not in self.active_ids: self.active_ids.append(int(eid)); self.active_ids.sort()
        self.lifecycle_events.append({"event":"activate","expert_id":int(eid),"at_interval":int(t),"reason":reason,
          "first_affected_prediction":int(t)+1,"dormant_predictions_before_reactivation":dormant_predictions,
          "model_hash":self._model_hash(x),"optimizer_hash":self._opt_hash(x),"optimizer_step":c.opt_step(x["optimizer"])})

    def _make_dormant(self,eid,t,reason):
        eid=int(eid); x=self.experts[eid]
        if eid in self.active_ids: self.active_ids.remove(eid)
        x["role"]="dormant"; x["dormant_since_prediction"]=int(t)+1
        x["dormant_model_hash"]=self._model_hash(x); x["dormant_optimizer_hash"]=self._opt_hash(x); x["dormant_optimizer_step"]=c.opt_step(x["optimizer"])
        # preserve a historical summary but never use it in current scoring.
        sc=self.tracker().score(eid,int(t)-2)
        x["historical_utility"].append({"epoch":self.deployment_epoch,"at_interval":int(t),"reason":reason,"score":sc})
        self.archive_events.append({"event":"archive_accepted","expert_id":eid,"at_interval":int(t),"reason":reason,
          "model_hash":x["dormant_model_hash"],"optimizer_hash":x["dormant_optimizer_hash"],"optimizer_step":x["dormant_optimizer_step"],"utility":sc})
        self.lifecycle_events.append({"event":"sleep","expert_id":eid,"at_interval":int(t),"reason":reason,"dormant_from_prediction":int(t)+1})

    def _first_birth_check(self,t):
        m=int(t)-2
        if m<319: return None
        recent=np.arange(m-63,m+1); prev=np.arange(m-319,m-63)
        R=float(self.b_loss[recent].mean()); P=float(self.b_loss[prev].mean()); y=self.src["B"]["labels"][recent]
        pos=int((y>0).sum()); neg=int((y<=0).sum())
        cand=bool(R>=1.25*max(P,1e-6) and R-P>=.02 and pos>=16 and neg>=16)
        before=self.birth_streak; self.birth_streak=before+1 if cand else 0
        row={"at_interval":int(t),"m":m,"R":R,"P":P,"positive_support":pos,"negative_support":neg,
             "candidate":cand,"streak_before":before,"streak_after":self.birth_streak}
        self.birth_checks.append(row); return row

    def _first_birth(self,t):
        if self.experts or self.next_id!=0: raise AssertionError("bad first birth state")
        if self.real_science and int(t)!=351: raise AssertionError("Protocol042 first birth must reproduce t351")
        row=self._begin_action("first_birth",t)
        x=c.new_expert(0,t,"active"); x["accepted"]=True; x["first_active_t"]=int(t)+1
        self.experts[0]=x; self.active_ids=[0]; self.next_id=1; self.first_birth_t=int(t)
        self.lifecycle_events.append({"event":"first_birth","expert_id":0,"at_interval":int(t),"first_affected_prediction":int(t)+1,
          "model_hash_before_update":self._model_hash(x),"optimizer_hash_before_update":self._opt_hash(x),"optimizer_step_before_update":c.opt_step(x["optimizer"])})
        self._topology_epoch(t,"first_birth"); self._commit_action(row); self.save_checkpoint("post_first_birth_t%d"%t,True)

    def _predict_expert(self,x,t):
        t0=time.perf_counter(); d=c.expert_delta_np(x["model"],self.src["core"]["T"]["z"][t]); self.prediction_seconds+=time.perf_counter()-t0
        return d

    def _candidate_plan(self,m,for_reuse=False):
        active=list(self.active_ids)
        if self.arm=="R_win128":
            if not active: return {"retained_ids":[],"victim_id":None,"table":[],"reason":"empty_add"}
            victim,table=self.tracker().victim(active,m,self.arm,update_streak=False)
            if victim is None: return None
            return {"retained_ids":[i for i in active if i!=victim],"victim_id":victim,"table":table,"reason":"forced_replace"}
        # A arms: vacant slot always adds without victim.
        if len(active)<2: return {"retained_ids":active,"victim_id":None,"table":[],"reason":"vacant_add"}
        victim,table=self.tracker().victim(active,m,self.arm,update_streak=False)
        if victim is None: return None
        return {"retained_ids":[i for i in active if i!=victim],"victim_id":victim,"table":table,"reason":"replace_nonuseful"}

    def _predict(self,t):
        B=self.src["B"]; z=self.src["core"]["T"]["z"][t]; bm=c.bmargin(B,t)
        deltas={}; total=np.zeros(c.HOSTS,np.float32)
        for slot,eid in enumerate(sorted(self.active_ids)):
            x=self.experts[eid]; d=self._predict_expert(x,t); self.deployed_forwards+=1; deltas[eid]=d; total+=d
            self.out["active_ids"][t,slot]=eid; self.out["expert_versions"][t,slot]=int(x["version"]); self.out["expert_hashes"][t,slot]=self._model_hash(x)
            self.out["contribution"][eid,t]=d
        lg=c.logits_from_margin(B["detection_logits"][t],total); margin=(bm+total).astype(np.float32)
        pr=c.sigmoid_margin(margin)
        self.out["probability"][t]=pr; self.out["detection_logits"][t]=lg; self.out["total_delta"][t]=total
        self.out["live_margin"][t]=margin; self.out["B_margin"][t]=bm; self.out["deployment_epoch"][t]=self.deployment_epoch
        self.out["active_count"][t]=len(self.active_ids); self.out["shadow_present"][t]=int(self.shadow is not None); self.out["accepted_count"][t]=len(self.experts)
        # Candidate validation uses actual live minus locked victim + frozen candidate.
        if self.shadow is not None and self.shadow.get("status")=="validating" and t>=int(self.shadow["validation_start"]) and len(self.shadow["issued"])<32:
            if self.shadow["proposal_epoch"]!=self.deployment_epoch: raise AssertionError("stale shadow survived topology")
            cd=self._predict_expert(self.shadow,t); self.shadow_preview_forwards+=1
            victim=self.shadow["victim_id"]; vm=np.zeros(c.HOSTS,np.float32) if victim is None else deltas.get(victim)
            if victim is not None and vm is None: raise AssertionError("shadow victim not active")
            cm=(margin-vm+cd).astype(np.float32)
            self.shadow["issued"][int(t)]={"candidate_prob":c.sigmoid_margin(cm),"live_prob":pr.copy(),"B_prob":B["probability"][t].copy(),
                                           "candidate_margin":cm,"live_margin":margin.copy(),"candidate_delta":cd}
        if self.reuse_slot is not None and t>=self.reuse_slot["prediction_start"] and len(self.reuse_slot["issued"])<32:
            if self.reuse_slot["proposal_epoch"]!=self.deployment_epoch: raise AssertionError("stale reuse survived topology")
            rr={"live_prob":pr.copy(),"B_prob":B["probability"][t].copy(),"candidates":{}}
            for eid,plan in self.reuse_slot["plans"].items():
                x=self.experts[int(eid)]; cd=self._predict_expert(x,t); self.reuse_preview_forwards+=1
                victim=plan["victim_id"]; vm=np.zeros(c.HOSTS,np.float32) if victim is None else deltas.get(int(victim))
                if victim is not None and vm is None: raise AssertionError("reuse victim not active")
                cm=(margin-vm+cd).astype(np.float32); rr["candidates"][int(eid)]={"prob":c.sigmoid_margin(cm),"margin":cm,"delta":cd}
            self.reuse_slot["issued"][int(t)]=rr
        if self.deployed_forwards>11232 or self.reuse_preview_forwards>3072 or self.shadow_preview_forwards>64:
            raise AssertionError("Protocol042 new-arm forward budget")
        self.max_active_seen=max(self.max_active_seen,len(self.active_ids)); self.max_resident_seen=max(self.max_resident_seen,self.resident_count())

    def _settle(self,t):
        i=int(t)-2
        if i<0 or i>=self.src["n"]: return
        if self.settled[i]: raise AssertionError("duplicate settlement")
        y=self.src["B"]["labels"][i]
        self.b_loss[i]=c.stable_bce_interval(self.out["B_margin"][i],y)
        self.d_loss[i]=c.stable_bce_interval(self.out["live_margin"][i],y); self.settled[i]=1
        ep=int(self.out["deployment_epoch"][i]); deltas={}
        for eid in range(3):
            d=self.out["contribution"][eid,i]
            if np.isfinite(d).all(): deltas[eid]=d.copy()
        self.trackers[ep].settle(i,ep,self.out["live_margin"][i],deltas,y)
        self.settlement_log.append({"at_interval":int(t),"settled_interval":i,"epoch":ep,
          "B_bce":float(self.b_loss[i]),"D_bce":float(self.d_loss[i]),"active_ids":sorted(deltas)})
        if self.shadow is not None and self.shadow.get("status")=="validating" and i in self.shadow.get("issued",{}):
            r=self.shadow["issued"][i]; self.shadow["settled_rows"][i]={**r,"y":np.asarray(y).copy()}
            if len(self.shadow["settled_rows"])==32: self.shadow["ready"]=True
        if self.reuse_slot is not None and i in self.reuse_slot.get("issued",{}):
            r=self.reuse_slot["issued"][i]
            self.reuse_slot["settled_rows"][i]={"y":np.asarray(y).copy(),"live_prob":r["live_prob"],"B_prob":r["B_prob"],"candidates":r["candidates"]}
            if len(self.reuse_slot["settled_rows"])==32: self.reuse_slot["ready"]=True

    def _pressure_check(self,t):
        if not self.experts: self.pressure_streak=0; return None
        m=int(t)-2
        if m<319: self.pressure_streak=0; return None
        recent=np.arange(m-63,m+1); prev=np.arange(m-319,m-63)
        same=bool(np.all(self.out["deployment_epoch"][recent]==self.deployment_epoch))
        matured=len(self._epoch_matured_indices(m)); minm=256 if self.active_ids else 64
        eligible=bool(same and matured>=minm); R=P=None; pos=neg=0; cand=False
        if eligible:
            R=float(self.d_loss[recent].mean()); P=float(self.d_loss[prev].mean()); y=self.src["B"]["labels"][recent]
            pos=int((y>0).sum()); neg=int((y<=0).sum())
            cand=bool(R>=1.25*max(P,1e-6) and R-P>=.02 and pos>=16 and neg>=16)
        before=self.pressure_streak; self.pressure_streak=before+1 if cand else 0; current=bool(cand and self.pressure_streak>=2)
        row={"at_interval":int(t),"m":m,"deployment_epoch":self.deployment_epoch,"active_ids":list(self.active_ids),
          "eligible":eligible,"matured_current_epoch":matured,"R":R,"P":P,"positive_support":pos,"negative_support":neg,
          "candidate":cand,"streak_before":before,"streak_after":self.pressure_streak,"current_pressure":current}
        self.pressure_checks.append(row); return row

    def _update_utility_streaks(self,t):
        m=int(t)-2; rows=[]; eligible=[]
        for eid in sorted(self.active_ids):
            r=self.tracker().check(eid,m,True); row={"at_interval":int(t),"epoch":self.deployment_epoch,"expert_id":eid,**r}
            rows.append(row); self.utility_checks.append(row)
            if r.get("eligible_three"): eligible.append(row)
        if not eligible: return None,rows
        eligible.sort(key=lambda r:(float(r["score"]),int(r["expert_id"])))
        return int(eligible[0]["expert_id"]),rows

    def _sleep(self,eid,t):
        row=self._begin_action("sleep",t,{"expert_id":int(eid)})
        self._make_dormant(eid,t,"nonuseful_three_checks"); self._topology_cleanup(t,"sleep")
        self._commit_action(row); self.save_checkpoint("post_sleep_t%d"%t,True)

    def _reuse_plan_map(self,m):
        plans={}
        for eid,x in sorted(self.experts.items()):
            if x["role"]!="dormant": continue
            plan=self._candidate_plan(m,True)
            if plan is not None: plans[int(eid)]=plan
        return plans

    def _start_reuse(self,t):
        if self.reuse_slot is not None or self.reuse_started>=32: return False
        m=int(t)-2
        if len(self._epoch_matured_indices(m))<128: return False
        dormant=[eid for eid,x in self.experts.items() if x["role"]=="dormant"]
        if not dormant: return False
        plans=self._reuse_plan_map(m)
        if not plans:
            self.reuse_decisions.append({"event":"reuse_skip_no_valid_admission_plan","at_interval":int(t),"epoch":self.deployment_epoch,
                                         "dormant_ids":sorted(dormant)}); return False
        hashes={eid:self._model_hash(self.experts[eid]) for eid in plans}
        self.reuse_slot={"start_t":int(t),"prediction_start":int(t)+1,"proposal_epoch":self.deployment_epoch,
          "plans":copy.deepcopy(plans),"candidate_hashes":hashes,"issued":{},"settled_rows":{},"ready":False}
        self.reuse_started+=1
        self.reuse_decisions.append({"event":"reuse_start","at_interval":int(t),"epoch":self.deployment_epoch,
          "candidate_ids":sorted(plans),"plans":plans,"candidate_hashes":hashes,"slot_number":self.reuse_started})
        self.save_checkpoint("reuse_start_t%d"%t,False); return True

    def _apply_accept(self,eid,plan,t,reason,new_shadow=False):
        victim=plan["victim_id"]
        if victim is not None: self._make_dormant(int(victim),t,reason+"_victim")
        if new_shadow:
            s=self.shadow; s["accepted"]=True; s["role"]="dormant"
            for k in ("status","updates","validation_start","issued","settled_rows","ready","proposal_epoch","retained_ids","victim_id","candidate_id"):
                s.pop(k,None)
            self.experts[int(eid)]=s; self.shadow=None
        self._make_active(int(eid),t,reason)
        self._topology_cleanup(t,reason,keep_shadow=new_shadow,keep_reuse=(reason=="reuse_accept"))

    def _evaluate_reuse(self,t):
        s=self.reuse_slot
        if s is None or not s.get("ready"): return False
        if s["proposal_epoch"]!=self.deployment_epoch:
            self._cancel_reuse(t,"epoch_changed"); return False
        keys=sorted(s["settled_rows"])
        if len(keys)!=32: raise AssertionError("reuse ready without 32")
        live=np.stack([s["settled_rows"][i]["live_prob"] for i in keys]); bp=np.stack([s["settled_rows"][i]["B_prob"] for i in keys]); y=np.stack([s["settled_rows"][i]["y"] for i in keys])
        rows=[]
        for eid,plan in sorted(s["plans"].items()):
            cp=np.stack([s["settled_rows"][i]["candidates"][int(eid)]["prob"] for i in keys])
            q=c.qualify(cp,live,bp,y); q.update({"candidate_id":int(eid),"plan":plan}); rows.append(q)
        passing=[r for r in rows if r["pass"]]
        winner=None if not passing else sorted(passing,key=lambda r:(r["candidate"]["bce"],r["candidate_id"]))[0]
        self.reuse_decisions.append({"event":"reuse_evaluate","at_interval":int(t),"start_t":s["start_t"],
          "epoch":self.deployment_epoch,"intervals":[keys[0],keys[-1]+1],"candidates":rows,
          "winner":None if winner is None else winner["candidate_id"]})
        if winner is None:
            if rows and all(r["support"] and r["finite"] for r in rows):
                self.last_reuse_quality_failure={"eval_t":int(t),"epoch":self.deployment_epoch,
                  "candidate_ids":sorted(s["plans"]),"candidate_hashes":copy.deepcopy(s["candidate_hashes"])}
            self.reuse_slot=None; self.save_checkpoint("reuse_reject_t%d"%t,False); return False
        eid=int(winner["candidate_id"]); plan=s["plans"][eid]
        self.reuse_slot=None; self._cancel_shadow(t,"cancelled_by_reuse_accept")
        self._apply_accept(eid,plan,t,"reuse_accept",False)
        self.save_checkpoint("reuse_accept_t%d"%t,True); return True

    def _victim_acceptance_ok(self,plan,t):
        victim=plan["victim_id"]
        if victim is None: return True,{"reason":"no_victim"}
        if int(victim) not in self.active_ids: return False,{"reason":"victim_not_active"}
        m=int(t)-2
        if self.arm=="R_win128":
            s=self.tracker().score(int(victim),m); return bool(s.get("valid")),s
        s=self.tracker().score(int(victim),m)
        ok=bool(s.get("valid") and self.tracker().streak.get(int(victim),0)>=3 and s["score"]<=0 and s["score_pos"]<=0 and s["score_neg"]<=0 and s["removal_fpr_delta"]<=.01 and s["removal_recall_delta"]>=-.02)
        return ok,s

    def _evaluate_shadow(self,t):
        s=self.shadow
        if s is None or s.get("status")!="validating" or not s.get("ready"): return False
        if s["proposal_epoch"]!=self.deployment_epoch:
            self._cancel_shadow(t,"epoch_changed_before_decision"); return False
        keys=sorted(s["settled_rows"])
        cp=np.stack([s["settled_rows"][i]["candidate_prob"] for i in keys]); live=np.stack([s["settled_rows"][i]["live_prob"] for i in keys])
        bp=np.stack([s["settled_rows"][i]["B_prob"] for i in keys]); y=np.stack([s["settled_rows"][i]["y"] for i in keys])
        q=c.qualify(cp,live,bp,y); plan={"retained_ids":list(s["retained_ids"]),"victim_id":s["victim_id"]}
        victim_ok,vscore=self._victim_acceptance_ok(plan,t)
        if q["pass"] and not victim_ok:
            q["pass"]=False; q["reason"]="reject_victim_no_longer_eligible"
        q.update({"event":"shadow_evaluate","at_interval":int(t),"candidate_id":int(s["id"]),"intervals":[keys[0],keys[-1]+1],
                  "updates":int(s["updates"]),"plan":plan,"victim_recheck":vscore})
        self.candidate_decisions.append(q); self.qualification_records.append(copy.deepcopy(q))
        eid=int(s["id"]); self.last_attempt_end_m=int(t)-2
        if q["pass"]:
            self._apply_accept(eid,plan,t,"shadow_accept",True)
            self.save_checkpoint("shadow_accept_t%d"%t,True); return True
        self.shadow=None; self.save_checkpoint("shadow_reject_t%d"%t,True); return False

    def _prior_reuse_allows_birth(self,t):
        dormant=sorted(eid for eid,x in self.experts.items() if x["role"]=="dormant")
        if not dormant: return True
        f=self.last_reuse_quality_failure
        if not f or int(t)-int(f["eval_t"])>64 or int(f["epoch"])!=self.deployment_epoch: return False
        hashes={eid:self._model_hash(self.experts[eid]) for eid in dormant}
        return f["candidate_ids"]==dormant and f["candidate_hashes"]==hashes

    def _cooldown_ok(self,t):
        if self.attempts_after_e0==0 or self.last_attempt_end_m is None: return True
        return int(t)-2-int(self.last_attempt_end_m)>=256

    def _start_shadow(self,t):
        if self.shadow is not None or self.reuse_slot is not None: return False
        if self.attempts_after_e0>=2 or self.next_id>=3 or self.resident_count()>=3: return False
        if not self._cooldown_ok(t) or not self._prior_reuse_allows_birth(t): return False
        m=int(t)-2; plan=self._candidate_plan(m,False)
        if plan is None:
            self.opportunity_log.append({"event":"candidate_skip_no_victim","at_interval":int(t),"epoch":self.deployment_epoch,
              "active_ids":list(self.active_ids),"reason":"capacity_active_all_useful_or_unknown"}); return False
        eid=self.next_id; self.next_id+=1; self.attempts_after_e0+=1
        s=c.new_expert(eid,t,"shadow")
        if any(torch.count_nonzero(p).item() for p in s["model"].parameters()): raise AssertionError("new residual must be zero")
        s.update({"status":"training","updates":0,"validation_start":None,"issued":{},"settled_rows":{},"ready":False,
          "candidate_id":eid,"proposal_epoch":self.deployment_epoch,"retained_ids":list(plan["retained_ids"]),"victim_id":plan["victim_id"]})
        self.shadow=s
        self.candidate_decisions.append({"event":"shadow_start","at_interval":int(t),"candidate_id":eid,"attempt":self.attempts_after_e0,
          "proposal_epoch":self.deployment_epoch,"retained_ids":list(plan["retained_ids"]),"victim_id":plan["victim_id"],
          "initialization":"zero_weights_bias_fresh_AdamW","model_hash":self._model_hash(s),"optimizer_state_empty":len(s["optimizer"].state)==0})
        self.save_checkpoint("shadow_start_t%d"%t,True); return True

    def _control(self,t):
        if not self._due16(t): return False
        t0=time.process_time(); transitioned=False
        if not self.experts:
            r=self._first_birth_check(t)
            if r and r["streak_after"]>=2: self._first_birth(t); transitioned=True
            if self.real_science and int(t)==351 and self.first_birth_t!=351: raise AssertionError("first birth t351 not reproduced")
        if not transitioned and self.reuse_slot is not None and self.reuse_slot.get("ready"):
            transitioned=self._evaluate_reuse(t)
        if not transitioned and self.shadow is not None and self.shadow.get("status")=="validating" and self.shadow.get("ready"):
            transitioned=self._evaluate_shadow(t)
        victim=None; table=[]
        if not transitioned and self.active_ids:
            victim,table=self._update_utility_streaks(t)
            self.victim_tables.append({"at_interval":int(t),"epoch":self.deployment_epoch,"purpose":"sleep","rows":table,"selected":victim})
            if victim is not None:
                self._sleep(victim,t); transitioned=True
        if transitioned:
            self.controller_cpu_seconds+=time.process_time()-t0; return True
        pressure=self._pressure_check(t); reuse_started=False
        if self._due64(t): reuse_started=self._start_reuse(t)
        shadow_started=False
        if pressure and pressure.get("current_pressure") and not reuse_started and self.reuse_slot is None:
            shadow_started=self._start_shadow(t)
        self.opportunity_log.append({"event":"due16","at_interval":int(t),"epoch":self.deployment_epoch,
          "pressure":None if pressure is None else bool(pressure.get("current_pressure")),"reuse_started":reuse_started,
          "shadow_started":shadow_started,"active_ids":list(self.active_ids),"resident":self.resident_count(),
          "attempts_after_e0":self.attempts_after_e0,"cooldown_ok":self._cooldown_ok(t)})
        self.controller_cpu_seconds+=time.process_time()-t0; return False

    def _batch(self,t):
        if t not in self.src["core"]["by_t"]: return None
        b=[int(i) for i in self.src["core"]["by_t"][t]["batch_indices"]]
        if b and max(b)+2>t: raise AssertionError("immature batch")
        return b

    def _live_update(self,t):
        batch=self._batch(t)
        if batch is None:
            self.pending_train_context=None; return False
        # Pre-live context is persisted for the shadow update at this same t.
        z=torch.from_numpy(self.src["core"]["T"]["z"][batch].astype(np.float32))
        bm=torch.from_numpy(c.bmargin(self.src["B"],batch).astype(np.float32))
        y=torch.from_numpy((self.src["B"]["labels"][batch]>0).astype(np.float32))
        active=list(self.active_ids); deltas={}; total=torch.zeros_like(bm)
        for eid in active:
            d=self.experts[eid]["model"].delta(z); deltas[eid]=d; total=total+d; self.live_training_forwards+=1
        retained=[]
        if self.shadow is not None and self.shadow.get("status")=="training":
            retained=list(self.shadow["retained_ids"])
        bg=torch.zeros_like(bm)
        for eid in retained:
            if eid not in deltas: raise AssertionError("locked retained expert not active")
            bg=bg+deltas[eid]
        self.pending_train_context={"at_interval":int(t),"batch_indices":batch,"retained_ids":retained,
                                    "B_margin":bm.detach().cpu().numpy().astype(np.float32),
                                    "background_delta":bg.detach().cpu().numpy().astype(np.float32)}
        if not active: return False
        if not self.allow_gradient: raise AssertionError("engineering prefix attempted gradient")
        if self.live_optimizer_steps+len(active)>704: raise AssertionError("live optimizer budget")
        row=self._begin_action("joint_live_optimizer_steps",t,{"expert_ids":active,"batch_indices":batch})
        before={eid:{"model":self._model_hash(self.experts[eid]),"opt":self._opt_hash(self.experts[eid]),"step":c.opt_step(self.experts[eid]["optimizer"])} for eid in active}
        for eid in active: self.experts[eid]["optimizer"].zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(bm+total,y)
        reg=sum((d*d).mean() for d in deltas.values())*.001; objective=loss+reg
        t0=time.perf_counter(); objective.backward()
        gns={}
        for eid in active:
            gn=torch.nn.utils.clip_grad_norm_(self.experts[eid]["model"].parameters(),1.0); gns[eid]=float(gn)
        for eid in active:
            self.experts[eid]["optimizer"].step(); self.actual_optimizer_calls+=1
            self.experts[eid]["version"]+=1; self.live_optimizer_steps+=1
        self.update_seconds+=time.perf_counter()-t0
        for eid in active:
            x=self.experts[eid]
            self.update_log.append({"at_interval":int(t),"kind":"live","expert_id":eid,"joint_active_ids":active,"batch_indices":batch,
              "loss":float(loss.detach()),"regularizer":float(reg.detach()),"objective":float(objective.detach()),"grad_norm":gns[eid],
              "model_hash_before":before[eid]["model"],"model_hash_after":self._model_hash(x),
              "optimizer_hash_before":before[eid]["opt"],"optimizer_hash_after":self._opt_hash(x),
              "optimizer_step_before":before[eid]["step"],"optimizer_step_after":c.opt_step(x["optimizer"]),"version_after":x["version"]})
        self._commit_action(row,{"optimizer_calls":len(active),"live_steps_after":self.live_optimizer_steps})
        self.save_checkpoint("live_update_t%d"%t,False); return True

    def _shadow_update(self,t):
        s=self.shadow; batch=self._batch(t)
        if s is None or s.get("status")!="training" or batch is None: self.pending_train_context=None; return False
        if not self.allow_gradient: raise AssertionError("engineering prefix attempted shadow gradient")
        if self.shadow_optimizer_steps>=32 or s["updates"]>=16: raise AssertionError("shadow optimizer budget")
        ctx=self.pending_train_context
        if ctx is None or ctx["at_interval"]!=int(t) or ctx["batch_indices"]!=batch or ctx["retained_ids"]!=list(s["retained_ids"]):
            raise AssertionError("missing pre-live detached shadow context")
        z=torch.from_numpy(self.src["core"]["T"]["z"][batch].astype(np.float32))
        bm=torch.from_numpy(ctx["B_margin"]); bg=torch.from_numpy(ctx["background_delta"]).detach()
        y=torch.from_numpy((self.src["B"]["labels"][batch]>0).astype(np.float32))
        d=s["model"].delta(z); self.shadow_training_forwards+=1
        s["optimizer"].zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(bm+bg+d,y); reg=.001*(d*d).mean(); objective=loss+reg
        row=self._begin_action("shadow_optimizer_step",t,{"expert_id":int(s["id"]),"batch_indices":batch,"update_number":int(s["updates"])+1})
        before={"model":self._model_hash(s),"opt":self._opt_hash(s),"step":c.opt_step(s["optimizer"])}
        t0=time.perf_counter(); objective.backward(); gn=float(torch.nn.utils.clip_grad_norm_(s["model"].parameters(),1.0))
        s["optimizer"].step(); self.actual_optimizer_calls+=1; self.update_seconds+=time.perf_counter()-t0
        s["version"]+=1; s["updates"]+=1; self.shadow_optimizer_steps+=1
        ev={"at_interval":int(t),"kind":"shadow","expert_id":int(s["id"]),"batch_indices":batch,"retained_ids":list(s["retained_ids"]),
          "victim_id":s["victim_id"],"loss":float(loss.detach()),"regularizer":float(reg.detach()),"objective":float(objective.detach()),"grad_norm":gn,
          "model_hash_before":before["model"],"model_hash_after":self._model_hash(s),"optimizer_hash_before":before["opt"],"optimizer_hash_after":self._opt_hash(s),
          "optimizer_step_before":before["step"],"optimizer_step_after":c.opt_step(s["optimizer"]),"version_after":s["version"],"shadow_update_after":s["updates"]}
        self.update_log.append(ev)
        if s["updates"]==16:
            s["status"]="validating"; s["validation_start"]=int(t)+1; s["issued"]={}; s["settled_rows"]={}; s["ready"]=False
            self.candidate_decisions.append({"event":"shadow_training_complete","at_interval":int(t),"candidate_id":int(s["id"]),"updates":16,
              "validation_start":int(t)+1,"model_hash":self._model_hash(s),"optimizer_step":c.opt_step(s["optimizer"])})
        self._commit_action(row,{"shadow_updates_after":int(s["updates"]),"status_after":s["status"],
                                 "optimizer_step_after":c.opt_step(s["optimizer"])})
        self.pending_train_context=None
        self.save_checkpoint(("shadow16_t%d" if s["updates"]==16 else "shadow_update_t%d")%t,s["updates"]==16); return True

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
                if len(self.active_ids)>2 or self.resident_count()>3 or self.next_id>3: raise AssertionError("capacity/id invariant")
                if any(self.experts[eid]["role"]!="active" for eid in self.active_ids): raise AssertionError("active role mismatch")
                self.cursor+=1; self.next_substep="predict"; self.pending_train_context=None
                if self.cursor%512==0: self.save_checkpoint("cursor_%04d"%self.cursor,True)
            else: raise AssertionError("bad substep "+str(self.next_substep))
        return self

    def terminal_settle(self):
        counters=(self.live_optimizer_steps,self.shadow_optimizer_steps,self.deployed_forwards,self.reuse_preview_forwards,self.shadow_preview_forwards,self.actual_optimizer_calls)
        if self.terminal_progress==0:
            self._settle(self.src["n"]); self.terminal_progress=1; self.next_substep="terminal_second_settle"; self.save_checkpoint("terminal_first_settle",True)
        if self.terminal_progress==1:
            self._settle(self.src["n"]+1); self.terminal_progress=2; self.next_substep="terminal_censor"; self.save_checkpoint("terminal_second_settle",True)
        if self.terminal_progress==2:
            if self.reuse_slot is not None:
                self.reuse_decisions.append({"event":"reuse_censored","at_interval":self.src["n"]+1,"start_t":self.reuse_slot["start_t"],
                  "issued_count":len(self.reuse_slot.get("issued",{})),"settled_count":len(self.reuse_slot.get("settled_rows",{}))}); self.reuse_slot=None
            if self.shadow is not None:
                self.candidate_decisions.append({"event":"shadow_censored","at_interval":self.src["n"]+1,"candidate_id":int(self.shadow["id"]),
                  "status":self.shadow.get("status"),"updates":int(self.shadow.get("updates",0)),"issued_count":len(self.shadow.get("issued",{}))})
            self.terminal_progress=3; self.next_substep="done"; self.save_checkpoint("terminal_done",True)
        after=(self.live_optimizer_steps,self.shadow_optimizer_steps,self.deployed_forwards,self.reuse_preview_forwards,self.shadow_preview_forwards,self.actual_optimizer_calls)
        self.terminal_counter_delta=[after[i]-counters[i] for i in range(len(after))]
        if any(self.terminal_counter_delta): raise AssertionError("terminal settlement changed online counters")
        return self

    def snapshot(self):
        return {
          "protocol":"042","revision":2,"arm":self.arm,"cursor":self.cursor,"next_substep":self.next_substep,"terminal_progress":self.terminal_progress,
          "deployment_epoch":self.deployment_epoch,"epoch_start_prediction":self.epoch_start_prediction,
          "experts":{int(k):c.pack_expert(v) for k,v in self.experts.items()},"active_ids":list(self.active_ids),
          "shadow":None if self.shadow is None else c.pack_expert(self.shadow),"next_id":self.next_id,
          "attempts_after_e0":self.attempts_after_e0,"last_attempt_end_m":self.last_attempt_end_m,"first_birth_t":self.first_birth_t,
          "birth_streak":self.birth_streak,"pressure_streak":self.pressure_streak,"reuse_slot":copy.deepcopy(self.reuse_slot),
          "reuse_started":self.reuse_started,"last_reuse_quality_failure":copy.deepcopy(self.last_reuse_quality_failure),
          "pending_train_context":copy.deepcopy(self.pending_train_context),"trackers":copy.deepcopy(self.trackers),
          "out":{k:v.copy() for k,v in self.out.items()},"settled":self.settled.copy(),"b_loss":self.b_loss.copy(),"d_loss":self.d_loss.copy(),
          "birth_checks":copy.deepcopy(self.birth_checks),"pressure_checks":copy.deepcopy(self.pressure_checks),"utility_checks":copy.deepcopy(self.utility_checks),
          "victim_tables":copy.deepcopy(self.victim_tables),"lifecycle_events":copy.deepcopy(self.lifecycle_events),
          "candidate_decisions":copy.deepcopy(self.candidate_decisions),"reuse_decisions":copy.deepcopy(self.reuse_decisions),
          "opportunity_log":copy.deepcopy(self.opportunity_log),"update_log":copy.deepcopy(self.update_log),
          "settlement_log":copy.deepcopy(self.settlement_log),"qualification_records":copy.deepcopy(self.qualification_records),
          "archive_events":copy.deepcopy(self.archive_events),
          "live_optimizer_steps":self.live_optimizer_steps,"shadow_optimizer_steps":self.shadow_optimizer_steps,
          "deployed_forwards":self.deployed_forwards,"reuse_preview_forwards":self.reuse_preview_forwards,
          "shadow_preview_forwards":self.shadow_preview_forwards,"live_training_forwards":self.live_training_forwards,
          "shadow_training_forwards":self.shadow_training_forwards,"controller_cpu_seconds":self.controller_cpu_seconds,
          "prediction_seconds":self.prediction_seconds,"update_seconds":self.update_seconds,"max_active_seen":self.max_active_seen,
          "max_resident_seen":self.max_resident_seen,"permanent_deletions":self.permanent_deletions,
          "action_seq":self.action_seq,"completed_action_ids":copy.deepcopy(self.completed_action_ids),"actual_optimizer_calls":self.actual_optimizer_calls,
          "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
        }

    def save_checkpoint(self,reason,named=False):
        if self.checkpoint_dir is None: return
        d=Path(self.checkpoint_dir); d.mkdir(parents=True,exist_ok=True); snap=self.snapshot()
        tmp=d/"latest.pt.tmp"; final=d/"latest.pt"; torch.save(snap,tmp); os.replace(tmp,final)
        meta={"protocol":"042","revision":2,"arm":self.arm,"cursor":self.cursor,"next_substep":self.next_substep,
          "terminal_progress":self.terminal_progress,"reason":reason,"live_steps":self.live_optimizer_steps,
          "shadow_steps":self.shadow_optimizer_steps,"actual_optimizer_calls":self.actual_optimizer_calls,
          "action_seq":self.action_seq,"sha256":sha256_file(final)}
        c.W(d/"latest.json",meta)
        if named:
            safe="".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in reason); q=d/(safe+".pt")
            torch.save(snap,q); c.W(str(q)+".json",{**meta,"sha256":sha256_file(q)})

    @classmethod
    def restore(cls,src,work_dir,path,arm=None,allow_gradient=True,real_science=False,strict_journal=True):
        x=torch.load(path,map_location="cpu")
        if (x.get("protocol"),x.get("revision"))!=("042",2): raise AssertionError("bad Protocol042 checkpoint")
        if arm is not None and x.get("arm")!=arm: raise AssertionError("arm mismatch")
        m=cls(src,work_dir,x["arm"],allow_gradient=allow_gradient,real_science=real_science)
        for k in ("cursor","next_substep","terminal_progress","deployment_epoch","epoch_start_prediction","active_ids","next_id","attempts_after_e0",
                  "last_attempt_end_m","first_birth_t","birth_streak","pressure_streak","reuse_slot","reuse_started","last_reuse_quality_failure",
                  "pending_train_context","trackers","birth_checks","pressure_checks","utility_checks","victim_tables","lifecycle_events",
                  "candidate_decisions","reuse_decisions","opportunity_log","update_log","settlement_log","qualification_records","archive_events",
                  "live_optimizer_steps","shadow_optimizer_steps","deployed_forwards","reuse_preview_forwards","shadow_preview_forwards",
                  "live_training_forwards","shadow_training_forwards","controller_cpu_seconds","prediction_seconds","update_seconds",
                  "max_active_seen","max_resident_seen","permanent_deletions","action_seq","completed_action_ids","actual_optimizer_calls"):
            setattr(m,k,copy.deepcopy(x[k]))
        m.experts={int(k):c.unpack_expert(v) for k,v in x["experts"].items()}
        m.shadow=None if x["shadow"] is None else c.unpack_expert(x["shadow"])
        m.out={k:v.copy() for k,v in x["out"].items()}; m.settled=x["settled"].copy(); m.b_loss=x["b_loss"].copy(); m.d_loss=x["d_loss"].copy()
        torch.set_rng_state(x["torch_rng"]); np.random.set_state(x["numpy_rng"]); random.setstate(x["python_rng"])
        if strict_journal and m.action_journal.exists():
            j=c.J(m.action_journal)
            if j.get("status")=="pending": raise RuntimeError("ambiguous_step: pending action journal")
            if int(j.get("action_seq",0))>m.action_seq: raise RuntimeError("ambiguous_step: journal newer than checkpoint")
        return m

def semantic_digest(m):
    rows={"arm":m.arm,"cursor":m.cursor,"next_substep":m.next_substep,"terminal_progress":m.terminal_progress,
      "deployment_epoch":m.deployment_epoch,"active_ids":list(m.active_ids),"next_id":m.next_id,
      "attempts_after_e0":m.attempts_after_e0,"first_birth_t":m.first_birth_t,
      "experts":{str(k):{"role":v["role"],"version":v["version"],"model":m._model_hash(v),"opt":m._opt_hash(v),"step":c.opt_step(v["optimizer"])}
                 for k,v in sorted(m.experts.items())},
      "shadow":None if m.shadow is None else {"id":m.shadow["id"],"status":m.shadow.get("status"),"updates":m.shadow.get("updates"),
        "model":m._model_hash(m.shadow),"opt":m._opt_hash(m.shadow),"step":c.opt_step(m.shadow["optimizer"])},
      "counters":[m.live_optimizer_steps,m.shadow_optimizer_steps,m.deployed_forwards,m.reuse_preview_forwards,m.shadow_preview_forwards,m.actual_optimizer_calls],
      "events":[m.lifecycle_events,m.candidate_decisions,m.reuse_decisions,m.update_log],
      "out_hash":hashlib.sha256(m.out["probability"].tobytes()).hexdigest(),
      "contrib_hash":hashlib.sha256(m.out["contribution"].tobytes()).hexdigest(),
      "rng":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
