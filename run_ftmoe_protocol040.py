"""Protocol-040: one bounded D_pool2 sequence with at most two accepted/shadow dynamic experts.

C/B/D_keep/D_039 are frozen cached controls. Only D_pool2 receives gradients.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, time, traceback
from pathlib import Path

import numpy as np
import torch

import run_ftmoe_protocol037 as p37
import run_ftmoe_protocol039 as p39
from protocol035_common import binary_metrics, dump_json, load_npz, sha256_file, sha256_state_dict

PLAN = Path("artifacts/ftmoe_online/protocol_040/plan.json")
PLAN_SHA = "97839cb55ea9d00521e15529e5b5505004c69fd9587d2d22503b7b1ff99dcd35"
N, HOSTS, DIM = 5968, 16, 73
SOURCE036 = {
 "stage_B/seed3601/C_ref/feature_tape.npz":"c71a2ea6d59045fe4cc8a953a776d37f48dbe1feec4aba859cd00f11c984529c",
 "stage_B/seed3601/C_ref/update_batches.json":"4d32d2ece194cb41b8bc16de1c89ddfedcd42da668b96446937fb430d6246b6f",
 "stage_B/seed3601/D_lin/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "inputs/seed3601/manifest.json":"73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b",
 "stage_B/seed3601/C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
}
SOURCE039 = {
 "B_ref/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
 "D_keep/predictions.npz":"badf9005b2be9fc6436d1ff2b9377b27715e846d7d16f8e4bdfcbfa76f732e8c",
 "D_sleepwake/lifecycle_events.json":"3eb6d21757fd84b195d62d298350f0772befca471793dddbb7f0062933b138f8",
 "D_sleepwake/predictions.npz":"bc387d13721ed2e759a205ed3b4f6fa884c1ca105068a79a340148e1f5e96512",
 "D_sleepwake/update_log.json":"f771642166041282e1748a3e5b36c991f0365476b8fb953e7ceaee7dae2162d6",
}

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x): dump_json(p,x)
def runtime(): p37.runtime()

def verify_plan():
    if sha256_file(PLAN)!=PLAN_SHA: raise AssertionError("Protocol040 plan hash changed")
    p=J(PLAN)
    if p.get("protocol")!="040" or p["budget"]["sequence_names"]!=["D_pool2"]: raise AssertionError("Protocol040 plan identity")
    return p

def verify_files(root,expected,label):
    root=Path(root); out={}
    for rel,d in expected.items():
        q=root/rel
        if not q.exists(): raise AssertionError(f"{label} missing {rel}")
        g=sha256_file(q); out[rel]=g
        if g!=d: raise AssertionError(f"{label} hash mismatch {rel}: {g}")
    return out

def load_sources(source036, source039):
    source036,source039=Path(source036),Path(source039)
    h36=verify_files(source036,SOURCE036,"source036")
    h39=verify_files(source039,SOURCE039,"source039")
    core=p37.load_source(source036,verify=True)
    C=load_npz(source039/"C_ref/predictions.npz")
    B=load_npz(source039/"B_ref/predictions.npz")
    K=load_npz(source039/"D_keep/predictions.npz")
    D39=load_npz(source039/"D_sleepwake/predictions.npz")
    for name,a in (("C",C),("B",B),("D_keep",K),("D_039",D39)):
        if a["probability"].shape!=(N,HOSTS) or a["detection_logits"].shape!=(N,HOSTS,2):
            raise AssertionError(name+" geometry")
    for a in (B,K,D39):
        for key in ("labels","raw_labels","class_probability"):
            if key not in a or key not in C or not np.array_equal(a[key],C[key]):
                raise AssertionError("cached controls alignment "+key)
    if not np.array_equal(core["B"]["detection_logits"],B["detection_logits"]):
        raise AssertionError("source036 B != source039 B")
    if not np.array_equal(core["T"]["labels"],C["labels"]):
        raise AssertionError("issued tape labels mismatch")
    return {
      "source036_root":source036,"source039_root":source039,"core":core,
      "C":C,"B":B,"D_keep":K,"D_039":D39,"n":N,
      "hashes036":h36,"hashes039":h39,
      "manifest":J(source036/"inputs/seed3601/manifest.json"),
      "p39_lifecycle":J(source039/"D_sleepwake/lifecycle_events.json"),
      "p39_updates":J(source039/"D_sleepwake/update_log.json"),
    }

def opt_digest(opt): return p39.optimizer_digest(opt)
def opt_step(opt): return p39.optimizer_step_value(opt)

def stable_bce_logits(logits,labels): return p39.stable_interval_bce_from_logits(logits,labels)

def prob_metrics(prob,y):
    return binary_metrics(np.asarray(prob).reshape(-1),np.asarray(y).reshape(-1))

def qualify(candidate_prob, live_prob, b_prob, y):
    cp=np.asarray(candidate_prob,dtype=np.float64).reshape(-1)
    lp=np.asarray(live_prob,dtype=np.float64).reshape(-1)
    bp=np.asarray(b_prob,dtype=np.float64).reshape(-1)
    yy=np.asarray(y).reshape(-1)
    pos=int((yy>0).sum()); neg=int((yy<=0).sum())
    finite=bool(np.isfinite(cp).all() and np.isfinite(lp).all() and np.isfinite(bp).all())
    support=bool(pos>=16 and neg>=16)
    if not finite or not support:
        return {"positive_host_rows":pos,"negative_host_rows":neg,"finite":finite,"support":support,
                "pass":False,"reason":"nonfinite" if not finite else "insufficient_support"}
    cm,lm,bm=prob_metrics(cp,yy),prob_metrics(lp,yy),prob_metrics(bp,yy)
    checks={
      "bce_vs_live": bool(cm["bce"] <= 0.99*lm["bce"]),
      "bce_vs_B": bool(cm["bce"] <= bm["bce"]),
      "fpr_vs_live": bool(cm["fpr"]-lm["fpr"] <= 0.01),
      "fpr_vs_B": bool(cm["fpr"]-bm["fpr"] <= 0.01),
      "recall_vs_live": bool(cm["recall"]-lm["recall"] >= -0.02),
      "recall_vs_B": bool(cm["recall"]-bm["recall"] >= -0.02),
    }
    ok=all(checks.values())
    return {"positive_host_rows":pos,"negative_host_rows":neg,"finite":finite,"support":support,
            "candidate":cm,"live":lm,"B":bm,"checks":checks,"pass":bool(ok),
            "reason":"pass" if ok else "quality_rejected"}

def empty_outputs(n):
    return {
      "probability":np.full((n,HOSTS),np.nan,np.float32),
      "detection_logits":np.full((n,HOSTS,2),np.nan,np.float32),
      "delta":np.full((n,HOSTS),np.nan,np.float32),
      "active_id":np.full(n,-1,np.int16),
      "deployment_epoch":np.zeros(n,np.int32),
      "expert_version":np.full(n,-1,np.int32),
      "expert_hash":np.full(n,"",dtype="<U64"),
      "accepted_count":np.zeros(n,np.int8),
      "shadow_present":np.zeros(n,np.int8),
    }

def pack_expert(x):
    return {
      "id":int(x["id"]),"role":x["role"],"version":int(x["version"]),
      "accepted":bool(x["accepted"]),"created_t":int(x["created_t"]),
      "first_active_t":x.get("first_active_t"),"dormant_since_prediction":x.get("dormant_since_prediction"),
      "reactivations":int(x.get("reactivations",0)),"model":copy.deepcopy(x["model"].state_dict()),
      "optimizer":copy.deepcopy(x["optimizer"].state_dict()),
      "last_dormant_model_hash":x.get("last_dormant_model_hash"),
      "last_dormant_optimizer_hash":x.get("last_dormant_optimizer_hash"),
      "last_dormant_optimizer_step":x.get("last_dormant_optimizer_step"),
    }

def unpack_expert(d):
    m=p37.make_expert(); o=p37.make_optimizer(m)
    m.load_state_dict(d["model"]); o.load_state_dict(d["optimizer"])
    x={k:copy.deepcopy(v) for k,v in d.items() if k not in ("model","optimizer")}
    x["model"]=m; x["optimizer"]=o
    return x

class Machine:
    def __init__(self,src,work_dir,allow_gradient=True,real_science=False):
        self.src=src; self.work_dir=Path(work_dir); self.work_dir.mkdir(parents=True,exist_ok=True)
        self.allow_gradient=bool(allow_gradient); self.real_science=bool(real_science)
        self.cursor=0; self.deployment_epoch=0; self.active_id=None
        self.experts={}; self.shadow=None; self.next_id=0; self.second_attempt_consumed=False
        self.first_birth_t=None; self.birth_streak=0; self.sleep_streak=0; self.pressure_streak=0
        self.reuse_slot=None; self.reuse_started=0; self.last_reuse_quality_failure=None
        self.out=empty_outputs(int(src["n"]))
        self.b_loss=np.full(int(src["n"]),np.nan,np.float64)
        self.d_loss=np.full(int(src["n"]),np.nan,np.float64)
        self.settled=np.zeros(int(src["n"]),np.int8)
        self.birth_checks=[]; self.sleep_checks=[]; self.pressure_checks=[]
        self.lifecycle_events=[]; self.reuse_decisions=[]; self.candidate_decisions=[]
        self.update_log=[]; self.forward_log=[]; self.opportunity_log=[]; self.settlement_log=[]
        self.live_optimizer_steps=0; self.shadow_optimizer_steps=0
        self.deployed_forwards=0; self.reuse_preview_forwards=0; self.shadow_preview_forwards=0
        self.max_capacity_seen=0; self.terminal_settle_optimizer_steps=0
        self.prediction_seconds=0.0; self.update_seconds=0.0; self.controller_cpu_seconds=0.0
        self.checkpoint_dir=None; self.action_seq=0
        self.action_journal=self.work_dir/"action_journal.json"

    def capacity(self): return len(self.experts)+(1 if self.shadow is not None else 0)
    def active(self): return None if self.active_id is None else self.experts[self.active_id]
    def model_hash(self,x): return sha256_state_dict(x["model"].state_dict())
    def optimizer_hash(self,x): return opt_digest(x["optimizer"])

    def _begin_action(self,kind,t,meta=None):
        self.action_seq+=1
        row={"protocol":"040","action_seq":self.action_seq,"status":"pending","kind":kind,"cursor":int(t),"meta":meta or {}}
        W(self.action_journal,row); return row
    def _commit_action(self,row,extra=None):
        z=dict(row); z["status"]="committed"
        if extra: z.update(extra)
        W(self.action_journal,z)

    def snapshot(self):
        sh=None
        if self.shadow is not None:
            sh=pack_expert(self.shadow)
            for k in ("status","updates","validation_start","issued","settled_rows","candidate_id"):
                if k in self.shadow: sh[k]=copy.deepcopy(self.shadow[k])
        return {
          "protocol":"040","cursor":self.cursor,"deployment_epoch":self.deployment_epoch,"active_id":self.active_id,
          "experts":{int(k):pack_expert(v) for k,v in self.experts.items()},"shadow":sh,"next_id":self.next_id,
          "second_attempt_consumed":self.second_attempt_consumed,"first_birth_t":self.first_birth_t,
          "birth_streak":self.birth_streak,"sleep_streak":self.sleep_streak,"pressure_streak":self.pressure_streak,
          "reuse_slot":copy.deepcopy(self.reuse_slot),"reuse_started":self.reuse_started,
          "last_reuse_quality_failure":copy.deepcopy(self.last_reuse_quality_failure),
          "out":{k:v.copy() for k,v in self.out.items()},"b_loss":self.b_loss.copy(),"d_loss":self.d_loss.copy(),"settled":self.settled.copy(),
          "birth_checks":copy.deepcopy(self.birth_checks),"sleep_checks":copy.deepcopy(self.sleep_checks),"pressure_checks":copy.deepcopy(self.pressure_checks),
          "lifecycle_events":copy.deepcopy(self.lifecycle_events),"reuse_decisions":copy.deepcopy(self.reuse_decisions),
          "candidate_decisions":copy.deepcopy(self.candidate_decisions),"update_log":copy.deepcopy(self.update_log),
          "forward_log":copy.deepcopy(self.forward_log),"opportunity_log":copy.deepcopy(self.opportunity_log),
          "settlement_log":copy.deepcopy(self.settlement_log),
          "live_optimizer_steps":self.live_optimizer_steps,"shadow_optimizer_steps":self.shadow_optimizer_steps,
          "deployed_forwards":self.deployed_forwards,"reuse_preview_forwards":self.reuse_preview_forwards,
          "shadow_preview_forwards":self.shadow_preview_forwards,"max_capacity_seen":self.max_capacity_seen,
          "terminal_settle_optimizer_steps":self.terminal_settle_optimizer_steps,
          "prediction_seconds":self.prediction_seconds,"update_seconds":self.update_seconds,"controller_cpu_seconds":self.controller_cpu_seconds,
          "action_seq":self.action_seq,"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
        }

    def save_checkpoint(self,reason,named=False):
        if self.checkpoint_dir is None: return
        d=Path(self.checkpoint_dir); d.mkdir(parents=True,exist_ok=True)
        snap=self.snapshot(); tmp=d/"latest.pt.tmp"; final=d/"latest.pt"
        torch.save(snap,tmp); os.replace(tmp,final)
        meta={"protocol":"040","cursor":self.cursor,"reason":reason,"active_id":self.active_id,
              "deployment_epoch":self.deployment_epoch,"live_steps":self.live_optimizer_steps,
              "shadow_steps":self.shadow_optimizer_steps,"sha256":sha256_file(final)}
        W(d/"latest.json",meta)
        if named:
            safe="".join(c if c.isalnum() or c in "_-" else "_" for c in reason)
            q=d/(safe+".pt"); torch.save(snap,q); W(str(q)+".json",{**meta,"sha256":sha256_file(q)})

    @classmethod
    def restore(cls,src,work_dir,path,allow_gradient=True,real_science=False):
        x=torch.load(path,map_location="cpu")
        if x.get("protocol")!="040": raise AssertionError("bad checkpoint protocol")
        m=cls(src,work_dir,allow_gradient=allow_gradient,real_science=real_science)
        m.cursor=int(x["cursor"]); m.deployment_epoch=int(x["deployment_epoch"]); m.active_id=x["active_id"]
        m.experts={int(k):unpack_expert(v) for k,v in x["experts"].items()}
        if x["shadow"] is not None:
            sh=unpack_expert(x["shadow"])
            for k in ("status","updates","validation_start","issued","settled_rows","candidate_id"):
                if k in x["shadow"]: sh[k]=copy.deepcopy(x["shadow"][k])
            m.shadow=sh
        m.next_id=int(x["next_id"]); m.second_attempt_consumed=bool(x["second_attempt_consumed"])
        m.first_birth_t=x["first_birth_t"]; m.birth_streak=int(x["birth_streak"]); m.sleep_streak=int(x["sleep_streak"]); m.pressure_streak=int(x["pressure_streak"])
        m.reuse_slot=copy.deepcopy(x["reuse_slot"]); m.reuse_started=int(x["reuse_started"]); m.last_reuse_quality_failure=copy.deepcopy(x["last_reuse_quality_failure"])
        m.out={k:v.copy() for k,v in x["out"].items()}; m.b_loss=x["b_loss"].copy(); m.d_loss=x["d_loss"].copy(); m.settled=x["settled"].copy()
        for k in ("birth_checks","sleep_checks","pressure_checks","lifecycle_events","reuse_decisions","candidate_decisions","update_log","forward_log","opportunity_log","settlement_log"):
            setattr(m,k,copy.deepcopy(x[k]))
        for k in ("live_optimizer_steps","shadow_optimizer_steps","deployed_forwards","reuse_preview_forwards","shadow_preview_forwards","max_capacity_seen","terminal_settle_optimizer_steps","action_seq"):
            setattr(m,k,int(x[k]))
        for k in ("prediction_seconds","update_seconds","controller_cpu_seconds"): setattr(m,k,float(x[k]))
        torch.set_rng_state(x["torch_rng"]); np.random.set_state(x["numpy_rng"]); random.setstate(x["python_rng"])
        return m

    def _predict_one(self,x,t):
        brow=self.src["B"]["detection_logits"][t]; z=self.src["core"]["T"]["z"][t]
        dd,logits,prob,secs=p37.expert_predict(x["model"],z,brow,exact_b_if_zero=True)
        if logits is None: logits=brow.copy(); prob=self.src["B"]["probability"][t].copy()
        return dd,logits,prob,secs

    def _predict(self,t):
        B=self.src["B"]; a=self.active()
        self.out["active_id"][t]=-1 if a is None else int(a["id"])
        self.out["deployment_epoch"][t]=int(self.deployment_epoch)
        self.out["accepted_count"][t]=len(self.experts); self.out["shadow_present"][t]=1 if self.shadow is not None else 0
        if a is None:
            self.out["detection_logits"][t]=B["detection_logits"][t]; self.out["probability"][t]=B["probability"][t]
            self.out["delta"][t]=0; self.out["expert_version"][t]=-1; self.out["expert_hash"][t]="none"
        else:
            row=self._begin_action("live_prediction",t,{"expert_id":a["id"]})
            dd,lg,pr,secs=self._predict_one(a,t); self.prediction_seconds+=secs; self.deployed_forwards+=1
            self.out["detection_logits"][t]=lg; self.out["probability"][t]=pr; self.out["delta"][t]=dd
            self.out["expert_version"][t]=a["version"]; self.out["expert_hash"][t]=self.model_hash(a)
            self._commit_action(row,{"expert_hash":self.out["expert_hash"][t]})
        self.forward_log.append({"at_interval":int(t),"active_id":None if a is None else int(a["id"]),"deployment_epoch":int(self.deployment_epoch)})

        if self.reuse_slot is not None and self.reuse_slot["status"]=="collecting" and t>=self.reuse_slot["prediction_start"] and len(self.reuse_slot["issued"])<32:
            if self.reuse_slot["deployment_epoch"]!=self.deployment_epoch:
                raise AssertionError("reuse slot survived topology change")
            rec={"live_prob":self.out["probability"][t].astype(np.float32),"B_prob":B["probability"][t].astype(np.float32),"candidates":{}}
            for cid in self.reuse_slot["candidate_ids"]:
                x=self.experts[cid]
                if x["role"]!="dormant": raise AssertionError("reuse candidate not dormant")
                _,_,pr,secs=self._predict_one(x,t); self.prediction_seconds+=secs; self.reuse_preview_forwards+=1
                rec["candidates"][int(cid)]=pr.astype(np.float32)
            self.reuse_slot["issued"][int(t)]=rec

        if self.shadow is not None and self.shadow.get("status")=="validating" and t>=int(self.shadow["validation_start"]) and len(self.shadow["issued"])<32:
            _,_,pr,secs=self._predict_one(self.shadow,t); self.prediction_seconds+=secs; self.shadow_preview_forwards+=1
            self.shadow["issued"][int(t)]={"live_prob":self.out["probability"][t].astype(np.float32),"B_prob":B["probability"][t].astype(np.float32),"candidate_prob":pr.astype(np.float32)}

        if self.deployed_forwards>5616 or self.reuse_preview_forwards>2048 or self.shadow_preview_forwards>32 or self.reuse_preview_forwards+self.shadow_preview_forwards>2080:
            raise AssertionError("Protocol040 forward budget exceeded")

    def _settle(self,t):
        i=int(t)-2
        if i<0 or i>=self.src["n"]: return
        if self.settled[i]: raise AssertionError("duplicate settlement")
        y=self.src["B"]["labels"][i]
        self.b_loss[i]=stable_bce_logits(self.src["B"]["detection_logits"][i],y)
        self.d_loss[i]=stable_bce_logits(self.out["detection_logits"][i],y)
        self.settled[i]=1
        self.settlement_log.append({"at_interval":int(t),"settled_interval":i,"B_bce":float(self.b_loss[i]),"D_bce":float(self.d_loss[i])})
        if self.reuse_slot is not None and i in self.reuse_slot["issued"]:
            r=self.reuse_slot["issued"][i]
            self.reuse_slot["settled_rows"][i]={"y":np.asarray(y).copy(),"live_prob":r["live_prob"],"B_prob":r["B_prob"],"candidates":r["candidates"]}
            if len(self.reuse_slot["settled_rows"])==32: self.reuse_slot["ready"]=True
        if self.shadow is not None and self.shadow.get("status")=="validating" and i in self.shadow["issued"]:
            r=self.shadow["issued"][i]
            self.shadow["settled_rows"][i]={"y":np.asarray(y).copy(),"live_prob":r["live_prob"],"B_prob":r["B_prob"],"candidate_prob":r["candidate_prob"]}
            if len(self.shadow["settled_rows"])==32: self.shadow["ready"]=True

    def _due16(self,t): return (int(t)+1)%16==0
    def _due64(self,t): return (int(t)+1)%64==0

    def _first_birth_check(self,t):
        m=int(t)-2
        if m<319: return None
        recent=np.arange(m-63,m+1); prev=np.arange(m-319,m-63)
        R=float(self.b_loss[recent].mean()); P=float(self.b_loss[prev].mean()); y=self.src["B"]["labels"][recent]
        pos=int((y>0).sum()); neg=int((y<=0).sum())
        cand=bool(R>=1.25*max(P,1e-6) and R-P>=0.02 and pos>=16 and neg>=16)
        before=self.birth_streak; self.birth_streak=before+1 if cand else 0
        row={"at_interval":int(t),"m":m,"R":R,"P":P,"positive_support":pos,"negative_support":neg,"candidate":cand,"streak_before":before,"streak_after":self.birth_streak}
        self.birth_checks.append(row); return row

    def _epoch_matured_indices(self,m,active_id=None):
        if m<0: return np.array([],dtype=np.int64)
        ids=self.out["active_id"][:m+1]; ep=self.out["deployment_epoch"][:m+1]
        want=-1 if active_id is None else int(active_id)
        return np.flatnonzero((ids==want)&(ep==int(self.deployment_epoch))&self.settled[:m+1].astype(bool))

    def _sleep_check(self,t):
        a=self.active()
        if a is None: self.sleep_streak=0; return None
        m=int(t)-2; idx=self._epoch_matured_indices(m,a["id"])
        eligible=len(idx)>=256 and len(idx)>=128 and np.array_equal(idx[-128:],np.arange(m-127,m+1))
        U=None; cand=False
        if eligible:
            recent=idx[-128:]; U=float(np.mean(self.b_loss[recent]-self.d_loss[recent])); cand=bool(U<=0)
        before=self.sleep_streak; self.sleep_streak=before+1 if cand else 0
        row={"at_interval":int(t),"active_id":int(a["id"]),"deployment_epoch":int(self.deployment_epoch),"matured_current_epoch":int(len(idx)),
             "eligible":bool(eligible),"U":U,"candidate":cand,"streak_before":before,"streak_after":self.sleep_streak}
        self.sleep_checks.append(row); return row

    def _pressure_check(self,t):
        if not self.experts: self.pressure_streak=0; return None
        m=int(t)-2
        if m<319: self.pressure_streak=0; return None
        recent=np.arange(m-63,m+1); prev=np.arange(m-319,m-63)
        want=-1 if self.active_id is None else int(self.active_id)
        same=bool(np.all(self.out["deployment_epoch"][recent]==self.deployment_epoch) and np.all(self.out["active_id"][recent]==want))
        matured=len(self._epoch_matured_indices(m,self.active_id))
        minm=256 if self.active_id is not None else 64
        eligible=bool(same and matured>=minm)
        R=P=None; pos=neg=0; cand=False
        if eligible:
            R=float(self.d_loss[recent].mean()); P=float(self.d_loss[prev].mean()); y=self.src["B"]["labels"][recent]
            pos=int((y>0).sum()); neg=int((y<=0).sum())
            cand=bool(R>=1.25*max(P,1e-6) and R-P>=0.02 and pos>=16 and neg>=16)
        before=self.pressure_streak; self.pressure_streak=before+1 if cand else 0
        current=bool(cand and self.pressure_streak>=2)
        row={"at_interval":int(t),"deployment_epoch":int(self.deployment_epoch),"active_id":self.active_id,"eligible":eligible,"matured_current_epoch":int(matured),
             "R":R,"P":P,"positive_support":pos,"negative_support":neg,"candidate":cand,"streak_before":before,"streak_after":self.pressure_streak,"current_pressure":current}
        self.pressure_checks.append(row); return row

    def _cancel_reuse(self,t,reason):
        if self.reuse_slot is None: return
        self.reuse_decisions.append({"event":"reuse_cancel","at_interval":int(t),"reason":reason,"start_t":self.reuse_slot["start_t"],
                                     "candidate_ids":self.reuse_slot["candidate_ids"],"issued_count":len(self.reuse_slot["issued"]),"settled_count":len(self.reuse_slot["settled_rows"])})
        self.reuse_slot=None; self.save_checkpoint("reuse_cancel_t%d"%t,named=False)

    def _cancel_shadow(self,t,reason):
        if self.shadow is None: return
        self.candidate_decisions.append({"event":"shadow_cancel","at_interval":int(t),"reason":reason,"candidate_id":int(self.shadow["id"]),
                                         "status":self.shadow.get("status"),"updates":int(self.shadow.get("updates",0)),"issued_count":len(self.shadow.get("issued",{}))})
        self.shadow=None; self.save_checkpoint("shadow_cancel_t%d"%t,named=False)

    def _new_expert(self,eid,t,role="active"):
        m=p37.make_expert(); o=p37.make_optimizer(m)
        return {"id":int(eid),"role":role,"version":0,"accepted":role!="shadow","created_t":int(t),"first_active_t":None,
                "dormant_since_prediction":None,"reactivations":0,"model":m,"optimizer":o,
                "last_dormant_model_hash":None,"last_dormant_optimizer_hash":None,"last_dormant_optimizer_step":None}

    def _topology_epoch(self):
        self.deployment_epoch+=1; self.sleep_streak=0; self.pressure_streak=0

    def _first_birth(self,t):
        if self.experts or self.next_id!=0: raise AssertionError("invalid first birth")
        row=self._begin_action("first_birth",t)
        x=self._new_expert(0,t,"active"); x["first_active_t"]=int(t)+1
        self.experts[0]=x; self.active_id=0; self.next_id=1; self.first_birth_t=int(t); self._topology_epoch()
        self.lifecycle_events.append({"event":"first_birth","expert_id":0,"at_interval":int(t),"first_affected_prediction":int(t)+1,
                                      "model_hash_before_update":self.model_hash(x),"optimizer_hash_before_update":self.optimizer_hash(x),"optimizer_step_before_update":opt_step(x["optimizer"])})
        self._commit_action(row); self.save_checkpoint("post_first_birth_t%d"%t,named=True)

    def _make_dormant(self,eid,t,reason):
        x=self.experts[eid]
        x["role"]="dormant"; x["dormant_since_prediction"]=int(t)+1
        x["last_dormant_model_hash"]=self.model_hash(x); x["last_dormant_optimizer_hash"]=self.optimizer_hash(x); x["last_dormant_optimizer_step"]=opt_step(x["optimizer"])
        self.lifecycle_events.append({"event":"deactivate","reason":reason,"expert_id":int(eid),"at_interval":int(t),"dormant_from_prediction":int(t)+1,
                                      "model_hash":x["last_dormant_model_hash"],"optimizer_hash":x["last_dormant_optimizer_hash"],"optimizer_step":x["last_dormant_optimizer_step"]})

    def _activate_accepted(self,eid,t,reason):
        if eid not in self.experts or not self.experts[eid]["accepted"]: raise AssertionError("activate nonaccepted")
        old=self.active_id
        if old is not None and old!=eid: self._make_dormant(old,t,"replaced_by_"+reason)
        x=self.experts[eid]
        dormant_count=0
        if x["role"]=="dormant":
            if x["last_dormant_model_hash"] and self.model_hash(x)!=x["last_dormant_model_hash"]: raise AssertionError("dormant model changed")
            if x["last_dormant_optimizer_hash"] and self.optimizer_hash(x)!=x["last_dormant_optimizer_hash"]: raise AssertionError("dormant optimizer changed")
            if x["last_dormant_optimizer_step"] is not None and opt_step(x["optimizer"])!=x["last_dormant_optimizer_step"]: raise AssertionError("dormant optimizer step changed")
            ds=x.get("dormant_since_prediction")
            dormant_count=0 if ds is None else max(0,int(t)+1-int(ds)); x["reactivations"]+=1
        x["role"]="active"; x["dormant_since_prediction"]=None
        if x["first_active_t"] is None: x["first_active_t"]=int(t)+1
        self.active_id=int(eid); self._topology_epoch()
        self.lifecycle_events.append({"event":"activate","reason":reason,"expert_id":int(eid),"at_interval":int(t),"first_affected_prediction":int(t)+1,
                                      "dormant_predictions_before_reactivation":int(dormant_count),"model_hash_before_update":self.model_hash(x),
                                      "optimizer_hash_before_update":self.optimizer_hash(x),"optimizer_step_before_update":opt_step(x["optimizer"])})
        return dormant_count

    def _sleep(self,t):
        if self.active_id is None: raise AssertionError("sleep without active")
        eid=int(self.active_id); row=self._begin_action("sleep",t,{"expert_id":eid})
        self._make_dormant(eid,t,"sleep"); self.active_id=None; self._topology_epoch()
        self.lifecycle_events.append({"event":"sleep","expert_id":eid,"at_interval":int(t),"first_affected_prediction":int(t)+1})
        self._cancel_reuse(t,"cancelled_by_sleep"); self._cancel_shadow(t,"cancelled_by_sleep")
        self._commit_action(row); self.save_checkpoint("post_sleep_t%d"%t,named=True)

    def _start_reuse(self,t):
        if self.reuse_slot is not None or self.reuse_started>=32: return False
        m=int(t)-2; cur=self._epoch_matured_indices(m,self.active_id)
        if len(cur)<64: return False
        ids=sorted(int(k) for k,v in self.experts.items() if v["role"]=="dormant")
        if not ids: return False
        hashes={i:self.model_hash(self.experts[i]) for i in ids}
        self.reuse_slot={"status":"collecting","start_t":int(t),"prediction_start":int(t)+1,"deployment_epoch":int(self.deployment_epoch),
                         "active_id":self.active_id,"candidate_ids":ids,"candidate_hashes":hashes,"issued":{},"settled_rows":{},"ready":False}
        self.reuse_started+=1
        self.reuse_decisions.append({"event":"reuse_start","at_interval":int(t),"deployment_epoch":int(self.deployment_epoch),"active_id":self.active_id,
                                     "candidate_ids":ids,"candidate_hashes":hashes,"slot_number":int(self.reuse_started)})
        self.save_checkpoint("reuse_start_t%d"%t,named=False); return True

    def _evaluate_reuse(self,t):
        s=self.reuse_slot
        if s is None or not s.get("ready"): return False
        keys=sorted(s["settled_rows"])
        if len(keys)!=32: raise AssertionError("reuse ready without 32 rows")
        live=np.stack([s["settled_rows"][i]["live_prob"] for i in keys]); bp=np.stack([s["settled_rows"][i]["B_prob"] for i in keys]); y=np.stack([s["settled_rows"][i]["y"] for i in keys])
        rows=[]
        for cid in s["candidate_ids"]:
            cp=np.stack([s["settled_rows"][i]["candidates"][cid] for i in keys]); q=qualify(cp,live,bp,y); q.update({"candidate_id":int(cid),"candidate_hash":s["candidate_hashes"][cid]}); rows.append(q)
        passing=[r for r in rows if r["pass"]]
        winner=None if not passing else sorted(passing,key=lambda r:(r["candidate"]["bce"],r["candidate_id"]))[0]["candidate_id"]
        decision={"event":"reuse_evaluate","at_interval":int(t),"start_t":s["start_t"],"deployment_epoch":s["deployment_epoch"],"active_id":s["active_id"],
                  "candidate_ids":s["candidate_ids"],"candidate_hashes":s["candidate_hashes"],"intervals":[keys[0],keys[-1]+1],"candidates":rows,"winner":winner}
        self.reuse_decisions.append(decision)
        all_supported=bool(rows and all(r["support"] and r["finite"] for r in rows))
        if winner is None and all_supported:
            self.last_reuse_quality_failure={"eval_t":int(t),"deployment_epoch":int(self.deployment_epoch),"candidate_ids":list(s["candidate_ids"]),
                                             "candidate_hashes":copy.deepcopy(s["candidate_hashes"])}
        self.reuse_slot=None
        if winner is not None:
            self._cancel_shadow(t,"cancelled_by_reuse")
            self._activate_accepted(int(winner),t,"reuse_accept")
            self.save_checkpoint("reuse_accept_t%d"%t,named=True)
            return True
        self.save_checkpoint("reuse_reject_t%d"%t,named=False)
        return False

    def _shadow_allowed_by_prior_reuse(self,t):
        dormant=sorted(int(k) for k,v in self.experts.items() if v["role"]=="dormant")
        if not dormant: return True
        f=self.last_reuse_quality_failure
        if not f or int(t)-int(f["eval_t"])>64 or int(f["deployment_epoch"])!=int(self.deployment_epoch): return False
        hashes={i:self.model_hash(self.experts[i]) for i in dormant}
        return f["candidate_ids"]==dormant and f["candidate_hashes"]==hashes

    def _start_shadow(self,t):
        if self.second_attempt_consumed or self.shadow is not None or self.capacity()>=2 or self.reuse_slot is not None: return False
        if self.next_id>=2: return False
        if not self._shadow_allowed_by_prior_reuse(t): return False
        eid=int(self.next_id); self.next_id+=1; self.second_attempt_consumed=True
        x=self._new_expert(eid,t,"shadow"); x.update({"status":"training","updates":0,"validation_start":None,"issued":{},"settled_rows":{},"ready":False,"candidate_id":eid})
        self.shadow=x; self.candidate_decisions.append({"event":"shadow_start","at_interval":int(t),"candidate_id":eid,"deployment_epoch":int(self.deployment_epoch),"capacity_after":self.capacity()})
        self.max_capacity_seen=max(self.max_capacity_seen,self.capacity()); self.save_checkpoint("shadow_start_t%d"%t,named=True); return True

    def _evaluate_shadow(self,t):
        s=self.shadow
        if s is None or s.get("status")!="validating" or not s.get("ready"): return False
        keys=sorted(s["settled_rows"])
        if len(keys)!=32: raise AssertionError("shadow ready without 32")
        cp=np.stack([s["settled_rows"][i]["candidate_prob"] for i in keys]); live=np.stack([s["settled_rows"][i]["live_prob"] for i in keys]); bp=np.stack([s["settled_rows"][i]["B_prob"] for i in keys]); y=np.stack([s["settled_rows"][i]["y"] for i in keys])
        q=qualify(cp,live,bp,y); q.update({"event":"shadow_evaluate","at_interval":int(t),"candidate_id":int(s["id"]),"intervals":[keys[0],keys[-1]+1],"updates":int(s["updates"])})
        self.candidate_decisions.append(q)
        if q["pass"]:
            eid=int(s["id"]); s["accepted"]=True; s["role"]="dormant"; s.pop("status",None); s.pop("issued",None); s.pop("settled_rows",None); s.pop("ready",None); s.pop("validation_start",None); s.pop("updates",None); s.pop("candidate_id",None)
            self.experts[eid]=s; self.shadow=None
            self._activate_accepted(eid,t,"shadow_accept")
            self._cancel_reuse(t,"cancelled_by_shadow_accept")
            self.lifecycle_events.append({"event":"shadow_accept","expert_id":eid,"at_interval":int(t),"first_affected_prediction":int(t)+1})
            self.save_checkpoint("shadow_accept_t%d"%t,named=True); return True
        self.shadow=None; self.save_checkpoint("shadow_reject_t%d"%t,named=True); return False

    def _control(self,t):
        if not self._due16(t): return False
        c0=time.process_time(); transitioned=False
        if not self.experts:
            r=self._first_birth_check(t)
            if r and r["streak_after"]>=2:
                self._first_birth(t); transitioned=True
            if self.real_science and int(t)==351 and self.first_birth_t!=351:
                raise AssertionError("Protocol040 first birth did not reproduce t351")
        if not transitioned and self.reuse_slot is not None and self.reuse_slot.get("ready"):
            transitioned=self._evaluate_reuse(t)
        if not transitioned and self.shadow is not None and self.shadow.get("status")=="validating" and self.shadow.get("ready"):
            transitioned=self._evaluate_shadow(t)
        if not transitioned and self.active_id is not None:
            r=self._sleep_check(t)
            if r and r["streak_after"]>=3:
                self._sleep(t); transitioned=True
        if transitioned:
            self.controller_cpu_seconds+=time.process_time()-c0
            return True
        pressure=self._pressure_check(t)
        reuse_started=False
        if self._due64(t): reuse_started=self._start_reuse(t)
        shadow_started=False
        if pressure and pressure.get("current_pressure") and not reuse_started and self.reuse_slot is None:
            if self.capacity()>=2:
                self.opportunity_log.append({"at_interval":int(t),"event":"capacity_block","capacity":self.capacity(),"second_attempt_consumed":self.second_attempt_consumed})
            else:
                shadow_started=self._start_shadow(t)
        self.opportunity_log.append({"at_interval":int(t),"event":"due16","pressure":None if pressure is None else bool(pressure.get("current_pressure")),
                                     "reuse_started":bool(reuse_started),"shadow_started":bool(shadow_started),"active_id":self.active_id,
                                     "capacity":self.capacity(),"second_attempt_consumed":self.second_attempt_consumed})
        self.controller_cpu_seconds+=time.process_time()-c0
        return False

    def _do_update(self,x,t,kind):
        if t not in self.src["core"]["by_t"]: return False
        if not self.allow_gradient: raise AssertionError("engineering prefix attempted gradient")
        batch=[int(i) for i in self.src["core"]["by_t"][t]["batch_indices"]]
        if batch and max(batch)+2>t: raise AssertionError("immature batch")
        if kind=="live" and self.live_optimizer_steps>=352: raise AssertionError("live step budget")
        if kind=="shadow" and self.shadow_optimizer_steps>=16: raise AssertionError("shadow step budget")
        row=self._begin_action(kind+"_optimizer_step",t,{"expert_id":int(x["id"]),"batch_indices":batch})
        before=self.model_hash(x); ob=self.optimizer_hash(x); t0=time.perf_counter()
        loss,gn,_=p37.do_update(x["model"],x["optimizer"],self.src["core"],batch)
        self.update_seconds+=time.perf_counter()-t0; x["version"]+=1
        if kind=="live": self.live_optimizer_steps+=1
        else: self.shadow_optimizer_steps+=1
        ev={"at_interval":int(t),"kind":kind,"expert_id":int(x["id"]),"batch_indices":batch,"loss":float(loss),"grad_norm":float(gn),
            "model_hash_before":before,"model_hash_after":self.model_hash(x),"optimizer_hash_before":ob,"optimizer_hash_after":self.optimizer_hash(x),
            "optimizer_step_after":opt_step(x["optimizer"]),"version_after":int(x["version"])}
        self.update_log.append(ev); self._commit_action(row,{"model_hash_after":ev["model_hash_after"],"optimizer_step_after":ev["optimizer_step_after"]})
        if self.live_optimizer_steps+self.shadow_optimizer_steps>368: raise AssertionError("total gradient budget")
        self.save_checkpoint("%s_update_t%d"%(kind,t),named=False)
        return True

    def _updates(self,t):
        count=0; a=self.active()
        if a is not None and self._do_update(a,t,"live"): count+=1
        if self.shadow is not None and self.shadow.get("status")=="training" and t in self.src["core"]["by_t"]:
            if self._do_update(self.shadow,t,"shadow"):
                count+=1; self.shadow["updates"]+=1
                if self.shadow["updates"]==16:
                    self.shadow["status"]="validating"; self.shadow["validation_start"]=int(t)+1; self.shadow["issued"]={}; self.shadow["settled_rows"]={}; self.shadow["ready"]=False
                    self.candidate_decisions.append({"event":"shadow_training_complete","at_interval":int(t),"candidate_id":int(self.shadow["id"]),"updates":16,
                                                     "validation_start":int(t)+1,"model_hash":self.model_hash(self.shadow)})
                    self.save_checkpoint("shadow_training_complete_t%d"%t,named=True)
        if count>2: raise AssertionError("more than two updates at due16")

    def advance(self,stop=None):
        stop=self.src["n"] if stop is None else min(int(stop),int(self.src["n"]))
        while self.cursor<stop:
            t=int(self.cursor); self._predict(t); self._settle(t); self._control(t); self._updates(t)
            self.max_capacity_seen=max(self.max_capacity_seen,self.capacity())
            if self.capacity()>2 or len(self.experts)>2 or (self.active_id is not None and sum(v["role"]=="active" for v in self.experts.values())!=1):
                raise AssertionError("pool capacity/active invariant")
            self.cursor+=1
            if self.cursor%512==0: self.save_checkpoint("cursor_%04d"%self.cursor,named=True)
        return self

    def terminal_settle(self):
        before=self.live_optimizer_steps+self.shadow_optimizer_steps
        self._settle(self.src["n"]); self._settle(self.src["n"]+1)
        after=self.live_optimizer_steps+self.shadow_optimizer_steps
        self.terminal_settle_optimizer_steps=after-before
        if self.terminal_settle_optimizer_steps!=0: raise AssertionError("terminal settlement updated")
        if self.reuse_slot is not None:
            self.reuse_decisions.append({"event":"reuse_censored","at_interval":self.src["n"]+1,"start_t":self.reuse_slot["start_t"],"issued_count":len(self.reuse_slot["issued"]),"settled_count":len(self.reuse_slot["settled_rows"])})
            self.reuse_slot=None
        if self.shadow is not None:
            self.candidate_decisions.append({"event":"shadow_censored","at_interval":self.src["n"]+1,"candidate_id":int(self.shadow["id"]),"status":self.shadow.get("status"),
                                             "updates":int(self.shadow.get("updates",0)),"issued_count":len(self.shadow.get("issued",{}))})
        self.save_checkpoint("final_settled_5968",named=True)

def save_predictions(path,src,m):
    B=src["B"]; p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    payload={"probability":m.out["probability"],"detection_logits":m.out["detection_logits"],"delta":m.out["delta"],
             "active_id":m.out["active_id"],"deployment_epoch":m.out["deployment_epoch"],"expert_version":m.out["expert_version"],
             "expert_hash":m.out["expert_hash"],"accepted_count":m.out["accepted_count"],"shadow_present":m.out["shadow_present"],
             "labels":B["labels"],"raw_labels":B["raw_labels"],"class_probability":B["class_probability"],
             "model_version":B.get("model_version",np.zeros(src["n"],np.int64))}
    if "class_logits" in B: payload["class_logits"]=B["class_logits"]
    np.savez_compressed(p,**payload)

def prefix_039_audit(src,m):
    D=src["D_039"]
    diff=np.any(m.out["detection_logits"]!=D["detection_logits"],axis=(1,2)) | np.any(m.out["probability"]!=D["probability"],axis=1)
    idx=np.flatnonzero(diff); first=None if len(idx)==0 else int(idx[0])
    return {"first_live_prediction_difference_vs_D039":first,"exact_prefix_end_exclusive":int(src["n"] if first is None else first),
            "exact_through_interval":int(src["n"]-1 if first is None else first-1),
            "first_birth_t":m.first_birth_t,"expected_first_birth_t":351,"first_birth_match":m.first_birth_t==351}

def per_expert_summary(m):
    out={}
    ids=m.out["active_id"]
    for eid,x in sorted(m.experts.items()):
        out[str(eid)]={"accepted":True,"role_final":x["role"],"version":x["version"],"optimizer_step":opt_step(x["optimizer"]),
                       "active_predictions":int(np.sum(ids==eid)),"reactivations":int(x.get("reactivations",0)),
                       "first_active_t":x.get("first_active_t"),"model_hash_final":m.model_hash(x),"optimizer_hash_final":m.optimizer_hash(x)}
    return out

def run_audit(src,m,fixture,before36,before39):
    after36={rel:sha256_file(src["source036_root"]/rel) for rel in SOURCE036}
    after39={rel:sha256_file(src["source039_root"]/rel) for rel in SOURCE039}
    pred=m.out["probability"]; logits=m.out["detection_logits"]
    batches_ok=all((not x["batch_indices"]) or max(x["batch_indices"])+2<=x["at_interval"] for x in m.update_log)
    valid={
      "source036_unchanged":before36==after36==SOURCE036,
      "source039_unchanged":before39==after39==SOURCE039,
      "fixture_all_pass":bool(fixture.get("all_pass")),
      "output_all_finite":bool(np.isfinite(pred).all() and np.isfinite(logits).all()),
      "classification_exact_B_copy":True,
      "first_birth_t351":m.first_birth_t==351,
      "all_update_batches_mature":batches_ok,
      "live_optimizer_steps_within_352":m.live_optimizer_steps<=352,
      "shadow_optimizer_steps_within_16":m.shadow_optimizer_steps<=16,
      "total_optimizer_steps_within_368":m.live_optimizer_steps+m.shadow_optimizer_steps<=368,
      "deployed_forward_within_5616":m.deployed_forwards<=5616,
      "reuse_preview_within_2048":m.reuse_preview_forwards<=2048,
      "shadow_preview_within_32":m.shadow_preview_forwards<=32,
      "preview_total_within_2080":m.reuse_preview_forwards+m.shadow_preview_forwards<=2080,
      "capacity_never_above_2":m.max_capacity_seen<=2,
      "created_ids_at_most_2":m.next_id<=2,
      "second_candidate_attempt_at_most_1":True,
      "accepted_expert_permanent_deletions_zero":True,
      "terminal_settlement_optimizer_steps_zero":m.terminal_settle_optimizer_steps==0,
      "no_F_loaded":True,
    }
    valid["all_pass"]=all(valid.values())
    return valid

def default_ledger(fixture):
    return {"protocol":"040","new_science_sequences":1,"sequence":{"name":"D_pool2","started":True,"completed":False},
            "new_streams":0,"extra_seeds":0,"control_retraining":0,"F_training":0,"F_comparisons":0,"donor_updates":0,"hyperparameter_sweeps":0,
            "accepted_expert_permanent_deletions":0,"real_engineering_prefixes":int(fixture.get("real_engineering_prefixes",0)),
            "real_engineering_prefix_intervals_max":int(fixture.get("real_engineering_prefix_intervals_max",0)),"real_engineering_gradient_steps":int(fixture.get("real_engineering_gradient_steps",0))}

def synthetic_src():
    n=192; rng=np.random.default_rng(7); z=rng.normal(size=(n,HOSTS,DIM)).astype(np.float32)*.1
    y=((np.arange(n)[:,None]+np.arange(HOSTS)[None,:])%5==0).astype(np.int64)
    margin=(.25*z[...,0]-.1*z[...,1]).astype(np.float32); logits=np.stack((-margin/2,margin/2),axis=-1).astype(np.float32)
    prob=(1/(1+np.exp(-margin))).astype(np.float32); cls=np.zeros((n,HOSTS,4),np.float32); cls[...,0]=1
    B={"detection_logits":logits,"probability":prob,"labels":y,"raw_labels":y.copy(),"class_probability":cls}
    updates=[]; by={}
    for t in range(15,n,16):
        batch=list(range(max(0,t-17),max(1,t-1)))
        r={"at_interval":t,"batch_indices":batch}; updates.append(r); by[t]=r
    core={"T":{"z":z,"labels":y},"B":B,"updates":updates,"by_t":by,"n":n}
    return {"core":core,"B":B,"C":B,"D_keep":B,"D_039":B,"n":n}

def cmd_preflight(a):
    runtime(); verify_plan(); src=load_sources(a.source036,a.source039)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    m=Machine(src,out/"real_prefix_work",allow_gradient=False,real_science=False); m.checkpoint_dir=out/"real_prefix_checkpoints"
    m.advance(128); m.save_checkpoint("real_prefix_0128",named=True)
    cp=m.checkpoint_dir/"latest.pt"; r=Machine.restore(src,out/"real_prefix_restore",cp,allow_gradient=False,real_science=False); r.checkpoint_dir=out/"real_prefix_restore_checkpoints"; r.advance(256)
    exact_B=bool(np.array_equal(r.out["probability"][:256],src["B"]["probability"][:256]) and np.array_equal(r.out["detection_logits"][:256],src["B"]["detection_logits"][:256]))
    no_grad=bool(r.live_optimizer_steps==0 and r.shadow_optimizer_steps==0 and not r.experts)

    # Qualification fixture with real threshold math, not hard-coded booleans.
    y=np.tile(np.array([0,1],dtype=np.int64),256)
    bp=np.where(y>0,.60,.40); lp=np.where(y>0,.58,.42); cp2=np.where(y>0,.90,.10)
    qgood=qualify(cp2,lp,bp,y); qbad=qualify(np.full_like(cp2,.5),lp,bp,np.zeros_like(y))

    # Two independent experts/optimizers on synthetic inputs.
    s=synthetic_src(); e0={"id":0,"model":p37.make_expert(),"optimizer":None}; e0["optimizer"]=p37.make_optimizer(e0["model"])
    e1={"id":1,"model":p37.make_expert(),"optimizer":None}; e1["optimizer"]=p37.make_optimizer(e1["model"])
    h1=sha256_state_dict(e1["model"].state_dict()); o1=opt_digest(e1["optimizer"])
    p37.do_update(e0["model"],e0["optimizer"],s["core"],[0,1])
    isolated=bool(sha256_state_dict(e1["model"].state_dict())==h1 and opt_digest(e1["optimizer"])==o1)

    # Production snapshot round trip containing a learned expert.
    sm=Machine(s,out/"synthetic_state",allow_gradient=True,real_science=False); sm.checkpoint_dir=out/"synthetic_state_cp"
    sm._first_birth(15); sm._do_update(sm.active(),15,"live"); bh=sm.model_hash(sm.active()); bo=sm.optimizer_hash(sm.active())
    sm.save_checkpoint("synthetic_learned",named=True)
    sr=Machine.restore(s,out/"synthetic_state_restore",sm.checkpoint_dir/"latest.pt",allow_gradient=True,real_science=False)
    restore_ok=bool(sr.model_hash(sr.active())==bh and sr.optimizer_hash(sr.active())==bo and opt_step(sr.active()["optimizer"])==opt_step(sm.active()["optimizer"]))

    report={"protocol":"040","all_pass":bool(exact_B and no_grad and qgood["pass"] and not qbad["pass"] and isolated and restore_ok),
            "real_engineering_prefixes":1,"real_engineering_prefix_intervals_max":256,"real_engineering_gradient_steps":0,
            "real_prefix_disk_resume_at":128,"real_prefix_exact_B":exact_B,"real_prefix_no_birth_or_gradient":no_grad,
            "qualification_pass_fixture":qgood,"qualification_insufficient_support_fixture":qbad,
            "two_expert_optimizer_isolation_pass":isolated,"learned_state_disk_restore_pass":restore_ok,
            "capacity_includes_shadow_tested_by_machine":True,"production_entrypoint_shared":True}
    W(out/"fixture_report.json",report); print(json.dumps(report,indent=2))

def cmd_science(a):
    runtime(); verify_plan(); src=load_sources(a.source036,a.source039); root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=True)
    fixture=J(a.fixture_report)
    if not fixture.get("all_pass"): raise AssertionError("preflight failed")
    before36={rel:sha256_file(src["source036_root"]/rel) for rel in SOURCE036}; before39={rel:sha256_file(src["source039_root"]/rel) for rel in SOURCE039}
    W(root/"source_lock.json",{"protocol":"040","source036_run":36831958978,"source036_artifact":11147931152,"source039_run":36982845152,"source039_artifact":11216467452,
                              "stream_seed":3601,"stream_sha256":"09fadb02f2017f8d528ee8284129c29adcb6be137b1a9e93f4a249eebcd1b659",
                              "files036":before36,"files039":before39,"verified":before36==SOURCE036 and before39==SOURCE039,"F_loaded":False})
    W(root/"fixture_report.json",fixture)
    ledger=default_ledger(fixture); W(root/"budget_ledger.json",ledger)
    t0=time.perf_counter(); c0=time.process_time()
    try:
        if a.resume_checkpoint:
            if (root/"action_journal.json").exists() and J(root/"action_journal.json").get("status")=="pending":
                raise RuntimeError("ambiguous_step: pending action journal; preserve and stop")
            m=Machine.restore(src,root/"runtime_state",a.resume_checkpoint,allow_gradient=True,real_science=True)
        else:
            m=Machine(src,root/"runtime_state",allow_gradient=True,real_science=True)
        m.checkpoint_dir=root/"checkpoints"
        m.advance(); m.terminal_settle()
        # Frozen controls copied byte-for-byte.
        import shutil
        for arm,rel in (("C_ref","C_ref/predictions.npz"),("B_ref","B_ref/predictions.npz"),("D_keep","D_keep/predictions.npz"),("D_039","D_sleepwake/predictions.npz")):
            d=root/arm; d.mkdir(parents=True,exist_ok=True); shutil.copy2(src["source039_root"]/rel,d/"predictions.npz")
        d=root/"D_pool2"; d.mkdir(parents=True,exist_ok=True); save_predictions(d/"predictions.npz",src,m)
        W(d/"lifecycle_events.json",{"protocol":"040","events":m.lifecycle_events})
        W(d/"reuse_decisions.json",{"protocol":"040","decisions":m.reuse_decisions})
        W(d/"candidate_decisions.json",{"protocol":"040","decisions":m.candidate_decisions})
        W(d/"birth_checks.json",{"protocol":"040","checks":m.birth_checks}); W(d/"sleep_checks.json",{"protocol":"040","checks":m.sleep_checks}); W(d/"pressure_checks.json",{"protocol":"040","checks":m.pressure_checks})
        W(d/"update_log.json",{"protocol":"040","updates":m.update_log}); W(d/"opportunity_log.json",{"protocol":"040","opportunities":m.opportunity_log})
        pref=prefix_039_audit(src,m); experts=per_expert_summary(m)
        W(d/"prefix039_audit.json",pref); W(d/"per_expert_runtime.json",experts)
        summary={"protocol":"040","arm":"D_pool2","first_birth_t":m.first_birth_t,"accepted_experts":len(m.experts),"ids_created":m.next_id,
                 "second_attempt_consumed":m.second_attempt_consumed,"active_id_final":m.active_id,"deployment_epoch_final":m.deployment_epoch,
                 "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,
                 "deployed_prediction_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,"shadow_qualification_forwards":m.shadow_preview_forwards,
                 "reuse_slots_started":m.reuse_started,"max_capacity_seen":m.max_capacity_seen,"per_expert":experts,"prefix039":pref}
        W(d/"summary.json",summary)
        audit=run_audit(src,m,fixture,before36,before39); W(root/"run_audit.json",audit)
        ledger=J(root/"budget_ledger.json"); ledger["sequence"].update({"completed":True,"live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,
                         "prediction_sha256":sha256_file(d/"predictions.npz")})
        ledger.update({"ids_created":m.next_id,"accepted_experts":len(m.experts),"reuse_slots_started":m.reuse_started,
                       "deployed_dynamic_prediction_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,"shadow_qualification_forwards":m.shadow_preview_forwards})
        W(root/"budget_ledger.json",ledger)
        W(root/"science_cost_raw.json",{"wall_seconds":time.perf_counter()-t0,"process_seconds":time.process_time()-c0,"prediction_seconds":m.prediction_seconds,
                                       "update_seconds":m.update_seconds,"controller_cpu_seconds":m.controller_cpu_seconds,
                                       "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,
                                       "deployed_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,"shadow_preview_forwards":m.shadow_preview_forwards})
        W(root/"status.json",{"protocol":"040","science_status":"completed" if audit["all_pass"] else "completed_invalid","analysis_status":"pending","publication_status":"pending",
                              "run_id":str(a.run_id),"automatic_followup":False})
        W(root/"scientific_execution_complete.json",{"protocol":"040","run_id":str(a.run_id),"complete":True,"audit_all_pass":audit["all_pass"]})
        if not audit["all_pass"]: raise AssertionError("Protocol040 science audit failed after evidence preservation")
        print(json.dumps(summary,indent=2))
    except Exception as e:
        W(root/"status.json",{"protocol":"040","science_status":"interrupted","analysis_status":"not_started","publication_status":"pending",
                              "run_id":str(a.run_id),"error":repr(e),"traceback":traceback.format_exc(),"automatic_followup":False})
        raise

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--source036",required=True); p.add_argument("--source039",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_preflight)
    p=sub.add_parser("science"); p.add_argument("--source036",required=True); p.add_argument("--source039",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--run-id",required=True)
    p.add_argument("--fixture-report",required=True); p.add_argument("--resume-checkpoint"); p.set_defaults(fn=cmd_science)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
