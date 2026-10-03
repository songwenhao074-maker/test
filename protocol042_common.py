"""Protocol-042 rev2 common source, additive expert and scoring primitives."""
from __future__ import annotations
import copy, hashlib, json, math
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import run_ftmoe_protocol037 as p37
import run_ftmoe_protocol040 as p40
from protocol035_common import binary_metrics, dump_json, load_npz, sha256_file, sha256_state_dict

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/"artifacts/ftmoe_online/protocol_042/plan.json"
PLAN_SHA="11eaf5e89c53a6ec0f122854d513e208da6b0aa0ceda95ec38999fa4961f4be7"
N,HOSTS,DIM=5968,16,73
ARMS=("U_parent","R_win128","A_hist","A_win128")
NEW_ARMS=("R_win128","A_hist","A_win128")
SCORE_TOL=1e-10

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x): dump_json(p,x)

def verify_plan():
    if sha256_file(PLAN)!=PLAN_SHA: raise AssertionError("Protocol042 rev2 plan hash changed")
    p=J(PLAN)
    if (p.get("protocol"),p.get("revision"))!=("042",2): raise AssertionError("wrong Protocol042 registration")
    if p["budget"]["order"]!=list(ARMS): raise AssertionError("Protocol042 arm order changed")
    return p

def verify_files(root,expected,label):
    root=Path(root); out={}
    for rel,d in expected.items():
        q=root/rel
        if not q.exists(): raise AssertionError(f"{label} source_unavailable {rel}")
        got=sha256_file(q); out[rel]=got
        if got!=d: raise AssertionError(f"{label} source_mismatch {rel}: {got}")
    return out

def load_sources(source036,source041):
    p=verify_plan(); source036,source041=Path(source036),Path(source041)
    h36=verify_files(source036,p["source036"]["files"],"036")
    h41=verify_files(source041,p["source041"]["files"],"041")
    core=p37.load_source(source036,verify=True)
    def L(rel): return load_npz(source041/rel)
    C=L("cached/C_ref/predictions.npz"); B=L("cached/B_ref/predictions.npz")
    K=L("cached/D_keep/predictions.npz"); D39=L("cached/D_039/predictions.npz")
    D40=L("cached/D_040/predictions.npz"); W41=L("W_parent/predictions.npz")
    for name,a in (("C",C),("B",B),("D_keep",K),("D_039",D39),("D_040",D40),("W_041",W41)):
        if a["probability"].shape!=(N,HOSTS) or a["detection_logits"].shape!=(N,HOSTS,2):
            raise AssertionError(name+" geometry")
        for key in ("labels","raw_labels","class_probability"):
            if key not in a or not np.array_equal(a[key],C[key]): raise AssertionError(name+" alignment "+key)
    if not np.array_equal(core["B"]["detection_logits"],B["detection_logits"]): raise AssertionError("036 B != 041 cached B")
    if not np.array_equal(core["T"]["labels"],C["labels"]): raise AssertionError("issued labels mismatch")
    return {
      "source036_root":source036,"source041_root":source041,"core":core,
      "C":C,"B":B,"D_keep":K,"D_039":D39,"D_040":D40,"W_041":W41,"n":N,
      "hashes036":h36,"hashes041":h41,"manifest":J(source036/"inputs/seed3601/manifest.json"),
      "W41_birth":J(source041/"W_parent/birth_checks.json"),
      "W41_candidate":J(source041/"W_parent/candidate_decisions.json"),
      "W41_init":J(source041/"W_parent/initialization_events.json"),
      "W41_lifecycle":J(source041/"W_parent/lifecycle_events.json"),
      "W41_opportunity":J(source041/"W_parent/opportunity_log.json"),
      "W41_pressure":J(source041/"W_parent/pressure_checks.json"),
      "W41_reuse":J(source041/"W_parent/reuse_decisions.json"),
      "W41_sleep":J(source041/"W_parent/sleep_checks.json"),
      "W41_updates":J(source041/"W_parent/update_log.json"),
      "W41_summary":J(source041/"W_parent/summary.json"),
      "W41_qualification":load_npz(source041/"W_parent/qualification_first.npz"),
    }

def bmargin(B,t_or_idx):
    x=np.asarray(B["detection_logits"][t_or_idx],dtype=np.float32)
    return (x[...,1]-x[...,0]).astype(np.float32)

def sigmoid_margin(m):
    x=np.asarray(m,dtype=np.float64)
    return (1.0/(1.0+np.exp(-x))).astype(np.float32)

def logits_from_margin(B_logits,delta_sum):
    b=np.asarray(B_logits,dtype=np.float32); d=np.asarray(delta_sum,dtype=np.float32)
    return np.stack((b[...,0]-d/2.0,b[...,1]+d/2.0),axis=-1).astype(np.float32)

def stable_bce_rows(margin,y):
    m=np.asarray(margin,dtype=np.float64); yy=(np.asarray(y)>0).astype(np.float64)
    return np.maximum(m,0.0)-m*yy+np.log1p(np.exp(-np.abs(m)))

def stable_bce_interval(margin,y): return float(np.mean(stable_bce_rows(margin,y)))

def expert_delta_np(model,z):
    with torch.no_grad():
        zz=torch.from_numpy(np.asarray(z,dtype=np.float32))
        d=model.delta(zz).cpu().numpy().astype(np.float32)
    if not np.isfinite(d).all(): raise FloatingPointError("nonfinite expert delta")
    return d

def expert_delta_tensor(model,z):
    if isinstance(z,np.ndarray): z=torch.from_numpy(np.asarray(z,dtype=np.float32))
    return model.delta(z)

def new_expert(eid,created_t,role):
    m=p37.make_expert(); o=p37.make_optimizer(m)
    return {"id":int(eid),"role":role,"accepted":role!="shadow","version":0,"created_t":int(created_t),
            "first_active_t":None,"dormant_since_prediction":None,"reactivations":0,
            "model":m,"optimizer":o,"historical_utility":[],"last_observed_t":None}

def opt_digest(opt): return p40.opt_digest(opt)
def opt_step(opt): return p40.opt_step(opt)

def pack_expert(x):
    keep={k:copy.deepcopy(v) for k,v in x.items() if k not in ("model","optimizer")}
    keep["model"]=copy.deepcopy(x["model"].state_dict()); keep["optimizer"]=copy.deepcopy(x["optimizer"].state_dict())
    return keep

def unpack_expert(d):
    m=p37.make_expert(); o=p37.make_optimizer(m)
    m.load_state_dict(d["model"]); o.load_state_dict(d["optimizer"])
    x={k:copy.deepcopy(v) for k,v in d.items() if k not in ("model","optimizer")}
    x["model"]=m; x["optimizer"]=o; return x

def qualify(candidate_prob,live_prob,b_prob,y):
    cp=np.asarray(candidate_prob,dtype=np.float64).reshape(-1); lp=np.asarray(live_prob,dtype=np.float64).reshape(-1)
    bp=np.asarray(b_prob,dtype=np.float64).reshape(-1); yy=(np.asarray(y).reshape(-1)>0).astype(np.int64)
    pos=int(yy.sum()); neg=int(len(yy)-pos); finite=bool(np.isfinite(cp).all() and np.isfinite(lp).all() and np.isfinite(bp).all())
    support=pos>=16 and neg>=16
    if not finite or not support:
        return {"positive_host_rows":pos,"negative_host_rows":neg,"finite":finite,"support":support,"pass":False,
                "reason":"nonfinite" if not finite else "insufficient_support"}
    cm=binary_metrics(cp,yy); lm=binary_metrics(lp,yy); bm=binary_metrics(bp,yy)
    checks={"bce_vs_live":cm["bce"]<=.99*lm["bce"],"bce_vs_B":cm["bce"]<=bm["bce"],
            "fpr_vs_live":cm["fpr"]-lm["fpr"]<=.01,"fpr_vs_B":cm["fpr"]-bm["fpr"]<=.01,
            "recall_vs_live":cm["recall"]-lm["recall"]>=-.02,"recall_vs_B":cm["recall"]-bm["recall"]>=-.02}
    return {"positive_host_rows":pos,"negative_host_rows":neg,"finite":finite,"support":support,
            "candidate":cm,"live":lm,"B":bm,"checks":checks,"pass":bool(all(checks.values())),
            "reason":"pass" if all(checks.values()) else "quality_rejected"}

class UtilityTracker:
    """Online current-epoch issued counterfactual utility; no model forwards."""

    def __init__(self,mode):
        if mode not in ("win128","hist"): raise ValueError(mode)
        self.mode=mode; self.epoch=0; self.epoch_start_prediction=0; self.rows={}
        self.streak={}; self.archive=[]; self.audit=[]

    def reset(self,new_epoch,start_prediction,active_ids,reason):
        if self.rows:
            self.archive.append({"epoch":self.epoch,"start_prediction":self.epoch_start_prediction,
                                 "end_prediction":int(start_prediction),"active_ids":sorted(int(x) for x in self.rows.keys()),
                                 "final_scores":{str(k):self.score(k,None) for k in sorted(self.rows)}})
        self.epoch=int(new_epoch); self.epoch_start_prediction=int(start_prediction)
        self.rows={int(eid):deque() for eid in active_ids}; self.streak={int(eid):0 for eid in active_ids}
        self.audit.append({"event":"epoch_reset","epoch":self.epoch,"start_prediction":int(start_prediction),
                           "active_ids":sorted(int(x) for x in active_ids),"reason":reason})

    def add_active(self,eid):
        eid=int(eid)
        if eid not in self.rows: self.rows[eid]=deque()
        self.streak[eid]=0

    def remove_active(self,eid):
        self.streak.pop(int(eid),None)

    def settle(self,i,epoch,live_margin,deltas,y):
        if int(epoch)!=self.epoch: raise AssertionError("utility settlement epoch mismatch")
        live=np.asarray(live_margin,dtype=np.float32); yy=np.asarray(y)
        live_loss=stable_bce_rows(live,yy)
        for eid,d in deltas.items():
            eid=int(eid)
            if eid not in self.rows: continue
            dd=np.asarray(d,dtype=np.float32)
            rem=live-dd
            rem_loss=stable_bce_rows(rem,yy)
            u=(rem_loss-live_loss).astype(np.float64)
            rec={"i":int(i),"u":u,"y":(yy>0).astype(np.int8),"live_margin":live.copy(),"removed_margin":rem.astype(np.float32),
                 "delta":dd.copy()}
            self.rows[eid].append(rec)
            if self.mode=="win128":
                cutoff=int(i)-127
                while self.rows[eid] and self.rows[eid][0]["i"]<cutoff: self.rows[eid].popleft()

    def _rows_for(self,eid,m,window_override=None):
        eid=int(eid)
        if eid not in self.rows: return None,"not_active_or_unobserved"
        rows=list(self.rows[eid])
        if not rows: return None,"no_observations"
        if window_override is not None:
            cutoff=int(m)-int(window_override)+1; rows=[r for r in rows if r["i"]>=cutoff]
        elif self.mode=="win128":
            cutoff=int(m)-127; rows=[r for r in rows if r["i"]>=cutoff]
        return rows,None

    def score(self,eid,m,window_override=None):
        rows,reason=self._rows_for(eid,m,window_override)
        if rows is None: return {"valid":False,"reason":reason,"score":None,"score_pos":None,"score_neg":None}
        if m is None: m=rows[-1]["i"]
        matured=int(m)-self.epoch_start_prediction+1
        if matured<128: return {"valid":False,"reason":"epoch_mature_lt128","score":None,"score_pos":None,"score_neg":None,
                                "epoch_matured":matured,"intervals":len(rows)}
        if (self.mode=="win128" or window_override is not None) and len({r["i"] for r in rows})<min(128,window_override or 128):
            return {"valid":False,"reason":"window_not_full","score":None,"score_pos":None,"score_neg":None,
                    "epoch_matured":matured,"intervals":len({r["i"] for r in rows})}
        u=np.concatenate([r["u"].reshape(-1) for r in rows]); y=np.concatenate([r["y"].reshape(-1) for r in rows])
        live=np.concatenate([r["live_margin"].reshape(-1) for r in rows]); rem=np.concatenate([r["removed_margin"].reshape(-1) for r in rows])
        pos=y>0; neg=~pos; posn=int(pos.sum()); negn=int(neg.sum()); pint=sum(bool(np.any(r["y"]>0)) for r in rows)
        if posn<16 or negn<16 or pint<4:
            return {"valid":False,"reason":"insufficient_support","score":None,"score_pos":None,"score_neg":None,
                    "positive_host_rows":posn,"negative_host_rows":negn,"positive_intervals":pint,
                    "epoch_matured":matured,"intervals":len({r["i"] for r in rows})}
        lp=sigmoid_margin(live); rp=sigmoid_margin(rem)
        lm=binary_metrics(lp,y); rm=binary_metrics(rp,y)
        return {"valid":True,"reason":"ok","score":float(u.mean()),"score_pos":float(u[pos].mean()),"score_neg":float(u[neg].mean()),
                "positive_host_rows":posn,"negative_host_rows":negn,"positive_intervals":pint,
                "fpr_live":lm["fpr"],"fpr_removed":rm["fpr"],"recall_live":lm["recall"],"recall_removed":rm["recall"],
                "removal_fpr_delta":float(rm["fpr"]-lm["fpr"]),"removal_recall_delta":float(rm["recall"]-lm["recall"]),
                "epoch_matured":matured,"intervals":len({r["i"] for r in rows}),
                "first_i":int(rows[0]["i"]),"last_i":int(rows[-1]["i"])}

    def check(self,eid,m,require_nonuseful=True):
        s=self.score(eid,m)
        ok=bool(s.get("valid"))
        if ok and require_nonuseful:
            ok=bool(s["score"]<=0 and s["score_pos"]<=0 and s["score_neg"]<=0 and s["removal_fpr_delta"]<=.01 and s["removal_recall_delta"]>=-.02)
        before=int(self.streak.get(int(eid),0)); after=before+1 if ok else 0; self.streak[int(eid)]=after
        out={**s,"eligible_now":ok,"streak_before":before,"streak_after":after,"eligible_three":bool(ok and after>=3)}
        return out

    def victim(self,active_ids,m,arm,update_streak=True):
        rows=[]
        for eid in sorted(int(x) for x in active_ids):
            if arm=="R_win128":
                s=self.score(eid,m)
                s={**s,"eligible_three":bool(s.get("valid"))}
            else:
                s=self.check(eid,m,True) if update_streak else {**self.score(eid,m),"eligible_three":self.streak.get(eid,0)>=3}
            rows.append({"expert_id":eid,**s})
        valid=[r for r in rows if r.get("valid")]
        if arm=="R_win128":
            cand=valid
        else:
            cand=[r for r in rows if r.get("eligible_three")]
        if not cand: return None,rows
        cand.sort(key=lambda r:(float(r["score"]),int(r["expert_id"])))
        return int(cand[0]["expert_id"]),rows

def score_from_issued_records(records,eid,mode,m,epoch_start):
    tr=UtilityTracker(mode); tr.reset(0,epoch_start,[eid],"offline_recompute")
    for r in records:
        tr.settle(r["i"],0,r["live_margin"],{eid:r["deltas"][str(eid)] if str(eid) in r["deltas"] else r["deltas"][eid]},r["y"])
    return tr.score(eid,m)

def synthetic_source(n=640,seed=4202):
    rng=np.random.default_rng(seed); z=rng.normal(0,.2,size=(n,HOSTS,DIM)).astype(np.float32)
    y=((np.arange(n)[:,None]*3+np.arange(HOSTS)[None,:])%11<2).astype(np.int64)
    margin=(.45*z[...,0]-.20*z[...,1]+.10*z[...,2]+rng.normal(0,.03,size=(n,HOSTS))).astype(np.float32)
    logits=np.stack((-margin/2,margin/2),axis=-1).astype(np.float32); prob=sigmoid_margin(margin)
    cls=np.zeros((n,HOSTS,4),np.float32); cls[...,0]=1
    B={"detection_logits":logits,"probability":prob,"labels":y.copy(),"raw_labels":y.copy(),"class_probability":cls.copy(),"model_version":np.zeros(n,np.int64)}
    updates=[]; by={}
    for t in range(15,n,16):
        batch=list(range(max(0,t-33),max(1,t-1)))[-32:]
        if not batch: batch=[0]
        r={"at_interval":int(t),"batch_indices":[int(i) for i in batch]}; updates.append(r); by[int(t)]=r
    core={"T":{"z":z,"labels":y.copy()},"B":B,"updates":updates,"by_t":by,"n":n}
    return {"core":core,"B":B,"C":B,"D_keep":B,"D_039":B,"D_040":B,"W_041":B,"n":n,"manifest":{"timeline":[]}}
