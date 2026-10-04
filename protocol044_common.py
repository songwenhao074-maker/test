"""Protocol-043 revision1 common primitives: bounded float64 utility and immutable inputs."""
from __future__ import annotations
import copy, hashlib, json, math
from collections import deque
from pathlib import Path
import numpy as np
import torch

import run_ftmoe_protocol037 as p37
import run_ftmoe_protocol040 as p40
from protocol035_common import binary_metrics, dump_json, load_npz, sha256_file, sha256_state_dict

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/"artifacts/ftmoe_online/protocol_043/plan.json"
PLAN_SHA="9cd9b6f63aed0e35361f763e532d7c785a7838c19d172ea0c97bbaae5b2b0d6b"
N,HOSTS,DIM=5968,16,73
ARMS=("D_no_gc","D_bounded")
SCORE_TOL=1e-10

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x): dump_json(p,x)

def verify_plan():
    if sha256_file(PLAN)!=PLAN_SHA: raise AssertionError("Protocol043 plan hash changed")
    p=J(PLAN)
    if (p.get("protocol"),p.get("revision"))!=("043",1): raise AssertionError("wrong Protocol043 registration")
    if p["arms"]["order"]!=list(ARMS): raise AssertionError("arm order")
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
    K=L("cached/D_keep/predictions.npz"); D39=L("cached/D_039/predictions.npz"); D40=L("cached/D_040/predictions.npz")
    for name,a in (("C",C),("B",B),("D_keep",K),("D_039",D39),("D_040",D40)):
        if a["probability"].shape!=(N,HOSTS) or a["detection_logits"].shape!=(N,HOSTS,2): raise AssertionError(name+" geometry")
        for key in ("labels","raw_labels","class_probability"):
            if key not in a or not np.array_equal(a[key],C[key]): raise AssertionError(name+" alignment "+key)
    if not np.array_equal(core["B"]["detection_logits"],B["detection_logits"]): raise AssertionError("036 B != 041 B")
    if not np.array_equal(core["T"]["labels"],C["labels"]): raise AssertionError("issued labels mismatch")
    return {"source036_root":source036,"source041_root":source041,"core":core,"C":C,"B":B,"D_keep":K,"D_039":D39,"D_040":D40,
      "n":N,"hashes036":h36,"hashes041":h41,"manifest":J(source036/"inputs/seed3601/manifest.json")}

def bmargin(B,t_or_idx):
    x=np.asarray(B["detection_logits"][t_or_idx],dtype=np.float32)
    return (x[...,1]-x[...,0]).astype(np.float32)

def sigmoid_margin(m):
    x=np.asarray(m,dtype=np.float64)
    return (1.0/(1.0+np.exp(-np.clip(x,-80,80)))).astype(np.float32)

def logits_from_margin(B_logits,delta_sum):
    b=np.asarray(B_logits,dtype=np.float32); d=np.asarray(delta_sum,dtype=np.float32)
    return np.stack((b[...,0]-d/2.0,b[...,1]+d/2.0),axis=-1).astype(np.float32)

def stable_bce_rows(margin,y):
    m=np.asarray(margin,dtype=np.float64); yy=(np.asarray(y)>0).astype(np.float64)
    return np.maximum(m,0.0)-m*yy+np.log1p(np.exp(-np.abs(m)))

def stable_bce_interval(margin,y): return float(np.mean(stable_bce_rows(margin,y)))

def metrics_from_margin(margin,y):
    m=np.asarray(margin,dtype=np.float64); yy=(np.asarray(y)>0).astype(np.int64)
    p=1.0/(1.0+np.exp(-np.clip(m,-80,80)))
    return binary_metrics(p,yy)

def expert_delta_np(model,z):
    with torch.no_grad():
        zz=torch.from_numpy(np.asarray(z,dtype=np.float32)); d=model.delta(zz).cpu().numpy().astype(np.float32)
    if not np.isfinite(d).all(): raise FloatingPointError("nonfinite expert delta")
    return d

def new_expert(eid,created_t,role):
    m=p37.make_expert(); o=p37.make_optimizer(m)
    return {"id":int(eid),"role":role,"accepted":role!="shadow","version":0,"created_t":int(created_t),
      "first_active_t":None,"last_active_prediction":None,"dormant_since_prediction":None,"reactivations":0,
      "model":m,"optimizer":o,"historical_utility":[],"last_observed_t":None}

def opt_digest(opt): return p40.opt_digest(opt)
def opt_step(opt): return p40.opt_step(opt)

def pack_expert(x):
    keep={k:copy.deepcopy(v) for k,v in x.items() if k not in ("model","optimizer")}
    keep["model"]=copy.deepcopy(x["model"].state_dict()); keep["optimizer"]=copy.deepcopy(x["optimizer"].state_dict()); return keep

def unpack_expert(d):
    m=p37.make_expert(); o=p37.make_optimizer(m); m.load_state_dict(d["model"]); o.load_state_dict(d["optimizer"])
    x={k:copy.deepcopy(v) for k,v in d.items() if k not in ("model","optimizer")}; x["model"]=m; x["optimizer"]=o; return x

def tensor_bytes(x):
    n=0
    for p in x["model"].parameters(): n+=p.numel()*p.element_size()
    for st in x["optimizer"].state.values():
        for v in st.values():
            if torch.is_tensor(v): n+=v.numel()*v.element_size()
    return int(n)

def _conf(m,y):
    yy=(np.asarray(y)>0); pred=np.asarray(m,dtype=np.float64)>=0
    return {"tp":int(np.sum(pred&yy)),"fp":int(np.sum(pred&~yy)),"fn":int(np.sum(~pred&yy)),"tn":int(np.sum(~pred&~yy))}

def qualify_margin(candidate_margin,live_margin,b_margin,y):
    cm=np.asarray(candidate_margin,dtype=np.float64); lm=np.asarray(live_margin,dtype=np.float64); bm=np.asarray(b_margin,dtype=np.float64)
    yy=(np.asarray(y)>0).astype(np.int64); flat=yy.reshape(-1); pos=int(flat.sum()); neg=int(flat.size-pos)
    finite=bool(np.isfinite(cm).all() and np.isfinite(lm).all() and np.isfinite(bm).all()); support=pos>=16 and neg>=16
    if not finite or not support:
        return {"positive_host_rows":pos,"negative_host_rows":neg,"finite":finite,"support":support,"pass":False,
          "reason":"nonfinite" if not finite else "insufficient_support"}
    cand=metrics_from_margin(cm,yy); live=metrics_from_margin(lm,yy); base=metrics_from_margin(bm,yy)
    checks={"bce_vs_live":cand["bce"]<=.99*live["bce"],"bce_vs_B":cand["bce"]<=base["bce"],
      "fpr_vs_live":cand["fpr"]-live["fpr"]<=.01,"fpr_vs_B":cand["fpr"]-base["fpr"]<=.01,
      "recall_vs_live":cand["recall"]-live["recall"]>=-.02,"recall_vs_B":cand["recall"]-base["recall"]>=-.02}
    return {"positive_host_rows":pos,"negative_host_rows":neg,"finite":finite,"support":support,
      "candidate":cand,"live":live,"B":base,"checks":checks,"pass":bool(all(checks.values())),
      "reason":"pass" if all(checks.values()) else "quality_rejected"}

def reuse_evidence(candidate_margin,live_margin,b_margin,y):
    q=qualify_margin(candidate_margin,live_margin,b_margin,y)
    lm=np.asarray(live_margin,dtype=np.float64); cm=np.asarray(candidate_margin,dtype=np.float64); yy=(np.asarray(y)>0)
    delta=stable_bce_rows(lm,yy)-stable_bce_rows(cm,yy)
    flat=delta.reshape(-1); fy=yy.reshape(-1); pos=fy; neg=~fy
    q.update({"Q":float(flat.mean()),"Q_pos":None if not np.any(pos) else float(flat[pos].mean()),
      "Q_neg":None if not np.any(neg) else float(flat[neg].mean()),
      "positive_intervals":int(sum(bool(np.any(row)) for row in yy))})
    return q

class UtilityTracker:
    """Current-epoch ring128 sufficient statistics; float64 operands before subtraction."""
    def __init__(self):
        self.epoch=0; self.epoch_start_prediction=0; self.rows={}; self.streak={}; self.reset_events=[]; self.fallback_count=0; self.fallback_seconds=0.0
    def reset(self,new_epoch,start_prediction,active_ids,reason):
        self.epoch=int(new_epoch); self.epoch_start_prediction=int(start_prediction)
        self.rows={int(e):deque(maxlen=128) for e in active_ids}; self.streak={int(e):0 for e in active_ids}
        self.reset_events.append({"epoch":self.epoch,"start_prediction":int(start_prediction),"active_ids":sorted(int(x) for x in active_ids),"reason":reason})
    def settle(self,i,epoch,live_margin,deltas,y):
        if int(epoch)!=self.epoch: raise AssertionError("utility epoch mismatch")
        live64=np.asarray(live_margin,dtype=np.float64); yy=(np.asarray(y)>0)
        ll=stable_bce_rows(live64,yy)
        lc=_conf(live64,yy)
        for eid,d in deltas.items():
            eid=int(eid)
            if eid not in self.rows: continue
            d64=np.asarray(d,dtype=np.float64); rem=live64-d64; rl=stable_bce_rows(rem,yy); u=rl-ll
            pos=yy; neg=~yy; rc=_conf(rem,yy)
            self.rows[eid].append({"i":int(i),"sum_u":float(u.sum()),"sum_pos":float(u[pos].sum()),"sum_neg":float(u[neg].sum()),
              "pos":int(pos.sum()),"neg":int(neg.sum()),"positive_interval":bool(np.any(pos)),
              "live_tp":lc["tp"],"live_fp":lc["fp"],"live_fn":lc["fn"],"live_tn":lc["tn"],
              "rem_tp":rc["tp"],"rem_fp":rc["fp"],"rem_fn":rc["fn"],"rem_tn":rc["tn"]})
    @staticmethod
    def _ratio(a,b): return None if b==0 else float(a/b)
    def score(self,eid,m):
        eid=int(eid); rows=list(self.rows.get(eid,()))
        if eid not in self.rows: return {"valid":False,"reason":"not_active","score":None,"score_pos":None,"score_neg":None}
        matured=int(m)-self.epoch_start_prediction+1
        if matured<128: return {"valid":False,"reason":"epoch_mature_lt128","score":None,"score_pos":None,"score_neg":None,"epoch_matured":matured,"intervals":len(rows)}
        if len(rows)!=128 or rows[0]["i"]!=int(m)-127 or rows[-1]["i"]!=int(m):
            return {"valid":False,"reason":"window_not_full","score":None,"score_pos":None,"score_neg":None,"epoch_matured":matured,"intervals":len(rows)}
        pos=sum(r["pos"] for r in rows); neg=sum(r["neg"] for r in rows); pint=sum(int(r["positive_interval"]) for r in rows)
        if pos<16 or neg<16 or pint<4:
            return {"valid":False,"reason":"insufficient_support","score":None,"score_pos":None,"score_neg":None,
              "positive_host_rows":pos,"negative_host_rows":neg,"positive_intervals":pint,"epoch_matured":matured,"intervals":128}
        su=sum(r["sum_u"] for r in rows); sp=sum(r["sum_pos"] for r in rows); sn=sum(r["sum_neg"] for r in rows)
        ltp=sum(r["live_tp"] for r in rows); lfp=sum(r["live_fp"] for r in rows); lfn=sum(r["live_fn"] for r in rows); ltn=sum(r["live_tn"] for r in rows)
        rtp=sum(r["rem_tp"] for r in rows); rfp=sum(r["rem_fp"] for r in rows); rfn=sum(r["rem_fn"] for r in rows); rtn=sum(r["rem_tn"] for r in rows)
        lfpr=self._ratio(lfp,lfp+ltn); rfpr=self._ratio(rfp,rfp+rtn); lrec=self._ratio(ltp,ltp+lfn); rrec=self._ratio(rtp,rtp+rfn)
        if None in (lfpr,rfpr,lrec,rrec): return {"valid":False,"reason":"confusion_support","score":None,"score_pos":None,"score_neg":None}
        return {"valid":True,"reason":"ok","score":float(su/(pos+neg)),"score_pos":float(sp/pos),"score_neg":float(sn/neg),
          "positive_host_rows":pos,"negative_host_rows":neg,"positive_intervals":pint,"fpr_live":lfpr,"fpr_removed":rfpr,
          "recall_live":lrec,"recall_removed":rrec,"removal_fpr_delta":float(rfpr-lfpr),"removal_recall_delta":float(rrec-lrec),
          "epoch_matured":matured,"intervals":128,"first_i":int(rows[0]["i"]),"last_i":int(rows[-1]["i"])}
    def check(self,eid,m):
        s=self.score(eid,m); ok=bool(s.get("valid") and s["score"]<=0 and s["score_pos"]<=0 and s["score_neg"]<=0 and s["removal_fpr_delta"]<=.01 and s["removal_recall_delta"]>=-.02)
        before=int(self.streak.get(int(eid),0)); after=before+1 if ok else 0; self.streak[int(eid)]=after
        return {**s,"eligible_now":ok,"streak_before":before,"streak_after":after,"eligible_three":bool(ok and after>=3)}
    def victim(self,active_ids,m,update=True):
        rows=[]
        for eid in sorted(int(x) for x in active_ids):
            s=self.check(eid,m) if update else {**self.score(eid,m),"eligible_three":self.streak.get(eid,0)>=3}
            rows.append({"expert_id":eid,**s})
        cand=[x for x in rows if x.get("eligible_three")]
        if not cand: return None,rows
        cand.sort(key=lambda x:(float(x["score"]),int(x["expert_id"])))
        return int(cand[0]["expert_id"]),rows

def independent_score_from_arrays(live_margin,delta,labels,indices):
    ii=np.asarray(indices,dtype=np.int64)
    if len(ii)!=128: return {"valid":False,"reason":"window_not_full"}
    live64=np.asarray(live_margin[ii],dtype=np.float64); d64=np.asarray(delta[ii],dtype=np.float64); y=(np.asarray(labels[ii])>0)
    rem=live64-d64; u=stable_bce_rows(rem,y)-stable_bce_rows(live64,y)
    pos=y.reshape(-1); neg=~pos; flat=u.reshape(-1); pn=int(pos.sum()); nn=int(neg.sum()); pint=int(sum(bool(np.any(r)) for r in y))
    if pn<16 or nn<16 or pint<4: return {"valid":False,"reason":"insufficient_support","positive_host_rows":pn,"negative_host_rows":nn,"positive_intervals":pint}
    lm=metrics_from_margin(live64,y); rm=metrics_from_margin(rem,y)
    return {"valid":True,"reason":"ok","score":float(flat.mean()),"score_pos":float(flat[pos].mean()),"score_neg":float(flat[neg].mean()),
      "positive_host_rows":pn,"negative_host_rows":nn,"positive_intervals":pint,"removal_fpr_delta":float(rm["fpr"]-lm["fpr"]),
      "removal_recall_delta":float(rm["recall"]-lm["recall"]),"fpr_live":lm["fpr"],"fpr_removed":rm["fpr"],
      "recall_live":lm["recall"],"recall_removed":rm["recall"],"intervals":128,"first_i":int(ii[0]),"last_i":int(ii[-1])}

def synthetic_source(n=900,seed=4301,regime_switch=None,feature_scale=.2):
    rng=np.random.default_rng(seed); z=rng.normal(0,feature_scale,size=(n,HOSTS,DIM)).astype(np.float32)
    y=((np.arange(n)[:,None]*3+np.arange(HOSTS)[None,:])%11<2).astype(np.int64)
    margin=(.45*z[...,0]-.20*z[...,1]+.10*z[...,2]+rng.normal(0,.03,size=(n,HOSTS))).astype(np.float32)
    if regime_switch is not None:
        s=int(regime_switch); margin[s:]=(-2.0*(2*y[s:]-1)).astype(np.float32)
    logits=np.stack((-margin/2,margin/2),axis=-1).astype(np.float32); prob=sigmoid_margin(margin)
    cls=np.zeros((n,HOSTS,4),np.float32); cls[...,0]=1
    B={"detection_logits":logits,"probability":prob,"labels":y.copy(),"raw_labels":y.copy(),"class_probability":cls.copy(),"model_version":np.zeros(n,np.int64)}
    updates=[]; by={}
    for t in range(15,n,16):
        batch=list(range(max(0,t-33),max(1,t-1)))[-32:]
        if not batch: batch=[0]
        r={"at_interval":int(t),"batch_indices":[int(i) for i in batch]}; updates.append(r); by[int(t)]=r
    core={"T":{"z":z,"labels":y.copy()},"B":B,"updates":updates,"by_t":by,"n":n}
    return {"core":core,"B":B,"C":B,"D_keep":B,"D_039":B,"D_040":B,"n":n,"manifest":{"timeline":[]}}

def set_linear(expert,weight=None,bias=0.0):
    with torch.no_grad():
        for p in expert["model"].parameters(): p.zero_()
        if weight is not None:
            w=next(iter(expert["model"].parameters()))
            arr=np.asarray(weight,dtype=np.float32).reshape(w.shape); w.copy_(torch.from_numpy(arr))
        # make_expert is one Linear under the residual wrapper; last parameter is bias.
        params=list(expert["model"].parameters())
        if params and params[-1].ndim==1: params[-1].fill_(float(bias))
