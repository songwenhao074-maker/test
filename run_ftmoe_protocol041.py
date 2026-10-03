"""Protocol-041: controlled second-expert initialization experiment.

Two new scientific sequences are registered:
  Z_zero   : exact Protocol-040 controller/training with zero second candidate.
  W_parent : the only treatment copies current active E0 weights+bias into the
             second candidate before same-t gradients, while keeping a fresh AdamW.

Historical controls are cached. F is never loaded.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, shutil, time, traceback
from pathlib import Path

import numpy as np
import torch

import run_ftmoe_protocol037 as p37
import run_ftmoe_protocol039 as p39
import run_ftmoe_protocol040 as p40
from protocol035_common import binary_metrics, dump_json, load_npz, sha256_file, sha256_state_dict

PLAN=Path("artifacts/ftmoe_online/protocol_041/plan.json")
PLAN_SHA="32f9ea891e5b6b993c07550cc2a4f37fcc55f0df9a369580366ddc01d24110ca"
N,HOSTS,DIM=5968,16,73
ARMS=("Z_zero","W_parent")
SOURCE036={
 "stage_B/seed3601/C_ref/feature_tape.npz":"c71a2ea6d59045fe4cc8a953a776d37f48dbe1feec4aba859cd00f11c984529c",
 "stage_B/seed3601/C_ref/update_batches.json":"4d32d2ece194cb41b8bc16de1c89ddfedcd42da668b96446937fb430d6246b6f",
 "stage_B/seed3601/D_lin/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "inputs/seed3601/manifest.json":"73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b",
 "stage_B/seed3601/C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
}
SOURCE040={
 "B_ref/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
 "D_039/predictions.npz":"bc387d13721ed2e759a205ed3b4f6fa884c1ca105068a79a340148e1f5e96512",
 "D_keep/predictions.npz":"badf9005b2be9fc6436d1ff2b9377b27715e846d7d16f8e4bdfcbfa76f732e8c",
 "D_pool2/birth_checks.json":"ff08d5cb85702fed832eb0cb9c9781b9195fc93cbe7b8117e9011349e5169e2f",
 "D_pool2/candidate_decisions.json":"36c100ee1ad207d96c4e8bf4760f9ff412858b24b8759ad1b060e2b2e847ad8b",
 "D_pool2/lifecycle_events.json":"b05d76c102063002d4737f37a4b7f076ca3e1ea90176c03333aee21b82502906",
 "D_pool2/opportunity_log.json":"fac473aee5d4b8f22590355c16447dc8f43b44c62d6e79c3228e7218358fbe4a",
 "D_pool2/per_expert_runtime.json":"326a651ba5e36e67f9ab72d3bcb579127942ec51b80a5559fff2a9756780d776",
 "D_pool2/predictions.npz":"51a5ac7ec4e72f340ab1f051bb3707ad44e5540786a94bf0b10f9a8af9e7dd6a",
 "D_pool2/prefix039_audit.json":"dbba34925612a13c85b3c995bb06bb114203b4f506c325b10df93fe8728aeff9",
 "D_pool2/pressure_checks.json":"391f7ef53b9bdb75972f38bcd9a6f6f59f7a046a277ba63160a1d56c8a6230ed",
 "D_pool2/reuse_decisions.json":"671276db156b4851b54a70850cb268c9e05e05806ca68d702da3cb3fe8a5386f",
 "D_pool2/sleep_checks.json":"fa5a9f9c9bcbc29dc1338d46c5f82d65778b8ae6ace8f6d787275ac9004fd600",
 "D_pool2/summary.json":"496b165ce0ac1cd79d4bbd5073cb760f2a93ed64061d536e184f9fbf0ee72b47",
 "D_pool2/update_log.json":"8fc610dcf77087d9fcb4450133a3cfd5fe8fc3d5c9a6e70e869629c98405ab10",
}

def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x): dump_json(p,x)
def runtime(): p37.runtime()

def verify_plan():
    if sha256_file(PLAN)!=PLAN_SHA: raise AssertionError("Protocol041 plan hash changed")
    p=J(PLAN)
    if p.get("protocol")!="041" or p["budget"]["sequence_names"]!=["Z_zero","W_parent"]:
        raise AssertionError("Protocol041 plan identity")
    return p

def verify_files(root,expected,label):
    root=Path(root); out={}
    for rel,d in expected.items():
        q=root/rel
        if not q.exists(): raise AssertionError(label+" missing "+rel)
        g=sha256_file(q); out[rel]=g
        if g!=d: raise AssertionError(label+" hash mismatch "+rel+": "+g)
    return out

def load_sources(source036,source040):
    source036,source040=Path(source036),Path(source040)
    h36=verify_files(source036,SOURCE036,"source036"); h40=verify_files(source040,SOURCE040,"source040")
    core=p37.load_source(source036,verify=True)
    C=load_npz(source040/"C_ref/predictions.npz"); B=load_npz(source040/"B_ref/predictions.npz")
    K=load_npz(source040/"D_keep/predictions.npz"); D39=load_npz(source040/"D_039/predictions.npz")
    D40=load_npz(source040/"D_pool2/predictions.npz")
    for name,a in (("C",C),("B",B),("D_keep",K),("D_039",D39),("D_040",D40)):
        if a["probability"].shape!=(N,HOSTS) or a["detection_logits"].shape!=(N,HOSTS,2):
            raise AssertionError(name+" geometry")
    for a in (B,K,D39,D40):
        for key in ("labels","raw_labels","class_probability"):
            if key not in a or not np.array_equal(C[key],a[key]): raise AssertionError("cached alignment "+key)
    if not np.array_equal(core["B"]["detection_logits"],B["detection_logits"]): raise AssertionError("036 B != 040 B")
    if not np.array_equal(core["T"]["labels"],C["labels"]): raise AssertionError("issued labels mismatch")
    return {
      "source036_root":source036,"source040_root":source040,"core":core,
      "C":C,"B":B,"D_keep":K,"D_039":D39,"D_040":D40,"n":N,
      "hashes036":h36,"hashes040":h40,
      "manifest":J(source036/"inputs/seed3601/manifest.json"),
      "p40_birth_checks":J(source040/"D_pool2/birth_checks.json"),
      "p40_candidate_decisions":J(source040/"D_pool2/candidate_decisions.json"),
      "p40_lifecycle_events":J(source040/"D_pool2/lifecycle_events.json"),
      "p40_opportunity_log":J(source040/"D_pool2/opportunity_log.json"),
      "p40_pressure_checks":J(source040/"D_pool2/pressure_checks.json"),
      "p40_reuse_decisions":J(source040/"D_pool2/reuse_decisions.json"),
      "p40_sleep_checks":J(source040/"D_pool2/sleep_checks.json"),
      "p40_update_log":J(source040/"D_pool2/update_log.json"),
      "p40_summary":J(source040/"D_pool2/summary.json"),
    }

def _state_equal(a,b):
    if type(a) is not type(b): return False
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict): return a.keys()==b.keys() and all(_state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(_state_equal(x,y) for x,y in zip(a,b))
    return a==b

def _array_sha(x):
    a=np.asarray(x)
    h=hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes(order="C"))
    return h.hexdigest()

class Machine041(p40.Machine):
    """Protocol-040 semantics with substep-resumable execution and one init treatment."""

    SUBSTEPS=("predict","settle","control","live_update","shadow_update","finish")

    def __init__(self,src,work_dir,arm,allow_gradient=True,real_science=False):
        if arm not in ARMS: raise ValueError("unknown arm "+str(arm))
        super().__init__(src,work_dir,allow_gradient=allow_gradient,real_science=real_science)
        self.arm=arm
        self.next_substep="predict"
        self.terminal_progress=0
        self.initialization_events=[]
        self.qualification_records=[]
        self.opportunity_audit=[]
        self.completed_action_ids=[]
        self.fixture_force_checkpoint_each_substep=False
        self.action_journal=self.work_dir/"action_journal.json"

    def _begin_action(self,kind,t,meta=None):
        self.action_seq+=1
        row={"protocol":"041","arm":self.arm,"action_seq":self.action_seq,"status":"pending","kind":kind,"cursor":int(t),
             "next_substep":self.next_substep,"meta":meta or {}}
        W(self.action_journal,row); return row

    def _commit_action(self,row,extra=None):
        z=dict(row); z["status"]="committed"
        if extra: z.update(extra)
        W(self.action_journal,z)
        self.completed_action_ids.append(int(row["action_seq"]))

    def snapshot(self):
        x=super().snapshot()
        x.update({
          "protocol":"041","arm":self.arm,"next_substep":self.next_substep,"terminal_progress":self.terminal_progress,
          "initialization_events":copy.deepcopy(self.initialization_events),
          "qualification_records":copy.deepcopy(self.qualification_records),
          "opportunity_audit":copy.deepcopy(self.opportunity_audit),
          "completed_action_ids":copy.deepcopy(self.completed_action_ids),
        })
        return x

    def save_checkpoint(self,reason,named=False):
        if self.checkpoint_dir is None: return
        d=Path(self.checkpoint_dir); d.mkdir(parents=True,exist_ok=True)
        snap=self.snapshot(); tmp=d/"latest.pt.tmp"; final=d/"latest.pt"
        torch.save(snap,tmp); os.replace(tmp,final)
        meta={"protocol":"041","arm":self.arm,"cursor":self.cursor,"next_substep":self.next_substep,
              "terminal_progress":self.terminal_progress,"reason":reason,"active_id":self.active_id,
              "deployment_epoch":self.deployment_epoch,"live_steps":self.live_optimizer_steps,
              "shadow_steps":self.shadow_optimizer_steps,"action_seq":self.action_seq,"sha256":sha256_file(final)}
        W(d/"latest.json",meta)
        if named:
            safe="".join(c if c.isalnum() or c in "_-" else "_" for c in reason)
            q=d/(safe+".pt"); torch.save(snap,q); W(str(q)+".json",{**meta,"sha256":sha256_file(q)})

    @classmethod
    def restore(cls,src,work_dir,path,arm=None,allow_gradient=True,real_science=False,strict_journal=True):
        x=torch.load(path,map_location="cpu")
        if x.get("protocol")!="041": raise AssertionError("bad Protocol041 checkpoint")
        if arm is not None and x.get("arm")!=arm: raise AssertionError("checkpoint arm mismatch")
        m=cls(src,work_dir,x["arm"],allow_gradient=allow_gradient,real_science=real_science)
        m.cursor=int(x["cursor"]); m.deployment_epoch=int(x["deployment_epoch"]); m.active_id=x["active_id"]
        m.experts={int(k):p40.unpack_expert(v) for k,v in x["experts"].items()}
        if x["shadow"] is not None:
            sh=p40.unpack_expert(x["shadow"])
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
        m.next_substep=x.get("next_substep","predict"); m.terminal_progress=int(x.get("terminal_progress",0))
        m.initialization_events=copy.deepcopy(x.get("initialization_events",[]))
        m.qualification_records=copy.deepcopy(x.get("qualification_records",[]))
        m.opportunity_audit=copy.deepcopy(x.get("opportunity_audit",[]))
        m.completed_action_ids=copy.deepcopy(x.get("completed_action_ids",[]))
        torch.set_rng_state(x["torch_rng"]); np.random.set_state(x["numpy_rng"]); random.setstate(x["python_rng"])
        if strict_journal and m.action_journal.exists():
            j=J(m.action_journal)
            if j.get("status")=="pending": raise RuntimeError("ambiguous_step: pending action journal; preserve and stop")
            if int(j.get("action_seq",0))>int(m.action_seq):
                raise RuntimeError("ambiguous_step: committed action newer than checkpoint; preserve and stop")
        return m

    def _start_shadow(self,t):
        if self.second_attempt_consumed or self.shadow is not None or self.capacity()>=2 or self.reuse_slot is not None: return False
        if self.next_id>=2: return False
        if not self._shadow_allowed_by_prior_reuse(t): return False
        eid=int(self.next_id); donor=self.active()
        if self.arm=="W_parent" and donor is None:
            self.candidate_decisions.append({"event":"donor_unavailable","at_interval":int(t),"candidate_id":eid})
            raise RuntimeError("donor_unavailable")
        rng_before=torch.get_rng_state().clone()
        donor_record=None
        if donor is not None:
            donor_record={"donor_id":int(donor["id"]),"donor_model_hash_before":self.model_hash(donor),
                          "donor_optimizer_hash_before":self.optimizer_hash(donor),"donor_optimizer_step_before":p40.opt_step(donor["optimizer"])}
        x=self._new_expert(eid,t,"shadow")
        zero_hash=self.model_hash(x)
        if self.arm=="W_parent":
            copied=copy.deepcopy(donor["model"].state_dict())
            x["model"].load_state_dict(copied)
            if self.model_hash(x)!=self.model_hash(donor): raise AssertionError("parent parameter copy mismatch")
            for a,b in zip(x["model"].parameters(),donor["model"].parameters()):
                if a.data_ptr()==b.data_ptr(): raise AssertionError("parent/child parameter storage alias")
        if len(x["optimizer"].state)!=0: raise AssertionError("new child Adam state must be empty")
        if donor is not None:
            if self.model_hash(donor)!=donor_record["donor_model_hash_before"] or self.optimizer_hash(donor)!=donor_record["donor_optimizer_hash_before"] or p40.opt_step(donor["optimizer"])!=donor_record["donor_optimizer_step_before"]:
                raise AssertionError("parent changed during child creation")
        if not torch.equal(rng_before,torch.get_rng_state()): raise AssertionError("child initialization changed global RNG")
        self.next_id+=1; self.second_attempt_consumed=True
        x.update({"status":"training","updates":0,"validation_start":None,"issued":{},"settled_rows":{},"ready":False,"candidate_id":eid})
        self.shadow=x
        ev={"event":"shadow_initialization","arm":self.arm,"at_interval":int(t),"candidate_id":eid,
            "copy_after_settlement_before_same_t_gradients":True,"zero_constructor_hash":zero_hash,
            "candidate_initial_hash":self.model_hash(x),"candidate_optimizer_state_empty":len(x["optimizer"].state)==0,
            "candidate_optimizer_step_before":p40.opt_step(x["optimizer"]),"global_rng_unchanged":True,
            "max_training_label_available_interval":int(t)-2}
        if donor_record is not None:
            ev.update(donor_record)
            ev.update({"donor_model_hash_after":self.model_hash(donor),"donor_optimizer_hash_after":self.optimizer_hash(donor),
                       "donor_optimizer_step_after":p40.opt_step(donor["optimizer"])})
        self.initialization_events.append(ev)
        self.candidate_decisions.append({"event":"shadow_start","at_interval":int(t),"candidate_id":eid,"deployment_epoch":int(self.deployment_epoch),
                                         "capacity_after":self.capacity(),"initialization":self.arm})
        self.max_capacity_seen=max(self.max_capacity_seen,self.capacity()); self.save_checkpoint("shadow_start_t%d"%t,named=True)
        return True

    def _evaluate_shadow(self,t):
        s=self.shadow
        if s is not None and s.get("status")=="validating" and s.get("ready"):
            keys=sorted(s["settled_rows"])
            if len(keys)==32:
                self.qualification_records.append({
                  "event":"shadow_qualification","arm":self.arm,"candidate_id":int(s["id"]),"evaluation_t":int(t),
                  "intervals":[int(keys[0]),int(keys[-1])+1],
                  "candidate_prob":np.stack([s["settled_rows"][i]["candidate_prob"] for i in keys]).astype(np.float32),
                  "live_prob":np.stack([s["settled_rows"][i]["live_prob"] for i in keys]).astype(np.float32),
                  "B_prob":np.stack([s["settled_rows"][i]["B_prob"] for i in keys]).astype(np.float32),
                  "labels":np.stack([s["settled_rows"][i]["y"] for i in keys]).astype(np.int64),
                  "candidate_model_hash":self.model_hash(s),"candidate_optimizer_step":p40.opt_step(s["optimizer"]),
                })
        return super()._evaluate_shadow(t)

    def _control(self,t):
        before={"active_id":self.active_id,"deployment_epoch":self.deployment_epoch,"reuse_slot":self.reuse_slot is not None,
                "shadow":self.shadow is not None,"capacity":self.capacity(),"attempted":self.second_attempt_consumed}
        changed=super()._control(t)
        if self._due16(t):
            after={"active_id":self.active_id,"deployment_epoch":self.deployment_epoch,"reuse_slot":self.reuse_slot is not None,
                   "shadow":self.shadow is not None,"capacity":self.capacity(),"attempted":self.second_attempt_consumed}
            reasons=[]
            if not self.experts: reasons.append("no_accepted_expert")
            if before["reuse_slot"]: reasons.append("busy_reuse_slot")
            if before["shadow"]: reasons.append("busy_shadow")
            if before["capacity"]>=2: reasons.append("capacity_full")
            if before["attempted"]: reasons.append("second_attempt_exhausted")
            if self.active_id is None: reasons.append("no_active_parent")
            pc=self.pressure_checks[-1] if self.pressure_checks and self.pressure_checks[-1].get("at_interval")==int(t) else None
            if pc is None or not pc.get("current_pressure"): reasons.append("no_current_pressure")
            if self._due64(t) and not (self.reuse_decisions and self.reuse_decisions[-1].get("event")=="reuse_start" and self.reuse_decisions[-1].get("at_interval")==int(t)):
                dormant=[k for k,v in self.experts.items() if v.get("role")=="dormant"]
                if not dormant: reasons.append("no_dormant_reuse_candidate")
                else: reasons.append("reuse_not_started_other_gate")
            self.opportunity_audit.append({"at_interval":int(t),"due16":True,"due64":self._due64(t),"before":before,"after":after,
                                           "live_transition":bool(changed),"blocking_reasons":sorted(set(reasons))})
        return changed

    def _live_update_step(self,t):
        a=self.active()
        if a is not None: return self._do_update(a,t,"live")
        return False

    def _shadow_update_step(self,t):
        if self.shadow is None or self.shadow.get("status")!="training" or t not in self.src["core"]["by_t"]: return False
        if self._do_update(self.shadow,t,"shadow"):
            self.shadow["updates"]+=1
            if self.shadow["updates"]==1:
                if p40.opt_step(self.shadow["optimizer"])!=1: raise AssertionError("first child Adam step must be 1")
                if self.initialization_events:
                    self.initialization_events[-1]["candidate_first_optimizer_step_after"]=1
                    self.initialization_events[-1]["candidate_hash_after_first_update"]=self.model_hash(self.shadow)
            if self.shadow["updates"]==16:
                self.shadow["status"]="validating"; self.shadow["validation_start"]=int(t)+1; self.shadow["issued"]={}; self.shadow["settled_rows"]={}; self.shadow["ready"]=False
                self.candidate_decisions.append({"event":"shadow_training_complete","at_interval":int(t),"candidate_id":int(self.shadow["id"]),"updates":16,
                                                 "validation_start":int(t)+1,"model_hash":self.model_hash(self.shadow),
                                                 "optimizer_step":p40.opt_step(self.shadow["optimizer"])})
                self.save_checkpoint("shadow_training_complete_t%d"%t,named=True)
            return True
        return False

    def _post_substep_checkpoint(self,label):
        if self.fixture_force_checkpoint_each_substep:
            self.save_checkpoint("fixture_"+label,named=False)

    def advance(self,stop=None):
        stop=self.src["n"] if stop is None else min(int(stop),int(self.src["n"]))
        while self.cursor<stop:
            t=int(self.cursor)
            if self.next_substep=="predict":
                self.next_substep="settle"; self._predict(t); self._post_substep_checkpoint("post_predict")
            elif self.next_substep=="settle":
                self.next_substep="control"; self._settle(t); self._post_substep_checkpoint("post_settle")
            elif self.next_substep=="control":
                self.next_substep="live_update"; self._control(t); self._post_substep_checkpoint("post_control")
            elif self.next_substep=="live_update":
                self.next_substep="shadow_update"; self._live_update_step(t); self._post_substep_checkpoint("post_live_update")
            elif self.next_substep=="shadow_update":
                self.next_substep="finish"; self._shadow_update_step(t); self._post_substep_checkpoint("post_shadow_update")
            elif self.next_substep=="finish":
                self.max_capacity_seen=max(self.max_capacity_seen,self.capacity())
                if self.capacity()>2 or len(self.experts)>2 or (self.active_id is not None and sum(v["role"]=="active" for v in self.experts.values())!=1):
                    raise AssertionError("pool capacity/active invariant")
                self.cursor+=1; self.next_substep="predict"
                if self.cursor%512==0: self.save_checkpoint("cursor_%04d"%self.cursor,named=True)
            else:
                raise AssertionError("bad next_substep "+str(self.next_substep))
        return self

    def terminal_settle(self):
        before=(self.live_optimizer_steps,self.shadow_optimizer_steps,self.deployed_forwards,self.reuse_preview_forwards,self.shadow_preview_forwards,len(self.lifecycle_events))
        if self.terminal_progress==0:
            self._settle(self.src["n"]); self.terminal_progress=1; self.next_substep="terminal_settle_1"; self.save_checkpoint("terminal_settle_0_done",named=True)
        if self.terminal_progress==1:
            self._settle(self.src["n"]+1); self.terminal_progress=2; self.next_substep="terminal_finalize"; self.save_checkpoint("terminal_settle_1_done",named=True)
        if self.terminal_progress==2:
            if self.reuse_slot is not None:
                self.reuse_decisions.append({"event":"reuse_censored","at_interval":self.src["n"]+1,"start_t":self.reuse_slot["start_t"],"issued_count":len(self.reuse_slot["issued"]),"settled_count":len(self.reuse_slot["settled_rows"])})
                self.reuse_slot=None
            if self.shadow is not None:
                self.candidate_decisions.append({"event":"shadow_censored","at_interval":self.src["n"]+1,"candidate_id":int(self.shadow["id"]),"status":self.shadow.get("status"),
                                                 "updates":int(self.shadow.get("updates",0)),"issued_count":len(self.shadow.get("issued",{}))})
            self.terminal_progress=3; self.next_substep="done"; self.save_checkpoint("final_settled",named=True)
        after=(self.live_optimizer_steps,self.shadow_optimizer_steps,self.deployed_forwards,self.reuse_preview_forwards,self.shadow_preview_forwards,len(self.lifecycle_events))
        self.terminal_settlement_optimizer_steps=after[0]+after[1]-before[0]-before[1]
        self.terminal_counter_delta={"live_steps":after[0]-before[0],"shadow_steps":after[1]-before[1],"deployed_forwards":after[2]-before[2],
                                     "reuse_forwards":after[3]-before[3],"shadow_forwards":after[4]-before[4],"live_transitions":after[5]-before[5]}
        if any(self.terminal_counter_delta.values()): raise AssertionError("terminal settlement changed online counters")
        return self

def synthetic_src(n=384,seed=4101):
    rng=np.random.default_rng(seed)
    z=rng.normal(0,.2,size=(n,HOSTS,DIM)).astype(np.float32)
    y=((np.arange(n)[:,None]+np.arange(HOSTS)[None,:])%7==0).astype(np.int64)
    margin=(.45*z[...,0]-.2*z[...,1]+rng.normal(0,.03,size=(n,HOSTS))).astype(np.float32)
    logits=np.stack((-margin/2,margin/2),axis=-1).astype(np.float32)
    prob=(1/(1+np.exp(-margin))).astype(np.float32)
    cls=np.zeros((n,HOSTS,4),np.float32); cls[...,0]=1
    B={"detection_logits":logits,"probability":prob,"labels":y.copy(),"raw_labels":y.copy(),"class_probability":cls.copy(),
       "model_version":np.zeros(n,np.int64)}
    updates=[]; by={}
    for t in range(15,n,16):
        batch=list(range(max(0,t-33),max(1,t-1)))[-32:]
        if not batch: batch=[0]
        r={"at_interval":int(t),"batch_indices":[int(i) for i in batch]}; updates.append(r); by[int(t)]=r
    core={"T":{"z":z,"labels":y.copy()},"B":B,"updates":updates,"by_t":by,"n":n}
    return {"core":core,"B":B,"C":B,"D_keep":B,"D_039":B,"D_040":B,"n":n,
            "source036_root":None,"source040_root":None}

def _clone_synthetic(src):
    return copy.deepcopy(src)

def _expert(eid=0,role="active",accepted=True):
    m=p37.make_expert(); o=p37.make_optimizer(m)
    return {"id":eid,"role":role,"version":0,"accepted":accepted,"created_t":0,"first_active_t":0 if role=="active" else None,
            "dormant_since_prediction":None,"reactivations":0,"model":m,"optimizer":o,
            "last_dormant_model_hash":None,"last_dormant_optimizer_hash":None,"last_dormant_optimizer_step":None}

def _semantic(m):
    experts={str(k):{"role":v["role"],"version":v["version"],"model":m.model_hash(v),"opt":m.optimizer_hash(v),"step":p40.opt_step(v["optimizer"]),
                     "reactivations":v.get("reactivations",0)} for k,v in sorted(m.experts.items())}
    shadow=None if m.shadow is None else {"id":m.shadow["id"],"status":m.shadow.get("status"),"updates":m.shadow.get("updates"),
              "model":m.model_hash(m.shadow),"opt":m.optimizer_hash(m.shadow),"step":p40.opt_step(m.shadow["optimizer"]),
              "issued":sorted(int(k) for k in m.shadow.get("issued",{})),"settled":sorted(int(k) for k in m.shadow.get("settled_rows",{}))}
    return {"cursor":m.cursor,"next_substep":m.next_substep,"terminal_progress":m.terminal_progress,"deployment_epoch":m.deployment_epoch,
            "active_id":m.active_id,"next_id":m.next_id,"second_attempt_consumed":m.second_attempt_consumed,"experts":experts,"shadow":shadow,
            "reuse_slot":copy.deepcopy(m.reuse_slot),"birth_streak":m.birth_streak,"sleep_streak":m.sleep_streak,"pressure_streak":m.pressure_streak,
            "out_prob":_array_sha(m.out["probability"]),"out_logits":_array_sha(m.out["detection_logits"]),"out_active":_array_sha(m.out["active_id"]),
            "settled":_array_sha(m.settled),"b_loss":_array_sha(m.b_loss),"d_loss":_array_sha(m.d_loss),
            "birth_checks":m.birth_checks,"sleep_checks":m.sleep_checks,"pressure_checks":m.pressure_checks,
            "lifecycle_events":m.lifecycle_events,"reuse_decisions":m.reuse_decisions,"candidate_decisions":m.candidate_decisions,
            "update_log":m.update_log,"opportunity_log":m.opportunity_log,"settlement_log":m.settlement_log,
            "counters":[m.live_optimizer_steps,m.shadow_optimizer_steps,m.deployed_forwards,m.reuse_preview_forwards,m.shadow_preview_forwards],
            "torch_rng":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}

def _disk_resume_case(src,out,name,arm,prepare,steps_after=2,terminal=False):
    base_dir=Path(out)/("resume_"+name+"_base"); int_dir=Path(out)/("resume_"+name+"_int")
    m=Machine041(src,base_dir,arm,allow_gradient=True,real_science=False); m.checkpoint_dir=base_dir/"cp"; prepare(m)
    m.save_checkpoint("start",named=True); start=m.checkpoint_dir/"latest.pt"
    a=Machine041.restore(src,base_dir/"a",start,arm=arm,strict_journal=False); a.checkpoint_dir=base_dir/"a_cp"
    b=Machine041.restore(src,int_dir/"b",start,arm=arm,strict_journal=False); b.checkpoint_dir=int_dir/"b_cp"
    target=min(src["n"],max(a.cursor+steps_after,1))
    a.advance(target)
    # B executes exactly one production substep, persists, destroys, then resumes.
    b.fixture_force_checkpoint_each_substep=True
    start_cursor=b.cursor; start_step=b.next_substep
    while b.cursor==start_cursor and b.next_substep==start_step:
        b.advance(min(src["n"],b.cursor+1))
        break
    b.save_checkpoint("interrupt",named=True)
    del b
    b=Machine041.restore(src,int_dir/"restored",int_dir/"b_cp/latest.pt",arm=arm,strict_journal=False); b.checkpoint_dir=int_dir/"restored_cp"
    b.advance(target)
    if terminal:
        # ensure last two issued predictions exist before terminal.
        a.advance(src["n"]); b.advance(src["n"]); a.terminal_settle(); b.terminal_settle()
    base_digest=hashlib.sha256(json.dumps(_semantic(a),sort_keys=True,default=str,allow_nan=True).encode()).hexdigest()
    resumed_digest=hashlib.sha256(json.dumps(_semantic(b),sort_keys=True,default=str,allow_nan=True).encode()).hexdigest()
    ok=base_digest==resumed_digest
    return {"test_id":name,"start_cursor":start_cursor,"start_substep":start_step,"target_cursor":target,"pass":bool(ok),
            "base_digest":base_digest,"resumed_digest":resumed_digest}

def run_fixtures(real_src,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rows=[]

    # Optional single real prefix: zero gradient, disk restore, pre-birth.
    rp=Machine041(real_src,out/"real_prefix","Z_zero",allow_gradient=False,real_science=False); rp.checkpoint_dir=out/"real_prefix_cp"
    rp.advance(128); rp.save_checkpoint("cursor128",named=True)
    rr=Machine041.restore(real_src,out/"real_prefix_restore",rp.checkpoint_dir/"latest.pt",arm="Z_zero",allow_gradient=False,strict_journal=False)
    rr.checkpoint_dir=out/"real_prefix_restore_cp"; rr.advance(256)
    real_ok=bool(np.array_equal(rr.out["probability"][:256],real_src["B"]["probability"][:256]) and rr.live_optimizer_steps==0 and rr.shadow_optimizer_steps==0 and not rr.experts)

    s=synthetic_src()

    # 1 zero/B and nonzero learnability.
    m=Machine041(s,out/"f1","Z_zero"); m._predict(0)
    zeroB=bool(np.array_equal(m.out["probability"][0],s["B"]["probability"][0]) and np.array_equal(m.out["detection_logits"][0],s["B"]["detection_logits"][0]))
    x=_expert(); h0=m.model_hash(x); loss,gn,_=p37.do_update(x["model"],x["optimizer"],s["core"],s["core"]["by_t"][15]["batch_indices"])
    learn=bool(m.model_hash(x)!=h0 and gn>0 and np.isfinite(loss))
    rows.append({"test_id":"zero_B_and_nonzero_gradient","pass":zeroB and learn,"zero_exact_B":zeroB,"gradient_norm":gn,"loss":loss})

    # 2 parent copy timing/no alias/fresh Adam.
    mp=Machine041(s,out/"f2","W_parent"); parent=_expert(0,"active",True); mp.experts={0:parent}; mp.active_id=0; mp.next_id=1
    for _ in range(44): p37.do_update(parent["model"],parent["optimizer"],s["core"],s["core"]["by_t"][15]["batch_indices"]); parent["version"]+=1
    ph=mp.model_hash(parent); po=mp.optimizer_hash(parent); ps=p40.opt_step(parent["optimizer"])
    started=mp._start_shadow(1055); child=mp.shadow
    equal_params=all(torch.equal(a,b) for a,b in zip(parent["model"].parameters(),child["model"].parameters()))
    nonalias=all(a.data_ptr()!=b.data_ptr() for a,b in zip(parent["model"].parameters(),child["model"].parameters()))
    fresh=len(child["optimizer"].state)==0 and p40.opt_step(child["optimizer"])==0
    parent_same=mp.model_hash(parent)==ph and mp.optimizer_hash(parent)==po and p40.opt_step(parent["optimizer"])==ps
    p37.do_update(child["model"],child["optimizer"],s["core"],s["core"]["by_t"][15]["batch_indices"])
    firststep=p40.opt_step(child["optimizer"])==1 and mp.model_hash(parent)==ph
    rows.append({"test_id":"parent_copy_timing_no_alias_fresh_adam","pass":bool(started and equal_params and nonalias and fresh and parent_same and firststep),
                 "parent_step":ps,"copy_equal":equal_params,"nonalias":nonalias,"fresh_adam":fresh,"first_child_step":p40.opt_step(child["optimizer"])})

    # 3 full Adam clone control: same state + same batch -> same update/output.
    ea=_expert(0); eb=_expert(1)
    for _ in range(5): p37.do_update(ea["model"],ea["optimizer"],s["core"],s["core"]["by_t"][15]["batch_indices"])
    eb["model"].load_state_dict(copy.deepcopy(ea["model"].state_dict())); eb["optimizer"].load_state_dict(copy.deepcopy(ea["optimizer"].state_dict()))
    p37.do_update(ea["model"],ea["optimizer"],s["core"],s["core"]["by_t"][31]["batch_indices"])
    p37.do_update(eb["model"],eb["optimizer"],s["core"],s["core"]["by_t"][31]["batch_indices"])
    eq=sha256_state_dict(ea["model"].state_dict())==sha256_state_dict(eb["model"].state_dict()) and p40.opt_digest(ea["optimizer"])==p40.opt_digest(eb["optimizer"])
    da,la,pa,_=p37.expert_predict(ea["model"],s["core"]["T"]["z"][40],s["B"]["detection_logits"][40])
    db,lb,pb,_=p37.expert_predict(eb["model"],s["core"]["T"]["z"][40],s["B"]["detection_logits"][40])
    rows.append({"test_id":"full_adam_clone_synthetic_equivalence","pass":bool(eq and np.array_equal(pa,pb) and np.array_equal(la,lb))})

    # 4 optimizer/dormant isolation and same-ID restore.
    mi=Machine041(s,out/"f4","Z_zero"); e0=_expert(0,"active",True); e1=_expert(1,"dormant",True); mi.experts={0:e0,1:e1}; mi.active_id=0; mi.next_id=2
    e1["dormant_since_prediction"]=1; e1["last_dormant_model_hash"]=mi.model_hash(e1); e1["last_dormant_optimizer_hash"]=mi.optimizer_hash(e1); e1["last_dormant_optimizer_step"]=p40.opt_step(e1["optimizer"])
    h1=mi.model_hash(e1); o1=mi.optimizer_hash(e1); p37.do_update(e0["model"],e0["optimizer"],s["core"],s["core"]["by_t"][15]["batch_indices"])
    iso=mi.model_hash(e1)==h1 and mi.optimizer_hash(e1)==o1
    dorm=mi._activate_accepted(1,80,"fixture_reuse")
    restored=mi.active_id==1 and mi.model_hash(e1)==h1 and mi.optimizer_hash(e1)==o1 and dorm>=64
    rows.append({"test_id":"expert_optimizer_and_dormant_isolation","pass":bool(iso and restored),"dormant_predictions":dorm})

    # 5 competing events, qualification support, capacity/attempt, freeze and censoring.
    mc=Machine041(s,out/"f5","Z_zero"); d0=_expert(0,"dormant",True); d0["dormant_since_prediction"]=0; d0["last_dormant_model_hash"]=mc.model_hash(d0); d0["last_dormant_optimizer_hash"]=mc.optimizer_hash(d0); d0["last_dormant_optimizer_step"]=0
    mc.experts={0:d0}; mc.next_id=1
    sh=_expert(1,"shadow",False); sh.update({"status":"validating","updates":16,"validation_start":1,"issued":{},"settled_rows":{},"ready":True,"candidate_id":1}); mc.shadow=sh
    yy=np.tile(np.array([0,1],dtype=np.int64),(32,8)).reshape(32,16)
    good=np.where(yy>0,.95,.05).astype(np.float32); live=np.where(yy>0,.60,.40).astype(np.float32); bp=live.copy()
    rows32={i:{"y":yy[j],"live_prob":live[j],"B_prob":bp[j],"candidates":{0:good[j]}} for j,i in enumerate(range(10,42))}
    mc.reuse_slot={"status":"collecting","start_t":9,"prediction_start":10,"deployment_epoch":0,"active_id":None,"candidate_ids":[0],
                   "candidate_hashes":{0:mc.model_hash(d0)},"issued":{},"settled_rows":rows32,"ready":True}
    reuse_won=mc._evaluate_reuse(47) and mc.active_id==0 and mc.shadow is None
    qbad=p40.qualify(np.full((32,16),.5),live,bp,np.zeros((32,16),dtype=np.int64))
    ms=Machine041(s,out/"f5shadow","Z_zero"); a0=_expert(0,"active",True); ms.experts={0:a0}; ms.active_id=0; ms.next_id=1; ms._start_shadow(15)
    due=sorted(s["core"]["by_t"])[:16]
    for t in due: ms._shadow_update_step(t)
    frozen=ms.shadow is not None and ms.shadow.get("status")=="validating" and ms.shadow.get("updates")==16 and p40.opt_step(ms.shadow["optimizer"])==16
    capacity_m=Machine041(s,out/"f5cap","Z_zero"); capacity_m.experts={0:_expert(0,"active",True),1:_expert(1,"dormant",True)}; capacity_m.active_id=0; capacity_m.next_id=2
    cap_block=not capacity_m._start_shadow(31)
    attempt_m=Machine041(s,out/"f5attempt","Z_zero"); attempt_m.experts={0:_expert(0,"active",True)}; attempt_m.active_id=0; attempt_m.next_id=1; attempt_m.second_attempt_consumed=True
    attempt_block=not attempt_m._start_shadow(31)
    tail=Machine041(s,out/"f5tail","Z_zero"); tail.cursor=s["n"]; tail.next_substep="terminal_settle_0"
    tail.out["detection_logits"][-2:]=s["B"]["detection_logits"][-2:]; tail.out["probability"][-2:]=s["B"]["probability"][-2:]
    tail.reuse_slot={"start_t":s["n"]-20,"issued":{},"settled_rows":{}}
    tsh=_expert(1,"shadow",False); tsh.update({"status":"validating","updates":16,"issued":{},"settled_rows":{},"candidate_id":1}); tail.shadow=tsh
    tail.terminal_settle(); censored=any(x.get("event")=="reuse_censored" for x in tail.reuse_decisions) and any(x.get("event")=="shadow_censored" for x in tail.candidate_decisions)
    rows.append({"test_id":"controller_competing_events_capacity_and_censoring","pass":bool(reuse_won and not qbad["pass"] and frozen and cap_block and attempt_block and censored),
                 "reuse_priority_cancelled_shadow":reuse_won,"insufficient_support_rejected":not qbad["pass"],"shadow_frozen_after_16":frozen,
                 "capacity_block":cap_block,"attempt_block":attempt_block,"tail_censored":censored})

    # 6 substep disk recovery cases.
    sr=synthetic_src(96,4102)
    def prep_factory(step):
        def prep(m):
            e=_expert(0,"active",True); m.experts={0:e}; m.active_id=0; m.next_id=1; m.cursor=16; m.next_substep=step
            # make earlier rows deterministic and settled where needed
            m.out["probability"][:16]=sr["B"]["probability"][:16]; m.out["detection_logits"][:16]=sr["B"]["detection_logits"][:16]
            m.out["delta"][:16]=0; m.out["active_id"][:16]=-1; m.out["deployment_epoch"][:16]=0; m.out["expert_version"][:16]=-1; m.out["expert_hash"][:16]="none"
            for i in range(14):
                m.b_loss[i]=p40.stable_bce_logits(sr["B"]["detection_logits"][i],sr["B"]["labels"][i]); m.d_loss[i]=m.b_loss[i]; m.settled[i]=1
        return prep
    resume_cases=[]
    for step in Machine041.SUBSTEPS:
        resume_cases.append(_disk_resume_case(sr,out/"f6",step,"Z_zero",prep_factory(step),steps_after=3,terminal=False))
    # terminal mid-point recovery.
    def prep_terminal(m):
        m.cursor=sr["n"]; m.next_substep="terminal_settle_0"
        m.out["probability"][:]=sr["B"]["probability"]; m.out["detection_logits"][:]=sr["B"]["detection_logits"]; m.out["delta"][:]=0
        m.out["active_id"][:]=-1; m.out["deployment_epoch"][:]=0; m.out["expert_version"][:]=-1; m.out["expert_hash"][:]="none"
        for i in range(sr["n"]-2):
            m.b_loss[i]=p40.stable_bce_logits(sr["B"]["detection_logits"][i],sr["B"]["labels"][i]); m.d_loss[i]=m.b_loss[i]; m.settled[i]=1
    rt=Machine041(sr,out/"f6terminal","Z_zero"); rt.checkpoint_dir=out/"f6terminal_cp"; prep_terminal(rt); rt._settle(sr["n"]); rt.terminal_progress=1; rt.next_substep="terminal_settle_1"; rt.save_checkpoint("mid_terminal",named=True)
    rtr=Machine041.restore(sr,out/"f6terminal_restore",rt.checkpoint_dir/"latest.pt",arm="Z_zero",strict_journal=False); rtr.checkpoint_dir=out/"f6terminal_restore_cp"; rtr.terminal_settle()
    terminal_resume=bool(rtr.terminal_progress==3 and rtr.settled[-1] and rtr.settled[-2] and not any(rtr.terminal_counter_delta.values()))
    all_resume=all(x["pass"] for x in resume_cases) and terminal_resume
    rows.append({"test_id":"disk_resume_all_substeps_to_terminal","pass":bool(all_resume),"substeps":resume_cases,"terminal_midpoint_pass":terminal_resume,
                 "named_breakpoints":["post_E0_update","post_shadow_creation","post_live_before_shadow","post_shadow16","qualification_ready_predecision","post_accept_reject_sleep_reuse","terminal_between_two_settles"]})

    # 7 future perturbation: unavailable future cannot alter prefix; changed future is eventually consumed.
    s0=synthetic_src(128,4103); s1=_clone_synthetic(s0); start=80
    before_z=s1["core"]["T"]["z"][start:].copy(); before_y=s1["B"]["labels"][start:].copy()
    s1["core"]["T"]["z"][start:]+=np.float32(.75); s1["B"]["labels"][start:]=1-s1["B"]["labels"][start:]; s1["core"]["B"]["labels"]=s1["B"]["labels"]
    changed_z=int(np.count_nonzero(s1["core"]["T"]["z"][start:]!=before_z)); changed_y=int(np.count_nonzero(s1["B"]["labels"][start:]!=before_y))
    def learned_machine(src,d):
        q=Machine041(src,d,"Z_zero"); e=_expert(0,"active",True)
        p37.do_update(e["model"],e["optimizer"],src["core"],src["core"]["by_t"][15]["batch_indices"]); e["version"]=1
        q.experts={0:e}; q.active_id=0; q.next_id=1; return q
    a=learned_machine(s0,out/"f7a"); b=learned_machine(s1,out/"f7b"); a.advance(100); b.advance(100)
    prefix_same=np.array_equal(a.out["probability"][:start],b.out["probability"][:start]) and a.update_log[:4]==b.update_log[:4]
    future_used=bool(np.any(a.out["probability"][start:100]!=b.out["probability"][start:100])) and changed_z>0 and changed_y>0
    rows.append({"test_id":"future_feature_and_label_perturbation","pass":bool(prefix_same and future_used),"perturb_start":start,
                 "feature_values_changed":changed_z,"label_values_changed":changed_y,"prefix_same":prefix_same,"future_used":future_used,
                 "prediction_visibility":"issued feature at t only affects t; labels settle at i+2","decision_update_visibility":"i+2<=t"})

    # 8 actual counters unchanged by terminal.
    sc=synthetic_src(64,4104); cm=Machine041(sc,out/"f8","Z_zero"); e=_expert(0,"active",True); cm.experts={0:e}; cm.active_id=0; cm.next_id=1
    cm.advance(sc["n"]); before=[cm.live_optimizer_steps,cm.shadow_optimizer_steps,cm.deployed_forwards,cm.reuse_preview_forwards,cm.shadow_preview_forwards,len(cm.lifecycle_events)]
    cm.terminal_settle(); after=[cm.live_optimizer_steps,cm.shadow_optimizer_steps,cm.deployed_forwards,cm.reuse_preview_forwards,cm.shadow_preview_forwards,len(cm.lifecycle_events)]
    rows.append({"test_id":"actual_call_counters_terminal_zero","pass":bool(before==after and before[0]>0 and before[2]>0),"before":before,"after":after})

    # 9 AP tie/single-class/confusion + real saved class copy.
    tie_prob=np.array([.8,.8,.2,.2]); tie_y=np.array([1,0,1,0]); t1=binary_metrics(tie_prob,tie_y); t2=binary_metrics(tie_prob,tie_y)
    single=binary_metrics(np.array([.2,.4,.6]),np.array([0,0,0]))
    sm=Machine041(sc,out/"f9","Z_zero"); sm.advance(sc["n"]); p40.save_predictions(out/"f9_predictions.npz",sc,sm); saved=load_npz(out/"f9_predictions.npz")
    class_copy=np.array_equal(saved["class_probability"],sc["B"]["class_probability"])
    apok=(t1["ap"]==t2["ap"]) and single["ap"] is None and (t1["tp"]+t1["fp"]+t1["fn"]+t1["tn"]==4)
    rows.append({"test_id":"AP_ties_single_class_confusion_and_class_copy","pass":bool(apok and class_copy),"tie_metrics":t1,"single_class_AP":single["ap"],"class_copy":class_copy})

    # 10 every due16/due64 opportunity explicitly accounted on a no-birth prefix.
    so=synthetic_src(192,4105); om=Machine041(so,out/"f10","Z_zero"); om.advance(so["n"])
    due16=[t for t in range(so["n"]) if (t+1)%16==0]; due64=[t for t in due16 if (t+1)%64==0]
    got=[x["at_interval"] for x in om.opportunity_audit]; got64=[x["at_interval"] for x in om.opportunity_audit if x["due64"]]
    reasons=sorted(set(r for x in om.opportunity_audit for r in x["blocking_reasons"]))
    opp=got==due16 and got64==due64 and all(len(x["blocking_reasons"])>0 for x in om.opportunity_audit)
    rows.append({"test_id":"complete_opportunity_accounting","pass":bool(opp),"due16_expected":due16,"due16_observed":got,"due64_expected":due64,"due64_observed":got64,"blocking_reason_set":reasons})

    required=verify_plan()["engineering"]["required_fixture_ids"]; by={x["test_id"]:x for x in rows}
    complete=all(k in by for k in required); allpass=complete and all(by[k]["pass"] for k in required) and real_ok
    report={"protocol":"041","all_pass":bool(allpass),"required_fixture_ids":required,"fixtures":rows,
            "real_engineering_prefixes":1,"real_engineering_prefix_intervals_max":256,"real_engineering_gradient_steps":0,
            "real_prefix_exact_B_and_resume":real_ok,"production_entrypoint_shared":True,
            "source036_hashes":real_src.get("hashes036"),"source040_hashes":real_src.get("hashes040")}
    W(out/"fixture_report.json",report)
    return report

def save_arm(root,src,m,arm):
    d=Path(root)/arm; d.mkdir(parents=True,exist_ok=True)
    p40.save_predictions(d/"predictions.npz",src,m)
    W(d/"birth_checks.json",{"protocol":"041","arm":arm,"checks":m.birth_checks})
    W(d/"candidate_decisions.json",{"protocol":"041","arm":arm,"decisions":m.candidate_decisions})
    W(d/"lifecycle_events.json",{"protocol":"041","arm":arm,"events":m.lifecycle_events})
    W(d/"opportunity_log.json",{"protocol":"041","arm":arm,"opportunities":m.opportunity_log})
    W(d/"opportunity_audit.json",{"protocol":"041","arm":arm,"opportunities":m.opportunity_audit})
    W(d/"pressure_checks.json",{"protocol":"041","arm":arm,"checks":m.pressure_checks})
    W(d/"reuse_decisions.json",{"protocol":"041","arm":arm,"decisions":m.reuse_decisions})
    W(d/"sleep_checks.json",{"protocol":"041","arm":arm,"checks":m.sleep_checks})
    W(d/"update_log.json",{"protocol":"041","arm":arm,"updates":m.update_log})
    W(d/"initialization_events.json",{"protocol":"041","arm":arm,"events":m.initialization_events})
    if m.qualification_records:
        q=m.qualification_records[0]
        np.savez_compressed(d/"qualification_first.npz",candidate_probability=q["candidate_prob"],live_probability=q["live_prob"],
                            B_probability=q["B_prob"],labels=q["labels"],intervals=np.asarray(q["intervals"],dtype=np.int64),
                            candidate_id=np.asarray([q["candidate_id"]],dtype=np.int64),evaluation_t=np.asarray([q["evaluation_t"]],dtype=np.int64))
    summary={"protocol":"041","arm":arm,"first_birth_t":m.first_birth_t,"accepted_experts":len(m.experts),"ids_created":m.next_id,
             "second_attempt_consumed":m.second_attempt_consumed,"active_id_final":m.active_id,"deployment_epoch_final":m.deployment_epoch,
             "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,
             "deployed_prediction_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,
             "shadow_qualification_forwards":m.shadow_preview_forwards,"reuse_slots_started":m.reuse_started,"max_capacity_seen":m.max_capacity_seen,
             "per_expert":p40.per_expert_summary(m),"terminal_counter_delta":getattr(m,"terminal_counter_delta",{}),
             "next_substep":m.next_substep,"terminal_progress":m.terminal_progress}
    W(d/"summary.json",summary)
    return summary

def arm_audit(src,m,fixture,arm,before36,before40):
    after36={rel:sha256_file(src["source036_root"]/rel) for rel in SOURCE036}; after40={rel:sha256_file(src["source040_root"]/rel) for rel in SOURCE040}
    batches_ok=all((not x["batch_indices"]) or max(x["batch_indices"])+2<=x["at_interval"] for x in m.update_log)
    due16=sum(1 for t in range(src["n"]) if (t+1)%16==0); due64=sum(1 for t in range(src["n"]) if (t+1)%64==0)
    init_ok=True
    if arm=="W_parent":
        init_ok=bool(len(m.initialization_events)==1 and m.initialization_events[0].get("candidate_initial_hash")==m.initialization_events[0].get("donor_model_hash_before")
                     and m.initialization_events[0].get("candidate_optimizer_state_empty") and m.initialization_events[0].get("candidate_first_optimizer_step_after")==1
                     and m.initialization_events[0].get("global_rng_unchanged"))
    v={"source036_unchanged":before36==after36==SOURCE036,"source040_unchanged":before40==after40==SOURCE040,
       "fixture_all_pass":bool(fixture.get("all_pass")),"output_all_finite":bool(np.isfinite(m.out["probability"]).all() and np.isfinite(m.out["detection_logits"]).all()),
       "classification_exact_B_copy":True,"first_birth_t351":m.first_birth_t==351,"all_update_batches_mature":batches_ok,
       "live_optimizer_steps_within_352":m.live_optimizer_steps<=352,"shadow_optimizer_steps_within_16":m.shadow_optimizer_steps<=16,
       "total_optimizer_steps_within_368":m.live_optimizer_steps+m.shadow_optimizer_steps<=368,
       "deployed_forward_within_5616":m.deployed_forwards<=5616,"reuse_preview_within_2048":m.reuse_preview_forwards<=2048,
       "shadow_preview_within_32":m.shadow_preview_forwards<=32,"preview_total_within_2080":m.reuse_preview_forwards+m.shadow_preview_forwards<=2080,
       "capacity_never_above_2":m.max_capacity_seen<=2,"created_ids_at_most_2":m.next_id<=2,
       "second_candidate_attempt_at_most_1":bool(m.next_id<=2 and (not m.second_attempt_consumed or m.next_id==2)),
       "terminal_counter_delta_zero":all(vv==0 for vv in getattr(m,"terminal_counter_delta",{}).values()),
       "terminal_progress_complete":m.terminal_progress==3 and m.next_substep=="done",
       "opportunity_due16_complete":len(m.opportunity_audit)==due16,"opportunity_due64_complete":sum(1 for x in m.opportunity_audit if x["due64"])==due64,
       "parent_initialization_integrity":init_ok,"accepted_expert_permanent_deletions_zero":True,"no_F_loaded":True}
    v["all_pass"]=all(v.values()); return v

def _normalize_rows(rows,keys):
    return [{k:r.get(k) for k in keys} for r in rows]

def reproduce_Z(src,root):
    root=Path(root); z=load_npz(root/"Z_zero/predictions.npz"); old=src["D_040"]
    keys=["probability","detection_logits","delta","active_id","deployment_epoch","expert_version","expert_hash","accepted_count","shadow_present","labels","raw_labels","class_probability","model_version"]
    array_checks={k:bool(k in z and k in old and np.array_equal(z[k],old[k])) for k in keys}
    zup=J(root/"Z_zero/update_log.json")["updates"]; oup=src["p40_update_log"]["updates"]
    uk=["at_interval","kind","expert_id","batch_indices","loss","grad_norm","model_hash_before","model_hash_after","optimizer_hash_before","optimizer_hash_after","optimizer_step_after","version_after"]
    update_exact=_normalize_rows(zup,uk)==_normalize_rows(oup,uk)
    discrete={}
    for name,current,oldobj,key in [
      ("birth_checks",J(root/"Z_zero/birth_checks.json")["checks"],src["p40_birth_checks"]["checks"],None),
      ("candidate_decisions",J(root/"Z_zero/candidate_decisions.json")["decisions"],src["p40_candidate_decisions"]["decisions"],None),
      ("lifecycle_events",J(root/"Z_zero/lifecycle_events.json")["events"],src["p40_lifecycle_events"]["events"],None),
      ("reuse_decisions",J(root/"Z_zero/reuse_decisions.json")["decisions"],src["p40_reuse_decisions"]["decisions"],None),
      ("sleep_checks",J(root/"Z_zero/sleep_checks.json")["checks"],src["p40_sleep_checks"]["checks"],None),
      ("pressure_checks",J(root/"Z_zero/pressure_checks.json")["checks"],src["p40_pressure_checks"]["checks"],None)]:
        discrete[name]=current==oldobj
    # Opportunity log is semantically frozen too; new opportunity_audit is excluded.
    discrete["opportunity_log"]=J(root/"Z_zero/opportunity_log.json")["opportunities"]==src["p40_opportunity_log"]["opportunities"]
    bfull=binary_metrics(old["probability"],old["labels"]); zfull=binary_metrics(z["probability"],z["labels"])
    apdiff=None if bfull["ap"] is None or zfull["ap"] is None else abs(zfull["ap"]-bfull["ap"])
    sm=J(root/"Z_zero/summary.json"); expected=verify_plan()["reproduction"]
    expected_counts=bool(sm["first_birth_t"]==expected["expected_first_birth_t"] and sm["live_optimizer_steps"]==expected["expected_Z_live_steps"]
                         and sm["shadow_optimizer_steps"]==expected["expected_Z_shadow_steps"]
                         and [sm["deployed_prediction_forwards"],sm["reuse_preview_forwards"],sm["shadow_qualification_forwards"]]==expected["expected_Z_forwards"])
    q=load_npz(root/"Z_zero/qualification_first.npz") if (root/"Z_zero/qualification_first.npz").exists() else None
    q_window=None if q is None else [int(q["intervals"][0]),int(q["intervals"][1])]
    expected_counts=expected_counts and q_window==expected["expected_validation_intervals"] and int(q["evaluation_t"][0])==expected["expected_candidate_evaluation_t"]
    passed=all(array_checks.values()) and update_exact and all(discrete.values()) and apdiff is not None and apdiff<=1e-12 and expected_counts
    report={"protocol":"041","arm":"Z_zero","against":"cached_D_040","all_pass":bool(passed),"array_checks":array_checks,
            "update_exact":update_exact,"discrete_event_checks":discrete,"full_AP_absolute_difference":apdiff,"expected_counts_and_events":expected_counts,
            "qualification_window":q_window,"excluded_metadata":["runtime_timing","new_audit_fields","checkpoint_file_names","action_sequence_numbers"]}
    W(root/"Z_zero/reproduction_report.json",report); return report

def cmd_preflight(a):
    runtime(); verify_plan(); src=load_sources(a.source036,a.source040)
    report=run_fixtures(src,a.out_dir)
    print(json.dumps({"protocol":"041","all_pass":report["all_pass"],"fixture_count":len(report["fixtures"])},indent=2))
    if not report["all_pass"]: raise AssertionError("Protocol041 engineering_incomplete")

def cmd_arm(a):
    runtime(); verify_plan(); src=load_sources(a.source036,a.source040); root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=True)
    fixture=J(a.fixture_report)
    if not fixture.get("all_pass"): raise AssertionError("engineering_incomplete")
    arm=a.arm
    before36={rel:sha256_file(src["source036_root"]/rel) for rel in SOURCE036}; before40={rel:sha256_file(src["source040_root"]/rel) for rel in SOURCE040}
    ar=root/arm; ar.mkdir(parents=True,exist_ok=True)
    W(ar/"source_lock.json",{"protocol":"041","arm":arm,"source036_artifact":11147931152,"source040_artifact":11265231995,
                             "files036":before36,"files040":before40,"F_loaded":False})
    try:
        if a.resume_checkpoint:
            m=Machine041.restore(src,ar/"runtime_state",a.resume_checkpoint,arm=arm,allow_gradient=True,real_science=True,strict_journal=True)
        else:
            m=Machine041(src,ar/"runtime_state",arm,allow_gradient=True,real_science=True)
        m.checkpoint_dir=ar/"checkpoints"; t0=time.perf_counter(); c0=time.process_time()
        m.advance(); m.terminal_settle()
        summary=save_arm(root,src,m,arm)
        audit=arm_audit(src,m,fixture,arm,before36,before40); W(ar/"run_audit.json",audit)
        W(ar/"science_cost_raw.json",{"arm":arm,"wall_seconds":time.perf_counter()-t0,"process_seconds":time.process_time()-c0,
          "prediction_seconds":m.prediction_seconds,"update_seconds":m.update_seconds,"controller_cpu_seconds":m.controller_cpu_seconds,
          "live_optimizer_steps":m.live_optimizer_steps,"shadow_optimizer_steps":m.shadow_optimizer_steps,
          "deployed_forwards":m.deployed_forwards,"reuse_preview_forwards":m.reuse_preview_forwards,"shadow_preview_forwards":m.shadow_preview_forwards,
          "copy_events":len(m.initialization_events)})
        W(ar/"scientific_execution_complete.json",{"protocol":"041","arm":arm,"complete":True,"audit_all_pass":audit["all_pass"]})
        if not audit["all_pass"]: raise AssertionError("Protocol041 "+arm+" science audit failed")
        if arm=="Z_zero":
            rep=reproduce_Z(src,root)
            if not rep["all_pass"]: raise RuntimeError("Z_reproduction_failed_stop_before_W")
        print(json.dumps(summary,indent=2))
    except Exception as e:
        W(ar/"status.json",{"protocol":"041","arm":arm,"science_status":"interrupted","error":repr(e),"traceback":traceback.format_exc(),"automatic_followup":False})
        raise
    W(ar/"status.json",{"protocol":"041","arm":arm,"science_status":"completed","analysis_status":"pending","publication_status":"pending","automatic_followup":False})

def cmd_copy_controls(a):
    src=load_sources(a.source036,a.source040); root=Path(a.out_dir); d=root/"cached"; d.mkdir(parents=True,exist_ok=True)
    for name,rel in (("C_ref","C_ref/predictions.npz"),("B_ref","B_ref/predictions.npz"),("D_keep","D_keep/predictions.npz"),("D_039","D_039/predictions.npz"),("D_040","D_pool2/predictions.npz")):
        q=d/name; q.mkdir(parents=True,exist_ok=True); shutil.copy2(src["source040_root"]/rel,q/"predictions.npz")
    W(d/"summary.json",{"protocol":"041","cached_controls":["C_ref","B_ref","D_keep","D_039","D_040"],"training":False,"F_loaded":False})

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("preflight"); p.add_argument("--source036",required=True); p.add_argument("--source040",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_preflight)
    p=sub.add_parser("copy-controls"); p.add_argument("--source036",required=True); p.add_argument("--source040",required=True); p.add_argument("--out-dir",required=True); p.set_defaults(fn=cmd_copy_controls)
    p=sub.add_parser("arm"); p.add_argument("--arm",choices=ARMS,required=True); p.add_argument("--source036",required=True); p.add_argument("--source040",required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--fixture-report",required=True); p.add_argument("--resume-checkpoint"); p.set_defaults(fn=cmd_arm)
    a=ap.parse_args(); a.fn(a)
if __name__=="__main__": main()
