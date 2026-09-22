"""Protocol-031 nonblocking dormant-specialist reuse policy."""
from __future__ import annotations
from copy import deepcopy
import math, time
import numpy as np
import torch

from ftmoe_protocol024_lifecycle import _loss_from_logits, _normal_anomaly_mean
from ftmoe_protocol024_v2a import proposal_due
from ftmoe_protocol024_v2b import _unit
from ftmoe_protocol024_v2c import V2C_DEFAULT
from ftmoe_protocol027_normal_guard import normal_detection_guard_report
from ftmoe_protocol028_memory_protected import (
    Protocol028MemoryProtectedLifecycle, Protocol028DynamicSession,
)

P031_CONFIG=dict(V2C_DEFAULT)
P031_CONFIG.update({
    "proposal_start_matured":600,
    "proposal_every_matured":1600,
    "candidate_train_intervals":128,
    "buffer_train_horizon_intervals":128,
    "buffer_size":64,
    "buffer_minibatch":32,
    "buffer_update_every_matured":4,
    "buffer_updates_per_opportunity":4,
    "validation_intervals":32,
    "reuse_every_matured":32,
    "reuse_recent_z_intervals":32,
    "reuse_validation_intervals":16,
    "crossfade_prediction_intervals":8,
    "reuse_enabled":False,  # inherited blocking entry is disabled; 031 owns pending slot.
})

def _normal_fp(logits,target):
    det=torch.as_tensor(logits,dtype=torch.float32).reshape(-1,2)
    y=torch.as_tensor(target,dtype=torch.long).reshape(-1)
    normal=y==0
    if not bool(normal.any()): return 0,0
    p=torch.softmax(det,-1)[:,1]
    return int((p[normal]>=0.5).sum().item()),int(normal.sum().item())

class Protocol031NonblockingReuseLifecycle(Protocol028MemoryProtectedLifecycle):
    """Memory-protected births plus one independent causal pending reuse slot."""

    def __init__(self,guard_anchor,config=None):
        cfg=dict(P031_CONFIG); cfg.update(config or {})
        super().__init__(guard_anchor=guard_anchor,config=cfg)
        self.p031_config=cfg
        self.pending_reuse=None
        self.reuse_records=[]
        self.reuse_due_accounting={
            "due":0,"started":0,"skip_pending":0,"skip_transition":0,
            "skip_no_memory":0,"skip_insufficient_z":0,
        }
        self.reuse_outcomes={
            "accepted":0,"rejected":0,"cancelled":0,"censored":0,"pending":0,
        }
        self.birth_cancelled_by_reuse=0
        self.birth_cancelled_ids=[]
        self.first_reuse_influence_cursor=None
        self.extra_compute.setdefault("p031_reuse_validation_forwards",0)
        self.extra_compute.setdefault("p031_reuse_validation_seconds",0.0)
        self.extra_compute.setdefault("p031_reuse_guard_forwards",0)
        self.extra_compute.setdefault("p031_reuse_guard_seconds",0.0)

    def state_dict(self):
        state=super().state_dict()
        state["protocol031_nonblocking_reuse"]=deepcopy({
            "p031_config":self.p031_config,
            "pending_reuse":self.pending_reuse,
            "reuse_records":self.reuse_records,
            "reuse_due_accounting":self.reuse_due_accounting,
            "reuse_outcomes":self.reuse_outcomes,
            "birth_cancelled_by_reuse":self.birth_cancelled_by_reuse,
            "birth_cancelled_ids":self.birth_cancelled_ids,
            "first_reuse_influence_cursor":self.first_reuse_influence_cursor,
        })
        return state

    def load_state_dict(self,state):
        payload=dict(state)
        p031=payload.pop("protocol031_nonblocking_reuse",None)
        super().load_state_dict(payload)
        if p031 is None:
            raise ValueError("Protocol031 lifecycle checkpoint lacks pending reuse state")
        if dict(p031["p031_config"])!=self.p031_config:
            raise ValueError("Protocol031 lifecycle config mismatch")
        for k,v in p031.items():
            if k!="p031_config": setattr(self,k,deepcopy(v))

    def _pending_topology_valid(self,session):
        if self.pending_reuse is None: return True
        slot=self.pending_reuse
        bank=session.model.learner
        key=str(slot["expert_id"])
        return (
            self.transition is None
            and str(self.active_specialist_id) if self.active_specialist_id is not None else None
        ) == slot["active_specialist_id"] and key in bank.dormant_experts

    def _cancel_pending(self,session,reason):
        slot=self.pending_reuse
        if slot is None: return
        slot["outcome"]="cancelled"
        slot["cancel_reason"]=str(reason)
        slot["cancel_cursor"]=int(session.cursor)
        slot["matured_at_cancel"]=int(self.matured_count)
        self.reuse_records.append(deepcopy(slot))
        self.reuse_outcomes["cancelled"]+=1
        self.pending_reuse=None
        self._event(session,"reuse_pending_cancelled",
                    expert_id=slot["expert_id"],reason=str(reason),
                    started_cursor=slot["started_cursor"],
                    started_matured=slot["started_matured"])

    def _start_pending_reuse(self,session):
        bank=session.model.learner
        dormant=sorted([
            str(k) for k in bank.dormant_experts
            if str(k) in self.specialist_memory
        ],key=int)
        if not dormant:
            self.reuse_due_accounting["skip_no_memory"]+=1
            self._event(session,"reuse_due_skipped_no_memory_nonblocking")
            return False
        width=int(self.p031_config["reuse_recent_z_intervals"])
        if len(self.recent_z)<width:
            self.reuse_due_accounting["skip_insufficient_z"]+=1
            self._event(session,"reuse_due_skipped_insufficient_z_nonblocking",
                        available=len(self.recent_z),required=width)
            return False
        current=_unit(np.asarray(self.recent_z[-width:],dtype=np.float64).mean(axis=0))
        scored=[]
        for key in dormant:
            mem=self.specialist_memory[key]
            sim=float(np.dot(current,np.asarray(mem["centroid"],dtype=np.float64)))
            scored.append((sim,-int(key),key,float(mem["similarity_threshold"])))
        scored.sort(reverse=True)
        sim,_,key,threshold=scored[0]
        slot={
            "expert_id":str(key),
            "active_specialist_id":None if self.active_specialist_id is None else str(self.active_specialist_id),
            "started_cursor":int(session.cursor),
            "started_matured":int(self.matured_count),
            "selection_similarity":float(sim),
            "logged_similarity_threshold":float(threshold),
            "similarity_threshold_used_as_gate":False,
            "prelabel":{},"pairs":[],
            "outcome":"pending","decision":None,
            "post_start_prediction_indices":[],
        }
        self.pending_reuse=slot
        self.reuse_due_accounting["started"]+=1
        self._event(session,"reuse_candidate_selected_nonblocking",
                    expert_id=key,similarity=float(sim),
                    logged_similarity_threshold=float(threshold),
                    similarity_threshold_used_as_gate=False,
                    selection="highest cosine; smaller expert ID tie-break",
                    active_specialist_id=slot["active_specialist_id"])
        return True

    def _reuse_guard_report_isolated(self,session,key):
        old_last_z=getattr(session.model,"_last_z",None)
        started=time.perf_counter()
        anchor=self.guard_anchor; labels=anchor["labels"]
        live_det=[]; cand_det=[]; targets=[]
        try:
            with torch.no_grad():
                for left in range(0,int(labels.shape[0]),32):
                    right=min(int(labels.shape[0]),left+32)
                    import run_ftmoe_protocol023_s4 as s4
                    context=s4.graph_context(
                        anchor["ids"][left:right],anchor["before"][left:right],
                        anchor["caps"][left:right])
                    context["observed_history_length"]=anchor["observed_history_length"][left:right]
                    out=session.model.predict_deployment(
                        anchor["x"][left:right],anchor["schedule"][left:right],
                        anchor["graph_x"][left:right],graph_context=context)
                    correction,_=self._preview_specialist(
                        session,key,session.model._last_z.detach())
                    live_det.append(out["detection_logits"].detach().cpu())
                    cand_det.append((out["base_final_detection_logits"]+
                                     correction[...,:2]).detach().cpu())
                    targets.append(labels[left:right].detach().cpu())
                    self.extra_compute["p031_reuse_guard_forwards"]+=1
        finally:
            session.model._last_z=old_last_z
        report=normal_detection_guard_report(
            torch.cat(live_det,0),torch.cat(cand_det,0),torch.cat(targets,0))
        report["preview_topology"]="four_generalists_plus_reused_specialist"
        report["guard_role"]="F0_known_normal_regression_guard"
        self.extra_compute["p031_reuse_guard_seconds"]+=time.perf_counter()-started
        return report

    def _score_pending(self,session,index,target):
        slot=self.pending_reuse
        if slot is None: return False
        rec=slot["prelabel"].pop(int(index),None)
        if rec is None: return False
        live_loss=_loss_from_logits(
            rec["live_detection_logits"],rec["live_class_logits"],target)
        cand_loss=_loss_from_logits(
            rec["candidate_detection_logits"],rec["candidate_class_logits"],target)
        live_normal,nrows=_normal_anomaly_mean(rec["live_detection_logits"],target)
        cand_normal,_=_normal_anomaly_mean(rec["candidate_detection_logits"],target)
        live_fp,n1=_normal_fp(rec["live_detection_logits"],target)
        cand_fp,n2=_normal_fp(rec["candidate_detection_logits"],target)
        if n1!=n2 or n1!=int(nrows):
            raise AssertionError("Protocol031 reuse normal-row accounting mismatch")
        slot["pairs"].append({
            "index":int(index),"prediction_cursor":int(rec["prediction_cursor"]),
            "label_published_cursor":int(session.cursor),
            "live_loss":float(live_loss),"candidate_loss":float(cand_loss),
            "normal_rows":int(nrows),
            "live_normal_probability":live_normal,
            "candidate_normal_probability":cand_normal,
            "live_fp_0p5":int(live_fp),"candidate_fp_0p5":int(cand_fp),
            "candidate_route_probability_mean":float(rec["candidate_route_probability_mean"]),
        })
        if len(slot["pairs"])>=int(self.p031_config["reuse_validation_intervals"]):
            self._decide_pending_reuse(session)
            return True
        return False

    def _cancel_shadow_for_reuse(self,session):
        if session.model.learner.shadow_id is None: return None
        cid=str(session.model.learner.shadow_id)
        session.discard_shadow()
        rec=self.candidate_records.get(cid)
        if rec is not None:
            rec["accepted"]=False
            rec["cancelled_by_reuse"]=True
            rec["cancelled_cursor"]=int(session.cursor)
            rec["decision"]={
                "accepted":False,"cancelled":True,
                "reason":"cancelled_by_accepted_reuse_before_birth_acceptance",
            }
        self.birth_cancelled_by_reuse+=1
        self.birth_cancelled_ids.append(cid)
        self.candidate_id=None; self.candidate_parent_id=None
        self.candidate_training_indices=[]; self.candidate_validation_indices=[]
        self.training_buffer=[]; self.training_targets={}; self.training_elapsed_distinct=0
        self.prelabel_candidate={}; self.validation_pairs=[]
        self.phase="monitoring"
        self._event(session,"candidate_cancelled_by_reuse",
                    candidate_id=cid,reason="accepted_reuse_has_same_point_priority")
        return cid

    def _decide_pending_reuse(self,session):
        slot=self.pending_reuse
        if slot is None: return
        needed=int(self.p031_config["reuse_validation_intervals"])
        pairs=slot["pairs"][:needed]
        key=str(slot["expert_id"])
        if len(pairs)<needed:
            raise AssertionError("Protocol031 reuse decision before 16 matured pairs")
        live=float(np.mean([x["live_loss"] for x in pairs]))
        cand=float(np.mean([x["candidate_loss"] for x in pairs]))
        improvement=(live-cand)/max(abs(live),1e-12)
        normal_pairs=[x for x in pairs if x["normal_rows"]>0]
        if normal_pairs:
            w=[x["normal_rows"] for x in normal_pairs]
            live_normal=float(np.average(
                [x["live_normal_probability"] for x in normal_pairs],weights=w))
            cand_normal=float(np.average(
                [x["candidate_normal_probability"] for x in normal_pairs],weights=w))
            normal_ok=cand_normal<=live_normal+0.01
        else:
            live_normal=cand_normal=None; normal_ok=False
        normal_rows=sum(x["normal_rows"] for x in pairs)
        live_fp=sum(x["live_fp_0p5"] for x in pairs)
        cand_fp=sum(x["candidate_fp_0p5"] for x in pairs)
        live_fpr=live_fp/float(normal_rows) if normal_rows else None
        cand_fpr=cand_fp/float(normal_rows) if normal_rows else None
        fpr_delta=None if live_fpr is None else float(cand_fpr-live_fpr)
        fpr_ok=fpr_delta is not None and fpr_delta<=0.01
        guard=self._reuse_guard_report_isolated(session,key)
        guard_loss_ok=bool(guard.get("available") and guard.get("normal_nll_ok_direct"))
        guard_fpr_ok=bool(
            guard.get("available") and guard.get("fpr_delta_0p5") is not None
            and float(guard["fpr_delta_0p5"])<=0.01)
        loss_ok=improvement>=0.01
        accept=bool(loss_ok and normal_ok and fpr_ok and guard_loss_ok and guard_fpr_ok)
        reasons=[]
        if not loss_ok: reasons.append("relative_loss_improvement_below_1pct")
        if not normal_ok: reasons.append("mean_normal_probability_guard_failed_or_unavailable")
        if not fpr_ok: reasons.append("validation_fpr_delta_failed_or_unavailable")
        if not guard.get("available"): reasons.append("F0_normal_guard_unavailable")
        elif not guard_loss_ok: reasons.append("F0_normal_nll_regression")
        elif not guard_fpr_ok: reasons.append("F0_normal_fpr_regression")
        decision={
            "expert_id":key,"live_mean_loss":live,"candidate_mean_loss":cand,
            "relative_improvement":float(improvement),"required_relative_improvement":0.01,
            "live_normal_probability":live_normal,
            "candidate_normal_probability":cand_normal,
            "normal_safety_ok":bool(normal_ok),
            "validation_live_fpr_0p5":live_fpr,
            "validation_candidate_fpr_0p5":cand_fpr,
            "validation_fpr_delta_0p5":fpr_delta,
            "validation_fpr_ok":bool(fpr_ok),
            "guard":guard,"guard_loss_ok":bool(guard_loss_ok),
            "guard_fpr_ok":bool(guard_fpr_ok),
            "validation_intervals":len(pairs),
            "accepted":accept,"reject_reasons":reasons,
            "selection_similarity":slot["selection_similarity"],
            "logged_similarity_threshold":slot["logged_similarity_threshold"],
            "similarity_threshold_used_as_gate":False,
        }
        slot["decision"]=deepcopy(decision)
        if accept:
            if self.transition is not None:
                raise AssertionError("Protocol031 accepted reuse during crossfade")
            old=self.active_specialist_id
            self._cancel_shadow_for_reuse(session)
            if key not in session.model.learner.dormant_experts:
                self._cancel_pending(session,"candidate_not_dormant_at_acceptance")
                return
            session.reactivate_expert(key)
            self.specialist_memory[key]["last_causal_accept_matured"]=int(self.matured_count)
            self.reuse_accepted+=1; self.reactivations+=1
            self.reuse_outcomes["accepted"]+=1
            slot["outcome"]="accepted"; slot["accepted_cursor"]=int(session.cursor)
            self.reuse_records.append(deepcopy(slot))
            self.pending_reuse=None
            self._event(session,"specialist_reactivated_nonblocking",**decision)
            self._begin_transition(session,old,key,"reactivation")
        else:
            self.reuse_rejected+=1
            self.reuse_outcomes["rejected"]+=1
            slot["outcome"]="rejected"; slot["rejected_cursor"]=int(session.cursor)
            self.reuse_records.append(deepcopy(slot))
            self.pending_reuse=None
            self._event(session,"reuse_candidate_rejected_nonblocking",**decision)
            # Crucial: do not mutate the independent birth phase/cooldown.

    def on_pre_label_prediction(self,session,index,output):
        # Parent handles z history, birth candidate preview, ramps and transition completion.
        super().on_pre_label_prediction(session,index,output)
        slot=self.pending_reuse
        if slot is None: return
        if self.transition is not None:
            self._cancel_pending(session,"crossfade_started_or_active")
            self._sync_session(session); return
        active=None if self.active_specialist_id is None else str(self.active_specialist_id)
        key=str(slot["expert_id"])
        if active!=slot["active_specialist_id"] or key not in session.model.learner.dormant_experts:
            self._cancel_pending(session,"topology_changed")
            self._sync_session(session); return
        started=time.perf_counter()
        z=session.model._last_z.detach()
        with torch.no_grad():
            correction,routed=self._preview_specialist(session,key,z)
            det=output["base_final_detection_logits"]+correction[...,:2]
            cls=output["base_final_class_logits"]+correction[...,2:]
        slot["prelabel"][int(index)]={
            "candidate_detection_logits":det[0].detach().cpu().numpy(),
            "candidate_class_logits":cls[0].detach().cpu().numpy(),
            "live_detection_logits":output["detection_logits"][0].detach().cpu().numpy(),
            "live_class_logits":output["class_logits"][0].detach().cpu().numpy(),
            "candidate_route_probability_mean":float(routed[...,-1].mean().cpu()),
            "prediction_cursor":int(session.cursor),"recorded_before_label":True,
        }
        slot["post_start_prediction_indices"].append(int(index))
        self.extra_compute["p031_reuse_validation_forwards"]+=1
        self.extra_compute["p031_reuse_validation_seconds"]+=time.perf_counter()-started
        self._sync_session(session)

    def _birth_due(self):
        return proposal_due(
            self.matured_count,
            int(self.p031_config["proposal_start_matured"]),
            int(self.p031_config["proposal_every_matured"]))

    def _reuse_due(self):
        return (
            self.matured_count>=int(self.p031_config["proposal_start_matured"])
            and self.matured_count%int(self.p031_config["reuse_every_matured"])==0
        )

    def _process_birth_phase(self,session,index,target):
        topology_before=None if self.active_specialist_id is None else str(self.active_specialist_id)
        if self.phase=="candidate_training":
            self._buffer_train(session,index,target)
        elif self.phase=="candidate_validation":
            self._score_candidate_validation(session,index,target)
        elif self.phase=="cooldown":
            self.cooldown_left-=1
            if self.cooldown_left<=0:
                self.phase="monitoring"; self._event(session,"cooldown_complete")
        elif self.phase=="transition":
            pass
        elif self.phase!="monitoring":
            raise RuntimeError("unexpected Protocol031 birth phase %s"%self.phase)
        topology_after=None if self.active_specialist_id is None else str(self.active_specialist_id)
        if self.pending_reuse is not None and (
            topology_after!=topology_before or self.transition is not None
        ):
            self._cancel_pending(session,"birth_acceptance_or_topology_changed")

    def on_matured(self,session,index,target):
        if session.arm!="D": return
        index=int(index)
        loss=self._current_live_loss(session,index,target)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite Protocol031 matured loss")
        self.loss_history.append(loss)
        self.matured_count+=1
        self._diagnostic(session,index,target,loss)

        # Existing pending reuse matures before birth processing at this label event.
        reuse_accepted_now=False
        before_accept=self.reuse_outcomes["accepted"]
        if self.pending_reuse is not None:
            self._score_pending(session,index,target)
            reuse_accepted_now=self.reuse_outcomes["accepted"]>before_accept

        birth_due=self._birth_due()
        if birth_due:
            self.birth_accounting["due"]+=1

        if not reuse_accepted_now:
            self._process_birth_phase(session,index,target)
        elif birth_due:
            self.birth_accounting["skipped_busy"]+=1
            self._event(session,"birth_due_skipped_busy",
                        reason="reuse_accepted_same_matured_event",
                        controller_phase=self.phase)

        if birth_due and not reuse_accepted_now:
            if self.phase!="monitoring" or session.model.learner.shadow_id is not None or self.transition is not None:
                self.birth_accounting["skipped_busy"]+=1
                self._event(session,"birth_due_skipped_busy",
                            reason="candidate_cooldown_or_transition_busy",
                            controller_phase=self.phase)
            else:
                self.proposal_opportunities+=1
                cap_before=self.capacity_blocked
                if self._start_candidate(session,"periodic_background_budget_1600"):
                    self.birth_accounting["started"]+=1
                elif self.capacity_blocked>cap_before:
                    self.birth_accounting["skipped_capacity"]+=1
                else:
                    self.birth_accounting["skipped_busy"]+=1

        # New reuse admission happens after all birth processing at this matured event.
        if self._reuse_due():
            self.reuse_due_accounting["due"]+=1
            if self.transition is not None:
                self.reuse_due_accounting["skip_transition"]+=1
                self._event(session,"reuse_due_skipped_transition_nonblocking")
            elif self.pending_reuse is not None:
                self.reuse_due_accounting["skip_pending"]+=1
                self._event(session,"reuse_due_skipped_pending_nonblocking",
                            expert_id=self.pending_reuse["expert_id"])
            else:
                self._start_pending_reuse(session)

        self._sync_session(session)

    def mark_stream_end(self):
        super().mark_stream_end()
        if self.pending_reuse is not None:
            slot=self.pending_reuse
            slot["outcome"]="censored"
            slot["censor_reason"]="stream_end_before_16_future_labels_matured"
            self.reuse_records.append(deepcopy(slot))
            self.reuse_outcomes["censored"]+=1
            self.pending_reuse=None
        self.reuse_outcomes["pending"]=0

    def due_conservation(self):
        base=super().due_conservation()
        d=self.reuse_due_accounting
        due_accounted=sum(int(d[k]) for k in (
            "started","skip_pending","skip_transition","skip_no_memory","skip_insufficient_z"))
        started=int(d["started"])
        outcomes=sum(int(self.reuse_outcomes[k]) for k in (
            "accepted","rejected","cancelled","censored","pending"))
        base["reuse_nonblocking"]={
            **{k:int(v) for k,v in d.items()},
            "due_accounted":int(due_accounted),
            "due_conserved":int(d["due"])==int(due_accounted),
            "outcomes":{k:int(v) for k,v in self.reuse_outcomes.items()},
            "started_outcome_accounted":int(outcomes),
            "started_conserved":started==outcomes,
        }
        return base

class Protocol031DynamicSession(Protocol028DynamicSession):
    """Protocol-028 memory protection with Protocol-031 nonblocking reuse."""

    def __init__(self,*args,guard_anchor,v2c_config=None,**kwargs):
        cfg=dict(P031_CONFIG); cfg.update(v2c_config or {})
        super().__init__(*args,guard_anchor=guard_anchor,v2c_config=cfg,**kwargs)
        self.lifecycle_controller=Protocol031NonblockingReuseLifecycle(
            guard_anchor=guard_anchor,config=cfg)
        self.lifecycle_state=self.lifecycle_controller.state_dict()
        self.lifecycle_state["enabled"]=True
        self._apply_specialist_freeze()
        self.lifecycle_controller._sync_session(self)
        self.comparator="D_nonblocking_reuse"

    def comparator_manifest(self):
        base=deepcopy(super().comparator_manifest())
        base.update({
            "name":"D_nonblocking_reuse","protocol":"031","plan_revision":2,
            "reuse_policy":"independent nonblocking pending slot; cosine ranks only; original causal quality gates",
            "birth_start_matured":600,"birth_every_matured":1600,
            "reuse_every_matured":32,"reuse_validation_future_intervals":16,
            "similarity_threshold_is_admission_gate":False,
            "online_update_every_scored_intervals":16,
        })
        return base
