"""Protocol-024 next_round_v2c: replacement-consistent birth qualification.

Only the v2 birth preview semantics and audit/recovery accounting change here.
The registered v2b memory/reuse policy, buffer128_4x4 budget, thresholds, and
common online settings remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import time

import numpy as np
import torch

import run_ftmoe_protocol023_s4 as s4
from ftmoe_protocol024_lifecycle import _loss_from_logits, _unit_rows
from ftmoe_protocol024_v2a import fixed_threshold_fpr, proposal_due
from ftmoe_protocol024_v2b import MemoryResidualLifecycle, V2BProtocol024Session, V2B_DEFAULT


V2C_DEFAULT = dict(V2B_DEFAULT)
V2C_DEFAULT.update({
    "replacement_consistent_birth": True,
    "reuse_enabled": True,
    "audit_all_theoretical_due": True,
})


def _hash_value(digest, value):
    if torch.is_tensor(value):
        digest.update(str(value.dtype).encode("utf8"))
        digest.update(str(tuple(value.shape)).encode("utf8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode("utf8"))
        digest.update(str(value.shape).encode("utf8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda x: str(x)):
            digest.update(str(key).encode("utf8")); _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value: _hash_value(digest, item)
    elif value is None:
        digest.update(b"<none>")
    else:
        digest.update(repr(value).encode("utf8"))


def optimizer_state_hash(optimizer):
    digest = hashlib.sha256(); _hash_value(digest, optimizer.state_dict()); return digest.hexdigest()


def model_tensor_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(str(key).encode("utf8")); _hash_value(digest, tensor)
    return digest.hexdigest()


def rng_state_hash():
    digest = hashlib.sha256(); _hash_value(digest, torch.get_rng_state()); _hash_value(digest, np.random.get_state())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all(): _hash_value(digest, state)
    return digest.hexdigest()


class ReplacementConsistentLifecycle(MemoryResidualLifecycle):
    """v2b lifecycle with birth train/validation/guard using final replacement topology."""

    def __init__(self, guard_anchor, config=None):
        cfg = dict(V2C_DEFAULT); cfg.update(config or {})
        super().__init__(guard_anchor=guard_anchor, config=cfg)
        self.v2c_config = cfg
        self.birth_accounting = {"due":0,"started":0,"skipped_busy":0,"skipped_no_memory":0,"skipped_no_match":0,"skipped_capacity":0,"not_yet_eligible":0}
        self.reuse_accounting = {"due":0,"started":0,"skipped_busy":0,"skipped_no_memory":0,"skipped_no_match":0,"skipped_capacity":0,"not_yet_eligible":0}
        self.reuse_disabled_due = 0
        self.candidate_records = {}
        self.freeze_checks = {"live_update":0,"shadow_train":0}
        self.extra_compute.setdefault("legacy_preview_audit_forwards", 0)

    def state_dict(self):
        state = super().state_dict()
        state["v2c"] = deepcopy({
            "v2c_config": self.v2c_config,
            "birth_accounting": self.birth_accounting,
            "reuse_accounting": self.reuse_accounting,
            "reuse_disabled_due": self.reuse_disabled_due,
            "candidate_records": self.candidate_records,
            "freeze_checks": self.freeze_checks,
        })
        return state

    def load_state_dict(self, state):
        payload = dict(state); v2c = payload.pop("v2c", None); super().load_state_dict(payload)
        if v2c is None: raise ValueError("v2c lifecycle checkpoint lacks v2c state")
        if dict(v2c["v2c_config"]) != self.v2c_config: raise ValueError("v2c config mismatch on resume")
        for key, value in v2c.items():
            if key != "v2c_config": setattr(self, key, deepcopy(value))

    def _birth_target_preview(self, session, z):
        """Final birth topology: current four generalists + shadow; old specialist excluded."""
        bank = session.model.learner
        if bank.shadow_id is None: raise RuntimeError("no shadow candidate for replacement preview")
        sid = str(bank.shadow_id); ids = list(self.generalist_ids) + [sid]
        experts, weights, biases, ramps = {}, {}, {}, {}
        for gid in self.generalist_ids:
            if gid not in bank.experts: raise AssertionError("generalist not active: %s" % gid)
            experts[gid] = bank.experts[gid]; weights[gid] = bank.router_weights[gid]; biases[gid] = bank.router_biases[gid]
            ramps[gid] = float(bank.ramp.get(gid, 1.0))
        experts[sid] = bank.shadow_experts[sid]; weights[sid] = bank.shadow_router_weights[sid]; biases[sid] = bank.shadow_router_biases[sid]; ramps[sid] = 1.0
        return bank._mixture(z, ids, experts, weights, biases, ramps)

    def _legacy_incremental_preview(self, session, z):
        """Audit-only v2 preview; never used for v2c optimization or decisions."""
        return session.model.learner.preview_with_shadow(z, shadow_ramp=1.0)

    def frozen_specialist_hash(self, session):
        bank = session.model.learner; digest = hashlib.sha256()
        for key in sorted(self.specialist_memory, key=lambda x: int(x)):
            digest.update(str(key).encode("utf8"))
            if key in bank.experts:
                expert, weight, bias, loc = bank.experts[key], bank.router_weights[key], bank.router_biases[key], "active"
            elif key in bank.dormant_experts:
                expert, weight, bias, loc = bank.dormant_experts[key], bank.dormant_router_weights[key], bank.dormant_router_biases[key], "dormant"
            else:
                continue
            digest.update(loc.encode("utf8"))
            for name, tensor in sorted(expert.state_dict().items()): digest.update(name.encode("utf8")); _hash_value(digest, tensor)
            _hash_value(digest, weight); _hash_value(digest, bias)
            prefix = str(key) + "|"
            for name in sorted(session.optimizer_archive):
                if str(name).startswith(prefix): digest.update(str(name).encode("utf8")); _hash_value(digest, session.optimizer_archive[name])
        return digest.hexdigest()

    def _start_candidate(self, session, reason):
        started = super()._start_candidate(session, reason)
        if started:
            cid = str(self.candidate_id)
            self.candidate_records[cid] = {
                "candidate_id":cid,"parent_id":None if self.candidate_parent_id is None else str(self.candidate_parent_id),
                "created_cursor":int(session.cursor),"created_matured":int(self.matured_count),"training_indices":[],"validation":[],
                "accepted":None,"accepted_cursor":None,"first_influence_cursor":None,"full_ramp_cursor":None,"stream_end_censored":False,
            }
        return started

    def _buffer_train(self, session, index, target):
        idx = int(index)
        if idx in self.training_targets: raise AssertionError("buffered candidate saw duplicate matured interval")
        if self.candidate_id is not None: self.candidate_records[str(self.candidate_id)]["training_indices"].append(idx)
        self.training_buffer.append(idx); self.training_targets[idx] = np.asarray(target, dtype=np.int64).copy()
        self.training_buffer = self.training_buffer[-int(self.v2_config["buffer_size"]):]; self.training_elapsed_distinct += 1
        every = int(self.v2_config["buffer_update_every_matured"]); batch = int(self.v2_config["buffer_minibatch"]); updates = int(self.v2_config["buffer_updates_per_opportunity"])
        if self.training_elapsed_distinct % every == 0 and len(self.training_buffer) >= batch:
            indices = list(self.training_buffer[-batch:]); targets = np.stack([self.training_targets[i] for i in indices], axis=0)
            x, sched, graph, context = s4.window_batch(session.replay, indices); frozen_before = self.frozen_specialist_hash(session)
            for _ in range(updates):
                started = time.perf_counter(); base, z = session.model._base_forward(x, sched, graph, graph_context=context, record=False)
                session.model.zero_grad(set_to_none=True); session.shadow_optimizer.zero_grad(set_to_none=True)
                correction, _ = self._birth_target_preview(session, z)
                detection = base["detection_logits"] + correction[..., :2]; classes = base["class_logits"] + correction[..., 2:]
                terms = s4.supervised_terms(detection, classes, torch.as_tensor(targets).long()); terms["total"].backward(); session.shadow_optimizer.step(); session.model.zero_grad(set_to_none=True)
                self.extra_compute["shadow_train_steps"] += 1; self.extra_compute["shadow_train_examples"] += len(indices); self.extra_compute["shadow_train_seconds"] += time.perf_counter() - started
            if frozen_before != self.frozen_specialist_hash(session): raise AssertionError("frozen specialist changed during shadow training")
            self.freeze_checks["shadow_train"] += 1
        if self.training_elapsed_distinct >= int(self.v2_config["buffer_train_horizon_intervals"]):
            self.candidate_training_indices = sorted(self.training_targets.keys())
            for parameter in session.model.learner.shadow_parameters(): parameter.requires_grad_(False); parameter.grad = None
            self.phase = "candidate_validation"; self.prelabel_candidate = {}; self.validation_pairs = []
            self._event(session,"candidate_training_complete",candidate_id=self.candidate_id,distinct_intervals=self.training_elapsed_distinct,
                        shadow_train_steps=self.extra_compute["shadow_train_steps"],shadow_train_examples=self.extra_compute["shadow_train_examples"],
                        parent_parameter_l2_change=self._shadow_parent_delta(session),preview_topology="four_generalists_plus_new_specialist")

    def _guard_report(self, session):
        anchor = self.guard_anchor; labels = anchor["labels"]; started = time.perf_counter(); live_det=[]; live_cls=[]; cand_det=[]; cand_cls=[]; targets=[]
        with torch.no_grad():
            for left in range(0, int(labels.shape[0]), 32):
                right=min(int(labels.shape[0]),left+32); context=s4.graph_context(anchor["ids"][left:right],anchor["before"][left:right],anchor["caps"][left:right])
                out=session.model.predict_deployment(anchor["x"][left:right],anchor["schedule"][left:right],anchor["graph_x"][left:right],graph_context=context)
                correction,_=self._birth_target_preview(session,session.model._last_z.detach())
                live_det.append(out["detection_logits"].detach().cpu()); live_cls.append(out["class_logits"].detach().cpu())
                cand_det.append((out["base_final_detection_logits"]+correction[...,:2]).detach().cpu()); cand_cls.append((out["base_final_class_logits"]+correction[...,2:]).detach().cpu())
                targets.append(labels[left:right].detach().cpu()); self.extra_compute["guard_forwards"] += 1
        live_det=torch.cat(live_det,0); live_cls=torch.cat(live_cls,0); cand_det=torch.cat(cand_det,0); cand_cls=torch.cat(cand_cls,0); target=torch.cat(targets,0)
        live_loss=_loss_from_logits(live_det,live_cls,target); candidate_loss=_loss_from_logits(cand_det,cand_cls,target)
        live_fpr,normal_rows=fixed_threshold_fpr(live_det,target); candidate_fpr,_=fixed_threshold_fpr(cand_det,target); positive_rows=int((target.reshape(-1)>0).sum().item())
        available=normal_rows>0 and positive_rows>0; rel=(candidate_loss-live_loss)/max(abs(live_loss),1e-12); fpr_delta=None if live_fpr is None or candidate_fpr is None else float(candidate_fpr-live_fpr)
        self.extra_compute["guard_seconds"] += time.perf_counter()-started
        return {"available":bool(available),"rows":int(target.numel()),"normal_rows":int(normal_rows),"positive_rows":int(positive_rows),"live_loss":float(live_loss),
                "candidate_loss":float(candidate_loss),"relative_loss_increase":float(rel),"live_fpr_0p5":live_fpr,"candidate_fpr_0p5":candidate_fpr,"fpr_delta_0p5":fpr_delta,
                "preview_topology":"four_generalists_plus_new_specialist"}

    def _record_recent_z(self, session, output):
        bank=session.model.learner; z_tensor=session.model._last_z.detach(); z=z_tensor.cpu().numpy(); routing=output.get("correction_router_probabilities")
        if routing is None: return z_tensor
        routing_np=routing.detach().cpu().numpy().reshape(-1,routing.shape[-1]); z_unit=_unit_rows(z)
        for column,key in enumerate(bank.ids):
            weight=routing_np[:,column]; total=float(weight.sum())
            if total<=0.0: continue
            vector=(z_unit*weight[:,None]).sum(axis=0); self.route_mass[key]=float(self.route_mass.get(key,0.0)+total)
            if key not in self.z_weighted_sum: self.z_weighted_sum[key]=np.zeros(vector.shape,dtype=np.float64); self.z_weight[key]=0.0
            self.z_weighted_sum[key]=np.asarray(self.z_weighted_sum[key])+vector; self.z_weight[key]=float(self.z_weight[key]+total)
        mean_z=z_unit.mean(axis=0); mean_z/=max(np.linalg.norm(mean_z),1e-12); self.recent_z.append(mean_z)
        if len(self.recent_z)>self.config["trigger_window"]: self.recent_z=self.recent_z[-self.config["trigger_window"]:]
        return z_tensor

    def on_pre_label_prediction(self, session, index, output):
        if session.arm!="D": return
        bank=session.model.learner; z_tensor=self._record_recent_z(session,output)
        if self.phase=="candidate_validation" and bank.shadow_id is not None:
            started=time.perf_counter()
            with torch.no_grad():
                replacement,routed=self._birth_target_preview(session,z_tensor); det=output["base_final_detection_logits"]+replacement[...,:2]; cls=output["base_final_class_logits"]+replacement[...,2:]
                legacy,legacy_routed=self._legacy_incremental_preview(session,z_tensor); legacy_det=output["base_final_detection_logits"]+legacy[...,:2]; legacy_cls=output["base_final_class_logits"]+legacy[...,2:]
            self.prelabel_candidate[int(index)]={"candidate_detection_logits":det[0].detach().cpu().numpy(),"candidate_class_logits":cls[0].detach().cpu().numpy(),
                "legacy_incremental_detection_logits":legacy_det[0].detach().cpu().numpy(),"legacy_incremental_class_logits":legacy_cls[0].detach().cpu().numpy(),
                "live_detection_logits":output["detection_logits"][0].detach().cpu().numpy(),"live_class_logits":output["class_logits"][0].detach().cpu().numpy(),
                "candidate_route_probability_mean":float(routed[...,-1].mean().cpu()),"legacy_route_probability_mean":float(legacy_routed[...,-1].mean().cpu()),
                "prediction_cursor":int(session.cursor),"recorded_before_label":True,"decision_preview":"replacement","legacy_preview_is_audit_only":True}
            self.extra_compute["shadow_validation_forwards"] += 1; self.extra_compute["route_diagnostic_forwards"] += 1; self.extra_compute["legacy_preview_audit_forwards"] += 1
            self.extra_compute["shadow_validation_seconds"] += time.perf_counter()-started
        for key in list(self.accepted_ids):
            if key not in bank.ramp: continue
            ramp=float(bank.ramp[key]); rec=self.candidate_records.get(str(key))
            if ramp>0.0 and key not in self.first_influence_seen:
                self.first_influence_seen.add(key)
                if rec is not None: rec["first_influence_cursor"]=int(index)
                self._event(session,"candidate_first_influence_prediction",expert_id=key,prediction_index=int(index),ramp=ramp,accepted_cursor=self.accepted_cursor.get(key))
            if ramp>=1.0 and key not in self.full_ramp_seen:
                self.full_ramp_seen.add(key)
                if rec is not None: rec["full_ramp_cursor"]=int(index)
                self._event(session,"candidate_full_ramp_prediction",expert_id=key,prediction_index=int(index),ramp=ramp,accepted_cursor=self.accepted_cursor.get(key))
        if self.phase=="reuse_validation" and self.reuse_candidate_id is not None:
            started=time.perf_counter()
            with torch.no_grad(): correction,routed=self._preview_specialist(session,self.reuse_candidate_id,z_tensor); det=output["base_final_detection_logits"]+correction[...,:2]; cls=output["base_final_class_logits"]+correction[...,2:]
            self.reuse_prelabel[int(index)]={"candidate_detection_logits":det[0].detach().cpu().numpy(),"candidate_class_logits":cls[0].detach().cpu().numpy(),"live_detection_logits":output["detection_logits"][0].detach().cpu().numpy(),
                "live_class_logits":output["class_logits"][0].detach().cpu().numpy(),"candidate_route_probability_mean":float(routed[...,-1].mean().cpu()),"prediction_cursor":int(session.cursor),"recorded_before_label":True}
            self.extra_compute["reuse_validation_forwards"] += 1; self.extra_compute["reuse_validation_seconds"] += time.perf_counter()-started
        self._finish_transition_after_prediction(session,index,output); self._sync_session(session)

    def _score_candidate_validation(self, session, index, target):
        record=self.prelabel_candidate.get(int(index))
        if record is not None and self.candidate_id is not None:
            cid=str(self.candidate_id); repl=_loss_from_logits(record["candidate_detection_logits"],record["candidate_class_logits"],target); old=_loss_from_logits(record["legacy_incremental_detection_logits"],record["legacy_incremental_class_logits"],target)
            self.candidate_records[cid]["validation"].append({"index":int(index),"prediction_cursor":int(record["prediction_cursor"]),"label_published_cursor":int(session.cursor),"replacement_loss":float(repl),"legacy_incremental_loss":float(old),
                "replacement_route_probability_mean":float(record["candidate_route_probability_mean"]),"legacy_route_probability_mean":float(record["legacy_route_probability_mean"])})
        return super()._score_candidate_validation(session,index,target)

    def _decide_candidate(self, session):
        cid=str(self.candidate_id); old_active=self.active_specialist_id; super()._decide_candidate(session); rec=self.candidate_records.get(cid)
        if rec is not None:
            rec["accepted"]=cid in self.accepted_ids; rec["accepted_cursor"]=int(self.accepted_cursor[cid]) if cid in self.accepted_cursor else None; rec["old_active_specialist_id"]=old_active; rec["decision"]=deepcopy(self.last_decision)

    def _due_flags(self):
        start=int(self.v2_config["proposal_start_matured"])
        if self.matured_count<start: self.birth_accounting["not_yet_eligible"] += 1
        birth_due=proposal_due(self.matured_count,start,int(self.v2_config["proposal_every_matured"])); reuse_tick=self.matured_count%int(self.v2b_config["reuse_every_matured"])==0
        if reuse_tick and self.matured_count<start: self.reuse_accounting["not_yet_eligible"] += 1
        reuse_due=reuse_tick and self.matured_count>=start
        if birth_due: self.birth_accounting["due"] += 1
        if reuse_due:
            if self.v2c_config.get("reuse_enabled",True): self.reuse_accounting["due"] += 1
            else: self.reuse_disabled_due += 1
        return birth_due,reuse_due

    def _account_busy(self, session, birth_due, reuse_due):
        if birth_due:
            self.birth_accounting["skipped_busy"] += 1; self._event(session,"birth_due_skipped_busy",reason="candidate_reuse_cooldown_or_transition_busy",controller_phase=self.phase)
        if reuse_due and self.v2c_config.get("reuse_enabled",True):
            self.reuse_accounting["skipped_busy"] += 1; self._event(session,"reuse_due_skipped_busy",reason="candidate_reuse_cooldown_or_transition_busy",controller_phase=self.phase)

    def on_matured(self, session, index, target):
        if session.arm!="D": return
        index=int(index); loss=self._current_live_loss(session,index,target)
        if not math.isfinite(loss): raise RuntimeError("non-finite matured supervised loss")
        self.loss_history.append(loss); self.matured_count += 1; self._diagnostic(session,index,target,loss); birth_due,reuse_due=self._due_flags()
        if self.phase!="monitoring":
            self._account_busy(session,birth_due,reuse_due)
            if self.phase=="candidate_training": self._buffer_train(session,index,target)
            elif self.phase=="candidate_validation": self._score_candidate_validation(session,index,target)
            elif self.phase=="reuse_validation": self._score_reuse(session,index,target)
            elif self.phase=="cooldown":
                self.cooldown_left-=1
                if self.cooldown_left<=0: self.phase="monitoring"; self._event(session,"cooldown_complete")
            elif self.phase=="transition": pass
            else: raise RuntimeError("unexpected v2c lifecycle phase %s"%self.phase)
            self._sync_session(session); return
        if reuse_due:
            if self.v2c_config.get("reuse_enabled",True):
                nm=self.reuse_skipped_no_memory; nn=self.reuse_skipped_no_match; started=self._maybe_start_reuse(session)
                if started:
                    self.reuse_accounting["started"] += 1
                    if birth_due: self.birth_accounting["skipped_busy"] += 1; self._event(session,"birth_due_skipped_busy",reason="reuse_started_same_due_slot",controller_phase=self.phase)
                    self._sync_session(session); return
                if self.reuse_skipped_no_memory>nm: self.reuse_accounting["skipped_no_memory"] += 1
                elif self.reuse_skipped_no_match>nn: self.reuse_accounting["skipped_no_match"] += 1
                else: self.reuse_accounting["skipped_capacity"] += 1
            else: self._event(session,"reuse_due_disabled_by_diagnostic_arm")
        if birth_due:
            self.proposal_opportunities += 1; cap_before=self.capacity_blocked
            if self.phase!="monitoring" or session.model.learner.shadow_id is not None:
                self.proposal_skipped += 1; self.birth_accounting["skipped_busy"] += 1; self._event(session,"proposal_skipped_busy",reason="candidate_or_reuse_busy",controller_phase=self.phase)
            elif self._start_candidate(session,"periodic_background_budget"):
                self.birth_accounting["started"] += 1; self._sync_session(session); return
            elif self.capacity_blocked>cap_before: self.birth_accounting["skipped_capacity"] += 1
            else: self.birth_accounting["skipped_busy"] += 1
        self._sync_session(session)

    def mark_stream_end(self):
        if self.candidate_id is not None and str(self.candidate_id) in self.candidate_records: self.candidate_records[str(self.candidate_id)]["stream_end_censored"] = True

    def due_conservation(self):
        fields=("started","skipped_busy","skipped_no_memory","skipped_no_match","skipped_capacity"); out={}
        for name,row in (("birth",self.birth_accounting),("reuse",self.reuse_accounting)):
            rhs=sum(int(row[x]) for x in fields); out[name]={"due":int(row["due"]),"accounted":int(rhs),"conserved":int(row["due"])==int(rhs),"not_yet_eligible":int(row["not_yet_eligible"]),**{x:int(row[x]) for x in fields}}
        return out


class V2CProtocol024Session(V2BProtocol024Session):
    """Production v2c session; tests/workflow import this exact implementation."""
    def __init__(self,*args,guard_anchor=None,v2c_config=None,**kwargs):
        cfg=dict(V2C_DEFAULT); cfg.update(v2c_config or {}); super().__init__(*args,guard_anchor=guard_anchor,v2b_config=cfg,**kwargs)
        self.lifecycle_controller=ReplacementConsistentLifecycle(guard_anchor=guard_anchor,config=cfg); self.lifecycle_state=self.lifecycle_controller.state_dict(); self.lifecycle_state["enabled"]=True; self._apply_specialist_freeze(); self.lifecycle_controller._sync_session(self)
    def update(self,*args,**kwargs):
        before=self.lifecycle_controller.frozen_specialist_hash(self); result=super().update(*args,**kwargs); after=self.lifecycle_controller.frozen_specialist_hash(self)
        if before!=after: raise AssertionError("frozen specialist changed during common live update")
        self.lifecycle_controller.freeze_checks["live_update"] += 1; return result
