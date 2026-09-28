"""Run the single registered Protocol-034 D_live full training replay."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import psutil
import torch

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
from run_ftmoe_protocol033 import validate_input
from ftmoe_protocol033_guard_budget import P033_CONFIG
from ftmoe_protocol034_live_memory import (
    GENERALISTS,
    PredictionMemoryObserver,
    Protocol034LiveSession,
    active_specialist_hash,
    dormant_state_hash,
    optimizer_names,
)


def _capture_py38(self, session, trigger, creation_prediction_index, specialist_id=None):
    bank = session.model.learner
    ctrl = session.lifecycle_controller
    key = str(specialist_id if specialist_id is not None else ctrl.active_specialist_id)
    if key in ("None", "") or key not in bank.experts:
        self.capture_log.append({"trigger": str(trigger), "creation_prediction_index": int(creation_prediction_index),
                                 "captured": False, "reason": "no_stable_resident_active_specialist"})
        return None
    ids = list(GENERALISTS) + [key]
    experts = {k: self._clone_expert(bank.experts[k]) for k in ids}
    weights = {k: bank.router_weights[k].detach().cpu().clone() for k in ids}
    biases = {k: bank.router_biases[k].detach().cpu().clone() for k in ids}
    digest = self._snapshot_hash(experts, weights, biases, ids, self.base_hash)
    dedup_key = (key, digest)
    if dedup_key in self._dedup:
        self.capture_log.append({"trigger": str(trigger), "creation_prediction_index": int(creation_prediction_index),
                                 "captured": False, "reason": "deduplicated_same_lineage_and_state",
                                 "lineage_id": key, "state_sha256": digest})
        return None
    self._dedup.add(dedup_key); self._seq += 1
    sid = "snapshot_%03d" % self._seq
    parameter_bytes = 0
    for k in ids:
        parameter_bytes += sum(int(p.numel() * p.element_size()) for p in experts[k].parameters())
        parameter_bytes += int(weights[k].numel() * weights[k].element_size())
        parameter_bytes += int(biases[k].numel() * biases[k].element_size())
    record = {
        "snapshot_id": sid, "lineage_id": key, "trigger": str(trigger),
        "creation_prediction_index": int(creation_prediction_index),
        "first_usable_prediction_index": int(creation_prediction_index) + 1,
        "source_phase_id_audit_only": p31.phase_name(max(0, int(creation_prediction_index)), self.phase_defs),
        "source_phase_used_for_selection": False, "state_sha256": digest,
        "base_reference_sha256": self.base_hash, "parameter_bytes": int(parameter_bytes),
        "full_definition": "frozen four-generalist plus one-specialist residual predictor; immutable base referenced by hash",
        "fragment_definition": "frozen specialist/router row plus current preprediction D_live generalists/router rows; unit ramps",
        "experts": experts, "weights": weights, "biases": biases,
        "full_probability": np.full((self.steps, self.hosts), np.nan, np.float32),
        "full_class_probability": np.full((self.steps, self.hosts, 3), np.nan, np.float32),
        "fragment_probability": np.full((self.steps, self.hosts), np.nan, np.float32),
        "fragment_class_probability": np.full((self.steps, self.hosts, 3), np.nan, np.float32),
    }
    self.snapshots.append(record); self._by_id[sid] = record
    log_entry = {k: p31.json_ready(v) for k, v in record.items()
                 if k not in ("experts", "weights", "biases", "full_probability", "full_class_probability",
                              "fragment_probability", "fragment_class_probability")}
    log_entry["captured"] = True; self.capture_log.append(log_entry)
    return sid

PredictionMemoryObserver.capture = _capture_py38


class InstrumentedProtocol034Session(Protocol034LiveSession):
    def __init__(self, *args, **kwargs):
        self.live_specialist_update_events = []
        super().__init__(*args, **kwargs)

    def update(self, t):
        ctrl = self.lifecycle_controller
        active = None if ctrl.active_specialist_id is None else str(ctrl.active_specialist_id)
        stable = active is not None and ctrl.transition is None
        before = active_specialist_hash(self) if stable else None
        d_before = dormant_state_hash(self); names_before = optimizer_names(self)
        result = super().update(t)
        after = active_specialist_hash(self) if stable else None
        d_after = dormant_state_hash(self); names_after = optimizer_names(self)
        if d_before != d_after: raise AssertionError("Protocol034 dormant hash changed across update")
        self.live_specialist_update_events.append({
            "at_interval": int(t), "matured_count": int(ctrl.matured_count),
            "stable_active_specialist_id": active if stable else None,
            "active_hash_before": before, "active_hash_after": after,
            "active_specialist_changed": bool(stable and before is not None and before != after),
            "optimizer_names_before": names_before, "optimizer_names_after": names_after,
            "transition_active": bool(ctrl.transition is not None), "dormant_hash_unchanged": True})
        return result


def _rss_tree():
    proc = psutil.Process(); total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try: total += child.memory_info().rss
        except Exception: pass
    return int(total)


def run_live(stream, out_dir, registration, input_lock, run_id):
    p31.deterministic_runtime(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    stream_sha, manifest, lock, reg = validate_input(stream, registration, input_lock)
    bundle = s4.build_replay(Path(stream)); pdefs = p31.phases(manifest)
    if int(bundle["steps"]) != 5968: raise AssertionError("Protocol034 scored-row count mismatch")
    guard = p31.build_guard(bundle)
    guard["meta"].update({"protocol":"034","plan_revision":1,
        "role":"F0_known_normal_regression_guard_inherited_from_033",
        "normal_nll_rule":"candidate <= live + max(0.02*live,0.01) + 1e-6",
        "candidate_training_on_guard_rows":False})
    p31.write_json(out/"guard_manifest.json", guard["meta"])
    runtime_registration = {"protocol":"034","plan_revision":1,"comparator":"D_live",
        "source_protocol":"033","source_run_id":36417604442,"scenario_id":reg["scenario_id"],
        "data_revision":reg["data_revision"],"stream_sha256":stream_sha,"replay_seed":700,"model_seed":1,
        "development_only":True,"statistical_confirmation":False,"input_lock_sha256":p31.sha(input_lock),
        "birth_start_matured":600,"birth_every_matured":1000,"snapshot_observer_learning_feedback":False}
    common = dict(seed=1,replay_bundle=bundle,budget=p31.budget(),out_dir=out,
                  run_id="protocol034_%s_D_live"%run_id,stream_dir=Path(stream),phase_defs=pdefs,
                  stream_sha=stream_sha,registration=runtime_registration,learning_rate=1e-4)
    session = InstrumentedProtocol034Session(guard_anchor=guard,v2c_config=P033_CONFIG,**common)
    init_prefix = p31.shared_prefix_hash(session.model,4)
    observer = PredictionMemoryObserver(session.steps,int(session.predictions["probability"].shape[1]),
                                        pdefs,session.model.frozen_hash())
    session.attach_snapshot_observer(observer)
    p31.write_json(out/"initialization.json",{"shared_first4_expert_and_router_rows_sha256":init_prefix,
        "model_seed":1,"registered_shared_prefix_initialization":True,"observer_attached_before_first_prediction":True})
    vm=psutil.virtual_memory(); effective=int(vm.total); soft=int(effective*0.8); mem=[]
    wall0=time.perf_counter(); cpu0=time.process_time(); peak_p=p31.param_bytes(session); peak_o=p31.opt_bytes(session)
    peak_live=len(session.model.learner.ids); peak_res=session.model.learner.resident_count()
    for i in range(session.steps):
        session.step()
        if i%20==0 or i+1==session.steps:
            rss=_rss_tree(); mem.append({"cursor":i,"process_tree_rss_bytes":rss,"system_available_bytes":int(psutil.virtual_memory().available)})
            if rss>=soft: raise MemoryError("Protocol034 D_live crossed 80% memory soft limit")
        if i%64==0 or i+1==session.steps:
            peak_p=max(peak_p,p31.param_bytes(session)); peak_o=max(peak_o,p31.opt_bytes(session))
        bank=session.model.learner; peak_live=max(peak_live,len(bank.ids)); peak_res=max(peak_res,bank.resident_count())
        if bank.resident_count()>8: raise AssertionError("Protocol034 resident capacity exceeded")
    session.lifecycle_controller.mark_stream_end(); session.finish(); session.save(); observer.save(out)
    wall=time.perf_counter()-wall0; cpu=time.process_time()-cpu0
    prob=session.predictions["probability"]; cls=session.predictions["class_probability"]; y=session.predictions["labels"]
    ctrl=session.lifecycle_controller; bank=session.model.learner; events=list(ctrl.events); records=list(ctrl.candidate_records.values())
    due=ctrl.due_conservation()
    if not due["birth"]["conserved"] or not due["reuse_nonblocking"]["due_conserved"] or not due["reuse_nonblocking"]["started_conserved"]:
        raise AssertionError("Protocol034 lifecycle accounting not conserved")
    stable=[x for x in session.live_specialist_update_events if x["stable_active_specialist_id"] is not None]
    changed=[x for x in stable if x["active_specialist_changed"]]
    if stable and not changed: raise AssertionError("Protocol034 active specialist never changed on stable update")
    if not observer.isolation_checks: raise AssertionError("Protocol034 observer isolation check missing")
    cost={"wall_seconds":float(wall),"cpu_seconds":float(cpu),"max_rss_bytes":max(x["process_tree_rss_bytes"] for x in mem),
          "effective_memory_limit_bytes":effective,"memory_soft_limit_bytes":soft,
          "p95_inference_seconds":float(np.percentile(session.predictions["prediction_seconds"],95)),
          "online_update_opportunities_completed":int(session.updates),"online_sample_draws":p31.samples(session),
          "final_resident_parameter_bytes":p31.param_bytes(session),"peak_resident_parameter_bytes":int(peak_p),
          "final_optimizer_state_bytes":p31.opt_bytes(session),"peak_optimizer_state_bytes":int(peak_o),
          "peak_live_expert_count":int(peak_live),"peak_resident_expert_count":int(peak_res)}
    p31.write_json(out/"memory_profile.json",{"protocol":"034","samples_every_intervals":20,"samples":mem,
        "effective_memory_limit_bytes":effective,"soft_limit_bytes":soft,"peak_process_tree_rss_bytes":cost["max_rss_bytes"]})
    lifecycle={"birth_candidates_created":len(records),"retirements":int(ctrl.retirements),"reactivations":int(ctrl.reactivations),
        "purges":int(ctrl.purges),"accepted_birth_ids":[str(x) for x in ctrl.accepted_ids],
        "memory_registry_ids":sorted([str(k) for k in ctrl.specialist_memory.keys()],key=int),
        "real_dormant_expert_ids":sorted([str(k) for k in bank.dormant_experts.keys()],key=int),
        "active_specialist_id":ctrl.active_specialist_id,"reuse_outcomes":ctrl.reuse_outcomes,"due_accounting":due,
        "final_topology":bank.topology_manifest(),"optimizer_transfer_events":session.optimizer_transfer_events,
        "stable_active_update_count":len(stable),"stable_active_parameter_change_update_count":len(changed),
        "dormant_hash_final":dormant_state_hash(session)}
    summary={"protocol":"034","plan_revision":1,"comparator":"D_live","run_id":str(run_id),"completed":True,
        "development_only":True,"statistical_confirmation":False,"stream_sha256":stream_sha,
        "initialization_shared_prefix_sha256":init_prefix,"manifest":session.comparator_manifest(),
        "full":p31.metrics(prob,cls,y),"phases":[],"cost":cost,"snapshot_count":len(observer.snapshots),"lifecycle":lifecycle}
    for phase in pdefs:
        a,b=int(phase["start"]),int(phase["end"]); yy=y[a:b]
        row={"phase":phase["name"],"service":phase["regime"],"intervals":[a,b],
             "positive_host_steps":int((yy>0).sum()),"negative_host_steps":int((yy==0).sum())}
        row.update(p31.metrics(prob[a:b],cls[a:b],yy)); summary["phases"].append(row)
    p31.write_jsonl(out/"lifecycle.jsonl",events); p31.write_json(out/"candidate_records.json",records)
    p31.write_json(out/"reuse_records.json",ctrl.reuse_records); p31.write_json(out/"reuse_event_records.json",ctrl.p033_reuse_event_records)
    p31.write_json(out/"opportunity_accounting.json",due); p31.write_json(out/"lifecycle_summary.json",lifecycle)
    p31.write_json(out/"live_specialist_update_events.json",session.live_specialist_update_events)
    p31.write_json(out/"optimizer_transfer_events.json",session.optimizer_transfer_events); p31.write_json(out/"summary.json",summary)
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stream",required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--registration",required=True); ap.add_argument("--input-lock",required=True); ap.add_argument("--run-id",required=True)
    a=ap.parse_args()
    try:
        result=run_live(a.stream,a.out_dir,a.registration,a.input_lock,a.run_id)
        print(json.dumps(p31.json_ready({"completed":True,"snapshot_count":result["snapshot_count"],"full_detection":result["full"]["detection"]}),indent=2))
    except Exception:
        root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=True)
        p31.write_json(root/"runner_exception.json",{"protocol":"034","comparator":"D_live","traceback":traceback.format_exc(),"no_automatic_extra_replay":True})
        raise

if __name__=="__main__": main()
