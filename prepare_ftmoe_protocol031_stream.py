"""Protocol-031 revision003 rare-recurrence data utilities.

This module reuses the pinned Protocol-025 simulator/workload physics but owns
the Protocol-031 timeline, audit, manifest and freeze semantics. No historical
Protocol-027 stream hash or nine-window gate is imported.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

import prepare_ftmoe_protocol025_stream as P25
from simulator.workload.BitbrainWorkloadProtocol025 import (
    SERVICE_LAWS, SERVICE_MECHANISM_IDS, FORBIDDEN_MODEL_INPUTS,
)

ROOT=Path(__file__).resolve().parent
REGISTRATION_PATH=ROOT/"artifacts/ftmoe_online/protocol_031/scenario_registration.json"
SCENARIO_ID="protocol031_rare_recurrence_v2"
DATA_REVISION="protocol031_data_revision_002"
PLAN_REVISION=3
REGISTERED_SEED=700
SOURCE_SERVICE={"U":"S1","V":"S3","W":"S4"}
LOGICAL_BY_SOURCE={v:k for k,v in SOURCE_SERVICE.items()}
COMMON_FEATURE_ORDER=[
    "cpu_pressure","ram_pressure","disk_pressure",
    "cpu_delta","ram_delta","disk_delta",
    "cpu_slope","ram_slope","disk_slope",
]

# Reuse only generic byte/chunk/simulator support; scientific audit below is 031-owned.
ROOT25=P25.ROOT
configure=P25.configure
guard=P25.guard
assert_no_other_experiment=P25.assert_no_other_experiment
SCENARIO_PATH=P25.SCENARIO_PATH
DRIFT_CONFIG=P25.DRIFT_CONFIG
FAMILIAR_PHASE=P25.FAMILIAR_PHASE
COHORT=P25.COHORT
Protocol025ServiceTurnoverBWGD2=P25.Protocol025ServiceTurnoverBWGD2
_allocate=P25._allocate
_chunk_name=P25._chunk_name
_save_chunk=P25._save_chunk
_verify_chunks=P25._verify_chunks
_load_chunks=P25._load_chunks
_checkpoint_paths=P25._checkpoint_paths
_common_features=P25._common_features
sha=P25.sha

def json_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def registration():
    reg=json.loads(REGISTRATION_PATH.read_text(encoding="utf8"))
    if reg.get("protocol")!="031" or int(reg.get("plan_revision",-1))!=PLAN_REVISION:
        raise AssertionError("Protocol031 registration revision mismatch")
    if reg.get("scenario_id")!=SCENARIO_ID or reg.get("data_revision")!=DATA_REVISION:
        raise AssertionError("Protocol031 scenario/data revision mismatch")
    if int(reg.get("replay_seed",-1))!=700 or int(reg.get("model_seed",-1))!=1:
        raise AssertionError("Protocol031 seed mismatch")
    steps=int(reg.get("scored_intervals",-1)); guard=int(reg.get("guard_intervals",-1))
    total=int(reg.get("total_intervals",steps+guard))
    if steps<=0 or guard!=1 or total!=steps+guard:
        raise AssertionError("Protocol031 registered length mismatch")
    if float(reg["generation"]["event_probability"])!=0.30:
        raise AssertionError("Protocol031 event probability changed")
    if int(reg["generation"]["chunk_intervals"])!=200:
        raise AssertionError("Protocol031 immutable chunk size changed")
    return reg

def phase_table(reg=None):
    reg=registration() if reg is None else reg
    rows=[]
    cursor=0
    for item in reg["timeline"]:
        start,end=int(item["start"]),int(item["end"])
        logical=item["service"]
        if start!=cursor or end-start!=int(item["length"]):
            raise AssertionError("Protocol031 timeline is not contiguous")
        source=None if logical=="baseline" else SOURCE_SERVICE[str(logical)]
        rows.append({
            "name":str(item["name"]),"start":start,"end":end,"length":end-start,
            "logical_service":None if logical=="baseline" else str(logical),
            "service":source,
            "event_probability":0.0 if source is None else float(reg["generation"]["event_probability"]),
            "kind":"baseline" if source is None else "service_response",
        })
        cursor=end
    if cursor!=int(reg["scored_intervals"]):
        raise AssertionError("Protocol031 timeline sum mismatch")
    return rows

def phase_at(t,phases):
    for i,p in enumerate(phases):
        if int(p["start"])<=int(t)<int(p["end"]):
            return i,p
    return len(phases)-1,phases[-1]

def summary_block(labels,ratio,host_event_any,start,end):
    y=labels[int(start):int(end)]
    r=ratio[int(start):int(end)]
    event=host_event_any[int(start):int(end)]
    return {
        "intervals":[int(start),int(end)],
        "host_steps":int(y.size),
        "positive_host_steps":int((y>0).sum()),
        "negative_host_steps":int((y==0).sum()),
        "positive_rate":float((y>0).mean()) if y.size else None,
        "class_counts":{str(k):int((y==k).sum()) for k in range(4)},
        "event_related_positive_host_steps":int(((y>0)&(event>0)).sum()),
        "peak_overload_ratio":float(r.max()) if r.size else None,
    }

def model_free_audit(reg,phases,arrays,workload,applied_switches):
    steps=int(reg["scored_intervals"])
    raw=np.asarray(arrays["raw_labels"],dtype=np.int64)
    ratio=np.asarray(arrays["overload_ratio"],dtype=np.float64)
    physical=np.where((ratio>1.0).any(-1),ratio.argmax(-1)+1,0)
    label_equal=bool(np.array_equal(raw,physical))
    common=_common_features(arrays["host_features"],arrays["capacities"])
    per_phase={}
    recurrence={}
    phase_coverage_ok=True
    for p in phases:
        block=summary_block(raw,ratio,arrays["audit_host_event_any"],p["start"],p["end"])
        block["logical_service"]=p["logical_service"]
        block["source_service"]=p["service"]
        feats=common[p["start"]:p["end"]]
        block["common_feature_abs_mean"]=np.mean(np.abs(feats),axis=(0,1)).tolist()
        per_phase[p["name"]]=block
        if p["logical_service"] is not None:
            if block["positive_host_steps"]<32 or block["negative_host_steps"]<32:
                phase_coverage_ok=False
        if "_rec" in p["name"]:
            y=raw[p["start"]:p["end"]]
            recurrence[p["name"]]={
                "intervals":[int(p["start"]),int(p["end"])],
                "positive_host_steps":int((y>0).sum()),
                "negative_host_steps":int((y==0).sum()),
                "positive_min_32":bool((y>0).sum()>=32),
                "negative_min_32":bool((y==0).sum()>=32),
                "ap_defined":bool((y>0).any() and (y==0).any()),
            }
    event_counts={}
    for logical,source in SOURCE_SERVICE.items():
        event_counts[logical]=int(sum(1 for e in workload.response_events if e.get("service_id")==source))
    source_param_match={}
    for logical,source in SOURCE_SERVICE.items():
        implemented=SERVICE_LAWS[source]["registered_parameters"]
        registered=reg["services"][logical]["parameters"]
        source_param_match[logical]=all(
            str(k) in registered and float(registered[k])==float(v)
            for k,v in implemented.items() if isinstance(v,(int,float))
        )
    age0_safe=all(
        float(SERVICE_LAWS[source]["cpu_shape"][0])==0.0
        and float(SERVICE_LAWS[source]["ram_shape"][0])==0.0
        and float(SERVICE_LAWS[source]["disk_shape"][0])==0.0
        for source in SOURCE_SERVICE.values()
    )
    schedules=np.asarray(arrays["schedules"][:steps],dtype=np.float64)
    caps=np.asarray(arrays["capacities"][:steps],dtype=np.float64)
    placements=np.concatenate([
        np.asarray(arrays["before_placement"][:steps]).reshape(-1),
        np.asarray(arrays["after_placement"][:steps]).reshape(-1),
    ])
    scheduler_ok=bool(np.isfinite(schedules).all() and
                      np.allclose(schedules.sum(-1),1.0,atol=1e-5) and
                      ((placements>=-1)&(placements<16)).all())
    capacity_ok=bool(np.isfinite(caps).all() and (caps>0).all())
    feature_finite=bool(np.isfinite(common[:steps]).all())
    guard_idx=np.arange(0,299,dtype=np.int64)
    guard_idx=guard_idx[guard_idx%5==0]
    guard_y=raw[guard_idx+1].reshape(-1)
    guard_normal=int((guard_y==0).sum())
    guard_positive=int((guard_y>0).sum())
    required_rec={"U_rec1","V_rec1","U_rec2","V_rec2","U_rec3","V_rec3"}
    rec_ok=(set(recurrence)==required_rec and all(
        x["positive_min_32"] and x["negative_min_32"] and x["ap_defined"]
        for x in recurrence.values()
    ))
    switches=[
        {"interval":int(x["interval"]),"phase":x["phase"],
         "source_service":x.get("service"),
         "logical_service":LOGICAL_BY_SOURCE.get(x.get("service"))}
        for x in applied_switches
    ]
    audit={
        "protocol":"031","plan_revision":PLAN_REVISION,"scenario_id":SCENARIO_ID,
        "data_revision":DATA_REVISION,"kind":"pre_model_data_audit",
        "model_results_seen":False,
        "causal_collection_order":[
            "host_features/demands/capacity","scheduler_decision",
            "simulationStep","post_totals/ratio/raw_label"
        ],
        "service_and_phase_ids_audit_only":True,
        "labels_recomputed_from_capacity_exceedance_equal":label_equal,
        "direct_label_assignment_from_service_id":False,
        "timeline_scored_intervals":int(sum(p["length"] for p in phases)),
        "per_phase":per_phase,
        "recurrence_first128_class_coverage":recurrence,
        "service_event_counts":event_counts,
        "short_trace_skips":{logical:int(workload.short_trace_skips[source])
                             for logical,source in SOURCE_SERVICE.items()},
        "registered_physical_parameters_match_implementation":source_param_match,
        "admission_age0_safe":age0_safe,
        "common_observable_features_finite":feature_finite,
        "common_feature_order":COMMON_FEATURE_ORDER,
        "scheduler_audit_pass":scheduler_ok,
        "capacity_audit_pass":capacity_ok,
        "normal_guard":{
            "prediction_indices":guard_idx.tolist(),
            "target":"same-host raw[t+1] inside F0",
            "normal_rows":guard_normal,"positive_rows":guard_positive,
            "positive_rows_required":False,
        },
        "applied_switches":switches,
    }
    gates={
        "label_recompute_exact":label_equal,
        "timeline_matches_registration":audit["timeline_scored_intervals"]==steps,
        "events_U_V_W":all(event_counts[x]>0 for x in ("U","V","W")),
        "physical_parameters_match":all(source_param_match.values()),
        "admission_age0_safe":age0_safe,
        "finite_causal_features":feature_finite,
        "scheduler_audit":scheduler_ok,
        "capacity_audit":capacity_ok,
        "all_nonbaseline_phases_min32_positive_and_negative":phase_coverage_ok,
        "six_recurrence_first128_min32_positive_and_negative":rec_ok,
        "normal_guard_available":guard_normal>0,
    }
    audit["gates"]=gates
    audit["audit_pass"]=bool(all(gates.values()))
    audit["reroll_after_failure_allowed"]=False
    return audit,common

def finalize(output,reg,phases,arrays,workload,chunks,applied_switches,
             scheduler_weight,started,reg_sha):
    import time
    output=Path(output)
    count=int(reg["scored_intervals"])+int(reg["guard_intervals"])
    if _verify_chunks(output,chunks)!=count:
        raise AssertionError("cannot finalize incomplete Protocol031 chunk sequence")
    assembled=_allocate(count)
    _load_chunks(output,assembled,chunks)
    stream=output/"stream.npz"
    if stream.exists():
        raise FileExistsError("refusing to overwrite final Protocol031 stream")
    np.savez_compressed(stream,**assembled,
                        overload_mask=(assembled["overload_ratio"]>1.0).astype(np.uint8))
    digest=sha(stream)
    audit,common=model_free_audit(reg,phases,assembled,workload,applied_switches)
    (output/"data_audit.json").write_text(
        json.dumps(audit,indent=2,allow_nan=False)+"\n",encoding="utf8")
    np.savez_compressed(output/"common_observable_features.npz",features=common)
    (output/"events.json").write_text(
        json.dumps(workload.response_events,indent=2,allow_nan=False)+"\n",encoding="utf8")
    source_law=ROOT/"simulator/workload/BitbrainWorkloadProtocol025.py"
    manifest={
        "protocol":"031","plan_revision":PLAN_REVISION,"scenario_id":SCENARIO_ID,
        "data_revision":DATA_REVISION,"seed":700,
        "steps":int(reg["scored_intervals"]),"guard_rows":int(reg["guard_intervals"]),
        "stream_file":"stream.npz","stream_sha256":digest,
        "registration_sha256":reg_sha,"timeline":phases,
        "logical_to_source_service":SOURCE_SERVICE,
        "source_service_laws":{k:SERVICE_LAWS[v] for k,v in SOURCE_SERVICE.items()},
        "event_probability":float(reg["generation"]["event_probability"]),
        "forbidden_model_inputs":list(FORBIDDEN_MODEL_INPUTS),
        "audit_only_npz_keys":[k for k in assembled if k.startswith("audit_")],
        "model_input_keys":["host_features","demands","schedules","capacities",
                            "creation_ids","before_placement"],
        "common_observable_features_file":"common_observable_features.npz",
        "common_observable_feature_order":COMMON_FEATURE_ORDER,
        "label_key":"raw_labels",
        "label_rule":"argmax(CPU,RAM,Disk aggregate/capacity) when any ratio>1 else 0",
        "scheduler_checkpoint":str(scheduler_weight),
        "scheduler_checkpoint_sha256":sha(scheduler_weight),
        "source_law_file":str(source_law.relative_to(ROOT)),
        "source_law_sha256":sha(source_law),
        "scenario_adapter_sha256":sha(SCENARIO_PATH),
        "drift_config_sha256":sha(DRIFT_CONFIG),
        "chunk_manifest":chunks,
        "generation_elapsed_seconds":float(time.perf_counter()-started),
        "data_audit_file":"data_audit.json","audit_pass":bool(audit["audit_pass"]),
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf8")
    return manifest,audit
