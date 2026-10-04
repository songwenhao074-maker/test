"""Protocol-036 seed-aware stream generation, assembly and pre-model audit.

Reuses the frozen Protocol-031/033 physical simulator loop while replacing only
protocol-facing registration/checkpoint/finalization hooks. Historical files are
never modified. Each registered seed is generated once and resumed from exact
simulator/RNG state across <=200-row segments.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, random, sys, time, traceback
from pathlib import Path

import dill
import numpy as np
import psutil

import prepare_ftmoe_protocol033_stream as P33
import maintenance.run_protocol031_segment as BASE

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "artifacts/ftmoe_online/protocol_036/plan.json"
OLD_REG = ROOT / "artifacts/ftmoe_online/protocol_033/scenario_registration.json"
PLAN_SHA_EXPECTED = "339980818a4d0d7518b6a33e696bcfa02ea3ec27d69467d4242264277c051a47"
ALLOWED_SEEDS = (3601, 3602, 3603)
PLAN_REVISION = 1
SOURCE_SERVICE = dict(P33.SOURCE_SERVICE)
COMMON_FEATURE_ORDER = list(P33.COMMON_FEATURE_ORDER)

# Frozen physical/simulator support.
configure = P33.configure
guard = P33.guard
assert_no_other_experiment = P33.assert_no_other_experiment
SCENARIO_PATH = P33.SCENARIO_PATH
DRIFT_CONFIG = P33.DRIFT_CONFIG
FAMILIAR_PHASE = P33.FAMILIAR_PHASE
COHORT = P33.COHORT
Protocol025ServiceTurnoverBWGD2 = P33.Protocol025ServiceTurnoverBWGD2
SERVICE_LAWS = P33.SERVICE_LAWS
SERVICE_MECHANISM_IDS = P33.SERVICE_MECHANISM_IDS
FORBIDDEN_MODEL_INPUTS = P33.FORBIDDEN_MODEL_INPUTS
ROOT25 = P33.ROOT25
_allocate = P33._allocate
_chunk_name = P33._chunk_name
_save_chunk = P33._save_chunk
_verify_chunks = P33._verify_chunks
_checkpoint_paths = P33._checkpoint_paths
_common_features = P33._common_features
sha = P33.sha

REGISTRATION_PATH = None
_MEMORY = None


def J(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def W(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def append_generation_event(output, event):
    p=Path(output)/"generation_ledger.jsonl"; p.parent.mkdir(parents=True,exist_ok=True)
    row={"protocol":"036","time_unix":time.time(),**event}
    with p.open("a",encoding="utf8") as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")


def json_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    if sha(PLAN) != PLAN_SHA_EXPECTED:
        raise AssertionError("Protocol036 plan hash changed")
    p = J(PLAN)
    if p.get("protocol") != "036" or int(p.get("revision", -1)) != 1:
        raise AssertionError("Protocol036 plan identity mismatch")
    return p


def set_registration_path(path):
    global REGISTRATION_PATH
    REGISTRATION_PATH = Path(path).resolve()


def make_registration(path, seed):
    seed = int(seed)
    if seed not in ALLOWED_SEEDS:
        raise ValueError("unregistered Protocol036 replay seed")
    p = plan(); reg = J(OLD_REG)
    reg["protocol"] = "036"
    reg["plan_revision"] = 1
    reg["scenario_id"] = f"protocol036_rare_recurrence_5969_seed{seed}_v1"
    reg["data_revision"] = f"protocol036_data_seed{seed}_revision_001"
    reg["status"] = "registered_runtime_pre_generation"
    reg["registered_before_generation"] = True
    reg["source_code_commit"] = p["code_base_commit"]
    reg["purpose"] = "Protocol036 fixed new-stream robustness replicate; physical laws unchanged from Protocol033"
    reg["replay_seed"] = seed
    reg["model_seed"] = int(p["data"]["model_seed"])
    reg["generation"]["full_stream_budget"] = 1
    reg["generation"]["expected_stream_sha256"] = None
    reg["generation"]["reroll_after_audit_failure"] = False
    reg["generation"]["protocol036_plan_sha256"] = PLAN_SHA_EXPECTED
    reg["interpretation"] = {
        "protocol036_new_stream": True,
        "paired_methods_share_one_C_trajectory_per_stream": True,
        "not_cross_model_initialization_confirmation": True,
        "no_dynamic_memory_claim": True,
    }
    W(path, reg)
    return reg


def registration():
    if REGISTRATION_PATH is None:
        raise RuntimeError("Protocol036 registration path not configured")
    reg = J(REGISTRATION_PATH); p = plan()
    seed = int(reg.get("replay_seed", -1))
    if reg.get("protocol") != "036" or int(reg.get("plan_revision", -1)) != 1:
        raise AssertionError("Protocol036 registration identity mismatch")
    if seed not in p["budget"]["stage_B_new_stream_seeds"]:
        raise AssertionError("Protocol036 unregistered stream seed")
    if int(reg.get("model_seed", -1)) != int(p["data"]["model_seed"]):
        raise AssertionError("Protocol036 model seed mismatch")
    if int(reg.get("scored_intervals", -1)) != 5968 or int(reg.get("total_intervals", -1)) != 5969:
        raise AssertionError("Protocol036 row geometry mismatch")
    g = reg["generation"]
    if float(g["event_probability"]) != float(p["data"]["event_probability"]):
        raise AssertionError("Protocol036 event probability changed")
    if int(g["chunk_intervals"]) != 200 or int(g["segment_max_intervals"]) != 200:
        raise AssertionError("Protocol036 chunk geometry changed")
    if int(g["expected_chunk_count"]) != 30 or int(g["final_chunk_intervals"]) != 169:
        raise AssertionError("Protocol036 registered chunk count changed")
    old = J(OLD_REG)
    # Exact physical timeline/services/causality must remain identical.
    for key in ("timeline", "services", "service_implementation_source", "source_law_revision", "causality"):
        if reg.get(key) != old.get(key):
            raise AssertionError("Protocol036 physical/causal registration changed: " + key)
    return reg


def phase_table(reg=None):
    reg = registration() if reg is None else reg
    rows=[]; cursor=0
    for item in reg["timeline"]:
        start,end=int(item["start"]),int(item["end"]); logical=item["service"]
        if start != cursor or end-start != int(item["length"]):
            raise AssertionError("Protocol036 timeline not contiguous")
        source = None if logical == "baseline" else SOURCE_SERVICE[str(logical)]
        rows.append({"name":str(item["name"]),"start":start,"end":end,"length":end-start,
                     "logical_service":None if logical=="baseline" else str(logical),"service":source,
                     "event_probability":0.0 if source is None else float(reg["generation"]["event_probability"]),
                     "kind":"baseline" if source is None else "service_response"})
        cursor=end
    if cursor != 5968: raise AssertionError("Protocol036 timeline sum mismatch")
    return rows


def _cgroup_limit_bytes():
    vals=[]
    for p in (Path("/sys/fs/cgroup/memory.max"),Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            t=p.read_text().strip()
            if t and t!="max":
                v=int(t)
                if 0<v<(1<<60): vals.append(v)
        except Exception: pass
    return min(vals) if vals else None


def _tree_rss_bytes():
    proc=psutil.Process(); total=proc.memory_info().rss
    for child in proc.children(recursive=True):
        try: total += child.memory_info().rss
        except Exception: pass
    return int(total)


def init_memory_monitor(output):
    global _MEMORY
    vm=psutil.virtual_memory(); cg=_cgroup_limit_bytes(); effective=min(int(vm.total),int(cg)) if cg else int(vm.total)
    _MEMORY={"protocol":"036","stage":"generation","seed":int(registration()["replay_seed"]),
             "effective_memory_limit_bytes":effective,"cgroup_memory_limit_bytes":cg,
             "system_total_bytes":int(vm.total),"soft_limit_fraction":.8,"soft_limit_bytes":int(.8*effective),
             "sample_every_intervals":20,"samples":[],"soft_limit_crossed":False,"started_at_unix":time.time()}
    sample_memory("generation_start",0); return _MEMORY


def sample_memory(label, step=None):
    if _MEMORY is None: return None
    vm=psutil.virtual_memory(); rss=_tree_rss_bytes(); row={"label":str(label),"step":None if step is None else int(step),
        "time_unix":time.time(),"process_tree_rss_bytes":rss,"system_available_bytes":int(vm.available)}
    _MEMORY["samples"].append(row)
    if rss>=int(_MEMORY["soft_limit_bytes"]): _MEMORY["soft_limit_crossed"]=True
    return row


def write_memory_report(output):
    if _MEMORY is None: return
    r=dict(_MEMORY); s=r["samples"]; r["peak_process_tree_rss_bytes"]=max((x["process_tree_rss_bytes"] for x in s),default=0)
    r["minimum_system_available_bytes"]=min((x["system_available_bytes"] for x in s),default=None)
    W(Path(output)/"memory_profile_generation.json",r)


def phase_at(t, phases):
    if int(t)%20==0: sample_memory("interval",int(t))
    for i,p in enumerate(phases):
        if int(p["start"])<=int(t)<int(p["end"]): return i,p
    return len(phases)-1,phases[-1]


def source_identity():
    files=[PLAN,ROOT/"protocol036_stream.py",ROOT/"maintenance/run_protocol031_segment.py",
           ROOT/"prepare_ftmoe_protocol033_stream.py",ROOT/"prepare_ftmoe_protocol031_stream.py",
           ROOT/"simulator/workload/BitbrainWorkloadProtocol025.py",SCENARIO_PATH,DRIFT_CONFIG]
    return {str(Path(p).relative_to(ROOT)):sha(p) for p in files if Path(p).is_file()}


def write_checkpoint(output,reg_sha,next_t,sim_state,chunks,arrays,active_service,active_probability,applied_switches):
    output=Path(output); sample_memory("checkpoint_before",next_t)
    state_path,manifest_path=_checkpoint_paths(output); tmp=state_path.with_suffix(".tmp")
    reg=registration(); payload={"protocol":"036","plan_revision":1,"scenario_id":reg["scenario_id"],
        "data_revision":reg["data_revision"],"replay_seed":int(reg["replay_seed"]),"registration_sha256":reg_sha,
        "source_sha256":source_identity(),"next_t":int(next_t),"sim_state":sim_state,
        "active_service":active_service,"active_probability":float(active_probability),"applied_switches":list(applied_switches),
        "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":sim_state["torch"].get_rng_state()}}
    with tmp.open("wb") as f: dill.dump(payload,f,protocol=dill.HIGHEST_PROTOCOL)
    os.replace(tmp,state_path); chunk_end=int(_verify_chunks(output,chunks)); next_t=int(next_t)
    if next_t<chunk_end: raise AssertionError("Protocol036 transient checkpoint precedes immutable chunk end")
    transient_path=output/"transient_partial.npz"; transient=None
    if next_t>chunk_end:
        np.savez_compressed(transient_path,**{k:v[chunk_end:next_t] for k,v in arrays.items()})
        transient={"file":transient_path.name,"start":chunk_end,"end":next_t,"sha256":sha(transient_path)}
    elif transient_path.exists(): transient_path.unlink()
    manifest={"protocol":"036","plan_revision":1,"scenario_id":reg["scenario_id"],"data_revision":reg["data_revision"],
        "replay_seed":int(reg["replay_seed"]),"registration_sha256":reg_sha,"source_sha256":source_identity(),
        "next_t":next_t,"state_file":state_path.name,"state_sha256":sha(state_path),"chunks":list(chunks),
        "engineering_transient_checkpoint":transient,"registered_chunk_semantics_unchanged":True}
    W(manifest_path,manifest); sample_memory("checkpoint_after",next_t); write_memory_report(output); return manifest


def read_checkpoint(output,reg_sha,arrays):
    output=Path(output); state_path,manifest_path=_checkpoint_paths(output)
    if not(state_path.is_file() and manifest_path.is_file()): raise FileNotFoundError("Protocol036 resume checkpoint absent")
    manifest=J(manifest_path); reg=registration()
    if manifest.get("protocol")!="036" or manifest.get("scenario_id")!=reg["scenario_id"]: raise AssertionError("wrong Protocol036 checkpoint identity")
    if manifest.get("registration_sha256")!=reg_sha or manifest.get("source_sha256")!=source_identity(): raise AssertionError("Protocol036 registration/source changed")
    if sha(state_path)!=manifest.get("state_sha256"): raise AssertionError("Protocol036 state digest mismatch")
    chunks=list(manifest["chunks"]); chunk_end=int(_verify_chunks(output,chunks)); next_t=int(manifest["next_t"])
    transient=manifest.get("engineering_transient_checkpoint")
    if transient is None:
        if chunk_end!=next_t: raise AssertionError("Protocol036 resume chunk boundary mismatch")
    else:
        start,end=int(transient["start"]),int(transient["end"]); path=output/transient["file"]
        if start!=chunk_end or end!=next_t or end<=start or not path.is_file() or sha(path)!=transient["sha256"]:
            raise AssertionError("invalid Protocol036 transient checkpoint")
        with np.load(path,allow_pickle=False) as d:
            for key in arrays: arrays[key][start:end]=d[key]
    with state_path.open("rb") as f: payload=dill.load(f)
    if payload.get("protocol")!="036" or payload.get("registration_sha256")!=reg_sha or int(payload["next_t"])!=next_t:
        raise AssertionError("wrong Protocol036 resume payload")
    if payload.get("source_sha256")!=source_identity(): raise AssertionError("Protocol036 source hash mismatch")
    random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
    payload["sim_state"]["torch"].set_rng_state(payload["rng"]["torch"])
    return payload,chunks,chunk_end


def initialize(output,reg,reg_sha,arrays):
    import torch
    from simulator.Simulator import Simulator
    from simulator.environment.RPiEdge import RPiEdge
    from simulator.environment.RPiCapacity import RPiCapacity
    from scheduler.GOBI import GOBIScheduler
    from recovery.Recovery import Recovery
    from stats.Stats import Stats
    from src.constants import MODEL_SAVE_PATH
    seed=int(reg["replay_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    scenario=J(SCENARIO_PATH); drift=J(DRIFT_CONFIG); familiar=next(x for x in drift["phases"] if x["name"]==FAMILIAR_PHASE)
    adapter=dict(scenario.get("adapter") or {}); adapter.update(familiar.get("adapter") or {})
    law_rel=scenario.get("disk_law_relative"); disk_law_path=(ROOT/law_rel).resolve() if law_rel else None
    dc=RPiEdge(16); workload=Protocol025ServiceTurnoverBWGD2(float(scenario.get("arrival_mean",1.0)),float(scenario.get("arrival_sigma",1.5)),seed,
        cohort=COHORT,adapter=adapter,disk_law_path=disk_law_path,active_service=None,event_probability=float(reg["generation"]["event_probability"]))
    scheduler=GOBIScheduler("energy_latency_16"); recovery=Recovery(); stats=Stats(workload,dc,scheduler)
    stats.feats_per_host=7; stats.history_limit=64; stats.series_tail=96
    env=Simulator(1000,10000,scheduler,recovery,stats,16,300,dc.generateHosts()); capacity=RPiCapacity(env.hostlist)
    capacity.apply(familiar["cpu_scale"],familiar["ram_scale"],familiar["disk_scale"])
    initial=workload.generateNewContainers(env.interval); deployed=env.addContainersInit(initial); decision=scheduler.placement(deployed); migrations=env.allocateInit(decision)
    workload.updateDeployedContainers(env.getCreationIDs(migrations,deployed)); stats.saveStats(deployed,migrations,[],deployed,decision,0)
    candidates=[ROOT/str(MODEL_SAVE_PATH)/"energy_latency_16_Trained.ckpt",ROOT/"scheduler/BaGTI"/str(MODEL_SAVE_PATH)/"energy_latency_16_Trained.ckpt"]
    scheduler_weight=next((x.resolve() for x in candidates if x.is_file()),None)
    if scheduler_weight is None: raise FileNotFoundError("GOBI trained checkpoint missing")
    sim_state={"env":env,"workload":workload,"scheduler":scheduler,"recovery":recovery,"stats":stats,"capacity":capacity,"torch":torch}
    write_checkpoint(output,reg_sha,0,sim_state,[],arrays,None,0.0,[]); return sim_state,scheduler_weight


def finalize(output,reg,phases,arrays,workload,chunks,applied_switches,scheduler_weight,started,reg_sha):
    output=Path(output); count=int(reg["total_intervals"]); end=int(_verify_chunks(output,chunks))
    if end!=count or len(chunks)!=30 or int(chunks[-1]["end"])-int(chunks[-1]["start"])!=169: raise AssertionError("Protocol036 incomplete chunk sequence")
    W(output/"events_final.json",workload.response_events); W(output/"applied_switches.json",applied_switches)
    generation={"protocol":"036","plan_revision":1,"scenario_id":reg["scenario_id"],"data_revision":reg["data_revision"],
        "replay_seed":int(reg["replay_seed"]),"complete":True,"row_target":count,"scored_target":5968,"chunk_count":len(chunks),
        "last_chunk_rows":169,"chunks":list(chunks),"registration_sha256":reg_sha,"source_sha256":source_identity(),
        "scheduler_checkpoint":str(scheduler_weight),"scheduler_checkpoint_sha256":sha(scheduler_weight),
        "generation_elapsed_seconds_this_process":float(time.perf_counter()-started),"assembly_pending":True,"input_audit_pending":True,
        "no_stream_npz_created_in_generation_process":True}
    W(output/"generation_manifest.json",generation); sample_memory("generation_complete",count); write_memory_report(output)
    return {"stream_sha256":None,**generation},{"audit_pass":False,"gates":{"input_audit_pending":True}}


def patch_base():
    BASE.P=sys.modules[__name__]
    BASE.initialize=initialize
    BASE.write_checkpoint=write_checkpoint
    BASE.read_checkpoint=read_checkpoint


def cmd_segment(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); regpath=out/"registration.json"
    if regpath.exists():
        set_registration_path(regpath); reg=registration()
        if int(reg["replay_seed"])!=int(a.seed): raise AssertionError("resume seed mismatch")
    else:
        make_registration(regpath,int(a.seed)); set_registration_path(regpath)
    snap=out/"registration_snapshot.json"
    if snap.exists() and snap.read_bytes()!=regpath.read_bytes(): raise AssertionError("Protocol036 registration snapshot changed")
    if not snap.exists(): snap.write_bytes(regpath.read_bytes())
    init_memory_monitor(out); patch_base(); before_next=None
    try:
        st=out/"generation_status.json"
        if st.exists():
            try: before_next=J(st).get("next_t")
            except Exception: before_next=None
        append_generation_event(out,{"event":"segment_start","seed":int(a.seed),"requested_max_intervals":int(a.max_intervals),"resume_from_next_t":before_next})
        result=BASE.collect_segment(out,int(a.max_intervals)); result=dict(result or {})
        result.update({"protocol":"036","seed":int(a.seed),"scenario_id":registration()["scenario_id"],"model_runs_started":0})
        if bool(result.get("complete")):
            result["stream_sha256"]=None; result["input_audit_pending"]=True; result["assembly_pending"]=True
            result.pop("audit_pass",None); result.pop("gates",None)
        W(out/"generation_status.json",result); append_generation_event(out,{"event":"segment_complete","seed":int(a.seed),"next_t":result.get("next_t"),"complete":bool(result.get("complete"))}); write_memory_report(out); print(json.dumps(result,indent=2)); return
    except Exception as exc:
        failure={"protocol":"036","seed":int(a.seed),"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),
                 "resume_available":_checkpoint_paths(out)[0].is_file(),"no_reroll":True,
                 "next_action":"repair deterministic engineering issue and resume this exact seed only from verified checkpoint"}
        W(out/"failure.json",failure); append_generation_event(out,{"event":"segment_exception","seed":int(a.seed),"error_type":type(exc).__name__,"error":str(exc)}); write_memory_report(out); raise


def cmd_assemble(a):
    src=Path(a.generation_root); out=Path(a.output); set_registration_path(src/"registration.json"); reg=registration()
    if out.exists() and any(out.iterdir()): raise FileExistsError("Protocol036 assembly output must be empty")
    out.mkdir(parents=True,exist_ok=True); field_dir=out/"fields"; field_dir.mkdir(); samples=[]
    def sample(label,chunk=None):
        vm=psutil.virtual_memory(); samples.append({"label":label,"chunk":chunk,"time_unix":time.time(),"process_tree_rss_bytes":_tree_rss_bytes(),"system_available_bytes":int(vm.available)})
    sample("assembly_start"); reg_sha=json_sha(REGISTRATION_PATH); gen=J(src/"generation_manifest.json"); resume=J(src/"resume_manifest.json")
    if gen.get("protocol")!="036" or resume.get("protocol")!="036": raise AssertionError("Protocol036 generation identity missing")
    if gen.get("registration_sha256")!=reg_sha or resume.get("registration_sha256")!=reg_sha: raise AssertionError("Protocol036 registration hash mismatch")
    if resume.get("source_sha256")!=source_identity(): raise AssertionError("Protocol036 source hash mismatch")
    chunks=list(resume["chunks"])
    if _verify_chunks(src,chunks)!=5969 or len(chunks)!=30 or int(chunks[-1]["end"])-int(chunks[-1]["start"])!=169: raise AssertionError("Protocol036 chunk coverage mismatch")
    with np.load(src/"chunks"/chunks[0]["file"],allow_pickle=False) as first:
        keys=list(first.files); schema={k:{"dtype":str(first[k].dtype),"tail_shape":list(first[k].shape[1:])} for k in keys}
        memmaps={k:np.lib.format.open_memmap(field_dir/(k+".npy"),mode="w+",dtype=first[k].dtype,shape=(5969,)+first[k].shape[1:]) for k in keys}
    for ci,rec in enumerate(chunks):
        aa,bb=int(rec["start"]),int(rec["end"])
        with np.load(src/"chunks"/rec["file"],allow_pickle=False) as z:
            if set(z.files)!=set(keys): raise AssertionError("Protocol036 chunk schema changed")
            for k in keys:
                expected=(bb-aa,)+tuple(schema[k]["tail_shape"])
                if z[k].shape!=expected or str(z[k].dtype)!=schema[k]["dtype"]: raise AssertionError("Protocol036 chunk shape/dtype mismatch "+k)
                memmaps[k][aa:bb]=z[k]
        for mm in memmaps.values(): mm.flush()
        sample("chunk_materialized",ci)
    common=_common_features(memmaps["host_features"],memmaps["capacities"]); np.save(out/"common_observable_features.npy",common); del common
    overload_mask=(memmaps["overload_ratio"]>1.0).astype(np.uint8); np.savez_compressed(out/"stream.npz",**memmaps,overload_mask=overload_mask); del overload_mask
    stream_sha=sha(out/"stream.npz"); phases=phase_table(reg)
    manifest={"protocol":"036","plan_revision":1,"scenario_id":reg["scenario_id"],"data_revision":reg["data_revision"],"seed":int(reg["replay_seed"]),
        "steps":5968,"guard_rows":1,"total_rows":5969,"stream_file":"stream.npz","stream_sha256":stream_sha,"registration_sha256":reg_sha,
        "timeline":phases,"logical_to_source_service":SOURCE_SERVICE,"event_probability":float(reg["generation"]["event_probability"]),
        "forbidden_model_inputs":list(FORBIDDEN_MODEL_INPUTS),"audit_only_npz_keys":[k for k in keys if k.startswith("audit_")]+["overload_mask"],
        "model_input_keys":["host_features","demands","schedules","capacities","creation_ids","before_placement"],
        "common_observable_features_file":"common_observable_features.npy","common_observable_feature_order":COMMON_FEATURE_ORDER,
        "common_feature_builder":"pressure[t], delta[t]=pressure[t]-pressure[t-1], slope[t]=(pressure[t]-pressure[t-4])/4; no future/full-stream statistics",
        "label_key":"raw_labels","label_rule":reg["causality"]["label_rule"],"target":reg["causality"]["target"],"target_maturity":reg["causality"]["target_maturity"],
        "chunk_manifest":chunks,"field_schema":schema,"assembled_in_fresh_process":True,"audit_pass":None}
    W(out/"manifest.json",manifest)
    for name in ("events_final.json","applied_switches.json","memory_profile_generation.json","generation_manifest.json","resume_manifest.json","registration_snapshot.json","registration.json"):
        p=src/name
        if p.is_file(): (out/name).write_bytes(p.read_bytes())
    sample("assembly_complete"); cg=_cgroup_limit_bytes(); vm=psutil.virtual_memory(); effective=min(int(vm.total),int(cg)) if cg else int(vm.total)
    W(out/"memory_profile_assembly.json",{"protocol":"036","seed":int(reg["replay_seed"]),"stage":"assembly","effective_memory_limit_bytes":effective,
        "soft_limit_bytes":int(.8*effective),"samples":samples,"peak_process_tree_rss_bytes":max(x["process_tree_rss_bytes"] for x in samples),
        "minimum_system_available_bytes":min(x["system_available_bytes"] for x in samples)})
    print(json.dumps({"protocol":"036","seed":int(reg["replay_seed"]),"assembled":True,"rows":5969,"chunks":30,"stream_sha256":stream_sha},indent=2))


def cmd_audit(a):
    genroot=Path(a.generation_root); data=Path(a.data_root); evidence=Path(a.evidence_root); evidence.mkdir(parents=True,exist_ok=True)
    set_registration_path(genroot/"registration.json"); reg=registration(); seed=int(reg["replay_seed"]); reg_sha=json_sha(REGISTRATION_PATH)
    manifest=J(data/"manifest.json"); resume=J(genroot/"resume_manifest.json"); generation=J(genroot/"generation_manifest.json"); chunks=list(resume.get("chunks") or [])
    gates={}; details={}; gates["registration_identity"]=bool(manifest.get("protocol")=="036" and manifest.get("seed")==seed and resume.get("protocol")=="036" and generation.get("protocol")=="036" and resume.get("registration_sha256")==reg_sha and resume.get("source_sha256")==source_identity())
    try: covered=int(_verify_chunks(genroot,chunks)); geometry=(covered==5969 and len(chunks)==30 and int(chunks[-1]["end"])-int(chunks[-1]["start"])==169)
    except Exception as exc: covered=-1; geometry=False; details["chunk_error"]=str(exc)
    gates["all_rows_and_chunks_complete"]=bool(geometry)
    field=data/"fields"; post=np.load(field/"post_totals.npy",mmap_mode="r"); caps=np.load(field/"capacities.npy",mmap_mode="r"); saved=np.load(field/"overload_ratio.npy",mmap_mode="r"); raw=np.load(field/"raw_labels.npy",mmap_mode="r"); schedules=np.load(field/"schedules.npy",mmap_mode="r"); before=np.load(field/"before_placement.npy",mmap_mode="r"); after=np.load(field/"after_placement.npy",mmap_mode="r"); host=np.load(field/"host_features.npy",mmap_mode="r"); audit_phase=np.load(field/"audit_phase_ids.npy",mmap_mode="r"); audit_service=np.load(field/"audit_service_ids.npy",mmap_mode="r")
    gates["exact_5969_row_geometry"]=all(int(x.shape[0])==5969 for x in (post,caps,saved,raw,schedules,before,after,host,audit_phase,audit_service))
    ratio_exact=label_exact=finite_features=capacity_ok=scheduler_ok=placement_ok=True
    for aa in range(0,5969,200):
        bb=min(5969,aa+200); c=np.asarray(caps[aa:bb],np.float64); po=np.asarray(post[aa:bb],np.float64); sr=np.asarray(saved[aa:bb],np.float64); y=np.asarray(raw[aa:bb],np.int64); calc=po/c; cy=np.where((calc>1.0).any(-1),calc.argmax(-1)+1,0)
        ratio_exact &= bool(np.array_equal(calc,sr)); label_exact &= bool(np.array_equal(cy,y)); finite_features &= bool(np.isfinite(np.asarray(host[aa:bb])).all()); capacity_ok &= bool(np.isfinite(c).all() and (c>0).all()); s=np.asarray(schedules[aa:bb],np.float64); scheduler_ok &= bool(np.isfinite(s).all() and (s>=-1e-7).all() and np.allclose(s.sum(-1),1.0,atol=1e-5)); bp=np.asarray(before[aa:bb]); ap=np.asarray(after[aa:bb]); placement_ok &= bool(((bp>=-1)&(bp<16)).all() and ((ap>=-1)&(ap<16)).all())
    gates.update({"overload_ratio_recomputed_exact":ratio_exact,"label_recompute_exact":label_exact,"finite_causal_features":finite_features,"capacity_audit":capacity_ok,"scheduler_audit":scheduler_ok,"placement_audit":placement_ok})
    common=np.load(data/"common_observable_features.npy",mmap_mode="r"); recomputed=_common_features(host,caps)
    gates["common_features_causal_formula_exact"]=bool(common.shape==(5969,16,9) and np.isfinite(common).all() and np.array_equal(np.asarray(common),recomputed))
    phases=phase_table(reg); recurrence={}; class_coverage=True; phase_ok=True; service_ok=True
    for pi,p in enumerate(phases):
        aa,bb=int(p["start"]),int(p["end"]); y=np.asarray(raw[aa+1:bb+1],np.int64); pos=int((y>0).sum()); neg=int((y==0).sum())
        if p["logical_service"] is not None: class_coverage &= pos>=32 and neg>=32
        if p["name"] in {"U_rec1","V_rec1","U_rec2","V_rec2","U_rec3","V_rec3"}: recurrence[p["name"]]={"positive_host_steps":pos,"negative_host_steps":neg,"ap_defined":pos>0 and neg>0}
        phase_ok &= bool((np.asarray(audit_phase[aa:bb])==pi).all()); sid=-1 if p["service"] is None else int(SERVICE_MECHANISM_IDS[p["service"]]); service_ok &= bool((np.asarray(audit_service[aa:bb])==sid).all())
    gates["all_nonbaseline_phases_min32_positive_and_negative_target_hoststeps"]=bool(class_coverage); gates["six_recurrence_windows_present_and_defined"]=bool(len(recurrence)==6 and all(x["ap_defined"] and x["positive_host_steps"]>=32 and x["negative_host_steps"]>=32 for x in recurrence.values())); gates["phase_annotation_matches_timeline"]=phase_ok; gates["service_annotation_matches_timeline"]=service_ok; gates["last_prediction_has_raw_t_plus_1_support"]=bool(raw.shape[0]==5969 and phases[-1]["end"]==5968)
    events=J(data/"events_final.json"); counts={logical:int(sum(1 for e in events if e.get("service_id")==source)) for logical,source in SOURCE_SERVICE.items()}; gates["events_U_V_W_present"]=all(counts[x]>0 for x in ("U","V","W"))
    switches=J(data/"applied_switches.json"); expected=[(int(p["start"]),p["name"],p["service"],p["logical_service"]) for p in phases[1:]]; actual=[(int(x["interval"]),x["phase"],x.get("service"),x.get("logical_service")) for x in switches]; gates["actual_phase_switches_match_registration"]=actual==expected
    source_match={}
    for logical,source in SOURCE_SERVICE.items():
        implemented=SERVICE_LAWS[source]["registered_parameters"]; registered=reg["services"][logical]["parameters"]; source_match[logical]=all(str(k) in registered and float(registered[k])==float(v) for k,v in implemented.items() if isinstance(v,(int,float)))
    gates["registered_physical_parameters_match_implementation"]=all(source_match.values())
    audit_pass=bool(all(gates.values())); stream_sha=sha(data/"stream.npz"); common_sha=sha(data/"common_observable_features.npy")
    audit={"protocol":"036","seed":seed,"kind":"pre_model_input_eligibility_audit","model_results_seen":False,"audit_pass":audit_pass,"reroll_after_failure_allowed":False,
           "future_statistic_preprocessing_used":False,"common_feature_causality":"pressure current row; delta past1; slope past4 only","gates":gates,"recurrence_target_coverage":recurrence,"service_event_counts":counts,"source_parameter_match":source_match,"details":details}
    W(evidence/"input_audit.json",audit)
    frozen=json.loads(json.dumps(reg)); frozen["status"]="frozen_pre_model" if audit_pass else "blocked_pre_model_audit"; frozen["generation"]["expected_stream_sha256"]=stream_sha; frozen["generation"]["actual_chunk_sha256"]=[x["sha256"] for x in chunks]; W(evidence/"registration_frozen.json",frozen); frsha=sha(evidence/"registration_frozen.json")
    lock={"protocol":"036","seed":seed,"locked":audit_pass,"input_audit_passed":audit_pass,"stream_sha256":stream_sha,"common_features_sha256":common_sha,
          "registration_original_sha256":reg_sha,"registration_frozen_sha256":frsha,"checkpoint_sha256":resume.get("state_sha256"),"checkpoint_next_t":resume.get("next_t"),"source_sha256":resume.get("source_sha256"),"chunk_sha256":[x["sha256"] for x in chunks],"chunk_count":len(chunks),"total_rows":5969,"scored_rows":5968,"reroll_after_failure":False}
    W(evidence/"input_lock.json",lock); W(data/"frozen_data_manifest.json",{"protocol":"036","seed":seed,"stream_sha256":stream_sha,"registration_frozen_sha256":frsha,"input_lock_sha256":sha(evidence/"input_lock.json"),"input_audit_sha256":sha(evidence/"input_audit.json"),"common_features_sha256":common_sha,"chunk_sha256":lock["chunk_sha256"],"eligibility_passed":audit_pass}); manifest["audit_pass"]=audit_pass; manifest["frozen_data_manifest"]="frozen_data_manifest.json"; W(data/"manifest.json",manifest)
    print(json.dumps({"protocol":"036","seed":seed,"audit_pass":audit_pass,"stream_sha256":stream_sha,"gates":gates},indent=2))
    if not audit_pass: raise SystemExit(2)


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("segment"); p.add_argument("--output",required=True); p.add_argument("--seed",type=int,required=True,choices=ALLOWED_SEEDS); p.add_argument("--max-intervals",type=int,default=200); p.set_defaults(fn=cmd_segment)
    p=sub.add_parser("assemble"); p.add_argument("--generation-root",required=True); p.add_argument("--output",required=True); p.set_defaults(fn=cmd_assemble)
    p=sub.add_parser("audit"); p.add_argument("--generation-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--evidence-root",required=True); p.set_defaults(fn=cmd_audit)
    a=ap.parse_args(); a.fn(a)

if __name__=="__main__": main()
