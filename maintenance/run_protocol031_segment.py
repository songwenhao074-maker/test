"""Resumable Protocol-031 simulator collector.

Each invocation advances only the same registered seed700/scenario stream.
Immutable 200-step chunks plus simulator/workload/scheduler/RNG state are
preserved. A transient partial block may bridge the latest immutable chunk.
"""
from __future__ import annotations
import argparse, contextlib, json, os, random, sys, time, traceback
from pathlib import Path
import dill
import numpy as np
import psutil

# The inherited collector uses a Windows priority-class symbol. On POSIX,
# Process.nice expects an integer niceness value; 10 is the historical
# BELOW_NORMAL compatibility value already used by the Protocol-025 CI recovery.
if not hasattr(psutil,"BELOW_NORMAL_PRIORITY_CLASS"):
    psutil.BELOW_NORMAL_PRIORITY_CLASS=10

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

BAGTI_ROOT=ROOT/"scheduler"/"BaGTI"
if str(BAGTI_ROOT) not in sys.path:
    sys.path.append(str(BAGTI_ROOT))

import prepare_ftmoe_protocol031_stream as P

TRANSIENT_NAME="transient_partial.npz"

def write_checkpoint(output,reg_sha,next_t,sim_state,chunks,arrays,
                     active_service,active_probability,applied_switches):
    output=Path(output)
    state_path,manifest_path=P._checkpoint_paths(output)
    tmp=state_path.with_suffix(".tmp")
    payload={
        "protocol":"031","plan_revision":2,"scenario_id":P.SCENARIO_ID,
        "data_revision":P.DATA_REVISION,
        "registration_sha256":reg_sha,"next_t":int(next_t),
        "sim_state":sim_state,"active_service":active_service,
        "active_probability":float(active_probability),
        "applied_switches":list(applied_switches),
        "rng":{
            "python":random.getstate(),"numpy":np.random.get_state(),
            "torch":sim_state["torch"].get_rng_state(),
        },
    }
    with tmp.open("wb") as f:
        dill.dump(payload,f,protocol=dill.HIGHEST_PROTOCOL)
    os.replace(tmp,state_path)
    chunk_end=int(P._verify_chunks(output,chunks))
    next_t=int(next_t)
    if next_t<chunk_end:
        raise AssertionError("transient checkpoint precedes immutable chunk end")
    transient_path=output/TRANSIENT_NAME
    transient=None
    if next_t>chunk_end:
        np.savez_compressed(transient_path,
                            **{k:v[chunk_end:next_t] for k,v in arrays.items()})
        transient={
            "file":transient_path.name,"start":chunk_end,"end":next_t,
            "sha256":P.sha(transient_path),
        }
    elif transient_path.exists():
        transient_path.unlink()
    manifest={
        "protocol":"031","plan_revision":2,"scenario_id":P.SCENARIO_ID,
        "data_revision":P.DATA_REVISION,
        "registration_sha256":reg_sha,"next_t":next_t,
        "state_file":state_path.name,"state_sha256":P.sha(state_path),
        "chunks":list(chunks),"engineering_transient_checkpoint":transient,
        "registered_chunk_semantics_unchanged":True,
    }
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",
                             encoding="utf8")
    return manifest

def read_checkpoint(output,reg_sha,arrays):
    output=Path(output)
    state_path,manifest_path=P._checkpoint_paths(output)
    if not (state_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError("Protocol031 resume checkpoint absent")
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("protocol")!="031" or manifest.get("scenario_id")!=P.SCENARIO_ID:
        raise AssertionError("wrong Protocol031 checkpoint identity")
    if manifest["registration_sha256"]!=reg_sha:
        raise AssertionError("registration changed since Protocol031 checkpoint")
    if P.sha(state_path)!=manifest["state_sha256"]:
        raise AssertionError("resume state digest mismatch")
    chunks=list(manifest["chunks"])
    chunk_end=int(P._load_chunks(output,arrays,chunks))
    next_t=int(manifest["next_t"])
    transient=manifest.get("engineering_transient_checkpoint")
    if transient is None:
        if chunk_end!=next_t:
            raise AssertionError("resume state/chunk boundary mismatch")
    else:
        start,end=int(transient["start"]),int(transient["end"])
        path=output/transient["file"]
        if start!=chunk_end or end!=next_t or end<=start:
            raise AssertionError("invalid Protocol031 transient coverage")
        if not path.is_file() or P.sha(path)!=transient["sha256"]:
            raise AssertionError("transient checkpoint digest mismatch")
        with np.load(path) as d:
            for key in arrays:
                arrays[key][start:end]=d[key]
    with state_path.open("rb") as f:
        payload=dill.load(f)
    if payload.get("protocol")!="031" or payload["registration_sha256"]!=reg_sha:
        raise AssertionError("wrong Protocol031 resume payload")
    if int(payload["next_t"])!=next_t:
        raise AssertionError("resume payload next_t mismatch")
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
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

    random.seed(700); np.random.seed(700); torch.manual_seed(700)
    scenario=json.loads(P.SCENARIO_PATH.read_text(encoding="utf8"))
    drift=json.loads(P.DRIFT_CONFIG.read_text(encoding="utf8"))
    familiar=next(x for x in drift["phases"] if x["name"]==P.FAMILIAR_PHASE)
    adapter=dict(scenario.get("adapter") or {})
    adapter.update(familiar.get("adapter") or {})
    law_rel=scenario.get("disk_law_relative")
    disk_law_path=(P.ROOT/law_rel).resolve() if law_rel else None
    dc=RPiEdge(16)
    workload=P.Protocol025ServiceTurnoverBWGD2(
        float(scenario.get("arrival_mean",1.0)),
        float(scenario.get("arrival_sigma",1.5)),
        700,cohort=P.COHORT,adapter=adapter,disk_law_path=disk_law_path,
        active_service=None,event_probability=float(reg["generation"]["event_probability"]))
    scheduler=GOBIScheduler("energy_latency_16")
    recovery=Recovery()
    stats=Stats(workload,dc,scheduler)
    stats.feats_per_host=7; stats.history_limit=64; stats.series_tail=96
    env=Simulator(1000,10000,scheduler,recovery,stats,16,300,dc.generateHosts())
    capacity=RPiCapacity(env.hostlist)
    capacity.apply(familiar["cpu_scale"],familiar["ram_scale"],familiar["disk_scale"])
    initial=workload.generateNewContainers(env.interval)
    deployed=env.addContainersInit(initial)
    decision=scheduler.placement(deployed)
    migrations=env.allocateInit(decision)
    workload.updateDeployedContainers(env.getCreationIDs(migrations,deployed))
    stats.saveStats(deployed,migrations,[],deployed,decision,0)

    candidates=[
        P.ROOT/str(MODEL_SAVE_PATH)/"energy_latency_16_Trained.ckpt",
        P.ROOT/"scheduler/BaGTI"/str(MODEL_SAVE_PATH)/"energy_latency_16_Trained.ckpt",
    ]
    scheduler_weight=next((x.resolve() for x in candidates if x.is_file()),None)
    if scheduler_weight is None:
        raise FileNotFoundError("GOBI trained checkpoint missing")
    sim_state={"env":env,"workload":workload,"scheduler":scheduler,
               "recovery":recovery,"stats":stats,"capacity":capacity,"torch":torch}
    write_checkpoint(output,reg_sha,0,sim_state,[],arrays,None,0.0,[])
    return sim_state,scheduler_weight

def collect_segment(output,max_intervals):
    output=Path(output); max_intervals=int(max_intervals)
    if max_intervals<=0: raise ValueError("max_intervals must be positive")
    reg=P.registration(); reg_sha=P.json_sha(P.REGISTRATION_PATH)
    phases=P.phase_table(reg)
    steps=int(reg["scored_intervals"]); count=steps+int(reg["guard_intervals"])
    chunk_size=int(reg["generation"]["chunk_intervals"])
    arrays=P._allocate(count)
    process_guard=None
    segment_started=time.perf_counter()

    if not output.exists():
        output.mkdir(parents=True)
        (output/"registration_snapshot.json").write_bytes(P.REGISTRATION_PATH.read_bytes())
    elif not output.is_dir():
        raise NotADirectoryError(output)

    try:
        P.configure(); P.guard(); process_guard=P.assert_no_other_experiment()
        os.chdir(P.ROOT)
        import torch
        from src.constants import MODEL_SAVE_PATH
        bitbrain=P.ROOT/"simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain/("%d.csv"%i)).is_file() for i in range(1,500)):
            raise FileNotFoundError("local Bitbrain rnd cohort incomplete")
        candidates=[
            P.ROOT/str(MODEL_SAVE_PATH)/"energy_latency_16_Trained.ckpt",
            P.ROOT/"scheduler/BaGTI"/str(MODEL_SAVE_PATH)/"energy_latency_16_Trained.ckpt",
        ]
        scheduler_weight=next((x.resolve() for x in candidates if x.is_file()),None)
        if scheduler_weight is None:
            raise FileNotFoundError("GOBI trained checkpoint missing")

        if not P._checkpoint_paths(output)[0].is_file():
            initialize(output,reg,reg_sha,arrays)

        with (output/"generation.log").open("a",encoding="utf8") as log, \
             contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            payload,chunks,chunk_start=read_checkpoint(output,reg_sha,arrays)
            sim=payload["sim_state"]
            env,workload,scheduler,recovery,stats,capacity=(
                sim[x] for x in ("env","workload","scheduler","recovery","stats","capacity"))
            t0=int(payload["next_t"])
            active_service=payload["active_service"]
            active_probability=float(payload["active_probability"])
            applied_switches=list(payload["applied_switches"])
            print(json.dumps({
                "protocol":"031","segment_resume_from":t0,
                "registered_chunk_start":chunk_start,"chunks":len(chunks),
                "registration_sha256":reg_sha,"max_intervals":max_intervals,
            }),flush=True)
            generated=0
            for t in range(t0,count):
                if t%50==0: P.guard()
                phase_index,phase=P.phase_at(t,phases)
                service=phase["service"]; probability=float(phase["event_probability"])
                if service!=active_service or probability!=active_probability:
                    workload.set_active_service(service,probability=probability)
                    active_service,active_probability=service,probability
                    if t<steps:
                        applied_switches.append({
                            "interval":int(t),"phase":phase["name"],
                            "service":service,"logical_service":phase["logical_service"],
                            "event_probability":probability,
                        })
                arrays["audit_phase_ids"][t]=phase_index
                arrays["audit_service_ids"][t]=(
                    -1 if service is None else P.SERVICE_MECHANISM_IDS[service])
                arrays["capacities"][t]=capacity.current()
                new=workload.generateNewContainers(env.interval)
                deployed,destroyed=env.addContainers(new)
                arrays["intervals"][t]=env.interval

                for slot,container in enumerate(env.containerlist):
                    if container is None: continue
                    if container.id!=slot or not container.active:
                        raise AssertionError("invalid live slot identity")
                    ram=container.getRAM(); disk=container.getDisk()
                    values=np.asarray([container.getBaseIPS(),*ram,*disk],dtype=np.float64)
                    arrays["demands"][t,slot]=values
                    arrays["creation_ids"][t,slot]=container.creationID
                    hid=container.getHostID(); arrays["before_placement"][t,slot]=hid
                    if hid>=0: arrays["host_features"][t,hid]+=values
                    eid=workload.response_event_id(container.creationID)
                    if eid is not None:
                        event=workload.response_events[eid]
                        arrays["audit_event_ids"][t,slot]=int(eid)
                        arrays["audit_event_service_ids"][t,slot]=int(event["mechanism_id"])
                        age=int(env.interval-container.startAt)
                        spec=P.SERVICE_LAWS[event["service_id"]]
                        shapes=(spec["cpu_shape"],spec["ram_shape"],spec["disk_shape"])
                        active=any(0<=age<len(shape) and float(shape[age])>0.0 for shape in shapes)
                        arrays["audit_event_active"][t,slot]=1 if active else 0
                        if hid>=0 and active: arrays["audit_host_event_any"][t,hid]=1

                selected=scheduler.selection()
                decision=scheduler.filter_placement(scheduler.placement(selected+deployed))
                arrays["schedules"][t]=np.asarray(scheduler.result_cache)
                np.testing.assert_allclose(arrays["schedules"][t].sum(-1),1.0,atol=1e-5)
                for cid,hid in decision:
                    c=env.containerlist[cid] if 0<=cid<len(env.containerlist) else None
                    if c is None: continue
                    if c.getHostID()==-1: arrays["audit_deploy_attempts"][t]+=1
                    else: arrays["audit_migrate_attempts"][t]+=1

                executed=env.simulationStep(recovery.run_model(stats.time_series,decision))
                executed_set={(cid,int(hid)) for cid,hid in executed}
                for cid,hid in decision:
                    if (cid,int(hid)) in executed_set: continue
                    c=env.containerlist[cid] if 0<=cid<len(env.containerlist) else None
                    if c is None or c.getHostID()==-1: arrays["audit_deploy_rejected"][t]+=1
                    else: arrays["audit_migrate_rejected"][t]+=1
                workload.updateDeployedContainers(env.getCreationIDs(executed,deployed))
                for slot,container in enumerate(env.containerlist):
                    if container is None: continue
                    hid=container.getHostID(); arrays["after_placement"][t,slot]=hid
                    if hid>=0:
                        arrays["post_totals"][t,hid]+=[
                            container.getBaseIPS(),container.getRAM()[0],container.getDisk()[0]]
                arrays["overload_ratio"][t]=arrays["post_totals"][t]/arrays["capacities"][t]
                arrays["raw_labels"][t]=np.where(
                    (arrays["overload_ratio"][t]>1.0).any(-1),
                    arrays["overload_ratio"][t].argmax(-1)+1,0)
                migrations=executed
                stats.saveStats(deployed,migrations,destroyed,selected,decision,0)
                generated+=1

                boundary=((t+1)%chunk_size==0) or (t+1==count)
                if boundary:
                    P._save_chunk(output,arrays,chunk_start,t+1,chunks)
                    chunk_start=t+1
                    sim_state={"env":env,"workload":workload,"scheduler":scheduler,
                               "recovery":recovery,"stats":stats,"capacity":capacity,"torch":torch}
                    write_checkpoint(output,reg_sha,t+1,sim_state,chunks,arrays,
                                     active_service,active_probability,applied_switches)
                    print(json.dumps({"checkpoint_next_t":t+1,"chunks":len(chunks)}),
                          flush=True)

                if generated>=max_intervals and t+1<count:
                    sim_state={"env":env,"workload":workload,"scheduler":scheduler,
                               "recovery":recovery,"stats":stats,"capacity":capacity,"torch":torch}
                    m=write_checkpoint(output,reg_sha,t+1,sim_state,chunks,arrays,
                                       active_service,active_probability,applied_switches)
                    result={
                        "protocol":"031","complete":False,"next_t":int(t+1),
                        "scored_target":steps,"row_target":count,
                        "immutable_chunk_end":int(P._verify_chunks(output,chunks)),
                        "transient":m["engineering_transient_checkpoint"],
                        "segment_generated":generated,
                    }
                    (output/"generation_status.json").write_text(
                        json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
                    print(json.dumps(result),flush=True)
                    return result

            manifest,audit=P.finalize(
                output,reg,phases,arrays,workload,chunks,applied_switches,
                scheduler_weight,segment_started,reg_sha)
            result={
                "protocol":"031","complete":True,"next_t":count,
                "scored_target":steps,"row_target":count,
                "stream_sha256":manifest["stream_sha256"],
                "audit_pass":bool(audit["audit_pass"]),"gates":audit["gates"],
            }
            (output/"generation_status.json").write_text(
                json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
            print(json.dumps(result,indent=2,allow_nan=False),flush=True)
            return result
    except Exception as exc:
        failure={
            "protocol":"031","error_type":type(exc).__name__,"error":str(exc),
            "traceback":traceback.format_exc(),
            "resume_available":P._checkpoint_paths(output)[0].is_file(),
            "next_action":"repair deterministic collector issue or resume the same registered stream; never reroll seed/scenario",
        }
        (output/"failure.json").write_text(
            json.dumps(failure,indent=2,allow_nan=False)+"\n",encoding="utf8")
        raise
    finally:
        _=process_guard

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--max-intervals",required=True,type=int)
    args=ap.parse_args()
    result=collect_segment(args.output,args.max_intervals)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=="__main__":
    main()
