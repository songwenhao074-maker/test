"""Generate the preregistered Protocol-025 seed700 service-turnover stream.

This collector uses the existing Bitbrain/GOBI/RPiEdge/simulationStep path.
S1..S6 alter only causal task-demand trajectories. Labels are recomputed only
from post-simulator aggregate CPU/RAM/Disk demand divided by current capacity.
Generation is saved as immutable 200-interval chunks plus a dill checkpoint at
every chunk boundary; resume restores the actual simulator/workload/scheduler
object graph before generating later chunks.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import dill
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_ftmoe_protocol023_stream import (
    configure, guard, assert_no_other_experiment, sha,
    SCENARIO_PATH, DRIFT_CONFIG, FAMILIAR_PHASE,
)
from simulator.workload.BitbrainWorkloadProtocol025 import (
    Protocol025ServiceTurnoverBWGD2, SERVICE_IDS, SERVICE_LAWS,
    SERVICE_MECHANISM_IDS, FORBIDDEN_MODEL_INPUTS,
)

REGISTRATION_PATH = ROOT / "artifacts/ftmoe_online/protocol_025/registration.json"
REGISTERED_SEED = 700
COHORT = "online"


def _json_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _registration():
    reg = json.loads(REGISTRATION_PATH.read_text(encoding="utf8"))
    if reg["protocol"] != "025" or int(reg["development_seed"]) != REGISTERED_SEED:
        raise AssertionError("unexpected Protocol-025 registration")
    if reg["confirmation_seeds_forbidden"] != [701, 702, 703]:
        raise AssertionError("confirmation seed seal changed")
    if reg["test_seeds_forbidden"] != [201, 202, 203, 204, 205]:
        raise AssertionError("test seed seal changed")
    if float(reg["generation"]["task_response_event_probability"]) != 0.30:
        raise AssertionError("unregistered event probability")
    return reg


def phase_table(reg):
    cursor, rows = 0, []
    p = float(reg["generation"]["task_response_event_probability"])
    for name, service, length in reg["timeline"]:
        length = int(length)
        rows.append({
            "name": str(name), "start": cursor, "end": cursor + length,
            "length": length, "service": None if service == "baseline" else str(service),
            "event_probability": 0.0 if service == "baseline" else p,
            "kind": "baseline" if service == "baseline" else "service_response",
        })
        cursor += length
    if cursor != int(reg["scored_intervals"]):
        raise AssertionError("registered Protocol-025 timeline sum mismatch")
    return rows


def phase_at(t, phases):
    for i, phase in enumerate(phases):
        if phase["start"] <= t < phase["end"]:
            return i, phase
    return len(phases) - 1, phases[-1]


def _allocate(count, slots=16):
    return {
        "host_features": np.zeros((count, slots, 7), np.float32),
        "demands": np.zeros((count, slots, 7), np.float32),
        "schedules": np.zeros((count, slots, slots), np.float32),
        "post_totals": np.zeros((count, slots, 3), np.float64),
        "capacities": np.zeros((count, slots, 3), np.float64),
        "overload_ratio": np.zeros((count, slots, 3), np.float64),
        "raw_labels": np.zeros((count, slots), np.int64),
        "before_placement": np.full((count, slots), -1, np.int64),
        "after_placement": np.full((count, slots), -1, np.int64),
        "creation_ids": np.full((count, slots), -1, np.int64),
        "intervals": np.zeros(count, np.int64),
        "audit_phase_ids": np.zeros(count, np.int64),
        "audit_service_ids": np.full(count, -1, np.int64),
        "audit_event_ids": np.full((count, slots), -1, np.int64),
        "audit_event_service_ids": np.full((count, slots), -1, np.int64),
        "audit_event_active": np.zeros((count, slots), np.uint8),
        "audit_host_event_any": np.zeros((count, slots), np.uint8),
        "audit_deploy_attempts": np.zeros(count, np.int64),
        "audit_deploy_rejected": np.zeros(count, np.int64),
        "audit_migrate_attempts": np.zeros(count, np.int64),
        "audit_migrate_rejected": np.zeros(count, np.int64),
    }


def _chunk_name(start, end):
    return "chunk_%06d_%06d.npz" % (int(start), int(end))


def _save_chunk(output, arrays, start, end, chunks):
    output = Path(output); chunk_dir = output / "chunks"; chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / _chunk_name(start, end)
    if path.exists():
        raise FileExistsError("immutable chunk already exists: %s" % path)
    np.savez_compressed(path, **{k: v[start:end] for k, v in arrays.items()})
    rec = {"start": int(start), "end": int(end), "file": path.name, "sha256": sha(path)}
    chunks.append(rec)
    return rec


def _verify_chunks(output, chunks):
    last = 0
    for rec in chunks:
        if int(rec["start"]) != last or int(rec["end"]) <= last:
            raise AssertionError("chunk coverage is not contiguous")
        path = Path(output) / "chunks" / rec["file"]
        if not path.is_file() or sha(path) != rec["sha256"]:
            raise AssertionError("chunk digest mismatch: %s" % path)
        last = int(rec["end"])
    return last


def _load_chunks(output, arrays, chunks):
    end = _verify_chunks(output, chunks)
    for rec in chunks:
        with np.load(Path(output) / "chunks" / rec["file"]) as d:
            for key in arrays:
                arrays[key][int(rec["start"]):int(rec["end"])] = d[key]
    return end


def _checkpoint_paths(output):
    return Path(output) / "resume_state.dill", Path(output) / "resume_manifest.json"


def _write_checkpoint(output, reg_sha, next_t, sim_state, chunks, active_service,
                      active_probability, applied_switches):
    state_path, manifest_path = _checkpoint_paths(output)
    tmp = state_path.with_suffix(".tmp")
    payload = {
        "protocol": "025", "registration_sha256": reg_sha, "next_t": int(next_t),
        "sim_state": sim_state, "active_service": active_service,
        "active_probability": float(active_probability),
        "applied_switches": list(applied_switches),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": sim_state["torch"].get_rng_state(),
        },
    }
    with tmp.open("wb") as f:
        dill.dump(payload, f, protocol=dill.HIGHEST_PROTOCOL)
    os.replace(tmp, state_path)
    state_sha = sha(state_path)
    manifest = {
        "protocol": "025", "registration_sha256": reg_sha,
        "next_t": int(next_t), "state_file": state_path.name,
        "state_sha256": state_sha, "chunks": list(chunks),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return manifest


def _read_checkpoint(output, reg_sha, arrays):
    state_path, manifest_path = _checkpoint_paths(output)
    if not (state_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError("resume requested but checkpoint files are absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest["registration_sha256"] != reg_sha:
        raise AssertionError("registration changed since generation checkpoint")
    if sha(state_path) != manifest["state_sha256"]:
        raise AssertionError("resume state digest mismatch")
    end = _load_chunks(output, arrays, manifest["chunks"])
    if end != int(manifest["next_t"]):
        raise AssertionError("resume state/chunk boundary mismatch")
    with state_path.open("rb") as f:
        payload = dill.load(f)
    if payload["registration_sha256"] != reg_sha or int(payload["next_t"]) != end:
        raise AssertionError("resume payload provenance mismatch")
    random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
    payload["sim_state"]["torch"].set_rng_state(payload["rng"]["torch"])
    return payload, list(manifest["chunks"])


def _common_features(host, caps):
    values = np.stack([host[..., 0], host[..., 1], host[..., 4]], axis=-1).astype(np.float64)
    pressure = values / np.maximum(caps, 1e-12)
    delta = np.zeros_like(pressure); delta[1:] = pressure[1:] - pressure[:-1]
    slope = np.zeros_like(pressure); slope[4:] = (pressure[4:] - pressure[:-4]) / 4.0
    return np.concatenate([pressure, delta, slope], axis=-1)


def _summary_block(labels, ratio, host_event_any, start, end):
    y = labels[start:end]; r = ratio[start:end]; event = host_event_any[start:end]
    hoststeps = int(y.size); pos = int((y > 0).sum()); neg = int((y == 0).sum())
    return {
        "intervals": [int(start), int(end)], "host_steps": hoststeps,
        "positive_host_steps": pos, "negative_host_steps": neg,
        "positive_rate": pos / float(hoststeps) if hoststeps else None,
        "class_counts": {str(k): int((y == k).sum()) for k in range(4)},
        "event_related_positive_host_steps": int(((y > 0) & (event > 0)).sum()),
        "peak_overload_ratio": float(r.max()) if r.size else None,
    }


def _model_free_audit(reg, phases, arrays, workload, applied_switches):
    steps = int(reg["scored_intervals"]); labels = arrays["raw_labels"]; ratio = arrays["overload_ratio"]
    recompute = np.where((ratio > 1.0).any(-1), ratio.argmax(-1) + 1, 0)
    label_equal = bool(np.array_equal(recompute, labels))
    common = _common_features(arrays["host_features"], arrays["capacities"])
    per_phase = {}; recurrence_first100_valid = {}; all_phase_classes = True
    for p in phases:
        block = _summary_block(labels, ratio, arrays["audit_host_event_any"], p["start"], p["end"])
        block["service"] = p["service"]
        features = common[p["start"]:p["end"]]
        block["common_feature_abs_mean"] = np.mean(np.abs(features), axis=(0, 1)).tolist()
        per_phase[p["name"]] = block
        if p["service"] is not None and (block["positive_host_steps"] == 0 or block["negative_host_steps"] == 0):
            all_phase_classes = False
        if "_rec" in p["name"]:
            y = labels[p["start"]:min(p["end"], p["start"] + 100)]
            recurrence_first100_valid[p["name"]] = {
                "positive": int((y > 0).sum()), "negative": int((y == 0).sum()),
                "ap_defined": bool((y > 0).any() and (y == 0).any()),
            }
    event_by_service = {s: int(sum(1 for e in workload.response_events if e["service_id"] == s)) for s in SERVICE_IDS}
    parameter_match = {}
    for s in SERVICE_IDS:
        registered = dict(reg["service_laws"][s]); implemented = SERVICE_LAWS[s]["registered_parameters"]
        parameter_match[s] = all(str(k) in registered and float(registered[k]) == float(v)
                                 for k, v in implemented.items() if isinstance(v, (int, float)))
    age0_safe = all(float(spec["cpu_shape"][0]) == 0.0 and float(spec["ram_shape"][0]) == 0.0 and
                    float(spec["disk_shape"][0]) == 0.0 for spec in SERVICE_LAWS.values())
    cue_finite = bool(np.isfinite(common[:steps]).all())
    audit = {
        "protocol": "025", "kind": "pre_model_data_audit", "model_results_seen": False,
        "causal_collection_order": ["host_features/demands/capacity", "scheduler_decision", "simulationStep", "post_totals/ratio/raw_label"],
        "service_ids_are_audit_only": True,
        "labels_recomputed_from_capacity_exceedance_equal": label_equal,
        "direct_label_assignment_from_service_id": False,
        "timeline_scored_intervals": int(sum(p["length"] for p in phases)),
        "per_phase": per_phase,
        "recurrence_first100_class_coverage": recurrence_first100_valid,
        "service_event_counts": event_by_service,
        "short_trace_skips": dict(workload.short_trace_skips),
        "registered_physical_parameters_match_implementation": parameter_match,
        "admission_age0_safe": age0_safe,
        "common_observable_features_finite": cue_finite,
        "common_feature_order": reg["causality"]["common_observable_features"],
        "applied_switches": applied_switches,
    }
    gates = {
        "label_rule": label_equal,
        "timeline": audit["timeline_scored_intervals"] == steps,
        "events_every_service": all(event_by_service[s] > 0 for s in SERVICE_IDS),
        "physical_parameters": all(parameter_match.values()),
        "age0_safe": age0_safe,
        "features_finite": cue_finite,
        "all_service_phases_have_positive_and_negative": all_phase_classes,
        "all_recurrence_first100_ap_defined": all(x["ap_defined"] for x in recurrence_first100_valid.values()),
    }
    audit["gates"] = gates; audit["audit_pass"] = bool(all(gates.values()))
    audit["revision_allowed_if_failed"] = int(reg["allowed_data_revision_count"])
    return audit, common


def _finalize(output, reg, phases, arrays, workload, chunks, applied_switches,
              scheduler_weight, started, reg_sha):
    output = Path(output); count = int(reg["scored_intervals"]) + int(reg["guard_intervals"])
    if _verify_chunks(output, chunks) != count:
        raise AssertionError("cannot finalize incomplete chunk sequence")
    assembled = _allocate(count); _load_chunks(output, assembled, chunks)
    stream_path = output / "stream.npz"
    if stream_path.exists(): raise FileExistsError("refusing to overwrite final stream")
    np.savez_compressed(stream_path, **assembled,
                        overload_mask=(assembled["overload_ratio"] > 1.0).astype(np.uint8))
    digest = sha(stream_path)
    audit, common = _model_free_audit(reg, phases, assembled, workload, applied_switches)
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8")
    np.savez_compressed(output / "common_observable_features.npz", features=common)
    (output / "events.json").write_text(json.dumps(workload.response_events, indent=2, allow_nan=False) + "\n", encoding="utf8")
    manifest = {
        "protocol": "025", "seed": REGISTERED_SEED,
        "steps": int(reg["scored_intervals"]), "guard_rows": int(reg["guard_intervals"]),
        "stream_file": "stream.npz", "stream_sha256": digest,
        "registration_sha256": reg_sha, "timeline": phases,
        "service_laws": SERVICE_LAWS,
        "event_probability": float(reg["generation"]["task_response_event_probability"]),
        "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
        "audit_only_npz_keys": [k for k in assembled if k.startswith("audit_")],
        "model_input_keys": ["host_features", "demands", "schedules", "capacities", "creation_ids", "before_placement"],
        "common_observable_features_file": "common_observable_features.npz",
        "common_observable_feature_order": reg["causality"]["common_observable_features"],
        "label_key": "raw_labels",
        "label_rule": "argmax(CPU,RAM,Disk aggregate/capacity) when any ratio>1 else 0",
        "scheduler_checkpoint": str(scheduler_weight), "scheduler_checkpoint_sha256": sha(scheduler_weight),
        "scenario_adapter_sha256": sha(SCENARIO_PATH), "drift_config_sha256": sha(DRIFT_CONFIG),
        "chunk_manifest": chunks, "generation_elapsed_seconds": time.perf_counter() - started,
        "data_audit_file": "data_audit.json", "audit_pass": bool(audit["audit_pass"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return manifest, audit


def collect(output, resume=False):
    output = Path(output); reg = _registration(); reg_sha = _json_sha(REGISTRATION_PATH)
    phases = phase_table(reg); steps = int(reg["scored_intervals"]); count = steps + int(reg["guard_intervals"])
    chunk_size = int(reg["generation"]["chunk_intervals"]); arrays = _allocate(count)
    started = time.perf_counter(); process_guard = None
    if resume:
        if not output.is_dir(): raise FileNotFoundError(output)
    else:
        if output.exists(): raise FileExistsError("refusing to overwrite Protocol-025 data dir %s" % output)
        output.mkdir(parents=True)
        (output / "registration_snapshot.json").write_bytes(REGISTRATION_PATH.read_bytes())
    try:
        configure(); guard(); process_guard = assert_no_other_experiment(); os.chdir(ROOT)
        import torch
        from simulator.Simulator import Simulator
        from simulator.environment.RPiEdge import RPiEdge
        from simulator.environment.RPiCapacity import RPiCapacity
        from scheduler.GOBI import GOBIScheduler
        from recovery.Recovery import Recovery
        from stats.Stats import Stats
        from src.constants import MODEL_SAVE_PATH

        bitbrain = ROOT / "simulator/workload/datasets/bitbrain/rnd"
        if not all((bitbrain / ("%d.csv" % i)).is_file() for i in range(1, 500)):
            raise FileNotFoundError("local Bitbrain rnd cohort is incomplete")
        candidates = [ROOT / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt",
                      ROOT / "scheduler/BaGTI" / str(MODEL_SAVE_PATH) / "energy_latency_16_Trained.ckpt"]
        scheduler_weight = next((p.resolve() for p in candidates if p.is_file()), None)
        if scheduler_weight is None: raise FileNotFoundError("GOBI trained checkpoint missing")

        log_mode = "a" if resume else "w"
        with (output / "generation.log").open(log_mode, encoding="utf8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if resume:
                payload, chunks = _read_checkpoint(output, reg_sha, arrays)
                sim = payload["sim_state"]
                env, workload, scheduler, recovery, stats, capacity = (sim[x] for x in ("env","workload","scheduler","recovery","stats","capacity"))
                t0 = int(payload["next_t"]); active_service = payload["active_service"]
                active_probability = float(payload["active_probability"]); applied_switches = list(payload["applied_switches"])
                print(json.dumps({"resume_from":t0,"chunks":len(chunks),"registration_sha256":reg_sha}), flush=True)
            else:
                random.seed(REGISTERED_SEED); np.random.seed(REGISTERED_SEED); torch.manual_seed(REGISTERED_SEED)
                scenario=json.loads(SCENARIO_PATH.read_text(encoding="utf8")); drift=json.loads(DRIFT_CONFIG.read_text(encoding="utf8"))
                familiar=next(p for p in drift["phases"] if p["name"]==FAMILIAR_PHASE); adapter=dict(scenario.get("adapter") or {}); adapter.update(familiar.get("adapter") or {})
                law_rel=scenario.get("disk_law_relative"); disk_law_path=(ROOT/law_rel).resolve() if law_rel else None
                dc=RPiEdge(16); workload=Protocol025ServiceTurnoverBWGD2(float(scenario.get("arrival_mean",1.0)),float(scenario.get("arrival_sigma",1.5)),REGISTERED_SEED,cohort=COHORT,adapter=adapter,disk_law_path=disk_law_path,active_service=None,event_probability=float(reg["generation"]["task_response_event_probability"]))
                scheduler=GOBIScheduler("energy_latency_16"); recovery=Recovery(); stats=Stats(workload,dc,scheduler); stats.feats_per_host=7; stats.history_limit=64; stats.series_tail=96
                env=Simulator(1000,10000,scheduler,recovery,stats,16,300,dc.generateHosts()); capacity=RPiCapacity(env.hostlist); capacity.apply(familiar["cpu_scale"],familiar["ram_scale"],familiar["disk_scale"])
                initial=workload.generateNewContainers(env.interval); deployed=env.addContainersInit(initial); decision=scheduler.placement(deployed); migrations=env.allocateInit(decision); workload.updateDeployedContainers(env.getCreationIDs(migrations,deployed)); stats.saveStats(deployed,migrations,[],deployed,decision,0)
                t0=0; chunks=[]; active_service=None; active_probability=0.0; applied_switches=[]

            chunk_start=t0
            for t in range(t0,count):
                if t % 50 == 0: guard()
                phase_index, phase = phase_at(t, phases); service=phase["service"]; probability=float(phase["event_probability"])
                if service != active_service or probability != active_probability:
                    workload.set_active_service(service, probability=probability); active_service, active_probability=service, probability
                    if t < steps: applied_switches.append({"interval":int(t),"phase":phase["name"],"service":service,"event_probability":probability})
                arrays["audit_phase_ids"][t]=phase_index; arrays["audit_service_ids"][t]=-1 if service is None else SERVICE_MECHANISM_IDS[service]
                arrays["capacities"][t]=capacity.current(); new=workload.generateNewContainers(env.interval); deployed,destroyed=env.addContainers(new); arrays["intervals"][t]=env.interval
                for slot,container in enumerate(env.containerlist):
                    if container is None: continue
                    if container.id != slot or not container.active: raise AssertionError("invalid live slot identity")
                    ram=container.getRAM(); disk=container.getDisk(); values=np.asarray([container.getBaseIPS(),*ram,*disk],dtype=np.float64); arrays["demands"][t,slot]=values; arrays["creation_ids"][t,slot]=container.creationID
                    hid=container.getHostID(); arrays["before_placement"][t,slot]=hid
                    if hid>=0: arrays["host_features"][t,hid]+=values
                    eid=workload.response_event_id(container.creationID)
                    if eid is not None:
                        event=workload.response_events[eid]; arrays["audit_event_ids"][t,slot]=int(eid); arrays["audit_event_service_ids"][t,slot]=int(event["mechanism_id"]); age=int(env.interval-container.startAt); spec=SERVICE_LAWS[event["service_id"]]
                        shapes=(spec["cpu_shape"],spec["ram_shape"],spec["disk_shape"]); active=any(0<=age<len(shape) and float(shape[age])>0.0 for shape in shapes); arrays["audit_event_active"][t,slot]=1 if active else 0
                        if hid>=0 and active: arrays["audit_host_event_any"][t,hid]=1
                selected=scheduler.selection(); decision=scheduler.filter_placement(scheduler.placement(selected+deployed)); arrays["schedules"][t]=np.asarray(scheduler.result_cache); np.testing.assert_allclose(arrays["schedules"][t].sum(-1),1.0,atol=1e-5)
                for cid,hid in decision:
                    c=env.containerlist[cid] if 0<=cid<len(env.containerlist) else None
                    if c is None: continue
                    if c.getHostID()==-1: arrays["audit_deploy_attempts"][t]+=1
                    else: arrays["audit_migrate_attempts"][t]+=1
                executed=env.simulationStep(recovery.run_model(stats.time_series,decision)); executed_set={(cid,int(hid)) for cid,hid in executed}
                for cid,hid in decision:
                    if (cid,int(hid)) in executed_set: continue
                    c=env.containerlist[cid] if 0<=cid<len(env.containerlist) else None
                    if c is None or c.getHostID()==-1: arrays["audit_deploy_rejected"][t]+=1
                    else: arrays["audit_migrate_rejected"][t]+=1
                workload.updateDeployedContainers(env.getCreationIDs(executed,deployed))
                for slot,container in enumerate(env.containerlist):
                    if container is None: continue
                    hid=container.getHostID(); arrays["after_placement"][t,slot]=hid
                    if hid>=0: arrays["post_totals"][t,hid]+=[container.getBaseIPS(),container.getRAM()[0],container.getDisk()[0]]
                arrays["overload_ratio"][t]=arrays["post_totals"][t]/arrays["capacities"][t]
                arrays["raw_labels"][t]=np.where((arrays["overload_ratio"][t]>1.0).any(-1),arrays["overload_ratio"][t].argmax(-1)+1,0)
                stats.saveStats(deployed,migrations,destroyed,selected,decision,0)

                boundary = ((t + 1) % chunk_size == 0) or (t + 1 == count)
                if boundary:
                    _save_chunk(output, arrays, chunk_start, t + 1, chunks); chunk_start=t+1
                    sim_state={"env":env,"workload":workload,"scheduler":scheduler,"recovery":recovery,"stats":stats,"capacity":capacity,"torch":torch}
                    _write_checkpoint(output,reg_sha,t+1,sim_state,chunks,active_service,active_probability,applied_switches)
                    print(json.dumps({"checkpoint_next_t":t+1,"chunks":len(chunks)}),flush=True)

            manifest,audit=_finalize(output,reg,phases,arrays,workload,chunks,applied_switches,scheduler_weight,started,reg_sha)
            print(json.dumps({"stream_sha256":manifest["stream_sha256"],"audit_pass":audit["audit_pass"],"gates":audit["gates"]},indent=2,allow_nan=False),flush=True)
            return manifest
    except Exception as exc:
        failure={"error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"resume_available":_checkpoint_paths(output)[0].is_file(),"next_action":"repair deterministic collector issue or resume from last immutable chunk; do not run models before data audit passes"}
        output.mkdir(parents=True,exist_ok=True); (output/"failure.json").write_text(json.dumps(failure,indent=2,allow_nan=False)+"\n",encoding="utf8"); raise
    finally:
        _=process_guard


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--resume",action="store_true"); a=p.parse_args(); collect(a.output,resume=a.resume)


if __name__=="__main__": main()
