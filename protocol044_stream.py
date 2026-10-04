"""Protocol-044 one-stream seed4401 generator wrapper around frozen Protocol-036 simulator source."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import dill,numpy as np,psutil

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/"artifacts/ftmoe_online/protocol_044/plan.json"
SCENARIO=ROOT/"artifacts/ftmoe_online/protocol_044/scenario_registration.json"
PLAN_SHA="a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51"
SEED=4401;STEPS=5952;ROWS=5953;CHUNK=200;FINAL=153
REGISTRATION_PATH=None;H=None;BASE=None
SOURCE_FILES={"protocol036_stream.py":"6ee73bbd16ba193164308db0a566843b4432f059f080d89e468ea6d7d7a6e4e7",
 "run_ftmoe_protocol036.py":"1cfadba564809df650235277470241682747e55c21cb9cb5e821ee8e3b3e2e7d",
 "simulator/workload/BitbrainWorkloadProtocol025.py":"71ca206e80d9fd55e64a6526a3e538598ed19fe92e59fe28a691a0f1017ace7c"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def configure_source(source):
 global H,BASE
 source=Path(source).resolve()
 if str(source) in sys.path:sys.path.remove(str(source))
 sys.path.insert(0,str(source))
 import protocol036_stream as hist
 import maintenance.run_protocol031_segment as base
 H=hist;BASE=base
 got={rel:sha(source/rel) for rel in SOURCE_FILES}
 if got!=SOURCE_FILES:raise AssertionError("frozen source036 hash mismatch "+repr(got))
 H.SOURCE_SERVICE={"U":"S1","V":"S3","W":"S4","X":"S2"}
 H.LOGICAL_BY_SOURCE={v:k for k,v in H.SOURCE_SERVICE.items()}
 return source,got
def plan():
 if sha(PLAN)!=PLAN_SHA:raise AssertionError("plan hash")
 p=J(PLAN)
 if (p.get("protocol"),p.get("revision"))!=("044",1):raise AssertionError("plan identity")
 return p
def runtime_registration():
 x=J(SCENARIO)
 if (x.get("protocol"),x.get("revision"),x.get("replay_seed"))!=("044",1,SEED):raise AssertionError("scenario identity")
 r=json.loads(json.dumps(x));r["plan_revision"]=1;r["scored_intervals"]=STEPS;r["guard_intervals"]=1;r["total_intervals"]=ROWS
 g=r["generation"];g["chunk_intervals"]=CHUNK;g["segment_max_intervals"]=CHUNK;g["expected_chunk_count"]=30;g["final_chunk_intervals"]=FINAL
 r["source_code_commit"]=plan()["source036"]["execution_commit"];r["source_scenario_sha256"]=sha(SCENARIO);r["registered_before_generation"]=True
 r["causality"]["label_rule"]="raw_labels[t]=argmax(overload_ratio[t]) + 1 iff any ratio>1 else 0"
 return r
def set_registration_path(p):
 global REGISTRATION_PATH;REGISTRATION_PATH=Path(p).resolve()
def registration():
 if REGISTRATION_PATH is None:raise RuntimeError("registration not configured")
 r=J(REGISTRATION_PATH)
 if r.get("protocol")!="044" or r.get("replay_seed")!=SEED or r.get("scored_intervals")!=STEPS or r.get("total_intervals")!=ROWS:raise AssertionError("runtime registration")
 return r
def json_sha(p):return sha(p)
def phase_table(reg=None):
 reg=registration() if reg is None else reg;rows=[];cur=0
 for item in reg["timeline"]:
  a,b=int(item["start"]),int(item["end"])
  if a!=cur or b-a!=int(item["length"]):raise AssertionError("timeline discontinuity")
  logical=None if item["service"]=="baseline" else str(item["service"]);source=item.get("source_service")
  rows.append({"name":str(item["name"]),"start":a,"end":b,"length":b-a,"logical_service":logical,"service":source,
    "event_probability":float(item["event_probability"]),"kind":"baseline" if source is None else "service_response"})
  cur=b
 if cur!=STEPS:raise AssertionError("timeline total")
 return rows
def phase_at(t,phases):
 for i,p in enumerate(phases):
  if int(p["start"])<=int(t)<int(p["end"]):return i,p
 # guard row continues final X physical state, audit phase remains final registered phase.
 return len(phases)-1,phases[-1]
def source_identity(source):
 d={rel:sha(Path(source)/rel) for rel in SOURCE_FILES}
 return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def append_event(out,event):
 p=Path(out)/"generation_ledger.jsonl";p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("a",encoding="utf8") as f:f.write(json.dumps({"protocol":"044","time_unix":time.time(),**event},allow_nan=False)+"\n")
def checkpoint_paths(out):return Path(out)/"resume_state.dill",Path(out)/"resume_manifest.json"
def write_checkpoint(output,reg_sha,next_t,sim_state,chunks,arrays,active_service,active_probability,applied_switches):
 out=Path(output);state,manifest=checkpoint_paths(out);tmp=state.with_suffix(".tmp");sid=source_identity(Path(H.__file__).resolve().parent)
 payload={"protocol":"044","revision":1,"scenario_id":registration()["scenario_id"],"replay_seed":SEED,"registration_sha256":reg_sha,
  "source_identity":sid,"next_t":int(next_t),"sim_state":sim_state,"active_service":active_service,"active_probability":float(active_probability),
  "applied_switches":list(applied_switches),"rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":sim_state["torch"].get_rng_state()}}
 with tmp.open("wb") as f:dill.dump(payload,f,protocol=dill.HIGHEST_PROTOCOL)
 os.replace(tmp,state);chunk_end=int(H._verify_chunks(out,chunks));next_t=int(next_t)
 trans=None;tp=out/"transient_partial.npz"
 if next_t>chunk_end:
  np.savez_compressed(tp,**{k:v[chunk_end:next_t] for k,v in arrays.items()});trans={"file":tp.name,"start":chunk_end,"end":next_t,"sha256":sha(tp)}
 elif tp.exists():tp.unlink()
 m={"protocol":"044","revision":1,"scenario_id":registration()["scenario_id"],"replay_seed":SEED,"registration_sha256":reg_sha,
   "source_identity":sid,"next_t":next_t,"state_file":state.name,"state_sha256":sha(state),"chunks":list(chunks),"engineering_transient_checkpoint":trans}
 W(manifest,m);return m
def read_checkpoint(output,reg_sha,arrays):
 out=Path(output);state,manifest=checkpoint_paths(out);m=J(manifest);sid=source_identity(Path(H.__file__).resolve().parent)
 if m.get("protocol")!="044" or m.get("registration_sha256")!=reg_sha or m.get("source_identity")!=sid:raise AssertionError("resume identity")
 if sha(state)!=m["state_sha256"]:raise AssertionError("resume digest")
 chunks=list(m["chunks"]);chunk_end=int(H._verify_chunks(out,chunks));next_t=int(m["next_t"]);tr=m.get("engineering_transient_checkpoint")
 if tr:
  p=out/tr["file"]
  if int(tr["start"])!=chunk_end or int(tr["end"])!=next_t or sha(p)!=tr["sha256"]:raise AssertionError("transient")
  with np.load(p,allow_pickle=False) as z:
   for k in arrays:arrays[k][chunk_end:next_t]=z[k]
 elif chunk_end!=next_t:raise AssertionError("chunk boundary")
 with state.open("rb") as f:x=dill.load(f)
 if x.get("protocol")!="044" or x.get("source_identity")!=sid or int(x["next_t"])!=next_t:raise AssertionError("resume payload")
 random.setstate(x["rng"]["python"]);np.random.set_state(x["rng"]["numpy"]);x["sim_state"]["torch"].set_rng_state(x["rng"]["torch"])
 return x,chunks,chunk_end
def finalize(output,reg,phases,arrays,workload,chunks,applied_switches,scheduler_weight,started,reg_sha):
 out=Path(output);end=int(H._verify_chunks(out,chunks))
 if end!=ROWS or len(chunks)!=30 or int(chunks[-1]["end"])-int(chunks[-1]["start"])!=FINAL:raise AssertionError("generation geometry")
 W(out/"events_final.json",workload.response_events);W(out/"applied_switches.json",applied_switches)
 g={"protocol":"044","revision":1,"scenario_id":reg["scenario_id"],"replay_seed":SEED,"complete":True,"row_target":ROWS,"scored_target":STEPS,
   "chunk_count":len(chunks),"last_chunk_rows":FINAL,"chunks":list(chunks),"registration_sha256":reg_sha,
   "source_identity":source_identity(Path(H.__file__).resolve().parent),"scheduler_checkpoint":str(scheduler_weight),
   "scheduler_checkpoint_sha256":sha(scheduler_weight),"elapsed_seconds":float(time.perf_counter()-started),"assembly_pending":True,"input_audit_pending":True}
 W(out/"generation_manifest.json",g);return {"stream_sha256":None,**g},{"audit_pass":False}
def patch_historical(source):
 H.registration=registration;H.phase_table=phase_table;H.phase_at=phase_at;H.json_sha=json_sha;H.source_identity=lambda:source_identity(source)
 H.write_checkpoint=write_checkpoint;H.read_checkpoint=read_checkpoint;H.finalize=finalize
 BASE.P=H;BASE.initialize=H.initialize;BASE.write_checkpoint=write_checkpoint;BASE.read_checkpoint=read_checkpoint
def cmd_segment(a):
 source,_=configure_source(a.source036);patch_historical(source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rp=out/"registration.json"
 if not rp.exists():W(rp,runtime_registration())
 set_registration_path(rp);snap=out/"registration_snapshot.json"
 if not snap.exists():snap.write_bytes(rp.read_bytes())
 elif snap.read_bytes()!=rp.read_bytes():raise AssertionError("registration changed")
 append_event(out,{"event":"segment_start","max_intervals":int(a.max_intervals)})
 try:
  res=BASE.collect_segment(out,int(a.max_intervals));res=dict(res or {});res.update({"protocol":"044","revision":1,"seed":SEED,"model_runs_started":0})
  W(out/"generation_status.json",res);append_event(out,{"event":"segment_complete","next_t":res.get("next_t"),"complete":bool(res.get("complete"))});print(json.dumps(res,indent=2))
 except Exception as e:
  W(out/"failure.json",{"protocol":"044","revision":1,"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),
    "resume_available":checkpoint_paths(out)[0].is_file(),"reroll":False});raise
def cmd_assemble(a):
 source,_=configure_source(a.source036);patch_historical(source);src=Path(a.generation_root);out=Path(a.output);set_registration_path(src/"registration.json");reg=registration()
 if out.exists() and any(out.iterdir()):raise FileExistsError(out)
 out.mkdir(parents=True,exist_ok=True);field=out/"fields";field.mkdir();resume=J(src/"resume_manifest.json");chunks=list(resume["chunks"])
 if int(H._verify_chunks(src,chunks))!=ROWS:raise AssertionError("coverage")
 with np.load(src/"chunks"/chunks[0]["file"],allow_pickle=False) as first:
  keys=list(first.files);schema={k:{"dtype":str(first[k].dtype),"tail_shape":list(first[k].shape[1:])} for k in keys}
  mm={k:np.lib.format.open_memmap(field/(k+".npy"),mode="w+",dtype=first[k].dtype,shape=(ROWS,)+first[k].shape[1:]) for k in keys}
 for rec in chunks:
  aa,bb=int(rec["start"]),int(rec["end"])
  with np.load(src/"chunks"/rec["file"],allow_pickle=False) as z:
   for k in keys:mm[k][aa:bb]=z[k]
  for x in mm.values():x.flush()
 common=H._common_features(mm["host_features"],mm["capacities"]);np.save(out/"common_observable_features.npy",common)
 overload=(mm["overload_ratio"]>1).astype(np.uint8);np.savez_compressed(out/"stream.npz",**mm,overload_mask=overload)
 manifest={"protocol":"044","revision":1,"scenario_id":reg["scenario_id"],"seed":SEED,"steps":STEPS,"guard_rows":1,"total_rows":ROWS,
   "stream_file":"stream.npz","stream_sha256":sha(out/"stream.npz"),"registration_sha256":sha(src/"registration.json"),
   "timeline":phase_table(reg),"logical_to_source_service":reg["logical_to_source_service"],"event_probability":.3,
   "forbidden_model_inputs":list(H.FORBIDDEN_MODEL_INPUTS),"audit_only_npz_keys":[k for k in keys if k.startswith("audit_")]+["overload_mask"],
   "model_input_keys":["host_features","demands","schedules","capacities","creation_ids","before_placement"],
   "common_observable_features_file":"common_observable_features.npy","field_schema":schema,"label_key":"raw_labels","target":reg["causality"]["target"],
   "chunk_manifest":chunks,"assembled_in_fresh_process":True,"audit_pass":None}
 W(out/"manifest.json",manifest)
 for name in ("events_final.json","applied_switches.json","generation_manifest.json","resume_manifest.json","registration_snapshot.json","registration.json"):
  p=src/name
  if p.is_file():(out/name).write_bytes(p.read_bytes())
 print(json.dumps({"assembled":True,"rows":ROWS,"stream_sha256":manifest["stream_sha256"]},indent=2))
def cmd_audit(a):
 source,_=configure_source(a.source036);patch_historical(source);gen=Path(a.generation_root);data=Path(a.data_root);ev=Path(a.evidence_root);ev.mkdir(parents=True,exist_ok=True)
 set_registration_path(gen/"registration.json");reg=registration();m=J(data/"manifest.json");field=data/"fields";gates={}
 resume=J(gen/"resume_manifest.json");chunks=list(resume["chunks"]);gates["all_chunks_rows_complete"]=int(H._verify_chunks(gen,chunks))==ROWS and len(chunks)==30 and int(chunks[-1]["end"])-int(chunks[-1]["start"])==FINAL
 post=np.load(field/"post_totals.npy",mmap_mode="r");caps=np.load(field/"capacities.npy",mmap_mode="r");saved=np.load(field/"overload_ratio.npy",mmap_mode="r");raw=np.load(field/"raw_labels.npy",mmap_mode="r")
 host=np.load(field/"host_features.npy",mmap_mode="r");sched=np.load(field/"schedules.npy",mmap_mode="r");phaseid=np.load(field/"audit_phase_ids.npy",mmap_mode="r");serviceid=np.load(field/"audit_service_ids.npy",mmap_mode="r")
 ratio=labels=finite=capacity=scheduler=True
 for aa in range(0,ROWS,200):
  bb=min(ROWS,aa+200);ca=np.asarray(caps[aa:bb],np.float64);po=np.asarray(post[aa:bb],np.float64);calc=po/ca;yy=np.where((calc>1).any(-1),calc.argmax(-1)+1,0)
  ratio&=np.array_equal(calc,np.asarray(saved[aa:bb]));labels&=np.array_equal(yy,np.asarray(raw[aa:bb]));finite&=np.isfinite(np.asarray(host[aa:bb])).all();capacity&=np.isfinite(ca).all() and (ca>0).all()
  ss=np.asarray(sched[aa:bb],np.float64);scheduler&=np.isfinite(ss).all() and (ss>=-1e-7).all() and np.allclose(ss.sum(-1),1,atol=1e-5)
 gates.update({"label_recompute_exact":bool(labels),"overload_ratio_recompute_exact":bool(ratio),"finite_causal_features":bool(finite),"scheduler_capacity_audit":bool(capacity and scheduler)})
 common=np.load(data/"common_observable_features.npy",mmap_mode="r");gates["common_features_causal_exact"]=bool(np.array_equal(np.asarray(common),H._common_features(host,caps)))
 coverage={};phase_ok=service_ok=True
 for pi,p in enumerate(phase_table(reg)):
  aa,bb=int(p["start"]),int(p["end"]);yy=np.asarray(raw[aa+1:bb+1]);pos=int((yy>0).sum());neg=int((yy==0).sum());coverage[p["name"]]={"positive":pos,"negative":neg}
  if p["logical_service"] is not None:gates["coverage_"+p["name"]]=pos>=32 and neg>=32
  phase_ok&=bool((np.asarray(phaseid[aa:bb])==pi).all());sid=-1 if p["service"] is None else int(H.SERVICE_MECHANISM_IDS[p["service"]]);service_ok&=bool((np.asarray(serviceid[aa:bb])==sid).all())
 gates["phase_annotation"]=phase_ok;gates["service_annotation"]=service_ok
 for p in phase_table(reg):
  if p["name"].endswith("_return"):
   aa=int(p["start"]);yy=np.asarray(raw[aa+1:aa+129]);gates["return128_"+p["name"]]=int((yy>0).sum())>=32 and int((yy==0).sum())>=32
 events=J(data/"events_final.json");counts={k:sum(1 for e in events if e.get("service_id")==v) for k,v in reg["logical_to_source_service"].items()}
 gates["all_four_services_have_events"]=all(counts.get(k,0)>0 for k in ("U","V","W","X"));gates["last_prediction_guard"]=raw.shape[0]==ROWS
 passed=bool(all(gates.values()));audit={"protocol":"044","revision":1,"kind":"pre_model_input_audit","seed":SEED,"model_results_seen":False,"audit_pass":passed,"reroll":False,"gates":gates,"coverage":coverage,"service_event_counts":counts}
 W(ev/"input_audit.json",audit)
 lock={"protocol":"044","revision":1,"seed":SEED,"locked":passed,"stream_sha256":sha(data/"stream.npz"),"common_features_sha256":sha(data/"common_observable_features.npy"),
   "scenario_registration_sha256":sha(SCENARIO),"runtime_registration_sha256":sha(gen/"registration.json"),"source_identity":resume["source_identity"],"chunk_sha256":[x["sha256"] for x in chunks]}
 W(ev/"input_lock.json",lock);m["audit_pass"]=passed;m["input_lock_sha256"]=sha(ev/"input_lock.json");W(data/"manifest.json",m)
 print(json.dumps({"audit_pass":passed,"gates":gates,"coverage":coverage},indent=2))
 if not passed:raise SystemExit(2)
def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
 p=sub.add_parser("segment");p.add_argument("--source036",required=True);p.add_argument("--output",required=True);p.add_argument("--max-intervals",type=int,default=200);p.set_defaults(fn=cmd_segment)
 p=sub.add_parser("assemble");p.add_argument("--source036",required=True);p.add_argument("--generation-root",required=True);p.add_argument("--output",required=True);p.set_defaults(fn=cmd_assemble)
 p=sub.add_parser("audit");p.add_argument("--source036",required=True);p.add_argument("--generation-root",required=True);p.add_argument("--data-root",required=True);p.add_argument("--evidence-root",required=True);p.set_defaults(fn=cmd_audit)
 a=ap.parse_args();a.fn(a)
if __name__=="__main__":main()
