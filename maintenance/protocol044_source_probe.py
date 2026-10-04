"""Protocol-044 fail-fast immutable source/input probe. No model forward and no gradient."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

PLAN_SHA="a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51"
SRC036_COMMIT="b620467aab4283e52d129924769c374124bbb4bf"
BASE_CKPT="artifacts/ftmoe_online/protocol_020/s6/adapted_v4_seed1/best.pt"
BASE_CKPT_SHA="10c44bdb0ea1a3134933d6a7eb5be98711ef4e48bd791594e4d8792519dfe03b"
GOBI_CKPT="scheduler/BaGTI/checkpoints/energy_latency_16_Trained.ckpt"
GOBI_CKPT_SHA="43574c7d2a3884cc1adbab5fd2166c7473476aa0e5fb500a59b383a9cec5a596"
INIT_FROZEN="ddf996d3060e9c0c8bc8c1983764c4e9392570fd92979e246a0548acdc9e2e58"
INIT_NORM="28b40256ac35b649fd9865677685ef860d0262c100ce7183fad563e7bed0ab0c"

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def git_blob(root,commit,path):
    return subprocess.check_output(["git","-C",str(root),"show",commit+":"+path])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True); ap.add_argument("--artifact035",required=True); ap.add_argument("--artifact033",required=True)
    ap.add_argument("--out",required=True); a=ap.parse_args()
    root=Path(a.repo).resolve(); plan_path=root/"artifacts/ftmoe_online/protocol_044/plan.json"; scen_path=root/"artifacts/ftmoe_online/protocol_044/scenario_registration.json"
    p=J(plan_path); scen=J(scen_path)
    checks={}
    checks["plan_sha"]=sha(plan_path)==PLAN_SHA
    checks["identity"]=(p.get("protocol"),p.get("revision"))==("044",1) and (scen.get("protocol"),scen.get("revision"))==("044",1)
    source_hashes={}
    for rel,exp in p["source036"]["source_files"].items():
        got=sha_bytes(git_blob(root,SRC036_COMMIT,rel)); source_hashes[rel]={"expected":exp,"actual":got,"pass":got==exp}
    checks["registered_source036_files"]=all(x["pass"] for x in source_hashes.values())
    base=root/BASE_CKPT; gobi=root/GOBI_CKPT
    checks["base_checkpoint_present"]=base.is_file()
    checks["base_checkpoint_sha"]=base.is_file() and sha(base)==BASE_CKPT_SHA
    checks["gobi_checkpoint_present"]=gobi.is_file()
    checks["gobi_checkpoint_sha"]=gobi.is_file() and sha(gobi)==GOBI_CKPT_SHA
    a35=Path(a.artifact035); a33=Path(a.artifact033)
    init=list(a35.rglob("initialization.json")); tapes=list(a35.rglob("feature_tape.npz")); batches=list(a35.rglob("update_batches.json")); resumes=list(a35.rglob("final_resume_state.pt"))
    checks["artifact035_required_files"]=bool(init and tapes and batches and resumes)
    initrow=J(init[0]) if init else {}
    checks["035_frozen_base_chain"]=initrow.get("frozen_base_sha256")==INIT_FROZEN
    checks["035_normalization_chain"]=initrow.get("normalization_buffers_sha256")==INIT_NORM
    checks["035_model_seed"]=initrow.get("model_seed")==1
    checks["artifact033_stream_present"]=bool(list(a33.rglob("stream.npz")) and list(a33.rglob("manifest.json")))
    checks["scenario_seed"]=scen.get("replay_seed")==4401 and scen.get("model_seed")==1 and scen.get("branch_seed")==3501
    checks["scenario_geometry"]=scen.get("prediction_intervals")==5952 and scen.get("raw_rows")==5953 and scen["generation"]["chunks_total"]==30 and scen["generation"]["last_chunk_raw_rows"]==153
    checks["no_model_forward"]=True; checks["no_gradient"]=True
    report={"protocol":"044","revision":1,"kind":"source_probe","checks":checks,"all_pass":bool(all(checks.values())),
      "source036_commit":SRC036_COMMIT,"source_hashes":source_hashes,
      "resolved_base_checkpoint":{"path":BASE_CKPT,"sha256":None if not base.is_file() else sha(base),"expected":BASE_CKPT_SHA},
      "resolved_gobi_checkpoint":{"path":GOBI_CKPT,"sha256":None if not gobi.is_file() else sha(gobi),"expected":GOBI_CKPT_SHA},
      "035_initialization":initrow,
      "artifact035_paths":{"initialization":None if not init else str(init[0]),"feature_tape":None if not tapes else str(tapes[0]),"update_batches":None if not batches else str(batches[0]),"final_resume":None if not resumes else str(resumes[0])}}
    W(a.out,report); print(json.dumps(report,indent=2))
    if not report["all_pass"]: raise SystemExit(2)
if __name__=="__main__": main()
