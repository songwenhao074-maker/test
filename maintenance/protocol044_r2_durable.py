"""Protocol-044 revision2 durable artifact/receipt primitives for generation and science."""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, time
import numpy as np

GEN_KEY="protocol044_revision2_seed4401_reconstruction1"
SCI_KEY="protocol044_revision1_stageS_seed4401_sequence"
R1_PLAN_SHA="a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51"
R2_PLAN_SHA="823365226f0e418bcc5dbdc79e0e1b77f5adf72c3e404b67fb28fdffd9cc20b3"
SCENARIO_SHA="6e6bd03efc0d84f5e885eacf36f1ac80e770198703463cc051216ad1a741683f"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def J(p): return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def files(root):
    root=pathlib.Path(root)
    skip={"state_manifest.json","artifact_receipt.json"}
    rows=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if pathlib.Path(rel).name in skip: continue
        rows.append({"path":rel,"size":p.stat().st_size,"sha256":sha(p)})
    return rows
def verify_contracts(repo):
    repo=pathlib.Path(repo)
    checks={
      "r1_plan":sha(repo/"artifacts/ftmoe_online/protocol_044/plan.json")==R1_PLAN_SHA,
      "r2_plan":sha(repo/"artifacts/ftmoe_online/protocol_044/revision_002/plan.json")==R2_PLAN_SHA,
      "scenario":sha(repo/"artifacts/ftmoe_online/protocol_044/scenario_registration.json")==SCENARIO_SHA}
    if not all(checks.values()): raise RuntimeError("contract hash failure "+repr(checks))
    return checks
def build_manifest(root,kind,execution_sha,next_t=None,cursor=None,arm=None,parent_receipt=None,out=None):
    root=pathlib.Path(root); verify_contracts(pathlib.Path(__file__).resolve().parents[1])
    parent_sha=None
    if parent_receipt:
        parent_sha=sha(parent_receipt)
    if kind=="generation":
        rm=J(root/"resume_manifest.json")
        if next_t is None: next_t=int(rm["next_t"])
        if int(rm["next_t"])!=int(next_t): raise RuntimeError("generation cursor mismatch")
        required=("resume_state.dill","resume_manifest.json","registration.json","registration_snapshot.json")
        if not all((root/x).is_file() for x in required): raise RuntimeError("generation state incomplete")
    elif kind=="science":
        if cursor is None or arm is None: raise ValueError("science manifest requires cursor/arm")
    else: raise ValueError(kind)
    rows=files(root)
    if not rows: raise RuntimeError("empty state")
    m={"protocol":"044","execution_revision":2,"science_config_revision":1,"kind":kind,"execution_sha":str(execution_sha),
       "generation_key":GEN_KEY,"science_key":SCI_KEY,"r1_plan_sha256":R1_PLAN_SHA,"r2_plan_sha256":R2_PLAN_SHA,
       "scenario_sha256":SCENARIO_SHA,"parent_receipt_sha256":parent_sha,"files":rows,"file_count":len(rows),
       "total_bytes":sum(x["size"] for x in rows),"created_unix":time.time()}
    if kind=="generation":m.update({"next_t":int(next_t),"model_runs_started":0})
    else:m.update({"arm":str(arm),"cursor":int(cursor)})
    out=pathlib.Path(out or root/"state_manifest.json");W(out,m);return m
def verify_manifest(root,manifest,expected_parent=None):
    root=pathlib.Path(root);m=J(manifest)
    if (m.get("protocol"),m.get("execution_revision"),m.get("science_config_revision"))!=("044",2,1):raise RuntimeError("manifest identity")
    if m.get("generation_key")!=GEN_KEY or m.get("science_key")!=SCI_KEY:raise RuntimeError("key identity")
    if (m.get("r1_plan_sha256"),m.get("r2_plan_sha256"),m.get("scenario_sha256"))!=(R1_PLAN_SHA,R2_PLAN_SHA,SCENARIO_SHA):raise RuntimeError("contract identity")
    if expected_parent is not None and m.get("parent_receipt_sha256")!=expected_parent:raise RuntimeError("parent receipt mismatch")
    got=[]
    for r in m["files"]:
        p=root/r["path"]
        if not p.is_file() or p.stat().st_size!=int(r["size"]) or sha(p)!=r["sha256"]:got.append(r["path"])
    if got:raise RuntimeError("state tree mismatch "+",".join(got[:10]))
    current={x["path"] for x in files(root)};expected={x["path"] for x in m["files"]}
    if current!=expected:raise RuntimeError("state tree extra/missing files")
    return m
def fresh_generation_restore(source036,root,manifest):
    import protocol044_stream as s
    root=pathlib.Path(root);m=verify_manifest(root,manifest)
    if m["kind"]!="generation":raise RuntimeError("not generation state")
    source,_=s.configure_source(source036);s.patch_historical(source);s.set_registration_path(root/"registration.json")
    arrays=s.H._allocate(s.ROWS);reg_sha=s.sha(root/"registration.json")
    payload,chunks,chunk_end=s.H.read_checkpoint(root,reg_sha,arrays)
    if int(payload["next_t"])!=int(m["next_t"]):raise RuntimeError("fresh restore cursor mismatch")
    if int(chunk_end)>int(m["next_t"]):raise RuntimeError("chunk coverage exceeds cursor")
    return {"next_t":int(payload["next_t"]),"chunk_end":int(chunk_end),"chunks":len(chunks),"fresh_restore":True}
def receipt(info,manifest,run_id,attempt,out,parent=None):
    info=J(info);m=J(manifest)
    if not info.get("id") or not info.get("name") or not info.get("digest"):raise RuntimeError("artifact metadata incomplete")
    parent_sha=None if parent is None else sha(parent)
    if m.get("parent_receipt_sha256")!=parent_sha:raise RuntimeError("manifest/receipt parent mismatch")
    r={"protocol":"044","execution_revision":2,"science_config_revision":1,"kind":m["kind"],"generation_key":GEN_KEY,"science_key":SCI_KEY,
       "execution_sha":m["execution_sha"],"parent_receipt_sha256":parent_sha,"run_id":str(run_id),"attempt":int(attempt),
       "artifact_id":int(info["id"]),"artifact_name":info["name"],"artifact_digest":info["digest"],"artifact_size":int(info["size_in_bytes"]),
       "artifact_expires_at":info.get("expires_at"),"state_manifest_sha256":sha(manifest),"committed":True,"model_runs_started":0 if m["kind"]=="generation" else None}
    if m["kind"]=="generation":r["next_t"]=int(m["next_t"])
    elif m["kind"]=="science":r.update({"arm":m["arm"],"cursor":int(m["cursor"])})
    elif m["kind"]=="synthetic_transaction_fixture":r.update({"cursor":int(m["cursor"]),"fixture_mode":True})
    else:raise RuntimeError("unsupported receipt kind "+str(m.get("kind")))
    W(out,r);return r
def verify_receipt(receipt_path,manifest):
    r=J(receipt_path);m=J(manifest)
    if r.get("committed") is not True or r.get("state_manifest_sha256")!=sha(manifest):raise RuntimeError("receipt state mismatch")
    if r.get("parent_receipt_sha256")!=m.get("parent_receipt_sha256"):raise RuntimeError("receipt parent mismatch")
    if r["kind"]=="generation" and int(r["next_t"])!=int(m["next_t"]):raise RuntimeError("receipt cursor")
    if r["kind"]=="science" and (r.get("arm")!=m.get("arm") or int(r["cursor"])!=int(m["cursor"])):raise RuntimeError("science receipt cursor")
    if r["kind"]=="synthetic_transaction_fixture" and int(r["cursor"])!=int(m["cursor"]):raise RuntimeError("fixture receipt cursor")
    return r

# Synthetic backend used only for cross-runner transaction tests.
def synthetic_init(root):
    root=pathlib.Path(root);shutil.rmtree(root,ignore_errors=True);root.mkdir(parents=True)
    rng=np.random.default_rng(4401002)
    payload=rng.integers(0,2**31,size=32,dtype=np.int64)
    np.save(root/"synthetic_rng.npy",payload)
    W(root/"synthetic_state.json",{"fixture":True,"cursor":0,"counter":0,"rng_sha256":sha(root/"synthetic_rng.npy")})
def synthetic_advance(root,rows):
    root=pathlib.Path(root);st=J(root/"synthetic_state.json");a=np.load(root/"synthetic_rng.npy")
    start=int(st["cursor"]);rows=int(rows);end=start+rows
    chunk=np.asarray([(int(a[i%len(a)])^(i*2654435761))&0x7fffffff for i in range(start,end)],dtype=np.int64)
    np.save(root/("chunk_%04d_%04d.npy"%(start,end)),chunk)
    st.update({"cursor":end,"counter":int(st["counter"])+rows});W(root/"synthetic_state.json",st)
def synthetic_manifest(root,execution_sha,parent=None,out=None):
    root=pathlib.Path(root);st=J(root/"synthetic_state.json");rows=files(root)
    m={"protocol":"044","execution_revision":2,"science_config_revision":1,"kind":"synthetic_transaction_fixture","fixture_mode":True,
       "execution_sha":execution_sha,"generation_key":GEN_KEY,"science_key":SCI_KEY,"parent_receipt_sha256":None if parent is None else sha(parent),
       "cursor":int(st["cursor"]),"files":rows,"file_count":len(rows),"total_bytes":sum(x["size"] for x in rows)}
    W(out or root/"state_manifest.json",m)
def synthetic_verify(root,manifest,expected_parent=None):
    root=pathlib.Path(root);m=J(manifest)
    if m.get("fixture_mode") is not True:raise RuntimeError("not fixture")
    if expected_parent is not None and m.get("parent_receipt_sha256")!=expected_parent:raise RuntimeError("fixture parent")
    exp={r["path"]:(r["size"],r["sha256"]) for r in m["files"]}
    cur={r["path"]:(r["size"],r["sha256"]) for r in files(root)}
    if exp!=cur:raise RuntimeError("synthetic artifact corruption")
    if int(J(root/"synthetic_state.json")["cursor"])!=int(m["cursor"]):raise RuntimeError("synthetic cursor")
    return m

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("manifest");p.add_argument("--root",required=True);p.add_argument("--kind",choices=("generation","science"),required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--next-t",type=int);p.add_argument("--cursor",type=int);p.add_argument("--arm");p.add_argument("--parent-receipt");p.add_argument("--out",required=True)
    p=sp.add_parser("verify");p.add_argument("--root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--expected-parent")
    p=sp.add_parser("fresh-generation-restore");p.add_argument("--source036",required=True);p.add_argument("--root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--out")
    p=sp.add_parser("receipt");p.add_argument("--info",required=True);p.add_argument("--manifest",required=True);p.add_argument("--run-id",required=True);p.add_argument("--attempt",type=int,required=True);p.add_argument("--out",required=True);p.add_argument("--parent")
    p=sp.add_parser("verify-receipt");p.add_argument("--receipt",required=True);p.add_argument("--manifest",required=True)
    p=sp.add_parser("synthetic-init");p.add_argument("--root",required=True)
    p=sp.add_parser("synthetic-advance");p.add_argument("--root",required=True);p.add_argument("--rows",type=int,required=True)
    p=sp.add_parser("synthetic-manifest");p.add_argument("--root",required=True);p.add_argument("--execution-sha",required=True);p.add_argument("--parent");p.add_argument("--out",required=True)
    p=sp.add_parser("synthetic-verify");p.add_argument("--root",required=True);p.add_argument("--manifest",required=True);p.add_argument("--expected-parent")
    a=ap.parse_args()
    if a.cmd=="manifest":build_manifest(a.root,a.kind,a.execution_sha,a.next_t,a.cursor,a.arm,a.parent_receipt,a.out)
    elif a.cmd=="verify":print(json.dumps(verify_manifest(a.root,a.manifest,a.expected_parent),indent=2))
    elif a.cmd=="fresh-generation-restore":
        x=fresh_generation_restore(a.source036,a.root,a.manifest);print(json.dumps(x,indent=2));W(a.out,x) if a.out else None
    elif a.cmd=="receipt":receipt(a.info,a.manifest,a.run_id,a.attempt,a.out,a.parent)
    elif a.cmd=="verify-receipt":print(json.dumps(verify_receipt(a.receipt,a.manifest),indent=2))
    elif a.cmd=="synthetic-init":synthetic_init(a.root)
    elif a.cmd=="synthetic-advance":synthetic_advance(a.root,a.rows)
    elif a.cmd=="synthetic-manifest":synthetic_manifest(a.root,a.execution_sha,a.parent,a.out)
    elif a.cmd=="synthetic-verify":print(json.dumps(synthetic_verify(a.root,a.manifest,a.expected_parent),indent=2))
if __name__=="__main__":main()
