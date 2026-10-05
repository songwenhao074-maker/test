"""Protocol-044 revision2 targeted production-entrypoint acceptance on synthetic data only."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, shutil, subprocess, sys
import numpy as np
import torch

ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
N=64
SCI=ROOT/"run_ftmoe_protocol044_science.py"
PLAN_SHA="a27a6cca6abde40b6395876eccd3db5f550a044e952a701fe63aa2e4306a5f51"
R2_SHA="823365226f0e418bcc5dbdc79e0e1b77f5adf72c3e404b67fb28fdffd9cc20b3"
SCENARIO_SHA="6e6bd03efc0d84f5e885eacf36f1ac80e770198703463cc051216ad1a741683f"
E_ID=11296492568
E_DIGEST="sha256:90b6bab6a149c46da0aa19e551c02f76b5665a27a83e97d7de192e21e08b6591"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def J(p): return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def env():
    e=os.environ.copy();e.update({"P044_R2_ENGINEERING_FIXTURE":"1","P044_R2_ENGINEERING_FIXTURE_N":str(N),
      "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONHASHSEED":"1"})
    return e
def run(cmd,cwd=ROOT,expect=0):
    q=subprocess.run([str(x) for x in cmd],cwd=str(cwd),env=env(),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if q.returncode!=expect:
        raise RuntimeError("command rc=%d expected=%d\n%s"%(q.returncode,expect,q.stdout[-12000:]))
    return q.stdout
def npz_equal(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        if set(x.files)!=set(y.files): return False,{"keys_a":x.files,"keys_b":y.files}
        bad=[]
        for k in x.files:
            aa=x[k];bb=y[k]
            if aa.dtype.kind=="f": ok=np.array_equal(aa,bb,equal_nan=True)
            else: ok=np.array_equal(aa,bb)
            if not ok: bad.append(k)
        return len(bad)==0,{"bad":bad}
def obj_equal(a,b):
    if torch.is_tensor(a) and torch.is_tensor(b): return bool(torch.equal(a,b))
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray): return bool(np.array_equal(a,b,equal_nan=True) if a.dtype.kind=="f" else np.array_equal(a,b))
    if isinstance(a,dict) and isinstance(b,dict): return set(a)==set(b) and all(obj_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)): return len(a)==len(b) and all(obj_equal(x,y) for x,y in zip(a,b))
    return a==b

def make_fixture(root,execution_sha):
    root=pathlib.Path(root);shutil.rmtree(root,ignore_errors=True);(root/"data").mkdir(parents=True)
    rows=N+1;rng=np.random.default_rng(4401002)
    host=(.05+rng.random((rows,16,7))*.02).astype(np.float32)
    demands=(.02+rng.random((rows,16,7))*.01).astype(np.float32)
    sched=np.repeat(np.eye(16,dtype=np.float32)[None],rows,axis=0)
    caps=np.zeros((rows,16,3),np.float64);caps[...,0]=100.;caps[...,1]=8.;caps[...,2]=100.
    ids=np.repeat(np.arange(16,dtype=np.int64)[None],rows,axis=0)
    before=ids.copy()
    raw=np.zeros((rows,16),np.int64)
    for t in range(rows):
        raw[t,(np.arange(16)+t)%4==0]=1
        raw[t,(np.arange(16)+t)%11==0]=2
        raw[t,(np.arange(16)+t)%17==0]=3
    np.savez_compressed(root/"data/stream.npz",host_features=host,demands=demands,schedules=sched,capacities=caps,
      creation_ids=ids,before_placement=before,raw_labels=raw)
    sh=sha(root/"data/stream.npz")
    manifest={"protocol":"044","revision":1,"seed":4401,"steps":N,"guard_rows":1,"total_rows":rows,"stream_file":"stream.npz","stream_sha256":sh,
      "timeline":[{"name":"F0","start":0,"end":N,"length":N,"logical_service":None,"service":None}]}
    W(root/"data/manifest.json",manifest)
    W(root/"input_lock.json",{"protocol":"044","revision":1,"seed":4401,"locked":True,"stream_sha256":sh,"engineering_fixture":True})
    gate={"protocol":"044","execution_revision":2,"science_config_revision":1,"fixture_mode":True,"E_recovery_gate":True,
      "inherited_E_pass":True,"compatibility_pass":True,"production_entrypoint_pass":True,"generation_recovery_pass":True,
      "remote_transaction_pass":True,"budget_idempotency_pass":True,"publication_pass":True,"execution_sha":str(execution_sha),
      "r1_plan_sha256":PLAN_SHA,"r2_plan_sha256":R2_SHA,"scenario_sha256":SCENARIO_SHA,
      "inherited_E":{"artifact_id":E_ID,"artifact_digest":E_DIGEST}}
    W(root/"gate.json",gate)
    return root

def cli(root,chain,arm,stop=None,resume=None):
    root=pathlib.Path(root);base=root/chain;ledger=base/"ledger.json";out=base/arm
    args=[sys.executable,SCI]
    common=["--input-lock",root/"input_lock.json","--gate",root/"gate.json","--execution-sha",J(root/"gate.json")["execution_sha"],
            "--ledger",ledger,"--out",out,"--run-id","engineering"]
    if arm=="C_ref":
        args+=["C_ref","--data",root/"data",*common]
        if resume: args+=["--resume-label",resume]
    elif arm=="D_lin":
        args+=["D_lin","--feature-tape",base/"C_ref/feature_tape.npz","--update-batches",base/"C_ref/update_batches.json",*common]
        if resume: args+=["--resume-from",resume]
    else:
        args+=["dynamic","--arm",arm,"--feature-tape",base/"C_ref/feature_tape.npz","--b-predictions",base/"D_lin/predictions.npz",
               "--update-batches",base/"C_ref/update_batches.json",*common]
        if resume: args+=["--resume-from",resume]
    if stop is not None:args+=["--stop-cursor",str(stop)]
    return run(args)

def prepare_chain(root,chain):
    import run_ftmoe_protocol044_science as s
    old=os.environ.get("P044_R2_ENGINEERING_FIXTURE");os.environ["P044_R2_ENGINEERING_FIXTURE"]="1";s.N=N
    try: W(pathlib.Path(root)/chain/"ledger.json",s.default_ledger())
    finally:
        if old is None: os.environ.pop("P044_R2_ENGINEERING_FIXTURE",None)
        else: os.environ["P044_R2_ENGINEERING_FIXTURE"]=old

def copy_fresh(src,dst):
    src=pathlib.Path(src);dst=pathlib.Path(dst);shutil.rmtree(dst,ignore_errors=True);shutil.copytree(src,dst);shutil.rmtree(src)

def run_continuous(root):
    prepare_chain(root,"continuous")
    for arm in ("C_ref","D_lin","D_no_gc","D_bounded"): cli(root,"continuous",arm,stop=N)

def run_segmented(root):
    root=pathlib.Path(root);prepare_chain(root,"segmented");base=root/"segmented"
    # C
    tmp=base/"C_ref_tmp"; final=base/"C_ref"
    cli(root,"segmented","C_ref",stop=32)
    # CLI wrote directly to final; move to new directory via temporary rename/copy cycle.
    shutil.copytree(final,tmp);shutil.rmtree(final);shutil.copytree(tmp,final);shutil.rmtree(tmp)
    cli(root,"segmented","C_ref",stop=N,resume="cursor_0032")
    # D_lin
    cli(root,"segmented","D_lin",stop=32)
    tmp=base/"D_lin_tmp";shutil.copytree(base/"D_lin",tmp);shutil.rmtree(base/"D_lin");shutil.copytree(tmp,base/"D_lin");shutil.rmtree(tmp)
    cli(root,"segmented","D_lin",stop=N,resume=base/"D_lin/checkpoints/cursor_0032.pt")
    # dynamic arms
    for arm in ("D_no_gc","D_bounded"):
        cli(root,"segmented",arm,stop=32)
        tmp=base/(arm+"_tmp");shutil.copytree(base/arm,tmp);shutil.rmtree(base/arm);shutil.copytree(tmp,base/arm);shutil.rmtree(tmp)
        cli(root,"segmented",arm,stop=N,resume=base/arm/"checkpoints/latest.pt")

def compare(root):
    root=pathlib.Path(root);a=root/"continuous";b=root/"segmented";rows=[]
    for arm in ("C_ref","D_lin","D_no_gc","D_bounded"):
        files=["predictions.npz"]
        if arm=="C_ref":files+=["feature_tape.npz"]
        details={}
        ok=True
        for fn in files:
            x,d=npz_equal(a/arm/fn,b/arm/fn);details[fn]=d;ok&=x
        rows.append({"id":arm+"_outputs_exact","pass":bool(ok),"details":details})
    rows.append({"id":"C_updates_exact","pass":J(a/"C_ref/update_batches.json")==J(b/"C_ref/update_batches.json")})
    rows.append({"id":"Dlin_updates_exact","pass":J(a/"D_lin/update_log.json")==J(b/"D_lin/update_log.json")})
    # C final learner hash and checkpoint payload.
    ca=J(a/"C_ref/checkpoints/final.meta.json");cb=J(b/"C_ref/checkpoints/final.meta.json")
    rows.append({"id":"C_learner_exact","pass":ca["learner_hash"]==cb["learner_hash"]})
    # Dlin model/Adam/RNG payload exact.
    xa=torch.load(a/"D_lin/checkpoints/cursor_0064.pt",map_location="cpu",weights_only=False)
    xb=torch.load(b/"D_lin/checkpoints/cursor_0064.pt",map_location="cpu",weights_only=False)
    rows.append({"id":"Dlin_full_checkpoint_state_exact","pass":obj_equal(xa,xb)})
    # Dynamic semantic endpoint incl. model/Adam/RNG/output/audit chain.
    import run_ftmoe_protocol044_science as s
    from protocol044_engine import Machine044,semantic_digest
    old=os.environ.get("P044_R2_ENGINEERING_FIXTURE");os.environ["P044_R2_ENGINEERING_FIXTURE"]="1";s.N=N
    try:
        for arm in ("D_no_gc","D_bounded"):
            srca=s.load_dynamic_source(a/"C_ref/feature_tape.npz",a/"D_lin/predictions.npz",a/"C_ref/update_batches.json")
            srcb=s.load_dynamic_source(b/"C_ref/feature_tape.npz",b/"D_lin/predictions.npz",b/"C_ref/update_batches.json")
            ma=Machine044.restore(srca,a/arm,a/arm/"checkpoints/latest.pt",arm=arm,allow_gradient=True,real_science=False,strict_journal=True)
            mb=Machine044.restore(srcb,b/arm,b/arm/"checkpoints/latest.pt",arm=arm,allow_gradient=True,real_science=False,strict_journal=True)
            rows.append({"id":arm+"_semantic_state_exact","pass":semantic_digest(ma)==semantic_digest(mb),"continuous":semantic_digest(ma),"segmented":semantic_digest(mb)})
    finally:
        if old is None: os.environ.pop("P044_R2_ENGINEERING_FIXTURE",None)
        else: os.environ["P044_R2_ENGINEERING_FIXTURE"]=old
    la=J(a/"ledger.json");lb=J(b/"ledger.json")
    stepsa={k:v["optimizer_steps_used"] for k,v in la["sequences"].items()}
    stepsb={k:v["optimizer_steps_used"] for k,v in lb["sequences"].items()}
    rows.append({"id":"budget_exact","pass":stepsa==stepsb and la["total_optimizer_steps_used"]==lb["total_optimizer_steps_used"],"continuous":stepsa,"segmented":stepsb})
    return rows

def negative(root):
    root=pathlib.Path(root);rows=[]
    sys.path.insert(0,str(ROOT));import run_ftmoe_protocol044_science as s
    s.N=N
    gate=J(root/"gate.json")
    # Fixture gate must be rejected without engineering env.
    old=os.environ.pop("P044_R2_ENGINEERING_FIXTURE",None)
    p=root/"negative_ledger.json";W(p,s.default_ledger())
    try:
        try:s.validate_gate(gate,root/"input_lock.json",gate["execution_sha"],"C_ref",p,0,resume=False);rej=False
        except Exception:rej=True
    finally:
        if old is not None:os.environ["P044_R2_ENGINEERING_FIXTURE"]=old
    rows.append({"id":"fixture_gate_rejected_in_production","pass":rej})
    os.environ["P044_R2_ENGINEERING_FIXTURE"]="1"
    def rejects(cid,mut,seq="C_ref",resume=False):
        x=s.default_ledger();mut(x);W(p,x)
        try:s.validate_gate(gate,root/"input_lock.json",gate["execution_sha"],seq,p,0,resume=resume);ok=False;err=None
        except Exception as e:ok=True;err=type(e).__name__+":"+str(e)
        rows.append({"id":cid,"pass":ok,"error":err,"optimizer_calls":0})
    rejects("started_without_resume",lambda x:x["sequences"]["C_ref"].update({"started":True}))
    rejects("resume_without_state",lambda x:x["sequences"]["C_ref"].update({"started":True}),resume=True)
    rejects("completed_rejected",lambda x:x["sequences"]["C_ref"].update({"started":True,"completed":True}))
    rejects("previous_sequence_incomplete",lambda x:None,seq="D_lin")
    rejects("negative_budget",lambda x:(x["sequences"]["C_ref"].update({"optimizer_steps_used":-1}),x.update({"total_optimizer_steps_used":-1})))
    x=s.default_ledger();x["sequences"]["C_ref"].update({"started":True,"cursor":32,"optimizer_steps_used":0,"state_sha256":"bad"});W(p,x)
    try:s.assert_resume_binding(p,"C_ref",32,"good");ok=False
    except Exception:ok=True
    rows.append({"id":"wrong_state_hash_rejected","pass":ok,"optimizer_calls":0})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--execution-sha",required=True);a=ap.parse_args()
    root=make_fixture(a.root,a.execution_sha)
    run_continuous(root);run_segmented(root)
    rows=compare(root)+negative(root)
    out={"protocol":"044","execution_revision":2,"science_config_revision":1,"kind":"targeted_production_entrypoint_acceptance",
      "synthetic_only":True,"fixture_horizon":N,"real_model_forwards":0,"real_gradients":0,"fixtures":rows,"all_pass":bool(all(r["pass"] for r in rows))}
    W(root/"acceptance.json",out);print(json.dumps(out,indent=2))
    if not out["all_pass"]:raise SystemExit(2)
if __name__=="__main__":main()
