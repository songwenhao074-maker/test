"""Fresh-process verifier for one committed Protocol044 r2 science checkpoint."""
from __future__ import annotations
import argparse,json,os,pathlib,random,sys
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_ftmoe_protocol044_science as s
import reference_protocol036_locked as r36
from protocol044_engine import Machine044
from protocol035_common import sha256_file
def J(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def W(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("C_ref","D_lin","D_no_gc","D_bounded"),required=True)
 ap.add_argument("--stage",required=True);ap.add_argument("--data");ap.add_argument("--out",required=True);a=ap.parse_args()
 root=pathlib.Path(a.stage);led=s.load_ledger(root/"budget_ledger.json");r=led["sequences"][a.arm]
 if r.get("started") is not True or r.get("completed") not in (False,True):raise RuntimeError("sequence state")
 cur=int(r["cursor"]);steps=int(r["optimizer_steps_used"]);state=str(r["state_sha256"])
 if a.arm=="C_ref":
  if not a.data:raise ValueError("C requires data")
  label="final" if cur==s.N else "cursor_%04d"%cur
  mp=root/"C_ref/checkpoints"/(label+".meta.json")
  if sha256_file(mp)!=state:raise RuntimeError("C ledger state hash")
  sess,_,_=s.make_c(a.data,root/"input_lock.json",root/"C_ref","verify")
  s.restore_c_ckpt(sess,root/"C_ref",label)
  ok=sess.cursor==cur and int(sess.updates)==steps
  detail={"cursor":int(sess.cursor),"optimizer_steps":int(sess.updates),"learner_hash":sess.learner_state_hash()}
 elif a.arm=="D_lin":
  p=root/"D_lin/checkpoints"/("cursor_%04d.pt"%cur)
  if sha256_file(p)!=state:raise RuntimeError("Dlin ledger state hash")
  tp=root/"C_ref/feature_tape.npz";tape_sha=sha256_file(tp)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(3501);b=r36.LinearCorrection()
  opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
  x=s.restore_branch(p,b,opt,tape_sha)
  ok=int(x["cursor"])==cur and int(x["version"])==steps
  detail={"cursor":int(x["cursor"]),"optimizer_steps":int(x["version"]),"model_hash":r36.bhash(b)}
 else:
  p=root/a.arm/"checkpoints/latest.pt"
  if sha256_file(p)!=state:raise RuntimeError("dynamic ledger state hash")
  src=s.load_dynamic_source(root/"C_ref/feature_tape.npz",root/"D_lin/predictions.npz",root/"C_ref/update_batches.json")
  m=Machine044.restore(src,root/a.arm,p,arm=a.arm,allow_gradient=True,real_science=True,strict_journal=True)
  ok=int(m.cursor)==cur and int(m.actual_optimizer_calls)==steps
  detail={"cursor":int(m.cursor),"optimizer_steps":int(m.actual_optimizer_calls),"active_ids":list(m.active_ids),"next_substep":m.next_substep}
 out={"protocol":"044","execution_revision":2,"arm":a.arm,"cursor":cur,"optimizer_steps_used":steps,
      "fresh_process_restore":bool(ok),"model_forward_calls":0,"optimizer_calls_during_verification":0,"detail":detail}
 W(a.out,out);print(json.dumps(out,indent=2))
 if not ok:raise SystemExit(2)
if __name__=="__main__":main()
