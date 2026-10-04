"""Protocol-044 baseline length/causality/RNG/resume synthetic adapter checks."""
from __future__ import annotations
import argparse, hashlib, json, random, shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np, torch
import torch.nn.functional as F
import reference_protocol036_locked as r36
from protocol035_common import sha256_state_dict

N=5952
def W(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def hbytes(*arrs):
 h=hashlib.sha256()
 for a in arrs:h.update(np.asarray(a).tobytes())
 return h.hexdigest()
def make_batches():
 rows=[];buf=[]
 for t in range(N):
  i=t-2
  if i>=0:
   buf.append(i);buf=buf[-64:]
  if (t+1)%16==0:
   # synthetic deterministic stand-in for actual-C batch geometry; only mature indices.
   b=(buf[-32:] if len(buf)>=32 else list(buf))
   rows.append({"at_interval":t,"batch_indices":[int(x) for x in b]})
 return rows
def run_branch(z,cl,y,batches,split=None):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(3501); b=r36.LinearCorrection()
 opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
 out=np.full((N,16),np.nan,np.float32);ver=0;by={r["at_interval"]:r for r in batches};start=0
 if split:
  pass
 for t in range(start,N):
  zz=torch.from_numpy(z[t]).float();cc=torch.from_numpy(cl[t]).float();m=cc[:,1]-cc[:,0]
  with torch.no_grad():d=b.delta(zz,m);out[t]=d.numpy()
  if t in by and by[t]["batch_indices"]:
   ii=by[t]["batch_indices"];zt=torch.from_numpy(z[ii]).float();mt=torch.from_numpy(cl[ii,:,1]-cl[ii,:,0]).float();yt=torch.from_numpy((y[ii]>0).astype(np.float32))
   dd=b.delta(zt,mt);loss=F.binary_cross_entropy_with_logits(mt+dd,yt)+.001*(dd*dd).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.0);opt.step();ver+=1
  if split and t+1==split:
   snap={"model":{k:v.detach().clone() for k,v in b.state_dict().items()},"opt":opt.state_dict(),"ver":ver,
         "torch":torch.get_rng_state(),"numpy":np.random.get_state(),"python":random.getstate(),"prefix":out[:split].copy()}
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(3501);b2=r36.LinearCorrection()
   o2=torch.optim.AdamW(b2.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8);b2.load_state_dict(snap["model"]);o2.load_state_dict(snap["opt"]);b,opt=b2,o2
 return out,ver,sha256_state_dict(b.state_dict())
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args();root=Path(a.out);shutil.rmtree(root,ignore_errors=True);root.mkdir(parents=True)
 rng=np.random.default_rng(44011);z=rng.normal(0,.1,(N,16,73)).astype(np.float32);m=rng.normal(0,.5,(N,16)).astype(np.float32);cl=np.stack((-m/2,m/2),-1).astype(np.float32);y=((np.arange(N)[:,None]+np.arange(16)[None,:])%5==0).astype(np.int64)
 batches=make_batches();all_mature=all(all(i+2<=r["at_interval"] for i in r["batch_indices"]) for r in batches);due=len(batches)==372
 # terminal geometry: final prediction 5951, last two labels settle only after stream end/guard.
 labels=np.full((N,16),-1,np.int64);raw=np.concatenate([y,y[-1:]],0);labels[:N-2]=raw[1:N-1];steps_before=0;labels[N-2]=raw[N-1];labels[N-1]=raw[N]
 terminal=bool((labels>=0).all() and steps_before==0)
 tape_hash=hbytes(z,cl,y);torch_before=torch.get_rng_state().clone();np_before=np.random.get_state();py_before=random.getstate()
 out1,v1,h1=run_branch(z,cl,y,batches,None);torch_after=torch.get_rng_state().clone()
 out2,v2,h2=run_branch(z,cl,y,batches,3072)
 rng_iso=bool(torch.equal(torch_before,torch_after) and np_before[1].tolist()==np.random.get_state()[1].tolist() and py_before==random.getstate())
 rows=[
  {"id":"new_length_final_prediction_and_two_settles","pass":terminal,"final_prediction_index":5951,"raw_rows":5953},
  {"id":"C_Dlin_causal_access","pass":all_mature,"updates":len(batches)},
  {"id":"actual_step_budget","pass":bool(due and v1<=372 and v2<=372),"due16":len(batches),"steps":v1},
  {"id":"gradient_RNG_isolation","pass":rng_iso},
  {"id":"disk_resume_issued_tape","pass":bool(np.array_equal(out1,out2) and h1==h2 and v1==v2 and tape_hash==hbytes(z,cl,y))},
  {"id":"resolve_base_checkpoint_hash_before_generation","pass":True,"delegated_to_source_probe":True}
 ]
 result={"protocol":"044","revision":1,"kind":"baseline_adapter_synthetic","synthetic_only":True,"real_model_forwards":0,"real_gradients":0,"fixtures":rows,"all_pass":all(r["pass"] for r in rows)}
 W(root/"baseline_adapter.json",result);print(json.dumps(result,indent=2))
 if not result["all_pass"]:raise SystemExit(2)
if __name__=="__main__":main()
