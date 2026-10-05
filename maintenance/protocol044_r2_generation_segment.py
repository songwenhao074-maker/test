"""Protocol044 r2 one-job generation worker: <=200 new rows, soft-deadline safe checkpoints."""
from __future__ import annotations
import argparse,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def J(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source036",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--max-new-rows",type=int,default=200);ap.add_argument("--subsegment-rows",type=int,default=50)
    ap.add_argument("--soft-deadline-seconds",type=int,default=3600);ap.add_argument("--status",required=True);a=ap.parse_args()
    if not (1<=a.max_new_rows<=200):raise ValueError("max new rows")
    if not (1<=a.subsegment_rows<=a.max_new_rows):raise ValueError("subsegment rows")
    out=pathlib.Path(a.output);m=J(out/"resume_manifest.json");start=int(m["next_t"]);target=5953
    if start<0 or start>target:raise RuntimeError("invalid start cursor")
    t0=time.monotonic();generated=0;calls=[]
    while generated<a.max_new_rows and start+generated<target:
        elapsed=time.monotonic()-t0
        # Do not begin another bounded collector call after the soft deadline.
        if elapsed>=a.soft_deadline_seconds:break
        n=min(a.subsegment_rows,a.max_new_rows-generated,target-(start+generated))
        cmd=[sys.executable,str(ROOT/"protocol044_stream.py"),"segment","--source036",str(a.source036),"--output",str(out),"--max-intervals",str(n)]
        st=time.monotonic();q=subprocess.run(cmd,cwd=str(ROOT),check=False,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if q.returncode!=0:
            W(a.status,{"protocol":"044","execution_revision":2,"start_t":start,"generated_this_job":generated,
              "next_t":int(J(out/"resume_manifest.json")["next_t"]) if (out/"resume_manifest.json").exists() else None,
              "collector_rc":q.returncode,"collector_tail":q.stdout[-12000:],"paused":False,"failed":True})
            raise RuntimeError("collector failure rc=%d"%q.returncode)
        nm=J(out/"resume_manifest.json");nxt=int(nm["next_t"]);delta=nxt-(start+generated)
        if delta<=0 or delta>n:raise RuntimeError("collector cursor delta invalid")
        generated+=delta;calls.append({"requested":n,"generated":delta,"next_t":nxt,"wall_seconds":time.monotonic()-st})
        if nxt==target:break
    next_t=int(J(out/"resume_manifest.json")["next_t"])
    if next_t-start!=generated or generated>a.max_new_rows:raise RuntimeError("job generation accounting")
    status={"protocol":"044","execution_revision":2,"science_config_revision":1,"start_t":start,"next_t":next_t,
      "generated_this_job":generated,"max_new_rows":a.max_new_rows,"soft_deadline_seconds":a.soft_deadline_seconds,
      "wall_seconds":time.monotonic()-t0,"calls":calls,"complete":next_t==target,"paused":next_t<target,"failed":False,
      "model_runs_started":0}
    W(a.status,status);print(json.dumps(status,indent=2))
if __name__=="__main__":main()
