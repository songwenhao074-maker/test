"""Protocol-044 read-only equivalence audit of sealed Protocol-043 D arms."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import protocol044_common as c

ARMS=("D_no_gc","D_bounded")
FIELDS=("score","score_pos","score_neg","removal_fpr_delta","removal_recall_delta","positive_host_rows","negative_host_rows","positive_intervals","first_i","last_i")
def J(p): return json.loads(Path(p).read_text(encoding="utf8"))
def W(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
def L(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def same(a,b):
    for k in FIELDS:
        x=a.get(k);y=b.get(k)
        if x is None or y is None:
            if x is not None or y is not None:return False
        elif isinstance(x,(float,np.floating)) or isinstance(y,(float,np.floating)):
            if abs(float(x)-float(y))>1e-10:return False
        elif int(x)!=int(y):return False
    for k in ("valid","reason","eligible_now","streak_before","streak_after","eligible_three"):
        if a.get(k)!=b.get(k):return False
    return True

def audit_arm(root,arm):
    d=Path(root)/"science"/arm; pred=L(d/"predictions.npz")
    checks=J(d/"utility_checks.json")["checks"]; tables=J(d/"sleep_tables.json")["tables"]
    by_t={}
    for r in checks:by_t.setdefault(int(r["at_interval"]),[]).append(r)
    tab={int(r["at_interval"]):r for r in tables}
    epochs=np.asarray(pred["deployment_epoch"],np.int64); n=len(epochs); mism=[]; action_mism=[]; rebuilt=0
    produced=0
    for ep in sorted(set(int(x) for x in epochs.tolist())):
        idx=np.flatnonzero(epochs==ep)
        if not len(idx):continue
        start,end=int(idx[0]),int(idx[-1])
        first_active=[int(x) for x in pred["active_ids"][start] if int(x)>=0]
        tr=c.UtilityTracker();tr.reset(ep,start,first_active,"p043_readonly_audit")
        for i in range(start,end+1):
            live=np.asarray(pred["live_margin"][i],np.float32);y=np.asarray(pred["labels"][i])
            ds={}
            for eid in range(pred["contribution"].shape[0]):
                z=np.asarray(pred["contribution"][eid,i],np.float32)
                if np.isfinite(z).all():ds[eid]=z
            tr.settle(i,ep,live,ds,y)
            t=i+2
            if t in by_t:
                rr=[]
                for old in sorted(by_t[t],key=lambda q:int(q["expert_id"])):
                    got=tr.check(int(old["expert_id"]),i);produced+=1
                    row={"at_interval":t,"epoch":ep,"expert_id":int(old["expert_id"]),**got};rr.append(row)
                    if not same(row,old):mism.append({"published":old,"replayed":row})
                if t in tab:
                    cand=[r for r in rr if r.get("eligible_three")]
                    selected=None if not cand else int(sorted(cand,key=lambda r:(float(r["score"]),int(r["expert_id"])))[0]["expert_id"])
                    if selected!=tab[t].get("selected"):action_mism.append({"at_interval":t,"published":tab[t].get("selected"),"replayed":selected})
        rebuilt+=tr.periodic_rebuild_count+tr.near_threshold_rebuild_count+tr.rank_rebuild_count
    return {"arm":arm,"expected_checks":592,"checks_replayed":produced,"check_mismatches":mism,"action_mismatches":action_mism,
      "rebuild_count":rebuilt,"pass":bool(produced==592 and not mism and not action_mism)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--raw043",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    rows=[audit_arm(a.raw043,x) for x in ARMS]
    out={"protocol":"044","revision":1,"kind":"fixed043_incremental_score_equivalence","model_forwards":0,"gradient_steps":0,
      "arms":rows,"semantic_change":not all(r["pass"] for r in rows),"all_pass":all(r["pass"] for r in rows)}
    W(a.out,out);print(json.dumps(out,indent=2))
    if not out["all_pass"]:raise SystemExit(3)
if __name__=="__main__":main()
