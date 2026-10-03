"""Materialize only Protocol039 locked C/B/D_keep inputs from verified artifact extractions.
F files may exist in the 037 archive but are never opened or copied.
"""
from pathlib import Path
import argparse, shutil, json

P036 = [
 "stage_B/seed3601/C_ref/feature_tape.npz",
 "stage_B/seed3601/C_ref/update_batches.json",
 "stage_B/seed3601/D_lin/predictions.npz",
 "inputs/seed3601/manifest.json",
 "stage_B/seed3601/C_ref/predictions.npz",
 "stage_B/seed3601/C_ref/summary.json",
 "stage_B/seed3601/D_lin/summary.json",
]
P037 = [
 "D_birth/predictions.npz",
 "D_birth/birth_checks.json",
 "D_birth/lifecycle_events.json",
 "D_birth/update_log.json",
 "D_birth/summary.json",
]

def locate(root,suffix):
    root=Path(root); suffix=Path(suffix).as_posix()
    hits=[p for p in root.rglob(Path(suffix).name) if p.is_file() and p.as_posix().endswith("/"+suffix)]
    if len(hits)!=1:
        raise RuntimeError("expected exactly one %s under %s, got %d: %s"%(suffix,root,len(hits),hits[:10]))
    return hits[0]

def copy_set(src,dst,rels):
    src,dst=Path(src),Path(dst); rows=[]
    for rel in rels:
        p=locate(src,rel); q=dst/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        rows.append({"relative":rel,"source":str(p),"dest":str(q),"size":q.stat().st_size})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--extracted036",required=True); ap.add_argument("--extracted037",required=True); ap.add_argument("--out036",required=True); ap.add_argument("--out037",required=True); ap.add_argument("--report",required=True); a=ap.parse_args()
    r36=copy_set(a.extracted036,a.out036,P036); r37=copy_set(a.extracted037,a.out037,P037)
    Path(a.report).write_text(json.dumps({"protocol":"039","source036":r36,"source037":r37,"F_files_opened_or_copied":False},indent=2)+"\n",encoding="utf8")
if __name__=="__main__": main()
