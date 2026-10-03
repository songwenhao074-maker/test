"""Materialize exact Protocol-036 and Protocol-039 inputs for Protocol-040 by locked SHA256."""
from pathlib import Path
import argparse, hashlib, json, shutil

P36 = {
 "stage_B/seed3601/C_ref/feature_tape.npz":"c71a2ea6d59045fe4cc8a953a776d37f48dbe1feec4aba859cd00f11c984529c",
 "stage_B/seed3601/C_ref/update_batches.json":"4d32d2ece194cb41b8bc16de1c89ddfedcd42da668b96446937fb430d6246b6f",
 "stage_B/seed3601/D_lin/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "inputs/seed3601/manifest.json":"73a1c6fd8d8d12c8e84958df55b2379ab2f2542ac34923bc3934d363329cb02b",
 "stage_B/seed3601/C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
}
P39 = {
 "B_ref/predictions.npz":"1ea811ed175817294f275505db85d0bf283c3d0aca6b883b036243ad7843bef6",
 "C_ref/predictions.npz":"da02b63c4a6cf7a2e4fe7c5b98dcd872ca1fa08114bb029f46fce4053b9e9576",
 "D_keep/predictions.npz":"badf9005b2be9fc6436d1ff2b9377b27715e846d7d16f8e4bdfcbfa76f732e8c",
 "D_sleepwake/lifecycle_events.json":"3eb6d21757fd84b195d62d298350f0772befca471793dddbb7f0062933b138f8",
 "D_sleepwake/predictions.npz":"bc387d13721ed2e759a205ed3b4f6fa884c1ca105068a79a340148e1f5e96512",
 "D_sleepwake/update_log.json":"f771642166041282e1748a3e5b36c991f0365476b8fb953e7ceaee7dae2162d6",
}

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def index(root):
    out={}
    for p in Path(root).rglob("*"):
        if p.is_file():
            d=sha(p)
            out.setdefault(d,[]).append(p)
    return out

def materialize(src_root, dest_root, expected, label):
    idx=index(src_root); dest_root=Path(dest_root); rows=[]
    for rel,digest in expected.items():
        hits=idx.get(digest,[])
        if len(hits)!=1:
            raise RuntimeError(f"{label} digest {digest} for {rel}: expected one file, found {len(hits)}")
        q=dest_root/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(hits[0],q)
        if sha(q)!=digest: raise RuntimeError("copy digest mismatch "+rel)
        rows.append({"path":rel,"sha256":digest,"source_path":str(hits[0])})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extracted036",required=True); ap.add_argument("--extracted039",required=True)
    ap.add_argument("--out036",required=True); ap.add_argument("--out039",required=True); ap.add_argument("--report",required=True)
    a=ap.parse_args()
    r36=materialize(a.extracted036,a.out036,P36,"Protocol036")
    r39=materialize(a.extracted039,a.out039,P39,"Protocol039")
    obj={"protocol":"040","source036":r36,"source039":r39,"all_locked_files_verified":True,"F_files_materialized":False}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(obj,indent=2),encoding="utf8")
    print(json.dumps(obj,indent=2))
if __name__=="__main__": main()
