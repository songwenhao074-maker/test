from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch

WINDOWS=(("U_rec1",3300,3428),("V_rec1",3808,3936),("U_rec2",4316,4444),("V_rec2",4824,4952),("U_rec3",5332,5460),("V_rec3",5840,5968))
LATE_NAMES={"U_rec2","V_rec2","U_rec3","V_rec3"}
W_BLOCKS=("W_long","W_gap1","W_gap2","W_gap3","W_gap4","W_gap5")


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def sha256_state_dict(state):
    h=hashlib.sha256()
    for key in sorted(state):
        h.update(str(key).encode('utf8')); v=state[key]
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        else:
            h.update(repr(v).encode('utf8'))
    return h.hexdigest()


def dump_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def grouped_average_precision(labels, scores):
    y=(np.asarray(labels).reshape(-1)>0).astype(np.int64)
    s=np.asarray(scores,dtype=np.float64).reshape(-1)
    if y.size==0: return None
    pos=int(y.sum()); neg=int(y.size-pos)
    if pos==0 or neg==0: return None
    order=np.argsort(-s,kind='mergesort'); y=y[order]; s=s[order]
    tp=0; fp=0; ap=0.0; prev_recall=0.0; i=0
    while i<y.size:
        j=i+1
        while j<y.size and s[j]==s[i]: j+=1
        tp += int(y[i:j].sum()); fp += int((j-i)-y[i:j].sum())
        recall=tp/pos; precision=tp/(tp+fp)
        ap += precision*(recall-prev_recall); prev_recall=recall; i=j
    return float(ap)


def binary_metrics(probability, labels, threshold=0.5):
    p=np.asarray(probability,dtype=np.float64).reshape(-1)
    y=(np.asarray(labels).reshape(-1)>0)
    if p.size != y.size: raise ValueError('probability/label shape mismatch')
    pred=p>=float(threshold); pos=int(y.sum()); neg=int((~y).sum())
    tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=pos-tp; tn=neg-fp
    eps=1e-12; clipped=np.clip(p,eps,1-eps)
    bce=float(-(y*np.log(clipped)+(~y)*np.log(1-clipped)).mean()) if y.size else None
    return {'ap':grouped_average_precision(y,p),'bce':bce,'rows':int(y.size),'positives':pos,'negatives':neg,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'recall':(tp/pos if pos else None),'fpr':(fp/neg if neg else None)}


def phase_bounds(manifest_path):
    m=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    rows=m.get('timeline') or m.get('phases') or m.get('phase_table')
    return {str(x.get('name',x.get('phase'))):(int(x.get('start',x.get('start_interval'))),int(x.get('end_exclusive',x.get('end',x.get('end_interval'))))) for x in rows}


def load_npz(path):
    with np.load(path,allow_pickle=False) as d: return {k:d[k].copy() for k in d.files}


def metrics_for_ranges(prob,labels,ranges):
    out=[]
    for name,a,b in ranges:
        out.append({'name':name,'intervals':[int(a),int(b)],**binary_metrics(prob[a:b],labels[a:b])})
    return out


def pooled_ranges(prob,labels,ranges):
    pp=np.concatenate([np.asarray(prob[a:b]) for _,a,b in ranges],axis=0)
    yy=np.concatenate([np.asarray(labels[a:b]) for _,a,b in ranges],axis=0)
    return binary_metrics(pp,yy)
