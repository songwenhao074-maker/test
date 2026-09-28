"""Protocol-034 fixed causal cache router for full or fragment snapshot memories."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

CLIP=1e-6
WINDOW=32
IMPROVEMENT=0.01

def J(p): return json.loads(Path(p).read_text(encoding='utf8'))

def ce(prob, label):
    p=np.clip(np.asarray(prob,dtype=np.float64),CLIP,1.0-CLIP)
    y=(np.asarray(label)>0).astype(np.float64)
    return -(y*np.log(p)+(1.0-y)*np.log(1.0-p))

def load_cache(root,meta,mode):
    out=[]; prefix='full' if mode=='full' else 'fragment'
    for item in meta:
        sid=item['snapshot_id']; path=Path(root)/'snapshot_cache'/(sid+'.npz')
        with np.load(path) as d:
            out.append({'meta':item,'probability':d[prefix+'_probability'].copy(),
                        'class_probability':d[prefix+'_class_probability'].copy()})
    return out

def route(mode,live_dir,cache_root,out_dir):
    if mode not in ('full','fragment'): raise ValueError(mode)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=False)
    meta=J(Path(cache_root)/'snapshot_metadata.json'); caches=load_cache(cache_root,meta,mode)
    with np.load(Path(live_dir)/'predictions.npz') as d:
        live_p=d['probability'].copy(); live_c=d['class_probability'].copy(); labels=d['labels'].copy()
    steps,hosts=live_p.shape; routed_p=live_p.copy(); routed_c=live_c.copy()
    selected=np.full((steps,hosts),-1,np.int16); lineage=np.full((steps,hosts),-1,np.int16)
    fallback_loss=np.full((steps,hosts),np.nan,np.float32); candidate_loss=np.full((steps,hosts),np.nan,np.float32)
    score_max_index=np.full(steps,-1,np.int32); max_label_cursor=np.full(steps,-1,np.int32)
    candidate_count=np.zeros(steps,np.int16); selected_count=np.zeros(steps,np.int32)
    sid_to_seq={m['snapshot_id']:i+1 for i,m in enumerate(meta)}
    lineage_ids=sorted({int(m['lineage_id']) for m in meta})
    for t in range(steps):
        last=t-3; first=last-(WINDOW-1)
        if first<0: continue
        support=np.arange(first,last+1,dtype=np.int64)
        if support.size!=WINDOW or int(support[-1])+2>=t: raise AssertionError('Protocol034 causal support cutoff violated')
        score_max_index[t]=int(support[-1]); max_label_cursor[t]=int(support[-1])+2
        live_losses=ce(live_p[support],labels[support]).mean(axis=0); fallback_loss[t]=live_losses.astype(np.float32)
        eligible=[]
        for cache in caches:
            cp=cache['probability'][support]
            if not np.isfinite(cp).all(): continue
            losses=ce(cp,labels[support]).mean(axis=0); eligible.append((cache,losses))
        candidate_count[t]=len(eligible)
        if not eligible: continue
        per_lineage={}
        for lid in lineage_ids:
            rows=[x for x in eligible if int(x[0]['meta']['lineage_id'])==lid]
            if not rows: continue
            best_loss=np.full(hosts,np.inf,np.float64); best_cache=[None]*hosts
            for cache,loss in rows:
                seq=sid_to_seq[cache['meta']['snapshot_id']]
                for h in range(hosts):
                    if loss[h]<best_loss[h]-1e-15:
                        best_loss[h]=loss[h]; best_cache[h]=cache
                    elif abs(float(loss[h]-best_loss[h]))<=1e-15 and best_cache[h] is not None:
                        if seq<sid_to_seq[best_cache[h]['meta']['snapshot_id']]: best_cache[h]=cache
            per_lineage[lid]=(best_loss,best_cache)
        for h in range(hosts):
            best=None
            for lid in sorted(per_lineage):
                loss,arr=per_lineage[lid]; cache=arr[h]
                if cache is None: continue
                seq=sid_to_seq[cache['meta']['snapshot_id']]; tup=(float(loss[h]),int(lid),int(seq),cache)
                if best is None or tup[:3]<best[:3]: best=tup
            if best is None: continue
            loss,lid,seq,cache=best; candidate_loss[t,h]=loss
            if loss<=((1.0-IMPROVEMENT)*float(live_losses[h])):
                routed_p[t,h]=cache['probability'][t,h]; routed_c[t,h]=cache['class_probability'][t,h]
                selected[t,h]=seq; lineage[t,h]=lid; selected_count[t]+=1
    if np.any((score_max_index>=0)&(max_label_cursor>=np.arange(steps))):
        raise AssertionError('Protocol034 router used a label available at/after current prediction')
    np.savez_compressed(out/'predictions.npz',probability=routed_p,class_probability=routed_c,labels=labels,
                        selected_snapshot_seq=selected,selected_lineage_id=lineage,fallback_loss=fallback_loss,
                        candidate_loss=candidate_loss,score_max_prediction_index=score_max_index,
                        max_label_available_cursor=max_label_cursor,eligible_snapshot_count=candidate_count,
                        selected_host_count=selected_count)
    usage={}
    for seq in range(1,len(meta)+1): usage[meta[seq-1]['snapshot_id']]=int((selected==seq).sum())
    report={'protocol':'034','mode':mode,'trained_parameters':False,'gradient_steps':0,'current_labels_allowed':False,
            'score_window_matured_intervals':WINDOW,'score_time_rule':'i + 2 < t','minimum_relative_loss_improvement':IMPROVEMENT,
            'snapshot_ids':[m['snapshot_id'] for m in meta],'lineage_ids':[m['lineage_id'] for m in meta],
            'selected_host_predictions':int((selected>=0).sum()),'total_host_predictions':int(selected.size),
            'fallback_host_predictions':int((selected<0).sum()),'snapshot_usage':usage,
            'max_logged_score_prediction_index':int(score_max_index.max()),'causal_cutoff_verified':True,'learning_feedback':False}
    (out/'routing_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2)); return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['full','fragment'],required=True)
    ap.add_argument('--live-dir',required=True); ap.add_argument('--cache-root',required=True); ap.add_argument('--out-dir',required=True)
    a=ap.parse_args(); route(a.mode,a.live_dir,a.cache_root,a.out_dir)
if __name__=='__main__': main()
