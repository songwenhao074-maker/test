"""Frozen inference sensitivity only; no training, new streams, or protocol edits."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='3'
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
import psutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from run_ftmoe_online import Replay, OnlineSession, resolve_checkpoint, resources
from analyze_ftmoe_online import summarize_arrays
torch.set_num_threads(3)
torch.set_num_interop_threads(1)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
resources()
out=Path(__file__).resolve().parent
arrays=dict(np.load(ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/streams/seed301_steps300/stream.npz'))
stored=dict(np.load(ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/runs/A_model1_replay301_lr0/predictions.npz'))
checkpoint,source=resolve_checkpoint(1)
training=np.load(ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical/time_series.npy')[:5].reshape(5,202,16,7)
lo=training.min(axis=(0,1));hi=training.max(axis=(0,1))
global_max=hi.max(axis=0)
resource_mask=np.zeros((16,7),bool)
resource_mask[:,[0,1,4]]=True
zero=(hi<=1e-8)&resource_mask
floor=zero.copy()
# OfflineTraceWorkloadV2 uses EPS_IPS=2 for an otherwise zero-demand task.
floor[:,0] |= hi[:,0]<=2.
report={'scope':'Post-hoc frozen inference diagnostics on adapted protocol016 pilot301 only; not online comparison results or selected normalization.',
        'source_checkpoint':source,'cases':{}}
for name,mask in [('original',np.zeros_like(zero)),('zero_resource_fallback',zero),('zero_and_cpu_floor_fallback',floor)]:
    normalization=copy.deepcopy(checkpoint['normalization'])
    scale=np.asarray(normalization['time_scale']).reshape(16,7).copy()
    scale[mask]=np.broadcast_to(global_max,scale.shape)[mask]
    normalization['time_scale']=scale.reshape(-1).tolist()
    replay=Replay(arrays,normalization,300)
    # Capacity remains the approved native value; only the explicitly listed input scales differ.
    session=OnlineSession(checkpoint,'A',1,replay,0.,301)
    p=[];classes=[]
    for start in range(0,300,32):
        resources()
        windows=[replay.window(t) for t in range(start,min(start+32,300))]
        x,s,g=[torch.stack([w[j] for w in windows]) for j in range(3)]
        with torch.inference_mode():
            result=session.model(x,s,g)
        p.append(result['detection_logits'].softmax(-1)[...,1].numpy())
        classes.append(result['class_logits'].softmax(-1).numpy())
    p=np.concatenate(p);classes=np.concatenate(classes)
    assert session.model.state_hash()==session.initial_state_hash
    scores=summarize_arrays(p,classes,stored['labels'],stored['raw_labels'])
    entry={'modified_host_feature_indices':np.argwhere(mask).tolist(),
           'replacement_values':scale[mask].tolist(),
           'full':scores['full'],'second_half':scores['second_half'],
           'max_probability_difference_vs_original_saved':float(np.abs(p-stored['probability']).max())}
    if name=='original':
        np.testing.assert_allclose(p,stored['probability'],atol=2e-6,rtol=1e-5)
    report['cases'][name]=entry
    print(name,json.dumps(entry))
(out/'normalization_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
