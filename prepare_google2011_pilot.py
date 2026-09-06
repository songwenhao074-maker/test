"""Download a bounded official Google 2011 prefix and construct a provenance-preserving pilot."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import sys
import time
import urllib.request

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import certifi
import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/ftmoe_online/google2011_018'
BUCKET='https://storage.googleapis.com/clusterdata-2011-2/'
CTX=ssl.create_default_context(cafile=certifi.where())
CALIBRATION=64
STEPS=300
BINS=CALIBRATION+STEPS+2
ORIGIN=600_000_000
WIDTH=300_000_000


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def guard():
    available=psutil.virtual_memory().available/2**30
    if available<4.5:raise RuntimeError('RAM guard below 4.5 GiB (available %.3f GiB)'%available)
    if shutil.disk_usage(ROOT).free/2**30<20:raise RuntimeError('Disk guard below 20 GiB')


def download(meta):
    target=OUT/'source'/Path(meta['name']).name
    size=int(meta['size'])
    if target.exists():
        if target.stat().st_size!=size:raise ValueError('Completed source size mismatch')
        return target
    partial=target.with_suffix(target.suffix+'.partial')
    guard()
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(BUCKET+meta['name'],headers={'Range':f'bytes={offset}-'} if offset else {})
    if offset<size:
        with urllib.request.urlopen(request,context=CTX,timeout=40) as response:
            if offset and response.status!=206:raise RuntimeError('Server ignored partial download Range')
            with partial.open('ab' if offset else 'wb') as stream:
                while True:
                    guard()
                    block=response.read(1024*1024)
                    if not block:break
                    stream.write(block)
    if partial.stat().st_size!=size:raise ValueError('Downloaded source size mismatch')
    digest=hashlib.md5()
    with partial.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    if base64.b64encode(digest.digest()).decode()!=meta['md5Hash']:raise ValueError('Official source MD5 mismatch')
    partial.replace(target)
    return target


def chunks(path):
    # Official zero-based fields: start/end, job/task IDs, CPU, canonical memory, local disk space.
    return pd.read_csv(path,header=None,usecols=[0,1,2,3,5,6,12],chunksize=50000,
                       dtype={0:'int64',1:'int64',2:'int64',3:'int64',5:'float64',6:'float64',12:'float64'})


def add_measurement(sums, weights, row, slot):
    start,end=int(row[0]),int(row[1]);values=np.asarray(row[4:7],dtype=float)
    if end<=start or not np.isfinite(values).all() or (values<0).any():return
    first=max(0,(start-ORIGIN)//WIDTH);last=min(BINS-1,(end-1-ORIGIN)//WIDTH)
    for index in range(first,last+1):
        length=max(0,min(end,ORIGIN+(index+1)*WIDTH)-max(start,ORIGIN+index*WIDTH))
        sums[slot,index]+=values*length;weights[slot,index]+=length


def fill_single_gaps(values, coverage):
    """Carry previous observation across isolated interior gaps; never use future values."""
    result=values.copy();valid=coverage.copy()
    for t in range(1,len(valid)-1):
        if not coverage[t] and coverage[t-1] and coverage[t+1]:
            result[t]=values[t-1];valid[t]=True
    return result,valid


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    protocol={'protocol':18,'source':'Google clusterdata-2011-2 official task_usage',
              'calibration_bins':CALIBRATION,'scoring_bins':STEPS,'bin_seconds':300,'maximum_source_parts':30,
              'candidate_selection':'up to 1024 task IDs observed with positive CPU/RAM/disk in first full bin, ordered by SHA256(job:task)',
              'task_selection':'first 16 hash-ordered candidates with >=98% full-bin coverage, only isolated interior gaps forward-filled, all bins valid after filling; future coverage used only for cohort eligibility, never future resource values for imputation',
              'mapping':'per-resource linear scale: offline training demand p95 / Google calibration p95; no per-row clipping; four I/O channels fixed 1 for legacy input semantics',
              'simulation':'16 long-running source tasks, global source time, no trace cycling, unchanged GOBI and no-op recovery, CPU capacity .8 and disk capacity .25, native RAM',
              'scope':'single model1/replay301/300-step A B C D pilot at lr3e-5; unchanged checkpoint normalization; no formal selection using pilot model scores',
              'limitation':'selected long-lived tasks with nonzero storage, normalized units mapped to simulator units; simulated overload is not original fault annotation'}
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==protocol
    else:write(p,protocol)
    guard()
    metadata_path=OUT/'source_metadata.json'
    if metadata_path.exists():metadata=json.loads(metadata_path.read_text())
    else:
        url='https://storage.googleapis.com/storage/v1/b/clusterdata-2011-2/o?prefix=task_usage%2F&maxResults=500'
        metadata=json.load(urllib.request.urlopen(url,context=CTX,timeout=40))
        write(metadata_path,metadata)
    schema=OUT/'source/schema.csv'
    if not schema.exists():schema.write_bytes(urllib.request.urlopen(BUCKET+'schema.csv',context=CTX,timeout=40).read())
    first=download(metadata['items'][0])
    candidates=set()
    for chunk in chunks(first):
        guard()
        good=chunk[(chunk[0]<=ORIGIN)&(chunk[1]>=ORIGIN+WIDTH)&(chunk[[5,6,12]]>0).all(axis=1)]
        candidates.update((int(j),int(t)) for j,t in good[[2,3]].itertuples(index=False,name=None))
    ids=sorted(candidates,key=lambda x:hashlib.sha256(f'{x[0]}:{x[1]}'.encode()).hexdigest())[:1024]
    if len(ids)<16:raise ValueError('Insufficient positive CPU/RAM/disk task candidates in initial source bin')
    index={key:i for i,key in enumerate(ids)}
    sums=np.zeros((len(ids),BINS,3));weights=np.zeros((len(ids),BINS))
    jobs=set(j for j,t in ids);sources=[];last_start=-1
    for meta in metadata['items'][:30]:
        file=download(meta)
        for chunk in chunks(file):
            guard();last_start=max(last_start,int(chunk[0].max()))
            chosen=chunk[chunk[2].isin(jobs)]
            for row in chosen.itertuples(index=False,name=None):
                slot=index.get((int(row[2]),int(row[3])))
                if slot is not None:add_measurement(sums,weights,row,slot)
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        sources.append({'name':meta['name'],'bytes':file.stat().st_size,'md5_base64':meta['md5Hash'],'sha256':digest.hexdigest(),'generation':meta['generation']})
        print(json.dumps({'source_parts':len(sources),'max_start_hours':(last_start-ORIGIN)/3.6e9,'source_bytes':sum(s['bytes'] for s in sources)}),flush=True)
        if last_start>=ORIGIN+BINS*WIDTH:break
    if last_start<ORIGIN+BINS*WIDTH:raise ValueError('Source prefix does not cover registered continuous horizon')
    # Less than 90% observed time is missing. More than a full bin indicates overlapping duplicate measurements.
    values=np.divide(sums,weights[...,None],out=np.zeros_like(sums),where=weights[...,None]>0)
    eligible=[]
    for i in range(len(ids)):
        coverage=(weights[i]>=WIDTH*.9)&(weights[i]<=WIDTH*1.01)
        if coverage.mean()<.98:continue
        fixed,valid=fill_single_gaps(values[i],coverage)
        if valid.all():eligible.append((i,fixed,int((~coverage).sum())))
    write(OUT/'coverage.json',{'candidates':len(ids),'eligible':len(eligible),'source_parts':sources})
    if len(eligible)<16:raise ValueError('Fewer than 16 sufficiently continuous tasks; do not cycle or silently fill long gaps')
    chosen=eligible[:16];observed=np.stack([v for _,v,_ in chosen],axis=1)
    demand=np.load(ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical/container_demand_series.npy')[:5].reshape(-1,7)
    target=np.quantile(demand[:,[0,1,4]],.95,axis=0)
    calibration=np.quantile(observed[:CALIBRATION],.95,axis=(0,1))
    if (calibration<=0).any():raise ValueError('Invalid calibration scale')
    factor=target/calibration;series=observed[CALIBRATION:]*factor
    np.savez_compressed(OUT/'trace.npz',resource_series=series.astype(np.float64),source_resource_series=observed,
                        source_ids=np.array([ids[i] for i,_,_ in chosen],dtype=np.int64),scale=factor)
    manifest={'source':BUCKET,'source_parts':sources,'schema_sha256':hashlib.sha256(schema.read_bytes()).hexdigest(),
              'source_ids':[list(ids[i]) for i,_,_ in chosen],'forward_filled_bins':[g for _,_,g in chosen],
              'calibration_p95':calibration.tolist(),'offline_training_p95':target.tolist(),'linear_scale':factor.tolist(),
              'source_start_microseconds':ORIGIN+CALIBRATION*WIDTH,'series_shape':list(series.shape),
              'trace_sha256':hashlib.sha256((OUT/'trace.npz').read_bytes()).hexdigest(),
              'source_is_fault_labeled':False,'synthetic_disk':False,'trace_repeated':False}
    write(OUT/'trace.json',manifest)
    write(OUT/'scenario.json',{'name':'google2011_018','workload':'google2011','ram_capacity_multiplier':1.,'disk_capacity_multiplier':1.,
                              'trace_path':str((OUT/'trace.npz').relative_to(ROOT)),'demand_adapter':manifest['linear_scale']})
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'preparation_failure.json',{'error':type(exc).__name__+': '+str(exc)})
        raise
