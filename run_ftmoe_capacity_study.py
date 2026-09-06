"""Protocol017: select capacity from data only, then complete a 2,000-step A/B/C/D replay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/ftmoe_online/capacity_017'


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def inspect_stream(folder, target):
    with np.load(folder/'stream.npz') as data:
        raw=data['raw_labels'];steps=len(raw)-1;y=raw[:-1].copy()
        for t in range(steps):
            for d in (-1,1):
                if 0<=t+d<len(raw):
                    use=(y[t]==0)&(raw[t+d]>0);y[t,use]=raw[t+d,use]
        counts=np.bincount(y.ravel(),minlength=4);p=counts/counts.sum()
        rawcounts=np.bincount(raw[:-1].ravel(),minlength=4)
        # Compare normal and each resource, rather than only total anomaly prevalence.
        similar=bool(abs(p[0]-target[0])<=.08 and np.all(p[1:]>=target[1:]*.5) and np.all(p[1:]<=target[1:]*2))
        score=float(np.abs(np.log((p[1:]+1e-6)/(target[1:]+1e-6))).mean()+abs(p[0]-target[0]))
        reconstructed=np.where((data['post_totals']/data['capacities']>1).any(-1),(data['post_totals']/data['capacities']).argmax(-1)+1,0)
        assert np.array_equal(reconstructed,raw)
        result={'stream':str(folder),'steps':steps,'main_class_counts':counts.tolist(),'main_class_fraction':p.tolist(),
                'raw_class_counts':rawcounts.tolist(),'distribution_score':score,'roughly_similar':similar,
                'mean_active_containers':float((data['after_placement'][:-1]>=0).sum(-1).mean()),
                'raw_label_mismatches':0}
    write(folder/'distribution.json',result)
    return result


def command(args, log):
    with log.open('w',encoding='utf8') as stream:
        completed=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError('Command failed; see '+str(log))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical/labels.npy'
    y=np.load(data)[:5];counts=np.bincount(y.ravel(),minlength=4);target=counts/counts.sum()
    candidates=[{'name':f'ram{r:g}_disk{d:g}','workload':'adapted_bwgd2','ram_capacity_multiplier':r,'disk_capacity_multiplier':d}
                for r in (1.,.75,.5) for d in (1.,.95,.9)]
    protocol={'protocol':17,'target_train_counts':counts.tolist(),'target_train_fraction':target.tolist(),
              'target_source_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'candidates':candidates,
              'development_seed':301,'development_steps':300,'confirmation_seed':302,'confirmation_steps':2000,
              'model_seed':1,'learning_rate':3e-5,'methods':list('ABCD'),
              'selection':'minimum mean absolute log class-ratio error plus normal fraction error; ties prefer less capacity reduction; no model scores',
              'rough_similarity':'normal absolute difference <= .08; each positive class fraction between .5x and 2x training fraction',
              'normalization':'unchanged protocol014 checkpoint; known host14 normalization limitation preserved for attribution',
              'diagnostic_boundary':'Run all 2000 steps and four arms even if approximate distribution matching fails, explicitly report failure; no automatic LR search or five-seed claims'}
    registration=OUT/'protocol.json'
    if registration.exists():assert json.loads(registration.read_text(encoding='utf8'))==protocol
    else:write(registration,protocol)
    results=[]
    for config in candidates:
        config_path=OUT/(config['name']+'.json')
        if not config_path.exists():write(config_path,config)
        folder=OUT/'development'/config['name']
        if config['name']=='ram1_disk1':
            folder=ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/streams/seed301_steps300'
            # Original outputs remain immutable; inspect in memory through a copied tiny npz and manifest.
            copied=OUT/'development'/config['name'];copied.mkdir(parents=True,exist_ok=True)
            import shutil
            for name in ('stream.npz','manifest.json'):
                if not (copied/name).exists():shutil.copy2(folder/name,copied/name)
            folder=copied
        if not (folder/'manifest.json').exists():
            command(['prepare_ftmoe_scenario.py','--seed',301,'--steps',300,'--configuration',config_path,'--output',folder],OUT/(config['name']+'.log'))
        result=inspect_stream(folder,target);result['configuration']=config;results.append(result)
        write(OUT/'development_results.json',results)
        print(json.dumps({'candidate':config['name'],'fractions':result['main_class_fraction'],'similar':result['roughly_similar']},ensure_ascii=False),flush=True)
    best=min(results,key=lambda r:(r['distribution_score'],-r['configuration']['ram_capacity_multiplier'],-r['configuration']['disk_capacity_multiplier']))
    selection=OUT/'selection.json'
    if selection.exists():assert json.loads(selection.read_text(encoding='utf8'))==best
    else:write(selection,best)
    config_path=OUT/(best['configuration']['name']+'.json')
    folder=OUT/'confirmation/seed302_steps2000'
    if not (folder/'manifest.json').exists():
        command(['prepare_ftmoe_scenario.py','--seed',302,'--steps',2000,'--configuration',config_path,'--output',folder],OUT/'confirmation.log')
    dist=inspect_stream(folder,target)
    for method in 'ABCD':
        output=OUT/'runs'/method
        if (output/'summary.json').exists():continue
        args=['run_ftmoe_online.py','--method',method,'--model-seed',1,'--stream',folder,'--learning-rate',3e-5,'--output',output]
        if (output/'resume.pt').exists():args+=['--resume']
        command(args,OUT/(method+'.log'))
    summaries={m:json.loads((OUT/'runs'/m/'summary.json').read_text()) for m in 'ABCD'}
    result={'status':'complete','distribution':dist,'selected_configuration':best['configuration'],
            'second_half_f1':{m:s['metrics']['second_half']['f1'] for m,s in summaries.items()},
            'limitation':'one model seed, one confirmation replay; scenario pilot, not five-seed confirmation'}
    write(OUT/'result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'failure.json',{'error':str(exc)})
        raise
