"""Causal replay of A/B/C/D online tuning, with complete resumable state."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='3'
import numpy as np
import psutil
import torch
from recovery.PreGANSrc.src.ftmoe_online import OnlineFTMoE
from train_ftmoe_ablation_existing import loss_fn
from analyze_ftmoe_online import summarize_arrays

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts/ftmoe_online'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)


def save_state(path, state):
    temporary=Path(str(path)+'.tmp');torch.save(state,temporary);temporary.replace(path)


def resources():
    available=psutil.virtual_memory().available/2**30;disk=shutil.disk_usage(ROOT).free/2**30
    if available<4.5 or disk<20:raise RuntimeError('Resource guard: RAM %.3f GiB, disk %.3f GiB'%(available,disk))
    return psutil.Process().memory_info().rss/2**30


def resolve_checkpoint(seed):
    base=ROOT/'artifacts/ftmoe_end_to_end'
    provenance=json.loads((base/'comparison_014_complete.json').read_text(encoding='utf8'))['source_sha256']
    if seed in (1,2,6):
        selection=json.loads((base/f'diagnosis_reassessment/physical_lr0003_e30/v4_seed{seed}.json').read_text())
        path=Path(selection['best_checkpoint'])
    elif seed in (17,42):path=base/f'runs/historical_confirm_lr0003_e30/v4_seed{seed}/best.pt'
    else:raise ValueError('Unregistered model seed')
    actual=sha(path)
    if provenance.get(str(path))!=actual:raise AssertionError('Checkpoint does not match protocol014 source hash')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint['variant']!='v4' or checkpoint['seed']!=seed:raise AssertionError('Checkpoint identity mismatch')
    return checkpoint,{'path':str(path),'sha256':actual,'epoch':checkpoint['epoch']}


def tolerance_label(raw_seen, index, observed_until):
    """Only mature a label after index+1 has been physically observed."""
    if index<0 or index+1>observed_until:raise ValueError('Tolerance label has not matured')
    result=raw_seen[index].copy()
    if (raw_seen[max(0,index-1):index+2]<0).any():raise ValueError('Unobserved raw labels')
    # Nearest anomaly wins; equal-distance ties choose the earlier interval.
    if index:
        use=(result==0)&(raw_seen[index-1]>0);result[use]=raw_seen[index-1][use]
    use=(result==0)&(raw_seen[index+1]>0);result[use]=raw_seen[index+1][use]
    return result


class Replay:
    def __init__(self, arrays, normalization, steps):
        self.arrays=arrays;self.steps=steps
        self.time_scale=np.asarray(normalization['time_scale'],np.float32).reshape(16,7)
        self.graph_scale=np.asarray(normalization['graph_scale'],np.float32)
        if arrays['host_features'].shape!=(steps+1,16,7) or arrays['raw_labels'].shape!=(steps+1,16):
            raise ValueError('Stream must contain exact scored horizon plus one guard interval')
        for name in ('host_features','demands','schedules','capacities'):
            if not np.isfinite(arrays[name]).all():raise ValueError('Nonfinite stream: '+name)
        if (self.time_scale<=0).any() or (self.graph_scale<=0).any():raise ValueError('Invalid normalization')

    def window(self, index):
        if not 0<=index<self.steps:raise IndexError(index)
        positions=np.maximum(np.arange(index-11,index+1),0)
        host=(self.arrays['host_features'][positions]/self.time_scale).transpose(1,0,2).copy()
        graph=(self.arrays['demands'][positions]/self.graph_scale).transpose(1,0,2).copy()
        return (torch.from_numpy(host).float(),torch.from_numpy(self.arrays['schedules'][positions].copy()).float(),
                torch.from_numpy(graph).float())


class OnlineSession:
    def __init__(self, checkpoint, method, seed, replay, learning_rate, replay_seed):
        self.model=OnlineFTMoE(checkpoint,method,seed);self.method=method;self.replay=replay
        self.learning_rate=learning_rate;self.seed=seed;self.replay_seed=replay_seed
        self.original_capacity=checkpoint['model']['graph_encoder.host_capacity'].clone()
        new_capacity=np.asarray(replay.arrays['capacities'])/replay.graph_scale[[0,1,4]]
        self.model.graph_encoder.host_capacity.copy_(torch.as_tensor(new_capacity,dtype=torch.float32))
        self.capacity_before=self.original_capacity.tolist();self.capacity_after=new_capacity.tolist()
        self.initial_state_hash=self.model.state_hash();self.initial_frozen_hash=self.model.frozen_hash()
        self.optimizer=self.make_optimizer()
        n=replay.steps
        self.predictions={'probability':np.zeros((n,16),np.float32),'class_probability':np.zeros((n,16,3),np.float32),
            'labels':np.full((n,16),-1,np.int64),'raw_labels':np.full((n,16),-1,np.int64),
            'model_version':np.zeros(n,np.int64),'expert_count':np.zeros(n,np.int64),
            'mean_active':np.zeros(n,np.float32),'unmatched_ratio':np.zeros(n,np.float32),
            'prediction_seconds':np.zeros(n,np.float64)}
        self.raw_seen=np.full((n+1,16),-1,np.int64)
        self.buffer=deque(maxlen=64);self.cursor=0;self.update_number=0;self.updates=[];self.reference=[]
        self.finished=False

    def make_optimizer(self):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=self.learning_rate,
                                 weight_decay=1e-4) if self.method!='A' else None

    def step(self, prediction_sink=None):
        t=self.cursor
        if t>=self.replay.steps:raise StopIteration
        x,s,g=self.replay.window(t)
        started=time.perf_counter();out=self.model.predict_online(x[None],s[None],g[None])
        probability=out['detection_logits'].softmax(-1)[0,:,1].numpy()
        classes=out['class_logits'].softmax(-1)[0].numpy()
        if not np.isfinite(probability).all() or not np.isfinite(classes).all():raise RuntimeError('Nonfinite prediction')
        self.predictions['probability'][t]=probability;self.predictions['class_probability'][t]=classes
        self.predictions['prediction_seconds'][t]=time.perf_counter()-started
        self.predictions['model_version'][t]=self.update_number
        for key,value in self.model.eagate.last_routing.items():self.predictions[key][t]=value
        if prediction_sink is not None:
            prediction_sink({'step':t+1,'model_version':self.update_number,'probability':probability.tolist(),
                             'class_probability':classes.tolist(),**self.model.eagate.last_routing})
        # Future labels stored in the replay are never presented before this line.
        self.raw_seen[t]=self.replay.arrays['raw_labels'][t]
        self.predictions['raw_labels'][t]=self.raw_seen[t]
        if t:
            self.predictions['labels'][t-1]=tolerance_label(self.raw_seen,t-1,t)
            self.buffer.append(t-1)
        self.cursor=t+1
        if self.cursor%10==0 and self.optimizer is not None and self.buffer:self.update()
        if self.cursor%100==0:
            if self.model.frozen_hash()!=self.initial_frozen_hash:raise AssertionError('Frozen parameters changed')
            if self.method=='A':self.model.eagate.reset_statistics()
        return probability,classes

    def update(self):
        started=time.perf_counter();self.model.eval();total_loss=0.;batches=0
        indices=list(self.buffer)
        if max(indices)>self.cursor-2:raise AssertionError('Immature training sample')
        for offset in range(0,len(indices),32):
            batch=indices[offset:offset+32];windows=[self.replay.window(i) for i in batch]
            x,s,g=[torch.stack([sample[j] for sample in windows]) for j in range(3)]
            y=torch.from_numpy(self.predictions['labels'][batch])
            self.optimizer.zero_grad(set_to_none=True)
            out=self.model(x,s,g)
            loss=loss_fn(self.model,out,y,.7,.3,0.,.01,2.,.5)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite online loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);self.optimizer.step()
            total_loss+=float(loss.detach());batches+=1
        self.update_number+=1;event=None
        if self.update_number%10==0:
            if self.method=='D':event=self.model.adapt(self.optimizer)
            else:self.model.eagate.reset_statistics()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise RuntimeError('Nonfinite online parameters')
        self.updates.append({'step':self.cursor,'update_number':self.update_number,'loss':total_loss/batches,
            'batches':batches,'buffer_indices':indices,'seconds':time.perf_counter()-started,'topology_event':event})

    def finish(self):
        if self.cursor!=self.replay.steps:raise ValueError('Cannot finalize partial stream')
        if not self.finished:
            self.raw_seen[self.cursor]=self.replay.arrays['raw_labels'][self.cursor]
            self.predictions['labels'][-1]=tolerance_label(self.raw_seen,self.cursor-1,self.cursor)
            self.finished=True

    def save(self):
        return {'model':self.model.state_dict(),'topology':self.model.eagate.topology_state(),
            'optimizer':None if self.optimizer is None else self.optimizer.state_dict(),
            'cursor':self.cursor,'update_number':self.update_number,'buffer':list(self.buffer),
            'raw_seen':self.raw_seen,'predictions':self.predictions,'updates':self.updates,'reference':self.reference,
            'initial_state_hash':self.initial_state_hash,'initial_frozen_hash':self.initial_frozen_hash,
            'finished':self.finished,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'python_rng':random.getstate()}

    def restore(self,state):
        self.model.eagate.restore_topology(state['topology']);self.model.load_state_dict(state['model'],strict=True)
        self.model.set_trainability();self.model.eval();self.optimizer=self.make_optimizer()
        if self.optimizer is not None:self.optimizer.load_state_dict(state['optimizer'])
        for name in ('cursor','update_number','raw_seen','predictions','updates','reference','initial_state_hash','initial_frozen_hash','finished'):
            setattr(self,name,state[name])
        self.buffer=deque(state['buffer'],maxlen=64)
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if self.model.frozen_hash()!=self.initial_frozen_hash:raise AssertionError('Resume frozen hash mismatch')

    def evaluate_reference(self, blocks):
        from train_ftmoe_end_to_end import evaluate
        current=self.model.graph_encoder.host_capacity.clone()
        self.model.graph_encoder.host_capacity.copy_(self.original_capacity)
        try:
            score=evaluate(self.model,blocks)
        finally:
            self.model.graph_encoder.host_capacity.copy_(current)
        self.reference.append({'step':self.cursor,**score})


def run(args):
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);resources()
    manifest=json.loads((args.stream/'manifest.json').read_text())
    if manifest['stream_sha256']!=sha(args.stream/'stream.npz'):raise AssertionError('Stream hash mismatch')
    checkpoint,source=resolve_checkpoint(args.model_seed)
    with np.load(args.stream/'stream.npz') as data:arrays={key:data[key] for key in data.files}
    replay=Replay(arrays,checkpoint['normalization'],manifest['steps'])
    lr=0. if args.method=='A' else args.learning_rate
    if lr not in ((0.,) if args.method=='A' else (1e-5,3e-5,1e-4)):raise ValueError('Unregistered learning rate')
    files=['run_ftmoe_online.py','analyze_ftmoe_online.py','train_ftmoe_ablation_existing.py','train_ftmoe_end_to_end.py',
        'recovery/PreGANSrc/src/ftmoe_online.py','recovery/PreGANSrc/src/ftmoe_ablation.py','recovery/PreGANSrc/src/ftmoe_end_to_end.py']
    config={'schema_version':1,'method':args.method,'model_seed':args.model_seed,'replay_seed':manifest['seed'],
        'steps':manifest['steps'],'learning_rate':lr,'source_checkpoint':source,'stream':str(args.stream.resolve()),
        'stream_sha256':manifest['stream_sha256'],'stream_manifest_sha256':sha(args.stream/'manifest.json'),
        'buffer_intervals':64,'update_interval':10,'batch_size':32,'epochs_per_update':1,'weight_decay':1e-4,
        'clip_grad_norm':1.,'threshold':.5,'dropout':0.,'expert_min':2,'expert_max':8,'max_active':4,
        'adapt_every_updates':10,'label_tolerance':1,'label_delay_intervals':1,
        'code_sha256':{name:sha(ROOT/name) for name in files}}
    out=args.output
    if out.exists():
        if not args.resume:raise FileExistsError(out)
        if json.loads((out/'configuration.json').read_text())!=config:raise ValueError('Resume configuration changed')
        if (out/'summary.json').exists():raise ValueError('Run already completed')
    else:
        if args.resume:raise FileNotFoundError(out)
        out.mkdir(parents=True);write_json(out/'configuration.json',config)
    session=OnlineSession(checkpoint,args.method,args.model_seed,replay,lr,manifest['seed'])
    from train_ftmoe_end_to_end import load_data
    _,validation,normalization,_=load_data(ROOT/'artifacts/ftmoe_end_to_end/data/protocol_004_physical')
    for key in ('time_scale','graph_scale'):
        np.testing.assert_allclose(normalization[key],checkpoint['normalization'][key],rtol=1e-7,atol=1e-8)
    elapsed_before=0.
    if args.resume:
        saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        session.restore(saved['session']);elapsed_before=saved['elapsed_seconds']
    else:session.evaluate_reference(validation)
    began=time.perf_counter();end=args.stop_after or replay.steps
    if not session.cursor<end<=replay.steps:raise ValueError('Invalid stop-after cursor')
    try:
        # Rebuild only this run's durable prediction journal on resume, removing uncommitted suffixes.
        with (out/'predictions.jsonl').open('w',encoding='utf8') as journal:
            for t in range(session.cursor):
                journal.write(json.dumps({'step':t+1,'model_version':int(session.predictions['model_version'][t]),
                    'probability':session.predictions['probability'][t].tolist(),
                    'class_probability':session.predictions['class_probability'][t].tolist(),
                    **{k:float(session.predictions[k][t]) for k in ('expert_count','mean_active','unmatched_ratio')}})+'\n')
            def sink(value):journal.write(json.dumps(value)+'\n');journal.flush()
            while session.cursor<end:
                if session.cursor%10==0:resources()
                session.step(sink)
                if session.cursor%500==0:session.evaluate_reference(validation)
                if session.cursor%100==0 or session.cursor==end:
                    state={'session':session.save(),'configuration':config,'elapsed_seconds':elapsed_before+time.perf_counter()-began}
                    save_state(out/f'step{session.cursor:05d}.pt',state);save_state(out/'resume.pt',state)
                    print(json.dumps({'method':args.method,'step':session.cursor,'experts':len(session.model.eagate.ids),
                        'elapsed_seconds':state['elapsed_seconds']}),flush=True)
        if end<replay.steps:return
        session.finish()
        if session.reference[-1]['step']!=session.cursor:session.evaluate_reference(validation)
        if session.model.frozen_hash()!=session.initial_frozen_hash:raise AssertionError('Frozen parameters changed')
        summary={'configuration':config,'metrics':summarize_arrays(**{k:session.predictions[k] for k in
            ('probability','class_probability','labels','raw_labels')}),
            'initial_state_hash':session.initial_state_hash,'final_state_hash':session.model.state_hash(),
            'frozen_parameters_unchanged':True,'capacity_before':session.capacity_before,'capacity_after':session.capacity_after,
            'updates':session.update_number,'final_experts':list(session.model.eagate.ids),
            'elapsed_seconds':elapsed_before+time.perf_counter()-began,'rss_gib':resources(),
            'prediction_mean_seconds':float(session.predictions['prediction_seconds'].mean()),
            'prediction_p95_seconds':float(np.percentile(session.predictions['prediction_seconds'],95)),
            'update_total_seconds':sum(item['seconds'] for item in session.updates)}
        np.savez_compressed(out/'predictions.npz',**session.predictions)
        write_json(out/'updates.json',session.updates);write_json(out/'reference.json',session.reference)
        state={'session':session.save(),'configuration':config,'elapsed_seconds':summary['elapsed_seconds']}
        save_state(out/'last.pt',state);save_state(out/'resume.pt',state)
        write_json(out/'summary.json',summary)
        print(json.dumps({'completed':str(out),'second_half':summary['metrics']['second_half'],'final_experts':summary['final_experts']}),flush=True)
    except Exception as exc:
        write_json(out/'failure.json',{'error':str(exc),'step':session.cursor,'traceback':traceback.format_exc(),
            'next_action':'Report before changing the registered experiment'})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=list('ABCD'),required=True)
    parser.add_argument('--model-seed',type=int,required=True);parser.add_argument('--stream',type=Path,required=True)
    parser.add_argument('--learning-rate',type=float,default=3e-5);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    run(parser.parse_args())
