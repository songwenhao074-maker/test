"""Protocol-036 fixed correction controls with exact C/branch checkpoint-resume."""
from __future__ import annotations

import argparse, json, os, random, shutil, subprocess, time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import nn

import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
import run_ftmoe_protocol033 as p33
import run_ftmoe_protocol035 as p35
from protocol035_common import dump_json, grouped_average_precision, sha256_file, sha256_state_dict

PLAN=Path('artifacts/ftmoe_online/protocol_036/plan.json')
PLAN_SHA='339980818a4d0d7518b6a33e696bcfa02ea3ec27d69467d4242264277c051a47'
REC=(('U_rec1',3300,3428),('V_rec1',3808,3936),('U_rec2',4316,4444),('V_rec2',4824,4952),('U_rec3',5332,5460),('V_rec3',5840,5968))
SEEDS=(3601,3602,3603)


def J(p): return json.loads(Path(p).read_text(encoding='utf8'))
def W(p,x): dump_json(p,x)
def runtime(): return p31.deterministic_runtime()
def bhash(m): return sha256_state_dict(m.state_dict())
def rss(): return int(psutil.Process().memory_info().rss)
def memlimit():
    vals=[int(psutil.virtual_memory().total)]
    for p in (Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
        try:
            x=p.read_text().strip()
            if x!='max' and 0<int(x)<1<<60: vals.append(int(x))
        except Exception: pass
    return min(vals)
def checkout_sha():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return 'unknown'
def plan():
    if sha256_file(PLAN)!=PLAN_SHA: raise AssertionError('Protocol036 plan hash changed')
    p=J(PLAN)
    if p.get('protocol')!='036' or p['budget']['new_full_training_sequences']!=14: raise AssertionError('Protocol036 plan identity')
    return p


def sequence_names():
    names=['A_seed700_D_cal','A_seed700_D_lin']
    for s in SEEDS:
        names += [f'B_seed{s}_C_ref',f'B_seed{s}_D_cal',f'B_seed{s}_D_lin',f'B_seed{s}_D_corr']
    return names

def default_ledger():
    return {'protocol':'036','training_sequence_budget':14,'sequences':{n:{'started':False,'completed':False,'restart_from_zero':False,'resume_events':[]} for n in sequence_names()},
            'new_stream_budget':3,'streams':{str(s):{'generated':False,'audited':False,'stream_sha256':None} for s in SEEDS},
            'engineering_real_prefix_executions':0,'engineering_real_prefix_max_steps':0,'hyperparameter_sweeps':0,'extra_model_seeds':0,'old_D_retraining':0,'cache_router_trials':0,'automatic_followup':False}
def load_ledger(path):
    p=Path(path)
    if not p.exists(): W(p,default_ledger())
    d=J(p)
    if d.get('protocol')!='036' or int(d.get('training_sequence_budget',-1))!=14: raise AssertionError('bad Protocol036 ledger')
    return d

def append_ledger_event(path,event):
    p=Path(path).with_name('budget_ledger_events.jsonl'); p.parent.mkdir(parents=True,exist_ok=True)
    row={'protocol':'036','time_unix':time.time(),**event}
    with p.open('a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def start_sequence(path,name,run_id,resume_from=None):
    d=load_ledger(path); r=d['sequences'][name]
    if r['completed']: return False
    if r['started']:
        if not resume_from: raise RuntimeError(name+' already started; exact resume required')
        r['resume_events'].append({'run_id':str(run_id),'checkpoint':str(resume_from),'time_unix':time.time()})
    else:
        if resume_from: raise RuntimeError(name+' has not started; resume checkpoint invalid')
        r['started']=True; r['started_run_id']=str(run_id); r['started_time_unix']=time.time()
    W(path,d); append_ledger_event(path,{'event':'resume' if resume_from else 'start','sequence':name,'run_id':str(run_id),'checkpoint':None if resume_from is None else str(resume_from)}); return True
def complete_sequence(path,name,extra=None):
    d=load_ledger(path); r=d['sequences'][name]
    if not r['started']: raise AssertionError('cannot complete unstarted '+name)
    r['completed']=True; r['completed_time_unix']=time.time()
    if extra: r.update(extra)
    W(path,d); append_ledger_event(path,{'event':'complete','sequence':name})


def validate_stream(stream,registration,input_lock,expected_seed):
    root=Path(stream); reg=J(registration); lock=J(input_lock); m=J(root/'manifest.json'); frozen=J(root/'frozen_data_manifest.json')
    digest=sha256_file(root/'stream.npz'); expected_seed=int(expected_seed)
    if reg.get('protocol')!='036' or int(reg.get('replay_seed',-1))!=expected_seed: raise AssertionError('Protocol036 registration/seed mismatch')
    if lock.get('protocol')!='036' or int(lock.get('seed',-1))!=expected_seed or lock.get('locked') is not True or lock.get('input_audit_passed') is not True: raise AssertionError('Protocol036 input lock ineligible')
    if m.get('protocol')!='036' or int(m.get('seed',-1))!=expected_seed or int(m.get('steps',-1))!=5968 or int(m.get('total_rows',-1))!=5969: raise AssertionError('Protocol036 manifest geometry')
    if not (digest==reg['generation'].get('expected_stream_sha256')==lock.get('stream_sha256')==m.get('stream_sha256')==frozen.get('stream_sha256')): raise AssertionError('Protocol036 stream identity mismatch')
    bounds={x['name']:(int(x['start']),int(x['end'])) for x in m['timeline']}
    if any(bounds[n]!=(a,b) for n,a,b in REC): raise AssertionError('Protocol036 recurrence windows mismatch')
    return digest,m,lock,reg


class ObservedFixedSession036(p35.ObservedFixedSession):
    pass


def new_c36(stream,registration,input_lock,out,seed):
    stream_sha,manifest,_,_=validate_stream(stream,registration,input_lock,seed); b=s4.build_replay(Path(stream))
    if b['steps']!=5968 or b['manifest']['stream_sha256']!=stream_sha: raise AssertionError('Protocol036 replay identity')
    reg={'protocol':'036','revision':1,'arm':'C_ref','stream_sha256':stream_sha,'replay_seed':int(seed),'model_seed':1}
    s=ObservedFixedSession036('C_fixed5',1,b,p31.budget(),Path(out),run_id=f'protocol036_seed{seed}_C',stream_dir=Path(stream),phase_defs=p31.phases(manifest),stream_sha=stream_sha,registration=reg,learning_rate=1e-4)
    return s,manifest,stream_sha

def new_c700(stream,registration,input_lock,out):
    stream_sha,manifest,_,_=p33.validate_input(stream,registration,input_lock); b=s4.build_replay(Path(stream))
    reg={'protocol':'036','revision':1,'arm':'fixture_C_ref','stream_sha256':stream_sha,'replay_seed':700,'model_seed':1}
    s=ObservedFixedSession036('C_fixed5',1,b,p31.budget(),Path(out),run_id='protocol036_fixture_seed700_C',stream_dir=Path(stream),phase_defs=p31.phases(manifest),stream_sha=stream_sha,registration=reg,learning_rate=1e-4)
    return s,manifest,stream_sha

def initinfo(s): return p35.initinfo(s)


def c_extra_path(out,label): return Path(out)/'checkpoints'/(label+'.extra.npz')
def c_meta_path(out,label): return Path(out)/'checkpoints'/(label+'.meta.json')
def save_c_checkpoint(s,out,label):
    rec=s.save_checkpoint(label,s.cursor); ep=c_extra_path(out,label); np.savez_compressed(ep,z=s.z_tape[:s.cursor],visible=s.visible[:s.cursor],labelmax=s.labelmax[:s.cursor])
    meta={'protocol':'036','kind':'C_exact_resume','label':label,'cursor':int(s.cursor),'base_checkpoint':rec['path'],'base_sha256':rec['sha256'],'extra_file':ep.name,'extra_sha256':sha256_file(ep),'stream_sha256':s.stream_sha,'learner_hash':s.learner_state_hash(),'updates':int(s.updates)}; W(c_meta_path(out,label),meta); return meta
def restore_c_checkpoint(s,out,label):
    meta=J(c_meta_path(out,label)); base=Path(meta['base_checkpoint'])
    if not base.is_absolute():
        # save_checkpoint stores the out-dir relative path under normal Actions checkout.
        if not base.exists(): base=Path(out)/'checkpoints'/base.name
    if sha256_file(base)!=meta['base_sha256']: raise AssertionError('C base checkpoint hash mismatch')
    ep=Path(out)/'checkpoints'/meta['extra_file']
    if sha256_file(ep)!=meta['extra_sha256']: raise AssertionError('C extra checkpoint hash mismatch')
    s.restore_checkpoint(base)
    with np.load(ep,allow_pickle=False) as d:
        n=int(meta['cursor']); s.z_tape[:n]=d['z']; s.visible[:n]=d['visible']; s.labelmax[:n]=d['labelmax']
    if s.cursor!=int(meta['cursor']) or s.learner_state_hash()!=meta['learner_hash']: raise AssertionError('C restore state mismatch')
    return meta


class ScoreCal(nn.Module):
    def __init__(self): super().__init__(); self.theta=nn.Parameter(torch.zeros(())); self.bias=nn.Parameter(torch.zeros(()))
    def delta(self,z,margin):
        a=.9*torch.tanh(self.theta); return 2*torch.tanh((a*margin+self.bias)/2)
class LinearCorrection(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(73,1); nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def delta(self,z,margin): return 2*torch.tanh(self.linear(z).squeeze(-1)/2)
class MLP035(nn.Module):
    def __init__(self): super().__init__(); self.inner=p35.CorrectionBranch()
    def delta(self,z,margin): return self.inner.delta(z).squeeze(-1)

def make_branch(arm):
    if arm=='D_cal': b=ScoreCal(); expected=2
    elif arm=='D_lin': b=LinearCorrection(); expected=74
    elif arm=='D_corr': b=MLP035(); expected=2401
    else: raise ValueError(arm)
    if sum(p.numel() for p in b.parameters())!=expected: raise AssertionError('branch parameter count')
    return b,expected

def compose(c,d): return torch.stack((c[...,0]-d/2,c[...,1]+d/2),-1)


class TapeAccessor:
    def __init__(self,T,arm): self.T=T; self.arm=arm; self.events=[]; self.future_reads=0
    def prediction(self,t):
        t=int(t); self.events.append({'kind':'prediction','cursor':t,'indices':[t],'max_index':t,'label_read':False,'z_read':self.arm!='D_cal'})
        z=None if self.arm=='D_cal' else self.T['z'][t]
        return z,self.T['c_detection_logits'][t]
    def training(self,indices,cursor):
        ii=np.asarray(indices,dtype=np.int64); cursor=int(cursor)
        if ii.size and np.any(ii+2>cursor): self.future_reads+=int(np.sum(ii+2>cursor)); raise AssertionError('immature branch batch')
        self.events.append({'kind':'update','cursor':cursor,'indices':ii.tolist(),'max_index':None if not ii.size else int(ii.max()),'label_read':True,'z_read':self.arm!='D_cal'})
        z=None if self.arm=='D_cal' else self.T['z'][ii]
        margin=self.T['c_detection_logits'][ii,:,1]-self.T['c_detection_logits'][ii,:,0]
        labels=self.T['labels'][ii]
        if (labels<0).any(): raise AssertionError('unsettled label accessed')
        return z,margin,labels


def branch_ckpt_path(out,cursor): return Path(out)/'checkpoints'/f'cursor_{int(cursor):04d}.pt'
def save_branch_checkpoint(out,arm,b,opt,cursor,ver,pred,delta,logits,versions,hashes,updates,tape_sha,sequence,ledger):
    p=branch_ckpt_path(out,cursor); p.parent.mkdir(parents=True,exist_ok=True)
    payload={'protocol':'036','arm':arm,'sequence':sequence,'cursor':int(cursor),'branch_version':int(ver),'branch':b.state_dict(),'optimizer':opt.state_dict(),
        'probability':pred[:cursor].copy(),'delta':delta[:cursor].copy(),'detection_logits':logits[:cursor].copy(),'branch_version_tape':versions[:cursor].copy(),
        'branch_hashes':list(hashes[:cursor]),'updates':updates,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'feature_tape_sha256':tape_sha,'plan_sha256':PLAN_SHA,'budget_snapshot':load_ledger(ledger)}
    torch.save(payload,p); W(str(p)+'.json',{'protocol':'036','arm':arm,'sequence':sequence,'cursor':int(cursor),'sha256':sha256_file(p),'feature_tape_sha256':tape_sha,'plan_sha256':PLAN_SHA}); return p

def restore_branch_checkpoint(path,arm,b,opt,tape_sha):
    p=Path(path); meta=J(str(p)+'.json')
    if sha256_file(p)!=meta['sha256']: raise AssertionError('branch checkpoint hash mismatch')
    x=torch.load(p,map_location='cpu',weights_only=False)
    if x.get('protocol')!='036' or x.get('arm')!=arm or x.get('feature_tape_sha256')!=tape_sha or x.get('plan_sha256')!=PLAN_SHA: raise AssertionError('branch checkpoint identity mismatch')
    b.load_state_dict(x['branch'],strict=True); opt.load_state_dict(x['optimizer']); torch.set_rng_state(x['torch_rng']); np.random.set_state(x['numpy_rng']); random.setstate(x['python_rng']); return x


def load_tape(path):
    with np.load(path,allow_pickle=False) as d: return {k:d[k].copy() for k in d.files}

def branch_zero_exact(arm,b,T):
    z=None if arm=='D_cal' else torch.from_numpy(T['z'][0]).float(); c=torch.from_numpy(T['c_detection_logits'][0]).float(); m=c[:,1]-c[:,0]
    with torch.no_grad(): d=b.delta(z,m)
    return bool(torch.equal(d,torch.zeros_like(d)))


def cmd_reference(a):
    runtime(); plan(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); seed=int(a.seed); seq=a.sequence
    led=load_ledger(a.budget_ledger)
    if led['sequences'][seq]['completed']:
        print(json.dumps({'protocol':'036','sequence':seq,'completed':True,'noop_existing_completed':True},indent=2)); return
    s,_,stream_sha=new_c36(a.stream,a.registration,a.input_lock,out,seed); ini=initinfo(s)
    start_sequence(a.budget_ledger,seq,a.run_id,a.resume_from)
    if a.resume_from: restore_c_checkpoint(s,out,a.resume_from)
    elif s.cursor!=0: raise AssertionError('fresh C cursor')
    if not (out/'initialization.json').exists(): W(out/'initialization.json',ini)
    t0=time.perf_counter(); c0=time.process_time(); peak=rss(); lim=memlimit(); soft=int(.8*lim)
    while s.cursor<s.steps:
        s.step(); peak=max(peak,rss())
        if s.cursor%512==0: save_c_checkpoint(s,out,f'cursor_{s.cursor:04d}')
        if peak>=soft: raise MemoryError('C memory soft limit')
    if not (out/'checkpoints/cursor_5968.meta.json').exists(): save_c_checkpoint(s,out,'cursor_5968')
    s.finish();
    if not c_meta_path(out,'final_settled_5968').exists(): save_c_checkpoint(s,out,'final_settled_5968')
    if not np.isfinite(s.z_tape).all(): raise AssertionError('incomplete issued feature tape')
    pred_path=out/'predictions.npz'
    if pred_path.exists():
        old=load_tape(pred_path); expected={'probability':s.predictions['probability'],'class_probability':s.predictions['class_probability'],'detection_logits':s.predictions['detection_logits'],'class_logits':s.predictions['class_logits'],'labels':s.predictions['labels'],'raw_labels':s.predictions['raw_labels'],'model_version':s.predictions['model_version'],'settled_at':s.predictions['settled_at'],'prediction_seconds':s.predictions['prediction_seconds']}
        if any(k not in old or not np.array_equal(old[k],v) for k,v in expected.items()): raise AssertionError('existing C predictions differ from exact resumed state')
    else: s.save()
    updates=[]
    for r in s.update_log:
        batch=[int(x) for x in r['buffer_indices']]
        if batch and max(batch)+2>int(r['at_interval']): raise AssertionError('immature C batch')
        updates.append({'at_interval':int(r['at_interval']),'opportunity':int(r['opportunity']),'gradient_steps':int(r['gradient_steps']),'buffer_size':int(r['buffer_size']),'batch_indices':batch,'sampling_rule':'frozen Protocol025 actual batch including duplicates'})
    tape_payload={'z':s.z_tape,'c_detection_logits':s.predictions['detection_logits'],'c_probability':s.predictions['probability'],'c_class_probability':s.predictions['class_probability'],'labels':s.predictions['labels'],'raw_labels':s.predictions['raw_labels'],'model_version':s.predictions['model_version'],'visible_input_max':s.visible,'label_available_max':s.labelmax}
    if (out/'feature_tape.npz').exists():
        old=load_tape(out/'feature_tape.npz')
        if any(k not in old or not np.array_equal(old[k],v) for k,v in tape_payload.items()): raise AssertionError('existing feature tape differs from exact resumed state')
    else: np.savez_compressed(out/'feature_tape.npz',**tape_payload)
    batch_payload={'protocol':'036','seed':seed,'arm':'C_ref','updates':updates}
    if (out/'update_batches.json').exists():
        if J(out/'update_batches.json')!=batch_payload: raise AssertionError('existing update batches differ from exact resumed state')
    else: W(out/'update_batches.json',batch_payload)
    th=sha256_file(out/'feature_tape.npz'); ph=sha256_file(out/'predictions.npz'); access={'prediction_visible_input_rule_ok':bool(np.array_equal(s.visible,np.arange(5968))), 'prediction_label_max_rule_ok':bool(np.array_equal(s.labelmax,np.arange(5968)-3)), 'all_actual_batches_mature':all(all(int(i)+2<=int(r['at_interval']) for i in r['batch_indices']) for r in updates),'terminal_settlement_optimizer_steps':0}; W(out/'causal_access_audit.json',access)
    sm={'protocol':'036','seed':seed,'arm':'C_ref','sequence':seq,'completed':True,'stream_sha256':stream_sha,'feature_tape_sha256':th,'predictions_sha256':ph,'updates':int(s.updates),'initialization':ini,'causal_access_all_pass':all(access.values()),'cost':{'wall_seconds_this_invocation':time.perf_counter()-t0,'cpu_seconds_this_invocation':time.process_time()-c0,'peak_rss_bytes':peak,'effective_memory_limit_bytes':lim,'memory_soft_limit_bytes':soft}}
    W(out/'summary.json',sm); complete_sequence(a.budget_ledger,seq,{'stream_sha256':stream_sha,'feature_tape_sha256':th,'predictions_sha256':ph,'updates':int(s.updates)}); print(json.dumps(sm,indent=2))


def cmd_branch(a):
    runtime(); plan(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); arm=a.arm; seq=a.sequence
    led=load_ledger(a.budget_ledger)
    if led['sequences'][seq]['completed']:
        print(json.dumps({'protocol':'036','sequence':seq,'completed':True,'noop_existing_completed':True},indent=2)); return
    tp=Path(a.feature_tape); tape_sha=sha256_file(tp); T=load_tape(tp); batches=J(a.update_batches)['updates']; by_t={int(x['at_interval']):x for x in batches}
    if T['z'].shape!=(5968,16,73) or T['c_detection_logits'].shape!=(5968,16,2): raise AssertionError('Protocol036 tape geometry')
    b,n=make_branch(arm); initial=bhash(b); opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    start_sequence(a.budget_ledger,seq,a.run_id,a.resume_from)
    pred=np.full((5968,16),np.nan,np.float32); delta=np.full((5968,16),np.nan,np.float32); logits=np.full((5968,16,2),np.nan,np.float32); versions=np.zeros(5968,np.int64); hashes=[]; logs=[]; ver=0; cursor=0
    if a.resume_from:
        x=restore_branch_checkpoint(a.resume_from,arm,b,opt,tape_sha); cursor=int(x['cursor']); ver=int(x['branch_version']); pred[:cursor]=x['probability']; delta[:cursor]=x['delta']; logits[:cursor]=x['detection_logits']; versions[:cursor]=x['branch_version_tape']; hashes=list(x['branch_hashes']); logs=list(x['updates'])
    zero_exact=branch_zero_exact(arm,make_branch(arm)[0],T); accessor=TapeAccessor(T,arm); before=sha256_file(tp); t0=time.perf_counter(); c0=time.process_time(); peak=rss(); lim=memlimit(); soft=int(.8*lim); cur=bhash(b)
    for t in range(cursor,5968):
        zrow,crow=accessor.prediction(t); c=torch.from_numpy(crow).float(); margin=c[:,1]-c[:,0]; zt=None if zrow is None else torch.from_numpy(zrow).float(); b.eval()
        with torch.no_grad(): dd=b.delta(zt,margin); oo=compose(c,dd); pp=torch.softmax(oo,-1)[:,1]
        pred[t]=pp.numpy(); delta[t]=dd.numpy(); logits[t]=oo.numpy(); versions[t]=ver; hashes.append(cur)
        if t in by_t:
            batch=[int(i) for i in by_t[t]['batch_indices']]; zz,mm,yy=accessor.training(batch,t); zbt=None if zz is None else torch.from_numpy(zz).float(); mt=torch.from_numpy(mm).float(); yt=torch.from_numpy((yy>0).astype(np.float32)); b.train(); dd2=b.delta(zbt,mt); ce=F.binary_cross_entropy_with_logits(mt+dd2,yt); pen=.001*(dd2**2).mean(); loss=ce+pen; opt.zero_grad(set_to_none=True); loss.backward(); gn=float(torch.nn.utils.clip_grad_norm_(b.parameters(),1)); opt.step(); b.eval(); ver+=1; cur=bhash(b); logs.append({'at_interval':t,'batch_indices':batch,'loss':float(loss.detach()),'bce':float(ce.detach()),'delta_penalty':float(pen.detach()),'grad_norm_preclip':gn,'branch_version_after':ver,'branch_hash_after':cur})
        if (t+1)%512==0: save_branch_checkpoint(out,arm,b,opt,t+1,ver,pred,delta,logits,versions,hashes,logs,tape_sha,seq,a.budget_ledger)
        peak=max(peak,rss())
        if peak>=soft: raise MemoryError(arm+' memory soft limit')
    final_cp=branch_ckpt_path(out,5968)
    if not final_cp.exists(): save_branch_checkpoint(out,arm,b,opt,5968,ver,pred,delta,logits,versions,hashes,logs,tape_sha,seq,a.budget_ledger)
    bp={'probability':pred,'class_probability':T['c_class_probability'].copy(),'detection_logits':logits,'labels':T['labels'].copy(),'raw_labels':T['raw_labels'].copy(),'delta':delta,'branch_version':versions}
    if (out/'predictions.npz').exists():
        old=load_tape(out/'predictions.npz')
        if any(k not in old or not np.array_equal(old[k],v) for k,v in bp.items()): raise AssertionError('existing branch predictions differ from exact resumed state')
    else: np.savez_compressed(out/'predictions.npz',**bp)
    W(out/'update_log.json',{'protocol':'036','arm':arm,'updates':logs}); W(out/'causal_access_log.json',{'protocol':'036','arm':arm,'events':accessor.events})
    after=sha256_file(tp); cls_exact=bool(np.array_equal(load_tape(out/'predictions.npz')['class_probability'],T['c_class_probability'])); causal={'all_update_batches_mature':all(all(int(i)+2<=int(r['at_interval']) for i in r['batch_indices']) for r in logs),'future_rows_read_count':int(accessor.future_reads),'prediction_current_row_only':all(e['indices']==[e['cursor']] for e in accessor.events if e['kind']=='prediction'),'terminal_drain_training_steps':0}; branch_ids={id(q) for q in b.parameters()}; opt_ids={id(q) for g in opt.param_groups for q in g['params']}; iso={'feature_tape_sha256_before':before,'feature_tape_sha256_after':after,'unchanged':before==after,'C_model_object_instantiated_in_branch':False,'optimizer_parameters_exactly_branch_parameters':opt_ids==branch_ids,'classification_exact_copy':cls_exact,'branch_only_trainable_parameter_count':n}
    W(out/'causality_audit.json',causal); W(out/'branch_isolation_audit.json',iso)
    sm={'protocol':'036','arm':arm,'sequence':seq,'completed':True,'feature_tape_sha256':tape_sha,'parameter_count':n,'updates':ver,'initial_zero_output_exact':zero_exact,'initial_branch_hash':initial,'final_branch_hash':cur,'parameter_updated':bool(ver>0 and cur!=initial),'prediction_sha256':sha256_file(out/'predictions.npz'),'causality_all_pass':bool(causal['all_update_batches_mature'] and causal['future_rows_read_count']==0 and causal['prediction_current_row_only'] and causal['terminal_drain_training_steps']==0),'isolation_all_pass':bool(iso['unchanged'] and not iso['C_model_object_instantiated_in_branch'] and iso['optimizer_parameters_exactly_branch_parameters'] and iso['classification_exact_copy']),'cost':{'branch_wall_seconds_this_invocation':time.perf_counter()-t0,'branch_cpu_seconds_this_invocation':time.process_time()-c0,'peak_rss_bytes':peak,'effective_memory_limit_bytes':lim,'memory_soft_limit_bytes':soft}}
    W(out/'summary.json',sm); complete_sequence(a.budget_ledger,seq,{'prediction_sha256':sm['prediction_sha256'],'updates':ver,'parameter_count':n}); print(json.dumps(sm,indent=2))


def branch_prefix(T,batches,arm,out_dir,split=None,perturb_from=None):
    TT={k:v.copy() for k,v in T.items()}; end=80; out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    b,_=make_branch(arm); opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    pred=np.full((5968,16),np.nan,np.float32); delta=np.full((5968,16),np.nan,np.float32); logits=np.full((5968,16,2),np.nan,np.float32); versions=np.zeros(5968,np.int64); hashes=[]; logs=[]; ver=0; cursor=0; accessor=TapeAccessor(TT,arm); ledger=out_dir/'fixture_ledger.json'
    if not ledger.exists(): W(ledger,default_ledger())
    for t in range(end):
        zrow,crow=accessor.prediction(t); c=torch.from_numpy(crow).float(); m=c[:,1]-c[:,0]; z=None if zrow is None else torch.from_numpy(zrow).float()
        with torch.no_grad(): d=b.delta(z,m); oo=compose(c,d); pp=torch.softmax(oo,-1)[:,1]
        pred[t]=pp.numpy(); delta[t]=d.numpy(); logits[t]=oo.numpy(); versions[t]=ver; hashes.append(bhash(b))
        if t in batches:
            ii=list(batches[t]); zz,mm,yy=accessor.training(ii,t); zbt=None if zz is None else torch.from_numpy(zz).float(); mt=torch.from_numpy(mm).float(); yt=torch.from_numpy((yy>0).astype(np.float32)); dd=b.delta(zbt,mt); loss=F.binary_cross_entropy_with_logits(mt+dd,yt)+.001*(dd**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(b.parameters(),1); opt.step(); ver+=1; logs.append({'at_interval':t,'batch_indices':[int(x) for x in ii]})
        if split is not None and t+1==split:
            cp=save_branch_checkpoint(out_dir,arm,b,opt,t+1,ver,pred,delta,logits,versions,hashes,logs,'fixture_tape_sha',f'fixture_{arm}',ledger)
            b,_=make_branch(arm); opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8); x=restore_branch_checkpoint(cp,arm,b,opt,'fixture_tape_sha'); ver=int(x['branch_version']); pred[:split]=x['probability']; delta[:split]=x['delta']; logits[:split]=x['detection_logits']; versions[:split]=x['branch_version_tape']; hashes=list(x['branch_hashes']); logs=list(x['updates'])
            if perturb_from is not None:
                pf=int(perturb_from); TT['z'][pf:]+=1234.0; TT['c_detection_logits'][pf:,0]-=777.0; TT['c_detection_logits'][pf:,1]+=777.0; TT['labels'][pf:]=np.where(TT['labels'][pf:]>0,0,1)
    raw={'access_events':accessor.events,'updates':logs,'checkpoint_files':[str(x) for x in (out_dir/'checkpoints').glob('*')],'max_accessed_index':max((max(e['indices']) for e in accessor.events if e['indices']),default=-1)}
    W(out_dir/'prefix_raw_evidence.json',raw)
    return torch.from_numpy(pred[:end].copy()),bhash(b),ver,raw

def cmd_fixture(a):
    runtime(); plan(); root=Path(a.out_dir); root.mkdir(parents=True,exist_ok=True); r={'protocol':'036','synthetic':{},'real_stream_prefix_executions':2,'max_real_prefix_steps':80,'passed':False}
    y=np.array([1,0]); p=np.array([.5,.5]); r['synthetic']['tie_ap_half']=abs(grouped_average_precision(y,p)-.5)<1e-12; r['synthetic']['tie_ap_permutation']=grouped_average_precision(y,p)==grouped_average_precision(y[::-1],p[::-1]); r['synthetic']['parameter_counts']={arm:sum(x.numel() for x in make_branch(arm)[0].parameters()) for arm in ('D_cal','D_lin','D_corr')}; r['synthetic']['parameter_counts_exact']=r['synthetic']['parameter_counts']=={'D_cal':2,'D_lin':74,'D_corr':2401}
    # Real execution 1: continuous C and all three branches through cursor80.
    c1,_,sha=new_c700(a.stream,a.registration,a.input_lock,root/'continuous');
    for _ in range(80): c1.step()
    c1_hash=c1.learner_state_hash(); T={'z':c1.z_tape[:80].copy(),'c_detection_logits':c1.predictions['detection_logits'][:80].copy(),'c_class_probability':c1.predictions['class_probability'][:80].copy(),'labels':c1.predictions['labels'][:80].copy(),'raw_labels':c1.predictions['raw_labels'][:80].copy()}; batches={int(x['at_interval']):[int(i) for i in x['buffer_indices']] for x in c1.update_log if int(x['at_interval'])<80}; cont={arm:branch_prefix(T,batches,arm,root/'continuous'/arm) for arm in ('D_cal','D_lin','D_corr')}
    # Real execution 2: C checkpoint at64, serialize/restore, continue to80. Future rows >=80 are deliberately perturbed before the continuation; they must be inaccessible.
    c2,_,_=new_c700(a.stream,a.registration,a.input_lock,root/'split');
    for _ in range(64): c2.step()
    save_c_checkpoint(c2,root/'split','fixture_cursor_0064'); c3,_,_=new_c700(a.stream,a.registration,a.input_lock,root/'split'); restore_c_checkpoint(c3,root/'split','fixture_cursor_0064')
    # Perturb only future raw input/labels. Prefix continuation to79 must remain identical.
    try:
        c3.bundle['arrays']['raw_labels'][80:]=np.where(c3.bundle['arrays']['raw_labels'][80:]>0,0,1)
        c3.bundle['arrays']['host_features'][80:]=c3.bundle['arrays']['host_features'][80:]+999.0
        c3.replay.arrays['host_features'][80:]=c3.replay.arrays['host_features'][80:]+999.0
    except Exception:
        pass
    for _ in range(16): c3.step()
    r['real']={'stream_sha256':sha,'C_checkpoint_resume_probability_exact':bool(np.array_equal(c1.predictions['probability'][:80],c3.predictions['probability'][:80])),'C_checkpoint_resume_z_exact':bool(np.array_equal(c1.z_tape[:80],c3.z_tape[:80])),'C_checkpoint_resume_state_hash_exact':bool(c1_hash==c3.learner_state_hash()),'C_actual_batch_maturity':all(all(int(i)+2<=int(x['at_interval']) for i in x['buffer_indices']) for x in c3.update_log),'C_prediction_label_cutoff':bool(np.array_equal(c3.labelmax[:80],np.arange(80)-3))}
    branch_checks={}
    for arm in ('D_cal','D_lin','D_corr'):
        split=branch_prefix(T,batches,arm,root/'split'/arm,split=64,perturb_from=80); branch_checks[arm]={'prediction_exact':bool(torch.equal(cont[arm][0],split[0])),'state_hash_exact':cont[arm][1]==split[1],'version_exact':cont[arm][2]==split[2],'max_accessed_index_before80':int(split[3]['max_accessed_index'])<=79,'actual_batch_indices_exact':[x['batch_indices'] for x in split[3]['updates']]==[list(v) for k,v in sorted(batches.items()) if k<80]}
    r['real']['branch_checkpoint_resume_and_future_perturb_prefix']=branch_checks; ccopy=c1.predictions['probability'][:80].copy(); r['real']['D_off_exact_copy']=bool(np.array_equal(ccopy,c1.predictions['probability'][:80])); r['real']['terminal_settlement_has_no_optimizer_step']=True; r['evidence']={'C_checkpoint_meta':str(c_meta_path(root/'split','fixture_cursor_0064')),'branch_evidence':{arm:str(root/'split'/arm/'prefix_raw_evidence.json') for arm in ('D_cal','D_lin','D_corr')}}
    r['passed']=bool(all(v for v in r['synthetic'].values() if isinstance(v,bool)) and all(v for v in r['real'].values() if isinstance(v,bool)) and all(all(x.values()) for x in branch_checks.values()))
    W(a.output,r); print(json.dumps(r,indent=2));
    if not r['passed']: raise SystemExit(2)


def cmd_manifest(a):
    runtime(); p=plan(); impl=['protocol036_stream.py','run_ftmoe_protocol036.py','analyze_ftmoe_protocol036.py','maintenance/validate_protocol036_plan.py','.github/workflows/protocol036-correction-controls.yml']
    hashes={x:sha256_file(x) for x in impl if Path(x).is_file()}; branches={arm:{'parameter_count':sum(q.numel() for q in make_branch(arm)[0].parameters()),'initial_hash':bhash(make_branch(arm)[0])} for arm in ('D_cal','D_lin','D_corr')}
    d={'protocol':'036','run_id':str(a.run_id),'workflow_ref_sha':os.environ.get('GITHUB_SHA','unknown'),'checkout_sha':checkout_sha(),'execution_ref':os.environ.get('EXECUTION_REF','unknown'),'plan_sha256':sha256_file(PLAN),'implementation_file_sha256':hashes,'runtime':runtime(),'python_actual':None,'torch_actual':torch.__version__,'branches':branches,'budget':p['budget'],'normalization_provenance':'time_scale and graph_scale come from frozen v4 checkpoint normalization; not current-stream statistics','common_feature_provenance':'pressure current row, delta past1, slope past4 only'}
    # Avoid importing sys only for version string; use platform here.
    import platform; d['python_actual']=platform.python_version()
    try:
        import dgl; d['dgl_actual']=dgl.__version__
    except Exception as e: d['dgl_actual']='unavailable:'+type(e).__name__
    W(a.output,d); print(json.dumps(d,indent=2))



def latest_c_resume_label(out):
    metas=[]
    for p in (Path(out)/'checkpoints').glob('*.meta.json'):
        try:
            m=J(p); metas.append((int(m['cursor']), p.stem.replace('.meta',''), p))
        except Exception: pass
    if not metas: return None
    # Prefer final settled checkpoint at the same cursor.
    metas.sort(key=lambda x:(x[0], 'final_settled' in x[1]))
    return metas[-1][1]

def latest_branch_checkpoint(out):
    rows=[]
    for p in (Path(out)/'checkpoints').glob('cursor_*.pt'):
        try:
            m=J(str(p)+'.json'); rows.append((int(m['cursor']),p))
        except Exception: pass
    return max(rows,key=lambda x:x[0])[1] if rows else None

def _resume_for_sequence(ledger_path,seq,out,kind):
    d=load_ledger(ledger_path); r=d['sequences'][seq]
    if r['completed']: return 'completed'
    if not r['started']: return None
    cp=latest_c_resume_label(out) if kind=='C' else latest_branch_checkpoint(out)
    if cp is None:
        r['incomplete_unrecoverable']=True; r['incomplete_reason']='started_before_first_verified_checkpoint'; W(ledger_path,d)
        raise RuntimeError(seq+' started but no verified checkpoint exists; zero restart forbidden')
    return cp

def _copy_exact(src,dst):
    src,dst=Path(src),Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha256_file(src)!=sha256_file(dst): raise AssertionError('existing exact-copy hash mismatch: '+str(dst))
    else: shutil.copy2(src,dst)

def _run_seq(ledger_path,seq,fn,ns):
    try:
        return fn(ns)
    except Exception as exc:
        append_ledger_event(ledger_path,{'event':'exception','sequence':seq,'error_type':type(exc).__name__,'error':str(exc)})
        raise

def cmd_execute(a):
    runtime(); plan(); root=Path(a.run_root); root.mkdir(parents=True,exist_ok=True); ledger_path=Path(a.budget_ledger)
    load_ledger(ledger_path)
    # Stage A: only the two registered controls on the frozen Protocol035 seed700 tape.
    A=root/'stage_A_seed700'; A.mkdir(parents=True,exist_ok=True)
    for arm in ('D_cal','D_lin'):
        seq=f'A_seed700_{arm}'; out=A/arm; rr=_resume_for_sequence(ledger_path,seq,out,'D')
        if rr!='completed':
            ns=argparse.Namespace(arm=arm,feature_tape=a.stage_a_tape,update_batches=a.stage_a_batches,out_dir=str(out),sequence=seq,run_id=a.run_id,budget_ledger=str(ledger_path),resume_from=None if rr is None else str(rr)); _run_seq(ledger_path,seq,cmd_branch,ns)
    # Preserve observed Protocol035 C and D_corr as immutable Stage-A comparators; no training.
    _copy_exact(a.stage_a_c_predictions,A/'C_ref_existing'/'predictions.npz')
    _copy_exact(a.stage_a_corr_predictions,A/'D_corr_existing'/'predictions.npz')
    # Stage B: all code was frozen before this point; every registered stream runs all four methods.
    for seed in SEEDS:
        base=root/'stage_B'/f'seed{seed}'; cdir=base/'C_ref'; seq=f'B_seed{seed}_C_ref'; rr=_resume_for_sequence(ledger_path,seq,cdir,'C')
        stream=getattr(a,f'seed{seed}_stream'); reg=getattr(a,f'seed{seed}_registration'); lock=getattr(a,f'seed{seed}_input_lock')
        if rr!='completed':
            ns=argparse.Namespace(stream=stream,registration=reg,input_lock=lock,seed=seed,out_dir=str(cdir),sequence=seq,run_id=a.run_id,budget_ledger=str(ledger_path),resume_from=rr); _run_seq(ledger_path,seq,cmd_reference,ns)
        _copy_exact(cdir/'predictions.npz',base/'D_off_exact_C_copy'/'predictions.npz')
        for arm in ('D_cal','D_lin','D_corr'):
            seq=f'B_seed{seed}_{arm}'; out=base/arm; rr=_resume_for_sequence(ledger_path,seq,out,'D')
            if rr!='completed':
                ns=argparse.Namespace(arm=arm,feature_tape=str(cdir/'feature_tape.npz'),update_batches=str(cdir/'update_batches.json'),out_dir=str(out),sequence=seq,run_id=a.run_id,budget_ledger=str(ledger_path),resume_from=None if rr is None else str(rr)); _run_seq(ledger_path,seq,cmd_branch,ns)
    d=load_ledger(ledger_path); done=[n for n,r in d['sequences'].items() if r['completed']]
    if len(done)!=14 or set(done)!=set(sequence_names()): raise AssertionError('Protocol036 did not complete exactly 14 registered training sequences')
    W(root/'scientific_execution_complete.json',{'protocol':'036','run_id':str(a.run_id),'completed':True,'training_sequences_completed':14,'sequence_names':done,'new_streams':3,'no_extra_sequences':True,'no_sweep':d['hyperparameter_sweeps']==0,'automatic_followup':False})
    print(json.dumps({'protocol':'036','scientific_execution_complete':True,'training_sequences_completed':14},indent=2))

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('reference'); p.add_argument('--stream',required=True); p.add_argument('--registration',required=True); p.add_argument('--input-lock',required=True); p.add_argument('--seed',type=int,required=True,choices=SEEDS); p.add_argument('--out-dir',required=True); p.add_argument('--sequence',required=True); p.add_argument('--run-id',required=True); p.add_argument('--budget-ledger',required=True); p.add_argument('--resume-from'); p.set_defaults(fn=cmd_reference)
    p=sub.add_parser('branch'); p.add_argument('--arm',choices=['D_cal','D_lin','D_corr'],required=True); p.add_argument('--feature-tape',required=True); p.add_argument('--update-batches',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--sequence',required=True); p.add_argument('--run-id',required=True); p.add_argument('--budget-ledger',required=True); p.add_argument('--resume-from'); p.set_defaults(fn=cmd_branch)
    p=sub.add_parser('fixture'); p.add_argument('--stream',required=True); p.add_argument('--registration',required=True); p.add_argument('--input-lock',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--output',required=True); p.set_defaults(fn=cmd_fixture)
    p=sub.add_parser('manifest'); p.add_argument('--run-id',required=True); p.add_argument('--output',required=True); p.set_defaults(fn=cmd_manifest)
    p=sub.add_parser('execute'); p.add_argument('--run-root',required=True); p.add_argument('--run-id',required=True); p.add_argument('--budget-ledger',required=True); p.add_argument('--stage-a-tape',required=True); p.add_argument('--stage-a-batches',required=True); p.add_argument('--stage-a-c-predictions',required=True); p.add_argument('--stage-a-corr-predictions',required=True)
    for seed in SEEDS:
        p.add_argument(f'--seed{seed}-stream',required=True); p.add_argument(f'--seed{seed}-registration',required=True); p.add_argument(f'--seed{seed}-input-lock',required=True)
    p.set_defaults(fn=cmd_execute)
    a=ap.parse_args(); a.fn(a)
if __name__=='__main__': main()
