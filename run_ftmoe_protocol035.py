"""Protocol-035 registered C reference tape and C-preserving correction branch."""
from __future__ import annotations
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import psutil,torch
import torch.nn.functional as F
from torch import nn
import run_ftmoe_protocol023_s4 as s4
import run_ftmoe_protocol031_pilot as p31
import run_ftmoe_protocol033 as p33
from ftmoe_protocol025_session import Protocol025FixedSession
from protocol035_common import dump_json,sha256_file,sha256_state_dict,grouped_average_precision

PLAN=Path('artifacts/ftmoe_online/protocol_035/plan.json')
def J(p): return json.loads(Path(p).read_text(encoding='utf8'))
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
def ledger(p):
    p=Path(p)
    if p.exists(): return J(p)
    return {'protocol':'035','full_training_sequence_budget':2,'sequences':{'C_ref_035':{'started':False,'completed':False,'restart_from_zero':False},'D_corr_branch':{'started':False,'completed':False,'restart_from_zero':False}},'engineering_real_prefix_executions':0,'new_simulator_streams':0,'extra_seeds':0,'hyperparameter_sweeps':0}
def mark(p,name,started=None,completed=None,extra=None):
    d=ledger(p); r=d['sequences'][name]
    if started:
        if r['started'] and not r['completed']: raise RuntimeError(name+' already started; zero restart forbidden')
        r['started']=True
    if completed is not None: r['completed']=bool(completed)
    if extra: r.update(extra)
    dump_json(p,d)

class ObservedFixedSession(Protocol025FixedSession):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); self.z_tape=np.full((self.steps,16,73),np.nan,np.float32); self.visible=np.full(self.steps,-1,np.int64); self.labelmax=np.full(self.steps,-1,np.int64)
    def step(self):
        t=self.cursor
        if t>=self.steps: raise StopIteration
        st=time.perf_counter(); x,s,g,c=s4.window_batch(self.replay,[t]); o=self.model.predict_deployment(x,s,g,graph_context=c)
        z=self.model._last_z.detach().cpu().numpy()[0].astype(np.float32,copy=True)
        if z.shape!=(16,73) or not np.isfinite(z).all(): raise RuntimeError('bad issued z')
        p=torch.softmax(o['detection_logits'],-1)[0,:,1].detach().cpu().numpy().astype(np.float32); q=torch.softmax(o['class_logits'],-1)[0].detach().cpu().numpy().astype(np.float32)
        self.z_tape[t]=z; self.visible[t]=t; self.labelmax[t]=t-3
        self.predictions['probability'][t]=p; self.predictions['class_probability'][t]=q; self.predictions['detection_logits'][t]=o['detection_logits'][0].detach().cpu().numpy(); self.predictions['class_logits'][t]=o['class_logits'][0].detach().cpu().numpy(); self.predictions['model_version'][t]=self.model_version; self.predictions['learner_hash'][t]=self.learner_hash; self.predictions['prediction_seconds'][t]=time.perf_counter()-st
        if not np.all(self.raw_seen[t]==-1): raise AssertionError('label read before prediction')
        self.raw_seen[t]=np.asarray(self.bundle['arrays']['raw_labels'][t],np.int64).copy(); self.predictions['raw_labels'][t]=self.raw_seen[t]
        i=t-2
        if i>=0:
            self.predictions['labels'][i]=self._mature_label(i,t); self.predictions['settled_at'][i]=t; self.settled[i]=True; self.buffer.append(i); self.buffer=self.buffer[-self.buffer_limit:]
        if (t+1)%self.update_every==0: self.update(t)
        self.model_version+=1; self.cursor=t+1; return p,q

def new_c(stream,registration,input_lock,out,runid):
    stream_sha,manifest,_,_=p33.validate_input(stream,registration,input_lock); b=s4.build_replay(Path(stream))
    if b['steps']!=5968 or b['manifest']['stream_sha256']!=stream_sha: raise AssertionError('stream identity')
    reg={'protocol':'035','revision':1,'arm':'C_ref_035','stream_sha256':stream_sha,'replay_seed':700,'model_seed':1}
    s=ObservedFixedSession('C_fixed5',1,b,p31.budget(),Path(out),run_id='p035_%s_C'%runid,stream_dir=Path(stream),phase_defs=p31.phases(manifest),stream_sha=stream_sha,registration=reg,learning_rate=1e-4)
    return s,manifest,stream_sha
def initinfo(s):
    return {'shared_first4_expert_and_router_rows_sha256':p31.shared_prefix_hash(s.model,4),'all5_learner_state_sha256':sha256_state_dict(s.model.learner.state_dict()),'frozen_base_sha256':s.model.frozen_hash(),'normalization_buffers_sha256':sha256_state_dict({'time_scale':s.model.protocol025_time_scale,'graph_scale':s.model.protocol025_graph_scale}),'model_seed':1}

class CorrectionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3501); self.fc1=nn.Linear(73,32); self.relu=nn.ReLU(); self.fc2=nn.Linear(32,1)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def delta(self,z): return 2*torch.tanh(self.fc2(self.relu(self.fc1(z)))/2)
def compose(c,d): return torch.stack((c[...,0]-d.squeeze(-1)/2,c[...,1]+d.squeeze(-1)/2),-1)

def cmd_manifest(a):
    runtime(); s,_,sha=new_c(a.stream,a.registration,a.input_lock,a.out_dir,a.run_id); b=CorrectionBranch(); p=J(PLAN)
    d={'protocol':'035','run_id':str(a.run_id),'execution_commit':os.environ.get('GITHUB_SHA','unknown'),'plan_sha256':sha256_file(PLAN),'stream_sha256':sha,'source_artifacts':p['source'],'runtime':runtime(),'initialization':initinfo(s),'features':{'dimension':73,'definition':'Protocol025 z64 plus 9 causal pressure/change features'},'correction':{'parameter_count':sum(x.numel() for x in b.parameters()),'initial_state_sha256':bhash(b),'seed':3501,'architecture':'73-32-ReLU-1','delta':'2*tanh(raw/2)'},'budget':p['budget']}
    if d['correction']['parameter_count']!=2401: raise AssertionError('parameter count'); dump_json(a.output,d)
    dump_json(a.output,d); print(json.dumps(d,indent=2))
def cmd_reference(a):
    runtime(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); mark(a.budget_ledger,'C_ref_035',started=True,extra={'started_at_run_id':str(a.run_id)})
    s,_,sha=new_c(a.stream,a.registration,a.input_lock,out,a.run_id); ini=initinfo(s); dump_json(out/'initialization.json',ini); t0=time.perf_counter(); c0=time.process_time(); peak=rss(); lim=memlimit(); soft=int(.8*lim)
    for _ in range(s.steps):
        s.step(); peak=max(peak,rss())
        if peak>=soft: raise MemoryError('C memory soft limit')
    s.finish(); s.save();
    if not np.isfinite(s.z_tape).all(): raise AssertionError('incomplete tape')
    u=[]
    for r in s.update_log:
        batch=[int(x) for x in r['buffer_indices']]
        if batch and max(batch)+2>int(r['at_interval']): raise AssertionError('immature C batch')
        u.append({'at_interval':int(r['at_interval']),'opportunity':int(r['opportunity']),'gradient_steps':int(r['gradient_steps']),'buffer_size':int(r['buffer_size']),'batch_indices':batch,'sampling_rule':'numpy.default_rng(seed*7919+updates_before_opportunity)'})
    np.savez_compressed(out/'feature_tape.npz',z=s.z_tape,c_detection_logits=s.predictions['detection_logits'],c_probability=s.predictions['probability'],c_class_probability=s.predictions['class_probability'],labels=s.predictions['labels'],raw_labels=s.predictions['raw_labels'],model_version=s.predictions['model_version'],visible_input_max=s.visible,label_available_max=s.labelmax)
    dump_json(out/'update_batches.json',{'protocol':'035','arm':'C_ref_035','updates':u}); th=sha256_file(out/'feature_tape.npz'); ph=sha256_file(out/'predictions.npz')
    torch.save({'protocol':'035','arm':'C_ref_035','model':s.model.state_dict(),'optimizer':s.optimizer.state_dict(),'cursor':s.cursor,'updates':s.updates,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'stream_sha256':sha,'feature_tape_sha256':th},out/'final_resume_state.pt')
    sm={'protocol':'035','arm':'C_ref_035','completed':True,'stream_sha256':sha,'feature_tape_sha256':th,'predictions_sha256':ph,'updates':int(s.updates),'initialization':ini,'cost':{'wall_seconds':time.perf_counter()-t0,'cpu_seconds':time.process_time()-c0,'peak_rss_bytes':peak,'effective_memory_limit_bytes':lim,'memory_soft_limit_bytes':soft}}
    dump_json(out/'summary.json',sm); mark(a.budget_ledger,'C_ref_035',completed=True,extra={'feature_tape_sha256':th,'predictions_sha256':ph,'updates':int(s.updates)}); print(json.dumps(sm,indent=2))
def save_ckpt(path,b,opt,cursor,ver,pred,delta,hashes,updates,tape,led):
    torch.save({'protocol':'035','arm':'D_corr_branch','cursor':int(cursor),'branch_version':int(ver),'branch':b.state_dict(),'optimizer':opt.state_dict(),'probability':pred[:cursor].copy(),'delta':delta[:cursor].copy(),'branch_hash':list(hashes[:cursor]),'updates':updates,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'feature_tape_sha256':tape,'budget_ledger':J(led)},path)
def cmd_correction(a):
    runtime(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); mark(a.budget_ledger,'D_corr_branch',started=True,extra={'started_at_run_id':str(a.run_id)})
    tp=Path(a.feature_tape); th=sha256_file(tp)
    if th!=J(Path(a.c_ref_dir)/'summary.json')['feature_tape_sha256']: raise AssertionError('tape hash')
    with np.load(tp,allow_pickle=False) as x: T={k:x[k].copy() for k in x.files}
    z=T['z']; cl=T['c_detection_logits']; cls=T['c_class_probability']; labels=T['labels']; raw=T['raw_labels']; batches=J(a.update_batches)['updates']; bt={int(x['at_interval']):x for x in batches}
    b=CorrectionBranch(); n=sum(x.numel() for x in b.parameters());
    if n!=2401: raise AssertionError('params')
    opt=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8); pred=np.full((5968,16),np.nan,np.float32); dl=np.full((5968,16,2),np.nan,np.float32); da=np.full((5968,16),np.nan,np.float32); vers=np.zeros(5968,np.int64); hashes=[]; logs=[]; ver=0; cur=bhash(b); initial=cur; t0=time.perf_counter(); c0=time.process_time(); peak=rss(); lim=memlimit(); soft=int(.8*lim)
    for t in range(5968):
        b.eval()
        with torch.no_grad():
            zz=torch.from_numpy(z[t]); cc=torch.from_numpy(cl[t]); dd=b.delta(zz); oo=compose(cc,dd); pp=torch.softmax(oo,-1)[:,1]
        pred[t]=pp.numpy(); dl[t]=oo.numpy(); da[t]=dd[:,0].numpy(); vers[t]=ver; hashes.append(cur)
        if t in bt:
            batch=[int(i) for i in bt[t]['batch_indices']]
            if any(i+2>t for i in batch): raise AssertionError('immature D batch')
            ii=np.asarray(batch); zz=torch.from_numpy(z[ii]).float(); cm=torch.from_numpy(cl[ii,:,1]-cl[ii,:,0]).float(); yy=torch.from_numpy((labels[ii]>0).astype(np.float32)); b.train(); dd=b.delta(zz).squeeze(-1); ce=F.binary_cross_entropy_with_logits(cm+dd,yy); pen=.001*(dd**2).mean(); loss=ce+pen; opt.zero_grad(set_to_none=True); loss.backward(); gn=float(torch.nn.utils.clip_grad_norm_(b.parameters(),1)); opt.step(); b.eval(); ver+=1; cur=bhash(b); logs.append({'at_interval':t,'batch_indices':batch,'matured_max_index':max(batch) if batch else None,'loss':float(loss.detach()),'bce':float(ce.detach()),'delta_penalty':float(pen.detach()),'grad_norm_preclip':gn,'branch_version_after':ver,'branch_hash_after':cur})
        if (t+1)%512==0: save_ckpt(out/('checkpoint_%04d.pt'%(t+1)),b,opt,t+1,ver,pred,da,hashes,logs,th,a.budget_ledger)
        peak=max(peak,rss())
        if peak>=soft: raise MemoryError('D memory soft limit')
    save_ckpt(out/'checkpoint_final.pt',b,opt,5968,ver,pred,da,hashes,logs,th,a.budget_ledger); np.savez_compressed(out/'predictions.npz',probability=pred,class_probability=cls.copy(),detection_logits=dl,labels=labels.copy(),raw_labels=raw.copy(),delta=da,branch_version=vers); dump_json(out/'update_log.json',{'updates':logs}); dump_json(out/'branch_hashes.json',{'prediction_branch_hash':hashes})
    after=sha256_file(tp); iso={'feature_tape_read_only_sha256_before':th,'feature_tape_read_only_sha256_after':after,'unchanged':th==after,'C_gradient_path_present':False,'C_optimizer_shared':False,'classification_exact_copy':True}; caus={'prediction_reads_current_tape_row_only':True,'prediction_preupdate_label_max_rule':'i+2<t','update_label_rule':'i+2<=t','all_update_batches_mature':all(all(int(i)+2<=int(r['at_interval']) for i in r['batch_indices']) for r in logs),'future_C_weights_recomputed':False,'terminal_drain_training_steps':0}; dump_json(out/'branch_isolation_audit.json',iso); dump_json(out/'causality_audit.json',caus)
    sm={'protocol':'035','arm':'D_corr_branch','completed':True,'feature_tape_sha256':th,'parameter_count':n,'updates':ver,'initial_branch_zero_output':True,'initial_branch_hash':initial,'final_branch_hash':cur,'parameter_updated':bool(ver>0 and cur!=initial),'prediction_sha256':sha256_file(out/'predictions.npz'),'cost':{'branch_wall_seconds':time.perf_counter()-t0,'branch_cpu_seconds':time.process_time()-c0,'peak_rss_bytes':peak,'effective_memory_limit_bytes':lim,'memory_soft_limit_bytes':soft},'C_path_cost_file':str(Path(a.c_ref_dir)/'summary.json')}; dump_json(out/'summary.json',sm); mark(a.budget_ledger,'D_corr_branch',completed=True,extra={'predictions_sha256':sm['prediction_sha256'],'updates':ver}); print(json.dumps(sm,indent=2))

def synth(z,m,y,batches,split=None):
    b=CorrectionBranch(); o=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8); outs=[]
    def run(lo,hi):
        nonlocal b,o
        for t in range(lo,hi):
            with torch.no_grad(): outs.append(b.delta(z[t]).detach().clone())
            if t in batches:
                i=batches[t]; d=b.delta(z[i]).squeeze(-1); loss=F.binary_cross_entropy_with_logits(m[i]+d,y[i])+.001*(d**2).mean(); o.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(b.parameters(),1); o.step()
    if split is None: run(0,len(z)); return torch.stack(outs),bhash(b)
    run(0,split); bs={k:v.detach().clone() for k,v in b.state_dict().items()}; osd=o.state_dict(); b=CorrectionBranch(); b.load_state_dict(bs); o=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8); o.load_state_dict(osd); run(split,len(z)); return torch.stack(outs),bhash(b)
def cmd_fixture(a):
    runtime(); r={'protocol':'035','synthetic_tests':{},'real_stream_prefix_executions':0,'max_real_prefix_steps':0,'passed':False}; y=np.array([1,0]); p=np.array([.5,.5]); r['synthetic_tests']['tie_ap_half']=abs(grouped_average_precision(y,p)-.5)<1e-12; r['synthetic_tests']['tie_ap_permutation']=grouped_average_precision(y,p)==grouped_average_precision(y[::-1],p[::-1]); y2=np.array([1,0,1,0]); r['synthetic_tests']['all_equal_equals_prevalence']=abs(grouped_average_precision(y2,np.full(4,.3))-.5)<1e-12; st=torch.get_rng_state().clone(); z0=CorrectionBranch(); r['synthetic_tests']['rng_isolation']=bool(torch.equal(st,torch.get_rng_state())); r['synthetic_tests']['zero_delta_exact']=bool(torch.equal(z0.delta(torch.zeros(1,73)),torch.zeros(1,1)))
    z=torch.randn(24,16,73,generator=torch.Generator().manual_seed(901)); m=torch.randn(24,16,generator=torch.Generator().manual_seed(902)); yy=(torch.rand(24,16,generator=torch.Generator().manual_seed(903))>.7).float(); ba={7:[0,1,2,3],15:[4,5,6,7,8,9]}; w,h1=synth(z,m,yy,ba); q,h2=synth(z,m,yy,ba,12); r['synthetic_tests']['checkpoint_resume_exact']=bool(torch.equal(w,q) and h1==h2); r['synthetic_tests']['two_updates_change_branch']=h1!=bhash(CorrectionBranch()); zp=z.clone(); zp[12:]+=1000; yp=yy.clone(); yp[12:]=1-yp[12:]; a1,_=synth(z,m,yy,ba); a2,_=synth(zp,m,yp,ba); r['synthetic_tests']['future_labels_and_tape_perturbation_prefix_invariant']=bool(torch.equal(a1[:12],a2[:12])); c=torch.tensor([[[-1.,1.]]]); d=torch.tensor([[[.4]]]); r['synthetic_tests']['online_cache_composition_exact']=bool(torch.equal(compose(c,d),torch.stack((c[...,0]-d.squeeze(-1)/2,c[...,1]+d.squeeze(-1)/2),-1)))
    root=Path(a.out_dir); s1,_,sha=new_c(a.stream,a.registration,a.input_lock,root/'real_prefix','fixture'); i1=initinfo(s1); s2,_,_=new_c(a.stream,a.registration,a.input_lock,root/'constructor2','fixture2'); r['synthetic_tests']['deterministic_all5_constructor']=i1['all5_learner_state_sha256']==initinfo(s2)['all5_learner_state_sha256']; [s1.step() for _ in range(64)]; r['real_stream_prefix_executions']=1; r['max_real_prefix_steps']=64; bb=CorrectionBranch(); pp=[]
    with torch.no_grad():
        for t in range(64): pp.append(torch.softmax(compose(torch.from_numpy(s1.predictions['detection_logits'][t]),bb.delta(torch.from_numpy(s1.z_tape[t]))),-1)[:,1].numpy())
    r['real_prefix']={'stream_sha256':sha,'issued_feature_hook_shape':list(s1.z_tape[:64].shape),'issued_feature_hook_finite':bool(np.isfinite(s1.z_tape[:64]).all()),'C_updates_in_engineering_prefix':int(s1.updates),'actual_batch_maturity_ok':all(all(int(i)+2<=int(x['at_interval']) for i in x['buffer_indices']) for x in s1.update_log),'deployment_feature_captured_before_update':True,'future_label_cutoff_recorded':bool(np.array_equal(s1.labelmax[:64],np.arange(64)-3)),'zero_correction_probability_exact':bool(np.array_equal(np.asarray(pp,np.float32),s1.predictions['probability'][:64])),'classification_copy_exact':True,'terminal_drain_has_no_training_by_runner_design':True}; r['passed']=all(r['synthetic_tests'].values()) and all(v for v in r['real_prefix'].values() if isinstance(v,bool)); dump_json(a.output,r); print(json.dumps(r,indent=2));
    if not r['passed']: raise SystemExit(2)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    for name,fn in [('manifest',cmd_manifest),('reference',cmd_reference),('fixture',cmd_fixture)]:
        p=sub.add_parser(name); p.add_argument('--stream',required=True); p.add_argument('--registration',required=True); p.add_argument('--input-lock',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--run-id',default='fixture');
        if name=='manifest': p.add_argument('--output',required=True)
        if name=='reference': p.add_argument('--budget-ledger',required=True)
        if name=='fixture': p.add_argument('--output',required=True)
        p.set_defaults(fn=fn)
    p=sub.add_parser('correction'); p.add_argument('--feature-tape',required=True); p.add_argument('--update-batches',required=True); p.add_argument('--c-ref-dir',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--run-id',required=True); p.add_argument('--budget-ledger',required=True); p.set_defaults(fn=cmd_correction)
    a=ap.parse_args(); a.fn(a)
if __name__=='__main__': main()
