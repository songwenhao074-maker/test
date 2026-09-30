import sys,json,hashlib,pathlib,numpy as np
root=pathlib.Path(sys.argv[1]);r=root/'raw/artifacts/ftmoe_online/protocol_035/runtime/run_36585058415'
def J(p):return json.loads(p.read_text())
def N(p):return dict(np.load(p,allow_pickle=False))
def ap(y,p):
 y=(y.ravel()>0);p=p.ravel(); idx=np.argsort(-p,kind='stable'); y=y[idx];p=p[idx];end=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1];tp=np.cumsum(y)[end];return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/(end+1)))
def met(y,p):
 yy=y>0;pp=p>=.5;return {'ap':ap(y,p),'recall':float(pp[yy].mean()),'fpr':float(pp[~yy].mean()),'tp':int((pp&yy).sum()),'fp':int((pp&~yy).sum())}
c=N(r/'C_ref_035/predictions.npz');d=N(r/'D_corr_branch/predictions.npz');t=N(r/'C_ref_035/feature_tape.npz');o=N(r/'D_off_exact_C_copy/predictions.npz');comp=J(r/'analysis/comparison.json')
checks={};checks['manifest_hashes']=all(hashlib.sha256((r/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in J(r/'raw_artifact_manifest.json')['files']); checks['off_all_arrays_exact']=all(np.array_equal(c[k],o[k],equal_nan=True) for k in c if c[k].dtype.kind in 'fiub');checks['class_exact']=np.array_equal(c['class_probability'],d['class_probability']);checks['labels_exact']=np.array_equal(c['labels'],d['labels']); checks['tape_C_exact']=np.array_equal(t['c_detection_logits'],c['detection_logits']);checks['finite_features']=bool(np.isfinite(t['z']).all());checks['visible_input_max']=np.array_equal(t['visible_input_max'],np.arange(5968));checks['label_available_max']=np.array_equal(t['label_available_max'],np.arange(5968)-3)
cb=J(r/'C_ref_035/update_batches.json')['updates'];db=J(r/'D_corr_branch/update_log.json')['updates'];checks['batch_exact']=len(cb)==len(db) and all(x['at_interval']==y['at_interval'] and x['batch_indices']==y['batch_indices'] for x,y in zip(cb,db));checks['maturity']=all(all(i+2<=x['at_interval'] for i in x['batch_indices']) for x in db);checks['frequency']= [x['at_interval'] for x in db]==list(range(15,5968,16));checks['prediction_versions']=np.array_equal(d['branch_version'],np.arange(5968)//16);checks['first16_zero_delta']=bool((d['delta'][:16]==0).all());checks['issued_logit_composition']=np.array_equal(d['detection_logits'][...,0],c['detection_logits'][...,0]-d['delta']/2) and np.array_equal(d['detection_logits'][...,1],c['detection_logits'][...,1]+d['delta']/2)
y=c['labels'];cp=c['probability'];dp=d['probability'];full={'C':met(y,cp),'D':met(y,dp)};checks['full_AP_match']=abs(full['D']['ap']-full['C']['ap']-comp['full_AP_delta'])<1e-12
wins=[]
for w in comp['windows']:
 a,b=w['intervals'];cc=met(y[a:b],cp[a:b]);dd=met(y[a:b],dp[a:b]);wins.append({'name':w['window'],'C':cc,'D':dd,'ap_delta':dd['ap']-cc['ap']});checks['AP_'+w['window']]=abs(wins[-1]['ap_delta']-w['ap_delta'])<1e-12
phases=[('F0',0,300),('U_first',300,1300),('V_first',1300,2300),('W_long',2300,3300)];pd=[]
for name,a,b in phases:
 cc=met(y[a:b],cp[a:b]);dd=met(y[a:b],dp[a:b]);pd.append({'phase':name,'C':cc,'D':dd,'delta':dd['ap']-cc['ap']})
margin=t['c_detection_logits'][...,1]-t['c_detection_logits'][...,0];delta=d['delta']; corr={'delta_margin_pearson':float(np.corrcoef(margin.ravel(),delta.ravel())[0,1]),'mean_within_time_host_std':float(delta.std(axis=1).mean()),'std_time_mean':float(delta.mean(axis=1).std()),'min':float(delta.min()),'max':float(delta.max())}
res={'source_artifact_id':11041705829,'checks':{k:bool(v) for k,v in checks.items()},'all_pass':bool(all(checks.values())),'updates':len(db),'last_update':db[-1]['at_interval'],'full':full,'windows':wins,'first_phases':pd,'correction_descriptive':corr,'array_shapes':{k:list(v.shape) for k,v in c.items()}}
(root/'independent_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(res,ensure_ascii=False,indent=2))
