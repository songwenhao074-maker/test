"""Collect registered capacity or Google trace scenarios using unchanged simulator core."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def configure(configuration):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='3'
    for name in ('RAM_SCALE','DISK_SCALE','CPU_CAP_SCALE','DISK_CAP_SCALE'):os.environ[name]='1.0'
    os.environ['CPU_CAP_SCALE']='0.8'
    os.environ['DISK_CAP_SCALE']=str(.25*configuration['disk_capacity_multiplier'])
    for name in ('FIXED_SCHEDULE_PATH','QOS_OVERLOAD_OUT','ONLINE_TUNE','ONLINE_LABEL_MODE'):os.environ.pop(name,None)
    import torch
    import psutil
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def guard():
    import psutil
    import shutil
    available=psutil.virtual_memory().available/2**30
    disk=shutil.disk_usage(ROOT).free/2**30
    if available < 4.5:raise RuntimeError('RAM guard below 4.5 GiB (available %.3f GiB)'%available)
    if disk < 20:raise RuntimeError('Disk guard below 20 GiB (available %.3f GiB)'%disk)


def collect(seed, steps, output, configuration_path):
    configuration=json.loads(configuration_path.read_text(encoding='utf8'))
    if seed not in (301,302):raise ValueError('Unregistered scenario seed')
    if steps not in (300,2000):raise ValueError('Unregistered scenario horizon')
    for key in ('ram_capacity_multiplier','disk_capacity_multiplier'):
        if not 0 < configuration[key] <= 1:raise ValueError('Invalid reduced capacity')
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    console=sys.stdout
    try:
        configure(configuration);guard();os.chdir(ROOT)
        import numpy as np
        import torch
        import psutil
        bitbrain=ROOT/'simulator/workload/datasets/bitbrain/rnd'
        if configuration['workload']=='adapted_bwgd2' and not all((bitbrain/f'{i}.csv').is_file() for i in range(1,500)):
            raise FileNotFoundError('Local Bitbrain dataset incomplete; downloading is not authorized')
        scheduler_weight=ROOT/'scheduler/BaGTI/checkpoints/energy_latency_16_Trained.ckpt'
        # Resolve the real scheduler path through its source constant, before construction.
        with (output/'generation.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            from simulator.Simulator import Simulator
            from simulator.environment.RPiEdge import RPiEdge
            from simulator.workload.BitbrainWorkloadShared import SharedAdaptedBWGD2 as AdaptedBWGD2
            from scheduler.GOBI import GOBIScheduler
            from recovery.Recovery import Recovery
            from stats.Stats import Stats
            from src.constants import MODEL_SAVE_PATH
            candidates=[ROOT/str(MODEL_SAVE_PATH)/'energy_latency_16_Trained.ckpt',
                        ROOT/'scheduler/BaGTI'/str(MODEL_SAVE_PATH)/'energy_latency_16_Trained.ckpt']
            scheduler_weight=next((p.resolve() for p in candidates if p.is_file()),None)
            if scheduler_weight is None:raise FileNotFoundError('GOBI trained checkpoint missing')
            started=time.perf_counter()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            dc=RPiEdge(16)
            for spec in dc.types.values():spec['RAMSize']*=configuration['ram_capacity_multiplier']
            if configuration['workload']=='adapted_bwgd2':
                workload=AdaptedBWGD2(1,1.5,seed)
            elif configuration['workload']=='google2011':
                from simulator.workload.Google2011Workload import Google2011Workload
                workload=Google2011Workload(ROOT/configuration['trace_path'])
            else:raise ValueError('Unknown registered workload')
            scheduler=GOBIScheduler('energy_latency_16')
            recovery=Recovery();stats=Stats(workload,dc,scheduler);stats.feats_per_host=7
            env=Simulator(1000,10000,scheduler,recovery,stats,16,300,dc.generateHosts())
            initial=workload.generateNewContainers(env.interval)
            deployed=env.addContainersInit(initial)
            decision=scheduler.placement(deployed)
            migrations=env.allocateInit(decision)
            workload.updateDeployedContainers(env.getCreationIDs(migrations,deployed))
            stats.saveStats(deployed,migrations,[],deployed,decision,0)
            capacities=np.array([[h.ipsCap,h.ramCap.size,h.diskCap.size] for h in env.hostlist],dtype=np.float64)
            count=steps+1
            host=np.zeros((count,16,7),np.float32);demands=np.zeros_like(host);schedules=np.zeros((count,16,16),np.float32)
            totals=np.zeros((count,16,3),np.float64);labels=np.zeros((count,16),np.int64)
            before=np.full((count,16),-1,np.int64);after=before.copy();creation=before.copy();after_creation=before.copy()
            intervals=np.zeros(count,np.int64);events=[]
            for t in range(count):
                guard()
                new=workload.generateNewContainers(env.interval)
                deployed,destroyed=env.addContainers(new)
                intervals[t]=env.interval
                for slot,c in enumerate(env.containerlist):
                    if c is None:continue
                    if c.id != slot or not c.active:raise AssertionError('Invalid live slot identity')
                    ram=c.getRAM();disk=c.getDisk()
                    values=np.array([c.getBaseIPS(),*ram,*disk],dtype=np.float64)
                    if not np.isfinite(values).all() or (values<0).any():raise AssertionError('Invalid pre-action demand')
                    demands[t,slot]=values;creation[t,slot]=c.creationID;before[t,slot]=c.getHostID()
                    if c.getHostID()>=0:host[t,c.getHostID()]+=values
                selected=scheduler.selection()
                decision=scheduler.filter_placement(scheduler.placement(selected+deployed))
                schedules[t]=np.asarray(scheduler.result_cache)
                if not np.isfinite(schedules[t]).all():raise AssertionError('Nonfinite proposed schedule')
                np.testing.assert_allclose(schedules[t].sum(-1),1,atol=1e-5)
                if (schedules[t]<-1e-6).any():raise AssertionError('Negative proposed schedule')
                migrations=env.simulationStep(recovery.run_model(stats.time_series,decision))
                workload.updateDeployedContainers(env.getCreationIDs(migrations,deployed))
                for slot,c in enumerate(env.containerlist):
                    if c is None:continue
                    hid=c.getHostID()
                    if not 0<=hid<16:raise AssertionError('Unallocated container survived simulationStep')
                    if c.creationID!=creation[t,slot]:raise AssertionError('Slot identity changed inside simulationStep')
                    after[t,slot]=hid;after_creation[t,slot]=c.creationID
                    totals[t,hid]+=[c.getBaseIPS(),c.getRAM()[0],c.getDisk()[0]]
                ratio=totals[t]/capacities
                labels[t]=np.where((ratio>1).any(-1),ratio.argmax(-1)+1,0)
                if not np.isfinite(totals[t]).all() or (totals[t]<0).any():raise AssertionError('Invalid post-action totals')
                events.append({'step':t+1,'simulator_interval':int(env.interval),
                    'proposed_decision':[[int(x),int(y)] for x,y in decision],
                    'actual_migrations':[[int(x),int(y)] for x,y in migrations]})
                stats.saveStats(deployed,migrations,destroyed,selected,decision,0)
                # The collector exports its own durable per-step records. Native all-container
                # snapshots redundantly retain every finished task at every interval.
                # Keep the most recent snapshot; this list is not an input to this simulation.
                stats.allcontainerinfo[:]=stats.allcontainerinfo[-1:]
                if (t+1)%50==0:print(json.dumps({'collected':t+1,'total':count,'elapsed_seconds':time.perf_counter()-started}),file=console,flush=True)
            np.savez_compressed(output/'stream.npz',host_features=host,demands=demands,schedules=schedules,
                raw_labels=labels,capacities=capacities,post_totals=totals,before_placement=before,after_placement=after,
                creation_ids=creation,after_creation_ids=after_creation,simulator_intervals=intervals)
            (output/'events.json').write_text(json.dumps(events,indent=2)+'\n',encoding='utf8')
            # Capture implementation, scheduler training inputs, and actual local workload sources.
            sources=[ROOT/'prepare_ftmoe_scenario.py', configuration_path.resolve(), ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/protocol.json', ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json', scheduler_weight]
            for base in ('simulator','scheduler','metrics','stats','utils'):
                sources.extend(p for p in (ROOT/base).rglob('*.py') if '__pycache__' not in p.parts)
            sources.extend((ROOT/'scheduler/BaGTI').rglob('*.npy'))
            if configuration['workload']=='adapted_bwgd2':sources.extend(sorted(bitbrain.glob('*.csv')))
            else:
                sources.extend([ROOT/configuration['trace_path'],(ROOT/configuration['trace_path']).with_suffix('.json')])
            source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))}
            manifest={'schema_version':1,'seed':seed,'steps':steps,'guard_steps':1,'workload':configuration['workload'],'scenario_configuration':configuration,'scenario_configuration_sha256':sha(configuration_path),
                'interval_seconds':300,'hosts':16,'containers':16,'arrival_mean':1,'arrival_sigma':1.5,
                'capacity_scales':{'CPU_CAP_SCALE':.8,'DISK_CAP_SCALE':.25*configuration['disk_capacity_multiplier'],'RAM_CAP_SCALE':configuration['ram_capacity_multiplier']},'artificial_drift':False,
                'synthetic_dynamic_disk':configuration['workload']=='adapted_bwgd2','demand_adapter':configuration.get('demand_adapter','unchanged_protocol016'),
                'protocol_sha256':sha(ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/protocol.json'),
                'disk_law_sha256':sha(ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json'),
                'recovery':'no_op','scheduler':'GOBI_energy_latency_16','capacities':capacities.tolist(),
                'selected_vm_indices':workload.possible_indices,'source_sha256':source_hashes,
                'stream_sha256':sha(output/'stream.npz'),'events_sha256':sha(output/'events.json'),
                'raw_class_counts_scored':np.bincount(labels[:steps].ravel(),minlength=4).tolist(),
                'elapsed_seconds':time.perf_counter()-started,'rss_gib':psutil.Process().memory_info().rss/2**30,
                'ordering':'new tasks, addContainers, pre-action features, proposed GOBI schedule, simulationStep, actual physical labels',
                'replay_protocol':'record once; consumers predict before exposing each step label; tolerance matures one interval later'}
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
        print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','selected_vm_indices')},indent=2),flush=True)
    except Exception as exc:
        failure={'seed':seed,'steps':steps,'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),
                 'completed_steps':len(locals().get('events',[])),
                 'simulator_behavior_modified':True,'authorized_configuration':str(configuration_path),'next_action':'Report to user before simulator changes'}
        (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf8')
        print(json.dumps(failure),file=console,flush=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,required=True)
    args=parser.parse_args()
    collect(args.seed,args.steps,args.output,args.configuration)
