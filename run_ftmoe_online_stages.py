"""Serial, gated execution of the approved online pilot, development and confirmation."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts/ftmoe_online'


def run_child(script, arguments):
    command=[sys.executable,'-u',str(ROOT/script)]+list(map(str,arguments))
    print('RUN '+json.dumps(command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def stream(seed,steps):
    folder=ART/f'streams/seed{seed}_steps{steps}'
    if not (folder/'manifest.json').exists():
        if folder.exists():raise RuntimeError('Incomplete stream must be reviewed before retry: '+str(folder))
        run_child('prepare_ftmoe_online.py',['--seed',seed,'--steps',steps])
    return folder


def model_run(stage,method,seed,replay_seed,steps,learning_rate):
    output=ART/f'{stage}/runs/{method}_model{seed}_replay{replay_seed}_lr{learning_rate:g}'
    if (output/'summary.json').exists():return
    args=['--method',method,'--model-seed',seed,'--stream',stream(replay_seed,steps),
          '--learning-rate',learning_rate,'--output',output]
    if output.exists():
        if not (output/'resume.pt').exists():raise RuntimeError('Incomplete run needs review: '+str(output))
        args+=['--resume']
    run_child('run_ftmoe_online.py',args)


def report(stage):
    output=ART/stage/'analysis'
    if not (output/'report.json').exists():
        run_child('analyze_ftmoe_online.py',['--runs',ART/stage/'runs','--output',output,'--stage',stage])
    return json.loads((output/'report.json').read_text(encoding='utf8'))


def pipeline(through):
    pilot=report('pilot') if (ART/'pilot/analysis/report.json').exists() else None
    if pilot is None:
        for method in 'ABCD':model_run('pilot',method,1,301,300,0. if method=='A' else 3e-5)
        pilot=report('pilot')
    if not pilot['ready_for_next_stage']:
        return {'status':'awaiting_user_discussion','stage':'pilot','problems':pilot['problems']}
    if through=='pilot':return {'status':'pilot_complete','stage':'pilot'}
    for method in 'ABCD':
        for seed in (1,2):
            for replay_seed in (301,302):
                for lr in ((0.,) if method=='A' else (1e-5,3e-5,1e-4)):
                    model_run('development',method,seed,replay_seed,1000,lr)
    development=report('development')
    if not development['ready_for_next_stage']:
        return {'status':'awaiting_user_discussion','stage':'development','problems':development['problems']}
    if through=='development':return {'status':'development_complete','stage':'development'}
    frozen={'selection':development['selection'],'model_seeds':[1,2,6,17,42],
            'replay_seeds':[401,402,403],'steps':2000,'methods':list('ABCD')}
    path=ART/'confirmation_frozen.json'
    if path.exists():
        if json.loads(path.read_text())!=frozen:raise RuntimeError('Frozen confirmation configuration changed')
    else:path.write_text(json.dumps(frozen,indent=2)+'\n')
    for method in 'ABCD':
        lr=0. if method=='A' else frozen['selection'][method]['learning_rate']
        for seed in frozen['model_seeds']:
            for replay_seed in frozen['replay_seeds']:model_run('confirmation',method,seed,replay_seed,2000,lr)
    confirmation=report('confirmation')
    return {'status':'confirmation_complete','stage':'confirmation','accepted':confirmation['accepted'],
            'problems':confirmation['problems']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',choices=['pilot','development','confirmation'],default='confirmation')
    args=parser.parse_args();ART.mkdir(parents=True,exist_ok=True)
    if not (ART/'protocol_015_online.json').is_file():raise FileNotFoundError('Approved registration missing')
    lock=ART/'pipeline.lock'
    if lock.exists():
        previous=json.loads(lock.read_text())
        if psutil.pid_exists(previous['pid']) and abs(psutil.Process(previous['pid']).create_time()-previous['created'])<.1:
            raise RuntimeError('Online pipeline is already running')
        lock.unlink()
    with lock.open('x') as handle:json.dump({'pid':os.getpid(),'created':psutil.Process().create_time()},handle)
    try:
        # A second experimental process may consume resources or change simulator inputs.
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid==os.getpid() or 'python' not in (process.info['name'] or '').lower():continue
            command=process.info['cmdline'] or []
            scripts=[Path(x).name.lower() for x in command[1:] if x.lower().endswith('.py')]
            if any(x.startswith(('train_ftmoe','prepare_ftmoe','run_ftmoe_online','run_qos','dump_replay')) or x=='main.py' for x in scripts):
                raise RuntimeError('Another experimental Python process is active: PID '+str(process.pid))
        result=pipeline(args.through)
    except Exception as exc:
        result={'status':'awaiting_user_discussion','stage':'runtime','error':str(exc),
                'next_action':'Report the failure; do not modify simulator behavior or experiment conditions'}
        raise
    finally:
        if 'result' in globals():
            result['updated_local']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
            (ART/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            print(json.dumps(result,ensure_ascii=False),flush=True)
        lock.unlink(missing_ok=True)
