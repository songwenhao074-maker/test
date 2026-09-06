"""One-off, auditable documentation cleanup; deletion is performed by PowerShell."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'maintenance/cleanup_20260905'
ARCHIVE = ROOT/'docs/archive/2026-09-05_pre_cleanup.zip'
OLD_DOCS = ['EXPERIMENT_001_V0_V1_ROUTER.md','EXPERIMENT_002_V0_V1_EXISTING_DATA.md',
    'FTMOE_ABLATION_EXECUTION_PLAN.md','FTMOE_ABLATION_EXECUTION_REPORT.md',
    'MEMORY_INDEX.md','MEMORY_v9_BOOST.md','notes_v9_experiment.md','PROJECT_CONTEXT.md','PROJECT_CONTEXT_full.md']
REWRITE_DOCS = ['PROJECT_CONTEXT_LATEST.md','FTMOE_END_TO_END_ABLATION_PLAN.md',
    'FTMOE_END_TO_END_EXPERIMENT_LOG.md','README.md']
LOOSE = ['bench_v0_one.log','COSCO.log','synth_v3b2_train.log','synth2_v0.log','synth2_v2c2.log',
    '_candidate_final_sched.npy','_candidate_final_test.npy','paper_text.txt','sim_timing.txt']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def files():
    for base,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:] = [d for d in dirs if d not in ['.git','.agents','.codex','.claude']
                   and not (os.lstat(Path(base)/d).st_file_attributes & 1024)]
        for name in names:
            p = Path(base)/name
            if not (os.lstat(p).st_file_attributes & 1024): yield p


def entry(path, reason):
    assert within(path.resolve(),ROOT)
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
            'sha256':sha(path),'reason':reason}


def prepare():
    import psutil
    import torch
    torch.set_num_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert not (OUT/'plan.json').exists()
    if ARCHIVE.exists():
        with zipfile.ZipFile(ARCHIVE) as existing: assert not existing.namelist()
    active = [p.info['cmdline'] for p in psutil.process_iter(['cmdline']) if p.pid!=os.getpid()
              and any(Path(a).name in ['train_ftmoe_end_to_end.py','dump_replay.py'] for a in (p.info['cmdline'] or []))]
    assert not active, active
    OUT.mkdir(parents=True,exist_ok=True); ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    archived = []; delete = []
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in OLD_DOCS+REWRITE_DOCS+LOOSE:
            p = ROOT/name
            e = entry(p,'original preserved in verified archive')
            z.write(p,name); archived.append(e)
            if name not in REWRITE_DOCS: delete.append(e)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for e in archived: assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
    skipped = []
    for p in sorted((ROOT/'artifacts/ftmoe_end_to_end/runs').glob('*/*/resume.pt')):
        d = p.parent; summary = d/'summary.json'
        if not summary.exists(): skipped.append(str(p.relative_to(ROOT))); continue
        s = json.loads(summary.read_text()); config = s['configuration']; epochs = config['epochs']
        if config.get('stop_after') is not None or not config.get('save_all_epochs'):
            skipped.append(str(p.relative_to(ROOT))); continue
        if not all((d/'checkpoints_by_epoch'/f'epoch{e:03d}.pt').exists() for e in range(1,epochs+1)):
            skipped.append(str(p.relative_to(ROOT))); continue
        history = list(csv.DictReader((d/'epochs.csv').open(encoding='utf8')))
        assert len(history)==epochs and int(history[-1]['epoch'])==epochs
        assert (d/'best.pt').exists() and (d/'last.pt').exists() and s['all_parameters_trainable']
        resume = torch.load(p,map_location='cpu',weights_only=False)
        last = torch.load(d/'last.pt',map_location='cpu',weights_only=False)
        assert resume['epoch']==last['epoch']==epochs
        assert resume['variant']==last['variant']==s['variant'] and resume['seed']==last['seed']==s['seed']
        assert resume['validation']==last['validation']==s['A_last']
        assert resume['model'].keys()==last['model'].keys()
        assert all(torch.equal(t,last['model'][k]) for k,t in resume['model'].items())
        delete.append(entry(p,'completed training recovery state; last model verified identical; every epoch, best/last, configuration and metrics retained; optimizer/RNG state discarded'))
        del resume,last
    cache_dirs = set()
    all_files = list(files())
    for p in all_files:
        if p.suffix=='.pyc' and '__pycache__' in p.parts:
            delete.append(entry(p,'regenerable Python bytecode cache')); cache_dirs.add(p.parent)
    names = {e['path'] for e in delete}
    protected = {}
    suffixes = {'.pt','.ckpt','.npy','.npz','.json','.csv','.py','.sha256'}
    for p in all_files:
        rel = p.relative_to(ROOT).as_posix()
        if rel in names or within(p,OUT): continue
        if p.suffix in suffixes or p.name=='FTMOE_ABLATION_FINAL_REPORT.md': protected[rel] = sha(p)
    write(OUT/'protected_sha256.json',protected)
    write(OUT/'plan.json',{'root':str(ROOT),'archive':str(ARCHIVE),'archive_sha256':sha(ARCHIVE),
        'archived_originals':archived,'delete_files':delete,'preserved_incomplete_or_partial_resume':skipped,
        'empty_directories':[str(p.relative_to(ROOT)) for p in sorted(cache_dirs,key=lambda p:-len(str(p)))]+['torchinductor_Lenovo'],
        'protected_files':len(protected),'protected_manifest_sha256':sha(OUT/'protected_sha256.json'),
        'root_markdown_before':{'files':len(list(ROOT.glob('*.md'))),'bytes':sum(p.stat().st_size for p in ROOT.glob('*.md'))}})
    print(json.dumps({'delete_files':len(delete),'bytes':sum(e['bytes'] for e in delete),
        'completed_resume_files':sum(e['path'].endswith('/resume.pt') for e in delete),
        'archived_originals':len(archived),'protected_files':len(protected),'retained_resume':skipped},indent=2))


def verify():
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf8'))
    assert sha(ARCHIVE)==plan['archive_sha256']
    assert sha(OUT/'protected_sha256.json')==plan['protected_manifest_sha256']
    for e in plan['delete_files']: assert not (ROOT/e['path']).exists(),e['path']
    protected = json.loads((OUT/'protected_sha256.json').read_text())
    for name,digest in protected.items(): assert (ROOT/name).exists() and sha(ROOT/name)==digest,name
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for e in plan['archived_originals']: assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
    result = {'deleted_files':len(plan['delete_files']),'deleted_file_bytes':sum(e['bytes'] for e in plan['delete_files']),
        'archive_bytes':ARCHIVE.stat().st_size,'protected_files_verified':len(protected),
        'all_retained_code_checkpoints_data_metrics_unchanged':True,
        'root_markdown_before':plan['root_markdown_before'],
        'root_markdown_after':{'files':len(list(ROOT.glob('*.md'))),'bytes':sum(p.stat().st_size for p in ROOT.glob('*.md'))},
        'scope':'Maintenance only; no training, test evaluation, model-selection change, or QoS run.'}
    write(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify'])
    {'prepare':prepare,'verify':verify}[parser.parse_args().action]()
