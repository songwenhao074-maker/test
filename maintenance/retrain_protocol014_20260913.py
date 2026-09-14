"""Rebuild the accepted 30-model ablation with historical, hash-verified sources."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'artifacts/ftmoe_protocol014_retraining_20260913'
RECOVERY = BASE / 'recovery'
WORK = BASE / 'workspace'
PYTHON = Path(r'D:\Anaconda\envs\dynmoe\python.exe')
SEEDS = [1, 2, 6, 17, 42]
METRICS = ['f1', 'diagnosis_hr_at_100pct', 'diagnosis_ndcg_at_100pct']
SOURCE_NAMES = [
    'train_ftmoe_end_to_end.py', 'train_ftmoe_ablation_existing.py',
    'recovery/PreGANSrc/src/ftmoe_ablation.py', 'recovery/PreGANSrc/src/ftmoe_end_to_end.py',
    'recovery/PreGANSrc/src/ftmoe_context.py', 'recovery/PreGANSrc/src/ftmoe_expert_regularization.py',
    'recovery/PreGANSrc/src/ftmoe_fusion_controls.py', 'recovery/PreGANSrc/src/ftmoe_source_fusion.py',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare():
    recovered = load(RECOVERY / 'recovered_files.json')
    original = {entry['path']: entry for entry in recovered}
    data_prefix = 'artifacts/ftmoe_end_to_end/data/protocol_004_physical/'
    required = SOURCE_NAMES + [data_prefix + name for name in (
        'manifest.json', 'time_series.npy', 'container_demand_series.npy', 'schedule_series.npy', 'labels.npy')]
    for name in required:
        if name not in original:
            raise RuntimeError('Historical prerequisite missing: ' + name)
    if not WORK.exists():
        WORK.mkdir(parents=True)
        for entry in recovered:
            # Keep the recovered historical tree separate from newly trained run names.
            source = RECOVERY / '_recovery_cache' / entry['sha256']
            destination = WORK / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha(destination) != entry['sha256']:
                raise RuntimeError('Recovered file hash mismatch: ' + entry['path'])
        shutil.copytree(RECOVERY / 'historical_documents', WORK / 'historical_documents')
    for name in required:
        if sha(WORK / name) != original[name]['sha256']:
            raise RuntimeError('Historical prerequisite changed: ' + name)
    data = load(WORK / data_prefix / 'manifest.json')
    assert data['seeds'] == [42, 1, 6, 17, 23, 31, 101, 102]
    assert data['train_blocks'] == [0, 1, 2, 3, 4] and data['validation_blocks'] == [5, 6, 7]
    registration = {
        'purpose': 'User-authorized retraining to replace the deleted 2026-09-05 accepted baseline archive',
        'original_snapshot_recovered': False, 'training_is_new': True,
        'model_seeds': SEEDS, 'variants': ['v0', 'v1', 'v2', 'v3', 'v4', 'v4_gated'],
        'expected_models': 30, 'epochs': 30, 'learning_rate': 0.0003,
        'expert_dropout': 0.0, 'moe_context': 'none', 'independent_initialization': True,
        'all_parameters_trainable': True, 'batch_size': 32, 'optimizer': 'AdamW',
        'weight_decay': 0.0001, 'warmup_epochs': 5, 'cosine_eta_min': 0.00001,
        'gradient_clip_norm': 1, 'eagate_temperature': [1.0, 0.2],
        'loss': {'detection_ce': 0.7, 'resource_ce': 0.3, 'ranking': 0.5,
                 'router_balance': 0.01, 'detection_class_weights': [0.6, 2.0]},
        'B_best': 'Per model seed, maximize (F1+HR+NDCG)/3 over three equally weighted development replays; earliest tie; threshold 0.5',
        'A_last': 'epoch 30', 'threads': 3, 'interop_threads': 1, 'dataloader_workers': 0,
        'priority': 'BelowNormal', 'available_ram_guard_gib': 4.5, 'disk_guard_gib': 20,
        'train_replay_seeds': [42, 1, 6, 17, 23], 'development_replay_seeds': [31, 101, 102],
        'reserved_tests_accessed': False, 'qos_started': False,
        'source_and_data_sha256': {name: original[name]['sha256'] for name in required},
        'retain': ['every epoch checkpoint', 'best.pt', 'last.pt', 'resume.pt with optimizer and RNG', 'metrics', 'source snapshots'],
        'scope': 'The accepted protocol-014 configuration only; earlier unrelated experiments and deleted plots are not retrained or fabricated.',
    }
    destination = BASE / 'retraining_registration.json'
    if destination.exists():
        if load(destination) != registration:
            raise RuntimeError('Retraining registration changed')
    else:
        write_json(destination, registration)
    write_json(BASE / 'environment.json', {
        'python': sys.version, 'executable': str(PYTHON),
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'scikit-learn', 'psutil']},
    })
    shutil.copyfile(BASE / 'environment.json', WORK / 'RETRAINING_ENVIRONMENT.json')
    shutil.copyfile(RECOVERY / 'recovery_report.json', WORK / 'ORIGINAL_ARCHIVE_RECOVERY_REPORT.json')
    shutil.copyfile(RECOVERY / 'missing_files.json', WORK / 'MISSING_ORIGINAL_FILES.json')
    shutil.copyfile(Path(__file__), WORK / 'retraining_controller_reference.py')
    write_json(WORK / 'RETRAINING_REGISTRATION.json', registration)
    return registration


def jobs():
    result = []
    for fusion, variants in [('attention', ['v0', 'v1', 'v2', 'v3', 'v4']), ('gated', ['v4'])]:
        for variant in variants:
            for seed in SEEDS:
                run = 'retrained_014_' + fusion + '_' + variant + '_seed' + str(seed)
                result.append((fusion, variant, seed, run))
    return result


def run_models():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid == os.getpid():
            continue
        try:
            command = process.info['cmdline'] or []
            if 'python' in (process.info['name'] or '').lower() and any(
                str(ROOT).lower() in s.lower() or any(key in s.lower() for key in ['train_ftmoe', 'run_ftmoe', 'dump_replay', 'run_qos'])
                for s in command):
                raise RuntimeError('Another project Python process is running: ' + str(process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logs = BASE / 'logs'
    logs.mkdir(exist_ok=True)
    completed = 0
    for fusion, variant, seed, run in jobs():
        summary = WORK / 'artifacts/ftmoe_end_to_end/runs' / run / (variant + '_seed' + str(seed)) / 'summary.json'
        if summary.exists():
            completed += 1
            continue
        command = [str(PYTHON), '-X', 'utf8', '-u', str(WORK / 'train_ftmoe_end_to_end.py'),
                   '--run', run, '--data', 'artifacts/ftmoe_end_to_end/data/protocol_004_physical',
                   '--epochs', '30', '--learning-rate', '0.0003', '--moe-context', 'none',
                   '--expert-dropout', '0', '--fusion', fusion, '--seeds', str(seed),
                   '--variants', variant, '--save-all-epochs']
        write_json(BASE / 'status.json', {'state': 'training', 'completed_models': completed,
                    'total_models': 30, 'current_run': run, 'controller_pid': os.getpid(),
                    'command': command, 'updated_local': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        print('MODEL_START', completed + 1, '/30', run, flush=True)
        env = os.environ.copy()
        env.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
                    'OMP_NUM_THREADS': '3', 'MKL_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '3'})
        with (logs / (run + '.log')).open('a', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=str(WORK), env=env, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        if code != 0 or not summary.exists():
            raise RuntimeError('Training failed: ' + run + ', exit=' + str(code))
        completed += 1
        print('MODEL_COMPLETE', completed, '/30', run, flush=True)


def summarize():
    import math
    groups = {}
    records = []
    for fusion, variant, seed, run in jobs():
        directory = WORK / 'artifacts/ftmoe_end_to_end/runs' / run / (variant + '_seed' + str(seed))
        row = load(directory / 'summary.json')
        expected = [directory / 'checkpoints_by_epoch' / ('epoch%03d.pt' % epoch) for epoch in range(1, 31)]
        assert all(p.is_file() and p.stat().st_size > 0 for p in expected)
        assert all((directory / name).is_file() for name in ['best.pt', 'last.pt', 'resume.pt', 'epochs.csv'])
        assert row['all_parameters_trainable'] and row['configuration']['epochs'] == 30
        assert row['configuration']['learning_rate'] == 0.0003 and row['configuration']['expert_dropout'] == 0
        assert row['configuration']['fusion'] == fusion and row['seed'] == seed
        assert load(directory / 'progress.json')['epoch'] == 30
        name = variant if fusion == 'attention' else 'v4_gated'
        groups.setdefault(name, []).append(row)
        records.append({'arm': name, 'seed': seed, 'best_epoch': row['best_epoch'],
                        'summary': str((directory / 'summary.json').relative_to(WORK)),
                        'A_last': row['A_last']['mean'], 'B_best': row['B_best']['mean']})
    aggregate = {}
    for selection in ['A_last', 'B_best']:
        aggregate[selection] = {}
        for name, rows in groups.items():
            aggregate[selection][name] = {}
            for metric in METRICS:
                values = [r[selection]['mean'][metric] for r in rows]
                assert all(math.isfinite(v) for v in values)
                mean = sum(values) / 5
                aggregate[selection][name][metric] = {'mean': mean, 'std': math.sqrt(sum((v-mean)**2 for v in values)/4)}
    result = {'status': 'all_30_models_retrained', 'models': records, 'aggregate': aggregate,
              'historical_results_not_overwritten': True, 'reserved_tests_accessed': False,
              'epoch_checkpoint_count': 900, 'retained_resume_files': 30}
    write_json(WORK / 'RETRAINING_RESULTS.json', result)
    write_json(BASE / 'retraining_results.json', result)
    return result


def archive():
    shutil.copytree(BASE / 'logs', WORK / 'retraining_logs', dirs_exist_ok=True)
    target = BASE / 'snapshot.zip'
    if target.exists():
        raise RuntimeError('Snapshot already exists; refusing to overwrite it')
    files = sorted(p for p in WORK.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest = {'snapshot_type': 'retrained_protocol014_baseline', 'original_snapshot_restored': False,
                'models_retrained': 30, 'files': []}
    with zipfile.ZipFile(BASE / 'snapshot.zip.partial', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in files:
            digest = sha(p)
            relative = p.relative_to(WORK).as_posix()
            z.write(p, 'snapshot/' + relative)
            manifest['files'].append({'path': relative, 'bytes': p.stat().st_size, 'sha256': digest})
        z.writestr('manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (BASE / 'snapshot.zip.partial').rename(target)
    write_json(BASE / 'snapshot_manifest.json', manifest)
    with zipfile.ZipFile(target) as z:
        assert json.loads(z.read('manifest.json')) == manifest
        assert len(z.namelist()) == len(files) + 1
        for row in manifest['files']:
            h = hashlib.sha256()
            size = 0
            with z.open('snapshot/' + row['path']) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(chunk); size += len(chunk)
            assert h.hexdigest() == row['sha256'] and size == row['bytes'], row['path']
    verification = {'passed': True, 'files_verified': len(files), 'archive_bytes': target.stat().st_size,
                    'archive_sha256': sha(target), 'original_snapshot_restored': False, 'models_retrained': 30}
    write_json(BASE / 'snapshot_verification.json', verification)
    for p in [target, BASE / 'snapshot_manifest.json', BASE / 'snapshot_verification.json']:
        os.chmod(p, 0o444)
    write_json(BASE / 'status.json', {'state': 'complete', 'completed_models': 30, 'total_models': 30,
                'snapshot': str(target), 'verification': verification,
                'updated_local': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print('SNAPSHOT_COMPLETE', json.dumps(verification), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        print('PREPARED', str(WORK), flush=True)
        return
    try:
        run_models()
        summarize()
        archive()
    except Exception as exc:
        write_json(BASE / 'failure.json', {'error': str(exc), 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        raise


if __name__ == '__main__':
    main()
