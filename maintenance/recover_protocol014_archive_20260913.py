"""Recover only byte-identical baseline assets from surviving files and Git objects."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'artifacts/ftmoe_protocol014_retraining_20260913/recovery'
CACHE = OUT / '_recovery_cache'
REFERENCE = ROOT / 'maintenance/cleanup_20260905/protected_sha256.json'


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(ROOT))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite an existing recovery directory')
    CACHE.mkdir(parents=True)
    targets = json.loads(REFERENCE.read_text(encoding='utf-8'))
    wanted = set(targets.values())
    found = {}
    counts = collections.Counter()

    def consider(data, source):
        variants = [('unchanged', data)]
        if len(data) < 4 * 1024 * 1024 and b'\0' not in data:
            lf = data.replace(b'\r\n', b'\n')
            variants.extend([('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))])
        for transform, content in variants:
            sha = hashlib.sha256(content).hexdigest()
            if sha in wanted and sha not in found:
                (CACHE / sha).write_bytes(content)
                found[sha] = {'source': source, 'line_endings': transform, 'bytes': len(content)}

    print('SEARCH_CURRENT_FILES', flush=True)
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '.agents', '.codex', '.claude', '__pycache__', '.pytest_cache'}
                   and Path(base) / d != OUT and not (Path(base) / d).is_symlink()]
        for name in names:
            p = Path(base) / name
            if p.is_symlink():
                continue
            consider(p.read_bytes(), {'kind': 'surviving_file', 'path': p.relative_to(ROOT).as_posix()})
            counts['current_files_examined'] += 1
    print('CURRENT_HASHES_FOUND', len(found), flush=True)

    rows = git('cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype) %(objectsize)').decode().splitlines()
    objects = [r.split()[0] for r in rows if r.split()[1] == 'blob']
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for index, oid in enumerate(objects, 1):
        process.stdin.write((oid + '\n').encode())
        process.stdin.flush()
        header = process.stdout.readline().split()
        if header[1] != b'blob':
            raise RuntimeError('Unexpected Git object')
        size = int(header[2])
        data = process.stdout.read(size)
        if len(data) != size or process.stdout.read(1) != b'\n':
            raise RuntimeError('Truncated Git object')
        if hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + data).hexdigest() != oid:
            raise RuntimeError('Git object hash mismatch')
        consider(data, {'kind': 'git_blob', 'object': oid})
        # Some historical documentation was itself preserved as a small ZIP.
        if data[:4] == b'PK\x03\x04' and len(data) < 2 * 1024 * 1024:
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as nested:
                    for info in nested.infolist():
                        if not info.is_dir() and info.file_size < 8 * 1024 * 1024:
                            consider(nested.read(info), {'kind': 'git_zip_member', 'object': oid, 'member': info.filename})
            except zipfile.BadZipFile:
                pass
        counts['git_blobs_examined'] += 1
        if index % 400 == 0:
            print('GIT_PROGRESS', index, 'HASHES_FOUND', len(found), flush=True)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Git object reader failed')

    upstream = Path(r'F:\PreGANPlus-master.zip')
    if upstream.exists():
        with zipfile.ZipFile(upstream) as source:
            for info in source.infolist():
                if not info.is_dir():
                    consider(source.read(info), {'kind': 'upstream_zip', 'archive': str(upstream), 'member': info.filename})
                    counts['upstream_zip_members_examined'] += 1

    files, missing = [], []
    for path, sha in sorted(targets.items()):
        if sha in found:
            files.append({'path': path, 'sha256': sha, **found[sha]})
        else:
            missing.append({'path': path, 'sha256': sha})

    comparison_path = 'artifacts/ftmoe_end_to_end/comparison_014_complete.json'
    comparison = json.loads((CACHE / targets[comparison_path]).read_text(encoding='utf-8'))
    required = []
    for absolute, sha in comparison['source_sha256'].items():
        path = Path(absolute).relative_to(ROOT).as_posix()
        required.append({'path': path, 'sha256': sha, 'recovered': sha in found})
    selected = []
    for entry in files:
        if '/runs/' in entry['path'] and entry['path'].endswith('.pt'):
            selected.append(entry['path'])
    summary = {
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'status': 'partial_reconstruction', 'complete_original_archive_restored': False,
        'original_archive': {'files': 9760, 'zip_bytes': 7972544300,
                             'sha256': '3ea269de62e0912466b8057e686fb74bce5741b9d319d703d6e9c32ea4857405',
                             'source': 'successful archive receipt in this task on 2026-09-05'},
        'reference': {'path': REFERENCE.relative_to(ROOT).as_posix(),
                      'sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                      'scope': 'pre-archive protected assets; not the missing original full archive manifest'},
        'reference_file_count': len(targets), 'recovered_reference_files': len(files),
        'missing_reference_files': len(missing), 'recovered_bytes': sum(e['bytes'] for e in files),
        'comparison_014_sources': {'total': len(required), 'recovered': sum(e['recovered'] for e in required)},
        'missing_by_group': dict(collections.Counter(e['path'].split('/')[0] for e in missing)),
        'recovered_checkpoints': selected, 'search_counts': dict(counts),
        'limitations': [
            'The original ZIP and full manifest are absent; this is not a complete rollback snapshot.',
            'Missing-file list covers only the surviving 7361-file protected-assets manifest.',
            'All recovered snapshot files match the historical SHA-256; line-ending variants are accepted only on exact hash match.',
            'Additional historical documents are preserved as Git evidence, not claimed to be the exact 2026-09-05 working tree.',
            'No training, test replay, model evaluation, or changes to the protocol-023 implementation were performed.',
        ],
    }
    write_json(OUT / 'recovered_files.json', files)
    write_json(OUT / 'missing_files.json', missing)
    write_json(OUT / 'comparison_014_source_recovery.json', required)
    write_json(OUT / 'recovery_report.json', summary)
    evidence = OUT / 'historical_documents'
    docs = ['docs/BASELINE_ACCEPTED_20260905.md', 'FTMOE_END_TO_END_ABLATION_PLAN.md',
            'FTMOE_END_TO_END_EXPERIMENT_LOG.md', 'FTMOE_HISTORY_REUSE_AUDIT.md',
            'FTMOE_HISTORICAL_CONFIRMATION.md', 'FTMOE_HISTORICAL_SELECTION_DIAGNOSTIC.md',
            'docs/MODEL_SELECTION_REVIEW.md', 'docs/MAINTENANCE_20260905.md',
            'docs/archive/README.md', 'docs/archive/2026-09-05_pre_cleanup.zip']
    document_sources = []
    for path in docs:
        content = git('show', '43650a9:' + path)
        destination = evidence / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        document_sources.append({'path': path, 'git_commit': git('rev-parse', '43650a9').decode().strip(),
                                 'sha256': hashlib.sha256(content).hexdigest()})
    write_json(OUT / 'historical_document_sources.json', document_sources)
    (OUT / 'protected_sha256_original.json').write_bytes(REFERENCE.read_bytes())
    print('RECOVERY_INVENTORY_COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
