"""Explicit Protocol-027 data identity and complete, reusable snapshot checks."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
REVISION_PATH = ROOT / 'artifacts/ftmoe_online/protocol_027/data_revision_002.json'
REGISTRATION_PATH = ROOT / 'artifacts/ftmoe_online/protocol_027/registration.json'
REVISION = json.loads(REVISION_PATH.read_text(encoding='utf8'))
REVISION_ID = REVISION['revision_id']
EXPECTED_STREAM_SHA = REVISION['expected_stream_sha256']
EXPECTED_FINAL_CHUNK_SHA = REVISION['expected_final_chunk_sha256']
BASE_FILES = ('stream.npz', 'common_observable_features.npz', 'manifest.json',
              'events.json', 'data_audit.json', 'data_revision_snapshot.json',
              'protocol027_revision.json', 'protocol027_registration.json',
              'data_eligibility.json', 'assembly_verification.json')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def required_files(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    chunks = manifest['chunk_manifest']
    end = 0
    paths = list(BASE_FILES)
    for item in chunks:
        if item['start'] != end or item['end'] <= end:
            raise ValueError('noncontiguous frozen chunks')
        name = item['file']
        if Path(name).name != name:
            raise ValueError('invalid chunk filename')
        path = root / 'chunks' / name
        if sha(path) != item['sha256']:
            raise ValueError('chunk digest mismatch: ' + name)
        end = item['end']
        paths.append('chunks/' + name)
    if end != 5521 or len(chunks) != 28 or chunks[-1]['sha256'] != EXPECTED_FINAL_CHUNK_SHA:
        raise ValueError('incomplete or wrong revision chunks')
    if sha(root / 'events.json') != REVISION['events_sidecar']['uncompressed_sha256']:
        raise ValueError('events do not match the registered sidecar')
    if manifest['stream_sha256'] != EXPECTED_STREAM_SHA or sha(root / 'stream.npz') != EXPECTED_STREAM_SHA:
        raise ValueError('stream does not match the separately registered revision')
    return paths


def verify_frozen(root):
    root = Path(root)
    frozen = json.loads((root / 'frozen_data_manifest.json').read_text(encoding='utf8'))
    if frozen.get('data_revision') != REVISION_ID:
        raise ValueError('wrong frozen data revision')
    if frozen.get('revision_registration_sha256') != sha(REVISION_PATH):
        raise ValueError('data revision registration changed')
    paths = required_files(root)
    if set(frozen['files']) != set(paths):
        raise ValueError('frozen file inventory is incomplete or changed')
    for name in paths:
        p = root / name
        if sha(p) != frozen['files'][name]['sha256'] or p.stat().st_size != frozen['files'][name]['bytes']:
            raise ValueError('frozen file changed: ' + name)
    if sha(root / 'protocol027_registration.json') != sha(REGISTRATION_PATH):
        raise ValueError('experiment registration changed after freezing')
    if sha(root / 'protocol027_revision.json') != sha(REVISION_PATH):
        raise ValueError('revision snapshot does not match registration')
    eligibility = json.loads((root / 'data_eligibility.json').read_text(encoding='utf8'))
    if eligibility.get('protocol027_data_eligible') is not True or eligibility.get('data_revision') != REVISION_ID:
        raise ValueError('frozen snapshot lacks eligible revision audit')
    if eligibility.get('stream_sha256') != EXPECTED_STREAM_SHA:
        raise ValueError('frozen eligibility belongs to different data')
    return frozen


def freeze(root, eligibility_path, export_dir=None):
    root = Path(root)
    if (root / 'frozen_data_manifest.json').exists():
        frozen = verify_frozen(root)
    else:
        eligibility = json.loads(Path(eligibility_path).read_text(encoding='utf8'))
        if eligibility.get('protocol027_data_eligible') is not True or eligibility.get('data_revision') != REVISION_ID:
            raise ValueError('successful current-revision eligibility required before freezing')
        shutil.copyfile(eligibility_path, root / 'data_eligibility.json')
        shutil.copyfile(REVISION_PATH, root / 'protocol027_revision.json')
        shutil.copyfile(REGISTRATION_PATH, root / 'protocol027_registration.json')
        files = {name: {'sha256': sha(root/name), 'bytes': (root/name).stat().st_size}
                 for name in required_files(root)}
        frozen = {'protocol': '027', 'data_revision': REVISION_ID,
                  'revision_registration_sha256': sha(REVISION_PATH),
                  'stream_sha256': EXPECTED_STREAM_SHA, 'files': files,
                  'equivalence_to_original_complete_stream': 'not_established',
                  'model_steps_run_before_freeze': 0}
        (root / 'frozen_data_manifest.json').write_text(json.dumps(frozen, indent=2)+'\n', encoding='utf8')
        verify_frozen(root)
    if export_dir:
        dest = Path(export_dir)
        dest.mkdir(parents=True, exist_ok=False)
        for name in list(frozen['files']) + ['frozen_data_manifest.json']:
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/name, out)
        verify_frozen(dest)
    return frozen


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--eligibility')
    p.add_argument('--export-dir')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    result = verify_frozen(a.data_root) if a.verify_only else freeze(a.data_root, a.eligibility, a.export_dir)
    print(json.dumps({'data_revision': result['data_revision'], 'files_verified': len(result['files'])}))
