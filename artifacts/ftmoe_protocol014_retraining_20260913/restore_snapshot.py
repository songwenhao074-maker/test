"""Verify a retrained snapshot or extract it into a new, empty directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'extract'])
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent / 'snapshot.zip')
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    target = None
    if args.action == 'extract':
        if args.destination is None:
            parser.error('extract requires --destination')
        target = args.destination.resolve()
        if target in archive.parents or target == archive:
            raise RuntimeError('Destination contains the archive')
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise RuntimeError('Destination must be new or empty')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))
        expected = {'manifest.json'}
        total = 0
        for row in manifest['files']:
            path = PurePosixPath(row['path'])
            if path.is_absolute() or '..' in path.parts or '\\' in row['path'] or ':' in row['path']:
                raise RuntimeError('Unsafe archive path')
            expected.add('snapshot/' + row['path'])
            total += row['bytes']
        if len(z.namelist()) != len(expected) or set(z.namelist()) != expected:
            raise RuntimeError('Unexpected or duplicate ZIP entries')
        if target is not None:
            target.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(target).free < total + 20 * 2**30:
                raise RuntimeError('Insufficient free space for restore plus 20 GiB reserve')
        for row in manifest['files']:
            digest, length = hashlib.sha256(), 0
            with z.open('snapshot/' + row['path']) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block); length += len(block)
            if digest.hexdigest() != row['sha256'] or length != row['bytes']:
                raise RuntimeError('Archive content mismatch: ' + row['path'])
        if target is not None:
            for row in manifest['files']:
                path = target / row['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                if target not in path.resolve().parents:
                    raise RuntimeError('Destination path escapes restore directory')
                with z.open('snapshot/' + row['path']) as source, path.open('xb') as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                    raise RuntimeError('Restored file mismatch: ' + row['path'])
    print(json.dumps({'passed': True, 'verified_files': len(manifest['files']),
                     'archive_sha256': sha(archive), 'restored_to': str(target) if target else None,
                     'snapshot_type': manifest['snapshot_type']}, indent=2))


if __name__ == '__main__':
    main()
