"""Assemble the authorized revision from existing bytes; never rerun simulation."""
import argparse
import gzip
import json
import shutil
from pathlib import Path
import numpy as np
from ftmoe_protocol027_data import ROOT, REVISION, REVISION_ID, EXPECTED_STREAM_SHA, EXPECTED_FINAL_CHUNK_SHA, sha


def assemble(preserved, candidate, output):
    preserved, candidate, output = map(Path, (preserved, candidate, output))
    if sha(preserved/'preservation_manifest.json') != REVISION['preservation_manifest_sha256']:
        raise ValueError('wrong preservation manifest')
    evidence=json.loads((preserved/'preservation_manifest.json').read_text(encoding='utf8'))
    resume_path=preserved/'state_5520/resume_manifest.json'
    if sha(resume_path) != evidence['files']['state_5520/resume_manifest.json']['sha256']:
        raise ValueError('wrong preserved resume manifest')
    resume=json.loads(resume_path.read_text(encoding='utf8'))
    manifest=json.loads((candidate/'manifest.json').read_text(encoding='utf8'))
    if resume['next_t'] != 5520 or len(resume['chunks']) != 27:
        raise ValueError('wrong preserved prefix')
    if sha(candidate/'stream.npz') != EXPECTED_STREAM_SHA:
        raise ValueError('wrong candidate stream')
    last=manifest['chunk_manifest'][-1]
    if (last['start'],last['end']) != (5400,5521) or sha(candidate/'chunks'/last['file']) != EXPECTED_FINAL_CHUNK_SHA:
        raise ValueError('wrong candidate final chunk')
    if manifest['chunk_manifest'][:-1] != resume['chunks']:
        raise ValueError('candidate prefix manifest differs from preservation')
    with np.load(candidate/'stream.npz') as stream:
        for c in resume['chunks']:
            p=preserved/'chunks'/c['file']
            if sha(p) != c['sha256']:
                raise ValueError('changed prefix chunk: '+c['file'])
            with np.load(p) as part:
                if not all(np.array_equal(part[k],stream[k][c['start']:c['end']]) for k in part.files):
                    raise ValueError('candidate differs from preserved prefix arrays')
        p=preserved/'state_5520/transient_partial.npz'
        if sha(p) != evidence['files']['state_5520/transient_partial.npz']['sha256']:
            raise ValueError('changed transient checkpoint')
        with np.load(p) as part:
            if not all(np.array_equal(part[k],stream[k][5400:5520]) for k in part.files):
                raise ValueError('candidate differs from preserved transient arrays')
        with np.load(candidate/'chunks'/last['file']) as part:
            if not all(np.array_equal(part[k],stream[k][5400:5521]) for k in part.files):
                raise ValueError('candidate stream and tail chunk disagree')
    sidecar=ROOT/REVISION['events_sidecar']['file']
    if sha(sidecar) != REVISION['events_sidecar']['gzip_sha256']:
        raise ValueError('events sidecar digest mismatch')
    if output.exists():
        raise FileExistsError('refusing to replace an existing dataset: '+str(output))
    (output/'chunks').mkdir(parents=True)
    for name in ('stream.npz','manifest.json','data_audit.json','common_observable_features.npz'):
        shutil.copyfile(candidate/name,output/name)
    for c in resume['chunks']:
        shutil.copyfile(preserved/'chunks'/c['file'],output/'chunks'/c['file'])
    shutil.copyfile(candidate/'chunks'/last['file'],output/'chunks'/last['file'])
    (output/'events.json').write_bytes(gzip.decompress(sidecar.read_bytes()))
    if sha(output/'events.json') != REVISION['events_sidecar']['uncompressed_sha256']:
        raise ValueError('events content digest mismatch')
    report={'data_revision':REVISION_ID,'passed':True,'simulation_steps_run':0,
            'prefix_array_equal_intervals':[0,5520],'prefix_chunks_verified':27,
            'candidate_stream_sha256':EXPECTED_STREAM_SHA,'candidate_final_chunk_sha256':EXPECTED_FINAL_CHUNK_SHA,
            'events_sha256':sha(output/'events.json'),'source_candidate_run':REVISION['candidate_run_id'],
            'equivalence_to_original_complete_stream':'not_established'}
    (output/'assembly_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--preserved',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(assemble(a.preserved,a.candidate,a.output),indent=2))
