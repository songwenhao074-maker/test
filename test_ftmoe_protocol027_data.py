"""A complete snapshot must remain reusable and detect changed or missing inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import ftmoe_protocol027_data as data

class FrozenDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'data'
        self.root.mkdir()
        (self.root/'chunks').mkdir()
        chunks=[]
        for start in range(0, 5400, 200):
            p=self.root/'chunks'/('chunk_%06d.npz'%start)
            p.write_bytes(str(start).encode())
            chunks.append({'start':start,'end':start+200,'file':p.name,'sha256':data.sha(p)})
        p=self.root/'chunks'/'last.npz';p.write_bytes(b'last chunk')
        chunks.append({'start':5400,'end':5521,'file':p.name,'sha256':data.sha(p)})
        (self.root/'stream.npz').write_bytes(b'fixed stream')
        self.stream_sha=data.sha(self.root/'stream.npz')
        for name in data.BASE_FILES:
            if not (self.root/name).exists(): (self.root/name).write_text('{}')
        (self.root/'manifest.json').write_text(json.dumps({'stream_sha256':self.stream_sha,'chunk_manifest':chunks}))
        self.eligibility=Path(self.tmp.name)/'eligibility.json'
        self.eligibility.write_text(json.dumps({'protocol027_data_eligible':True,'data_revision':data.REVISION_ID,'stream_sha256':self.stream_sha}))
        event_patch=patch.dict(data.REVISION['events_sidecar'], {'uncompressed_sha256':data.sha(self.root/'events.json')})
        event_patch.start();self.addCleanup(event_patch.stop)
        for name,value in [('EXPECTED_STREAM_SHA',self.stream_sha),('EXPECTED_FINAL_CHUNK_SHA',chunks[-1]['sha256'])]:
            m=patch.object(data,name,value);m.start();self.addCleanup(m.stop)

    def test_complete_export_can_be_reused_without_recovery_files(self):
        dest=Path(self.tmp.name)/'export'
        data.freeze(self.root,self.eligibility,dest)
        first=data.sha(dest/'frozen_data_manifest.json')
        data.freeze(dest,self.eligibility)
        self.assertEqual(first,data.sha(dest/'frozen_data_manifest.json'))
        self.assertFalse((dest/'resume_state.dill').exists())
        self.assertEqual(len(data.verify_frozen(dest)['files']),38)

    def test_changed_common_features_are_rejected(self):
        data.freeze(self.root,self.eligibility)
        (self.root/'common_observable_features.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'frozen file changed'):
            data.verify_frozen(self.root)

    def test_missing_events_prevent_freezing(self):
        (self.root/'events.json').unlink()
        with self.assertRaises(FileNotFoundError):
            data.freeze(self.root,self.eligibility)
        self.assertFalse((self.root/'frozen_data_manifest.json').exists())

if __name__=='__main__': unittest.main()
