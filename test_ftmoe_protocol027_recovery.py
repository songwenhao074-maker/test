"""A changed recovery artifact must remain blocked even with a matching manifest."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import audit_ftmoe_protocol027 as audit

class RecoveryTests(unittest.TestCase):
    def test_registered_files_pass_but_replacement_with_updated_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'chunks').mkdir()
            stream = root / 'stream.npz'
            chunk = root / 'chunks' / 'chunk.npz'
            stream.write_bytes(b'registered stream')
            chunk.write_bytes(b'registered chunk')
            expected_stream = audit.sha256(stream)
            expected_chunk = audit.sha256(chunk)
            manifest = {'stream_sha256': expected_stream, 'chunk_manifest': [
                {'start': 5400, 'end': 5521, 'file': chunk.name, 'sha256': expected_chunk}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'resume_manifest.json').write_text(json.dumps({'next_t': 5521}))
            with patch.object(audit, 'EXPECTED_STREAM_SHA', expected_stream), patch.object(audit, 'EXPECTED_FINAL_CHUNK_SHA', expected_chunk):
                self.assertTrue(audit.verify_recovery(root, root / 'report')['passed'])
                stream.write_bytes(b'replacement stream')
                manifest['stream_sha256'] = audit.sha256(stream)
                (root / 'manifest.json').write_text(json.dumps(manifest))
                result = audit.verify_recovery(root, root / 'report')
                self.assertFalse(result['passed'])
                self.assertFalse(result['gates']['stream_sha256_matches_registered'])
                self.assertEqual(json.loads((root / 'report/recovery_verification.json').read_text()), result)

    def test_missing_recovery_files_are_diagnosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit.verify_recovery(tmp, Path(tmp) / 'report')
            self.assertFalse(result['passed'])
            self.assertIsNone(result['stream_sha256'])
            self.assertTrue((Path(tmp) / 'report/recovery_verification.json').is_file())

if __name__ == '__main__':
    unittest.main()
