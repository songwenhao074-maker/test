"""Exercise the 044 finalizer with the inherited collector's actual return contract."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import protocol044_stream as stream


class TerminalContract(unittest.TestCase):
    def test_last_153_rows_report_generation_not_data_acceptance(self):
        chunks = [{"start": t, "end": min(t + 200, 5953)} for t in range(0, 5953, 200)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = root / "scheduler"
            scheduler.write_bytes(b"synthetic fixture weight")
            support = SimpleNamespace(__file__=__file__, _verify_chunks=lambda _, rows: rows[-1]["end"])
            with patch.object(stream, "H", support), patch.object(stream, "source_identity", return_value="fixture"):
                manifest, audit = stream.finalize(root, {"scenario_id": "fixture"}, [], {},
                    SimpleNamespace(response_events=[]), chunks, [], scheduler, 0., "fixture")
                self.assertIsNone(manifest["stream_sha256"])
                self.assertFalse(audit["audit_pass"])
                self.assertEqual(audit["gates"], {})
                self.assertTrue(json.loads((root / "generation_manifest.json").read_text())["input_audit_pending"])
                self.assertFalse((root / "input_lock.json").exists())
                with self.assertRaisesRegex(AssertionError, "generation geometry"):
                    stream.finalize(root, {}, [], {}, None, chunks[:-1], [], scheduler, 0., "fixture")


if __name__ == "__main__":
    unittest.main()
