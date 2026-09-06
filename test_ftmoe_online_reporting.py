"""Reporting tests with artificial predictions, never experimental results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analyze_ftmoe_online import analyze,summarize_arrays


class ReportingTests(unittest.TestCase):
    def build(self,root,positive=True,mismatch=False):
        y=np.zeros((300,16),np.int64)
        if positive:y[::2,0]=1
        for method in 'ABCD':
            path=root/method;path.mkdir(parents=True)
            p=np.full((300,16),.1,np.float32)
            if method!='A':p[y>0]=.9
            if method=='B':p[::4,0]=.1
            if method=='C':p[::8,0]=.1
            c=np.full((300,16,3),1/3,np.float32)
            config={'method':method,'model_seed':1,'replay_seed':301,'steps':300,
                    'learning_rate':0. if method=='A' else 3e-5,'stream_sha256':'fixture'}
            (path/'configuration.json').write_text(json.dumps(config))
            np.savez(path/'predictions.npz',probability=p,class_probability=c,labels=y,raw_labels=y,
                     expert_count=np.full(300,4),unmatched_ratio=np.zeros(300))
            (path/'summary.json').write_text(json.dumps({'metrics':summarize_arrays(p,c,y,y),
                'initial_state_hash':method if mismatch else 'shared','frozen_parameters_unchanged':True}))
            (path/'reference.json').write_text(json.dumps([{'step':0,'mean':{'f1':.8}}]))

    def test_all_normal_remains_undefined_and_plots(self):
        with tempfile.TemporaryDirectory(prefix='ftmoe_reporting_test_') as directory:
            root=Path(directory);self.build(root/'runs',False)
            with contextlib.redirect_stdout(io.StringIO()):r=analyze(root/'runs',root/'report','pilot')
            self.assertFalse(r['accepted']);self.assertFalse(r['ready_for_next_stage'])
            self.assertIsNone(r['runs'][0]['metrics']['second_half']['f1'])
            self.assertTrue((root/'report/online_comparison.png').is_file())

    def test_complete_paired_grid_and_acceptance(self):
        with tempfile.TemporaryDirectory(prefix='ftmoe_reporting_test_') as directory:
            root=Path(directory);self.build(root/'runs')
            with contextlib.redirect_stdout(io.StringIO()):r=analyze(root/'runs',root/'report','pilot')
            self.assertTrue(r['accepted']);self.assertTrue(r['ready_for_next_stage'])
            self.assertGreater(r['second_half_means']['D'],r['second_half_means']['B'])

    def test_mismatched_start_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ftmoe_reporting_test_') as directory:
            root=Path(directory);self.build(root/'runs',mismatch=True)
            with self.assertRaisesRegex(ValueError,'identical state/data'):analyze(root/'runs',root/'report','pilot')


if __name__=='__main__':unittest.main(verbosity=2)
